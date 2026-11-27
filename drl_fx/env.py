"""Trading environment: the agent sees a window of data and chooses buy / sell / hold.

Reward stage 1 ("pnl"):    position * next return + position * carry - transaction costs (spread + impact + fees)
Reward stage 2 ("sharpe"): differential Sharpe ratio (Moody & Saffell, 1998) of the same net return,
                           i.e. "PnL first, Sharpe ratio in later stages" as in the course.

The benchmark is being flat (benchmark=0.0, the default): the agent is rewarded for what its
position earns. For FX this is the natural choice, as a currency pair has no long-term drift.
benchmark=1.0 makes both rewards relative to buy-and-hold instead: (position - 1) * (return + carry)
- costs, so simply staying long earns nothing. That suits markets with a long-term drift, which currency pairs do not have.

Carry is the interest difference earned or paid for holding a position until the next bar
(data.make_features, from rate_base and rate_quote); it is zero when no rates are given.
The logged "net" is always the agent's real PnL, whatever the reward.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

HOLD, BUY, SELL = 0, 1, 2


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, f: pd.DataFrame, obs_cols: list[str], window: int = 30,
                 episode_len: int | None = 2048, reward: str = "pnl", allow_short: bool = True,
                 fee: float = 0.0, cost_mult: float = 1.0, dsr_eta: float = 0.01,
                 random_start: bool = True, seed: int | None = None, benchmark: float = 0.0,
                 turnover_penalty: float = 0.0):
        super().__init__()
        self.X = f[obs_cols].to_numpy(np.float32)
        self.r = f["fwd_ret"].to_numpy(np.float64)
        self.c = f["cost"].to_numpy(np.float64) * cost_mult + fee
        self.k = (f["carry"].to_numpy(np.float64) if "carry" in f.columns else np.zeros(len(f)))
        self.fin = (f["financing"].to_numpy(np.float64) if "financing" in f.columns else np.zeros(len(f)))
        # extra charge per unit traded that only shapes the reward (discourages flipping on noise);
        # it is not a real cost, so the logged net PnL does not include it
        self.turnover_penalty = turnover_penalty
        self.index = f.index
        self.window, self.reward_mode = window, reward
        self.allow_short, self.dsr_eta = allow_short, dsr_eta
        self.episode_len = episode_len or (len(f) - window - 1)
        self.episode_len = min(self.episode_len, len(f) - window - 1)
        self.random_start, self.benchmark = random_start, benchmark
        self.rng = np.random.default_rng(seed)

        n_obs = window * len(obs_cols) + 1          # flattened window + current position
        self.observation_space = spaces.Box(-10, 10, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    # ------------------------------------------------------------------
    def _obs(self):
        w = self.X[self.t - self.window + 1: self.t + 1].ravel()
        return np.append(w, np.float32(self.pos)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:                         # Gymnasium convention: reset(seed=...) fixes the episode starts
            self.rng = np.random.default_rng(seed)
        lo, hi = self.window - 1, len(self.X) - self.episode_len - 1
        self.t0 = int(self.rng.integers(lo, hi + 1)) if (self.random_start and hi > lo) else lo
        self.t, self.pos, self.steps = self.t0, 0.0, 0
        self.A = self.B = 0.0                        # DSR running moments
        self.log = []
        return self._obs(), {}

    def step(self, action: int):
        target = {HOLD: self.pos, BUY: 1.0, SELL: -1.0 if self.allow_short else 0.0}[int(action)]
        trade = abs(target - self.pos)
        cost = trade * self.c[self.t]
        carry = target * self.k[self.t] - abs(target) * self.fin[self.t]   # interest gap, less broker markup
        net = target * self.r[self.t] + carry - cost    # simple-ish log-return approximation
        excess = net - self.benchmark * (self.r[self.t] + self.k[self.t])   # == net when benchmark is 0
        self.pos = target

        excess = excess - trade * self.turnover_penalty     # reward only
        if self.reward_mode == "sharpe":
            dA, dB = excess - self.A, excess ** 2 - self.B
            denom = max(self.B - self.A ** 2, 0.0) ** 1.5    # running variance can dip below 0 by rounding
            reward = (self.B * dA - 0.5 * self.A * dB) / denom if denom > 1e-12 else 0.0
            self.A += self.dsr_eta * dA
            self.B += self.dsr_eta * dB
            reward = float(np.clip(reward, -5, 5))
        else:
            reward = float(excess * 100)                 # scale to ~unit size for PPO

        self.log.append((self.index[self.t], target, self.r[self.t], self.k[self.t], cost, carry, net))
        self.t += 1
        self.steps += 1
        # The data running out is a real end; the episode length is only a training cut-off, so it is
        # reported as truncated and PPO keeps the value of the state instead of treating it as zero.
        terminated = self.t >= len(self.X) - 1
        truncated = self.steps >= self.episode_len and not terminated
        return self._obs(), reward, terminated, truncated, {"net": net, "cost": cost, "carry": carry,
                                                        "excess": excess}

    def history(self) -> pd.DataFrame:
        # ret / carry_long: what a long position earns this bar; carry: what this position earned
        cols = ["time", "pos", "ret", "carry_long", "cost", "carry", "net"]
        return pd.DataFrame(self.log, columns=cols).set_index("time")


class MultiAssetEnv(gym.Env):
    """Trains one agent on several instruments: each episode picks one of them at random.
    All sub-environments must share the same observation columns and window."""
    metadata = {"render_modes": []}

    def __init__(self, envs: list[TradingEnv], seed: int | None = None):
        super().__init__()
        self.envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.rng = np.random.default_rng(seed)
        self.cur = envs[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:                         # one seed fixes the instrument order and every episode start
            self.rng = np.random.default_rng(seed)
            for e in self.envs:
                e.rng = np.random.default_rng(int(self.rng.integers(2**31)))
        self.cur = self.envs[int(self.rng.integers(len(self.envs)))]
        return self.cur.reset()

    def step(self, action):
        return self.cur.step(action)
