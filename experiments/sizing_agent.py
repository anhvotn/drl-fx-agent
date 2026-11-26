"""Option 2: a rule picks the direction, the RL agent only decides how much of it to hold.

The rule is carry on six pairs (EUR/USD, GBP/USD, AUD/USD, USD/JPY, USD/CHF, USD/CAD), each pair
volatility-scaled, equal weights; see experiments/multi_pair_rules.py. Each day the agent picks a
scale between 0 (out) and 2 (double) for the whole basket. It sees only information known at
that day's close: the basket's own recent returns, volatility and drawdown, VIX, and how far
apart the interest rates are. Changing the scale costs the spread on the change in notional.

The agent is retrained every year on all earlier years (walk-forward, 2016 to 2026) with several
random seeds. It passes only if it beats, out of sample and on every seed:
  - the rule at a fixed size (scale 1)
  - two hand-made sizing rules: volatility targeting, and switching off when VIX is high
  - its own scales handed out at random within each year (95th percentile)
Results are shown without and with IBKR's financing markup on borrowed currency.

    python -m experiments.sizing_agent --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd

from experiments import multi_pair_rules as M
from experiments.predictability import read_daily_series

warnings.filterwarnings("ignore")
MAX_SCALE = 2.0
FEATURES = ["ret_5", "ret_20", "ret_60", "vol_20", "vol_60", "drawdown_250",
            "log_vix", "vix_chg_5", "vix_pct_250", "rate_gap_abs"]


# ------------------------------------------------------------------ data
def basket(data_dir: str, bis_path: str, financing: float) -> pd.DataFrame:
    """Daily carry basket at scale 1, plus the inputs the agent may see."""
    bis = M.read_bis(bis_path)
    ecb = read_daily_series(os.path.join(data_dir, "ECBDFR.csv"))
    vix = read_daily_series(os.path.join(data_dir, "VIXCLS.csv"))
    nets, expo, cost, gaps, vixs, held = {}, {}, {}, {}, {}, {}
    for p in M.PAIRS:
        d = M.pair_frame(p, data_dir, bis, ecb, vix)
        s = M.strategy_returns(d, "carry")
        nets[p], expo[p], cost[p] = s["net"], s["pos"].abs(), d["cost"]
        gaps[p] = (d["rate_base"] - d["rate_quote"]).abs()
        vixs[p] = d["vix"]
        held[p] = (d["close_time"].shift(-1) - d["close_time"]).dt.total_seconds() / 86400
    idx = sorted(set().union(*[set(v.index) for v in nets.values()]))
    f = pd.DataFrame(index=pd.DatetimeIndex(idx))
    f["net"] = pd.DataFrame(nets).reindex(f.index).fillna(0).mean(axis=1)            # basket return at scale 1
    f["exposure"] = pd.DataFrame(expo).reindex(f.index).fillna(0).mean(axis=1)       # gross notional per unit equity
    f["cost"] = pd.DataFrame(cost).reindex(f.index).mean(axis=1)
    f["days"] = pd.DataFrame(held).reindex(f.index).median(axis=1).clip(upper=5).fillna(1)
    # financing markup: paid on the borrowed leg, every calendar day the position is held
    f["financing"] = f["exposure"] * financing * f["days"] / 365
    f["vix"] = pd.DataFrame(vixs).reindex(f.index).mean(axis=1).ffill()
    f["rate_gap_abs"] = pd.DataFrame(gaps).reindex(f.index).mean(axis=1).ffill()

    # inputs, all from data up to and including the close of day t (net[t] is earned after it)
    past = f["net"].shift(1)                         # returns already realised at the close of day t
    f["ret_5"], f["ret_20"], f["ret_60"] = (past.rolling(n).sum() for n in (5, 20, 60))
    f["vol_20"], f["vol_60"] = (past.rolling(n).std() * np.sqrt(252) for n in (20, 60))
    eq = past.fillna(0).cumsum()
    f["drawdown_250"] = eq - eq.rolling(250, min_periods=20).max()
    f["log_vix"] = np.log(f["vix"])
    f["vix_chg_5"] = f["log_vix"].diff(5)
    f["vix_pct_250"] = f["vix"].rolling(250, min_periods=126).rank(pct=True)
    return f.dropna()


# ------------------------------------------------------------------ environment
class SizingEnv(gym.Env):
    """One step = one day. Action: scale of the carry basket in [0, MAX_SCALE]."""

    def __init__(self, f: pd.DataFrame, X: np.ndarray, episode_len: int = 252, eta: float = 0.01,
                 random_start: bool = True, seed: int | None = None):
        super().__init__()
        self.r = f["net"].to_numpy()
        self.fin = f["financing"].to_numpy()
        self.tc = (f["exposure"] * f["cost"]).to_numpy()
        self.X = X.astype(np.float32)
        self.episode_len = min(episode_len, len(f) - 1)
        self.random_start, self.eta = random_start, eta
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10, 10, shape=(X.shape[1] + 1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def _obs(self):
        return np.append(self.X[self.t], np.float32(self.scale / MAX_SCALE)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        hi = len(self.r) - self.episode_len - 1
        self.t = int(self.rng.integers(0, hi + 1)) if (self.random_start and hi > 0) else 0
        self.t0, self.scale, self.A, self.B = self.t, 1.0, 0.0, 0.0
        return self._obs(), {}

    @staticmethod
    def to_scale(action) -> float:
        return float((np.clip(np.asarray(action).ravel()[0], -1, 1) + 1) / 2 * MAX_SCALE)

    def step(self, action):
        new = self.to_scale(action)
        net = new * (self.r[self.t] - self.fin[self.t]) - abs(new - self.scale) * self.tc[self.t]
        self.scale = new
        dA, dB = net - self.A, net ** 2 - self.B                      # differential Sharpe ratio
        denom = (self.B - self.A ** 2) ** 1.5
        reward = (self.B * dA - 0.5 * self.A * dB) / denom if denom > 1e-12 else 0.0
        self.A += self.eta * dA
        self.B += self.eta * dB
        self.t += 1
        terminated = self.t >= len(self.r) - 1
        truncated = (self.t - self.t0) >= self.episode_len and not terminated
        return self._obs(), float(np.clip(reward, -5, 5)), terminated, truncated, {"net": net}


# ------------------------------------------------------------------ evaluation
def scaled_returns(f: pd.DataFrame, scale: pd.Series) -> pd.Series:
    scale = scale.reindex(f.index).astype(float)
    change = scale.diff().abs().fillna((scale - 1.0).abs())
    return scale * (f["net"] - f["financing"]) - change * f["exposure"] * f["cost"]


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def summary(x: pd.Series, scale: pd.Series | None = None) -> dict:
    eq = x.cumsum()
    out = {"sharpe": round(sharpe(x), 2), "return_%_a_year": round(x.mean() * 252 * 100, 2),
           "vol_%": round(x.std() * np.sqrt(252) * 100, 2),
           "max_drawdown_%": round(float((eq - eq.cummax()).min()) * 100, 1)}
    if scale is not None:
        out["avg_scale"] = round(float(scale.mean()), 2)
        out["scale_changes_a_year"] = round(float((scale.diff().abs() > 0.05).mean() * 252), 1)
    return out


def hand_rules(f: pd.DataFrame) -> dict[str, pd.Series]:
    target = (f["vol_60"].median() / f["vol_60"]).clip(upper=MAX_SCALE)       # same average size as scale 1
    return {"fixed size": pd.Series(1.0, index=f.index),
            "volatility targeting": target,
            "off when VIX is high": (f["vix_pct_250"] <= 0.8).astype(float)}


def train_and_predict(f: pd.DataFrame, seed: int, first_year: int, steps: int) -> pd.Series:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    scales = []
    for year in range(first_year, f.index.max().year + 1):
        tr = f[f.index < f"{year}-01-01"]
        te = f[(f.index >= f"{year}-01-01") & (f.index < f"{year + 1}-01-01")]
        if len(te) == 0:
            continue
        mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
        Xtr = ((tr[FEATURES] - mu) / sd).clip(-5, 5).to_numpy()
        Xte = ((te[FEATURES] - mu) / sd).clip(-5, 5).to_numpy()
        venv = DummyVecEnv([lambda i=i: SizingEnv(tr, Xtr, seed=seed * 100 + i) for i in range(4)])
        model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=5,
                    gamma=0.95, ent_coef=0.005, policy_kwargs=dict(net_arch=[32, 32]),
                    seed=seed, verbose=0, device="cpu")
        model.learn(steps)
        # trade the test year day by day: the scale held yesterday is part of today's input
        env = SizingEnv(te, Xte, episode_len=len(te), random_start=False)
        obs, _ = env.reset()
        out = []
        for _ in range(len(te)):
            a, _ = model.predict(obs, deterministic=True)
            out.append(SizingEnv.to_scale(a))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        scales.append(pd.Series(out, index=te.index[:len(out)]))
        print(f"  seed {seed} {year}: mean scale {np.mean(out):.2f}", flush=True)
    return pd.concat(scales)


def random_scales(scale: pd.Series, f: pd.DataFrame, n: int, seed: int = 0) -> float:
    """95th percentile Sharpe when the agent's own daily scales are reshuffled within each year,
    in blocks of 20 days so the reshuffled scales change about as often as the agent's."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        parts = []
        for _, s in scale.groupby(scale.index.year):
            blocks = [s.values[i:i + 20] for i in range(0, len(s), 20)]
            rng.shuffle(blocks)
            parts.append(pd.Series(np.concatenate(blocks), index=s.index))
        res.append(sharpe(scaled_returns(f, pd.concat(parts))))
    return round(float(np.percentile(res, 95)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--bis", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=60_000, help="PPO steps per yearly retraining")
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--financing", type=float, default=0.015, help="IBKR markup on borrowed currency, per year")
    ap.add_argument("--out", default="outputs/sizing_agent")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bis = a.bis or next(p for p in [os.path.join(a.data, "WS_CBPOL_csv_col.csv"),
                                    os.path.join(a.data, "WS_CBPOL_csv_col", "WS_CBPOL_csv_col.csv")]
                        if os.path.exists(p))

    # train without financing (it would only teach the agent to hold less); report both
    f0 = basket(a.data, bis, 0.0)
    f1 = basket(a.data, bis, a.financing)
    oos = f0.index >= f"{a.first_test_year}-01-01"
    years = oos.sum() / 252
    print(f"\nCarry basket, {len(f0):,} days; out of sample {f0.index[oos][0].date()} .. {f0.index[-1].date()} "
          f"({years:.1f} years, Sharpe standard error about {1 / np.sqrt(years):.2f})\n")

    agent = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        agent[f"agent, seed {s}"] = train_and_predict(f0, s, a.first_test_year, a.steps)
    scales = {**{k: v[oos] for k, v in hand_rules(f0).items()}, **agent}
    scales["agent, average of seeds"] = pd.concat(agent.values(), axis=1).mean(axis=1)
    pd.DataFrame(scales).to_csv(os.path.join(a.out, "scales.csv"))

    rows = {}
    for name, sc in scales.items():
        x0, x1 = scaled_returns(f0[oos], sc), scaled_returns(f1[oos], sc)
        r = summary(x0, sc)
        r["sharpe_after_financing"] = round(sharpe(x1), 2)
        r["return_%_after_financing"] = round(x1.mean() * 252 * 100, 2)
        if name.startswith("agent"):
            r["random_scales_p95"] = random_scales(sc, f0[oos], 300)
        rows[name] = r
    table = pd.DataFrame(rows).T
    table.to_csv(os.path.join(a.out, "results.csv"))
    print("\n" + table.to_string())

    yearly = pd.DataFrame({k: scaled_returns(f0[oos], v).groupby(f0.index[oos].year).apply(sharpe)
                           for k, v in scales.items()}).round(2)
    yearly.to_csv(os.path.join(a.out, "yearly_sharpe.csv"))
    print("\nSharpe per year:\n" + yearly.to_string())

    best_hand = max(rows[k]["sharpe"] for k in hand_rules(f0))
    seeds = [k for k in rows if k.startswith("agent, seed")]
    verdict = {
        "beats fixed size on every seed": all(rows[k]["sharpe"] > rows["fixed size"]["sharpe"] for k in seeds),
        "beats the best hand-made sizing rule on every seed": all(rows[k]["sharpe"] > best_hand for k in seeds),
        "above random scales (p95) on every seed": all(rows[k]["sharpe"] > rows[k]["random_scales_p95"] for k in seeds),
        "positive after financing": rows["agent, average of seeds"]["sharpe_after_financing"] > 0,
    }
    print("\nPass checks:")
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"years": round(years, 1), "results": rows, "checks": verdict}, fh, indent=2, default=float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for k in ["fixed size", "volatility targeting", "off when VIX is high", "agent, average of seeds"]:
            ax1.plot(scaled_returns(f0[oos], scales[k]).cumsum() * 100, label=k, lw=1.8 if "agent" in k else 1)
        ax1.set_ylabel("cumulative return, %")
        ax1.set_title("Carry on six pairs: who should decide the size?")
        ax1.legend(fontsize=8, frameon=False)
        ax2.plot(scales["agent, average of seeds"], lw=0.8, color="tab:red")
        ax2.set_ylabel("agent's scale")
        ax2.set_ylim(0, MAX_SCALE)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "equity.png"), dpi=120)
    except Exception as err:
        print("chart skipped:", err)


if __name__ == "__main__":
    main()
