"""Backtesting, go-live checks and ensemble weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .env import TradingEnv


# ---------------------------------------------------------------- backtest
def backtest(model, f: pd.DataFrame, obs_cols, window: int, cost_mult: float = 1.0,
             allow_short: bool = True) -> pd.DataFrame:
    """Run the policy deterministically over the whole segment, one pass, no random starts.

    Fast path: the policy's only state is the current position (-1/0/+1), so we precompute its
    action for every bar under each possible position in large batches, then walk forward.
    The result is identical to stepping the env bar by bar."""
    import torch
    env = TradingEnv(f, obs_cols, window=window, episode_len=None, random_start=False,
                     cost_mult=cost_mult, allow_short=allow_short)
    env.reset()
    T = np.arange(window - 1, window - 1 + env.episode_len)
    win = np.lib.stride_tricks.sliding_window_view(env.X, window, axis=0)   # (n-w+1, cols, w)
    flat = win.transpose(0, 2, 1).reshape(win.shape[0], -1)[T - window + 1]
    acts = {}
    with torch.no_grad():
        for p in (-1.0, 0.0, 1.0):
            obs = np.hstack([flat, np.full((len(T), 1), p, np.float32)])
            out = []
            for i in range(0, len(obs), 8192):
                a, _ = model.predict(obs[i:i + 8192], deterministic=True)
                out.append(a)
            acts[p] = np.concatenate(out)
    for i in range(len(T)):
        _, _, terminated, truncated, _ = env.step(int(acts[env.pos][i]))
        if terminated or truncated:
            break
    return env.history()


def metrics(h: pd.DataFrame, periods_per_year: int = 252) -> dict:
    net = h["net"]
    eq = net.cumsum()
    sd = net.std()
    bh = h["ret"] + (h["carry_long"] if "carry_long" in h else 0.0)
    ex = net - bh                                     # return in excess of buy-and-hold
    ex_sd = ex.std()
    corr = float(np.corrcoef(net, h["ret"])[0, 1]) if sd > 0 and h["ret"].std() > 0 else 0.0
    return {
        "total_return_%": round(100 * (np.exp(eq.iloc[-1]) - 1), 2) if len(eq) else 0.0,
        "sharpe": round(float(net.mean() / sd * np.sqrt(periods_per_year)), 2) if sd > 0 else 0.0,
        "max_drawdown_%": round(100 * float((np.exp(eq) / np.exp(eq.cummax()) - 1).min()), 2),
        "turnover_per_bar": round(float(h["pos"].diff().abs().mean()), 3),
        "cost_drag_%": round(100 * float(h["cost"].sum()), 2),
        "carry_%": round(100 * float(h["carry"].sum()), 2) if "carry" in h else 0.0,
        "time_in_market_%": round(100 * float((h["pos"] != 0).mean()), 1),
        # an agent that tracks buy-and-hold almost exactly (under 0.5% a year apart) has no
        # meaningful excess Sharpe: dividing tiny differences by a tiny spread only gives noise
        "excess_sharpe_vs_bh": round(float(ex.mean() / ex_sd * np.sqrt(periods_per_year)), 2)
        if ex_sd * np.sqrt(periods_per_year) > 0.005 else 0.0,
        "corr_with_bh": round(corr, 3),
        "bars": len(h),
    }


def buy_and_hold(f: pd.DataFrame, window: int) -> pd.DataFrame:
    seg = f.iloc[window - 1:-1]
    k = seg["carry"] if "carry" in seg else 0.0
    fin = seg["financing"] if "financing" in seg else 0.0
    return pd.DataFrame({"pos": 1.0, "ret": seg["fwd_ret"], "carry_long": k, "cost": 0.0, "carry": k - fin,
                         "net": seg["fwd_ret"] + k - fin}, index=seg.index)


def positions_history(pos: pd.Series, seg: pd.DataFrame) -> pd.DataFrame:
    """History frame for any position series on a data segment, with the same accounting as the
    environment: the position at t earns the return and carry from t to t+1 and pays costs on changes."""
    seg = seg.loc[pos.index]
    k = seg["carry"] if "carry" in seg else pd.Series(0.0, index=seg.index)
    cost = pos.diff().abs().fillna(pos.abs()) * seg["cost"]
    fin = seg["financing"] if "financing" in seg else 0.0
    carry = pos * k - pos.abs() * fin
    return pd.DataFrame({"pos": pos, "ret": seg["fwd_ret"], "carry_long": k, "cost": cost, "carry": carry,
                         "net": pos * seg["fwd_ret"] + carry - cost})


# ---------------------------------------------------------------- simple FX benchmarks
def add_benchmarks(f: pd.DataFrame, ppy: int, trend_months: float = 3.0) -> pd.DataFrame:
    """Positions of two classic FX strategies, computed on the full history before splitting
    (each uses only past data):
      bm_trend  long if the pair rose over the past `trend_months`, short if it fell
      bm_carry  hold the higher-yielding currency (long if the first currency pays more interest)
    """
    f = f.copy()
    look = max(int(ppy * trend_months / 12), 2)
    past = f["ret_1"].rolling(look).sum()                 # returns up to bar t, known at t
    f["bm_trend"] = np.sign(past).fillna(0.0)
    f["bm_carry"] = np.sign(f["carry"]) if "carry" in f else 0.0
    return f


def benchmark_metrics(seg: pd.DataFrame, window: int, ppy: int) -> dict:
    """Metrics of the benchmark strategies on a segment, aligned with the agent's backtest bars."""
    s = seg.iloc[window - 1:-1]
    out = {}
    if "bm_trend" in s:
        out["trend"] = metrics(positions_history(s["bm_trend"], s), ppy)
    if "bm_carry" in s and (s["bm_carry"] != 0).any():
        out["carry"] = metrics(positions_history(s["bm_carry"], s), ppy)
    return out


def random_benchmark(h: pd.DataFrame, seg: pd.DataFrame, ppy: int, n: int = 50, seed: int = 0) -> dict:
    """Random positions that change as often as the agent's (h). If the agent does not beat
    most of these, its trading is indistinguishable from noise that pays costs."""
    rng = np.random.default_rng(seed)
    p_change = float((h["pos"].diff().fillna(0) != 0).mean())
    states = np.array([-1.0, 0.0, 1.0])
    sharpes = []
    for _ in range(n):
        change = rng.random(len(h)) < p_change
        pos = np.empty(len(h)); cur = rng.choice(states)
        for i in range(len(h)):
            if change[i]:
                cur = rng.choice(states[states != cur])
            pos[i] = cur
        sharpes.append(metrics(positions_history(pd.Series(pos, index=h.index), seg), ppy)["sharpe"])
    return {"sharpe": round(float(np.mean(sharpes)), 2),
            "sharpe_p95": round(float(np.percentile(sharpes, 95)), 2), "runs": n}


def by_regime(h: pd.DataFrame, regimes: pd.Series, ppy: int) -> pd.DataFrame:
    lab = regimes.reindex(h.index)
    rows = {r: metrics(h[lab == r], ppy) for r in lab.dropna().unique() if (lab == r).sum() > 20}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- go-live checks
def diagnose(train_sr: float, dev_sr: float, test_sr: float, good: float = 0.5) -> str:
    """Course's three cases mapped to remedies."""
    if train_sr < good * 0.5:
        return ("UNDERFIT: poor on training data. Add inputs, deepen the network, tune PPO/network "
                "hyper-parameters, check weight initialisation.")
    if test_sr < 0 or test_sr < 0.4 * train_sr:
        return ("OVERFIT: good in training, weak/negative on test. Bring in more data, use a shallower "
                "network, or raise dropout.")
    return "OK: decent in train, dev and test. Proceed to holdout, then fine-tune with a small LR and paper trade."


def is_bh_copy(m: dict) -> bool:
    """True when the agent just holds one fixed position: always long (a buy-and-hold copy),
    always short (its mirror image) or always flat. All three barely trade."""
    return m["turnover_per_bar"] < 0.002 or (abs(m["corr_with_bh"]) > 0.95 and m["turnover_per_bar"] < 0.005)


def go_live_checklist(res: dict, ppy: int, min_sharpe: float = 0.5) -> pd.DataFrame:
    """res holds metrics dicts and regime tables produced by run.py."""
    m = res["metrics"]
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})

    add("Data size >= 100k rows", res["data_size"]["ok"], res["data_size"]["note"])
    add("Train / dev / test consistent", diagnose(m["train"]["sharpe"], m["dev"]["sharpe"],
                                                   m["test"]["sharpe"]).startswith("OK"),
        diagnose(m["train"]["sharpe"], m["dev"]["sharpe"], m["test"]["sharpe"]))
    add("Holdout (never seen) Sharpe >= threshold", m["holdout"]["sharpe"] >= min_sharpe,
        f"holdout Sharpe {m['holdout']['sharpe']} vs threshold {min_sharpe}")
    bms = {k: m[f"holdout_{k}"]["sharpe"] for k in ["carry", "trend"] if f"holdout_{k}" in m}
    if bms:
        add("Beats the carry and trend strategies on holdout", m["holdout"]["sharpe"] > max(bms.values()),
            f"agent {m['holdout']['sharpe']} vs " + ", ".join(f"{k} {v}" for k, v in bms.items()))
    if "holdout_random" in m:
        r = m["holdout_random"]
        add("Beats random trading with the same turnover", m["holdout"]["sharpe"] > r["sharpe_p95"],
            f"agent {m['holdout']['sharpe']} vs random: mean {r['sharpe']}, best 5% above {r['sharpe_p95']}")
    copy = is_bh_copy(m["holdout"])
    tim, corr = m["holdout"]["time_in_market_%"], m["holdout"]["corr_with_bh"]
    side = "flat" if tim < 5 else ("long" if corr > 0 else "short")
    add("Not stuck in one position (always long, short or flat)", not copy,
        f"correlation with B&H {m['holdout']['corr_with_bh']}, turnover/bar {m['holdout']['turnover_per_bar']}"
        + (f" -> the agent learned to stay {side}; it never times the market" if copy else ""))
    reg = res["regimes_test_holdout"]
    covered = set(reg.index) & {"up", "down", "steady", "volatile"}
    add("Out-of-sample data covers up/down/steady/volatile", len(covered) == 4,
        f"regimes present: {sorted(covered)}")
    neg = [r for r in reg.index if reg.loc[r, "sharpe"] < -0.5]
    add("No regime with badly negative Sharpe", not neg, f"weak regimes: {neg or 'none'}")
    add("Survives 2x transaction costs", m["holdout_2x_cost"]["sharpe"] > 0,
        f"holdout Sharpe at 2x costs: {m['holdout_2x_cost']['sharpe']}")
    if "transfer" in m:
        add("Works on an untrained instrument, same timestep", m["transfer"]["sharpe"] > 0,
            f"transfer Sharpe {m['transfer']['sharpe']}")
    else:
        add("Works on an untrained instrument, same timestep", False, "not run: pass --transfer-csv")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ensemble
def ensemble_weights(recent_sharpes: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Higher weight to agents that did better recently; agents with Sharpe <= 0 get zero."""
    names = list(recent_sharpes)
    s = np.array([max(recent_sharpes[n], 0.0) for n in names])
    if s.sum() == 0:
        return {n: 0.0 for n in names}            # nobody works recently -> stay flat
    w = np.exp(s / temperature) * (s > 0)
    w /= w.sum()
    return dict(zip(names, np.round(w, 3)))


def ensemble_action(models: dict, weights: dict, obs) -> int:
    """Weighted vote over buy/sell/hold; 'hold' if no agent has weight."""
    votes = np.zeros(3)
    for n, m in models.items():
        if weights.get(n, 0) > 0:
            a, _ = m.predict(obs, deterministic=True)
            votes[int(a)] += weights[n]
    return int(votes.argmax()) if votes.sum() > 0 else 0
