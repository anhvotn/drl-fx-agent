"""Walk-forward test: retrain every few months on all data so far, trade the months that follow.

    python -m experiments.walk_forward --csv data/eurusd_15m_news.csv --fee-bp 0.2

Fold k trains on everything before its test start (expanding window), normalises with that training
data only, and is scored on the next --test-months. The test periods never overlap and each one is
unseen by the agent that trades it. Gluing them together gives one long out-of-sample track record,
which says much more than a single holdout period.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from drl_fx import data as D
from drl_fx.evaluate import backtest, buy_and_hold, metrics, positions_history, random_benchmark
from experiments.common import load, save_plot, train_agent

G = {}


def _init(csv, fee_bp, benchmark=0.0):
    f, cols, groups, ppy = load(csv, fee_bp)
    G.update(f=f, cols=cols, groups=groups, ppy=ppy, benchmark=benchmark)


def _job(args):
    k, test_start, test_end, window, seed, steps_pnl, steps_sharpe, out, reuse = args
    f, cols = G["f"], G["cols"]
    train = f[f.index < test_start]
    # include window-1 bars before the test start so the first test bar already has a full window
    first = f.index.searchsorted(test_start)
    test = f.iloc[max(first - window + 1, 0):][lambda d: d.index < test_end]
    train, test = D.normalize(train, [test], cols)
    path = os.path.join(out, "models", f"fold{k:02d}")
    if reuse and os.path.exists(path + ".zip"):
        from stable_baselines3 import PPO
        model = PPO.load(path, device="cpu")
    else:
        model = train_agent(train, cols, G["groups"], window, seed, steps_pnl, steps_sharpe,
                            benchmark=G["benchmark"])
        model.save(path)
    h = backtest(model, test, cols, window)
    h = h[h.index >= test_start]
    bh = buy_and_hold(test, window)
    bh = bh[bh.index >= test_start]
    print(f"  fold {k}: {test_start:%Y-%m} .. {test_end:%Y-%m}, Sharpe {metrics(h, G['ppy'])['sharpe']}", flush=True)
    return k, test_start, test_end, len(train), h, bh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--fee-bp", type=float, default=0.0)
    p.add_argument("--first-train-years", type=float, default=4.0)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-pnl", type=int, default=200_000)
    p.add_argument("--steps-sharpe", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="outputs/walk_forward")
    p.add_argument("--vs-buy-and-hold", action="store_true", help="reward relative to holding long")
    p.add_argument("--reuse-models", action="store_true",
                   help="load fold agents saved by an earlier run instead of training again")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)

    bench = 1.0 if a.vs_buy_and_hold else 0.0
    _init(a.csv, a.fee_bp, bench)
    f, ppy = G["f"], G["ppy"]
    start = f.index.min() + pd.DateOffset(months=int(round(a.first_train_years * 12)))
    edges = []
    t = start
    while t < f.index.max() - pd.Timedelta(days=15 * a.test_months):   # skip a stub fold shorter than half a period
        edges.append((t, min(t + pd.DateOffset(months=a.test_months), f.index.max() + pd.Timedelta(minutes=1))))
        t = t + pd.DateOffset(months=a.test_months)
    jobs = [(k, s, e, a.window, a.seed, a.steps_pnl, a.steps_sharpe, a.out, a.reuse_models)
            for k, (s, e) in enumerate(edges)]
    print(f"{len(jobs)} folds, first test period starts {start:%Y-%m-%d}, {a.workers} workers ...", flush=True)

    with Pool(a.workers, initializer=_init, initargs=(a.csv, a.fee_bp, bench)) as pool:
        results = sorted(pool.map(_job, jobs, chunksize=1), key=lambda r: r[0])

    rows, agent_all, bh_all, trend_all, carry_all = [], [], [], [], []
    has_carry = (f["bm_carry"] != 0).any()
    for k, s, e, n_train, h, bh in results:
        m, mb = metrics(h, ppy), metrics(bh, ppy)
        seg = f.loc[h.index]
        ht = positions_history(seg["bm_trend"], seg)
        hc = positions_history(seg["bm_carry"], seg) if has_carry else None
        trend_all.append(ht)
        if hc is not None:
            carry_all.append(hc)
        rows.append({"fold": k, "test_from": f"{s:%Y-%m-%d}", "test_to": f"{e:%Y-%m-%d}", "train_bars": n_train,
                     "agent_sharpe": m["sharpe"], "bh_sharpe": mb["sharpe"],
                     "trend_sharpe": metrics(ht, ppy)["sharpe"],
                     "carry_sharpe": metrics(hc, ppy)["sharpe"] if hc is not None else None,
                     "agent_return_%": m["total_return_%"], "bh_return_%": mb["total_return_%"],
                     "excess_sharpe_vs_bh": m["excess_sharpe_vs_bh"], "turnover": m["turnover_per_bar"],
                     "long_%": round(100 * float((h["pos"] > 0).mean()), 1),
                     "short_%": round(100 * float((h["pos"] < 0).mean()), 1),
                     "cost_drag_%": m["cost_drag_%"], "carry_%": m["carry_%"]})
        agent_all.append(h); bh_all.append(bh)
    folds = pd.DataFrame(rows).set_index("fold")
    A, B = pd.concat(agent_all), pd.concat(bh_all)
    A.to_csv(os.path.join(a.out, "positions_out_of_sample.csv"))
    mA, mB = metrics(A, ppy), metrics(B, ppy)
    mT = metrics(pd.concat(trend_all), ppy)
    mC = metrics(pd.concat(carry_all), ppy) if carry_all else None
    rnd = random_benchmark(A, f.loc[A.index], ppy)
    summary = {
        "folds": len(folds),
        "out_of_sample_from": rows[0]["test_from"], "out_of_sample_to": rows[-1]["test_to"],
        "agent_sharpe_all_folds": mA["sharpe"], "bh_sharpe_all_folds": mB["sharpe"],
        "agent_return_%_all_folds": mA["total_return_%"], "bh_return_%_all_folds": mB["total_return_%"],
        "agent_max_drawdown_%": mA["max_drawdown_%"],
        "excess_sharpe_vs_bh_all_folds": mA["excess_sharpe_vs_bh"],
        "folds_with_positive_sharpe": int((folds["agent_sharpe"] > 0).sum()),
        "folds_beating_bh": int((folds["agent_sharpe"] > folds["bh_sharpe"]).sum()),
        "cost_drag_%_all_folds": mA["cost_drag_%"],
        "carry_%_all_folds": mA["carry_%"],
        "reward": "excess vs buy-and-hold" if bench else "PnL vs flat",
        "trend_sharpe_all_folds": mT["sharpe"], "trend_return_%_all_folds": mT["total_return_%"],
        "carry_sharpe_all_folds": mC["sharpe"] if mC else None,
        "carry_return_%_all_folds": mC["total_return_%"] if mC else None,
        "random_same_turnover_sharpe_mean": rnd["sharpe"], "random_same_turnover_sharpe_p95": rnd["sharpe_p95"],
    }
    folds.to_csv(os.path.join(a.out, "folds.csv"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(A["net"].cumsum() * 100, color="#4c78a8", lw=1.4, label="agent, retrained each fold (net of costs)")
    ax.plot(B["net"].cumsum() * 100, color="#9aa5b1", ls="--", lw=1.1, label="buy & hold")
    ax.plot(pd.concat(trend_all)["net"].cumsum() * 100, color="#f58518", lw=0.9, alpha=0.8, label="trend (3 months)")
    if carry_all:
        ax.plot(pd.concat(carry_all)["net"].cumsum() * 100, color="#54a24b", lw=0.9, alpha=0.8, label="carry")
    for _, r in folds.iterrows():
        ax.axvline(pd.Timestamp(r["test_from"]), color="#ddd", lw=0.8)
    ax.axhline(0, color="#ccc", lw=0.6); ax.grid(alpha=0.2)
    ax.set_title(f"Walk-forward, out of sample only: {len(folds)} folds of {a.test_months} months (grey lines = retraining)")
    ax.set_ylabel("cumulative log return, %"); ax.legend(frameon=False, fontsize=8)
    save_plot(fig, os.path.join(a.out, "walk_forward_equity.png"))

    pd.set_option("display.width", 220)
    print(folds.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
