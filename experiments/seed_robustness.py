"""Do agents trained with different random starts and window sizes behave alike?

    python -m experiments.seed_robustness --csv data/eurusd_15m_news.csv --fee-bp 0.2

Trains one agent per (window, seed) on the same train split as drl_fx.run, scores each on every
split, and compares their holdout positions. If the edge is real, most agents should make money in the
holdout and hold similar positions; if it is luck, results scatter around zero and positions disagree.
Also scores an equal-weight vote of all agents.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from drl_fx import data as D
from drl_fx.evaluate import backtest, benchmark_metrics, buy_and_hold, metrics
from experiments.common import load, save_plot, score_positions, train_agent

G = {}


def _init(csv, fee_bp, benchmark=0.0):
    f, cols, groups, ppy = load(csv, fee_bp)
    sp = D.chronological_splits(f)
    sp["train"], sp["dev"], sp["test"], sp["holdout"] = D.normalize(
        sp["train"], [sp["dev"], sp["test"], sp["holdout"]], cols)
    G.update(sp=sp, cols=cols, groups=groups, ppy=ppy, benchmark=benchmark)


def _job(args):
    window, seed, steps_pnl, steps_sharpe, out = args
    sp, cols = G["sp"], G["cols"]
    model = train_agent(sp["train"], cols, G["groups"], window, seed, steps_pnl, steps_sharpe,
                        benchmark=G["benchmark"])
    name = f"w{window}_s{seed}"
    model.save(os.path.join(out, "models", name))
    res, pos = {}, {}
    for k in ["train", "dev", "test", "holdout"]:
        h = backtest(model, sp[k], cols, window)
        res[k] = metrics(h, G["ppy"])
        pos[k] = h["pos"]
    print(f"  done {name}: holdout Sharpe {res['holdout']['sharpe']}", flush=True)
    return name, res, pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--fee-bp", type=float, default=0.0)
    p.add_argument("--windows", default="20,40,80")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps-pnl", type=int, default=200_000)
    p.add_argument("--steps-sharpe", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="outputs/seed_robustness")
    p.add_argument("--vs-buy-and-hold", action="store_true", help="reward relative to holding long")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)

    jobs = [(int(w), s, a.steps_pnl, a.steps_sharpe, a.out)
            for w in a.windows.split(",") for s in range(a.seeds)]
    print(f"Training {len(jobs)} agents with {a.workers} workers ...", flush=True)
    bench = 1.0 if a.vs_buy_and_hold else 0.0
    with Pool(a.workers, initializer=_init, initargs=(a.csv, a.fee_bp, bench)) as pool:
        results = pool.map(_job, jobs, chunksize=1)

    _init(a.csv, a.fee_bp, bench)                # the main process needs the data too
    sp, ppy = G["sp"], G["ppy"]

    # ---- per-agent table
    rows = {}
    for name, res, pos in results:
        hp = pos["holdout"]
        rows[name] = {f"{k}_sharpe": res[k]["sharpe"] for k in ["train", "dev", "test", "holdout"]}
        rows[name].update({"holdout_excess_vs_bh": res["holdout"]["excess_sharpe_vs_bh"],
                           "holdout_turnover": res["holdout"]["turnover_per_bar"],
                           "holdout_long_%": round(100 * float((hp > 0).mean()), 1),
                           "holdout_short_%": round(100 * float((hp < 0).mean()), 1)})
    table = pd.DataFrame(rows).T
    bh = {k: metrics(buy_and_hold(sp[k], 20), ppy)["sharpe"] for k in ["train", "dev", "test", "holdout"]}
    table.loc["buy_and_hold"] = {f"{k}_sharpe": v for k, v in bh.items()}
    for name in ["trend", "carry"]:
        row = {}
        for k in ["train", "dev", "test", "holdout"]:
            bm = benchmark_metrics(sp[k], 20, ppy)
            if name in bm:
                row[f"{k}_sharpe"] = bm[name]["sharpe"]
        if row:
            table.loc[f"{name}_strategy"] = row

    # ---- do they agree? correlation of holdout positions between agents
    P = pd.DataFrame({name: pos["holdout"] for name, _, pos in results}).dropna()
    corr = P.corr()
    off = corr.values[np.triu_indices_from(corr.values, k=1)]
    off = off[~np.isnan(off)]                     # an agent that never changes position has no correlation
    same_side = float((np.sign(P).nunique(axis=1) == 1).mean())

    # ---- equal-weight vote: each bar, the side most agents are on (flat on a tie)
    vote = np.sign(P.mean(axis=1))
    ens = metrics(score_positions(vote, sp["holdout"]), ppy)
    table.loc["vote_of_all_agents", ["holdout_sharpe", "holdout_excess_vs_bh", "holdout_turnover"]] = \
        [ens["sharpe"], ens["excess_sharpe_vs_bh"], ens["turnover_per_bar"]]

    agents = table.loc[[n for n, _, _ in results]]
    summary = {
        "agents": len(agents),
        "holdout_sharpe_median": float(agents["holdout_sharpe"].median()),
        "holdout_sharpe_min": float(agents["holdout_sharpe"].min()),
        "holdout_sharpe_max": float(agents["holdout_sharpe"].max()),
        "agents_with_positive_holdout": int((agents["holdout_sharpe"] > 0).sum()),
        "agents_with_positive_test": int((agents["test_sharpe"] > 0).sum()),
        "agents_positive_in_both_test_and_holdout": int(((agents["test_sharpe"] > 0) & (agents["holdout_sharpe"] > 0)).sum()),
        "mean_pairwise_position_correlation": round(float(off.mean()), 3) if len(off) else None,
        "agents_that_never_change_position": int((P.std() == 0).sum()),
        "share_of_bars_all_agents_on_same_side": round(same_side, 3),
        "vote_holdout_sharpe": ens["sharpe"],
        "buy_and_hold_holdout_sharpe": bh["holdout"],
    }
    table.to_csv(os.path.join(a.out, "agents.csv"))
    corr.round(3).to_csv(os.path.join(a.out, "holdout_position_correlation.csv"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- chart: holdout equity of every agent, the vote and buy-and-hold
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, _, pos in results:
        h = score_positions(pos["holdout"], sp["holdout"])
        ax.plot(h["net"].cumsum() * 100, lw=0.9, alpha=0.6, label=name)
    ax.plot(score_positions(vote, sp["holdout"])["net"].cumsum() * 100, color="black", lw=1.8, label="vote")
    bhh = buy_and_hold(sp["holdout"], 20)
    ax.plot(bhh["net"].cumsum() * 100, color="grey", ls="--", lw=1.2, label="buy & hold")
    ax.axhline(0, color="#ccc", lw=0.6); ax.grid(alpha=0.2)
    ax.set_title("Holdout period: each agent (window_seed), their vote, and buy & hold")
    ax.set_ylabel("cumulative log return, %"); ax.legend(ncol=6, fontsize=7, frameon=False)
    save_plot(fig, os.path.join(a.out, "holdout_equity.png"))

    pd.set_option("display.width", 220)
    print(table.round(2).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
