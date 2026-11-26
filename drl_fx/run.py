"""End-to-end pipeline.

    python -m drl_fx.run --demo                                         # synthetic data, quick smoke test
    python -m drl_fx.run --csv data/eurusd_15m_news.csv --fee-bp 0.2    # EUR/USD with IBKR FX commission
    python -m drl_fx.run --csv data/eurusd_15m.csv,data/usdjpy_15m.csv  # one agent, several pairs
    python -m drl_fx.run --csv data/eurusd_15m.csv --transfer-csv data/gbpusd_15m.csv
    python -m drl_fx.run --csv data/eurusd_15m.csv --vs-buy-and-hold    # reward relative to holding long

Steps: load -> clean -> costs -> features -> splits -> train N agents (different windows/seeds)
-> weight them by dev performance -> test -> holdout -> regimes -> transfer -> go-live checklist.

By default the reward is what the position earns, measured against being flat (see env.py), and
agents are ranked by their dev-period Sharpe. The holdout is also scored for three FX benchmarks:
carry (hold the higher-yielding currency), trend (follow the past 3 months) and random trading. With several CSVs, one agent is trained on all
of them (a random instrument per episode) and evaluated on each separately.
Bars per year is measured from the data unless --ppy is given.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import data as D
from .agent import build_agent, train_two_stage
from .env import MultiAssetEnv, TradingEnv
from .evaluate import (add_benchmarks, backtest, benchmark_metrics, buy_and_hold, by_regime,
                       ensemble_weights, go_live_checklist, metrics, random_benchmark)


def prepare(df, trade_notional, fee_bp: float = 0.0):
    df, clean_rep = D.clean(df)
    cost = D.estimate_costs(df, trade_notional) + fee_bp / 1e4   # spread + impact + commission
    f, price_cols, sent_cols = D.make_features(df, cost)
    return f, price_cols, sent_cols, clean_rep


def _name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].split("_")[0].upper()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="one CSV, or several separated by commas")
    p.add_argument("--transfer-csv")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--ppy", type=int, default=0, help="bars per year; 0 = measure from the data")
    p.add_argument("--notional", type=float, default=100_000)
    p.add_argument("--fee-bp", type=float, default=0.0,
                   help="commission per trade in basis points of the amount traded (IBKR FX: about 0.2)")
    p.add_argument("--windows", default="20,40", help="one agent per window size")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--steps-pnl", type=int, default=60_000)
    p.add_argument("--steps-sharpe", type=int, default=30_000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--vs-buy-and-hold", action="store_true",
                   help="reward PnL in excess of holding long instead of plain PnL (for drifting markets)")
    p.add_argument("--out", default="outputs")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    benchmark = 1.0 if a.vs_buy_and_hold else 0.0
    short = not a.no_short

    # ---- 1. data, one entry per instrument
    raws = {}
    if a.demo:
        from .synthetic import make_market
        raws["DEMO"] = make_market(n=120_000, seed=0).set_index("timestamp")
        raw_t = make_market(n=40_000, seed=7).set_index("timestamp")
        a.ppy = a.ppy or 6552                    # demo annualisation kept as in the published demo results
    else:
        for path in a.csv.split(","):
            raws[_name(path.strip())] = D.load_csv(path.strip())
        raw_t = D.load_csv(a.transfer_csv) if a.transfer_csv else None

    assets = {}
    for name, raw in raws.items():
        f, price_cols, sent_cols, rep = prepare(raw, a.notional, a.fee_bp)
        ppy = a.ppy or D.bars_per_year(f)
        f = add_benchmarks(f, ppy)
        assets[name] = dict(f=f, sent=sent_cols, size=D.check_data_size(raw), clean=rep, ppy=ppy)
        print(f"{name}: {assets[name]['size']['rows']:,} rows, {f.index.min():%Y-%m-%d} .. "
              f"{f.index.max():%Y-%m-%d}, {assets[name]['ppy']:,} bars/year, sentiment: {sent_cols}")

    # every instrument must offer the same inputs: keep sentiment columns common to all
    common_sent = sorted(set.intersection(*[set(x["sent"]) for x in assets.values()]))
    obs_cols = D.PRICE_FEATURES + common_sent
    groups = [list(range(len(D.PRICE_FEATURES)))]
    if common_sent:
        groups.append(list(range(len(D.PRICE_FEATURES), len(obs_cols))))

    for x in assets.values():
        sp = D.chronological_splits(x["f"])
        sp["train"], sp["dev"], sp["test"], sp["holdout"] = D.normalize(
            sp["train"], [sp["dev"], sp["test"], sp["holdout"]], obs_cols)
        x["sp"], x["regimes"] = sp, D.label_regimes(x["f"])

    # ---- 2. train several agents that behave differently
    def make_env_fn(w, reward):
        def fn(seed):
            envs = [TradingEnv(x["sp"]["train"], obs_cols, window=w, reward=reward, allow_short=short,
                               seed=seed + 31 * i, benchmark=benchmark) for i, x in enumerate(assets.values())]
            return envs[0] if len(envs) == 1 else MultiAssetEnv(envs, seed=seed)
        return fn

    agents = {}
    for w in [int(x) for x in a.windows.split(",")]:
        for s in range(a.seeds):
            name = f"w{w}_s{s}"
            model = build_agent(make_env_fn(w, "pnl"), w, groups, dropout=a.dropout, seed=s)
            print(f"training {name} ({'vs buy-and-hold' if benchmark else 'raw PnL'} reward) ...", flush=True)
            train_two_stage(model, make_env_fn(w, "sharpe"), a.steps_pnl, a.steps_sharpe, seed=s)
            agents[name] = (model, w)

    # ---- 3. rank agents on the dev period (averaged over instruments)
    key = "excess_sharpe_vs_bh" if benchmark else "sharpe"
    per_agent, score = {}, {}
    for name, (m, w) in agents.items():
        per_agent[name] = {an: {k: metrics(backtest(m, x["sp"][k], obs_cols, w, allow_short=short), x["ppy"])
                                for k in ["train", "dev", "test"]} for an, x in assets.items()}
        score[name] = round(float(np.mean([r["dev"][key] for r in per_agent[name].values()])), 2)
    weights = ensemble_weights(score)
    best = max(score, key=score.get)
    print(f"dev {key}:", score, "-> weights:", weights, "| best:", best)

    # ---- 4. selected agent: all splits, holdout, cost stress, regimes, transfer, checklist
    m, w = agents[best]
    transfer = None
    if raw_t is not None:
        ft, *_ = prepare(raw_t, a.notional, a.fee_bp)
        for c in obs_cols:
            if c not in ft.columns:
                ft[c] = 0.0
        cut = int(len(ft) * 0.6)                 # normalise with its own early history
        ft = D.normalize(ft.iloc[:cut], [ft], obs_cols)[1]
        transfer = metrics(backtest(m, ft.iloc[cut:], obs_cols, w, allow_short=short),
                           a.ppy or D.bars_per_year(ft))

    multi = len(assets) > 1
    summary = {}
    pd.set_option("display.width", 200)
    for an, x in assets.items():
        sp, ppy = x["sp"], x["ppy"]
        hist = {k: backtest(m, sp[k], obs_cols, w, allow_short=short) for k in sp}
        M = {k: metrics(h, ppy) for k, h in hist.items()}
        M["holdout_bh"] = metrics(buy_and_hold(sp["holdout"], w), ppy)
        M["holdout_2x_cost"] = metrics(backtest(m, sp["holdout"], obs_cols, w, cost_mult=2.0,
                                                allow_short=short), ppy)
        for k, v in benchmark_metrics(sp["holdout"], w, ppy).items():
            M[f"holdout_{k}"] = v
        M["holdout_random"] = random_benchmark(hist["holdout"], sp["holdout"], ppy)
        if transfer is not None:
            M["transfer"] = transfer
        reg_tbl = by_regime(pd.concat([hist["test"], hist["holdout"]]), x["regimes"], ppy)
        checks = go_live_checklist({"data_size": x["size"], "metrics": M,
                                    "regimes_test_holdout": reg_tbl}, ppy)
        sfx = f"_{an}" if multi else ""
        pd.DataFrame(M).T.to_csv(os.path.join(a.out, f"metrics{sfx}.csv"))
        reg_tbl.to_csv(os.path.join(a.out, f"regimes_oos{sfx}.csv"))
        checks.to_csv(os.path.join(a.out, f"go_live_checklist{sfx}.csv"), index=False)
        _plot(hist, sp, w, os.path.join(a.out, f"equity{sfx}.png"), title=an if multi else None)
        summary[an] = {"holdout_sharpe": M["holdout"]["sharpe"], "holdout_bh_sharpe": M["holdout_bh"]["sharpe"],
                       "holdout_trend_sharpe": M.get("holdout_trend", {}).get("sharpe"),
                       "holdout_carry_sharpe": M.get("holdout_carry", {}).get("sharpe"),
                       "holdout_random_p95": M["holdout_random"]["sharpe_p95"],
                       "holdout_excess_sharpe": M["holdout"]["excess_sharpe_vs_bh"],
                       "holdout_turnover": M["holdout"]["turnover_per_bar"],
                       "holdout_time_in_market_%": M["holdout"]["time_in_market_%"],
                       "checks_passed": f"{int(checks['pass'].sum())}/{len(checks)}"}

        cols = ["sharpe", "total_return_%", "max_drawdown_%", "turnover_per_bar",
                "time_in_market_%", "cost_drag_%", "carry_%", "corr_with_bh"]
        print(f"\n== {an}: metrics (selected agent {best}) ==\n", pd.DataFrame(M).T[cols])
        print(f"\n== {an}: out-of-sample by regime ==\n", reg_tbl[["sharpe", "excess_sharpe_vs_bh", "bars"]])
        print(f"\n== {an}: go-live checklist ==\n", checks.to_string(index=False))

    # ---- 5. outputs
    m.save(os.path.join(a.out, f"agent_{best}"))
    if multi:
        pd.DataFrame(summary).T.to_csv(os.path.join(a.out, "summary.csv"))
        print("\n== summary ==\n", pd.DataFrame(summary).T)
    with open(os.path.join(a.out, "agents.json"), "w") as fh:
        json.dump({"reward": "excess vs buy-and-hold" if benchmark else "PnL vs flat",
                   "instruments": list(assets), "per_agent": per_agent, f"dev_{key}": score,
                   "ensemble_weights": weights, "selected": best,
                   "cleaning": {an: x["clean"] for an, x in assets.items()}}, fh, indent=2, default=float)


def _plot(hist, sp, w, path, title=None):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    for ax, k in zip(axes, ["train", "dev", "test", "holdout"]):
        eq = hist[k]["net"].cumsum() * 100
        bh = buy_and_hold(sp[k], w)["net"].cumsum() * 100
        ax.plot(eq.index, eq, color="#4c78a8", lw=1.3, label="agent (net of costs)")
        ax.plot(bh.index, bh, color="#9aa5b1", lw=1.0, ls="--", label="buy & hold")
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.set_title(k + (" (never seen)" if k == "holdout" else ""), fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7); ax.grid(alpha=0.2)
    axes[0].set_ylabel("cumulative log return, %")
    axes[0].legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
