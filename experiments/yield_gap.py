"""Does the gap between German and US 2-year yields predict EUR/USD?

Two-year yields price where each central bank is expected to take rates over the next one to two
years, so they move before the policy rates do. Three inputs, fixed before looking at results:

    y_gap          German minus US 2-year yield (EUR minus USD)
    y_gap_chg_20   its change over the past 20 days
    y_slope        y_gap minus the policy rate gap: the expected change in the policy gap

Both yields are used from the day after their date, as published. Tests:
  1. same-day check: do EUR/USD and the gap move together on the same day? (they should; this
     checks the data, not a trading signal)
  2. each input against the next day's move, and a yearly-retrained ridge model with the three
     inputs, alone and added to the price and rate inputs of experiments/predictability.py
  3. two rules: follow the 20-day change of the gap, and follow the slope; net of costs and carry,
     against carry, trend and random trading at the same pace.

    python -m experiments.yield_gap
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from drl_fx.evaluate import metrics, positions_history, random_benchmark
from experiments.predictability import attach, build, read_daily_series

warnings.filterwarnings("ignore")
YIELD_INPUTS = ["y_gap", "y_gap_chg_20", "y_slope"]
PRICE_RATE = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "z_20", "rate_gap", "rate_gap_chg_60"]


def ridge_walk_forward(f: pd.DataFrame, cols: list[str], first_year: int, alpha: float = 10.0) -> pd.Series:
    from sklearn.linear_model import Ridge
    out = []
    for year in range(first_year, f.index.max().year + 1):
        tr = f[f.index < f"{year}-01-01"]
        te = f[(f.index >= f"{year}-01-01") & (f.index < f"{year + 1}-01-01")]
        if len(te) == 0 or len(tr) < 400:
            continue
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        Xtr, Xte = ((tr[cols] - mu) / sd).clip(-5, 5), ((te[cols] - mu) / sd).clip(-5, 5)
        out.append(pd.Series(Ridge(alpha=alpha).fit(Xtr, tr["fwd_ret"]).predict(Xte), index=te.index))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/eurusd_duka_15m.csv")
    ap.add_argument("--de2y", default="data/BBSSY.D.REN.EUR.A610.000000WT0202.A.csv")
    ap.add_argument("--us2y", default="data/DGS2.csv")
    ap.add_argument("--rate-base", default="data/ECBDFR.csv")
    ap.add_argument("--rate-quote", default="data/DFF.csv")
    ap.add_argument("--fee-bp", type=float, default=0.2)
    ap.add_argument("--first-test-year", type=int, default=2016)
    ap.add_argument("--out", default="outputs/yield_gap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    de, us = read_daily_series(a.de2y), read_daily_series(a.us2y)
    start = str(de.index.min().date())
    f, _ = build(a.prices, a.rate_base, a.rate_quote, None, None, a.fee_bp, start)

    # attach needs close_time; rebuild it from the date index (17:00 New York on that date)
    d = f[[]].copy()
    d["close_time"] = d.index + pd.Timedelta(hours=17)
    d.index.name = "date"
    d = attach(d, de, pd.Timedelta(days=1), "de2y")
    d = attach(d, us, pd.Timedelta(days=1), "us2y")
    f["y_gap"] = d["de2y"] - d["us2y"]
    f["y_gap_chg_20"] = f["y_gap"].diff(20)
    f["y_slope"] = f["y_gap"] - f["rate_gap"]
    f = f.dropna()

    # same-day check (not tradable): gap change over the day vs EUR/USD over the same day
    same = pd.DataFrame({"gap": (de - us).dropna()})
    same["d_gap"] = same["gap"].diff()
    same = same.join(f["ret_1"].rename("ret"), how="inner").dropna()
    same_day_corr = round(float(same["d_gap"].corr(same["ret"])), 3)

    oos = f[f.index >= f"{a.first_test_year}-01-01"]
    y = oos["fwd_ret"]
    n = len(y)
    band = round(1.96 / np.sqrt(n), 3)
    ppy = int(round(len(f) / ((f.index.max() - f.index.min()).days / 365.25)))
    print(f"\n{n:,} out-of-sample days {oos.index.min():%Y-%m-%d} .. {oos.index.max():%Y-%m-%d}; "
          f"noise band for correlations ±{band}")
    print(f"Same day: correlation of the change in the 2-year gap with EUR/USD's move = {same_day_corr} "
          f"({len(same):,} days)\n")

    single = {c: round(float(np.corrcoef(oos[c], y)[0, 1]), 3) for c in YIELD_INPUTS}
    print("Each input against the next day's move:", single)

    models = {}
    for name, cols in {"yield inputs only": YIELD_INPUTS, "price + rates": PRICE_RATE,
                       "price + rates + yields": PRICE_RATE + YIELD_INPUTS}.items():
        p = ridge_walk_forward(f, cols, a.first_test_year).reindex(oos.index)
        sse, sse0 = float(((y - p) ** 2).sum()), float((y ** 2).sum())
        models[name] = {"correlation": round(float(np.corrcoef(p, y)[0, 1]), 3),
                        "r2_oos_%": round(100 * (1 - sse / sse0), 3),
                        "hit_rate_%": round(100 * float((np.sign(p) == np.sign(y)).mean()), 1),
                        "_pos": np.sign(p)}
    print("\nRidge, retrained each year:")
    for k, v in models.items():
        print(f"  {k:24s} corr {v['correlation']:+.3f}  R² {v['r2_oos_%']:+.2f}%  hit {v['hit_rate_%']}%")

    rules = {
        "follow 20-day change of the 2y gap": np.sign(oos["y_gap_chg_20"]),
        "follow the slope (expected policy change)": np.sign(oos["y_slope"]),
        "ridge, yield inputs only": models["yield inputs only"]["_pos"],
        "carry: higher policy rate": np.sign(oos["rate_gap"]),
        "trend: last 60 days": np.sign(oos["ret_60"]),
        "buy & hold": pd.Series(1.0, index=oos.index),
    }
    res = {}
    for name, pos in rules.items():
        pos = pos.astype(float).fillna(0.0)
        h = positions_history(pos, oos)
        m = metrics(h, ppy)
        row = {"sharpe": m["sharpe"], "return_%": m["total_return_%"], "max_drawdown_%": m["max_drawdown_%"],
               "trades_per_year": round(float((pos.diff().abs() > 0).sum()) / n * ppy, 1),
               "carry_%": m["carry_%"], "cost_%": m["cost_drag_%"]}
        if name.startswith(("follow", "ridge")):
            rb = random_benchmark(h, oos, ppy, n=200)
            row["random_same_pace_p95"] = rb["sharpe_p95"]
        res[name] = row
    table = pd.DataFrame(res).T
    print("\nTrading it, net of spread, commission and carry:")
    print(table.to_string())
    yearly = pd.DataFrame({k: positions_history(v.astype(float).fillna(0.0), oos)["net"]
                           .groupby(oos.index.year).sum().mul(100).round(2)
                           for k, v in list(rules.items())[:2]})
    print("\nReturn per year, %:\n", yearly.to_string())

    table.to_csv(os.path.join(a.out, "rules.csv"))
    yearly.to_csv(os.path.join(a.out, "yearly_return.csv"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"out_of_sample_days": n, "noise_band": band, "same_day_corr": same_day_corr,
                   "single_input_corr": single,
                   "ridge": {k: {kk: vv for kk, vv in v.items() if kk != "_pos"} for k, v in models.items()},
                   "rules": res}, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
