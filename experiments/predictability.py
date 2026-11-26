"""Do the inputs predict tomorrow's move at all? A cheap check before spending hours on RL training.

    python -m experiments.predictability --prices data/eurusd_duka_15m.csv \
        --rate-base data/ECBDFR.csv --rate-quote data/DFF.csv \
        --epu data/USEPUINDXD.csv --sffed data/news_sentiment_data.xlsx

1. Rolls the 15-minute bars into FX trading days, which close at 17:00 New York time.
2. Builds daily inputs, each only from information public at that day's close:
   past returns (1, 5, 20, 60 days), volatility, distance from the 20-day average,
   the interest-rate gap and its 60-day change, the EPU index and the SF Fed news index.
3. Walk-forward: every January, fits a linear model (ridge) and a direction model (logistic)
   on all earlier days and predicts every day of that year. Nothing is fitted on the year it predicts.
4. Scores the predictions: out-of-sample R² (against predicting zero), correlation with the actual
   move, share of days with the right direction, and a simple strategy (long if the model expects a
   rise, short otherwise) net of spread, commission and carry, next to carry, trend, buy-and-hold
   and random trading with the same turnover.

If the models cannot beat these yardsticks, an RL agent given the same inputs is unlikely to either.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from drl_fx.evaluate import metrics, positions_history, random_benchmark

FX_DAY_SHIFT = pd.Timedelta(hours=7)       # 17:00 New York + 7h = midnight: label each FX day by its closing date


# ---------------------------------------------------------------- data
def to_daily(px: pd.DataFrame, min_bars: int = 40) -> pd.DataFrame:
    """15-minute bars (New York time) -> FX trading days ending 17:00 New York time."""
    day = (px.index + FX_DAY_SHIFT).normalize()
    spread = (px["ask"] - px["bid"]) / ((px["ask"] + px["bid"]) / 2) if {"bid", "ask"} <= set(px.columns) else None
    g = px.groupby(day)
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "bars": g["close"].size()})
    if spread is not None:
        d["spread"] = spread.groupby(day).median()
    d = d[d["bars"] >= min_bars]                       # drop holidays and half-empty days
    d.index.name = "date"
    d["close_time"] = d.index - FX_DAY_SHIFT + pd.Timedelta(days=1)   # 17:00 New York on the closing date
    return d


def read_daily_series(path: str) -> pd.Series:
    """Two-column daily file (date + value), CSV or the SF Fed Excel file."""
    from add_daily_features import _read_daily
    return _read_daily(path)


def attach(d: pd.DataFrame, s: pd.Series, delay: pd.Timedelta, name: str) -> pd.DataFrame:
    """Value for date x becomes usable at x + delay; each day sees the latest usable value at its close."""
    avail = s.copy()
    avail.index = avail.index.normalize() + delay
    avail = avail.rename(name).to_frame().sort_index()
    out = pd.merge_asof(d.reset_index().sort_values("close_time"), avail, left_on="close_time",
                        right_index=True, direction="backward")
    return out.set_index("date")


def build(prices: str, rate_base: str, rate_quote: str, epu: str | None, sffed: str | None,
          fee_bp: float, start: str) -> tuple[pd.DataFrame, list[str]]:
    px = pd.read_csv(prices, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    px = px[px.index >= pd.Timestamp(start) - pd.Timedelta(days=120)]      # warm-up for 60-day inputs
    d = to_daily(px)
    d = attach(d, read_daily_series(rate_base), pd.Timedelta(days=1), "rate_base")
    d = attach(d, read_daily_series(rate_quote), pd.Timedelta(days=1), "rate_quote")
    if epu:
        e = read_daily_series(epu)
        e = np.log(e.rolling(7, min_periods=3).mean())
        d = attach(d, e, pd.Timedelta(days=1, hours=9, minutes=30), "epu")
    if sffed:
        d = attach(d, read_daily_series(sffed), pd.Timedelta(days=7), "sffed")

    lc = np.log(d["close"])
    r = lc.diff()
    f = pd.DataFrame(index=d.index)
    f["ret_1"], f["ret_5"], f["ret_20"], f["ret_60"] = r, lc.diff(5), lc.diff(20), lc.diff(60)
    f["vol_20"] = r.rolling(20).std()
    f["z_20"] = (d["close"] - d["close"].rolling(20).mean()) / d["close"].rolling(20).std()
    f["rate_gap"] = d["rate_base"] - d["rate_quote"]
    f["rate_gap_chg_60"] = f["rate_gap"].diff(60)
    if epu:
        f["epu"] = d["epu"]
        f["epu_chg_20"] = d["epu"].diff(20)
    if sffed:
        f["sffed"] = d["sffed"]
        f["sffed_chg_20"] = d["sffed"].diff(20)
    inputs = list(f.columns)

    # what a position held from this close to the next one earns or pays
    days_held = (d["close_time"].shift(-1) - d["close_time"]).dt.total_seconds() / 86400
    f["fwd_ret"] = r.shift(-1)
    f["carry"] = f["rate_gap"] / 100 * days_held / 365
    f["cost"] = (d["spread"] / 2 if "spread" in d else 0.5e-4) + fee_bp / 1e4
    f = f[f.index >= start].replace([np.inf, -np.inf], np.nan).dropna()
    return f, inputs


# ---------------------------------------------------------------- models
def _standardise(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return ((train[cols] - mu) / sd).clip(-5, 5), ((test[cols] - mu) / sd).clip(-5, 5)


def walk_forward(f: pd.DataFrame, inputs: list[str], first_test_year: int, alpha: float = 10.0):
    from sklearn.linear_model import LogisticRegression, Ridge
    preds = []
    for year in range(first_test_year, f.index.max().year + 1):
        train = f[f.index < f"{year}-01-01"]
        test = f[(f.index >= f"{year}-01-01") & (f.index < f"{year + 1}-01-01")]
        if len(test) == 0 or len(train) < 500:
            continue
        Xtr, Xte = _standardise(train, test, inputs)
        ridge = Ridge(alpha=alpha).fit(Xtr, train["fwd_ret"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logit = LogisticRegression(C=0.1, max_iter=1000).fit(Xtr, (train["fwd_ret"] > 0).astype(int))
        preds.append(pd.DataFrame({"year": year, "pred_ridge": ridge.predict(Xte),
                                   "p_up": logit.predict_proba(Xte)[:, 1]}, index=test.index))
    return pd.concat(preds)


def score(y: pd.Series, pred: pd.Series) -> dict:
    sse_model = float(((y - pred) ** 2).sum())
    sse_zero = float((y ** 2).sum())
    return {"r2_oos_%": round(100 * (1 - sse_model / sse_zero), 3),
            "correlation": round(float(np.corrcoef(pred, y)[0, 1]), 3),
            "hit_rate_%": round(100 * float((np.sign(pred) == np.sign(y)).mean()), 1),
            "days": int(len(y))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--rate-base", required=True)
    p.add_argument("--rate-quote", required=True)
    p.add_argument("--epu")
    p.add_argument("--sffed")
    p.add_argument("--fee-bp", type=float, default=0.2)
    p.add_argument("--start", default="2012-04-01")
    p.add_argument("--first-test-year", type=int, default=2016)
    p.add_argument("--out", default="outputs/predictability")
    p.add_argument("--daily-csv", default=None, help="also save the daily dataset here")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    f, inputs = build(a.prices, a.rate_base, a.rate_quote, a.epu, a.sffed, a.fee_bp, a.start)
    ppy = int(round(len(f) / ((f.index.max() - f.index.min()).days / 365.25)))
    print(f"{len(f):,} trading days, {f.index.min():%Y-%m-%d} .. {f.index.max():%Y-%m-%d}, {ppy} per year")
    print("inputs:", inputs)
    if a.daily_csv:
        f.to_csv(a.daily_csv)

    P = walk_forward(f, inputs, a.first_test_year)
    oos = f.loc[P.index]
    y = oos["fwd_ret"]
    n = len(y)
    band = round(1.96 / np.sqrt(n), 3)

    # ---- 1. forecast quality
    quality = {"ridge (expected return)": score(y, P["pred_ridge"]),
               "logistic (direction)": score(y, P["p_up"] - 0.5)}
    quality["logistic (direction)"].pop("r2_oos_%")

    # ---- 2. each input on its own, out-of-sample years only: does it line up with tomorrow's move?
    single = pd.DataFrame({c: {"correlation_next_day": round(float(np.corrcoef(oos[c], y)[0, 1]), 3),
                               "years_with_same_sign": int(sum(
                                   np.sign(np.corrcoef(g[c], g["fwd_ret"])[0, 1]) ==
                                   np.sign(np.corrcoef(oos[c], y)[0, 1])
                                   for _, g in oos.groupby(oos.index.year) if g[c].std() > 0))}
                           for c in inputs}).T
    n_years = oos.index.year.nunique()

    # ---- 3. trading the forecasts, net of costs and carry
    strategies = {
        "ridge: long if expected rise": np.sign(P["pred_ridge"]),
        "logistic: long if P(up) > 0.5": np.sign(P["p_up"] - 0.5),
        "carry: hold the higher-yielding currency": np.sign(oos["rate_gap"]),
        "trend: direction of last 60 days": np.sign(oos["ret_60"]),
        "buy & hold": pd.Series(1.0, index=oos.index),
    }
    res = {}
    for name, pos in strategies.items():
        h = positions_history(pos.astype(float), oos)
        m = metrics(h, ppy)
        res[name] = {"sharpe": m["sharpe"], "return_%": m["total_return_%"], "max_drawdown_%": m["max_drawdown_%"],
                     "trades_per_year": round(float((pos.diff().abs() > 0).sum()) / n * ppy, 1),
                     "cost_%": m["cost_drag_%"], "carry_%": m["carry_%"]}
    for name in ["ridge: long if expected rise", "logistic: long if P(up) > 0.5"]:
        h = positions_history(strategies[name].astype(float), oos)
        rb = random_benchmark(h, oos, ppy, n=200)
        res[name]["random_same_turnover_p95"] = rb["sharpe_p95"]
    trading = pd.DataFrame(res).T
    sharpe_se = round(float(np.sqrt(ppy / n)), 2)     # standard error of an annualised Sharpe ratio near zero

    # ---- yearly view
    yearly = pd.DataFrame({yr: {
        "ridge_correlation": round(float(np.corrcoef(P.loc[g.index, "pred_ridge"], g["fwd_ret"])[0, 1]), 3),
        "ridge_hit_%": round(100 * float((np.sign(P.loc[g.index, "pred_ridge"]) == np.sign(g["fwd_ret"])).mean()), 1),
        "ridge_strategy_sharpe": metrics(positions_history(np.sign(P.loc[g.index, "pred_ridge"]).astype(float), g), ppy)["sharpe"],
        "buy_hold_sharpe": metrics(positions_history(pd.Series(1.0, index=g.index), g), ppy)["sharpe"],
    } for yr, g in oos.groupby(oos.index.year)}).T

    summary = {"days_out_of_sample": n, "years_out_of_sample": n_years,
               "from": f"{oos.index.min():%Y-%m-%d}", "to": f"{oos.index.max():%Y-%m-%d}",
               "correlation_noise_band_95": band, "sharpe_standard_error": sharpe_se,
               "forecast_quality": quality, "trading": res}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    single.to_csv(os.path.join(a.out, "inputs_one_by_one.csv"))
    trading.to_csv(os.path.join(a.out, "trading.csv"))
    yearly.to_csv(os.path.join(a.out, "yearly.csv"))
    P.to_csv(os.path.join(a.out, "predictions.csv"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colours = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#9aa5b1"]
    for (name, pos), c in zip(strategies.items(), colours):
        h = positions_history(pos.astype(float), oos)
        ax.plot(h["net"].cumsum() * 100, color=c, lw=1.3 if "ridge" in name or "logistic" in name else 1.0,
                ls="--" if name == "buy & hold" else "-", label=name)
    ax.axhline(0, color="#ccc", lw=0.6); ax.grid(alpha=0.2)
    ax.set_title(f"EUR/USD daily, out of sample {oos.index.min():%Y}–{oos.index.max():%Y}, net of costs and carry")
    ax.set_ylabel("cumulative log return, %"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "equity.png"), dpi=130)

    pd.set_option("display.width", 220)
    print(f"\nOut of sample: {n:,} days in {n_years} years. Correlations within ±{band} are noise; "
          f"Sharpe ratios have a standard error of about {sharpe_se}.")
    print("\nForecast quality:\n", pd.DataFrame(quality).T.to_string())
    print("\nEach input on its own:\n", single.to_string(), f"\n(out of {n_years} years)")
    print("\nTrading the forecasts:\n", trading.to_string())
    print("\nBy year:\n", yearly.to_string())


if __name__ == "__main__":
    main()
