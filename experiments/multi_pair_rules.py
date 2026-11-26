"""Carry, trend and mean reversion on six currency pairs, and whether weighting them by recent
performance beats weighting them equally (the course's suggestion: trade the algos that did best
in the last few months, or give them more weight).

Everything is daily (FX days ending 17:00 New York), net of half the bid/ask spread plus
commission, with interest carry. Each rule is sized to a 6% annual volatility target from the
past 60 days. Weights are set at each month end from past data only and held for the next month.

    python -m experiments.multi_pair_rules

Inputs in data/: eurusd_duka_15m.csv and the five *_duka_1h.csv files from download_dukascopy.py,
the BIS policy rate file (data.bis.org bulk download, WS_CBPOL_csv_col.csv), ECBDFR.csv (the ECB
deposit rate is closer to euro money-market rates than the main refinancing rate the BIS lists)
and VIXCLS.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from experiments.predictability import FX_DAY_SHIFT, attach, read_daily_series, to_daily

warnings.filterwarnings("ignore")

PAIRS = {  # pair -> (price file, base currency, quote currency)
    "EURUSD": ("eurusd_duka_15m.csv", "EUR", "USD"),
    "GBPUSD": ("gbpusd_duka_1h.csv", "GBP", "USD"),
    "AUDUSD": ("audusd_duka_1h.csv", "AUD", "USD"),
    "USDJPY": ("usdjpy_duka_1h.csv", "USD", "JPY"),
    "USDCHF": ("usdchf_duka_1h.csv", "USD", "CHF"),
    "USDCAD": ("usdcad_duka_1h.csv", "USD", "CAD"),
}
BIS_AREA = {"USD": "US", "GBP": "GB", "JPY": "JP", "AUD": "AU", "CHF": "CH", "CAD": "CA", "EUR": "XM"}
RULES = ["carry", "trend", "reversion"]
COMMISSION = 0.2e-4          # IBKR FX: 0.2 basis points of the amount traded
TREND_DAYS, VOL_DAYS, Z_DAYS = 63, 60, 20


# ------------------------------------------------------------------ data
def read_bis(path: str) -> dict[str, pd.Series]:
    """BIS policy rates, wide 'csv_col' layout -> one daily series per reference area."""
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["FREQ"] == "D"]
    days = [c for c in raw.columns if len(c) == 10 and c[4] == "-" and c[7] == "-"]
    out = {}
    for _, row in raw.iterrows():
        s = pd.to_numeric(row[days], errors="coerce")
        s.index = pd.to_datetime(days)
        out[row["REF_AREA"]] = s.dropna()
    return out


def load_prices(path: str) -> pd.DataFrame:
    px = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    step = px.index.to_series().diff().median()
    per_day = pd.Timedelta(days=1) / step
    return to_daily(px, min_bars=int(per_day * 0.4))      # same 40% rule as for 15-minute bars


def rate_series(ccy: str, bis: dict, ecb: pd.Series) -> pd.Series:
    if ccy == "EUR":
        return ecb
    return bis[BIS_AREA[ccy]]


def pair_frame(pair: str, data_dir: str, bis: dict, ecb: pd.Series, vix: pd.Series) -> pd.DataFrame:
    fname, base, quote = PAIRS[pair]
    d = load_prices(os.path.join(data_dir, fname))
    d = attach(d, rate_series(base, bis, ecb), pd.Timedelta(days=1), "rate_base")
    d = attach(d, rate_series(quote, bis, ecb), pd.Timedelta(days=1), "rate_quote")
    d = attach(d, vix, pd.Timedelta(days=1), "vix")
    d[["rate_base", "rate_quote"]] = d[["rate_base", "rate_quote"]].ffill()   # policy rates are step functions
    lc = np.log(d["close"])
    d["fwd_ret"] = lc.shift(-1) - lc                                  # earned by the position held at this close
    held = (d["close_time"].shift(-1) - d["close_time"]).dt.total_seconds() / 86400
    held = held.where(held <= 5)
    d["carry_long"] = (d["rate_base"] - d["rate_quote"]) / 100 * held / 365
    d["cost"] = d["spread"].rolling(20, min_periods=1).median() / 2 + COMMISSION
    ret = lc.diff()
    vol = ret.rolling(VOL_DAYS).std() * np.sqrt(252)
    d["size"] = (0.06 / vol).clip(upper=3.0)
    z = (lc - lc.rolling(Z_DAYS).mean()) / lc.rolling(Z_DAYS).std()
    d["sig_carry"] = np.sign(d["rate_base"] - d["rate_quote"])
    d["sig_trend"] = np.sign(lc - lc.shift(TREND_DAYS))
    d["sig_reversion"] = np.where(z > 1, -1.0, np.where(z < -1, 1.0, 0.0))
    return d.dropna(subset=["fwd_ret", "carry_long", "size", "cost"])


def strategy_returns(d: pd.DataFrame, rule: str, vix_filter: bool = False) -> pd.DataFrame:
    pos = d[f"sig_{rule}"].fillna(0) * d["size"]
    if vix_filter:
        calm = d["vix"] <= d["vix"].rolling(252, min_periods=126).quantile(0.8)
        pos = pos.where(calm, 0.0)
    trade = pos.diff().abs().fillna(pos.abs())
    gross = pos * d["fwd_ret"]
    carry = pos * d["carry_long"]
    cost = trade * d["cost"]
    return pd.DataFrame({"net": gross + carry - cost, "carry": carry, "cost": cost, "pos": pos})


# ------------------------------------------------------------------ statistics
def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def summarise(x: pd.Series, extra: dict | None = None) -> dict:
    x = x.dropna()
    eq = x.cumsum()
    out = {"sharpe": round(sharpe(x), 2), "return_%_a_year": round(x.mean() * 252 * 100, 2),
           "vol_%": round(x.std() * np.sqrt(252) * 100, 2),
           "max_drawdown_%": round(float((eq - eq.cummax()).min()) * 100, 1)}
    if extra:
        out.update(extra)
    return out


def newey_west_t(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, m = len(x), x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(m / np.sqrt(s / n))


# ------------------------------------------------------------------ weighting
def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().values)


def trailing_sharpe(R: pd.DataFrame, end: pd.Timestamp, months: int) -> pd.Series:
    win = R.loc[(R.index > end - pd.DateOffset(months=months)) & (R.index <= end)]
    sd = win.std()
    return (win.mean() / sd * np.sqrt(252)).where(sd > 0, 0.0).fillna(0.0)


def weight_schedule(R: pd.DataFrame, scheme: str, months: int = 3, rng=None, top: int = 3) -> pd.DataFrame:
    """Weights per day; set at each month end from data up to that day, used the following month."""
    ends = month_ends(R.index)
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    n = R.shape[1]
    for i, e in enumerate(ends[:-1]):
        if scheme == "equal":
            w = pd.Series(1.0 / n, index=R.columns)
        else:
            s = trailing_sharpe(R, e, months)
            if scheme == "recent":
                w = s.clip(lower=0)
                w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / n, index=R.columns)
            elif scheme == "top":
                w = pd.Series(0.0, index=R.columns)
                w[s.nlargest(top).index] = 1.0 / top
            if rng is not None:                       # same weights, given to strategies at random
                w = pd.Series(rng.permutation(w.values), index=R.columns)
        nxt = (R.index > e) & (R.index <= ends[i + 1])
        W.loc[nxt] = w.values
    return W


def portfolio(R: pd.DataFrame, W: pd.DataFrame) -> pd.Series:
    return (R * W).sum(axis=1, min_count=1).where(W.notna().any(axis=1))


def persistence(R: pd.DataFrame, months: int) -> dict:
    """Do strategies that did well over the past `months` do better next month? Rank correlation per month."""
    ends = month_ends(R.index)
    rows = []
    for i, e in enumerate(ends[:-1]):
        if e - pd.DateOffset(months=months) < R.index[0]:
            continue
        past = trailing_sharpe(R, e, months)
        nxt = R.loc[(R.index > e) & (R.index <= ends[i + 1])].sum()
        rows.append(past.rank().corr(nxt.rank()))
    rho = np.array(rows)
    return {"months": len(rho), "mean_rank_corr": round(float(np.nanmean(rho)), 3),
            "t_stat": round(newey_west_t(rho, max(months - 1, 1)), 2),
            "share_positive_%": round(float(np.mean(rho > 0) * 100), 1)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--bis", default=None, help="default: data/WS_CBPOL_csv_col.csv or data/WS_CBPOL_csv_col/WS_CBPOL_csv_col.csv")
    ap.add_argument("--start", default="2013-01-01", help="first day scored (2012 is warm-up)")
    ap.add_argument("--random-draws", type=int, default=500)
    ap.add_argument("--out", default="outputs/multi_pair_rules")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bis_path = a.bis or next(p for p in [os.path.join(a.data, "WS_CBPOL_csv_col.csv"),
                                         os.path.join(a.data, "WS_CBPOL_csv_col", "WS_CBPOL_csv_col.csv")]
                             if os.path.exists(p))
    bis = read_bis(bis_path)
    ecb = read_daily_series(os.path.join(a.data, "ECBDFR.csv"))
    vix = read_daily_series(os.path.join(a.data, "VIXCLS.csv"))

    frames = {p: pair_frame(p, a.data, bis, ecb, vix) for p in PAIRS}
    cols, carry_vix = {}, {}
    detail = {}
    for p, d in frames.items():
        for r in RULES:
            s = strategy_returns(d, r)
            cols[f"{r} {p}"] = s["net"]
            detail[f"{r} {p}"] = s
        carry_vix[p] = strategy_returns(d, "carry", vix_filter=True)["net"]
    R = pd.DataFrame(cols).sort_index()
    R = R.loc[R.index >= pd.Timestamp(a.start) - pd.DateOffset(months=12)].fillna(0.0)
    score = R.index >= pd.Timestamp(a.start)
    years = score.sum() / 252
    print(f"{R.shape[1]} strategies, scored {R.index[score][0].date()} to {R.index[-1].date()} "
          f"({years:.1f} years; Sharpe standard error about {1 / np.sqrt(years):.2f})\n")

    # single strategies
    rows = {}
    for c in R.columns:
        s = detail[c].reindex(R.index[score])
        rows[c] = summarise(R.loc[score, c], {"carry_%_a_year": round(s["carry"].mean() * 252 * 100, 2),
                                              "cost_%_a_year": round(s["cost"].mean() * 252 * 100, 2)})
    strat = pd.DataFrame(rows).T
    strat.to_csv(os.path.join(a.out, "strategies.csv"))
    print(strat[["sharpe", "return_%_a_year", "carry_%_a_year", "cost_%_a_year"]].to_string(), "\n")

    # baskets per rule, VIX filter
    baskets = {f"{r} (6 pairs)": R[[c for c in R.columns if c.startswith(r)]].mean(axis=1) for r in RULES}
    cv = pd.DataFrame(carry_vix).reindex(R.index).fillna(0.0)
    baskets["carry (6 pairs), off when VIX is high"] = cv.mean(axis=1)

    # ensembles
    ens = {"equal weights (18)": portfolio(R, weight_schedule(R, "equal"))}
    for m in (3, 6, 12):
        ens[f"weighted by last {m} months"] = portfolio(R, weight_schedule(R, "recent", m))
    ens["best 3 of last 3 months"] = portfolio(R, weight_schedule(R, "top", 3))

    rng = np.random.default_rng(0)
    rand = {k: [] for k in ("recent", "top")}
    for _ in range(a.random_draws):
        rand["recent"].append(sharpe(portfolio(R, weight_schedule(R, "recent", 3, rng=rng))[score]))
        rand["top"].append(sharpe(portfolio(R, weight_schedule(R, "top", 3, rng=rng))[score]))
    p95 = {k: round(float(np.percentile(v, 95)), 2) for k, v in rand.items()}
    med = {k: round(float(np.median(v)), 2) for k, v in rand.items()}

    table = {k: summarise(v[score]) for k, v in {**baskets, **ens}.items()}
    table["weighted by last 3 months"].update({"same weights at random: median": med["recent"], "p95": p95["recent"]})
    table["best 3 of last 3 months"].update({"same weights at random: median": med["top"], "p95": p95["top"]})
    ens_df = pd.DataFrame(table).T
    ens_df.to_csv(os.path.join(a.out, "ensembles.csv"))
    print(ens_df.to_string(), "\n")

    pers = {f"last {m} months": persistence(R.loc[R.index >= pd.Timestamp(a.start) - pd.DateOffset(months=m)], m)
            for m in (1, 3, 6, 12)}
    pers_df = pd.DataFrame(pers).T
    pers_df.to_csv(os.path.join(a.out, "persistence.csv"))
    print("Does recent performance carry over to the next month? (rank correlation across the 18 strategies)")
    print(pers_df.to_string(), "\n")

    # yearly Sharpe of the main lines
    yearly = pd.DataFrame({k: v[score].groupby(v[score].index.year).apply(sharpe)
                           for k, v in {**baskets, "equal weights (18)": ens["equal weights (18)"],
                                        "weighted by last 3 months": ens["weighted by last 3 months"]}.items()}).round(2)
    yearly.to_csv(os.path.join(a.out, "yearly_sharpe.csv"))
    print(yearly.to_string())

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"years_scored": round(years, 1), "sharpe_standard_error": round(1 / np.sqrt(years), 2),
                   "ensembles": table, "persistence": pers}, fh, indent=2, default=float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in ["carry (6 pairs)", "trend (6 pairs)", "reversion (6 pairs)", "carry (6 pairs), off when VIX is high"]:
            ax.plot(baskets[k][score].cumsum() * 100, label=k, lw=1)
        for k in ["equal weights (18)", "weighted by last 3 months"]:
            ax.plot(ens[k][score].cumsum() * 100, label=k, lw=1.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("cumulative return, %")
        ax.set_title("Rules on six FX pairs, net of costs and carry")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "equity.png"), dpi=120)
    except Exception as err:                                  # chart is optional
        print("chart skipped:", err)


if __name__ == "__main__":
    main()
