"""Add daily news indices to a 15-minute price CSV, delayed to when they were really available.

    python add_daily_features.py --prices data/eurusd_15m.csv --sffed data/news_sentiment_data.xlsx --epu data/USEPUINDXD.csv
    -> writes data/eurusd_15m_news.csv (same bars plus sentiment_sffed and sentiment_epu columns)

Sources (download in a browser):
  EPU    FRED series USEPUINDXD, "Download" -> CSV        https://fred.stlouisfed.org/series/USEPUINDXD
  SF Fed Daily News Sentiment Index, "Download Data"       https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/

Interest rates for carry (both from FRED, daily, in percent):
  --rate-base   rate of the first currency of the pair  (EUR/USD: ECBDFR, ECB deposit facility rate)
  --rate-quote  rate of the second currency             (EUR/USD: DFF, US federal funds effective rate)
  -> columns rate_base and rate_quote. drl_fx turns them into the overnight carry of a position
     and gives the agent their difference as an input (macro_rate_diff).

    python add_daily_features.py --prices data/eurusd_15m.csv --sffed data/news_sentiment_data.xlsx \
        --rate-base data/ECBDFR.csv --rate-quote data/DFF.csv

Publication delays (so the backtest never sees a value before it was published):
  EPU    the value for day D is posted on FRED the next morning -> usable from 09:30 New York time on day D+1
  SF Fed the index is updated once a week -> a value for day D is treated as usable 7 days later
  Rates  the rate for day D is published the next day -> usable from midnight New York time on day D+1
Each value is carried forward bar by bar until the next one becomes available.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd


def _date_col(df: pd.DataFrame):
    for c in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = len(df) and pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9
        if ok:
            return c
    return None


def _read_daily(path: str) -> pd.Series:
    """Read a two-column daily file (date + value) from CSV or Excel, whatever the column names."""
    if path.lower().endswith((".xlsx", ".xls")):
        # use the first sheet that has a date column (the SF Fed file starts with a "Methodology" sheet)
        sheets = pd.read_excel(path, sheet_name=None)
        df = next((d for d in sheets.values() if _date_col(d) is not None), None)
        if df is None:
            raise ValueError(f"No sheet with a date column in {path}: {list(sheets)}")
    else:
        df = pd.read_csv(path)
    date_col = _date_col(df)
    if date_col is None:
        raise ValueError(f"No date column found in {path}: {list(df.columns)}")
    # keep rows with a real date: drops metadata rows such as the Bundesbank's header block
    df = df[pd.to_datetime(df[date_col], errors="coerce", format="mixed").notna()]
    num = [c for c in df.columns if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    if not num:
        raise ValueError(f"No numeric column found in {path}: {list(df.columns)}")
    s = pd.Series(pd.to_numeric(df[num[0]], errors="coerce").values,
                  index=pd.to_datetime(df[date_col], errors="coerce"), name=num[0])
    s = s[s.index.notna()].dropna().sort_index()
    print(f"{os.path.basename(path)}: column '{num[0]}', {len(s):,} days, {s.index.min():%Y-%m-%d} .. {s.index.max():%Y-%m-%d}")
    return s


def attach(prices: pd.DataFrame, daily: pd.Series, delay: pd.Timedelta, name: str) -> pd.DataFrame:
    """Stamp each daily value with the time it became usable, then carry it forward onto the bars."""
    avail = daily.copy()
    avail.index = avail.index.normalize() + delay
    avail = avail.rename(name).to_frame().sort_index()
    out = pd.merge_asof(prices.sort_index(), avail, left_index=True, right_index=True, direction="backward")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--epu")
    p.add_argument("--sffed")
    p.add_argument("--rate-base", help="daily interest rate of the first currency, e.g. data/ECBDFR.csv")
    p.add_argument("--rate-quote", help="daily interest rate of the second currency, e.g. data/DFF.csv")
    p.add_argument("--out")
    a = p.parse_args()

    px = pd.read_csv(a.prices, parse_dates=["timestamp"]).set_index("timestamp").sort_index()

    if a.epu:
        epu = _read_daily(a.epu)
        # very noisy day to day: use the log of a 7-day average
        epu = np.log(epu.rolling(7, min_periods=3).mean()).dropna()
        # value for day D usable from 09:30 New York time on day D+1
        px = attach(px, epu, pd.Timedelta(days=1, hours=9, minutes=30), "sentiment_epu")
    if a.sffed:
        sf = _read_daily(a.sffed)
        px = attach(px, sf, pd.Timedelta(days=7), "sentiment_sffed")

    if bool(a.rate_base) != bool(a.rate_quote):
        p.error("--rate-base and --rate-quote go together")
    if a.rate_base:
        for arg, name in [(a.rate_base, "rate_base"), (a.rate_quote, "rate_quote")]:
            px = attach(px, _read_daily(arg), pd.Timedelta(days=1), name)

    added = [c for c in px.columns if c.startswith(("sentiment_", "rate_"))]
    before = len(px)
    px = px.dropna(subset=added)
    out = a.out or a.prices.replace(".csv", "_news.csv")
    px.index.name = "timestamp"
    px.to_csv(out)
    print(f"Added {added}. Dropped {before - len(px):,} early bars without values. "
          f"Saved {len(px):,} bars to {out}")
    for c in added:
        print(f"  {c}: last value {px[c].iloc[-1]:.3f} (as of {px.index[-1]})")


if __name__ == "__main__":
    main()
