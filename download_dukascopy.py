"""Download free FX history from Dukascopy into a CSV that drl_fx can read.

    pip install dukascopy-python==7.0.0
    python download_dukascopy.py                          # EUR/USD, 15-min, from 2003 -> data/eurusd_duka_15m.csv
    python download_dukascopy.py --pair GBP-USD --start 2010-01-01
    python download_dukascopy.py --pair GBP-USD --start 2012-01-01 --hourly   # hourly bars, ~4x faster -> data/gbpusd_duka_1h.csv

Fetches bid and ask candles month by month and writes:
    timestamp (New York time), open/high/low/close (mid = average of bid and ask), volume, bid, ask
It saves after every month, so you can stop it with Ctrl+C and run the same command again to continue.
No IBKR or TWS needed: Dukascopy's historical feed is public.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

TZ = "US/Eastern"          # same clock as the IBKR files, so instruments line up


def fetch_side(pair, start, end, side, minutes):
    import dukascopy_python as dk
    if minutes == 60:
        candles = dk.fetch_candles(pair, 1, dk.TimeUnit.HOUR, side, start, end)
    else:
        candles = dk.fetch_candles(pair, minutes, dk.TimeUnit.MINUTE, side, start, end)
    df = dk.to_dataframe(candles)
    if df.empty:
        return df
    return df[["open", "high", "low", "close", "volume"]]


def fetch_month(pair, start, end, minutes):
    import dukascopy_python as dk
    bid = fetch_side(pair, start, end, dk.OfferSide.BID, minutes)
    ask = fetch_side(pair, start, end, dk.OfferSide.ASK, minutes)
    if bid.empty or ask.empty:
        return pd.DataFrame()
    j = bid.join(ask, lsuffix="_bid", rsuffix="_ask", how="inner")
    out = pd.DataFrame(index=j.index)
    for c in ["open", "high", "low", "close"]:
        out[c] = (j[f"{c}_bid"] + j[f"{c}_ask"]) / 2
    out["volume"] = j["volume_bid"] + j["volume_ask"]
    out["bid"], out["ask"] = j["close_bid"], j["close_ask"]
    out = out[out["volume"] > 0]                      # Dukascopy fills closed-market periods with flat, zero-volume candles
    out.index = out.index.tz_convert(TZ).tz_localize(None)
    out.index.name = "timestamp"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="EUR-USD", help="Dukascopy code, e.g. EUR-USD, USD-JPY, GBP-USD")
    p.add_argument("--minutes", type=int, default=15)
    p.add_argument("--hourly", action="store_true", help="hourly bars (same as --minutes 60); enough for daily tests")
    p.add_argument("--start", default="2003-05-05")
    p.add_argument("--end", default=None, help="default: today")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.hourly:
        a.minutes = 60

    tag = "1h" if a.minutes == 60 else f"{a.minutes}m"
    out = a.out or os.path.join("data", f"{a.pair.replace('-', '').lower()}_duka_{tag}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    have = pd.DataFrame()
    start = pd.Timestamp(a.start, tz="UTC")
    if os.path.exists(out):
        have = pd.read_csv(out, parse_dates=["timestamp"]).set_index("timestamp")
        if len(have):
            last = have.index.max().tz_localize(TZ).tz_convert("UTC")
            start = (last + pd.Timedelta(minutes=a.minutes)).normalize()
            print(f"Resuming: {len(have):,} bars already in {out}, last {have.index.max()}")
    end = pd.Timestamp(a.end, tz="UTC") if a.end else pd.Timestamp.now(tz="UTC").normalize()

    months = pd.date_range(start.normalize().replace(day=1), end, freq="MS", tz="UTC")
    bounds = [max(start, months[0])] + list(months[1:]) + [end]
    try:
        for s, e in zip(bounds[:-1], bounds[1:]):
            if s >= e:
                continue
            chunk = fetch_month(a.pair, s.to_pydatetime(), e.to_pydatetime(), a.minutes)
            if chunk.empty:
                print(f"  {s:%Y-%m}: no data")
                continue
            have = pd.concat([have, chunk]) if len(have) else chunk
            have = have[~have.index.duplicated(keep="last")].sort_index()
            have.to_csv(out)
            print(f"  {s:%Y-%m}: +{len(chunk):,} bars (total {len(have):,})", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")

    print(f"Saved {len(have):,} bars to {out}")


if __name__ == "__main__":
    main()
