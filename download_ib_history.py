"""Download FX history from Interactive Brokers (TWS or IB Gateway) into a CSV that drl_fx can read.

    python download_ib_history.py                          # EUR/USD, 15-min, 8 years -> data/eurusd_15m.csv
    python download_ib_history.py --symbol USDJPY --years 5
    python download_ib_history.py --no-spread              # skip bid/ask (twice as fast, cruder cost model)

IBKR gives FX history as MIDPOINT bars around the clock (no volume). Bid and ask are fetched too,
because the spread is most of the trading cost in FX.

Works backwards one week per request and saves after every chunk, so you can stop it with
Ctrl+C and run the same command again later: it continues where it stopped.
TWS must be running, logged in, with the API enabled.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from ib_async import IB, Forex, util

TZ = "US/Eastern"


def fetch_chunk(ib, contract, end, duration, bar, what):
    bars = ib.reqHistoricalData(contract, endDateTime=end, durationStr=duration,
                                barSizeSetting=bar, whatToShow=what, useRTH=False,
                                formatDate=2, timeout=120)
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    return df.set_index("date")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EURUSD", help="currency pair, six letters, e.g. EURUSD, USDJPY")
    p.add_argument("--bar", default="15 mins")
    p.add_argument("--years", type=float, default=8)
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--out", default=None)
    p.add_argument("--no-spread", action="store_true", help="do not fetch bid/ask")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between requests")
    a = p.parse_args()
    a.symbol = a.symbol.upper().replace("/", "")
    if not (len(a.symbol) == 6 and a.symbol.isalpha()):
        p.error("--symbol must be a currency pair like EURUSD")

    out = a.out or os.path.join("data", f"{a.symbol.lower()}_{a.bar.replace(' mins', 'm').replace(' ', '')}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ib = IB()
    ib.connect("127.0.0.1", a.port, clientId=7, timeout=15)
    ib.reqMarketDataType(3)                       # delayed data is fine for history
    contract = Forex(a.symbol)
    chunk_len = "1 W"                             # about 480 bars of 15 minutes per request
    step = pd.Timedelta(weeks=1)
    ib.qualifyContracts(contract)

    head = ib.reqHeadTimeStamp(contract, whatToShow="MIDPOINT", useRTH=False, formatDate=2)
    if head:
        head = pd.Timestamp(head)
        head = (head.tz_convert(TZ) if head.tzinfo else head.tz_localize("UTC").tz_convert(TZ)).tz_localize(None)
    else:
        head = None
    stop_at = pd.Timestamp.now() - pd.Timedelta(days=365 * a.years)
    if head is not None and head > stop_at:
        stop_at = head
    print(f"{a.symbol}: downloading back to {stop_at:%Y-%m-%d} (earliest available: {head})")

    # resume support
    if os.path.exists(out):
        have = pd.read_csv(out, parse_dates=["timestamp"]).set_index("timestamp")
        end = have.index.min()
        print(f"Resuming: {len(have):,} bars already in {out}, earliest {end}")
    else:
        have, end = pd.DataFrame(), None          # None = "up to now"

    empty_in_a_row = 0
    try:
        while end is None or end > stop_at:
            end_str = "" if end is None else end.tz_localize(TZ).to_pydatetime()
            mid = fetch_chunk(ib, contract, end_str, chunk_len, a.bar, "MIDPOINT")
            if mid.empty:
                empty_in_a_row += 1
                print(f"  no data for the period ending {end or 'now'}")
                if empty_in_a_row >= 3:
                    print("  several empty periods in a row, stopping")
                    break
                end = (end or pd.Timestamp.now()) - step
                time.sleep(a.pause)
                continue
            empty_in_a_row = 0
            chunk = mid[["open", "high", "low", "close"]].copy()
            chunk["volume"] = -1.0                    # midpoint bars carry no volume

            if not a.no_spread:
                time.sleep(a.pause)
                ba = fetch_chunk(ib, contract, end_str, chunk_len, a.bar, "BID_ASK")
                if not ba.empty:
                    # IBKR BID_ASK bars: open = average bid, close = average ask over the bar
                    chunk = chunk.join(ba[["open", "close"]].rename(columns={"open": "bid", "close": "ask"}))

            have = pd.concat([chunk, have]) if not have.empty else chunk
            have = have[~have.index.duplicated(keep="last")].sort_index()
            have.index.name = "timestamp"
            have.to_csv(out)
            new_end = chunk.index.min()
            # guard: if IBKR returned the same week again, step back manually
            end = new_end if end is None or new_end < end else end - step
            print(f"  {chunk.index.min():%Y-%m-%d} .. {chunk.index.max():%Y-%m-%d}  "
                  f"(+{len(chunk)} bars, total {len(have):,})")
            time.sleep(a.pause)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
    finally:
        ib.disconnect()

    print(f"Saved {len(have):,} bars to {out}")
    if len(have) < 100_000:
        print("Note: under 100k rows. Fine for a first test, but download more for real training.")


if __name__ == "__main__":
    main()
