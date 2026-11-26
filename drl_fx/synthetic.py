"""Regime-switching synthetic market with bid/ask, volume and a noisy sentiment signal.
Only for testing the pipeline end-to-end; replace with real data for anything meaningful."""
import numpy as np
import pandas as pd


def make_market(n: int = 120_000, seed: int = 0, freq: str = "15min", start: str = "2018-01-01",
                missing_frac: float = 0.002, edge: float = 0.0012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # regimes: (drift per bar, vol per bar)
    regimes = [(0.00004, 0.0015), (-0.00005, 0.0025), (0.0, 0.0010), (0.0, 0.0040)]
    P = np.full((4, 4), 0.0005); np.fill_diagonal(P, 1 - 0.0015)
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = rng.choice(4, p=P[s[t - 1]] / P[s[t - 1]].sum())
    mu = np.array([regimes[k][0] for k in s]); sig = np.array([regimes[k][1] for k in s])
    # weak, persistent signal: sentiment leads drift a little
    sent = np.convolve(rng.normal(0, 1, n), np.ones(50) / 50, "same") + 400 * mu
    eps = rng.normal(0, 1, n)
    r = mu + sig * eps + edge * np.roll(sent, 1) * sig / 0.002
    close = 100 * np.exp(np.cumsum(r))
    spread = close * (0.00005 + 0.03 * sig)
    vol = rng.lognormal(10, 0.5, n) * (1 + 200 * sig)
    idx = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({"timestamp": idx, "open": np.roll(close, 1), "high": close * (1 + sig),
                       "low": close * (1 - sig), "close": close, "volume": vol,
                       "bid": close - spread / 2, "ask": close + spread / 2,
                       "sentiment_news": sent + rng.normal(0, 0.05, n)})
    m = rng.random((n, 3)) < missing_frac
    for j, col in enumerate(["close", "volume", "sentiment_news"]):
        df.loc[m[:, j], col] = np.nan
    return df
