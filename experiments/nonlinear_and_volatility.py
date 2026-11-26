"""Two checks on the daily data, before giving the inputs to the RL agent again.

    python -m experiments.predictability ... --daily-csv data/eurusd_daily.csv   # builds the daily file
    python -m experiments.nonlinear_and_volatility --daily data/eurusd_daily.csv

1. Direction, nonlinear. The linear test found no signal, but a linear model misses relations like
   "news matters only when volatility is high". Two nonlinear models predict the next day's return:
     - gradient boosting (shallow trees)
     - a small neural network built like the agent: one encoder for price inputs, a separate one for
       news and rate inputs, joined before the output (the course's advice), averaged over 5 seeds
   Same yearly walk-forward as the linear test. A planted nonlinear signal checks that the test
   can find such a relation when one exists.

2. Volatility. Direction is nearly unpredictable, volatility much less so. Forecast the volatility
   of the next 5 days with:
     - naive: today's 20-day volatility
     - price only: a standard model from past volatility over 1, 5 and 20 days (HAR)
     - price + news and rates: the same plus EPU, SF Fed index and rate gap
     - gradient boosting with everything
   and test whether news and rates improve the forecast. Finally, a practical check: carry and trend
   positions scaled to a constant volatility target with each forecast.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from drl_fx.evaluate import metrics, positions_history, random_benchmark

PRICE = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "z_20"]
CONTEXT = ["rate_gap", "rate_gap_chg_60", "epu", "epu_chg_20", "sffed", "sffed_chg_20"]


# ---------------------------------------------------------------- helpers
def yearly_splits(f, first_year):
    for year in range(first_year, f.index.max().year + 1):
        train = f[f.index < f"{year}-01-01"]
        test = f[(f.index >= f"{year}-01-01") & (f.index < f"{year + 1}-01-01")]
        if len(test) and len(train) >= 500:
            yield year, train, test


def standardise(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return ((train[cols] - mu) / sd).clip(-5, 5).values, ((test[cols] - mu) / sd).clip(-5, 5).values


def newey_west_t(d: np.ndarray, lags: int) -> float:
    """t-statistic of the mean of d, robust to overlap and autocorrelation."""
    n = len(d); m = d.mean(); e = d - m
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(m / np.sqrt(v / n))


# ---------------------------------------------------------------- 1. direction models
def fit_gbm(Xtr, ytr, Xte, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300, l2_regularization=1.0,
                                      min_samples_leaf=50, early_stopping=True, validation_fraction=0.15,
                                      random_state=seed)
    return m.fit(Xtr, ytr).predict(Xte)


def fit_two_encoder_net(Xtr, ytr, Xte, n_price, seeds=5, epochs=200):
    """Price inputs -> encoder A, news/rate inputs -> encoder B, joined -> prediction.
    Early stopping on the last 15% of the training years; averaged over several seeds."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)

    class Net(nn.Module):
        def __init__(self, n_p, n_c):
            super().__init__()
            self.a = nn.Sequential(nn.Linear(n_p, 16), nn.ReLU())
            self.b = nn.Sequential(nn.Linear(n_c, 16), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            return self.head(torch.cat([self.a(x[:, :n_price]), self.b(x[:, n_price:])], 1)).squeeze(1)

    cut = int(len(Xtr) * 0.85)
    ysd = ytr.std()
    xt, yt = torch.tensor(Xtr[:cut], dtype=torch.float32), torch.tensor(ytr[:cut] / ysd, dtype=torch.float32)
    xv, yv = torch.tensor(Xtr[cut:], dtype=torch.float32), torch.tensor(ytr[cut:] / ysd, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    preds = []
    for s in range(seeds):
        torch.manual_seed(s)
        net = Net(n_price, Xtr.shape[1] - n_price)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            net.train()
            for idx in torch.randperm(len(xt)).split(128):
                opt.zero_grad()
                loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = float(((net(xv) - yv) ** 2).mean())
            if v < best - 1e-5:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 15:
                    break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            preds.append(net(xe).numpy() * ysd)
    return np.mean(preds, axis=0)


def direction_test(f, target, first_year):
    cols = PRICE + CONTEXT
    out = []
    for year, train, test in yearly_splits(f, first_year):
        Xtr, Xte = standardise(train, test, cols)
        ytr = train[target].values
        out.append(pd.DataFrame({"gbm": fit_gbm(Xtr, ytr, Xte), "net": fit_two_encoder_net(Xtr, ytr, Xte, len(PRICE)),
                                 "linear": _ridge(Xtr, ytr, Xte)}, index=test.index))
    return pd.concat(out)


def _ridge(Xtr, ytr, Xte):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)


def score_direction(pred, y):
    return {"correlation": round(float(np.corrcoef(pred, y)[0, 1]), 3),
            "hit_rate_%": round(100 * float((np.sign(pred) == np.sign(y)).mean()), 1),
            "r2_oos_%": round(100 * (1 - float(((y - pred) ** 2).sum()) / float((y ** 2).sum())), 2)}


# ---------------------------------------------------------------- 2. volatility
def add_vol_columns(f):
    f = f.copy()
    r = f["ret_1"]
    f["lv_1"] = np.log(r.abs().clip(lower=1e-5))
    f["lv_5"] = np.log(np.sqrt((r ** 2).rolling(5).mean()).clip(lower=1e-5))
    f["lv_20"] = np.log(np.sqrt((r ** 2).rolling(20).mean()).clip(lower=1e-5))
    fut = (f["fwd_ret"] ** 2).rolling(5).mean().shift(-4)           # days t+1 .. t+5
    f["target_lv5"] = np.log(np.sqrt(fut).clip(lower=1e-5))
    return f.dropna(subset=["lv_5", "lv_20", "target_lv5"])


def volatility_test(f, first_year):
    from sklearn.linear_model import Ridge
    har = ["lv_1", "lv_5", "lv_20"]
    plus = har + CONTEXT
    out = []
    for year, train, test in yearly_splits(f, first_year):
        train = train.iloc[:-5]                                   # targets of the last 5 training days reach into the test year
        ytr = train["target_lv5"].values
        row = pd.DataFrame(index=test.index)
        row["naive"] = test["lv_20"].values
        for name, cols in [("price_only", har), ("price_news_rates", plus)]:
            Xtr, Xte = standardise(train, test, cols)
            row[name] = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
        Xtr, Xte = standardise(train, test, plus)
        row["gbm_all"] = fit_gbm(Xtr, ytr, Xte)
        out.append(row)
    return pd.concat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", required=True, help="daily file from experiments.predictability --daily-csv")
    p.add_argument("--first-test-year", type=int, default=2016)
    p.add_argument("--target-vol", type=float, default=0.06, help="annual volatility target for sizing")
    p.add_argument("--out", default="outputs/nonlinear_volatility")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = pd.read_csv(a.daily, parse_dates=["date"]).set_index("date")
    ppy = int(round(len(f) / ((f.index.max() - f.index.min()).days / 365.25)))
    res = {}

    # ---- 1a. direction on the real data
    P = direction_test(f, "fwd_ret", a.first_test_year)
    oos = f.loc[P.index]; y = oos["fwd_ret"].values; n = len(y)
    band = round(1.96 / np.sqrt(n), 3)
    direction = {}
    for m in ["linear", "gbm", "net"]:
        s = score_direction(P[m].values, y)
        pos = pd.Series(np.sign(P[m].values), index=oos.index)
        h = positions_history(pos, oos)
        s["strategy_sharpe"] = metrics(h, ppy)["sharpe"]
        s["random_same_turnover_p95"] = random_benchmark(h, oos, ppy, n=100)["sharpe_p95"]
        direction[m] = s
    res["direction"] = {"days": n, "correlation_noise_band_95": band, "models": direction}
    print(f"\n1. Direction, {n:,} days out of sample. Correlations within ±{band} are noise.")
    print(pd.DataFrame(direction).T.to_string())

    # ---- 1b. can the test find a nonlinear signal? plant one: tomorrow's move partly follows the
    #          5-day trend, but only when uncertainty (EPU) is rising. Linear models cannot see this.
    g = f.copy()
    inter = np.sign(g["epu_chg_20"]) * np.sign(g["ret_5"])
    k = 0.12                                                        # strength of the planted relation
    g["fwd_planted"] = g["fwd_ret"] + k * g["fwd_ret"].std() * inter
    Pp = direction_test(g, "fwd_planted", a.first_test_year)
    yp = g.loc[Pp.index, "fwd_planted"].values
    planted = {m: score_direction(Pp[m].values, yp) for m in ["linear", "gbm", "net"]}
    true_corr = round(float(np.corrcoef(inter.loc[Pp.index], yp)[0, 1]), 3)
    res["planted_nonlinear_signal"] = {"true_correlation_of_planted_pattern": true_corr, "models": planted}
    print(f"\nPlanted nonlinear pattern (correlation {true_corr} with the planted target):")
    print(pd.DataFrame(planted).T.to_string())

    # ---- 2. volatility
    fv = add_vol_columns(f)
    V = volatility_test(fv, a.first_test_year)
    vo = fv.loc[V.index]; yv = vo["target_lv5"].values
    base_sse = float(((yv - V["naive"].values) ** 2).sum())
    vol = {}
    for m in V.columns:
        e = (yv - V[m].values) ** 2
        vol[m] = {"r2_vs_mean_%": round(100 * (1 - e.sum() / float(((yv - yv.mean()) ** 2).sum())), 1),
                  "improvement_vs_naive_%": round(100 * (1 - e.sum() / base_sse), 1),
                  "correlation": round(float(np.corrcoef(V[m].values, yv)[0, 1]), 3)}
    d = (yv - V["price_only"].values) ** 2 - (yv - V["price_news_rates"].values) ** 2
    t_news = round(newey_west_t(d, lags=10), 2)
    d2 = (yv - V["price_only"].values) ** 2 - (yv - V["gbm_all"].values) ** 2
    t_gbm = round(newey_west_t(d2, lags=10), 2)
    res["volatility"] = {"days": len(yv), "models": vol,
                         "news_and_rates_vs_price_only_t_stat": t_news,
                         "gbm_all_vs_price_only_t_stat": t_gbm,
                         "note": "t above about 2 means the extra inputs improve the forecast reliably"}
    print(f"\n2. Volatility of the next 5 days (log), {len(yv):,} days out of sample:")
    print(pd.DataFrame(vol).T.to_string())
    print(f"News and rates on top of price-only: t = {t_news}; gradient boosting with all: t = {t_gbm} "
          "(about 2 or more = reliable improvement)")

    # ---- 2b. practical check: carry and trend at constant risk, sized with each forecast
    tv_daily = a.target_vol / np.sqrt(ppy)
    sizing = {}
    for rule, side in [("carry", np.sign(vo["rate_gap"])), ("trend", np.sign(vo["ret_60"]))]:
        for name, lv in [("fixed size", None), ("naive vol", V["naive"]), ("price-only forecast", V["price_only"]),
                         ("price+news+rates forecast", V["price_news_rates"])]:
            size = 1.0 if lv is None else (tv_daily / np.exp(lv)).clip(upper=3.0)
            pos = (side * size).astype(float)
            m = metrics(positions_history(pos, vo), ppy)
            sizing[f"{rule}: {name}"] = {"sharpe": m["sharpe"], "return_%": m["total_return_%"],
                                         "max_drawdown_%": m["max_drawdown_%"], "cost_%": m["cost_drag_%"]}
    res["volatility_sized_rules"] = sizing
    print("\nCarry and trend, fixed size vs scaled to a constant volatility target:")
    print(pd.DataFrame(sizing).T.to_string())

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    pd.DataFrame(direction).T.to_csv(os.path.join(a.out, "direction.csv"))
    pd.DataFrame(vol).T.to_csv(os.path.join(a.out, "volatility.csv"))
    pd.DataFrame(sizing).T.to_csv(os.path.join(a.out, "volatility_sized_rules.csv"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    real = np.exp(pd.Series(yv, index=vo.index)) * np.sqrt(ppy) * 100
    ax.plot(real.rolling(5).mean(), color="#9aa5b1", lw=0.8, label="actual (next 5 days)")
    ax.plot(np.exp(V["price_only"]) * np.sqrt(ppy) * 100, color="#4c78a8", lw=1.1, label="forecast: price only")
    ax.plot(np.exp(V["price_news_rates"]) * np.sqrt(ppy) * 100, color="#f58518", lw=1.0, alpha=0.8,
            label="forecast: price + news + rates")
    ax.set_ylabel("annualised volatility, %"); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    ax.set_title("EUR/USD volatility: out-of-sample forecasts vs what followed")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "volatility_forecast.png"), dpi=130)


if __name__ == "__main__":
    main()
