"""How strong must a nonlinear pattern be before the models in
nonlinear_and_volatility.py can find it?

Plants sign(epu_chg_20) * sign(ret_5) into next-day EUR/USD returns at
three strengths and checks what gradient boosting recovers out of sample,
once with all 12 inputs and once with only the two inputs that matter.

    python -m experiments.power_check --daily data/eurusd_daily.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.nonlinear_and_volatility import CONTEXT, PRICE, standardise, yearly_splits

warnings.filterwarnings("ignore")

GBM = dict(max_depth=2, learning_rate=0.02, max_iter=300, min_samples_leaf=200,
           early_stopping=False, l2_regularization=1.0, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="data/eurusd_daily.csv")
    ap.add_argument("--out", default="outputs/nonlinear_volatility/power_check.csv")
    a = ap.parse_args()
    f = pd.read_csv(a.daily, parse_dates=["date"]).set_index("date")
    inter = np.sign(f["epu_chg_20"]) * np.sign(f["ret_5"])
    rows = []
    for k in [0.12, 0.2, 0.3]:
        f["planted"] = f["fwd_ret"] + k * f["fwd_ret"].std() * inter
        strength = np.corrcoef(inter, f["planted"])[0, 1]
        for label, cols in [("all 12 inputs", PRICE + CONTEXT), ("2 relevant inputs", ["epu_chg_20", "ret_5"])]:
            preds, ys = [], []
            for _, tr, te in yearly_splits(f, 2016):
                Xtr, Xte = standardise(tr, te, cols)
                m = HistGradientBoostingRegressor(**GBM).fit(Xtr, tr["planted"].values)
                preds.append(m.predict(Xte))
                ys.append(te["planted"].values)
            p, y = np.concatenate(preds), np.concatenate(ys)
            found = np.corrcoef(p, y)[0, 1]
            rows.append(dict(planted_corr=round(strength, 3), inputs=label, found_corr=round(found, 3)))
            print(f"planted {strength:.2f} | {label:18s} | found {found:+.3f}")
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
