"""Shared helpers for the experiments: load one instrument, train one agent, score a position series."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from drl_fx import data as D
from drl_fx.evaluate import add_benchmarks, positions_history
from drl_fx.run import prepare


def load(csv: str, fee_bp: float, notional: float = 100_000):
    """Clean data with costs and features, plus the observation columns and input groups."""
    raw = D.load_csv(csv)
    f, price_cols, sent_cols, _ = prepare(raw, notional, fee_bp)
    cols = D.PRICE_FEATURES + sent_cols
    groups = [list(range(len(D.PRICE_FEATURES)))]
    if sent_cols:
        groups.append(list(range(len(D.PRICE_FEATURES), len(cols))))
    ppy = D.bars_per_year(f)
    return add_benchmarks(f, ppy), cols, groups, ppy


def train_agent(train_df, cols, groups, window, seed, steps_pnl, steps_sharpe, dropout=0.1, benchmark=0.0):
    """One agent, trained the same way as drl_fx.run (PnL then Sharpe; benchmark 0 = flat, 1 = buy-and-hold)."""
    import torch
    torch.set_num_threads(1)                      # several agents train side by side
    from drl_fx.agent import build_agent, train_two_stage
    from drl_fx.env import TradingEnv

    def fn(reward):
        return lambda s: TradingEnv(train_df, cols, window=window, reward=reward, seed=s, benchmark=benchmark)
    model = build_agent(fn("pnl"), window, groups, dropout=dropout, seed=seed)
    return train_two_stage(model, fn("sharpe"), steps_pnl, steps_sharpe, seed=seed)


score_positions = positions_history          # kept for older scripts; same accounting as the environment


def save_plot(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
