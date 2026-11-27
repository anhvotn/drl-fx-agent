"""Fast checks for the parts of the pipeline where silent bugs would fake a good backtest.

Run:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from drl_fx import data as D
from drl_fx.env import BUY, HOLD, SELL, TradingEnv
from drl_fx.evaluate import backtest, ensemble_weights
from drl_fx.synthetic import make_market


@pytest.fixture(scope="module")
def market():
    raw = make_market(n=4000, seed=1, missing_frac=0.0).set_index("timestamp")
    df, _ = D.clean(raw)
    cost = D.estimate_costs(df, 1e5)
    f, price_cols, sent_cols = D.make_features(df, cost)
    return df, f, price_cols + sent_cols


# ---------------------------------------------------------------- data
def test_features_have_no_look_ahead(market):
    """Features at time t must not change when data after t is removed."""
    df, f_full, cols = market
    cut = 3000
    df_cut = df.iloc[:cut]
    f_cut, *_ = D.make_features(df_cut, D.estimate_costs(df_cut, 1e5))
    common = f_cut.index.intersection(f_full.index)
    pd.testing.assert_frame_equal(f_cut.loc[common, cols], f_full.loc[common, cols])


def test_clean_fills_only_from_the_past():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({"close": [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10.0], "volume": 1.0}, index=idx)
    out, _ = D.clean(df)
    assert out.loc[idx[3], "close"] == 3.0          # previous value, not the interpolated 4.0


def test_clean_drops_periods_with_too_much_missing_data():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    close = np.arange(48, dtype=float) + 1
    close[24:40] = np.nan                            # day 2 is 2/3 missing
    out, rep = D.clean(pd.DataFrame({"close": close, "volume": 1.0}, index=idx))
    assert rep["periods_dropped"] == 1
    assert (out.index.date == idx[0].date()).all()


def test_splits_are_chronological_and_disjoint(market):
    _, f, _ = market
    sp = D.chronological_splits(f)
    parts = [sp[k] for k in ["train", "dev", "test", "holdout"]]
    assert sum(len(p) for p in parts) == len(f)
    for a, b in zip(parts, parts[1:]):
        assert a.index.max() < b.index.min()


def test_normalization_uses_train_statistics_only(market):
    _, f, cols = market
    sp = D.chronological_splits(f)
    tr, te = D.normalize(sp["train"], [sp["test"]], cols)
    assert np.allclose(tr[cols].mean().abs().max(), 0, atol=0.05)
    raw_te = sp["test"][cols]
    expected = ((raw_te - sp["train"][cols].mean()) / sp["train"][cols].std()).clip(-5, 5)
    pd.testing.assert_frame_equal(te[cols], expected)


# ---------------------------------------------------------------- environment
def test_costs_charged_only_when_position_changes(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=10)
    env.reset()
    t0 = env.t
    _, _, _, _, info = env.step(BUY)                 # flat -> long: pays 1x cost
    assert info["cost"] == pytest.approx(f["cost"].iloc[t0])
    _, _, _, _, info = env.step(HOLD)                # no trade: no cost
    assert info["cost"] == 0.0
    _, _, _, _, info = env.step(SELL)                # long -> short: pays 2x cost
    assert info["cost"] == pytest.approx(2 * f["cost"].iloc[t0 + 2])


def test_long_only_mode_never_goes_short(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=50, allow_short=False)
    env.reset()
    for a in [SELL, BUY, SELL, SELL]:
        env.step(a)
    assert env.history()["pos"].min() >= 0


def test_reward_uses_next_bar_return(market):
    """Position chosen at t earns the return from t to t+1, never the return already observed."""
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=5, cost_mult=0.0)
    env.reset()
    t0 = env.t
    _, _, _, _, info = env.step(BUY)
    assert info["net"] == pytest.approx(f["fwd_ret"].iloc[t0])
    assert f["fwd_ret"].iloc[t0] == pytest.approx(f["ret_1"].iloc[t0 + 1])


# ---------------------------------------------------------------- evaluation
def test_fast_backtest_matches_step_by_step(market):
    from drl_fx.agent import build_agent
    _, f, cols = market
    f = D.normalize(f, [], cols)[0]
    groups = [list(range(8)), list(range(8, len(cols)))]
    model = build_agent(lambda s: TradingEnv(f, cols, window=20, seed=s), 20, groups, n_envs=1)
    model.learn(512)

    fast = backtest(model, f, cols, window=20)

    env = TradingEnv(f, cols, window=20, episode_len=None, random_start=False)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(a))
        done = terminated or truncated
    pd.testing.assert_frame_equal(fast, env.history())


def test_ensemble_weights():
    w = ensemble_weights({"a": 2.0, "b": 1.0, "c": -0.5})
    assert w["c"] == 0
    assert w["a"] > w["b"] > 0
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v == 0 for v in ensemble_weights({"a": -1, "b": 0}).values())


# ---------------------------------------------------------------- reward vs buy-and-hold
def test_excess_reward_makes_staying_long_worth_zero(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=30, cost_mult=0.0, benchmark=1.0)
    env.reset()
    env.step(BUY)
    rewards = [env.step(HOLD)[1] for _ in range(20)]
    assert all(r == pytest.approx(0.0, abs=1e-9) for r in rewards)


def test_excess_reward_pays_for_being_flat_in_a_fall(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=30, cost_mult=0.0, benchmark=1.0)
    env.reset()
    t0 = env.t
    _, reward, _, _, info = env.step(HOLD)           # flat while buy-and-hold is long
    assert info["net"] == 0.0                         # real PnL is still logged
    assert reward == pytest.approx(-100 * f["fwd_ret"].iloc[t0])


def test_buy_and_hold_copy_is_flagged(market):
    from drl_fx.evaluate import buy_and_hold, is_bh_copy, metrics
    _, f, _ = market
    assert is_bh_copy(metrics(buy_and_hold(f, 20)))


def test_multi_asset_env_switches_instruments(market):
    from drl_fx.env import MultiAssetEnv
    _, f, cols = market
    a = TradingEnv(f.iloc[:2000], cols, window=20, episode_len=50, seed=1)
    b = TradingEnv(f.iloc[2000:], cols, window=20, episode_len=50, seed=2)
    env = MultiAssetEnv([a, b], seed=0)
    used = set()
    for _ in range(20):
        env.reset()
        used.add(id(env.cur))
    assert used == {id(a), id(b)}


# ---------------------------------------------------------------- FX data
def test_fx_data_without_volume_still_gives_features():
    raw = make_market(n=3000, seed=2, missing_frac=0.0).set_index("timestamp")
    raw["volume"] = -1.0                              # IBKR midpoint bars have no volume
    df, _ = D.clean(raw)
    cost = D.estimate_costs(df, 1e5)
    f, *_ = D.make_features(df, cost)
    assert len(f) > 2900
    assert (f["vol_ratio"] == 0).all()
    assert f["cost"].notna().all() and (f["cost"] > 0).all()


def test_bars_per_year_is_measured_from_data(market):
    _, f, _ = market
    # synthetic market: 15-min bars around the clock -> about 35,000 per year
    assert 30_000 < D.bars_per_year(f) < 40_000


def test_always_short_is_flagged_too(market):
    from drl_fx.evaluate import buy_and_hold, is_bh_copy, metrics
    _, f, _ = market
    h = buy_and_hold(f, 20)
    h["pos"], h["net"] = -1.0, -h["ret"]
    assert is_bh_copy(metrics(h))


def test_always_flat_is_flagged(market):
    from drl_fx.evaluate import buy_and_hold, is_bh_copy, metrics
    _, f, _ = market
    h = buy_and_hold(f, 20)
    h["pos"], h["net"] = 0.0, 0.0
    assert is_bh_copy(metrics(h))


def test_near_copy_of_buy_and_hold_has_zero_excess_sharpe(market):
    from drl_fx.evaluate import buy_and_hold, metrics
    _, f, _ = market
    h = buy_and_hold(f, 20)
    h.iloc[0, h.columns.get_loc("pos")] = 0.0      # flat for one bar, then long forever
    h.iloc[0, h.columns.get_loc("net")] = 0.0
    assert metrics(h)["excess_sharpe_vs_bh"] == 0.0


def test_fee_is_added_to_trading_costs():
    from drl_fx.run import prepare
    raw = make_market(n=2000, seed=5, missing_frac=0.0).set_index("timestamp")
    f0, *_ = prepare(raw, 1e5, 0.0)
    f1, *_ = prepare(raw, 1e5, 0.2)
    assert (f1["cost"] - f0["cost"]).round(12).eq(0.2 / 1e4).all()


# ---------------------------------------------------------------- carry and FX benchmarks
@pytest.fixture(scope="module")
def fx_market():
    raw = make_market(n=3000, seed=8, missing_frac=0.0).set_index("timestamp")
    raw["rate_base"], raw["rate_quote"] = 2.0, 4.5          # EUR pays 2%, USD 4.5%: long EUR/USD pays carry
    df, _ = D.clean(raw)
    f, price_cols, ctx = D.make_features(df, D.estimate_costs(df, 1e5))
    return f, price_cols + ctx


def test_carry_follows_rate_gap_and_time_held(fx_market):
    f, cols = fx_market
    assert "macro_rate_diff" in cols and (f["macro_rate_diff"] == -2.5).all()
    expected = -2.5 / 100 * (15 / (60 * 24)) / 365            # 15-minute bars
    assert np.allclose(f["carry"], expected)


def test_weekend_carry_and_data_gaps():
    fri = pd.date_range("2024-01-05 09:45", "2024-01-05 17:00", freq="15min")      # ends Friday 17:00
    sun = pd.date_range("2024-01-07 17:00", periods=40, freq="15min")              # reopens Sunday 17:00
    later = pd.date_range("2024-03-04 10:00", periods=40, freq="15min")            # after an 8-week hole
    idx = fri.append(sun).append(later)
    rng = np.random.default_rng(0)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(idx))))
    df = pd.DataFrame({"close": close, "volume": -1.0, "rate_base": 3.0, "rate_quote": 1.0}, index=idx)
    f, _, _ = D.make_features(df, pd.Series(1e-5, index=idx))
    assert f.loc[fri[-1], "carry"] == pytest.approx(2.0 / 100 * 2 / 365)          # Friday -> Sunday: 2 days
    assert f.loc[sun[5], "carry"] == pytest.approx(2.0 / 100 * (15 / 1440) / 365)
    assert sun[-1] not in f.index                                                   # bar before the hole is dropped


def test_flat_benchmark_is_default_and_long_earns_return_plus_carry(fx_market):
    f, cols = fx_market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=10, cost_mult=0.0)
    env.reset()
    t0 = env.t
    env.step(BUY)
    _, reward, _, _, info = env.step(HOLD)
    t = t0 + 1
    assert info["carry"] == pytest.approx(f["carry"].iloc[t])
    assert reward == pytest.approx(100 * (f["fwd_ret"].iloc[t] + f["carry"].iloc[t]))
    _, reward, _, _, info = env.step(SELL)                          # short now earns the carry back
    assert info["carry"] == pytest.approx(-f["carry"].iloc[t + 1])


def test_buy_and_hold_includes_carry(fx_market):
    from drl_fx.evaluate import buy_and_hold
    f, _ = fx_market
    bh = buy_and_hold(f, 20)
    assert np.allclose(bh["net"], bh["ret"] + f["carry"].iloc[19:-1].values)


def test_trend_benchmark_uses_only_the_past(fx_market):
    from drl_fx.evaluate import add_benchmarks
    f, _ = fx_market
    g = add_benchmarks(f, ppy=4000)                                 # 3 months = 1000 bars
    t = 2000
    past = f["fwd_ret"].iloc[t - 1000:t].sum()
    assert g["bm_trend"].iloc[t] == np.sign(past)
    assert (g["bm_carry"] == -1).all()                              # USD pays more: hold USD = short EUR/USD


def test_random_benchmark_matches_turnover(fx_market):
    from drl_fx.evaluate import positions_history, random_benchmark
    f, _ = fx_market
    seg = f.iloc[:2000]
    pos = pd.Series(np.where((np.arange(2000) // 50) % 2 == 0, 1.0, -1.0), index=seg.index)
    r = random_benchmark(positions_history(pos, seg), seg, ppy=35000, n=10)
    assert set(r) == {"sharpe", "sharpe_p95", "runs"} and r["sharpe_p95"] >= r["sharpe"]


# ---------------------------------------------------------------- fixes from the code reviews
def test_episode_length_is_truncation_and_data_end_is_termination(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, episode_len=5, random_start=False)
    env.reset()
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(HOLD)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(HOLD)
    assert truncated and not terminated              # a training cut-off, not the end of the data

    env = TradingEnv(f, cols, window=20, episode_len=None, random_start=False)
    env.reset()
    env.t = len(f) - 2
    _, _, terminated, truncated, _ = env.step(HOLD)
    assert terminated and not truncated              # the data ran out


def test_reset_seed_fixes_episode_starts(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, episode_len=100)
    starts_a = [env.reset(seed=7)[0] is not None and env.t0] + [env.reset() and env.t0 for _ in range(3)]
    starts_b = [env.reset(seed=7)[0] is not None and env.t0] + [env.reset() and env.t0 for _ in range(3)]
    assert starts_a == starts_b


def test_multi_asset_reset_seed_is_reproducible(market):
    from drl_fx.env import MultiAssetEnv
    _, f, cols = market
    make = lambda: MultiAssetEnv([TradingEnv(f, cols, window=20, episode_len=100),
                                  TradingEnv(f.iloc[:3000], cols, window=20, episode_len=100)])
    a, b = make(), make()
    seq_a = [(a.reset(seed=3), a.envs.index(a.cur), a.cur.t0)[1:]]
    seq_b = [(b.reset(seed=3), b.envs.index(b.cur), b.cur.t0)[1:]]
    seq_a += [(a.reset(), a.envs.index(a.cur), a.cur.t0)[1:] for _ in range(5)]
    seq_b += [(b.reset(), b.envs.index(b.cur), b.cur.t0)[1:] for _ in range(5)]
    assert seq_a == seq_b


def test_encoder_groups_must_cover_each_column_once():
    import gymnasium as gym
    from drl_fx.agent import SplitGRUExtractor
    space = gym.spaces.Box(-10, 10, shape=(10 * 4 + 1,), dtype=np.float32)
    SplitGRUExtractor(space, window=10, groups=[[0, 1], [2, 3]])               # fine
    with pytest.raises(ValueError):
        SplitGRUExtractor(space, window=10, groups=[[0, 1], [1, 3]])           # overlap, column 2 missing
    with pytest.raises(ValueError):
        SplitGRUExtractor(space, window=12, groups=[[0, 1], [2, 3]])           # wrong window


def test_clean_limits_filling_for_prices_and_slow_series():
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    close = np.linspace(1.0, 2.0, len(idx))
    close[100:110] = np.nan                                        # 10 missing price bars: longer than max_gap
    rate = np.full(len(idx), 3.0)
    rate[200:200 + 24 * 3] = np.nan                                # rate missing 3 days: carried forward
    rate[300 + 24 * 3:] = np.nan                                   # rate stops updating for 10+ days
    rate[300:300 + 24 * 3] = 2.5
    df = pd.DataFrame({"close": close, "volume": 1.0, "rate_base": rate}, index=idx)
    with pytest.warns(UserWarning):
        out, rep = D.clean(df, max_missing_frac=1.0)
    assert idx[105] not in out.index                               # long price gap not filled
    assert out.loc[idx[230], "rate_base"] == 3.0                   # short rate gap filled from the past
    stale = idx[300 + 24 * 3 + 24 * 8]                             # 8 days after the rate stopped
    assert stale not in out.index                                  # not carried forward for ever
    assert rep["rows_dropped_missing"] > 0


def test_macro_rate_diff_is_not_listed_twice():
    raw = make_market(n=1000, seed=8, missing_frac=0.0).set_index("timestamp")
    raw["rate_base"], raw["rate_quote"], raw["macro_rate_diff"] = 2.0, 4.5, -2.5
    df, _ = D.clean(raw)
    f, _, sent = D.make_features(df, D.estimate_costs(df, 1e5))
    assert sent.count("macro_rate_diff") == 1


# ---------------------------------------------------------------- fixes from the second round of reviews
def _fx_features(financing_pct=0.0, markup_bp=0.0):
    raw = make_market(n=1500, seed=8, missing_frac=0.0).set_index("timestamp")
    raw["rate_base"], raw["rate_quote"] = 2.0, 4.5
    df, _ = D.clean(raw)
    return D.make_features(df, D.estimate_costs(df, 1e5, retail_markup_bps=markup_bp), financing_pct=financing_pct)


def test_retail_markup_adds_to_every_trade():
    f0, *_ = _fx_features()
    f1, *_ = _fx_features(markup_bp=1.0)
    assert np.allclose(f1["cost"] - f0["cost"], 1e-4)


def test_financing_markup_costs_long_and_short_alike():
    f, pc, sc = _fx_features(financing_pct=1.5)
    cols = pc + sc
    assert (f["financing"] > 0).all()
    infos = {}
    for side in (BUY, SELL):
        env = TradingEnv(f, cols, window=20, random_start=False)
        env.reset()
        env.step(side)
        infos[side] = env.step(HOLD)[4]
    k, fin = f["carry"].iloc[20], f["financing"].iloc[20]
    assert np.isclose(infos[BUY]["carry"], k - fin)        # long pays the gap and the markup
    assert np.isclose(infos[SELL]["carry"], -k - fin)      # short earns the gap but still pays the markup


def test_turnover_penalty_changes_reward_not_pnl(market):
    _, f, cols = market
    out = {}
    for pen in (0.0, 5e-4):
        env = TradingEnv(f, cols, window=20, random_start=False, turnover_penalty=pen)
        env.reset()
        _, reward, _, _, info = env.step(BUY)
        out[pen] = (reward, info["net"])
    assert out[0.0][1] == out[5e-4][1]                     # same real PnL
    assert np.isclose(out[0.0][0] - out[5e-4][0], 5e-4 * 100)   # reward lower by the penalty (x100 scale)


def test_sharpe_reward_survives_zero_variance(market):
    _, f, cols = market
    f = f.copy()
    f["fwd_ret"], f["carry"], f["cost"] = 0.0, 0.0, 0.0   # every step earns exactly nothing
    env = TradingEnv(f, cols, window=20, reward="sharpe", random_start=False)
    env.reset()
    rewards = [env.step(HOLD)[1] for _ in range(30)]
    assert all(np.isfinite(rewards))
