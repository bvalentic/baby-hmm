import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baselines import persistence_signal, ma_trend_signal, random_walk_baseline_sharpe


def test_persistence_signal_is_always_bullish():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    signal = persistence_signal(returns)
    assert (signal == 1).all()
    assert len(signal) == len(returns)


def test_persistence_signal_reduces_to_buy_and_hold_returns():
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
    persistence_returns = persistence_signal(returns) * returns
    pd.testing.assert_series_equal(persistence_returns, returns, check_names=False)


def test_ma_trend_signal_uptrend_is_always_long_after_window_fills():
    close = pd.Series(range(1, 21), dtype=float)  # strictly increasing
    signal = ma_trend_signal(close, window=5)
    # once the rolling mean is defined, price > its own trailing mean in a strict uptrend
    assert (signal.iloc[4:] == 1).all()


def test_ma_trend_signal_downtrend_is_always_flat_after_window_fills():
    close = pd.Series(range(20, 0, -1), dtype=float)  # strictly decreasing
    signal = ma_trend_signal(close, window=5)
    assert (signal.iloc[4:] == 0).all()


def test_random_walk_baseline_sharpe_near_zero_on_zero_mean_returns():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    mean_sharpe, p95_sharpe = random_walk_baseline_sharpe(returns, n_sims=200)
    # random long/flat on pure noise shouldn't show a strong edge on average
    assert abs(mean_sharpe) < 1.0
    assert p95_sharpe > mean_sharpe
