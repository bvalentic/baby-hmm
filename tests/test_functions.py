import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions import sharpe_ratio, compute_bull_regimes, flatten_yfinance_columns


def test_sharpe_ratio_known_value():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0])
    result = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    expected = (returns.mean() / returns.std()) * np.sqrt(252)
    assert np.isclose(result, expected)


def test_sharpe_ratio_zero_std_returns_nan():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_insufficient_data_returns_nan():
    returns = pd.Series([0.01])
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_drops_nans():
    returns = pd.Series([0.01, np.nan, 0.02, -0.01])
    result = sharpe_ratio(returns)
    assert not np.isnan(result)


def test_sharpe_ratio_applies_risk_free_rate():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0])
    zero_rf = sharpe_ratio(returns, risk_free_rate=0.0)
    positive_rf = sharpe_ratio(returns, risk_free_rate=0.05)
    # a higher risk-free hurdle should lower the Sharpe on the same returns
    assert positive_rf < zero_rf


class _DummyModel:
    def __init__(self, means):
        self.means_ = np.array(means)


def test_compute_bull_regimes_absolute_threshold():
    model = _DummyModel([[0.001, 0.05], [-0.002, 0.03]])  # state 0: +return, low vol
    assert compute_bull_regimes(model, volatility_threshold=0.07) == [0]


def test_compute_bull_regimes_excludes_high_volatility_positive_state():
    model = _DummyModel([[0.001, 0.10], [0.002, 0.02]])  # state 0 positive but too volatile
    assert compute_bull_regimes(model, volatility_threshold=0.07) == [1]


def test_compute_bull_regimes_empty_when_no_state_qualifies():
    model = _DummyModel([[-0.001, 0.02], [-0.002, 0.01]])  # neither state has positive return
    assert compute_bull_regimes(model, volatility_threshold=0.07) == []


def test_compute_bull_regimes_relative_picks_lower_range_state():
    model = _DummyModel([[0.001, 0.09], [0.002, 0.03]])  # both positive; relative picks lower Range
    assert compute_bull_regimes(model, volatility_threshold=0.07, relative=True) == [1]


def test_flatten_yfinance_columns_flattens_multiindex():
    df = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=pd.MultiIndex.from_tuples([("Close", "SPY"), ("Open", "SPY")]),
    )
    flattened = flatten_yfinance_columns(df)
    assert list(flattened.columns) == ["Close", "Open"]


def test_flatten_yfinance_columns_noop_on_flat_columns():
    df = pd.DataFrame({"Close": [1, 2], "Open": [3, 4]})
    flattened = flatten_yfinance_columns(df)
    assert list(flattened.columns) == ["Close", "Open"]
    