import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions import sharpe_ratio


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
