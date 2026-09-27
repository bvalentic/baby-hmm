import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baby_algo import signal_trader, buy_and_hold_pure_strategy

# signal_trader trades tomorrow on today's already-known Signal (internally
# shifted by 1) rather than on the same day's Signal, which would require
# knowing that day's own close before its open. These tests exercise both
# the cash/share mechanics and that look-ahead fix directly.


def test_signal_trader_first_day_never_trades_on_its_own_signal():
    # a lone day with Signal=1 has no *prior* day's signal to trade on,
    # so it should sit in cash rather than buy against its own close
    df = pd.DataFrame({"Open": [100.0], "Close": [110.0], "Signal": [1]})
    history = signal_trader(df, initial_funds=1000, buy_percentage=0.25)
    assert history == [1000.0]


def test_signal_trader_buys_one_day_after_the_signal():
    # Signal=1 on day 0 should only produce a buy on day 1 (using day 1's
    # open), not on day 0 itself
    df = pd.DataFrame({
        "Open":  [100.0, 105.0],
        "Close": [105.0, 95.0],
        "Signal": [1, 0],
    })
    history = signal_trader(df, initial_funds=1000, buy_percentage=0.25, sell_percentage=0.5)
    assert len(history) == 2
    assert history[0] == 1000.0  # day 0: no prior signal yet, no trade
    # day 1: buys 25% of 1000 = 250 at open=105 -> 250/105 shares; cash left 750
    # valuation at close=95: 750 + (250/105)*95
    assert history[1] == pytest.approx(750 + (250 / 105) * 95)


def test_signal_trader_sell_day():
    df = pd.DataFrame({"Open": [100.0], "Close": [90.0], "Signal": [0]})
    history = signal_trader(df, initial_funds=1000, shares=10, sell_percentage=0.5)
    # a lone day with no prior signal falls through to the sell branch either
    # way (NaN != 1), so this exercises the sell-side mechanics regardless
    # sell 50% of 10 shares = 5 shares at open=100 -> +500 cash; cash=1500; shares left=5
    # valuation at close=90: 1500 + 5*90 = 1950
    assert history == [1950.0]


def test_buy_and_hold_tracks_close_price():
    df = pd.DataFrame({"Open": [100.0], "Close": [110.0]})
    history = buy_and_hold_pure_strategy(df, initial_funds=1000)
    # buys all shares at day-0 open=100 -> 10 shares; valuation at close=110 -> 1100
    assert history == [1100.0]


def test_buy_and_hold_never_trades_after_day_zero():
    df = pd.DataFrame({
        "Open":  [100.0, 999.0, 999.0],
        "Close": [100.0, 120.0, 80.0],
    })
    history = buy_and_hold_pure_strategy(df, initial_funds=1000)
    # 10 shares bought once at day 0; value should track Close 1:1 afterward
    assert history == [1000.0, 1200.0, 800.0]
