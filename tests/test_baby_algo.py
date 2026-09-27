import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baby_algo import signal_trader, buy_and_hold_pure_strategy

# These tests exercise signal_trader's cash/share mechanics on hand-picked
# numbers, using the Signal column exactly as the function currently reads
# it (same-day, contemporaneous). They don't assert anything about whether
# that date alignment is the *right* choice for a real backtest -- that's
# the separate look-ahead-bias question flagged in review, which is a
# design decision, not a mechanics bug. These tests are the safety net for
# whichever fix gets picked there.


def test_signal_trader_buy_day():
    df = pd.DataFrame({"Open": [100.0], "Close": [110.0], "Signal": [1]})
    history = signal_trader(df, initial_funds=1000, buy_percentage=0.25)
    # spend 25% of 1000 = 250 at open=100 -> 2.5 shares; cash left 750
    # valuation at close=110: 750 + 2.5*110 = 1025
    assert history == [1025.0]


def test_signal_trader_sell_day():
    df = pd.DataFrame({"Open": [100.0], "Close": [90.0], "Signal": [0]})
    history = signal_trader(df, initial_funds=1000, shares=10, sell_percentage=0.5)
    # sell 50% of 10 shares = 5 shares at open=100 -> +500 cash; cash=1500; shares left=5
    # valuation at close=90: 1500 + 5*90 = 1950
    assert history == [1950.0]


def test_signal_trader_multi_day_runs_sequentially():
    df = pd.DataFrame({
        "Open":  [100.0, 105.0],
        "Close": [105.0, 95.0],
        "Signal": [1, 0],
    })
    history = signal_trader(df, initial_funds=1000, buy_percentage=0.25, sell_percentage=0.5)
    assert len(history) == 2
    # day 1 (buy): as in test_signal_trader_buy_day but close=105
    #   cash=750, shares=2.5 -> value = 750 + 2.5*105 = 1012.5
    assert history[0] == 1012.5


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
