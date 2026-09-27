"""
Baseline strategies used to sanity-check the HMM regime model against
model-free (or nearly model-free) alternatives on the same out-of-sample
data. See hmm.py's TODO / Work section and modeling-principles for why
these exist: base-rate regime persistence and pure luck can both look
like "signal" if you don't benchmark against them.

Signal-lag convention (read this before wiring these in):
- persistence_signal() is already lagged internally (it only ever looks
  at yesterday-or-earlier HMM labels) -- multiply its output directly by
  the same-day Returns series. Do NOT shift() it again.
- ma_trend_signal() is contemporaneous, same as the HMM's own `Signal`
  column (today's signal uses today's close) -- shift(1) it yourself
  before multiplying by Returns, exactly like you do for the HMM signal.
"""
import numpy as np
import pandas as pd

from functions import sharpe_ratio


def persistence_signal(reference_series: pd.Series) -> pd.Series:
    """
    Control-group baseline: always bullish / always fully invested, no
    regime detection involved at all. Tests whether the HMM's selective
    positioning (only being in the market on days it calls bullish) beats
    simply staying invested the whole time.

    `reference_series` is only used for its index (pass e.g. the Returns
    series) -- its values are ignored. A constant signal needs no shift, so
    multiply the result directly by the same-day Returns series. This
    baseline is mathematically identical to buy-and-hold.
    """
    return pd.Series(1, index=reference_series.index)


def random_walk_baseline_sharpe(returns_series: pd.Series, n_sims: int = 500,
                                 p_long: float = 0.5, percentile: float = 95):
    """
    Score `n_sims` random long/flat strategies on the same Returns series to
    build a null-hypothesis Sharpe distribution, rather than comparing
    against one lucky/unlucky random draw.

    Returns (mean_sharpe, percentile_sharpe). Compare the HMM's Sharpe
    against percentile_sharpe (default: 95th) -- beating the mean isn't a
    meaningful bar, since roughly half of random draws clear it by chance.
    """
    sharpes = []
    for _ in range(n_sims):
        signal = pd.Series(
            np.random.binomial(1, p_long, len(returns_series)),
            index=returns_series.index,
        )
        strat_returns = signal.shift(1) * returns_series
        sharpes.append(sharpe_ratio(strat_returns))
    sharpes = np.array(sharpes, dtype=float)
    return float(np.nanmean(sharpes)), float(np.nanpercentile(sharpes, percentile))


def ma_trend_signal(close_series: pd.Series, window: int = 50) -> pd.Series:
    """
    Classic trend filter, no HMM involved: long when price is above its own
    rolling average, flat otherwise. Contemporaneous, same as the HMM's own
    Signal column -- shift(1) this before multiplying by Returns.
    """
    ma = close_series.rolling(window).mean()
    return (close_series > ma).astype(int)
