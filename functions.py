## Functions that are used by the model multiple times
import numpy as np
import pandas as pd
import random

def create_HMM():
    print("Creating HMM...")

def print_means():
    print("Means and variances of each state:")

def flatten_yfinance_columns(data_frame):
    """
    Some yfinance versions return MultiIndex columns (e.g. ('Close','SPY'))
    even for a single-ticker download. Flatten to plain column names so the
    rest of the pipeline can rely on data_frame['Close'] etc. No-op if the
    columns are already flat.
    """
    if isinstance(data_frame.columns, pd.MultiIndex):
        data_frame.columns = data_frame.columns.get_level_values(0)
    return data_frame

def compute_bull_regimes(model, volatility_threshold, relative=False):
    """
    Components with positive mean Returns (column 0) and "low" mean Range
    (column 1, the volatility proxy). This is the single source of truth
    for the bull-regime logic that used to be copy-pasted at every refit
    site (initial training, backtest, each rolling-window iteration, and
    the post-loop recheck) -- one definition means they can't quietly
    drift out of sync with each other.

    By default (relative=False) "low" means below the fixed
    `volatility_threshold`, matching current behavior exactly.

    Pass relative=True to instead use "below this fit's own median Range
    mean" -- since each independent re-fit (e.g. in the rolling window)
    can land on a different absolute Range scale, a fixed cutoff doesn't
    always travel well across fits. NOTE: this is a real behavior change,
    not just a robustness tweak -- with a 2-state model it always marks
    exactly one state as "low volatility" (whichever is lower), even if
    both states are genuinely calm, or neither is. Try it deliberately as
    its own experiment against the walk-forward Sharpe baseline; don't
    flip the default silently.
    """
    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
    cutoff = np.median(model.means_[:, 1]) if relative else volatility_threshold
    low_volatility_regimes = np.where(model.means_[:, 1] < cutoff)[0]
    return [i for i in positive_return_regimes if i in low_volatility_regimes]

    
# Return the variables used in original two-state models: returns (close vs. next day close) and range (volatility: high minus low)
# (I think it's actually setting the 'Returns' and 'Range' in the data_frame and the return is NaN)
def get_two_state_data(data_frame):
    data_frame['Returns'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame['Range'] = (data_frame['High'] - data_frame['Low']) / data_frame['Close']
    data_frame.dropna(inplace=True)

    return data_frame['Returns'], data_frame['Range']

# Return the price "delta" between market open and close.
def get_delta_data(data_frame):
    data_frame['Delta'] = (data_frame['Open'] - data_frame['Close']) / data_frame['Close']
    
    return data_frame['Delta']

def get_four_state_data(data_frame):
    data_frame['Open_Normal'] = np.log(data_frame['Open'] / data_frame['Open'].shift(1))
    data_frame['High_Normal'] = np.log(data_frame['High'] / data_frame['High'].shift(1))
    data_frame['Low_Normal'] = np.log(data_frame['Low'] / data_frame['Low'].shift(1))
    data_frame['Close_Normal'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame.dropna(inplace=True)

    return data_frame['Open_Normal'], data_frame['High_Normal'], data_frame['Low_Normal'], data_frame['Close_Normal']

# Returns the "X" value used to fit the model and predict states
def get_two_state_values(data_frame):
    return data_frame[['Returns', 'Range']].values

def get_modified_two_state_values(data_frame):
    return data_frame[['Delta', 'Returns']].values

def normalize_four_state_data(data_frame):
    data_frame['Open_Normal'] = np.log(data_frame['Open'] / data_frame['Open'].shift(1))
    data_frame['High_Normal'] = np.log(data_frame['High'] / data_frame['High'].shift(1))
    data_frame['Low_Normal'] = np.log(data_frame['Low'] / data_frame['Low'].shift(1))
    data_frame['Close_Normal'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame.dropna(inplace=True)

    return data_frame[['Open_Normal','High_Normal','Low_Normal','Close_Normal']].values

def normalize_six_state_data(data_frame):
    pass

def print_most_recent_dates_and_states_table(end_date_range, data_set, data_frame, bull_regimes):
    # table of most recent dates and states
    print(f"\nMarket: {data_set}")
    print("|--- Date ---|-- State --|---Bull?---|")
    for i in range(0, end_date_range):
        # reverse index to go in order of dates, from -10 to -1
        index = end_date_range - i
        print_date = data_frame.index[-index].strftime("%Y-%m-%d")
        print_state = data_frame['State'].iloc[-index]
        print(f"| {print_date} |     {print_state}     |    {"Yes" if print_state in bull_regimes else "No "}    |") # formatting
    print("|------------|-----------|-----------|\n")

def guess_list(data_frame, n_components = 2):
    random.seed()
    length = len(data_frame)
    guesses = []
    for i in range(length):
        rng = random.randrange(0, n_components)
        guesses.append(rng)
    return guesses

def guess_mc(data_frame, mc_count, n_components = 2):
    random.seed()
    guess_lists = []
    winning_guess = 0
    winning_score = 0
    for sim in range(mc_count):
        guesses = guess_list(data_frame, n_components)
        guess_score = 0
        for item in range(len(data_frame)):
            if data_frame[item] == guesses[item]:
                guess_score += 1
            else:
                guess_score -= 1
        win_rate = guess_score / len(data_frame)
        if guess_score > winning_score:
            winning_guess = sim
            winning_score = guess_score
        guess_tuple = (sim, guesses, guess_score, win_rate)
        guess_lists.append(guess_tuple)
    return guess_lists[winning_guess]

def sharpe_ratio(returns_series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio from a series of log or simple daily returns.

    Parameters
    ----------
    returns_series   : daily strategy returns (log or simple, no NaNs)
    periods_per_year : trading days per year (252 for equities)
    risk_free_rate   : annualised risk-free rate (default 0.0)

    Returns
    -------
    Annualised Sharpe ratio, or np.nan if std == 0 / fewer than 2 observations.
    """
    clean = returns_series.dropna()
    if len(clean) < 2:
        return np.nan
    daily_rf = risk_free_rate / periods_per_year
    excess = clean - daily_rf
    if excess.std() == 0:
        return np.nan
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))

def print_sharpe_block(label: str, strategy_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> None:
    """Print a formatted Sharpe ratio comparison block."""
    s_sharpe = sharpe_ratio(strategy_returns, risk_free_rate, periods_per_year)
    m_sharpe = sharpe_ratio(market_returns, risk_free_rate, periods_per_year)
    print(f"\n── Sharpe Ratio ({label}) ──────────────────────")
    print(f"  HMM Strategy : {s_sharpe:+.4f}")
    print(f"  Buy & Hold   : {m_sharpe:+.4f}")
    if np.isnan(s_sharpe) or np.isnan(m_sharpe):
        print("  (insufficient data for comparison)")
    elif s_sharpe > m_sharpe:
        print(f"  ✅ HMM has a better Sharpe by {s_sharpe - m_sharpe:.4f}")
    else:
        print(f"  ❌ Buy & Hold has a better Sharpe by {m_sharpe - s_sharpe:.4f}")
    print("────────────────────────────────────────────────\n")
