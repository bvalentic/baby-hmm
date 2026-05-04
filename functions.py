## Functions that are used by the model multiple times
import numpy as np
import random

def create_HMM():
    print("Creating HMM...")

def print_means():
    print("Means and variances of each state:")
    
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
