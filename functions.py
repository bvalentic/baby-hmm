## Functions that are used by the model multiple times
import numpy as np

def create_HMM():
    print("Creating HMM...")

def print_means():
    print("Means and variances of each state:")
    
# this returns the "X" value used to fit the model and predict states
def normalize_two_state_data(data_frame):
    # we need features that define the "state" of the market
    # common choices are returns and volatility
    data_frame['Returns'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame['Range'] = (data_frame['High'] - data_frame['Low']) / data_frame['Close']
    data_frame.dropna(inplace=True)

    return data_frame[['Returns', 'Range']].values

def get_two_state_data(data_frame):
    data_frame['Returns'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame['Range'] = (data_frame['High'] - data_frame['Low']) / data_frame['Close']
    data_frame.dropna(inplace=True)
    return data_frame['Returns'], data_frame['Range']

def normalize_four_state_data(data_frame):
    pass

def normalize_six_state_data(data_frame):
    pass
