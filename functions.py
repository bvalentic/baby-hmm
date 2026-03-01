## Functions that are used by the model multiple times
import numpy as np

def create_HMM():
    print("Creating HMM...")

def print_means():
    print("Means and variances of each state:")
    
# Return the variables used in original two-state models: returns (close vs. next day close) and range (volatility: high minus low)
def get_two_state_data(data_frame):
    data_frame['Returns'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame['Range'] = (data_frame['High'] - data_frame['Low']) / data_frame['Close']
    data_frame.dropna(inplace=True)

    return data_frame['Returns'], data_frame['Range']

def get_modified_two_state_data(data_frame):
    data_frame['Delta'] = (data_frame['Open'] - data_frame['Close']) / data_frame['Close']
    data_frame['Range'] = (data_frame['High'] - data_frame['Low']) / data_frame['Close']
    
    return data_frame['Delta'], data_frame['Range']

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
    return data_frame[['Delta', 'Range']].values

def normalize_four_state_data(data_frame):
    data_frame['Open_Normal'] = np.log(data_frame['Open'] / data_frame['Open'].shift(1))
    data_frame['High_Normal'] = np.log(data_frame['High'] / data_frame['High'].shift(1))
    data_frame['Low_Normal'] = np.log(data_frame['Low'] / data_frame['Low'].shift(1))
    data_frame['Close_Normal'] = np.log(data_frame['Close'] / data_frame['Close'].shift(1))
    data_frame.dropna(inplace=True)

    return data_frame[['Open_Normal','High_Normal','Low_Normal','Close_Normal']].values

def normalize_six_state_data(data_frame):
    pass
