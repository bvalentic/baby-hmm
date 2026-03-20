import functions
import baby_algo as algo

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from datetime import datetime

# use SPY (S&P 500 ETF) for a good mix of regimes
data_set = "SPY"

# pick a good start date?
start_date = "2022-01-01"
end_date = "2025-01-01"

# interval of less than 1d if start - end < 60 days
interval = "1d"

data = yf.download(data_set, start=start_date, end=end_date, interval=interval)

# separate dataset into training and testing data
train_size = int(len(data) * 0.70)
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()
# get the initial start and end dates of testing
test_start = test_data['Close'].iloc[0]
test_end = test_data['Close'].iloc[-1]

# using returns and volatility:
train_data['Returns'], train_data['Range'] = functions.get_two_state_data(train_data)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = functions.get_two_state_values(train_data)

# number of market regimes
n_components = 2
# "full" allows features to correlate within a state
# "diag" allows features to be modeled w/o diagonal correlation
covariance_type = "full"
# number of model iterations
n_iter = 100
# add min_covar to prevent "non-positive definite" error
min_covar=1e-4
# use Viterbi algorithm
algorithm = "viterbi"
# "" keeps set variables
# "stmc" reinitializes parameters each time 
init_params = "stmc"

model_number = 0
max_model_count = 20

model_list = []
score_list = []
win_rate_list = []

while model_number < max_model_count:

    model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type=covariance_type,
        min_covar=min_covar,
        n_iter=n_iter, 
        algorithm=algorithm,
        init_params=init_params
    )
    model.fit(X)

    # model estimates which "hidden state" generated the data for each day
    hidden_states = model.predict(X)

    # add states back to the dataframe for analysis
    train_data['State'] = hidden_states

    # define colors for up to 6 states
    colors = ['green', 'red', 'blue', 'orange', "purple", "brown"]

    # list out names of parameters for easier printing
    two_state_shape = ['Returns', 'Volatility']
    four_state_shape = ['Open', 'High', 'Low', 'Close']
    six_state_shape = ['Returns', 'Range', 'Open', 'High', 'Low', 'Close']

    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]

    # try adding volatility check
    volatility_threshold = 0.070
    low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]

    bull_regimes = []
    for i in positive_return_regimes:
        if i in low_volatility_regimes:
            bull_regimes.append(i)

    train_bull_state_list = []
    # create a signal: 1 if in bullish state, 0 otherwise
    train_data['Signal'] = np.where(train_data['State'].isin(bull_regimes), 1, 0)

    # calculate returns on HMM
    train_data['Strategy_Returns'] = algo.buy_and_hold_strategy(train_data)
    train_data['Algorithm_Portfolio'] = algo.basic_algo(train_data)
    train_data['New_Strategy_Returns'] = algo.new_buy_and_hold(train_data)

    # calculate buy & hold returns
    train_data['Cumulative_Market'] = np.exp(train_data['Returns'].cumsum())
    train_data['Cumulative_Strategy'] = np.exp(train_data['Strategy_Returns'].cumsum())
    train_data['Cumulative_Algorithm'] = np.exp(train_data['Algorithm_Portfolio'].cumsum())

    market_final_train = train_data['Cumulative_Market'].iloc[-1]
    strategy_final_train = train_data['Cumulative_Strategy'].iloc[-1]

    # check if the model makes a good prediction:
    # get the transitional matrix for the final state
    train_features_final = train_data.iloc[-1:][['Returns', 'Range']].values
    train_prediction_final = model.predict(train_features_final)[0]
    train_transmat_final = model.transmat_[train_prediction_final]
    # get largest state, to see if first index of new model.predict matches 
    train_predicted_chance_final = train_transmat_final.max()
    train_predicted_state_final = np.where(train_transmat_final == train_predicted_chance_final)[0]

    # prepare the test features (must be the same columns as training)
    test_data['Returns'], test_data['Range'] = functions.get_two_state_data(test_data)

    X_test = functions.get_two_state_values(test_data)

    # predict uses the existing model parameters to predict the next state
    test_states = model.predict(X_test)
    test_data['State'] = test_states

    # now, check the training predicted state with the first testing state
    test_prediction_first = test_states[0]

    # add to dataframe and calculate returns
    test_data = test_data.copy() # Avoid SettingWithCopyWarning
    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
    low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
    bull_regimes = []
    for i in positive_return_regimes:
        if i in low_volatility_regimes:
            bull_regimes.append(i)

    test_data['Signal'] = np.where(test_data['State'].isin(bull_regimes), 1, 0)

    # calculate returns
    test_data['Strategy_Returns'] = algo.buy_and_hold_strategy(test_data)

    # calculate cumulative growth
    test_data['Cumulative_Market'] = np.exp(test_data['Returns'].cumsum())
    test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())

    market_final_test = test_data['Cumulative_Market'].iloc[-1]
    strategy_final_test = test_data['Cumulative_Strategy'].iloc[-1]

    # next phase - rolling window and walk-forward
    # roll up to present day; 
    # guess latest regime for most recent market close; 
    # compare with actual results for a final test.
    # then apply model to the next day?

    last_date = test_data.index[-1].strftime("%Y-%m-%d")
    most_recent_date = datetime.today().strftime("%Y-%m-%d")

    new_data = yf.download(data_set, start=last_date, end=most_recent_date)

    # we'll do a 1-year rolling window
    # 252 trading days in a year
    window_size = 252 

    # flatten MultiIndex columns if they exist
    # if isinstance(new_data.columns, pd.MultiIndex):
    #     new_data.columns = new_data.columns.get_level_values(0)

    # combine with a bit of old data so the first "new" prediction has a training window

    # use most recent trading year from old data
    # drop the 'Returns' and 'Range' columns from the tail of test_data 
    # so they don't create NaN columns in the new_data section during concat
    # then concatenate and remove duplicates (the overlapping last_date)
    buffer_data = test_data.tail(window_size)[['Open', 'High', 'Low', 'Close', 'Volume']]
    full_df = pd.concat([buffer_data, new_data])
    full_df = full_df[~full_df.index.duplicated(keep='last')]

    full_df['Returns'], full_df['Range'] = functions.get_two_state_data(full_df)

    # plot market in new data to get an idea of how it has behaved in recent past
    # and also to catch breath before the rolling window

    # now before rolling window, check the model's prediction
    # get the transitional matrix for the final state
    test_features_final = test_data.iloc[-1:][['Returns', 'Range']].values
    test_prediction_final = model.predict(test_features_final)[0]
    test_transmat_final = model.transmat_[test_prediction_final]
    # get largest state, to see if first index of new model.predict matches 
    test_predicted_chance_final = test_transmat_final.max()
    test_predicted_state_final = np.where(test_transmat_final == test_predicted_chance_final)[0][0]

    run_count = 0
    signals = []
    states = []
    exception_list = []

    # use prediction from test data
    current_predicted_high_chance = test_predicted_chance_final 
    current_predicted_index = test_predicted_state_final 
    model_score = 0
    correct_predictions = 0
    # add high streak?

    for i in range(window_size, len(full_df)):
        run_count += 1
        X_train = full_df.iloc[i-window_size:i][['Returns', 'Range']].values
        current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
        
        try:
            model.fit(X_train)
            current_state = model.predict(current_features)[0]

            positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
            low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
            bull_regimes = []
            for regime in positive_return_regimes:
                if regime in low_volatility_regimes:
                    bull_regimes.append(regime)

            signal = 1 if current_state in bull_regimes else 0
            signals.append(signal)
            states.append(current_state)

            # "score" model based on whether or not prediction is correct
            # using percentage like a 0-100 confidence scale
            if current_state == current_predicted_index:
                model_score += current_predicted_high_chance
                correct_predictions += 1
            else:
                model_score -= current_predicted_high_chance
            # set next values to "current"
            current_transmat = model.transmat_[current_state]
            current_predicted_high_chance = current_transmat.max()
            current_predicted_index = np.where(current_transmat == current_predicted_high_chance)[0][0]

        except Exception as e:
            # if model fails to converge, use signal from previous day
            print(f"Exception caught on window {i}! Exception: {e}")

            signals.append(signals[-1] if signals else 0)
            states.append(states[-1] if states else 0)
            exception_list.append(i)
            continue

    # print("\nRolling window complete.")
    # print(f"Executed {run_count}/{len(full_df)-window_size} runs.")
    # print(f"Exceptions: {exception_list}")

    # print(f"Model score: {model_score:.4f}")
    win_rate = correct_predictions / run_count
    # print(f"Win rate: {win_rate:.2%} ({correct_predictions}/{run_count})")

    model_list.append(model_number)
    score_list.append(model_score)
    win_rate_list.append(win_rate)

    # add model number and continue loop
    model_number += 1

high_index = np.argmax(score_list)
high_model = model_list[high_index]
high_score = score_list[high_index]
high_win_rate = win_rate_list[high_index]

# dot plot of win rate
plt.plot(model_list, win_rate_list, '.', color='green')
plt.title("Model Win Rates")
plt.show()

# dot plot of model scores
plt.plot(model_list, score_list, '.', color='red')
plt.title("Model Scores")
plt.show()

# histogram of model scores
plt.hist(score_list, bins=8, linewidth=0.5, edgecolor="white")
plt.title("Model Score Histogram")
plt.show()

# histogram of win rate
plt.hist(win_rate_list, bins=8, linewidth=0.5, edgecolor="white")
plt.title("Model Score Histogram")
plt.show()

print(f"Winning model: {high_model}")
print(f"High score: {high_score:.2f}")
print(f"High win rate: {high_win_rate:.2%}")
