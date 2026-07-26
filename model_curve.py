import functions
import baby_algo as algo

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from datetime import datetime

import time

# use SPY (S&P 500 ETF) for a good mix of regimes
data_set = "SPY"

# pick a good start date?
start_date = "2020-01-01"
end_date = "2025-01-01"

# interval of less than 1d if start - end < 60 days
interval = "1d"

data = yf.download(data_set, start=start_date, end=end_date, interval=interval)

# separate dataset into training and testing data
if(data is not None):
    train_size = int(len(data) * 0.70)
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()

# more data
last_date = test_data.index[-1].strftime("%Y-%m-%d")
most_recent_date = datetime.today().strftime("%Y-%m-%d")
new_data = yf.download(data_set, start=last_date)

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
max_model_count = 32
model_list = []
score_list = []
win_rate_list = []
signals_and_states = []

# before loop, start time
start = time.time()

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

    # add to dataframe and calculate returns
    test_data = test_data.copy() # Avoid SettingWithCopyWarning
    # reset bull market signal using new data
    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
    low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
    bull_regimes = []
    for i in positive_return_regimes:
        if i in low_volatility_regimes:
            bull_regimes.append(i)

    test_data['Signal'] = np.where(test_data['State'].isin(bull_regimes), 1, 0)

    # next phase - rolling window and walk-forward
    # roll up to present day; 
    # guess latest regime for most recent market close; 
    # compare with actual results for a final test.
    # then apply model to the next day?

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

    win_rate = correct_predictions / run_count
    model_list.append(model)
    score_list.append(model_score)
    win_rate_list.append(win_rate)
    signals_and_states.append((signals, states))

    # add model number and continue loop
    model_number += 1

end = time.time()

# determine winningest model and use that one
high_index = np.argmax(win_rate_list)
high_model = model_list[high_index]
model = high_model
signals = signals_and_states[high_index][0]
states = signals_and_states[high_index][1]

print(f"\nRun time: {end - start:.2f}s")
print(f"Model count: {len(model_list)}")
print(f"Winning model: {high_index}")
print(f"High score: {score_list[high_index]:.2f}")
print(f"High win rate: {win_rate_list[high_index]:.2%}")

# set new bullish states in case they've changed
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

# Add the signals and states to dataframe
full_results = full_df.copy()
new_results = full_results[window_size:]
new_results['Signal'] = signals
new_results['State'] = states

# subplots:
fig, axs = plt.subplots(2, 2)
fig.set_figwidth(10)
fig.set_figheight(7)
# plot 1: histogram of model scores
axs[0, 0].hist(score_list)
axs[0, 0].set_title('Model Score Histogram')
# plot 2: dot plot of model scores
axs[0, 1].plot(score_list, '.', color='red')
axs[0, 1].set_title('Model Scores')
# plot 3: histogram of win rate
axs[1, 0].hist(win_rate_list)
axs[1, 0].set_title('Win Rate Histogram')
# plot 4: dot plot of win rate
axs[1, 1].plot(range(0, len(win_rate_list)), win_rate_list, '.', color='green')
axs[1, 1].set_title('Model Win Rates')
plt.show()

# get the state for most recent time interval
# use iloc[-1:] to get the latest data point
most_recent_features = new_results.iloc[-1:][['Returns', 'Range']].values
most_recent_state = model.predict(most_recent_features)[0]
# most_recent_state = model.predict(most_recent_features)[-1]

# access the transition matrix
# a matrix of [Current State, Next State] probabilities
# shape is (n_components, n_components)
transition_matrix = model.transmat_

# find the most likely next state
probs_for_next_state = transition_matrix[most_recent_state]
next_predicted_state = np.argmax(probs_for_next_state)

# check if future state is bullish
is_bullish = 1 if next_predicted_state in bull_regimes else 0

# leaving this for now so that I have some idea of what's going on
print("\nMeans and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}{" (Bullish)" if i in bull_regimes else ""}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Volatility: {model.means_[i][1]:.5f}")

# print table of recent states and prediction
functions.print_most_recent_dates_and_states_table(10, data_set, new_results, bull_regimes)

print(f"Most recent date used: {new_results.index[-1].strftime("%Y-%m-%d")}")
print(f"Model prediction of most recent state: {most_recent_state}")
print("Probabilities for tomorrow:")

for i in range(0, probs_for_next_state.size):
    print(f"  State {i}{" (Bullish)" if i in bull_regimes else ""}: {probs_for_next_state[i]:.2%}")

print(f"Predicted state for {data_set} tomorrow: {next_predicted_state}")
print(f"Action for {data_set} Tomorrow: {'🚀 BUY BUY BUY' if is_bullish else '💰 SELL SELL SELL'}")

# plot transmat of winning_model after seeing table
plt.imshow(high_model.transmat_, aspect='auto', cmap='YlOrRd')
plt.title('Generated Transition Matrix')
plt.xticks([0, 1])
plt.xlabel('State To')
plt.yticks([0, 1])
plt.ylabel('State From')
plt.show()
