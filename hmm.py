## An HMM-based algorithmic trading system in Python
import functions
import baby_algo as algo

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from datetime import datetime

# TODO list:
# use highest-scoring model as basis for another round of models
# attempt adding a third parameter of "volume" 

print("\n|Phase 1: Fetch data|")

# use SPY (S&P 500 ETF) for a good mix of regimes
data_set = "SPY"
# BTC-USD for regimes in crypto
# Silver? Oil? Anything?
print(f"Market: {data_set}")

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
test_start = test_data['Close'].index[0]
test_end = test_data['Close'].index[-1]

# using returns and volatility:
train_data['Returns'], train_data['Range'] = functions.get_two_state_data(train_data)

# need to normalize OHLC data before attempting to train on it
train_data['Open_Normal'], train_data['High_Normal'], train_data['Low_Normal'], train_data['Close_Normal'] = functions.get_four_state_data(train_data)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = functions.get_two_state_values(train_data)

print("\n|Phase 2: Build & train model|")

# number of market regimes
n_components = 2
# "spherical" - each state uses a single variance value that applies to all features (default)
# "diag" - each state uses a diagonal covariance matrix
# "full" - each state uses a full (i.e. unrestricted) covariance matrix
# (originally said it 'allows features to correlate within a state')
# "tied" - all states use the same full covariance matrix
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

print("Creating model...")

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

print("Initial model training complete.")

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


print("Means and variances of each state (Returns & Volatility):")
for i in range(model.n_components):
    print(f"State {i}{" (Bullish)" if i in bull_regimes else ""}:")
    for j in range(X.shape[1]):
        print(f"  Mean {two_state_shape[j]}: {model.means_[i][j]:.5f}")

print("\n|Phase 3: Plot and verify|")

# plot price, colored by state
plt.figure(figsize=(15, 6))

# plot line chart of market close (for now)
plt.plot(train_data['Close'], '-', label=f"{data_set}", color='grey', markersize=1)

# plot each state's closing price
for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(
        train_data.index[state], 
        train_data['Close'][state], 
        '.', 
        label=f'State {i}', 
        color=colors[i], 
        markersize=3
        )
plt.legend()
plt.title('Regimes Detected by HMM - Training')
plt.show()

print("\n|Phase 4: Test against training data|")
train_bull_state_list = []
# create a signal: 1 if in bullish state, 0 otherwise
print("Setting bull market signal...")
train_data['Signal'] = np.where(train_data['State'].isin(bull_regimes), 1, 0)

# calculate returns on HMM
train_data['Strategy_Returns'] = algo.buy_and_hold_strategy(train_data)
train_data['Algorithm_Portfolio'] = algo.basic_algo(train_data)
train_data['New_Strategy_Returns'] = algo.new_buy_and_hold(train_data)

# Plot using the index explicitly for X-axis stability
print("Plotting portfolio returns:")
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Algorithm_Portfolio'], 
         label='Total Portfolio Balance', color='green')
plt.plot(train_data.index, train_data['New_Strategy_Returns'], 
         label='Full Buy & Hold Returns', color='red')
plt.title(f'{data_set}: Portfolio on HMM - Training')
plt.legend()
plt.show()

# calculate buy & hold returns
train_data['Cumulative_Market'] = np.exp(train_data['Returns'].cumsum())
train_data['Cumulative_Strategy'] = np.exp(train_data['Strategy_Returns'].cumsum())
train_data['Cumulative_Algorithm'] = np.exp(train_data['Algorithm_Portfolio'].cumsum())

# plot training performance
print("Plotting training performance:")
plt.figure(figsize=(12, 6))
plt.plot(train_data['Cumulative_Market'], label=data_set, color='gray')
plt.plot(train_data['Cumulative_Strategy'], label='Buy & Hold w/HMM', color='orange')
plt.title(f'{data_set}: HMM Strategy vs Market - Training')
plt.legend()
plt.show()

print("\n|Phase 5: Analyze training results|")

print("Return on investment during training period:")
market_final_train = train_data['Cumulative_Market'].iloc[-1]
strategy_final_train = train_data['Cumulative_Strategy'].iloc[-1]

print(f"  Training Period Market Return: {(market_final_train - 1):.2%}")
print(f"  Training Period Strategy Return: {(strategy_final_train - 1):.2%}")

# TODO: calculate Sharpe ratio

if market_final_train > strategy_final_train:
    print("📈 Buy & Hold outperformed the HMM in training.")
else:
    print("🤖 The HMM strategy beat the market in training!")

# check if the model makes a good prediction:
# get the transitional matrix for the final state
train_features_final = train_data.iloc[-1:][['Returns', 'Range']].values
train_prediction_final = model.predict(train_features_final)[0]
train_transmat_final = model.transmat_[train_prediction_final]
print(f"Probabilities for next state: {train_transmat_final}")
# get largest state, to see if first index of new model.predict matches 
train_predicted_chance_final = train_transmat_final.max()
train_predicted_state_final = np.where(train_transmat_final == train_predicted_chance_final)[0]

print("\n|Phase 6: Initial test on new data|")

# prepare the test features (must be the same columns as training)
test_data['Returns'], test_data['Range'] = functions.get_two_state_data(test_data)

X_test = functions.get_two_state_values(test_data)

# predict uses the existing model parameters to predict the next state
test_states = model.predict(X_test)
test_data['State'] = test_states
print(f"Size of test data frame: {len(test_data['State'])}")

# now, check the training predicted state with the first testing state
test_prediction_first = test_states[0]
print(f"Model predicted {train_predicted_state_final} with a {train_predicted_chance_final:.2%} chance")
print(f"First test prediction: {test_prediction_first}")

# add to dataframe and calculate returns
test_data = test_data.copy() # Avoid SettingWithCopyWarning
print("Resetting bull market signal...")
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

print("Bull regime(s):")
for i in range(0, len(bull_regimes)):
    print(f"  {bull_regimes[i]}")

test_data['Signal'] = np.where(test_data['State'].isin(bull_regimes), 1, 0)

print("Plotting predicted regimes:")
plt.figure(figsize=(12, 6))
# plot market close, in grey, behind regime plot
plt.plot(test_data['Close'], '-', label=f"{data_set}", color='grey', markersize=1)
for i in range(model.n_components):
    state = (test_states == i)
    plt.plot(
        test_data.index[state], 
        test_data['Close'][state], 
        '.', 
        label=f'State {i}', 
        color=colors[i], 
        markersize=3)
plt.legend()
plt.title(f'{data_set}: Regimes Detected by HMM - Backtesting')
plt.show()

# calculate returns
test_data['Strategy_Returns'] = algo.buy_and_hold_strategy(test_data)

# calculate cumulative growth
test_data['Cumulative_Market'] = np.exp(test_data['Returns'].cumsum())
test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())

# plot test performance
print("Plotting initial test performance:")
plt.figure(figsize=(12, 6))
plt.plot(test_data['Cumulative_Market'], label='Buy & Hold', color='gray')
plt.plot(test_data['Cumulative_Strategy'], label='HMM Strategy', color='orange')
plt.title(f'{data_set}: HMM Strategy vs Buy & Hold - Backtesting')
plt.legend()
plt.show()

market_final_test = test_data['Cumulative_Market'].iloc[-1]
strategy_final_test = test_data['Cumulative_Strategy'].iloc[-1]

print("Return on investment during initial testing period:")
print(f"  Test Period Market Return: {(market_final_test - 1):.2%}")
print(f"  Test Period Strategy Return: {(strategy_final_test - 1):.2%}")

if strategy_final_test > market_final_test:
    print("\n✅ The HMM beat the market in backtesting!")
else:
    print("\n❌ The HMM underperformed. It might need different features or state counts.")

# next phase - rolling window and walk-forward
# roll up to present day; 
# guess latest regime for most recent market close; 
# compare with actual results for a final test.
# then apply model to the next day?

print("\n|Phase 7: Rolling window|")
last_date = test_data.index[-1].strftime("%Y-%m-%d")
most_recent_date = datetime.today().strftime("%Y-%m-%d")
print(f"Last date of test window: {last_date}")
print(f"Fetching data up to {most_recent_date}")

# don't include #end=most_recent_date; it excludes it from results
new_data = yf.download(data_set, start=last_date)

# we'll do a 1-year rolling window
# 252 trading days in a year
window_size = 252 

# flatten MultiIndex columns if they exist
# if isinstance(new_data.columns, pd.MultiIndex):
#     new_data.columns = new_data.columns.get_level_values(0)

# combine with a bit of old data so the first "new" prediction has a training window
print("Creating new data series for rolling window...")

# use most recent trading year from old data
# drop the 'Returns' and 'Range' columns from the tail of test_data 
# so they don't create NaN columns in the new_data section during concat
# then concatenate and remove duplicates (the overlapping last_date)
buffer_data = test_data.tail(window_size)[['Open', 'High', 'Low', 'Close', 'Volume']]
full_df = pd.concat([buffer_data, new_data])
full_df = full_df[~full_df.index.duplicated(keep='last')]
print(f"Size of full data frame: {len(full_df)}")

full_df['Returns'], full_df['Range'] = functions.get_two_state_data(full_df)

print(f"New data successfully merged. Total rows: {len(full_df)}")
print(f"End date: {full_df.index[-1].strftime("%Y-%m-%d")}")
print(f"Compare with end date on test_data: {test_end.strftime("%Y-%m-%d")}")

# plot market in new data to get an idea of how it has behaved in recent past
# and also to catch breath before the rolling window
print("Plotting market between training date and now:\n")
plt.figure(figsize=(12, 6))
plt.plot(full_df['Close'], label=data_set, color='black')
plt.title(f'{data_set} Market Outlook - Rolling Window')
plt.legend()
plt.show()

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

print(f"Executing {window_size}-day rolling window from {last_date} to {most_recent_date}:")

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

print("\nRolling window complete.")
print(f"Executed {run_count}/{len(full_df)-window_size} runs.")
print(f"Exceptions: {exception_list}\n")
print("Compiling data...")

print(f"Model score: {model_score:.4f}")
win_rate = correct_predictions / run_count
print(f"Win rate: {win_rate:.2%} ({correct_predictions}/{run_count})")

# set new bullish states in case they've changed
# TODO: remove these checks?
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

# add the signals to dataframe
full_results = full_df.copy()
new_results = full_results[window_size:]
new_results['Signal'] = signals
new_results['State'] = states

new_results['Strategy_Returns'] = algo.buy_and_hold_strategy(new_results)
new_results['Algorithm_Portfolio'] = algo.basic_algo(new_results)

# get "control group" of random guesses
new_results_guesses = functions.guess_list(signals, n_components)
new_results['Guesses'] = new_results_guesses

# compare with signal
guess_score = 0
for item in range(len(signals)):
    if signals[item] == new_results_guesses[item]:
        guess_score += 1
guess_win_rate = guess_score / len(signals)

print(f"Guessing win rate: {guess_win_rate}")

print("\nPlotting new data:")
# plot algorithm portfolio
# plot using the index explicitly for X-axis stability
plt.figure(figsize=(12, 6))
plt.plot(new_results.index, new_results['Algorithm_Portfolio'], 
         label='Total Portfolio Balance', color='green')
plt.title(f'{data_set}: Portfolio on HMM - Training')
plt.legend()
plt.show()

new_results['Cumulative_Market'] = np.exp(new_results['Returns'].cumsum())
new_results['Cumulative_Strategy'] = np.exp(new_results['Strategy_Returns'].cumsum())

market_final = new_results['Cumulative_Market'].iloc[-1]
strategy_final = new_results['Cumulative_Strategy'].iloc[-1]

# plot "old" method of market + buy & hold
plt.figure(figsize=(12, 6))
plt.plot(new_results['Cumulative_Market'], label='Buy & Hold', color='black')
plt.plot(new_results['Cumulative_Strategy'], label='HMM Strategy', color='green')
plt.title(f'{data_set}: HMM Strategy vs Buy & Hold - Rolling Window')
plt.legend()
plt.show()

print(f"\nFrom {last_date} to {most_recent_date}:")
print(f"  Classic Market Final Value:")
print(f"    {market_final:.2%}")

print(f"  Rolling Strategy Final Value:")
print(f"    {strategy_final:.2%}")

print("Bull state(s):")
for i in range(0, len(bull_regimes)):
    print(f"  {bull_regimes[i]}")

print("\n|Phase 8: Recent states and prediction|")

# get the state for today
# use iloc[-1:] to get the latest data point
latest_features = functions.get_two_state_values(new_results)
today_state = model.predict(latest_features)[0]
# TODO: figure out if [0] or [-1] is correct index to use

# access the transition matrix
# a matrix of [Current State, Next State] probabilities
# shape is (n_components, n_components)
transition_matrix = model.transmat_

# find the most likely next state
probs_for_tomorrow = transition_matrix[today_state]
next_predicted_state = np.argmax(probs_for_tomorrow)

# check if future state is bullish
is_bullish = 1 if next_predicted_state in bull_regimes else 0

print("Means and variances of each state (Returns & Volatility):")
for i in range(model.n_components):
    print(f"State {i}{" (Bullish)" if i in bull_regimes else ""}:")
    for j in range(X.shape[1]):
        print(f"  Mean {two_state_shape[j]}: {model.means_[i][j]:.5f}")

functions.print_most_recent_dates_and_states_table(
    end_date_range=10,
    data_set=data_set,
    data_frame=new_results,
    bull_regimes=bull_regimes
    )
print(f"Most recent date used: {new_results.index[-1].strftime("%Y-%m-%d")}")
print(f"Model prediction of most recent state: {today_state}")
print("Probabilities for tomorrow:")

for i in range(0, probs_for_tomorrow.size):
    print(f"  State {i}{" (Bullish)" if i in bull_regimes else ""}: {probs_for_tomorrow[i]:.2%}")

print(f"Predicted state for {data_set} tomorrow: {next_predicted_state}")
print(f"Action for {data_set} Tomorrow: {'🚀 BUY BUY BUY' if is_bullish else '💰 SELL SELL SELL'}")
print("Transitional matrix:")
# plot heatmap of transmat
plt.imshow(model.transmat_, aspect='auto', cmap='magma')
plt.title('Generated Transition Matrix')
plt.xticks([0, 1])
plt.xlabel('State To')
plt.yticks([0, 1])
plt.ylabel('State From')
plt.show()
