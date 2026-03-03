## An HMM-based algorithmic trading system in Python
import functions
import baby_algo as algo

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from datetime import datetime

print("\n|Phase 1: Fetch data|")

# use SPY (S&P 500 ETF) for a good mix of regimes
data_set = "SPY"
# BTC-USD for regimes in crypto
# data_set = "BTC-USD"
# Silver? Oil? Anything?
print(f"Market: {data_set}")

# pick a good start date?
start_date = "2021-01-01"
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

# need to normalize OHLC data before attempting to train on it
train_data['Open_Normal'] = np.log(train_data['Open'] / train_data['Open'].shift(1))
train_data['High_Normal'] = np.log(train_data['High'] / train_data['High'].shift(1))
train_data['Low_Normal'] = np.log(train_data['Low'] / train_data['Low'].shift(1))
train_data['Close_Normal'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data.dropna(inplace=True)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = functions.normalize_two_state_data(train_data)

# try the OHLC now
# X = train_data[['Open_Normal', 'High_Normal', 'Low_Normal', 'Close_Normal']].values

# why not both?
# X = train_data[['Returns', 'Range', 'Open_Normal', 'High_Normal', 'Low_Normal', 'Close_Normal']].values


print("\n|Phase 2: Build & train model|")

# number of market regimes
n_components = 2
# "full" allows features to correlate within a state
# "diag" allows features to be modeled w/o diagonal correlation
covariance_type = "full"
# number of model iterations
n_iter = 100
# add min_covar to prevent "non-positive definite" error
min_covar=1e-3
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

two_state_shape = ['Returns', 'Volatility']
four_state_shape = ['Open', 'High', 'Low', 'Close']
six_state_shape = ['Returns', 'Range', 'Open', 'High', 'Low', 'Close']

print("Means and variances of each state (Returns & Volatility):")
for i in range(model.n_components):
    print(f"State {i}:")
    for j in range(X.shape[1]):
        print(f"  Mean {j}: {model.means_[i][j]:.5f}")

# print("Means and variances of each state (OHLC):")
# for i in range(model.n_components):
#     print(f"State {i}:")
#     for j in range(X.shape[1]):
#         print(f"  Mean {six_state_shape[j]}: {model.means_[i][j]:.5f}")

positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]

print("Positive return regime(s):")
for i in range(0, len(positive_return_regimes)):
    print(f"  {positive_return_regimes[i]}")

# try adding volatility check
volatility_threshold = 0.070
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]

bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

print("Bull regime(s):")
for i in range(0, len(bull_regimes)):
    print(f"  {bull_regimes[i]}")

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
# for now, grabbing any state with positive mean returns (not factoring in volatility)
print("Setting bull market signal...")
train_data['Signal'] = np.where(train_data['State'].isin(bull_regimes), 1, 0)

# calculate returns on HMM
# We shift signal by 1 because we trade at the close based on today's state for tomorrow
train_data['Strategy_Returns'] = train_data['Signal'].shift(1) * train_data['Returns']

# calculate buy & hold returns
train_data['Cumulative_Market'] = np.exp(train_data['Returns'].cumsum())
train_data['Cumulative_Strategy'] = np.exp(train_data['Strategy_Returns'].cumsum())

# plot training performance
print("Plotting training performance:")
plt.figure(figsize=(12, 6))
plt.plot(train_data['Cumulative_Market'], label='Buy & Hold', color='gray')
plt.plot(train_data['Cumulative_Strategy'], label='HMM Strategy', color='orange')
plt.title(f'{data_set}: HMM Strategy vs Buy & Hold - Training')
plt.legend()
plt.show()

# Run the algorithm
print("Running new algorithm...")
algorithm_portfolio = algo.algorithm(train_data)


train_data['Algorithm_Portfolio'] = algorithm_portfolio
train_data['Algorithm_Portfolio'] = pd.to_numeric(train_data['Algorithm_Portfolio'])

# Plot using the index explicitly for X-axis stability
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Algorithm_Portfolio'], 
         label='Total Portfolio Balance', color='green')
plt.title(f'{data_set}: Portfolio on HMM - Training')
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

print("\n|Phase 6: Initial test on new data|")

# prepare the test features (must be the same columns as training)
test_data['Returns'] = np.log(test_data['Close'] / test_data['Close'].shift(1))
test_data['Range'] = (test_data['High'] - test_data['Low']) / test_data['Close']
test_data.dropna(inplace=True)
X_test = test_data[['Returns', 'Range']].values

# need to normalize OHLC data before attempting to train on it
# test_data['Open_Normal'] = np.log(test_data['Open'] / test_data['Open'].shift(1))
# test_data['High_Normal'] = np.log(test_data['High'] / test_data['High'].shift(1))
# test_data['Low_Normal'] = np.log(test_data['Low'] / test_data['Low'].shift(1))
# test_data['Close_Normal'] = np.log(test_data['Close'] / test_data['Close'].shift(1))
# test_data.dropna(inplace=True)
# X_test = test_data[['Returns', 'Range', 'Open_Normal', 'High_Normal', 'Low_Normal', 'Close_Normal']].values

# predict uses the existing model parameters to predict the next state
test_states = model.predict(X_test)
test_data['State'] = test_states
print(f"Size of test data frame: {len(test_data['State'])}")

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
plt.title(f'{data_set}: Regimes Detected by HMM - Backesting')
plt.show()

# calculate returns (shift by 1 to avoid look-ahead bias)
test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']

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

# next phase - rolling window and walk-forward?
# roll up to present day; 
# guess latest regime for most recent market close; 
# compare with actual results for a final test.
# then apply model to the next day?

print("\n|Phase 7: Rolling window|")
last_date = test_data.index[-1].strftime("%Y-%m-%d")
most_recent_date = datetime.today().strftime("%Y-%m-%d")
print(f"Last date of test window: {last_date}")
print(f"Fetching data up to {most_recent_date}")

new_data = yf.download(data_set, start=last_date, end=most_recent_date)

# flatten MultiIndex columns if they exist
# if isinstance(new_data.columns, pd.MultiIndex):
#     new_data.columns = new_data.columns.get_level_values(0)

# combine with a bit of old data so the first "new" prediction has a training window
print("Creating new data series for rolling window...")

# use most recent trading year from old data
# drop the 'Returns' and 'Range' columns from the tail of test_data 
# so they don't create NaN columns in the new_data section during concat
# then concatenate and remove duplicates (the overlapping last_date)
buffer_data = test_data.tail(252)[['Open', 'High', 'Low', 'Close', 'Volume']]
full_df = pd.concat([buffer_data, new_data])
full_df = full_df[~full_df.index.duplicated(keep='last')]
print(f"Size of full data frame: {len(full_df)}")

full_df['Returns'] = np.log(full_df['Close'] / full_df['Close'].shift(1))
full_df['Range'] = (full_df['High'] - full_df['Low']) / full_df['Close']
# full_df.dropna(inplace=True)

full_df['Open_Normal'] = np.log(full_df['Open'] / full_df['Open'].shift(1))
full_df['High_Normal'] = np.log(full_df['High'] / full_df['High'].shift(1))
full_df['Low_Normal'] = np.log(full_df['Low'] / full_df['Low'].shift(1))
full_df['Close_Normal'] = np.log(full_df['Close'] / full_df['Close'].shift(1))
full_df.dropna(inplace=True)

print(f"New data successfully merged. Total rows: {len(full_df)}")
print(f"End date: {full_df.index[-1].strftime("%Y-%m-%d")}")

# plot market in new data to get an idea of how it has behaved in recent past
# and also to catch breath before the rolling window
print("Plotting market between training date and now:\n")
plt.figure(figsize=(12, 6))
plt.plot(full_df['Close'], label=data_set, color='black')
plt.title(f'{data_set} Market Outlook - Rolling Window')
plt.legend()
plt.show()

# we'll do a 1-year rolling window
# 252 trading days in a year
window_size = 252 
run_count = 0
signals = []
states = []
exception_list = []

print(f"Executing {window_size}-day rolling window from {last_date} to {most_recent_date}:")

for i in range(window_size, len(full_df)):
    run_count += 1
    X_train = full_df.iloc[i-window_size:i][['Returns', 'Range']].values
    current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
    # X_train = full_df.iloc[i-window_size:i][['Returns', 'Range', 'Open_Normal', 'High_Normal', 'Low_Normal', 'Close_Normal']].values
    # current_features = full_df.iloc[i:i+1][['Returns', 'Range', 'Open_Normal', 'High_Normal', 'Low_Normal', 'Close_Normal']].values
    
    try:
        model.fit(X_train)
        current_state = model.predict(current_features)[0]

        positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
        low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
        bull_regimes = []
        for i in positive_return_regimes:
            if i in low_volatility_regimes:
                bull_regimes.append(i)

        signal = 1 if current_state in bull_regimes else 0

        signals.append(signal)
        states.append(current_state)
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

# set new bullish states in case they've changed
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

# Add the signals to your dataframe
full_results = full_df.copy()
new_results = full_results[window_size:]
new_results['Signal'] = signals
new_results['State'] = states

new_results['Strategy_Returns'] = new_results['Signal'].shift(1) * new_results['Returns']
new_results['Cumulative_Market'] = np.exp(new_results['Returns'].cumsum())
new_results['Cumulative_Strategy'] = np.exp(new_results['Strategy_Returns'].cumsum())

market_final = new_results['Cumulative_Market'].iloc[-1]
strategy_final = new_results['Cumulative_Strategy'].iloc[-1]

print("Plotting new data:")

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

# print("Means and variances of each state (OHLC):")
# for i in range(model.n_components):
#     print(f"State {i}:")
#     for j in range(X.shape[1]):
#         print(f"  Mean {six_state_shape[j]}: {model.means_[i][j]:.5f}")

# get the state for today
# use iloc[-1:] to get the latest data point
today_features = new_results.iloc[-1:][['Returns', 'Range']].values
# today_features = new_results.iloc[-1:][['Returns', 'Range', 'Open_Normal', 'High_Normal', 'Low_Normal', 'Close_Normal']].values
today_state = model.predict(today_features)[0]

# access the transition matrix
# a matrix of [Current State, Next State] probabilities
# shape is (n_components, n_components)
transition_matrix = model.transmat_

# find the most likely next state
probs_for_tomorrow = transition_matrix[today_state]
next_predicted_state = np.argmax(probs_for_tomorrow)

# check if future state is bullish
is_bullish = 1 if next_predicted_state in bull_regimes else 0

print("\nMeans and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}{" (Bullish)" if i in bull_regimes else ""}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Volatility: {model.means_[i][1]:.5f}")

# table of most recent dates and states
end_date_range = 10
print(f"\nMarket: {data_set}")
print("|--- Date ---|-- State --|---Bull?---|")
for i in range(0, end_date_range):
    # reverse index to go in order of dates, from -10 to -1
    index = end_date_range - i
    print_date = new_results.index[-index].strftime("%Y-%m-%d")
    print_state = new_results['State'].iloc[-index]
    print(f"| {print_date} |     {print_state}     |    {"Yes" if print_state in bull_regimes else "No "}    |") # formatting
print("|------------|-----------|-----------|\n")

print(f"Today's state: {today_state}")
print("Probabilities for tomorrow:")

for i in range(0, probs_for_tomorrow.size):
    print(f"  State {i}{" (Bullish)" if i in bull_regimes else ""}: {probs_for_tomorrow[i]:.2%}")

print(f"Predicted state for {data_set} tomorrow: {next_predicted_state}")
print(f"Action for {data_set} Tomorrow: {'🚀 BUY BUY BUY' if is_bullish else '💰 SELL SELL SELL'}")
