## An HMM-based algorithmic trading system in Python

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
# Silver? Oil? Anything?
start_date_spy = "2019-01-01"
end_date_spy = "2023-01-01"
start_date_btc = "2017-06-01"
end_date_btc = "2022-06-01"
data = yf.download(data_set, start=start_date_spy, end=end_date_spy)

# separate dataset into training and testing data
train_size = int(len(data) * 0.70)
train_data = data[:train_size]
test_data = data[train_size:]
# get the initial start and end dates of testing
test_start = test_data['Close'].iloc[0]
test_end = test_data['Close'].iloc[-1]

# we need features that define the "state" of the market
# common choices are returns and volatility
train_data['Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data['Range'] = (train_data['High'] - train_data['Low']) / train_data['Close']
train_data.dropna(inplace=True)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = train_data[['Returns', 'Range']].values

print(f"Data shape: {X.shape}")

print("\n|Phase 2: Build & train model|")

# number of market regimes
n_components = 4
# "full" allows features to correlate within a state
# "diag" allows features to be modeled w/o diagonal correlation
covariance_type = "diag"
# number of model iterations
n_iter = 100
# add min_covar to prevent "non-positive definite" error
min_covar=1e-3
# use Viterbi algorithm
algorithm = "viterbi"
# reinitialize parameters each time
init_params = ""

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

# define colors for up to 4 states
colors = ['green', 'red', 'blue', 'orange']
potential_model_states = ["bull", "bear", "crash", "spike"]

print("Means and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Volatility: {model.means_[i][1]:.5f}")

positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]

print("Positive return regime(s):")
for i in range(0, len(positive_return_regimes)):
    print(f"  {positive_return_regimes[i]}")

print("\n|Phase 3: Plot and verify|")

# plot price, colored by state
plt.figure(figsize=(15, 6))

for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(train_data.index[state], train_data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('Regimes Detected by HMM - Training')
plt.show()

print("\n|Phase 4: Test against training data|")
train_bull_state_list = []
# create a signal: 1 if in bullish state, 0 otherwise
# for now, grabbing any state with positive mean returns (not factoring in volatility)
print("Setting bull market signal...")
train_data['Signal'] = np.where(train_data['State'].isin(positive_return_regimes), 1, 0)


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
plt.title('HMM Strategy vs Buy & Hold - Training')
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

# predict uses the existing model parameters to predict the next state
test_states = model.predict(X_test)
test_data['State'] = test_states
print(f"Size of test data frame: {len(test_data['State'])}")

# add to dataframe and calculate returns
test_data = test_data.copy() # Avoid SettingWithCopyWarning
print("Resetting bull market signal...")
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
print("Positive return regime(s):")
for i in range(0, len(positive_return_regimes)):
    print(f"  {positive_return_regimes[i]}")

# signal = 1 if current_state in positive_state_indices else 0
test_data['Signal'] = np.where(test_data['State'].isin(positive_return_regimes), 1, 0)

print("Plotting predicted regimes:")
plt.figure(figsize=(12, 6))
for i in range(model.n_components):
    state = (test_states == i)
    plt.plot(test_data.index[state], test_data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('Regimes Detected by HMM - Initial Testing')
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
plt.title('HMM Strategy vs Buy & Hold - Backtesting')
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

print(f"Last date of test window: {last_date}")
print(f"Fetching data up to {datetime.today().strftime("%Y-%m-%d")}")

# data = yf.download(data_set, start=start_date_spy, end=end_date_spy)
new_data = yf.download(data_set, start=last_date, end=datetime.today().strftime('%Y-%m-%d'))

# combine with a bit of old data so the first "new" prediction has a training window
print("Creating new data series for rolling window...")
# use most recent trading year from old data
full_df = pd.concat([test_data.tail(252), new_data]) 

print(f"Size of full data frame: {len(full_df)}")
print(f"End date: {full_df.index[-1].strftime("%Y-%m-%d")}")

full_df['Returns'] = np.log(full_df['Close'] / full_df['Close'].shift(1))
full_df['Range'] = (full_df['High'] - full_df['Low']) / full_df['Close']
# full_df.dropna(inplace=True)

# we'll do a 1-year rolling window
# 252 trading days in a year
window_size = 252 
signals = []

print(f"Size of full data frame after dropna: {len(full_df)}")
print(f"End date: {full_df.index[-1].strftime("%Y-%m-%d")}")

for i in range(window_size, len(full_df)):
    X_train = full_df.iloc[i-window_size:i][['Returns', 'Range']].values
    current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
    
    try:
        model.fit(X_train)
        
        bull_indices = np.where(model.means_[:, 0] > 0)[0]

        print(f"Rolling window run: {i}")
        print(f"Bull indices: {bull_indices}")

        current_state = model.predict(current_features)[0]

        print(f"Next state predicted: {current_state}")
        
        signal = 1 if current_state in bull_indices else 0

        print(f"Next state is bull: {bool(signal)}")

        signals.append(signal)
    except Exception as e:
        # if model fails to converge, use signal from previous day
        print(f"Exception caught on window {i}! Exception: {e}")
        signals.append(signals[-1] if signals else 0)
        continue

# Add the signals to your dataframe
# full_results = full_df.copy()
# new_results = full_results[window_size:]
new_results = full_df.copy()
new_results['Signal'] = signals

new_results['Strategy_Returns'] = new_results['Signal'].shift(1) * new_results['Returns']
new_results['Cumulative_Market'] = np.exp(new_results['Returns'].cumsum())
new_results['Cumulative_Strategy'] = np.exp(new_results['Strategy_Returns'].cumsum())

market_final = new_results['Cumulative_Market'].iloc[-1]
strategy_final = new_results['Cumulative_Strategy'].iloc[-1]

plt.figure(figsize=(12, 6))
plt.plot(new_results['Cumulative_Market'], label='Buy & Hold', color='black')
plt.plot(new_results['Cumulative_Strategy'], label='HMM Strategy', color='green')
plt.title(f'HMM Strategy vs Buy & Hold: New Results')
plt.legend()
plt.show()

print(f"Classic Market Final Value on {new_results.index[-1]}:")
print(f"  {market_final:.2%}")

print(f"Rolling Strategy Final Value on {full_df.index[-1]}:")
print(f"  {strategy_final:.2%}")

print("Bull state(s):")
for i in range(0, len(positive_return_regimes)):
    print(f"  {positive_return_regimes[i]}")

# TODO: return 10 most recent dates and states
end_date_range = 10

# make table
# print("|--- Date ---|--- State ---|")
# for i in range(0, end_date_range):
#     # reverse index to go in order of dates, from -10 to -1
#     index = 10 - i
#     print(f"| {new_results.index[-index]} | {new_results['State'].iloc[-index]} |")
