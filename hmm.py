## An HMM-based algorithmic trading system in Python

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm

print("Phase 1: Fetch data\n")

start_date = "2017-06-01"
end_date = "2023-01-01"
start_date_btc = "2016-01-01"
end_date_btc = "2022-01-01"
# use SPY (S&P 500 ETF) for a good mix of regimes
data = yf.download("SPY", start=start_date, end=end_date)
# BTC-USD for regimes in crypto
# data = yf.download("BTC-USD", start=start_date_btc, end=end_date_btc)

# separate dataset into training and testing data
train_size = int(len(data) * 0.70)
train_data = data[:train_size]
test_data = data[train_size:]

# we need features that define the "state" of the market
# common choices are returns and volatility
train_data['Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data['Range'] = (train_data['High'] - train_data['Low']) / train_data['Close']
train_data.dropna(inplace=True)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = train_data[['Returns', 'Range']].values

print(f"Data shape: {X.shape}")

print("\nPhase 2: Build & train model")

# number of market regimes
n_components = 4
# "full" allows features to correlate within a state
covariance_type = "full"
print("Creating model...")
model = hmm.GaussianHMM(n_components, covariance_type, algorithm="viterbi", n_iter=100)

print("Fitting model to data...")
model.fit(X)

# model estimates which "hidden state" generated the data for each day
print("Estimating states...")
hidden_states = model.predict(X)

# add states back to the dataframe for analysis
train_data['State'] = hidden_states

print("Model training complete.")

bullish_mean_return = model.means_[0][0]
bull_index = 0

print("Means and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Range (Vol): {model.means_[i][1]:.5f}")

    if model.means_[i][0] > bullish_mean_return:
        bullish_mean_return = model.means_[i][0]
        bull_index = i

print(f"\nExpected bullish regime: {bull_index}")

print("\nPhase 3: Plot and show")

# plot price, colored by state
plt.figure(figsize=(15, 6))

# define colors for 4 states
colors = ['green', 'red', 'blue', 'orange'] 

for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(train_data.index[state], train_data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('Regimes Detected by HMM')

# if able to display:
plt.show()
# if using container or headless:
#plt.savefig('hmm_regimes.png')
#print("Plot saved as hmm_regimes.png")

expected_bullish_state = bull_index
observed_bullish_state = int(input("Which state is the bullish state? "))
if (expected_bullish_state == observed_bullish_state):
    print("\n🤖 Observed bullish state matched expected state!")
else:
    print("\n👀 Observed bullish state did not match expected state!")

print("\nNext phase: Testing against training data \n")

# create a signal: 1 if in bullish state, 0 otherwise
print("Setting bull market signal...")
train_data['Signal'] = np.where(train_data['State'] == observed_bullish_state, 1, 0)

# calculate returns on HMM
# We shift signal by 1 because we trade at the close based on today's state for tomorrow
train_data['Strategy_Returns'] = train_data['Signal'].shift(1) * train_data['Returns']

# calculate buy & hold returns
train_data['Cumulative_Market'] = np.exp(train_data['Returns'].cumsum())
train_data['Cumulative_Strategy'] = np.exp(train_data['Strategy_Returns'].cumsum())

# plot training performance
print("\nPlotting training performance:\n")
plt.figure(figsize=(12, 6))
plt.plot(train_data['Cumulative_Market'], label='Buy & Hold', color='gray')
plt.plot(train_data['Cumulative_Strategy'], label='HMM Strategy', color='orange')
plt.title('HMM Strategy vs Buy & Hold - Training')
plt.legend()
plt.show()

print("Phase 4: Analyze training results")

# Use .iloc[-1] to get the value in the last row of that specific column
print("Simple comparison: ")
market_final_train = train_data['Cumulative_Market'].iloc[-1]
strategy_final_train = train_data['Cumulative_Strategy'].iloc[-1]

print(f"Training Period Market Return: {(market_final_train - 1):.2%}")
print(f"Training Period Strategy Return: {(strategy_final_train - 1):.2%}")

if market_final_train > strategy_final_train:
    print("\n📈 Buy & Hold outperformed the HMM in training.\n")
else:
    print("\n🤖 The HMM strategy beat the market in training!\n")

# calculate Sharpe ratio

print("Phase 5: Test on new data")

# prepare the test features (must be the same columns as training)
test_data['Returns'] = np.log(test_data['Close'] / test_data['Close'].shift(1))
test_data['Range'] = (test_data['High'] - test_data['Low']) / test_data['Close']
test_data.dropna(inplace=True)
X_test = test_data[['Returns', 'Range']].values

# predict uses the existing model parameters to predict the next state
test_states = model.predict(X_test)

# add to dataframe and calculate returns
test_data = test_data.copy() # Avoid SettingWithCopyWarning
test_data['State'] = test_states
test_data['Signal'] = np.where(test_data['State'] == observed_bullish_state, 1, 0)

# plot predicted regimes
for i in range(model.n_components):
    state = (test_states == i)
    plt.plot(test_data.index[state], test_data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('Regimes Detected by HMM')
plt.show()

# calculate returns (shift by 1 to avoid look-ahead bias)
test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']

# calculate cumulative growth
test_data['Cumulative_Market'] = np.exp(test_data['Returns'].cumsum())
test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())

# plot test performance
print("\nPlotting test performance:\n")
plt.figure(figsize=(12, 6))
plt.plot(test_data['Cumulative_Market'], label='Buy & Hold', color='gray')
plt.plot(test_data['Cumulative_Strategy'], label='HMM Strategy', color='orange')
plt.title('HMM Strategy vs Buy & Hold - Backtesting')
plt.legend()
plt.show()

market_final_test = test_data['Cumulative_Market'].iloc[-1]
strategy_final_test = test_data['Cumulative_Strategy'].iloc[-1]

print(f"Test Period Market Return: {(market_final_test - 1):.2%}")
print(f"Test Period Strategy Return: {(strategy_final_test - 1):.2%}")

if strategy_final_test > market_final_test:
    print("\n✅ The HMM beat the market!")
else:
    print("\n❌ The HMM underperformed. It might need different features or state counts.")

# next phase - apply to the next day?
