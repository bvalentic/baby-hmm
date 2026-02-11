## An HMM-based algorithmic trading system in Python

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from datetime import datetime

print("\n|Phase 1: Fetch data|")

start_date_spy = "2018-01-01"
end_date_spy = "2023-01-01"
start_date_btc = "2017-06-01"
end_date_btc = "2022-06-01"
# use SPY (S&P 500 ETF) for a good mix of regimes
data = yf.download("SPY", start=start_date_spy, end=end_date_spy)
# BTC-USD for regimes in crypto
# data = yf.download("BTC-USD", start=start_date_btc, end=end_date_btc)
# Silver? Oil? Anything?

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

# define initial bull state as state 0
bullish_mean_return = model.means_[0][0]
bullish_mean_range = model.means_[0][1]
bull_index = 0
spike_exists = False

# define colors for up to 4 states
colors = ['green', 'red', 'blue', 'orange']
potential_model_states = ["bull", "bear", "crash", "spike"]
positive_return_regimes = []

print("Means and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Volatility: {model.means_[i][1]:.5f}")

    if model.means_[i][0] > 0:
        positive_return_regimes.append(i)

if len(positive_return_regimes) > 1:
    spike_exists = True
    # Set regime variable (bull, bear, etc.) based on mean returns & range
    for i in range(len(positive_return_regimes) - 1):
        # "spike" state has a positive return with higher volatility
        regime = positive_return_regimes[i]
        next_regime = positive_return_regimes[i + 1]
        if model.means_[regime][1] > model.means_[next_regime][1]:
            spike_index = regime
            bull_index = next_regime
        else:
            bull_index = regime
            spike_index = next_regime
# else only set the bull regime
else:
    bull_index = positive_return_regimes[0]

print(f"Expected bullish regime: {bull_index}")
if(spike_exists):
    expected_spike_state = spike_index
    print("Spike regime detected!")
    print(f"Expected spike regime: {spike_index}")

print("\n|Phase 3: Plot and verify|")

# plot price, colored by state
plt.figure(figsize=(15, 6))

for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(train_data.index[state], train_data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('Regimes Detected by HMM - Training')

# if able to display:
plt.show()
# if using container or headless:
#plt.savefig('hmm_regimes_training.png')
#print("Plot saved as hmm_regimes_training.png")

expected_bullish_state = bull_index
observed_bullish_state = int(input("Which state is the bullish state? "))

if (expected_bullish_state == observed_bullish_state):
    print("🤖 Observed bullish state matched expected state!")
else:
    print("👀 Observed bullish state did not match expected state!")

if(spike_exists):
    observed_spike_state = int(input("Which state is the spike? "))
    if (expected_spike_state == observed_spike_state):
        print("🤖 Observed spike state matched expected state!")
    else:
        print("👀 Observed spike state did not match expected state!")

print("\n|Phase 4: Test against training data|")

# create a signal: 1 if in bullish state, 0 otherwise
print("Setting bull market signal...")
is_bull_regime = train_data['State'] == observed_bullish_state
if(spike_exists):
    is_spike_regime = train_data['State'] == observed_spike_state
    is_good_market = train_data['State'] == (observed_bullish_state or observed_spike_state)
    train_data['Signal'] = np.where(is_good_market, 1, 0)
else:
    train_data['Signal'] = np.where(is_bull_regime, 1, 0)

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

# add to dataframe and calculate returns
test_data = test_data.copy() # Avoid SettingWithCopyWarning
test_data['State'] = test_states
test_data['Signal'] = np.where(test_data['State'] == observed_bullish_state, 1, 0)

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
