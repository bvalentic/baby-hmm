## An HMM-based algorithmic trading system in Python

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Phase 1 - get data
print("Fetching data...")

# use SPY (S&P 500 ETF) for a good mix of regimes
# data = yf.download("SPY", start="2017-06-01", end="2023-01-01")
# BTC-USD for crypto generalities
data = yf.download("BTC-USD", start="2017-06-01", end="2023-01-01")

# using dataset of choice, separate into training and testing data
dataset = data
#train_size = int(len(dataset) * 0.75)
train_size = int(len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:]

# Feature engineering
# we need features that define the "state" of the market
# common choices are returns and volatility
train_data['Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data['Range'] = (train_data['High'] - train_data['Low']) / train_data['Close']
train_data.dropna(inplace=True)

# Prepare data for hmmlearn
# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = train_data[['Returns', 'Range']].values

print(f"Data shape: {X.shape}")

# Phase 2 - build and train model
print("Creating model...")

# n_components = 3 (we assume 3 market regimes)
# covariance_type = "full" allows features to correlate within a state
model = hmm.GaussianHMM(n_components=3, covariance_type="full", algorithm="viterbi", n_iter=100)

# fit the model to the data
print("Fitting model...")
model.fit(X)

# Predict states
# model estimates which "hidden state" generated the data for each day
print("Estimating states...")
hidden_states = model.predict(X)

# Add states back to the dataframe for analysis
train_data['State'] = hidden_states

print("Model training complete.")
print("Means and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Range (Vol): {model.means_[i][1]:.5f}")

# Phase 3 - plot and show

# plot price, colored by state
plt.figure(figsize=(15, 6))

# define colors for states (0, 1, 2)
colors = ['green', 'red', 'blue'] 

for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(train_data.index[state], train_data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('Regimes Detected by HMM')

# If able to display:
plt.show()

# If using container or headless:
#plt.savefig('hmm_regimes.png')
#print("Plot saved as hmm_regimes.png")

