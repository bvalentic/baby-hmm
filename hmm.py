## An HMM-based algorithmic trading system in Python

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Phase 1 - get data
print("Fetching data...")

# use SPY (S&P 500 ETF) for a good mix of regimes
data = yf.download("SPY", start="2018-01-01", end="2023-01-01")

# Feature engineering
# we need features that define the "state" of the market
# common choices are returns and volatility
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['Range'] = (data['High'] - data['Low']) / data['Close']
data.dropna(inplace=True)

# Prepare data for hmmlearn
# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = data[['Returns', 'Range']].values

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
data['State'] = hidden_states

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
    plt.plot(data.index[state], data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('S&P 500 Regimes Detected by HMM')

# If able to display:
plt.show()

# If using container or headless:
#plt.savefig('hmm_regimes.png')
#print("Plot saved as hmm_regimes.png")

