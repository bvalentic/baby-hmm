## An HMM-based algorithmic trading system in Python

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm

# 1. Get Data
# Let's use the SPY (S&P 500 ETF) for a good mix of regimes
data = yf.download("SPY", start="2018-01-01", end="2025-06-01")

# 2. Feature Engineering
# We need features that define the "state" of the market. 
# Common choices: Log Returns and Volatility (Range).
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['Range'] = (data['High'] - data['Low']) / data['Close']
data.dropna(inplace=True)

# 3. Prepare Data for hmmlearn
# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = data[['Returns', 'Range']].values

print(f"Data shape: {X.shape}")

# Phase 2 - build and train

# 4. Build and Train
# n_components = 3 (we assume 3 market regimes)
# covariance_type = "full" allows features to correlate within a state
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)

# Fit the model to the data
model.fit(X)

# 5. Predict States
# The model estimates which "hidden state" generated the data for each day
hidden_states = model.predict(X)

# Add states back to the dataframe for analysis
data['State'] = hidden_states

print("Model training complete.")
print("Means and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Range (Vol): {model.means_[i][1]:.5f}")
