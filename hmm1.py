## An HMM-based algorithmic trading system in Python

import numpy as np
import pandas as pd
import yfinance as yf
##import matplotlib.pyplot as plt
##from hmmlearn import hmm

# 1. Get Data
# Let's use the SPY (S&P 500 ETF) for a good mix of regimes
data = yf.download("SPY", start="2018-01-01", end="2023-01-01")

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

