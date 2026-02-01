## An HMM-based algorithmic trading system in Python

import numpy as np
import yfinance as yf
import pandas as pd

# Download data
ticker = yf.Ticker("SPY")
data = ticker.history(period="5y")

# Calculate returns and features

data['returns'] = data['Close'].pct_change()
data['log_returns'] = np.log_data['Close'] / data['Close'].shift(1)
data['volatility'] = data['returns'].rolling(window=20).std()


