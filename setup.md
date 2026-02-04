# Python HMM for Algorithmic Trading

Welcome back to Python! Using Hidden Markov Models (HMMs) for algorithmic trading is a fascinating approach. Unlike standard forecasting models that try to predict the exact price tomorrow, HMMs are typically used for **Regime Detection**.

In trading, the "Hidden States" usually correspond to market conditions (e.g., Bull Market, Bear Market, Sideways/Choppy) that drive the "Observable" data (price returns, volatility).

Here is a step-by-step guide to building, training, and evaluating a simple Gaussian HMM using the popular library `hmmlearn`.

### 1. Prerequisites and Setup

We will use:

* `yfinance`: To easily get stock data.
* `hmmlearn`: The standard library for HMMs in Python.
* `matplotlib`: For visualization.

First, you would run this in your terminal to install the libraries:

```bash
pip install hmmlearn yfinance matplotlib pandas numpy

```

---

### 2. The Implementation (Step-by-Step)

Here is a complete, runnable script. I have broken it down into the phases you requested: Build, Train, and Test (Backtest).

#### Phase A: Data Preparation

HMMs work best on stationary data (data that doesn't trend to infinity like raw prices). We will calculate **daily returns** and **volatility** as our features.

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm

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

```

#### Phase B: Building and Training the Model

We will use a `GaussianHMM`. We have to guess the number of states (components). `3` is a common starting point (Bull, Bear, Neutral).

```python
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

```

---

### 3. Visual Evaluation (The "Eye Test")

In unsupervised learning (like HMMs), you don't calculate "accuracy" because you don't have "True Labels" (no one tells you *for a fact* that yesterday was a Bear market).

The best evaluation is visual: **Do the states align with reality?**

```python
# Plotting the price colored by State
plt.figure(figsize=(15, 6))

# Define colors for states (0, 1, 2)
colors = ['green', 'red', 'blue'] 

for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(data.index[state], data['Close'][state], '.', label=f'State {i}', color=colors[i], markersize=3)

plt.legend()
plt.title('S&P 500 Regimes Detected by HMM')
plt.show()

```

**How to interpret:**
Look at the graph generated.

* **High Volatility State:** Does one color appear mostly during crashes (e.g., March 2020)? That is your "Bear/Crash" state.
* **Low Volatility/Positive Return:** Does one color appear during smooth uptrends? That is your "Bull" state.

---

### 4. Mathematical Evaluation (Backtesting)

To "Test" the model for trading, you run a backtest.

* **Strategy:** If the model predicts the "Bull State" (let's say State 0), we hold the stock. If it predicts "Bear State" (State 1), we go to cash.

*Note: You must identify which state is which based on the printout of `means_` in Phase B. For this code, I assume State 0 has positive returns.*

```python
# Assume State 0 is the "Bullish" state based on our inspection
# (You might need to swap this number based on your specific run)
bullish_state = 0 

# Create a signal: 1 if in bullish state, 0 otherwise
data['Signal'] = np.where(data['State'] == bullish_state, 1, 0)

# Calculate Strategy Returns
# We shift signal by 1 because we trade at the close based on today's state for tomorrow
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']

# Calculate Cumulative Returns
data['Cumulative_Market'] = np.exp(data['Returns'].cumsum())
data['Cumulative_Strategy'] = np.exp(data['Strategy_Returns'].cumsum())

# Plot Performance
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Market'], label='Buy & Hold', color='gray')
plt.plot(data['Cumulative_Strategy'], label='HMM Strategy', color='orange')
plt.title('HMM Strategy vs Buy & Hold')
plt.legend()
plt.show()

```




