# Parking Lot

Markdown parking lot for miscellaneous code as more features are onboarded

```python
from datetime import datetime

# 1. Fetch "New" Data up to today
last_date = test_data.index[-1]
new_data = yf.download("SPY", start=last_date, end=datetime.now().strftime('%Y-%m-%d'))

# Combine with a bit of old data so the first "new" prediction has a training window
full_df = pd.concat([test_data.tail(500), new_data]) 
full_df['Returns'] = np.log(full_df['Close'] / full_df['Close'].shift(1))
full_df['Range'] = (full_df['High'] - full_df['Low']) / full_df['Close']
full_df.dropna(inplace=True)

# 2. The Walk-Forward Loop
window_size = 252 # Use 1 year of trading days to train
signals = []

# We start from the window_size and move 1 step at a time
for i in range(window_size, len(full_df)):
    # Slice the training window
    train_window = full_df.iloc[i-window_size:i]
    X_train = train_window[['Returns', 'Range']].values
    
    # Current day features to predict
    current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
    
    # Fit model on the window
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    model.fit(X_train)
    
    # Identify the 'Bull' state programmatically (highest mean return)
    # This prevents the "State Flipping" issue I mentioned earlier!
    bull_state = np.argmax(model.means_[:, 0])
    
    # Predict today's state
    current_state = model.predict(current_features)[0]
    
    # Record signal: 1 if Bull, 0 otherwise
    signals.append(1 if current_state == bull_state else 0)

# Add signals back to the dataframe (matched to the dates after the first window)
new_results = full_df.iloc[window_size:].copy()
new_results['Signal'] = signals

new_results['Strategy_Returns'] = new_results['Signal'].shift(1) * new_results['Returns']
new_results['Cumulative_Strategy'] = np.exp(new_results['Strategy_Returns'].cumsum())

print(f"Rolling Strategy Final Value: {new_results['Cumulative_Strategy'].iloc[-1]:.2f}")
```

