# training day
# using ModelTrainer class to build and return winning model
# compare score and time with model_curve and day_trip

import model_trainer
import functions
import time
import numpy as np

data_set = "SPY"
start_date = "2020-01-01"
end_date = "2025-01-01"
max_model_count = 32

# create new model trainer; get best model
trainer = model_trainer.ModelTrainer(data_set, start_date, end_date)
model, new_results = trainer.train_new_day_model(max_model_count)

# get the state for most recent time interval
# use iloc[-1:] to get the latest data point
most_recent_features = new_results.iloc[-1:][['Returns', 'Range']].values
most_recent_state = model.predict(most_recent_features)[0]
# most_recent_state = model.predict(most_recent_features)[-1]

# access the transition matrix
# a matrix of [Current State, Next State] probabilities
# shape is (n_components, n_components)
transition_matrix = model.transmat_

# find the most likely next state
probs_for_next_state = transition_matrix[most_recent_state]
next_predicted_state = np.argmax(probs_for_next_state)

# get bull regimes and check if future state is bullish
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
volatility_threshold = 0.070
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for regime in positive_return_regimes:
    if regime in low_volatility_regimes:
        bull_regimes.append(regime)
is_bullish = 1 if next_predicted_state in bull_regimes else 0

# leaving this for now so that I have some idea of what's going on
print("\nMeans and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}{" (Bullish)" if i in bull_regimes else ""}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Volatility: {model.means_[i][1]:.5f}")

# print table of recent states and prediction
functions.print_most_recent_dates_and_states_table(10, data_set, new_results, bull_regimes)

print(f"Most recent date used: {new_results.index[-1].strftime("%Y-%m-%d")}")
print(f"Model prediction of most recent state: {most_recent_state}")
print("Probabilities for tomorrow:")

for i in range(0, probs_for_next_state.size):
    print(f"  State {i}{" (Bullish)" if i in bull_regimes else ""}: {probs_for_next_state[i]:.2%}")

print(f"Predicted state for {data_set} tomorrow: {next_predicted_state}")
print(f"Action for {data_set} Tomorrow: {'🚀 BUY BUY BUY' if is_bullish else '💰 SELL SELL SELL'}")
