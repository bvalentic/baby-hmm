import functions

import time
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from datetime import datetime

class ModelTrainer():
    def __init__(self, data_set, start_date, end_date):
        self.data_set = data_set
        self.start_date = start_date
        self.end_date = end_date

    def train_new_day_model(self, max_model_count):
        # because this is the day model, use 1-day interval
        interval = "1d"

        data = yf.download(self.data_set, start=self.start_date, end=self.end_date, interval=interval)

        # separate dataset into training and testing data
        train_size = int(len(data) * 0.70)
        train_data = data[:train_size].copy()
        test_data = data[train_size:].copy()

        # more data
        last_date = test_data.index[-1].strftime("%Y-%m-%d")
        most_recent_date = datetime.today().strftime("%Y-%m-%d")

        new_data = yf.download(self.data_set, start=last_date, end=most_recent_date)

        # using returns and volatility:
        train_data['Returns'], train_data['Range'] = functions.get_two_state_data(train_data)

        # hmmlearn expects a 2D array of shape (n_samples, n_features)
        X = functions.get_two_state_values(train_data)

        # number of market regimes
        n_components = 2
        # "full" allows features to correlate within a state
        # "diag" allows features to be modeled w/o diagonal correlation
        covariance_type = "full"
        # number of model iterations
        n_iter = 100
        # add min_covar to prevent "non-positive definite" error
        min_covar=1e-4
        # use Viterbi algorithm
        algorithm = "viterbi"
        # "" keeps set variables
        # "stmc" reinitializes parameters each time 
        init_params = "stmc"

        model_number = 0

        model_list = []
        score_list = []
        win_rate_list = []
        signals_and_states = []

        start = time.time()

        while model_number < max_model_count:

            model = hmm.GaussianHMM(
                n_components=n_components, 
                covariance_type=covariance_type,
                min_covar=min_covar,
                n_iter=n_iter, 
                algorithm=algorithm,
                init_params=init_params
            )
            model.fit(X)

            # model estimates which "hidden state" generated the data for each day
            hidden_states = model.predict(X)

            # add states back to the dataframe for analysis
            train_data['State'] = hidden_states

            positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]

            # try adding volatility check
            volatility_threshold = 0.070
            low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]

            bull_regimes = []
            for i in positive_return_regimes:
                if i in low_volatility_regimes:
                    bull_regimes.append(i)

            # create a signal: 1 if in bullish state, 0 otherwise
            train_data['Signal'] = np.where(train_data['State'].isin(bull_regimes), 1, 0)
            
            # prepare the test features (must be the same columns as training)
            test_data['Returns'], test_data['Range'] = functions.get_two_state_data(test_data)

            X_test = functions.get_two_state_values(test_data)

            # predict uses the existing model parameters to predict the next state
            test_states = model.predict(X_test)
            test_data['State'] = test_states

            # add to dataframe and calculate returns
            test_data = test_data.copy() # Avoid SettingWithCopyWarning
            positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
            low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
            bull_regimes = []
            for i in positive_return_regimes:
                if i in low_volatility_regimes:
                    bull_regimes.append(i)

            test_data['Signal'] = np.where(test_data['State'].isin(bull_regimes), 1, 0)

            # next phase - rolling window and walk-forward
            # roll up to present day; 
            # guess latest regime for most recent market close; 
            # compare with actual results for a final test.
            # then apply model to the next day?

            # we'll do a 1-year rolling window
            # 252 trading days in a year
            window_size = 252 

            # flatten MultiIndex columns if they exist
            # if isinstance(new_data.columns, pd.MultiIndex):
            #     new_data.columns = new_data.columns.get_level_values(0)

            # combine with a bit of old data so the first "new" prediction has a training window

            # use most recent trading year from old data
            # drop the 'Returns' and 'Range' columns from the tail of test_data 
            # so they don't create NaN columns in the new_data section during concat
            # then concatenate and remove duplicates (the overlapping last_date)
            buffer_data = test_data.tail(window_size)[['Open', 'High', 'Low', 'Close', 'Volume']]
            full_df = pd.concat([buffer_data, new_data])
            full_df = full_df[~full_df.index.duplicated(keep='last')]

            full_df['Returns'], full_df['Range'] = functions.get_two_state_data(full_df)

            # plot market in new data to get an idea of how it has behaved in recent past
            # and also to catch breath before the rolling window

            # now before rolling window, check the model's prediction
            # get the transitional matrix for the final state
            test_features_final = test_data.iloc[-1:][['Returns', 'Range']].values
            test_prediction_final = model.predict(test_features_final)[0]
            test_transmat_final = model.transmat_[test_prediction_final]
            # get largest state, to see if first index of new model.predict matches 
            test_predicted_chance_final = test_transmat_final.max()
            test_predicted_state_final = np.where(test_transmat_final == test_predicted_chance_final)[0][0]

            run_count = 0
            signals = []
            states = []
            exception_list = []

            # use prediction from test data
            current_predicted_high_chance = test_predicted_chance_final 
            current_predicted_index = test_predicted_state_final 
            model_score = 0
            correct_predictions = 0

            for i in range(window_size, len(full_df)):
                run_count += 1
                X_train = full_df.iloc[i-window_size:i][['Returns', 'Range']].values
                current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
                
                try:
                    model.fit(X_train)
                    current_state = model.predict(current_features)[0]

                    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
                    low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
                    bull_regimes = []
                    for regime in positive_return_regimes:
                        if regime in low_volatility_regimes:
                            bull_regimes.append(regime)

                    signal = 1 if current_state in bull_regimes else 0
                    signals.append(signal)
                    states.append(current_state)

                    # "score" model based on whether or not prediction is correct
                    # using percentage like a 0-100 confidence scale
                    if current_state == current_predicted_index:
                        model_score += current_predicted_high_chance
                        correct_predictions += 1
                    else:
                        model_score -= current_predicted_high_chance
                    # set next values to "current"
                    current_transmat = model.transmat_[current_state]
                    current_predicted_high_chance = current_transmat.max()
                    current_predicted_index = np.where(current_transmat == current_predicted_high_chance)[0][0]

                except Exception as e:
                    # if model fails to converge, use signal from previous day

                    signals.append(signals[-1] if signals else 0)
                    states.append(states[-1] if states else 0)
                    exception_list.append(i)
                    continue

            win_rate = correct_predictions / run_count
            model_list.append(model)
            score_list.append(model_score)
            win_rate_list.append(win_rate)
            signals_and_states.append((signals, states))

            # add model number and continue loop
            model_number += 1

        end = time.time()

        high_index = np.argmax(score_list)
        high_model = model_list[high_index]
        high_signals = signals_and_states[high_index][0]
        high_states = signals_and_states[high_index][1]

        print(f"\nRun time: {end - start:.2f}s")

        print(f"\nModel count: {len(model_list)}")
        print(f"Winning model: {high_index}")
        print(f"High score: {score_list[high_index]:.2f}")
        print(f"High win rate: {win_rate_list[high_index]:.2%}")

        full_results = full_df.copy()
        new_results = full_results[window_size:]
        new_results['Signal'] = high_signals
        new_results['State'] = high_states

        return high_model, new_results
