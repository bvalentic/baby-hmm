# Baby-HMM

AKA Baby's First HMM
A growing set of hidden Markov models

## Getting Started

### Create Virtual Env and Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Run Program

```bash
python hmm.py # or whichever py file you want to run
```

## Concepts

### Summary of Key Concepts

| Concept | Explanation |
| --- | --- |
| **Observation** | The data you see (Returns, Volatility). |
| **Hidden State** | The invisible regime (Bear, Bull, Choppy) causing the observations. |
| **Transition Matrix** | The probability of moving from one state to another (e.g., How likely is it to stay in a Bull market tomorrow if we are in one today?). |
| **Stationarity** | HMMs fail if you feed them raw prices (100, 101, 102). You must feed them changes (returns) or ratios (volatility). |

### Market Regimes

Some examples of regimes found by model:

| n_components | Likely Regimes |
| --- | --- |
| 1 | Market (no variance) |
| 2 | Bull, Bear |
| 3 | Bull, Bear, Crash |
| 4 | Bull, Bear, Crash, Recovery/Spike |
| --- | --- |

| Regime | Description | Mean Returns | Mean Range |
| --- | --- | --- | --- |
| Bull | High growth, low-med volatility | Low positive | Low volume |
| Bear | Low or no growth, low-med volatility | Low negative | High volume |
| Crash | Sharp decline, high volatility | High negative | High volume |
| Recovery | High growth, high volatility | High positive | Low volume |
| --- | --- | --- | --- |

## Phases

The program is broken down into phases:

1. Fetch data
2. Build and train model
3. Plot and show
4. Test against training data
5. Analyze training results
6. Test on new data
7. Use rolling window to train up to present day
8. Return recent states and predict tomorrow's state
