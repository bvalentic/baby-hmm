# Baby-HMM
AKA Baby's First HMM
A growing set of hidden Markov models

## Getting Started

### Create Virtual Env and Install Dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
### Run Program:

```bash
python hmm.py # or whichever py file you want to run
```

## Concepts

### Summary of Key Concepts:

| Concept | Explanation |
| --- | --- |
| **Observation** | The data you see (Returns, Volatility). |
| **Hidden State** | The invisible regime (Bear, Bull, Choppy) causing the observations. |
| **Transition Matrix** | The probability of moving from one state to another (e.g., How likely is it to stay in a Bull market tomorrow if we are in one today?). |
| **Stationarity** | HMMs fail if you feed them raw prices (100, 101, 102). You must feed them changes (returns) or ratios (volatility). |

### Phases
The program is broken down into phases:

1. Fetch data
2. Build and train model
3. Plot and show
4. Analyze training results
5. Test on new data

### Market Regimes
Depending on the number of regimes you search for, they could be broken down into multiple different categories.
Here's what I've broken the regimes down into, based purely on what I saw on the plots.

| n_components | Likely Regimes |
| --- | --- |
| 1 | Market (no variance) |
| 2 | Bull, Bear |
| 3 | Bull, Bear, Crash |
| 4 | Bull, Bear, Crash, Recovery |
| --- | --- |

| Regime | Characteristics |
| --- | --- |
| Bull | High growth, low-med volatility |
| Bear | Low or no growth, low-med volatility |
| Crash | Sharp decline, high volatility |
| Recovery | High growth, high volatility |
| --- | --- |

