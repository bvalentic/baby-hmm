# Baby-HMM
AKA Baby's First HMM
A growing set of hidden Markov models

## Create Virtual Env and Install Dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
## Run Program:

```bash
python hmm.py # or whichever py file you want to run
```
### Summary of Key Concepts:

| Concept | Explanation |
| --- | --- |
| **Observation** | The data you see (Returns, Volatility). |
| **Hidden State** | The invisible regime (Bear, Bull, Choppy) causing the observations. |
| **Transition Matrix** | The probability of moving from one state to another (e.g., How likely is it to stay in a Bull market tomorrow if we are in one today?). |
| **Stationarity** | HMMs fail if you feed them raw prices (100, 101, 102). You must feed them changes (returns) or ratios (volatility). |


