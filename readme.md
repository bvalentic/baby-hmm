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

### Phases

The program is broken down into phases:

1. Fetch data
2. Build and train model
3. Plot and show
4. Test against training data
5. Analyze training results
6. Test on new data
7. Use rolling window to train up to present day
8. Return recent states and predict tomorrow's state

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

## Roadmap to Prod

### Risk Management & Position Sizing

- **Position sizing algorithm** (Kelly Criterion, fixed fraction, or volatility-adjusted)
- **Portfolio-level risk limits** (max drawdown, max sector exposure, max correlation between positions)
- **Risk per trade** limits (e.g., never risk more than 1-2% of account)
- **Stress testing** - how does the system perform during market crashes or unusual volatility?

### Data Quality & Management

- **Data validation/cleaning pipeline** (handling splits, dividends, corporate actions, bad ticks)
- **Backfill strategy** for missing data during market hours
- **Point-in-time data** handling for accurate backtesting (avoiding look-ahead bias)
- **Multiple data source fallback** if primary API fails

### System Architecture & Operations

- **Message queue/event bus** for decoupling components (data ingestion → signal generation → order execution)
- **Circuit breakers** at multiple levels (if too many errors, pause trading)
- **State persistence** so system can recover from crashes without losing context
- **Time synchronization** and handling of market hours, holidays, early closes
- **Latency monitoring** - track how long each pipeline stage takes

### Execution Logic

- **Order types and routing logic** (market, limit, stop-loss logic)
- **Partial fills handling** and order lifecycle management
- **Slippage modeling** in both backtesting and live trading
- **Position reconciliation** - compare your records with broker's
- **Emergency liquidation procedure** if kill switch triggered

### Monitoring & Operations

- **Real-time dashboard** showing P&L, open positions, model confidence, system health
- **Alerting system** (email/SMS) for critical events (kill switch triggered, API failures, margin calls)
- **Performance metrics tracking** (Sharpe ratio, max drawdown, win rate) updated continuously
- **Logging strategy** - structured logs for both debugging and audit trails

### Compliance & Legal

- **Trade journal** for audit trail (regulatory requirements)
- **Check against restricted securities** (if in US, pattern day trader rules if under $25k)
- **Tax lot tracking** for tax reporting

### Deployment & Infrastructure

- **Configuration management** (separate from code - feature flags, model parameters)
- **Secrets management** for API keys
- **Containerization/Docker** for reproducibility
- **Orchestration** (how services start, stop, and scale)
- **Disaster recovery plan** (backup servers, failover strategy)

### Model-Specific Additions

- **Feature drift detection** - monitor if input distributions change
- **Model versioning** and A/B testing framework for comparing versions
- **Retraining trigger logic** (not just time-based, but performance-based)
- **Explainability tools** for understanding why model made certain predictions

### Testing Expansion

- **Paper trading phase** before going live
- **Market replay testing** against historical data
- **Chaos engineering** - test how system handles API outages, bad data, etc.
- **Forward walk testing** (out-of-sample validation over rolling windows)

## More Roadmaps

See [7 Steps to Systematic Investing](https://www.osqf.org/archive/2025/JeffRyan-JustinShea-seminar.pdf) by Jeff Ryan & Justin M Shea for a similar list
