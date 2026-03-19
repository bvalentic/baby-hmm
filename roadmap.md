# Roadmap

## I. Risk Management & Position Sizing Framework

### A. Account-Level Risk Controls

1. **Maximum Drawdown Limits**
   - Hard stop: Pause trading if account drops X% from peak
   - Soft warning: Alert at Y% drawdown
   - Daily loss limit: Stop trading for the day if down $Z
   - Monthly loss limit: Reduce position sizes after hitting threshold

2. **Exposure Limits**
   - Max gross exposure (% of account in positions)
   - Max net exposure (longs - shorts, if applicable)
   - Max notional value per $1 of capital
   - Leverage limits (if margin allowed)

3. **Concentration Limits**
   - Max position size as % of account
   - Max sector exposure (e.g., no more than 20% in tech)
   - Max correlation-based exposure (reduce positions in highly correlated assets)
   - Max position relative to average daily volume (ADV) - e.g., never own >1% of 20-day ADV

4. **Margin & Liquidity Management**
   - Maintain cash buffer for margin calls
   - Monitor overnight risk (if holding positions)
   - Gap risk assessment for hard-to-trade periods (earnings, news events)

### B. Per-Trade Risk Management

1. **Stop Loss Logic**
   - **Hard stops**: Fixed percentage or dollar loss (e.g., -2% from entry)
   - **Volatility-adjusted stops**: Based on ATR (e.g., 1.5× ATR)
   - **Time stops**: Exit if position hasn't moved after N periods
   - **Trailing stops**: Lock in profits as position moves favorably
   - **Technical stops**: Below recent swing low, below moving average, etc.
   - **Partial profit-taking**: Scale out at predetermined levels

2. **Risk Per Trade Calculation**
   - Fixed fractional: Risk 1% of account on each trade
   - Volatility-adjusted: Risk based on current market conditions
   - Kelly Criterion: Optimal f based on historical win rate and avg win/loss
   - Half-Kelly or fractional Kelly for more conservative approach
   - Dynamic sizing based on model confidence score

### C. Position Sizing Algorithms

1. **Core Sizing Methods**

   ```text
   Position Size = f(account_equity, risk_per_trade, stop_distance, asset_price)
   
   Example: If risking 1% of $100k account = $1,000 risk
   Stop distance = $2.50 per share
   Position size = $1,000 / $2.50 = 400 shares
   ```

2. **Advanced Sizing Factors**
   - **Volatility weighting**: Smaller positions in high-volatility environments

     ```text
     Size = Base_Size × (ATR_20 / Current_ATR)
     ```

   - **Correlation penalty**: Reduce size if highly correlated with existing positions

     ```text
     Size = Base_Size × (1 - avg_correlation_with_portfolio)
     ```

   - **Model confidence multiplier**: Scale based on prediction confidence (0.5x to 1.5x)
   - **Market regime filter**: Reduce all sizes during identified high-risk periods

3. **Portfolio-Level Optimization**
   - **Risk parity**: Equalize risk contribution across positions
   - **Minimum correlation method**: Position sizes inversely proportional to correlation
   - **Monte Carlo simulation**: Test size impact on drawdown and returns
   - **Scenario analysis**: How portfolio performs under various market conditions

### D. Dynamic Risk Adjustment

1. **Volatility Regimes**
   - CBOE VIX levels or custom volatility metric
   - Regime definitions: Low/Normal/High/Extreme
   - Position size multipliers per regime (e.g., 1.5x/1.0x/0.5x/0x)
   - Automatic deleveraging during volatility spikes

2. **Win/Loss Streak Adjustments**
   - After N consecutive losses, reduce size by X%
   - After large win, potentially reduce size (protect gains)
   - Reset to base size after drawdown recovery

3. **Time-Based Adjustments**
   - Reduce size before major economic releases
   - Scale down into market close (if not holding overnight)
   - Weekend/holiday position limits
   - Month-end/quarter-end effects

### E. Stress Testing & Scenario Analysis

1. **Historical Crash Scenarios**
   - 2008 Financial Crisis
   - 2020 COVID crash
   - 1987 Black Monday
   - 2000 Dot-com bubble burst
   - Flash crashes (2010, 2015)
   - Individual stock gaps (down 20%+ on earnings)

2. **Hypothetical Scenarios**
   - Liquidity dry-up (wider spreads, no buyers)
   - Multiple correlated positions moving against simultaneously
   - Broker/API failure during adverse move
   - Circuit breaker halts with open positions

3. **Metrics to Calculate**
   - Maximum expected drawdown (historical/VaR)
   - Time to recover from worst-case scenario
   - Margin call probability
   - Stress test P&L under each scenario

### F. Risk Monitoring & Reporting

1. **Real-Time Risk Dashboard**
   - Current drawdown from peak
   - Risk used (% of daily/monthly limits)
   - Current exposure by sector/asset
   - Largest position as % of account
   - Current volatility regime
   - Margin usage

2. **Pre-Trade Checks**
   Before each trade, verify:
   - Account drawdown < limit
   - Daily loss limit not hit
   - Position would not exceed concentration limits
   - Asset has sufficient liquidity (volume/spread)
   - Not in blackout period (earnings/news)

3. **Post-Trade Monitoring**
   - Recalculate portfolio risk after each fill
   - Alert if any limits breached
   - Update win/loss streaks
   - Log all risk decisions for audit trail

### G. Implementation Examples

1. **Python Pseudocode Structure**

```python
class RiskManager:
    def __init__(self, account_value, max_drawdown=0.15, max_position_pct=0.05):
        self.account_value = account_value
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        self.peak_value = account_value
        self.consecutive_losses = 0
        
    def check_pre_trade(self, symbol, quantity, price, existing_positions):
        """Run all pre-trade checks"""
        if self.current_drawdown() > self.max_drawdown:
            return False, "Max drawdown exceeded"
            
        new_exposure = quantity * price / self.account_value
        if new_exposure > self.max_position_pct:
            return False, "Position size limit exceeded"
            
        # Check correlation limits
        if self.would_exceed_correlation_limit(symbol, quantity, existing_positions):
            return False, "Correlation limit exceeded"
            
        return True, "Trade approved"
    
    def calculate_position_size(self, signal_strength, atr, stop_pct):
        """Dynamic position sizing"""
        base_risk = self.account_value * 0.01  # Risk 1% per trade
        
        # Adjust for volatility
        vol_multiplier = self.normal_atr / max(atr, 0.01)
        
        # Adjust for model confidence
        confidence_multiplier = signal_strength  # 0.5 to 1.5
        
        # Adjust for consecutive losses
        streak_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.1))
        
        risk_amount = base_risk * vol_multiplier * confidence_multiplier * streak_multiplier
        
        # Convert risk to shares
        stop_distance = price * stop_pct
        shares = risk_amount / stop_distance
        
        return min(shares, self.max_shares_by_volume(symbol))
```

## Data Quality & Management

- **Data validation/cleaning pipeline** (handling splits, dividends, corporate actions, bad ticks)
- **Backfill strategy** for missing data during market hours
- **Point-in-time data** handling for accurate backtesting (avoiding look-ahead bias)
- **Multiple data source fallback** if primary API fails

## System Architecture & Operations

- **Message queue/event bus** for decoupling components (data ingestion → signal generation → order execution)
- **Circuit breakers** at multiple levels (if too many errors, pause trading)
- **State persistence** so system can recover from crashes without losing context
- **Time synchronization** and handling of market hours, holidays, early closes
- **Latency monitoring** - track how long each pipeline stage takes

## Execution Logic

- **Order types and routing logic** (market, limit, stop-loss logic)
- **Partial fills handling** and order lifecycle management
- **Slippage modeling** in both backtesting and live trading
- **Position reconciliation** - compare your records with broker's
- **Emergency liquidation procedure** if kill switch triggered

## Monitoring & Operations

- **Real-time dashboard** showing P&L, open positions, model confidence, system health
- **Alerting system** (email/SMS) for critical events (kill switch triggered, API failures, margin calls)
- **Performance metrics tracking** (Sharpe ratio, max drawdown, win rate) updated continuously
- **Logging strategy** - structured logs for both debugging and audit trails

## Compliance & Legal

- **Trade journal** for audit trail (regulatory requirements)
- **Check against restricted securities** (if in US, pattern day trader rules if under $25k)
- **Tax lot tracking** for tax reporting

## Deployment & Infrastructure

- **Configuration management** (separate from code - feature flags, model parameters)
- **Secrets management** for API keys
- **Containerization/Docker** for reproducibility
- **Orchestration** (how services start, stop, and scale)
- **Disaster recovery plan** (backup servers, failover strategy)

## Model-Specific Additions

- **Feature drift detection** - monitor if input distributions change
- **Model versioning** and A/B testing framework for comparing versions
- **Retraining trigger logic** (not just time-based, but performance-based)
- **Explainability tools** for understanding why model made certain predictions

## Testing Expansion

- **Paper trading phase** before going live
- **Market replay testing** against historical data
- **Chaos engineering** - test how system handles API outages, bad data, etc.
- **Forward walk testing** (out-of-sample validation over rolling windows)

## I. Data Architecture Overview

### A. Data Pipeline Layers

```text
Raw Data Layer → Validation Layer → Cleaning Layer → Enrichment Layer → Storage Layer → API Layer
```

## II. Data Validation Pipeline

### 1. Real-Time Validation Checks

#### A. Tick-Level Validation (1-minute and lower)

```python
class TickValidator:
    def validate_tick(self, tick):
        checks = [
            self.check_timestamp_sequence(tick),
            self.check_price_reasonability(tick),
            self.check_volume_reasonability(tick),
            self.check_spread_reasonability(tick),
            self.check_exchange_operating_hours(tick)
        ]
        return all(checks)
    
    def check_price_reasonability(self, tick):
        """Prevent obvious bad ticks"""
        if tick.price <= 0:
            return False
        if tick.price > self.max_historical_price * 10:  # 10x historical high
            return False
        if abs(tick.price - self.prev_price) / self.prev_price > 0.5:  # 50% move
            return False, "Price jump too large"
        return True
    
    def check_spread_reasonability(self, tick):
        """Bid/ask spread validation"""
        if tick.ask <= tick.bid:
            return False, "Crossed market"
        if (tick.ask - tick.bid) / tick.mid > self.max_spread_pct:
            return False, "Spread too wide"
        return True
```

#### **B. Bar-Level Validation (1-minute to daily)**

- **OHLC consistency**: High >= Low, High >= Open/Close, Low <= Open/Close
- **Volume > 0** (except during halt periods)
- **Timestamp alignment**: Bar ends exactly on interval boundary
- **Price continuity**: No unexplained gaps without volume
- **Settlement price alignment** (for daily data)

### 2. Statistical Anomaly Detection

```python
class StatisticalValidator:
    def __init__(self, lookback_days=30):
        self.price_history = deque(maxlen=lookback_days)
        self.volume_history = deque(maxlen=lookback_days)
        
    def detect_anomalies(self, new_bar):
        anomalies = []
        
        # Z-score based detection
        price_zscore = (new_bar.close - self.mean_price) / self.std_price
        if abs(price_zscore) > 5:
            anomalies.append(f"Price {price_zscore:.2f} sigma from mean")
            
        # Volume spike detection
        if new_bar.volume > self.mean_volume + 5 * self.std_volume:
            anomalies.append(f"Volume spike: {new_bar.volume/self.mean_volume:.1f}x normal")
            
        # Price reversal patterns
        if self.detect_topple_reversal(new_bar):
            anomalies.append("Potential topple pattern - data error?")
            
        return anomalies
    
    def detect_topple_reversal(self, bar):
        """Detect common data errors where high/low are reversed"""
        if bar.high < bar.low:
            return True
        if bar.open > bar.high or bar.open < bar.low:
            return True
        return False
```

### 3. Cross-Source Validation

- Compare across multiple data providers
- Flag discrepancies > threshold (e.g., 0.1% price difference)
- Use consensus price when sources disagree
- Log all discrepancies for manual review

## III. Data Cleaning Pipeline

### 1. Handling Corporate Actions

#### **A. Action Detection & Adjustment**

```python
class CorporateActionProcessor:
    def __init__(self):
        self.action_types = {
            'split': self.adjust_for_split,
            'reverse_split': self.adjust_for_reverse_split,
            'dividend': self.adjust_for_dividend,
            'spin_off': self.adjust_for_spinoff,
            'merger': self.adjust_for_merger,
            'symbol_change': self.update_symbol
        }
    
    def adjust_historical_data(self, symbol, action_date, action_type, ratio):
        """Adjust all historical data before action_date"""
        historical_data = self.load_data(symbol, before=action_date)
        
        if action_type == 'split':
            # 2-for-1 split: divide price by 2, multiply volume by 2
            historical_data.price /= ratio
            historical_data.volume *= ratio
            
        elif action_type == 'dividend':
            # Subtract dividend from historical prices
            historical_data.price -= self.dividend_amount
            
        return self.save_adjusted_data(symbol, historical_data)
```

#### **B. Adjustment Methodologies**

- **Price adjustments**: Apply ratio to all pre-event prices
- **Volume adjustments**: Inverse ratio for splits (more shares traded)
- **Index membership changes**: Track and adjust for rebalancing
- **Currency adjustments**: Handle forex for international stocks

### 2. Bad Tick Handling

#### **A. Interpolation Strategies**

```python
class BadDataHandler:
    def handle_missing_tick(self, timestamp, symbol):
        strategies = [
            self.linear_interpolation,
            self.last_price_hold,
            self.sector_etf_correlation,
            self.option_implied_price
        ]
        
        for strategy in strategies:
            price = strategy(timestamp, symbol)
            if price:
                return price
        return None
    
    def linear_interpolation(self, ts, symbol):
        """Fill missing data with linear interpolation"""
        before = self.get_last_valid_tick(ts, symbol)
        after = self.get_next_valid_tick(ts, symbol)
        if before and after:
            ratio = (ts - before.ts) / (after.ts - before.ts)
            return before.price + ratio * (after.price - before.price)
        return None
```

#### **B. Outlier Treatment**

- **Winsorization**: Cap extreme values at X percentile
- **Removal**: Delete obviously bad ticks
- **Flagging**: Mark suspect data for model awareness
- **Replacement**: Use secondary source when primary fails

### 3. Gap Filling

#### **A. Non-Trading Periods**

- **Regular gaps**: Overnight, weekends, holidays
- **Trading halts**: News pending, volatility circuit breakers
- **Limit up/down**: When price hits exchange limits

#### **B. Gap Handling Logic**

```python
class GapHandler:
    def process_gap(self, from_tick, to_tick):
        gap_duration = to_tick.timestamp - from_tick.timestamp
        expected_duration = self.get_expected_interval(from_tick, to_tick)
        
        if gap_duration > expected_duration * 1.5:
            # Significant gap - check for news/corporate actions
            if self.news_during_gap(from_tick, to_tick):
                return self.handle_news_gap(from_tick, to_tick)
            else:
                return self.handle_silent_gap(from_tick, to_tick)
        else:
            # Normal gap - linear fill is fine
            return self.linear_fill_gap(from_tick, to_tick)
```

## IV. Backfill Strategy

### 1. Initial Historical Backfill

#### **A. Prioritization Strategy**

```python
class BackfillOrchestrator:
    def prioritize_backfill(self, symbols):
        return sorted(symbols, key=lambda s: (
            -self.trading_volume(s),  # High volume first
            -self.model_reliance(s),   # Core model symbols first
            self.data_completeness(s), # Sparse data last
        ))
    
    def backfill_tiered(self):
        # Tier 1: Core universe (S&P 500) - full history
        self.backfill_symbols(self.sp500, years=10)
        
        # Tier 2: Secondary universe - shorter history
        self.backfill_symbols(self.secondary, years=5)
        
        # Tier 3: Opportunistic - only recent
        self.backfill_symbols(self.opportunistic, months=6)
```

#### **B. Multi-Source Aggregation**

- Primary source: Highest quality, most expensive
- Secondary source: Backup, may have gaps
- Free source: Last resort, extensive validation needed
- Vendor-specific: For corporate actions, dividends

### 2. Incremental Backfill (Catch-up)

```python
class IncrementalBackfiller:
    def catch_up_missing_data(self, symbol, latest_stored):
        """Get all data since last stored timestamp"""
        current_time = datetime.now()
        missing_intervals = []
        
        # Identify gaps
        for source in self.data_sources:
            source_data = source.get_data(symbol, latest_stored, current_time)
            missing_intervals.extend(self.find_gaps(source_data))
        
        # Prioritize gaps
        for gap in sorted(missing_intervals, key=lambda g: g.duration):
            self.fill_gap(symbol, gap)
    
    def fill_gap(self, symbol, gap):
        """Fill specific gap with best available source"""
        for source in self.priority_sources:
            data = source.get_data(symbol, gap.start, gap.end)
            if data and self.validate_data(data):
                self.store_data(symbol, data)
                return True
        return False
```

### 3. Real-Time Backfill (Live Trading)

- **Continuous catch-up**: Run every N minutes
- **Priority queue**: Missing ticks > missing minutes > missing days
- **Rate limiting**: Respect API rate limits during catch-up
- **Partial fills**: Store what you can, retry later

## V. Point-in-Time Data Handling

### 1. Data Versioning

```python
class PointInTimeDatabase:
    def __init__(self):
        self.data_versions = {}  # as_of_date -> dataset
        
    def store_snapshot(self, data, as_of_date):
        """Store data exactly as it existed at as_of_date"""
        # Apply corporate actions known ON or BEFORE as_of_date
        adjusted_data = self.apply_known_actions(data, as_of_date)
        
        # Store frozen version
        self.data_versions[as_of_date] = frozencopy(adjusted_data)
        
    def get_data_as_of(self, symbol, date, as_of_date):
        """Get data for symbol on date, as it would have looked on as_of_date"""
        if as_of_date not in self.data_versions:
            raise ValueError(f"No data version for {as_of_date}")
            
        return self.data_versions[as_of_date].get(symbol, date)
```

### 2. Survivorship Bias Prevention

#### **A. Dead Symbol Tracking**

```python
class SurvivorshipBiasPreventer:
    def __init__(self):
        self.live_symbols = set()
        self.dead_symbols = {}  # symbol -> delisting_date
        
    def maintain_universe(self, as_of_date):
        """Get the exact universe as it existed on as_of_date"""
        universe = []
        
        # Symbols that were live on this date
        for symbol in self.live_symbols:
            if self.was_live_on(symbol, as_of_date):
                universe.append(symbol)
        
        # Symbols that died after this date
        for symbol, delist_date in self.dead_symbols.items():
            if delist_date > as_of_date:
                universe.append(symbol)
        
        return universe
```

#### **B. Backtesting Requirements**

- Never use future data in training
- Use only symbols that existed at training time
- Account for delisting returns (often -100%)
- Include IPO dates correctly

### 3. Look-Ahead Prevention

```python
class LookAheadPreventer:
    def prepare_features(self, raw_data, timestamp):
        """Ensure no future data leaks into features"""
        features = {}
        
        # Technical indicators using only past data
        features['sma_20'] = self.calculate_sma(raw_data[:timestamp], 20)
        features['rsi'] = self.calculate_rsi(raw_data[:timestamp], 14)
        
        # Fundamental data - use only what was known
        if hasattr(timestamp, 'quarter_end'):
            # Only use earnings released BEFORE this timestamp
            earnings = self.get_earnings_before(symbol, timestamp)
            features['pe_ratio'] = self.calculate_pe(earnings, timestamp)
        
        return features
```

## VI. Multiple Data Source Fallback

### 1. Source Prioritization

```python
class DataSourceManager:
    def __init__(self):
        self.sources = {
            'primary': BloombergSource(priority=1),
            'secondary': IQFeedSource(priority=2),
            'free': YahooFinanceSource(priority=3),
            'backup': PolygonSource(priority=4)
        }
        
        self.source_status = {s: True for s in self.sources}
        
    def get_data(self, symbol, start, end, max_retries=3):
        for source_name, source in sorted(self.sources.items(), 
                                         key=lambda x: x[1].priority):
            if not self.source_status[source_name]:
                continue
                
            for attempt in range(max_retries):
                try:
                    data = source.get_data(symbol, start, end)
                    if self.validate_data(data):
                        return data
                except Exception as e:
                    self.log_failure(source_name, e)
                    
            # Mark source as down if all retries failed
            self.source_status[source_name] = False
            
        raise DataUnavailableError(f"No data for {symbol}")
```

### 2. Source Synchronization

```python
class SourceSynchronizer:
    def align_sources(self, symbol, date):
        """Get aligned data from multiple sources"""
        data_points = []
        
        for source in self.active_sources:
            data = source.get_tick(symbol, date)
            if data:
                data_points.append(data)
        
        if not data_points:
            return None
            
        # Use median if multiple sources agree
        if len(data_points) >= 3:
            prices = [d.price for d in data_points]
            return {
                'price': median(prices),
                'confidence': 'high',
                'sources': len(data_points),
                'std_dev': stdev(prices)
            }
        
        # Use single source with warning
        return {
            'price': data_points[0].price,
            'confidence': 'low',
            'sources': 1
        }
```

### 3. Source Health Monitoring

```python
class SourceHealthMonitor:
    def __init__(self):
        self.source_metrics = {}
        
    def track_source_health(self, source_name):
        metrics = {
            'latency': self.measure_latency(source_name),
            'success_rate': self.calculate_success_rate(source_name),
            'data_quality': self.assess_data_quality(source_name),
            'uptime': self.track_uptime(source_name)
        }
        
        self.source_metrics[source_name] = metrics
        
        # Auto-disable poor performing sources
        if metrics['success_rate'] < 0.95:
            self.disable_source(source_name, "Low success rate")
        
        return metrics
    
    def switch_strategy(self, failing_source):
        """Determine fallback strategy based on failure type"""
        if failing_source == 'primary':
            return 'immediate_failover'  # Switch immediately
        elif failing_source == 'secondary':
            return 'retry_then_failover'  # Retry 3x then switch
        else:
            return 'degrade_gracefully'  # Accept lower quality
```

### 4. Data Reconciliation

```python
class DataReconciler:
    def reconcile_sources(self, symbol, date_range):
        """Compare data across sources and flag discrepancies"""
        discrepancies = []
        
        for date in date_range:
            source_data = {}
            for name, source in self.sources.items():
                data = source.get_bar(symbol, date)
                if data:
                    source_data[name] = data
            
            if len(source_data) >= 2:
                # Check for significant differences
                max_price = max(d.close for d in source_data.values())
                min_price = min(d.close for d in source_data.values())
                
                if (max_price - min_price) / min_price > 0.001:  # 0.1% difference
                    discrepancies.append({
                        'date': date,
                        'symbol': symbol,
                        'prices': {n: d.close for n, d in source_data.items()},
                        'max_diff_pct': (max_price - min_price) / min_price
                    })
        
        return discrepancies
```

## VII. Storage & Retrieval Optimization

### 1. Database Schema Design

```sql
-- Raw ticks table (partitioned by date)
CREATE TABLE ticks_raw (
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    price DECIMAL(10,4),
    volume BIGINT,
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    source VARCHAR(20),
    quality_score FLOAT,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- Adjusted bars table (point-in-time aware)
CREATE TABLE bars_adjusted (
    symbol VARCHAR(10),
    date DATE,
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume BIGINT,
    as_of_date DATE,  -- Point-in-time version
    adjusted_for VARCHAR(255),  -- Corporate actions applied
    PRIMARY KEY (symbol, date, as_of_date)
);

-- Data quality log
CREATE TABLE data_quality_log (
    timestamp TIMESTAMP,
    symbol VARCHAR(10),
    issue_type VARCHAR(50),
    severity VARCHAR(20),
    description TEXT,
    action_taken VARCHAR(100)
);
```

### 2. Caching Strategy

```python
class DataCache:
    def __init__(self):
        self.l1_cache = LRUCache(maxsize=1000)  # Hot data
        self.l2_cache = RedisCache(ttl=3600)    # Warm data
        self.cold_storage = Database()           # Cold data
        
    def get_data(self, symbol, date_range):
        # Check L1 first
        key = f"{symbol}:{date_range}"
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2
        data = self.l2_cache.get(key)
        if data:
            self.l1_cache[key] = data
            return data
        
        # Get from DB
        data = self.cold_storage.query(symbol, date_range)
        self.l2_cache.set(key, data)
        self.l1_cache[key] = data
        return data
```
