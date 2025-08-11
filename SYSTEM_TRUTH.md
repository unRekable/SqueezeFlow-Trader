# System Truth - What Actually Works

**Last Updated: 2025-08-11**

## 📊 DASHBOARD & VISUALIZATION STATUS

### ✅ What's Working
- Multi-page dashboard with navigation (Main, Portfolio, Exchange Analytics)
- TradingView lightweight-charts with all data visualization
- Exchange-colored volume bars (Binance yellow, Bybit orange, OKX green, Coinbase blue)
- Portfolio panel on right side with charts
- OI candlesticks visualization (OHLC format)
- Strategy scoring with confidence bands
- Chart synchronization (zoom/pan together)
- Timeframe switching (1s, 1m, 5m, 15m, 1h)
- Self-validation system with browser MCP support

### ⚠️ Recently Fixed
- 1s timeframe NaN values (fixed with pd.isna checks)
- Chart container height issues (explicit height calculations)
- Variable scoping in JavaScript (fixed repeated const declarations)
- Python f-string syntax in JavaScript templates

### 📁 Dashboard Files Structure
- **Main Visualizer:** `backtest/reporting/visualizer.py` (creates all 3 pages)
- **Enhanced Implementation:** `backtest/reporting/enhanced_visualizer.py` (main dashboard logic)
- **Validation:** `validate_dashboard.py` (unified validator with browser support)
- **Requirements:** `DASHBOARD_REQUIREMENTS.md` (complete specification)

### 📊 Generated Pages
1. **dashboard.html** - Main trading dashboard with all charts
2. **portfolio.html** - Portfolio analytics and trade history
3. **exchange_analytics.html** - Exchange statistics and volume analysis

## 🔍 VISUAL VALIDATION SYSTEM (CRITICAL FOR SELF-DEBUGGING)
**Claude can now SEE the dashboard output to debug issues:**
- **Automatic Validation**: Runs after every backtest with visualization
- **Screenshot Capability**: Can capture dashboard for visual inspection  
- **Data Analysis**: Validates HTML content, data structures, metrics
- **Issue Detection**: Identifies missing data, empty charts, trade problems
- **Self-Debugging**: Claude can use Read tool on screenshots to see actual output

**How to Use:**
```bash
# Automatic - runs with every backtest
python3 backtest/engine.py --symbol ETH ...

# Manual validation of any dashboard
python3 backtest/reporting/visual_validator.py

# View screenshot to see what user sees
Read tool: backtest/results/report_*/dashboard_screenshot_*.png
```

**This enables the optimization framework to improve the system by seeing results!**

## 🚨 DEVELOPMENT ENVIRONMENT CONFIGURATION
**THIS IS A LOCAL DEVELOPMENT ENVIRONMENT** that connects to remote data sources:
- **1-Second Market Data**: Collected on remote server, read from remote InfluxDB
- **Open Interest Data**: Collected on remote server, read from remote InfluxDB  
- **Local Services**: Strategy runner, backtesting, analysis tools
- **Remote Services**: Data collection (aggr-server), OI tracker, InfluxDB storage

### Data Flow Architecture
```
REMOTE SERVER (Production)          LOCAL DEVELOPMENT
┌─────────────────────┐            ┌──────────────────┐
│  aggr-server (1s)   │───────────▶│                  │
│  OI Tracker         │            │  InfluxDB Remote │
│  InfluxDB Storage   │◀───────────│  (Read Only)     │
└─────────────────────┘            │                  │
                                   │  Local Strategy   │
                                   │  Local Backtest   │
                                   │  Local Analysis   │
                                   └──────────────────┘
```

## Critical Understanding (DO NOT FORGET)

### 1. Data Location & Storage
- **Database:** `significant_trades` (on remote server)
- **Retention Policy:** `aggr_1s` (NOT default, NOT rp_1s)
- **Measurement:** `trades_1s`
- **Query:** `SELECT * FROM "aggr_1s"."trades_1s"`
- **Data Available:** 7 days of 1-second data (extended from 24h)
- **Aggregation:** FIRST/LAST/MAX/MIN for proper OHLCV candles
- **Remote Access:** Use `INFLUX_HOST=213.136.75.120` for all local operations (IP from `.env` file)
- **Market Aggregation:** ALL markets for a symbol are automatically aggregated (e.g., TON includes BINANCE:tonusdt, BYBIT:TONUSDT-SPOT, etc.)

### 1.5. CRITICAL: Data Availability Reality (AS OF 2025-08-10)
**⚠️ ALWAYS CHECK DATA EXISTENCE BEFORE RUNNING BACKTESTS:**
- **ETH data starts:** 2025-08-10 09:54 UTC (NOT midnight, NOT Aug 9)
- **ETH data ends:** Current time minus ~15 minutes (real-time collection)
- **Other symbols:** May have different start times - ALWAYS VERIFY
- **Markets use tags:** Data stored as `market='BINANCE:ethusdt'` not `symbol='ETH'`
- **Backtest engine:** Aggregates markets into symbols internally

**🚨 BEFORE ANY BACKTEST, ALWAYS RUN:**
```python
# Check what data actually exists
from influxdb import InfluxDBClient
client = InfluxDBClient(host='213.136.75.120', port=8086, database='significant_trades')

# Check ETH data range
query = '''
SELECT close FROM "aggr_1s"."trades_1s"
WHERE market = 'BINANCE:ethusdt'
ORDER BY time ASC LIMIT 1
'''
result = client.query(query)
for point in result.get_points():
    print(f"First data: {point['time']}")

# Check last data point
query_last = '''
SELECT close FROM "aggr_1s"."trades_1s"
WHERE market = 'BINANCE:ethusdt'
ORDER BY time DESC LIMIT 1
'''
result_last = client.query(query_last)
for point in result_last.get_points():
    print(f"Last data: {point['time']}")
```

**❌ WRONG: Assuming data exists for dates**
```bash
# WRONG - assumes Aug 9 data exists
python3 backtest/engine.py --start-date 2025-08-09 --end-date 2025-08-10
```

**✅ RIGHT: Use actual data range**
```bash
# RIGHT - use dates where data actually exists
python3 backtest/engine.py --start-date 2025-08-10 --end-date 2025-08-10
```

### 2. 1-Second Data Purpose & MANDATORY Usage
- **1s is for EXECUTION LATENCY, not analysis timeframe**
- **ALWAYS use `--timeframe 1s` for ALL crypto pairs (BTC, ETH, AVAX, SOL, etc.)**
- We collect every 1 second for fast execution
- We ANALYZE on 1m, 5m, 15m, 30m, 1h, 4h built FROM 1s data
- This gives us 1-2 second signal latency vs 60+ seconds
- Strategy has access to ALL historical data (no artificial windowing)
- **CRITICAL**: Never use 1m, 5m, etc. as base timeframe - ALWAYS 1s

### 3. Strategy Reality
- **5 Phases:** Context → Divergence → Reset → Scoring → Exit
- **CRITICAL:** Phase 2 (Divergence) MUST detect divergence for any trade
- **Phase 3 (Reset) IS the liquidity provision moment** - not separate
- **NO fixed thresholds** - Dynamic adaptation only (EXCEPT: see bug below)
- **Score 4.0+ required** to enter trades (but ONLY if divergence exists)
- **Trade frequency:** Unlimited - trades when conditions are met

### 3.5. CRITICAL BUG: Hardcoded Divergence Threshold (Line 230 phase2_divergence.py)
**⚠️ MAJOR ISSUE DISCOVERED Aug 10, 2025:**
- **Bug:** `min_change_threshold = 1e6` (1 million volume) hardcoded in phase2_divergence.py
- **Impact:** Low-volume symbols (TON, AVAX, SOL, etc.) can NEVER trade
- **Evidence:** 
  - ETH max CVD change: 106M (works fine)
  - TON max CVD change: 202K (blocked by 1M threshold)
- **Result:** TON backtest shows 0 trades because divergence is never detected
- **File:** `strategies/squeezeflow/components/phase2_divergence.py` line 230
- **Fix Needed:** Dynamic threshold based on symbol's typical volume or percentile-based approach

### 4. CVD (Cumulative Volume Delta)
- **Formula:** `(buy_volume - sell_volume).cumsum()`
- **TRUE Divergence:** When spot CVD and futures CVD move OPPOSITE directions
  - Long Setup: Spot CVD UP + Futures CVD DOWN
  - Short Setup: Spot CVD DOWN + Futures CVD UP
- **NOT Divergence:** Both moving same direction (even if one moves more)
- **Key:** This is about spot vs futures DISAGREEMENT, not magnitude differences
- **Data Source:** Remote InfluxDB server (not local collection)

### 5. Architecture That Works
```
REMOTE SERVER:
aggr-server (Node.js) 
    ↓ [1s data from 80+ markets]
InfluxDB (aggr_1s.trades_1s)
    ↓
OI Tracker (Python)
    ↓ [OI data from exchanges]
InfluxDB (open_interest)

LOCAL DEVELOPMENT:
DataPipeline → Remote InfluxDB
    ↓
SqueezeFlowStrategy (5 phases)
    ↓
Backtest Engine / Live Trading
```

### 6. Fixed Issues (2025-08-10)
- **Timezone sync:** All services use UTC via docker-compose.timezone.yml
- **InfluxDB connection:** Uses INFLUX_HOST env var (points to remote server)
- **OHLCV aggregation:** Fixed to use FIRST/LAST/MAX/MIN (not mean)
- **Data access:** Full historical data available at each point (no windowing)
- **Data queries:** Always use `aggr_1s` retention policy
- **Parallel processing:** Working with 4 workers
- **Backtest speed:** ~4 seconds for 24 hours of data
- **OI Integration:** Now reads from remote InfluxDB (not direct API calls)

## Current Configuration

### Environment Variables for Local Development
```yaml
# Connect to remote InfluxDB server
INFLUX_HOST: 213.136.75.120  # Remote server IP (from .env file)
INFLUX_PORT: 8086
INFLUX_DATABASE: significant_trades

# Local Redis for caching
REDIS_HOST: localhost
REDIS_PORT: 6379

# Strategy configuration
SQUEEZEFLOW_RUN_INTERVAL: 60  # Strategy runs every 60s
SQUEEZEFLOW_MAX_SYMBOLS: 3
TZ: UTC  # All services use UTC
```

### Service Status
- **Remote Server:**
  - aggr-server: Collecting 1s data from 80+ markets
  - OI Tracker: Collecting OI data from 4 exchanges
  - InfluxDB: Storing in aggr_1s retention policy (7 days)
  
- **Local Development:**
  - Strategy Runner: Processing BTC/ETH with 4h lookback
  - Backtest Engine: Using remote data for analysis
  - FreqTrade: Connected for live trading (optional)
  - Redis: Local caching and pub/sub

## Open Interest (OI) Data Structure

### OI Collection & Aggregation (Remote Server)
- **Exchanges:** 4 major exchanges (BINANCE_FUTURES, BYBIT, OKX, DERIBIT)
- **Base Symbols:** Aggregated by base asset (BTCUSDT+BTCUSDC → BTC)
- **Storage:** 3 types of records per symbol in remote InfluxDB

### OI Query Patterns (From Local to Remote)
```bash
# Individual exchange OI
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='BINANCE_FUTURES'

# Top 3 futures combined (BINANCE + BYBIT + OKX)
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='FUTURES_AGG'

# All exchanges combined (includes DERIBIT options)
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='TOTAL_AGG'

# Compare all OI types for BTC
SELECT exchange, open_interest_usd FROM open_interest 
WHERE symbol='BTC' AND time > now() - 1h ORDER BY time DESC
```

### OI Data Tags
- **aggregate_type:** 'individual', 'futures_aggregate', 'total_aggregate'
- **exchange:** Exchange name or 'FUTURES_AGG'/'TOTAL_AGG'
- **symbol:** Base symbol (BTC, ETH, etc.)

### Using OI Data in Strategies
```python
# Use InfluxDB-based OI tracker (reads from remote)
from strategies.squeezeflow.components.oi_tracker_influx import OITrackerInflux

# Initialize with remote InfluxDB connection
tracker = OITrackerInflux(rise_threshold=5.0)

# Get OI metrics from remote server
oi_data = tracker.get_oi_change_sync('BTC')

# Data includes:
# - current_oi: Current OI in USD
# - oi_change_pct: Percentage change
# - is_rising: Boolean for trend
# - exceeds_threshold: Boolean for signal
# - data_source: 'influxdb' (remote)
```

## Commands That Work

### Data Availability Checks (Remote)
```bash
# Check remote 1s data availability
curl http://213.136.75.120:8086/ping

# Check data via local client to remote
INFLUX_HOST=213.136.75.120 python -c "
from data.loaders.influx_client import OptimizedInfluxClient
client = OptimizedInfluxClient()
print(client.client.query('SELECT COUNT(*) FROM aggr_1s.trades_1s WHERE time > now() - 24h'))
"

# Check OI data availability from remote
INFLUX_HOST=213.136.75.120 python test_oi_remote.py
```

### Run Backtest (Using Remote Data)
```bash
# ⚠️ CRITICAL: ALWAYS CHECK DATA EXISTENCE FIRST (see section 1.5 above)

# WRONG - DO NOT BLINDLY USE DATES:
# python3 backtest/engine.py --symbol BTC --start-date 2025-08-09 --end-date 2025-08-09

# RIGHT - Check data first, then use actual dates:
# 1. First check what data exists (see section 1.5)
# 2. Then run backtest with VERIFIED dates:
export INFLUX_HOST=213.136.75.120
python3 backtest/engine.py --symbol ETH --start-date 2025-08-10 --end-date 2025-08-10 --timeframe 1s

# Or inline with CORRECT parameters:
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol ETH --start-date 2025-08-10 --end-date 2025-08-10 --timeframe 1s
```

### Local Development Commands
```bash
# Start local Redis for caching
redis-server

# Run strategy locally with remote data
INFLUX_HOST=213.136.75.120 python services/strategy_runner.py

# Monitor remote data flow
INFLUX_HOST=213.136.75.120 python scripts/monitor_performance.py

# Test OI integration
INFLUX_HOST=213.136.75.120 python -c "
from strategies.squeezeflow.components.oi_tracker_influx import OITrackerInflux
tracker = OITrackerInflux()
print(tracker.get_detailed_oi_analysis('BTC'))
"
```

## Recent Successful Tests (August 10, 2025)

### Backtest Results Examples

#### ETH Backtest (Aug 10, 2025)
```bash
# CORRECT Command (after verifying data exists):
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py \
  --symbol ETH \
  --start-date 2025-08-10 \
  --end-date 2025-08-10 \
  --timeframe 1s  # ALWAYS use 1s!

# Results:
- 80+ trades generated (high divergence detected)
- Aggregated 47 SPOT + 21 FUTURES markets
- Data range: 09:54-19:44 UTC
```

#### TON Backtest (Aug 10, 2025)
```bash
# Command:
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py \
  --symbol TON \
  --start-date 2025-08-10 \
  --end-date 2025-08-10 \
  --timeframe 1s

# Results:
- 0 trades (BLOCKED BY BUG - see section 3.5)
- Aggregated 16 SPOT + 9 FUTURES markets correctly ✅
- Data loaded: 16,201 points with CVD calculated ✅
- Divergences exist but MAX CVD change = 202K < 1M threshold ❌

# Analysis:
- TON max spot CVD change: 93K
- TON max futures CVD change: 202K
- Required threshold: 1,000,000 (hardcoded)
- Result: NO divergence can ever trigger for TON
```

**ROOT CAUSE:** Hardcoded 1M volume threshold in phase2_divergence.py line 230 blocks all low-volume symbols from trading. ETH works (106M volume) but TON/AVAX/SOL cannot meet threshold.

## Key Learnings to Remember

1. **DON'T ASSUME DATA EXISTS** - ALWAYS check actual data availability first (see section 1.5)
2. **DON'T use localhost/Docker for InfluxDB** - Always use remote server (213.136.75.120)
3. **DON'T forget INFLUX_HOST** - Backtest won't work without it
4. **DON'T blindly use date ranges** - Data may not exist for those dates
5. **DON'T use timeframe other than 1s** - ALWAYS use `--timeframe 1s`
6. **DON'T look for server IP in docker-compose** - It's in `.env` file
7. **DON'T use mean() for OHLCV** - Use FIRST/LAST/MAX/MIN
8. **DON'T use short lookbacks** - Need 4h for multi-timeframe
9. **DON'T treat 1s as analysis timeframe** - It's collection frequency
10. **DON'T run data collection locally** - It runs on the server

## File Locations

### Strategy Components
- **Strategy:** `strategies/squeezeflow/strategy.py`
- **OI Tracker (Remote):** `strategies/squeezeflow/components/oi_tracker_influx.py`
- **OI Tracker (Old/API):** `strategies/squeezeflow/components/oi_tracker.py` (deprecated)

### Data Pipeline
- **Pipeline:** `data/pipeline.py`
- **InfluxDB Client:** `data/loaders/influx_client.py` (now with OI methods)
- **Market Discovery:** `data/loaders/market_discovery.py`

### Backtest & Analysis
- **Backtest Engine:** `backtest/engine.py`
- **OI Test Script:** `test_oi_remote.py`

### Configuration
- **Docker Compose:** `docker-compose.yml` (for local services)
- **Server IP:** Located in `.env` file as `SERVER_IP=213.136.75.120`
- **Environment Setup:** Export `INFLUX_HOST=213.136.75.120` before running

## Essential Claude Code Agents

We use 16 carefully selected agents (not 110!) for SqueezeFlow Trader:

### Priority 1: Core Trading (6 agents)
- `quant-analyst` - Strategy optimization, backtesting
- `python-pro` - Python optimization, async code
- `data-engineer` - Pipeline optimization, InfluxDB
- `performance-engineer` - Latency reduction, profiling
- `devops-engineer` - Docker, deployment
- `fintech-engineer` - Trading system architecture

### Priority 2: Quality & Analysis (5 agents)
- `risk-manager` - Position sizing, portfolio risk
- `qa-expert` - Testing, validation
- `database-optimizer` - InfluxDB query optimization
- `react-specialist` - Dashboard development
- `data-scientist` - Statistical analysis, ML

### Priority 3: Support (5 agents)
- `sre-engineer`, `code-reviewer`, `multi-agent-coordinator`, `debugger`, `typescript-pro`

**Agent Setup Script:** `scripts/setup_essential_agents.sh`

## Development Workflow

1. **Setup Environment:**
   ```bash
   export INFLUX_HOST=213.136.75.120  # From .env file
   export INFLUX_PORT=8086
   ```

2. **Test Data Connection:**
   ```bash
   python test_oi_remote.py
   ```

3. **Run Backtest:**
   ```bash
   python backtest/engine.py --symbol BTC --timeframe 1s
   ```

4. **Monitor Performance:**
   ```bash
   python scripts/monitor_performance.py
   ```

5. **Deploy Changes:**
   - Test locally with remote data
   - Validate with backtests
   - Deploy to server if needed

## Important Notes

- **All timestamps are in UTC** - No timezone conversions
- **Data is read-only from local** - Cannot write to remote InfluxDB
- **OI updates every 5 minutes** - Not real-time like 1s market data
- **Use caching wisely** - Remote queries have network latency
- **Monitor connection health** - Network issues affect everything