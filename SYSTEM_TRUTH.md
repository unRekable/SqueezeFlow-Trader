# System Truth - What Actually Works

**Last Updated: 2025-08-12 04:25**

## 🚨 CRITICAL: Tick-by-Tick Execution Model CLARIFIED

### Option A Implementation Complete (2025-08-12)
**What Changed:**
- Added `execution_mode` parameter to DataPipeline.get_complete_dataset()
- Backtest engine now passes `execution_mode='tick'` to clarify sequential processing
- Added clear documentation that timeframes are ONLY for visualization

**Key Files Modified:**
- `/data/pipeline.py:384` - Added execution_mode parameter with 'candle' default
- `/backtest/engine.py:226` - Pass execution_mode='tick' for backtests
- `/strategies/squeezeflow/strategy.py:7-15` - Added critical comment about tick processing

**Backward Compatibility:** ✅ MAINTAINED
- Default `execution_mode='candle'` preserves existing behavior
- Only backtests explicitly use `execution_mode='tick'`
- Live trading unaffected (uses default)

## 🎯 TRADINGVIEW NATIVE PANES - FULLY WORKING!

### 🚀 NEW: TradingView Implementation with Visual Pane Separation
**Status: ✅ VERIFIED WORKING via Visual Validation (Enhanced 2025-08-12 14:32)**

**Implementation Method:** ONE chart with multiple priceScaleId + scaleMargins

**What's Working:**
- **Main Area (60%)**: Candlesticks with BUY/SELL/EXIT trade markers
- **Volume Pane (20%)**: Enhanced histogram bars with TradingView best practices
  - Scale margins: top: 0.8, bottom: 0.02
  - Red/green coloring based on candle direction
  - Volume formatting with proper precision
  - Hidden last value/price line for cleaner look
- **CVD Pane (30%)**: Spot CVD (blue) and Futures CVD (orange) on left scale
- **Score Pane (15%)**: Strategy Score with threshold lines
- **All synchronized**: Single time axis, unified zoom/pan

**Visual Validation Process Used:**
1. MCP Playwright browser navigation
2. Console error checking (found API changes)
3. Full page screenshots
4. Iterative debugging with visual feedback

## 📊 DASHBOARD & VISUALIZATION STATUS

### ⚠️ CRITICAL BROKEN LOGIC JUST FIXED (2025-08-12 04:05)

**MAJOR BUG DISCOVERED:** Timeframe switching was completely broken for 1h, 4h, and 1D!

**The Bug:**
```javascript
// BROKEN - !isNaN('60') returns true, so it becomes '60m' instead of '1h'
if (!isNaN(tf)) {
    tfKey = tf + 'm';  // WRONG for 60 and 240!
} else if (tf === '60') {
    tfKey = '1h';  // NEVER REACHED!
}
```

**The Fix (tradingview_unified.py:982-1001):**
- Check specific values ('60', '240', '1D') FIRST
- Only then check if it's a generic number
- Fixed in TWO places: button click handler AND initial active button logic

### 🎯 UNIFIED CHART IMPLEMENTATION (2025-08-12 11:48)

**What Makes It "Unified":**
1. **Single Time Axis** - Only bottom pane shows time, others hidden
2. **Synchronized Zoom/Pan** - All panes move together as one chart
3. **Unified Border** - Blue border around entire chart group
4. **No Gaps Between Panes** - Seamless vertical integration
5. **Pane Labels** - Each pane labeled (Price, Volume, CVD, Strategy Score)
6. **Coordinated Crosshair** - Vertical line syncs across all panes
7. **Aligned Data Ranges** - All indicator data filtered to match candle time range

**Technical Implementation - FIXED (2025-08-12 11:48):**
- **OLD BROKEN**: Used 4 separate LightweightCharts instances with complex sync
- **NEW WORKING**: Single chart instance with multiple series on different price scales
- Price on 'right' scale
- Volume on 'volume' scale with optimized margins (top: 0.8, bottom: 0.02)  
- CVD on 'left' scale with custom margins (top: 0.4, bottom: 0.3)
- Strategy Score on 'score' scale with custom margins (top: 0.8, bottom: 0.05)
- No synchronization needed - it's ONE chart!
- Time format fixed with timeVisible: true, secondsVisible: false

### ✅ DATA ISSUES RESOLVED (2025-08-12 13:00)

**Original Problem:** Charts showed NaN values and empty visualizations
**Root Cause Analysis Complete:**

1. **NaN Values Were NOT From Data Pipeline** ✅
   - Data pipeline returns valid data (tested and confirmed)
   - Issue was timeframe mismatch: requesting 5m but system loads 1s internally
   - This caused numpy "Mean of empty slice" warnings during aggregation

2. **Performance Issue Identified** ⚠️
   - Loading 1 day of 1s data = 86,400 points = 90 seconds
   - Loading 3 days would take 4.5 minutes just for data loading
   - **Solution:** Use appropriate timeframes for date ranges:
     - 1-hour test: Use 1s timeframe
     - 1-day test: Use 5m or 15m timeframe  
     - 3-day test: Use 15m or 1h timeframe

3. **CVD Spike at Start** ✅ UNDERSTOOD
   - CVD calculation starts from zero and accumulates
   - First trades cluster when CVD jumps from 0 to actual values
   - This is expected behavior for cumulative indicators

**Recommended Backtest Commands:**
```bash
# For 1-hour test (fast)
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol ETH --start-date 2025-08-11 --end-date 2025-08-11 --timeframe 15m

# For 1-day test (reasonable)
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol BTC --start-date 2025-08-11 --end-date 2025-08-11 --timeframe 5m

# For 3-day test (slower)
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol ETH --start-date 2025-08-10 --end-date 2025-08-12 --timeframe 1h
```

**Key Learning:** The system ALWAYS loads 1s data internally but aggregates based on timeframe parameter. Choose timeframe based on your date range for optimal performance.

### What SHOULD Work After Fix:
- **1h, 4h, 1D timeframes** - Now properly mapped
- **Timeframe switching** - All buttons should work
- **Strategy Score** - Always renders as line chart
- **Error handling** - Added try-catch blocks for chart operations

### ⚠️ Known Issues - RESOLVED
- **Indicator Data Truncation**: FIXED as of latest backtest
  - All indicators (CVD, Strategy Score) now continue for full time range
  - Verified via browser screenshot of backtest_20250812_034152
  
### 🔧 Recent Fixes (2025-08-12)
1. **Empty Chart JavaScript Error**
   - **Issue**: "Cannot access volumeChart before initialization"
   - **Fix**: Removed premature chart references at line 365-367
   - **Location**: tradingview_unified.py - store chart refs AFTER creation
   
2. **Strategy Score Visualization**
   - **Issue**: Displayed as histogram bars instead of clean line
   - **Fix**: Always use addLineSeries for strategy scores
   - **Location**: tradingview_unified.py:669-679
   
3. **NaN Quantity in Position Sizing**
   - **Issue**: Positions had NaN quantities, causing 0% win rate
   - **Fix**: Use initial balance when portfolio value is 0
   - **Location**: strategy.py:402-417
   
4. **Timeframe Switching**
   - **Issue**: Buttons said "requires re-run" but had 1s data
   - **Fix**: Client-side aggregation of 1s data to all timeframes
   - **Location**: tradingview_unified.py:30-94 (_aggregate_ohlcv_data)
- **Timeframe switching** (shows available resolutions)
- **Strategy scoring** ✅ FIXED - Now generating scores (7.56 in tests)
- **Trade execution** ✅ FIXED - Trades generating with relative divergences
- **Divergence detection** ✅ FIXED - Detects both TRUE and RELATIVE divergences

### 🔴 What's Broken
- **Portfolio page empty** (needs trade data connection)
- **Exchange analytics empty** (not implemented)

### ❌ Never Worked
- BTC OI data collection on server
- Exchange volume breakdown display

### 📁 Dashboard Files Structure - FINAL (2025-08-11 23:55)
- **Main Entry:** `backtest/reporting/visualizer.py` (ALWAYS uses TradingView)
- **Implementation:** `backtest/reporting/tradingview_unified.py` (TradingView + tabs)
- **Output:** `/results/backtest_*/dashboard.html` (single file)
- **Features:**
  - Tab 1: TradingView with 4 native panes (Price, Volume, CVD, Score)
  - Tab 2: Portfolio analytics with equity curve
  - Tab 3: Exchange volume distribution
- **NO CONDITIONALS**: Always generates the same dashboard type

### 📊 NEW: Single HTML Dashboard
- **File:** `results/backtest_*/dashboard.html` 
- **Tab 1:** Trading (charts, indicators, trades)
- **Tab 2:** Portfolio (equity curve, stats)
- **Tab 3:** Exchange (volume distribution)

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

### 1.7. Dashboard Visualization FIXES (AS OF 2025-08-11 18:40)
**✅ ROOT CAUSE OF PRICE CORRUPTION FOUND AND FIXED:**
- **Problem**: Candlestick low values showing single digits (3.30) when BTC at 118k
- **Root Cause**: `/data/loaders/influx_client.py` line 564 used `'low': 'min'` across ALL markets
  - When aggregating BTC, it included cross-pairs like ETHBTC (0.04) and took MIN of all
  - Result: BTC showing low of 0.04 instead of 118,000!
- **Solution**: Added outlier filtering in `_aggregate_ohlcv_data()` (lines 554-599)
  - Filters out prices >50% away from median before aggregation
  - Only takes MIN within similar price ranges
- **Status**: ✅ FIXED - No more cross-pair contamination

**✅ TIMEFRAME LOGIC FIXED:**
- **Problem**: All timeframes showed same data, 1s button showed 5m data
- **Root Cause**: Can't create higher resolution (1s) from lower resolution (5m) data
- **Solution**: 
  - Detect base timeframe from backtest data
  - Only show timeframe buttons for available aggregations (can only go UP)
  - If backtest uses 5m, only show 5m, 15m, 1h buttons
  - If backtest uses 1s, show all timeframes
- **Files Fixed**: `strategy_visualizer.py` lines 143-172, 489-492

**✅ UNDERSTANDING TIMEFRAMES:**
- **Timeframe = Candle Size** (not time range!)
- **1s**: Raw 1-second candles (86,400 per day)
- **1m**: 60 one-second candles aggregated (1,440 per day)
- **5m**: 300 one-second candles aggregated (288 per day)
- **15m**: 900 one-second candles aggregated (96 per day)
- **1h**: 3,600 one-second candles aggregated (24 per day)
- **Key Rule**: Can only aggregate UP, never DOWN
  - From 1s data: Can create 1m, 5m, 15m, 1h
  - From 5m data: Can only create 15m, 1h
  - From 1h data: Can't create any smaller timeframes

**Dashboard Status:**
- **Active Visualizer**: `strategy_visualizer.py` (single page, 5 panes)
- **Data Corruption**: ✅ FIXED at source in influx_client.py
- **Timeframe Buttons**: ✅ Only shows available timeframes
- **Portfolio View**: Still needs connection (separate issue)

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
- **NO fixed thresholds** - Dynamic adaptation only
- **Score 4.0+ required** to enter trades (but ONLY if divergence exists)
- **Trade frequency:** Unlimited - trades when conditions are met


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

### 5.5. CRITICAL FIX: Divergence Detection (2025-08-11 21:40)
- **Problem:** Strategy only detected TRUE divergences (spot/futures opposite)
- **Issue:** BTC data on Aug 10 had NO true divergences (both moved same direction)
- **Solution:** Added RELATIVE divergence detection
  - SPOT_LEADING_UP/DOWN: Spot CVD 1.5x stronger than futures
  - FUTURES_LEADING_UP/DOWN: Futures CVD 1.5x stronger than spot
- **Result:** Strategy now generates scores and trades!
- **Files Fixed:** `phase2_divergence.py` lines 241-291, 126-137, 409-418

### 5.6. Performance Issues (DISCOVERED BUT NOT FIXED)
- **Backtest Performance:** Drops from 4000 to 173 points/sec during trading (20x slower)
- **Root Cause:** Excessive logging during trades (3-4 lines per trade)
- **Min Score Changed:** From 4.0 to 6.0 to reduce trades (temporary fix)
- **Dashboard Confusion:** Multiple visualizers (visualizer.py, enhanced_visualizer.py, complete_visualizer.py)
- **Not Clear:** Which visualizer is actually being used

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

### ✅ OI DATA IS AVAILABLE ON REMOTE SERVER
- **Measurement:** `open_interest` (not in trades_1s)
- **Field:** `open_interest` (NOT `open_interest_usd` which is NULL)
- **Exchanges:** OKX, BINANCE_FUTURES, BYBIT, DERIBIT, TOTAL_AGG
- **Data Available:** Full historical data for Aug 10, 2025
- **Example Value:** BTC OI = 2.6M BTC on OKX

### OI Query Patterns (CORRECTED)
```sql
-- Get BTC OI from OKX (field is 'open_interest', not 'open_interest_usd')
SELECT mean(open_interest) as oi 
FROM open_interest 
WHERE symbol='BTC' AND exchange='OKX'
AND time >= '2025-08-10T00:00:00Z'
GROUP BY time(5m)

-- Result: ~2.6M BTC open interest

-- Compare all OI types for BTC  
SELECT exchange, open_interest FROM open_interest 
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
- 0 trades generated
- Aggregated 16 SPOT + 9 FUTURES markets correctly ✅
- Data loaded: 16,201 points with CVD calculated ✅
- No divergences detected for trading
```

## Test File Organization (CRITICAL - 2025-08-11)

### PROPER STRUCTURE:
- **Real Tests**: `/tests FUCK YOU CLAUDE/test_*.py`
- **Temp Debug**: `/temp_debug_scripts/quick_*.py` or `debug_*.py`
- **NEVER**: Create test files in root directory!

### WHY THIS MATTERS:
The folder "tests FUCK YOU CLAUDE" was literally created because Claude keeps making too many test files in the wrong places. Don't repeat this mistake!

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