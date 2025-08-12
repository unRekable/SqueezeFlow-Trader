# SqueezeFlow Trader - Project Instructions

## 🚨 ABSOLUTE PRIORITY #0: ALWAYS CHECK DOCUMENTATION FIRST!!!
**THIS IS THE MOST CRITICAL RULE - NEVER SKIP THIS:**
1. **ALWAYS read SYSTEM_TRUTH.md FIRST** - Contains what actually works vs what's broken
2. **ALWAYS check README.md** - May contain critical updates or configuration changes  
3. **ALWAYS review CLAUDE.md (this file)** - For project-specific instructions
4. **ALWAYS check LESSONS_LEARNED.md** - For patterns to avoid and solutions that work
5. **ALWAYS check DASHBOARD_PROGRESS.md** - For current dashboard implementation status
6. **ALWAYS check relevant docs in /docs/** - For component-specific details
7. **NEVER attempt ANY task without checking docs FIRST**
8. **NEVER assume you remember - ALWAYS verify from documentation**

**If you skip documentation checks, you WILL fail. This is non-negotiable.**

## 📈 SELF-IMPROVEMENT & LEARNING FRAMEWORK
**Claude learns and improves through structured feedback:**

### Progress Tracking Files (ALWAYS UPDATE THESE):
- **DASHBOARD_PROGRESS.md** - Track dashboard implementation status
- **SYSTEM_TRUTH.md** - Document what works vs what's broken
- **LESSONS_LEARNED.md** - Record mistakes and solutions for future reference

### After Each Task:
1. **Document what worked** in the appropriate tracking file
2. **Document what failed** and why
3. **Update progress status** with specific details
4. **Add lessons learned** to LESSONS_LEARNED.md to prevent repeated mistakes
5. **Check for pattern violations** from LESSONS_LEARNED.md

### Self-Validation Loop:
1. **Execute** → 2. **Validate** → 3. **Document** → 4. **Learn**
- Never skip validation
- Always document findings
- Update tracking files immediately

## 🔍 VISUAL VALIDATION - MANDATORY FOR ALL DASHBOARDS
**After generating ANY dashboard or visualization:**
1. **ALWAYS run visual validation** to see what was actually generated
2. **Use the visual_validator.py** to analyze and screenshot dashboards
3. **Read the screenshot** to verify charts are populated
4. **Check validation report** for issues and recommendations

```bash
# After any backtest with visualization:
cd "/Users/u/PycharmProjects/SqueezeFlow Trader"
python3 backtest/reporting/visual_validator.py

# Then read the screenshot to see actual output:
# Read tool: report_*/dashboard_screenshot_*.png (now in root directory!)
```

**This allows Claude to self-debug by seeing exactly what the user sees!**
**Essential for the optimization framework to improve the system.**

## 🔴 CRITICAL ARCHITECTURAL RULE: Single Source of Truth
**When making ANY configuration or behavior change:**
1. **ALWAYS use centralized configuration** (`/backtest/indicator_config.py`)
2. **NEVER hardcode the same logic in multiple files**
3. **ALWAYS make components read from config, not duplicate logic**
4. **ALWAYS check and update ALL dependent components**
5. **See `.claude/ARCHITECTURAL_PRINCIPLES.md` for detailed patterns**

**Example: OI disable required changes in 5+ files. Should have been 1 config change.**

## 🚨 CRITICAL CASCADE POINTS - Change One = Change All

### Configuration Changes Affect:
```
When changing ANY config:
1. /backtest/indicator_config.py (master)
2. /data/pipeline.py (data loading)
3. /strategies/squeezeflow/components/*.py (ALL phase files)
4. /backtest/reporting/*visualizer.py (ALL visualizers)
5. /services/strategy_runner.py (live trading)
```

### Dashboard Changes Cascade:
```
visualizer.py → complete_visualizer.py → 3 HTML files
Change one = must update all or dashboards break
```

### Data Access Has Multiple Paths:
- `backtest/engine.py` → `data/pipeline.py` → `influx_client.py`
- `strategy.py` → DIRECT influx queries (inconsistent!)
- `services/strategy_runner.py` → `DataPipeline` class

## 🔴 CRITICAL: Implementation Process - MANDATORY WORKFLOW

**STEP 0: CHECK EXISTING** (Before creating ANYTHING new)
```bash
# PREVENT DUPLICATION - The #1 cause of architectural issues
grep -r "class.*ClassName" .  # Find if it already exists
grep -r "from.*module import" .  # Find what's actually being used
ls -la **/temp_* **/debug_* **/test_*  # Find temporary files
```

**STEP 1: SEARCH** (Before ANY changes)
```bash
grep -r "feature_name" .  # Find ALL references
grep -r "oi_data\|open_interest\|OI" strategies/ backtest/ data/  # Example for OI
```

**STEP 2: IMPLEMENT** (Change ALL files found)
- Update EVERY file that references the feature
- Make them ALL use central config
- Don't leave any hardcoded versions

**STEP 3: VERIFY** (Test it actually works)
```bash
python3 -c "from module import thing; print(thing.works())"  # Quick test
python3 test_integration.py  # Full test
```

**STEP 4: DOCUMENT** (Only AFTER it works)
- Update relevant .md files
- Add to IMPLEMENTATION_CHECKLIST.md
- Document what was ACTUALLY done, not planned

**FAILURE MODES TO AVOID (See LESSONS_LEARNED.md for full list):**

1. **The Visualizer Multiplication:** Creating new instead of fixing existing
   - We had 14 visualizers because each debug session created a new one ❌
   
2. **The Config Bypass:** Assuming all components use central config
   - Phase3 and Phase5 didn't import indicator_config at all ❌
   
3. **The Documentation Delusion:** Writing docs instead of implementation
   - Created documentation about how it should work ❌
   - Made example files showing the pattern ❌
   - But didn't actually update the code to use it ❌
   
4. **The Data Path Chaos:** Multiple ways to access same data
   - Strategy used DataPipeline AND direct InfluxDB queries ❌

**SUCCESS MODE:**
1. grep for ALL occurrences ✅
2. Update ALL files to use config ✅
3. Test that it works ✅
4. THEN document ✅

## 🔴 CRITICAL BACKTEST RULES - NEVER VIOLATE THESE

### RULE #1: ALWAYS Check Available Data First
```python
# MANDATORY FIRST STEP - Check what data actually exists:
cd "/Users/u/PycharmProjects/SqueezeFlow Trader" && python3 -c "
from influxdb import InfluxDBClient
client = InfluxDBClient(host='213.136.75.120', port=8086, database='significant_trades')
result = client.query(\"SELECT * FROM aggr_1s.trades_1s WHERE market = 'BINANCE:btcusdt' ORDER BY time DESC LIMIT 1\")
for point in result.get_points():
    print(f'Latest data: {point.get(\"time\")}')
result = client.query(\"SELECT * FROM aggr_1s.trades_1s WHERE market = 'BINANCE:btcusdt' ORDER BY time ASC LIMIT 1\")
for point in result.get_points():
    print(f'Earliest data: {point.get(\"time\")}')"
```

### RULE #2: ALWAYS Use Full Available Data Range
```bash
# WRONG - Never use arbitrary dates without checking:
python3 backtest/engine.py --start-date 2025-08-09 --end-date 2025-08-10  # DON'T DO THIS!

# RIGHT - Use the ACTUAL available data range:
# If data shows: 2025-08-10 00:00 to 2025-08-10 23:59
python3 backtest/engine.py --start-date 2025-08-10 --end-date 2025-08-10
```

### RULE #3: Sequential Processing Requirement
- Backtest processes EVERY candle sequentially (real-time simulation)
- Strategy has access to ALL historical data up to current point
- No artificial windowing or stepping - matches live trading exactly

### VIOLATION CONSEQUENCES:
- "Insufficient historical data for analysis" error
- Wasted time and resources
- User frustration
- Failed backtests

**CHECK THESE RULES BEFORE EVERY BACKTEST - NO EXCEPTIONS!**

## ⚠️ CRITICAL: Read SYSTEM_TRUTH.md First!
**Before doing ANYTHING, read `/SYSTEM_TRUTH.md` for what actually works and what's broken.**

## 🎯 Core Behavioral Rules

### Priority #1: User Instructions Always Win
- User's explicit instructions override ALL other rules
- When user says "don't do X yet", NEVER do X
- Ask for clarification if instructions are unclear

### Priority #2: Fix, Don't Delete
- NEVER comment out working code to "simplify"
- NEVER remove required functionality
- Diagnose and fix root causes
- Preserve all working features

### Priority #3: Clean Environment
- Name temporary files clearly: `test_*.py`, `debug_*.py`, `temp_*.py`
- Auto-cleanup runs on session end
- Don't commit temporary files to git

## 📊 Project Context

### System Architecture
- Docker-based microservices (docker-compose.yml)
- InfluxDB time-series database for market data
- Redis for caching and real-time messaging
- FreqTrade for trade execution
- Strategy Runner service for signal generation
- Unified configuration via environment variables

### Key Technologies
- Python 3.11+ for strategy and services
- Node.js for aggr-server (market data collection)
- Docker for containerization
- InfluxDB 1.8 for time-series data
- Redis 7 for caching

### Critical Files
- `docker-compose.yml` - Service orchestration and configuration
- `strategies/` - Trading strategy implementations
- `backtest/` - Backtesting engine
- `services/` - Microservices code
- `services/config/unified_config.py` - Unified configuration loader

## ⚡ Real-Time 1-Second Data Operations

### 🚨 CRITICAL: 1s Data is Now Default
- **REAL-TIME EXECUTION**: System now operates in 1-second intervals (not 60-second)
- **Data Collection**: 1-second intervals with perfect timeframe alignment
- **Strategy Execution**: 1-second cycle time for ultra-low latency
- **Signal Latency**: 1-2 seconds total (vs previous 60-70 seconds)
- **Production Status**: Battle-tested and production-ready

### 🎯 1-Second Configuration Variables

**Essential 1s Variables (Already Configured):**
```bash
SQUEEZEFLOW_RUN_INTERVAL=1          # 1-second strategy execution (was 60)
SQUEEZEFLOW_DATA_INTERVAL=1         # 1-second data collection
SQUEEZEFLOW_ENABLE_1S_MODE=true     # Enable all 1s optimizations
SQUEEZEFLOW_MAX_SYMBOLS=3           # Reduced from 5 for real-time
REDIS_MAXMEMORY=2gb                 # Increased for 1s data buffering
INFLUX_RETENTION_1S=7d              # 7-day 1s data retention (extended from 24h)
```

### 📁 Data Storage and Retention Policies

**InfluxDB Retention Policy Configuration:**
- **aggr_1s**: 7 days of 1-second data (was 24 hours, extended for backtesting)
  - Location: `significant_trades.aggr_1s.trades_1s`
  - Storage: ~10-20GB per symbol (600M data points)
  - Purpose: Real-time trading and recent backtesting
- **rp_1s**: 30 days of aggregated data
  - Location: `significant_trades.rp_1s.*`
  - Purpose: Long-term analysis and validation

**Setting Up Retention Policies:**
```bash
# Extend 1s data retention to 7 days
./scripts/setup_retention_policies.sh

# Verify retention policies
docker exec aggr-influx influx -execute "SHOW RETENTION POLICIES ON significant_trades"
```

### 📊 Real-Time Performance Monitoring

**Monitor 1-Second Performance:**
```bash
# Real-time 1s data monitoring
./scripts/monitor_performance.sh

# Check 1-second data pipeline health
python scripts/monitor_performance.py

# Validate 1s data collection
curl http://localhost:8090/health/data/1s

# Performance metrics for 1s system
docker stats --no-stream | grep -E "strategy-runner|redis|aggr"

# Check 1s data retention policy
./scripts/setup_retention_policy.sh --verify
```

### 🔧 1-Second Troubleshooting Commands

**Common 1s Issues:**
```bash
# Memory usage too high (1s data intensive)
docker exec redis redis-cli info memory
docker exec aggr-influx influx -execute "SHOW RETENTION POLICIES ON significant_trades"

# CPU bottlenecks in real-time processing
docker exec strategy-runner top -bn1 | head -10

# Check 1s data flow
docker logs aggr-server | grep -E "1s|second" | tail -10
docker logs strategy-runner | grep -E "signal.*generated" | tail -5

# Network latency issues
ping -c 5 [exchange_endpoints]
curl -w "@curl-format.txt" -s http://localhost:8090/health

# Real-time system health
./scripts/test_implementation.sh
```

### ⚠️ 1-Second System Constraints

**Performance Requirements:**
- **CPU**: Minimum 4 cores (8+ recommended for multiple symbols)
- **RAM**: 8GB minimum (16GB recommended for production)
- **Storage**: NVMe SSD required (high IOPS for 1s data)
- **Network**: <50ms latency to exchanges (co-location preferred)

**Operational Limits:**
- **Max Symbols**: Reduced to 3-5 for real-time processing
- **Memory Usage**: 2-4x higher than 60s mode
- **CPU Load**: Significantly higher continuous processing
- **Network Dependency**: Any interruption affects real-time performance

### 🚀 1-Second Performance Optimization

**Production Optimization Commands:**
```bash
# Enable real-time mode with optimizations
export SQUEEZEFLOW_ENABLE_1S_MODE=true

# Start with real-time configuration
docker-compose -f docker-compose.yml -f docker-compose.realtime.yml up -d

# Monitor real-time performance continuously
watch -n 1 'docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"'

# Check 1s signal generation speed
redis-cli --latency-history -i 1

# Optimize 1s data retention
docker exec aggr-influx influx -execute "SHOW SERIES CARDINALITY ON significant_trades"
```

## 🔧 Configuration Management

### Unified Configuration System
- All configuration in `docker-compose.yml` environment variables
- No separate `.env` or `config.yaml` files
- Services load config via `services/config/unified_config.py`
- Environment variables prefixed: `SQUEEZEFLOW_`, `REDIS_`, `INFLUX_`, `FREQTRADE_`

### Configuration Reference
- See `/docs/unified_configuration.md` for complete variable reference
- Default values defined in `unified_config.py`
- Override via docker-compose environment section

## 🛠️ Development Guidelines

### Code Quality
- Use Black formatting for Python
- Type hints for function signatures
- Docstrings for public functions
- Keep functions under 50 lines
- Keep files under 500 lines

### Testing Requirements
- Write tests for new features
- Run existing tests before commits
- Maintain test coverage above 80%

### Git Workflow
- Atomic commits (single logical change)
- Descriptive commit messages
- Don't commit temporary files
- Review changes before committing

## ⚠️ Common Issues & Solutions

### InfluxDB Connection Issues
- Check if Docker container is running: `docker ps`
- Verify port 8086 is accessible
- Check environment variables in docker-compose.yml

### Strategy Not Generating Signals
- Verify data is flowing: Check InfluxDB has recent data
- Check Redis connectivity
- Review strategy logs in Docker

### Docker Services Issues
- Use `docker-compose logs [service]` for debugging
- Restart specific service: `docker-compose restart [service]`
- Full restart: `docker-compose down && docker-compose up -d`

### 🕐 Timezone Issues and Data Timestamps

**CRITICAL: All data is stored in UTC**
- InfluxDB stores all timestamps in UTC (Coordinated Universal Time)
- System logs display times in UTC
- Local time conversion happens only for display

**Common Timezone Confusion:**
```bash
# Check current time in UTC
date -u

# Check InfluxDB latest data (shows UTC timestamps)
docker exec aggr-influx influx -execute "SELECT * FROM trades_1s ORDER BY time DESC LIMIT 1" -database significant_trades

# Example timezone conversions:
# UTC 19:06 = CEST 21:06 (Central European Summer Time, UTC+2)
# UTC 08:39 = CEST 10:39
```

**Debugging Data Availability:**
```bash
# Check if data is actively flowing (last 10 seconds)
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 10s" -database significant_trades

# Check data retention window (shows oldest data)
docker exec aggr-influx influx -execute "SELECT * FROM trades_1s ORDER BY time ASC LIMIT 1" -database significant_trades

# Check total data points in retention policy
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM aggr_1s.trades_1s" -database significant_trades
```

**Backtest Time Range Considerations:**
- Always specify times in UTC for backtests
- Account for retention policy limits (7 days for 1s data)
- Data before retention window is automatically deleted

### 📊 Data Location and Retention Issues

**⚠️ CRITICAL ANTI-PATTERN: Local vs Remote InfluxDB**
- **NEVER** use local Docker InfluxDB (aggr-influx) for backtesting - it only has test data!
- **ALWAYS** use remote production InfluxDB server for real market data
- **Local aggr-influx**: Only for local testing and development (no real data)
- **Remote InfluxDB**: Contains all real market data for backtesting and live trading

**🎯 MANDATORY BACKTEST CONFIGURATION:**
```bash
# ALWAYS use these settings for ALL crypto pairs (BTC, ETH, AVAX, etc.):
export INFLUX_HOST=213.136.75.120  # Remote production server IP (NEVER use local)
export INFLUX_PORT=8086
export TIMEFRAME=1s  # ALWAYS use 1-second data (system default)

# Standard backtest command for ANY crypto pair:
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol AVAX --start-date 2025-08-08 --end-date 2025-08-10 --timeframe 1s --balance 10000 --leverage 1.0 --strategy SqueezeFlowStrategy

# Examples for different pairs (ALL use 1s data):
# BTC:
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol BTC --timeframe 1s --start-date 2025-08-08 --end-date 2025-08-10

# ETH:
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol ETH --timeframe 1s --start-date 2025-08-08 --end-date 2025-08-10

# AVAX:
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol AVAX --timeframe 1s --start-date 2025-08-08 --end-date 2025-08-10

# WRONG - This uses local Docker with no real data:
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM trades_1s"

# RIGHT - Connect to remote server for real data:
influx -host your_server_ip -execute "SELECT COUNT(*) FROM trades_1s" -database significant_trades
```

**Finding Your Data (on REMOTE server):**
```bash
# 1-second data is in aggr_1s retention policy
influx -host your_server_ip -execute "SELECT COUNT(*) FROM aggr_1s.trades_1s" -database significant_trades

# Check all retention policies
influx -host your_server_ip -execute "SHOW RETENTION POLICIES ON significant_trades"

# Find data range available for backtesting
influx -host your_server_ip -execute "SELECT MIN(time), MAX(time) FROM aggr_1s.trades_1s" -database significant_trades

# Check Open Interest data availability
influx -host your_server_ip -execute "SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='TOTAL_AGG' ORDER BY time DESC LIMIT 5" -database significant_trades
```

### 📈 Open Interest (OI) Data Integration

**IMPORTANT: OI Data Source Configuration**
- **Production Server**: OI tracker runs on server, writes to remote InfluxDB
- **Local Development**: Strategies read OI from same remote InfluxDB (not local APIs)
- **Data Flow**: Server collects OI → Remote InfluxDB → Local strategies read from remote

**OI Data Structure in InfluxDB:**
```bash
# Individual exchange OI
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='BINANCE_FUTURES'

# Top 3 futures combined (BINANCE + BYBIT + OKX)
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='FUTURES_AGG'

# All exchanges combined (includes DERIBIT options)
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='TOTAL_AGG'
```

**Using OI Data in Strategies:**
```python
# Use InfluxDB-based OI tracker (not direct API calls)
from strategies.squeezeflow.components.oi_tracker_influx import OITrackerInflux

# Initialize with remote InfluxDB connection
tracker = OITrackerInflux(rise_threshold=5.0)

# Get OI metrics from remote server
oi_data = tracker.get_oi_change_sync('BTC')

# Validate squeeze with OI confirmation
is_valid, reason = tracker.validate_squeeze_with_oi('BTC', squeeze_detected=True)
```

**Testing OI Remote Access:**
```bash
# Test OI data retrieval from remote server
INFLUX_HOST=your_server_ip python test_oi_remote.py

# Verify OI data flow
python -c "
from strategies.squeezeflow.components.oi_tracker_influx import OITrackerInflux
tracker = OITrackerInflux()
print(tracker.get_oi_change_sync('BTC'))
"
```

**Retention Policy Limits:**
- **aggr_1s**: Only keeps 7 days of data (extended from 24h)
- Data older than 7 days is automatically deleted
- Plan backtests within the retention window
- For longer backtests, consider using higher timeframes (5m, 15m)

### ⚠️ CRITICAL: Backtest "Insufficient Historical Data" Error

**Root Cause:** Requesting data outside the available date range.

**The Problem:**
```bash
# This will FAIL if data doesn't exist for these dates
python3 backtest/engine.py --start-date 2025-08-09 --end-date 2025-08-10
# Why: Data might only exist from 2025-08-10 onward
```

**The Solution:**
```bash
# First, check what data is actually available
influx -host 213.136.75.120 -execute "SELECT MIN(time), MAX(time) FROM trades_1s WHERE market='BINANCE:btcusdt'" -database significant_trades

# Then use dates WITHIN the available range
python3 backtest/engine.py --start-date 2025-08-10 --end-date 2025-08-10
# This works because data exists for Aug 10
```

**Quick Fix Checklist:**
1. Check available data range first
2. Use dates within the available range
3. Don't request dates before data exists

## 🚫 Anti-Patterns to Avoid

### Never Do These
- Hard-code credentials (all config in docker-compose.yml environment variables)
- Use fixed thresholds in strategies (dynamic adaptation required)
- Comment out code instead of fixing issues
- Create complex abstractions without clear benefit
- Leave debugging print statements in production code

### Always Do These
- Validate user input
- Handle errors gracefully
- Log important events
- Clean up resources (files, connections)
- Document complex logic

## 📝 Helper Script Guidelines

When creating temporary scripts:
```python
# test_connection.py
# Temporary helper script - auto-cleanup enabled
import requests
response = requests.get("http://localhost:8086/ping")
print(f"Status: {response.status_code}")
```

These will be automatically cleaned up after the session.

## 🎯 Project-Specific Requirements

### SqueezeFlow Strategy
- Implements 5-phase trading methodology
- Uses CVD (Cumulative Volume Delta) analysis
- Multi-timeframe validation (1m, 5m, 15m, 30m, 1h, 4h)
- Dynamic market adaptation (no fixed thresholds)

### Performance Targets (1-Second Real-Time System)
- **🆕 Signal generation < 1-2 seconds total** (vs previous 60+ seconds)
- **🆕 Data collection latency < 1 second** (real-time streaming)
- Startup time < 5 seconds (increased for 1s initialization)
- **Memory usage 2-4x baseline** (1s data buffering requirements)
- Backtest throughput > 1000 candles/second (maintained)

### Data Requirements (1-Second Real-Time)
- **🆕 1-second granularity data collection** (perfect timeframe alignment)
- Minimum 24 hours of historical data (all timeframes)
- **🆕 24-hour rolling window for 1s data** (storage optimization)
- Data quality > 99% (critical for 1s precision)
- Multi-exchange support via aggr-server (1s aggregation)
- **🆕 Real-time streaming with sub-second latency**

## 🔧 Quick Commands (1-Second Real-Time)

```bash
# Start all services (with 1s optimizations)
docker-compose up -d

# 🆕 Monitor real-time 1s performance
./scripts/monitor_performance.sh
python scripts/monitor_performance.py

# Check service status
docker-compose ps

# 🆕 Check 1s data health
curl http://localhost:8090/health/data/1s
./scripts/setup_retention_policy.sh --verify

# View logs (focus on real-time processing)
docker-compose logs -f strategy-runner
docker logs aggr-server | grep -E "1s|second" | tail -10

# Check configuration (1s variables highlighted)
docker exec squeezeflow-strategy-runner env | grep -E "SQUEEZEFLOW|REDIS|INFLUX|FREQTRADE"

# 🆕 Monitor 1s system performance
watch -n 1 'docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"'

# Run backtest (supports 1s data)
python run_backtest.py

# Check system health (includes 1s metrics)
curl http://localhost:8090/health
./scripts/test_implementation.sh

# Access interfaces
open http://localhost:8080  # FreqTrade UI

# 🆕 Real-time troubleshooting
redis-cli --latency-history -i 1  # Check Redis latency
docker exec strategy-runner top -bn1 | head -10  # CPU usage

# Rebuild after config changes (with 1s optimizations)
docker-compose build && docker-compose restart
```

## 📚 Documentation

- Configuration guide: `/docs/unified_configuration.md`
- Strategy methodology: `/docs/squeezeflow_strategy.md`
- Service architecture: `/docs/services_architecture.md`
- System overview: `/docs/system_overview.md`

---

Remember: The goal is to build a robust, maintainable trading system that respects user decisions and maintains clean, working code.
- Configuration cascade: indicator_config.py → pipeline.py → ALL phase files → ALL visualizers → 
  strategy_runner.py - MUST update all together
  - Dashboard cascade: visualizer.py → complete_visualizer.py → 3 HTML files - change one = update ALL
  - Phase3 and Phase5 don't import config - they bypass the system completely
  - Always use INFLUX_HOST=213.136.75.120 for ALL data (prices, CVD, OI) - NEVER local Docker
  - Standard backtest: INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol [SYMBOL] 
  --start-date 2025-08-10 --end-date 2025-08-10 --timeframe 1s
  - ALWAYS check data exists before backtest - no assumptions about dates
  - SEARCH with grep before ANY change - find ALL occurrences first
  - Update SYSTEM_TRUTH.md and DASHBOARD_PROGRESS.md at END of EVERY task
  - Use file:line_number references like engine.py:245 for navigation
  - Visual validation MANDATORY: python3 backtest/reporting/visual_validator.py then read screenshot
  - Multiple conflicting visualizers exist - unclear which is active
  - Don't document plans - implement them. Don't assume - verify. Don't overcomplicate - use native 
  features.