# SqueezeFlow Trader - Project Instructions

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

**Finding Your Data:**
```bash
# 1-second data is in aggr_1s retention policy
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM aggr_1s.trades_1s" -database significant_trades

# Check all retention policies
docker exec aggr-influx influx -execute "SHOW RETENTION POLICIES ON significant_trades"

# Find data range available for backtesting
docker exec aggr-influx influx -execute "SELECT MIN(time), MAX(time) FROM aggr_1s.trades_1s" -database significant_trades
```

**Retention Policy Limits:**
- **aggr_1s**: Only keeps 7 days of data (extended from 24h)
- Data older than 7 days is automatically deleted
- Plan backtests within the retention window
- For longer backtests, consider using higher timeframes (5m, 15m)

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