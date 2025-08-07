# SqueezeFlow Trader - Project Instructions

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

### Performance Targets
- Startup time < 1 second
- Memory usage < 100MB baseline
- Signal generation < 100ms
- Backtest throughput > 1000 candles/second

### Data Requirements
- Minimum 24 hours of historical data
- Data quality > 95% (minimal gaps)
- Multi-exchange support via aggr-server
- Real-time data processing

## 🔧 Quick Commands

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f [service]

# Check configuration
docker exec squeezeflow-strategy-runner env | grep -E "SQUEEZEFLOW|REDIS|INFLUX|FREQTRADE"

# Run backtest
python run_backtest.py

# Check system health
curl http://localhost:8090/health

# Access FreqTrade UI
open http://localhost:8080

# Rebuild after config changes
docker-compose build && docker-compose restart
```

## 📚 Documentation

- Configuration guide: `/docs/unified_configuration.md`
- Strategy methodology: `/docs/squeezeflow_strategy.md`
- Service architecture: `/docs/services_architecture.md`
- System overview: `/docs/system_overview.md`

---

Remember: The goal is to build a robust, maintainable trading system that respects user decisions and maintains clean, working code.