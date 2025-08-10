# SqueezeFlow Trader

A sophisticated CVD (Cumulative Volume Delta) convergence exhaustion trading system that identifies market equilibrium restoration moments through spot vs futures flow analysis.

## 🚀 Quick Start

### Single Machine Setup (All Local)
```bash
# Start all services locally
docker-compose up -d

# Run backtest
python3 backtest/engine.py --symbol ETH --start-date 2025-08-09 --end-date 2025-08-09

# Check system health
curl http://localhost:8090/health
```

### 🌐 Distributed Setup (Recommended for Production)
For reliable 24/7 data collection, see [DISTRIBUTED_SETUP.md](./DISTRIBUTED_SETUP.md) for:
- **Server**: Dedicated data collection (aggr-server + InfluxDB)
- **MacBook**: Development environment with remote data access

```bash
# Server setup (VPS/dedicated server)
docker-compose -f docker-compose.server.yml up -d

# MacBook setup (development)
cp .env.example .env  # Set SERVER_IP to your server
docker-compose -f docker-compose.local.yml up -d

# Test connection
./scripts/setup_distributed.sh test-connection
```

## Architecture

- **Data Collection**: aggr-server (1s intervals, 80+ market pairs)
- **Storage**: InfluxDB (`significant_trades.aggr_1s.trades_1s`)
- **Strategy**: 5-phase SqueezeFlow methodology
- **Execution**: FreqTrade integration
- **Monitoring**: Real-time performance tracking

## Key Files

- `docker-compose.yml` - Service configuration
- `strategies/squeezeflow/` - Trading strategy
- `backtest/engine.py` - Backtesting system
- `data/pipeline.py` - Data processing
- `CLAUDE.md` - Development guidelines

## Data Location

```sql
-- Data is in aggr_1s retention policy
SELECT * FROM "aggr_1s"."trades_1s" WHERE market =~ /ETH/
```

## Performance

- Data collection: 1-second intervals
- Signal generation: 1-2 seconds total latency
- Analysis timeframes: 1m, 5m, 15m, 30m, 1h, 4h (built from 1s data)

## 📚 Documentation

- [DISTRIBUTED_SETUP.md](./DISTRIBUTED_SETUP.md) - **Distributed architecture setup**
- [SYSTEM_TRUTH.md](./SYSTEM_TRUTH.md) - **What actually works (read first!)**
- [CLAUDE.md](./CLAUDE.md) - Development guidelines and project instructions
- `/docs/squeezeflow_strategy.md` - Strategy methodology
- `/docs/system_overview.md` - System architecture  
- `/docs/unified_configuration.md` - Configuration reference

## 🔧 Helper Scripts

- `./scripts/setup_distributed.sh` - Distributed setup and connection testing
- `./scripts/setup_retention_policies.sh` - InfluxDB retention policy setup
- `./scripts/monitor_performance.sh` - Real-time performance monitoring