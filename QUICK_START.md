# Quick Start Guide

## Prerequisites
- Docker installed
- 8GB RAM
- Remote InfluxDB server IP

## Setup (2 minutes)

```bash
# 1. Configure remote data server
export INFLUX_HOST=<server_ip>

# 2. Start services
docker-compose -f docker-compose.local.yml up -d

# 3. Run backtest
python backtest/engine.py --symbol BTC --timeframe 1s

# 4. View results
open backtest/results/charts/latest/report.html
```

## Troubleshooting

### No Data?
```bash
# Check remote connection
curl http://$INFLUX_HOST:8086/ping

# Verify data exists
python test_oi_remote.py
```

### Slow Performance?
- Check network latency: `ping $INFLUX_HOST`
- Increase Redis cache in docker-compose.yml
- Use 5m timeframe instead of 1s

### No Trades?
- Needs CVD divergence (Phase 2)
- Score must be > 4.0
- Check logs: `docker logs strategy-runner`

## Commands Reference

```bash
# Services
docker-compose ps                    # Check status
docker-compose logs -f [service]     # View logs
docker-compose restart [service]     # Restart

# Backtesting
python backtest/engine.py --help     # All options
--symbol [BTC/ETH/SOL]              # Choose symbol
--timeframe [1s/1m/5m]              # Set timeframe
--start-date YYYY-MM-DD             # Start date
--end-date YYYY-MM-DD               # End date

# Monitoring
docker stats --no-stream             # Resource usage
curl http://localhost:8090/health   # System health
```

## Next Steps
- Read [STRATEGY_IMPLEMENTATION.md](STRATEGY_IMPLEMENTATION.md) to understand trading logic
- Check [SYSTEM_TRUTH.md](SYSTEM_TRUTH.md) for configuration details
- See [README.md](README.md) for project overview