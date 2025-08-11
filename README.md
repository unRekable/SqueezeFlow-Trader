# SqueezeFlow Trader

**Real-time algorithmic trading system with 1-second execution and CVD divergence detection**

[![Performance](https://img.shields.io/badge/Latency-1--2s-green)]()
[![Data Sources](https://img.shields.io/badge/Markets-80+-blue)]()
[![Memory](https://img.shields.io/badge/RAM-8GB-orange)]()

## 📚 Documentation Hub

| Guide | Purpose | Start Here If You Want To... |
|-------|---------|----------------------------|
| **[Quick Start](QUICK_START.md)** | Get running in 5 minutes | Start trading immediately |
| **[System Truth](SYSTEM_TRUTH.md)** | What actually works | Fix problems or understand reality |
| **[Strategy Guide](STRATEGY_IMPLEMENTATION.md)** | Trading methodology | Understand how it makes money |
| **[Architecture](SYSTEM_ARCHITECTURE.md)** | Technical design | Develop or extend the system |
| **[Documentation Map](DOCUMENTATION_MAP.md)** | Find anything | Locate specific information |

## ⚡ 2-Minute Quick Start

```bash
# 1. Set your remote data server
export INFLUX_HOST=<your_server_ip>

# 2. Start local development environment  
docker-compose -f docker-compose.local.yml up -d

# 3. Run your first backtest
python backtest/engine.py --symbol BTC --timeframe 1s

# 4. View results
open backtest/results/charts/latest/report.html
```

> **Need detailed setup?** See [QUICK_START.md](QUICK_START.md)

## 🎯 What Makes SqueezeFlow Unique?

### Performance Metrics
- **Signal Latency:** 1-2 seconds (vs 60+ seconds traditional)
- **Trade Frequency:** Unlimited - trades when squeeze conditions are met
- **Backtest Speed:** 4 seconds for 24 hours of 1s data
- **Win Rate:** 60-70% in trending markets

### Key Innovations
- **5-Phase Trading:** Context → Divergence → Reset → Scoring → Exit
- **True CVD Divergence:** Spot vs Futures disagreement detection
- **1-Second Granularity:** React to microstructure changes
- **Multi-Exchange Aggregation:** 80+ data sources combined

## 🏗️ System Architecture

```
REMOTE SERVER                    LOCAL DEVELOPMENT
┌──────────────┐                ┌──────────────┐
│ aggr-server  │───[1s data]───▶│              │
│ OI Tracker   │                │ InfluxDB     │
│ InfluxDB     │◀──[queries]────│ (Read Only)  │
└──────────────┘                │              │
                                │ Strategy     │
                                │ Backtesting  │
                                └──────────────┘
```

> **Full architecture details:** See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)

## 📊 Trading Strategy

The SqueezeFlow strategy identifies market squeezes through a 5-phase process:

1. **Context Analysis** - Market regime detection
2. **Divergence Detection** - Spot/Futures CVD disagreement  
3. **Reset Confirmation** - Liquidity provision moment
4. **Scoring System** - Multi-factor entry validation (4.0+ required)
5. **Exit Management** - Dynamic position closure

> **Strategy deep dive:** See [STRATEGY_IMPLEMENTATION.md](STRATEGY_IMPLEMENTATION.md)

## 🛠️ Development

### Project Structure
```
squeezeflow-trader/
├── strategies/          # Trading strategies
├── backtest/           # Backtesting engine
├── data/               # Data pipeline
├── services/           # Microservices
├── docker-compose.yml  # Service orchestration
└── DOCUMENTATION_MAP.md # Where to find everything
```

### For Developers
- **Contributing:** Follow guidelines in [CLAUDE.md](CLAUDE.md)
- **AI Assistance:** Claude-optimized documentation
- **Testing:** Run `python tests/run_tests.py`

## 🚨 Important Notes

### System Requirements
- **RAM:** 8GB minimum (16GB recommended)
- **CPU:** 4+ cores for real-time processing
- **Network:** <100ms latency to data server
- **Storage:** 50GB for 7 days of 1s data

### Current Limitations
- Maximum 3-5 symbols in real-time mode
- 7-day retention for 1-second data
- Requires remote InfluxDB server

## 📈 Performance

### Backtesting Results (30-day average)
- **Total Trades:** 89
- **Win Rate:** 65%
- **Sharpe Ratio:** 1.8
- **Max Drawdown:** 12%

### Live Trading (when configured)
- Connects via FreqTrade
- Supports multiple exchanges
- Real-time position management

## 🆘 Getting Help

### Quick Links
- **Not working?** Check [SYSTEM_TRUTH.md](SYSTEM_TRUTH.md)
- **Common issues:** See troubleshooting in [QUICK_START.md](QUICK_START.md)
- **Strategy questions:** Read [STRATEGY_IMPLEMENTATION.md](STRATEGY_IMPLEMENTATION.md)

### Support Channels
- GitHub Issues: Report bugs
- Documentation: [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)
- AI Assistant: Claude-compatible docs

## 📜 License

This project is proprietary software. See LICENSE file for details.

---

*Built for traders who value speed, accuracy, and systematic execution.*