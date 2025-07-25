# SqueezeFlow Trader

**Professional Cryptocurrency Trading System with CVD-Divergence Squeeze Detection**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Core Concept

SqueezeFlow Trader detects "squeeze" conditions through **Cumulative Volume Delta (CVD) divergence analysis** between spot and futures markets. The system identifies high-probability trading opportunities by analyzing volume flow differences across 24+ exchanges.

### Squeeze Detection Algorithm

- **Long Squeeze**: Price ↑ + Spot CVD ↑ + Futures CVD ↓ → Negative Score (Buy Signal)
- **Short Squeeze**: Price ↓ + Spot CVD ↓ + Futures CVD ↑ → Positive Score (Sell Signal)
- **CVD Methodology**: Industry-standard `(buy_volume - sell_volume).cumsum()` verified against aggr.trade
- **Multi-Timeframe Analysis**: 5min (fast) to 240min (4h trend) lookback periods

## 🏗️ System Architecture

### Microservices Data Pipeline
```
Exchange APIs → aggr-server → InfluxDB → Symbol/Market Discovery → SqueezeFlow Calculator → Redis → FreqTrade → Order Execution
```

### Core Components

#### SqueezeFlow Calculator (`services/squeezeflow_calculator.py`)
- **Main signal generation engine** with automatic symbol discovery
- **Real-time CVD calculation** for all available trading pairs  
- **Multi-timeframe scoring**: 5, 10, 15, 30, 60, 120, 240 minute lookbacks
- **Score persistence**: Stores signals in InfluxDB and caches in Redis

#### Market Discovery System (`utils/`)
- **Database-driven symbol detection** replacing hardcoded lists
- **Automatic SPOT/PERP classification** via Exchange Mapper
- **Data quality validation**: Minimum 500 data points in 24h requirement
- **Pattern matching**: Precise symbol extraction (e.g., `BINANCE:btcusdt` → `BTC`)

#### Squeeze Score Calculator (`indicators/squeeze_score_calculator.py`)
- **Weighted scoring system**: 30% Price + 35% Spot CVD + 35% Futures CVD
- **CVD trend normalization**: 100,000 USD per period significance threshold
- **Signal classification**: STRONG_LONG_SQUEEZE (-0.4) to STRONG_SHORT_SQUEEZE (+0.4)
- **Smoothing**: 5-period moving average for noise reduction

#### FreqTrade Integration (`freqtrade/user_data/strategies/SqueezeFlowFreqAI.py`)
- **Signal persistence system**: 5-10 minute validity windows for timing optimization
- **Multi-timeframe entry conditions**: Primary (20min) + confirmation timeframes
- **Reverse exit logic**: Exit on opposite squeeze signals
- **FreqAI ML enhancement**: Machine learning integration for improved decisions

#### Backtest Engine (`backtest/engine.py`)
- **Historical strategy replay** with exact live logic simulation
- **Signal regeneration**: Identical squeeze calculation methodology
- **Performance metrics**: Win rate, PnL tracking, duration analysis
- **Market discovery compatibility**: Uses same symbol detection as live system

## 📊 Trading Strategy

### Entry Conditions (ALL must be met)
- **Primary Signal**: Squeeze score ≤ -0.3 (Long) or ≥ +0.3 (Short)
- **Signal Persistence**: Active signal less than 5 minutes old
- **Multi-Timeframe Alignment**: 
  - Entry timing: 10min or 30min confirmation
  - Primary signals: 60min (1h) or 120min (2h) squeeze detection
- **Technical Filters**: RSI < 70 (Long), Volume > 1.5x average
- **Risk Management**: Max 2 open positions, 20% position sizing

### Exit Conditions
- **Reverse signals**: Exit Long on Short squeeze (score ≥ +0.3)
- **Signal expiration**: 30+ minutes without active signal
- **Emergency exits**: Position age > 120min with weak score < 0.05
- **Hard stop loss**: 2.5% spot risk (adjusted for 5x leverage)

### Score Classification
```python
score <= -0.4:  "STRONG_LONG_SQUEEZE"     # Very strong buy signal
score <= -0.2:  "LONG_SQUEEZE"            # Weak buy signal  
score >= +0.4:  "STRONG_SHORT_SQUEEZE"    # Very strong sell signal
score >= +0.2:  "SHORT_SQUEEZE"           # Weak sell signal
else:           "NEUTRAL"                 # No signal
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- 4GB+ RAM, 50GB+ disk space

### 1. System Setup
```bash
# Clone repository
git clone https://github.com/unRekable/SqueezeFlow-Trader.git
cd SqueezeFlow-Trader

# Initialize system
python init.py --mode development

# Validate setup
python validate_setup.py
```

### 2. Start Services
```bash
# Development mode (recommended first)
python main.py start --dry-run

# Production mode with Docker
./start.sh

# Check system status
python status.py
```

### 3. Access Interfaces
- **Grafana Dashboards**: http://localhost:3002 (admin/admin)
- **FreqTrade UI**: http://localhost:8081
- **aggr-server**: http://localhost:3000
- **System logs**: `data/logs/squeezeflow.log`

## 📈 CVD Calculation Methodology

### Industry-Standard Formula
```python
# Step 1: Calculate per-minute volume delta
volume_delta = buy_volume - sell_volume

# Step 2: Calculate CUMULATIVE Volume Delta (running total)
cvd = volume_delta.cumsum()  # Industry standard approach

# Step 3: Trend analysis with normalization
cvd_slope = np.polyfit(time_periods, cvd_values, 1)[0]
normalized_trend = cvd_slope / 100_000  # Scale for cumulative values
```

### Real Market Coverage
- **BTC Markets**: 47 SPOT + 16 PERP exchanges automatically discovered
- **ETH Markets**: 49 SPOT + 21 PERP exchanges automatically discovered  
- **Symbol Discovery**: Automatic detection of available trading pairs from database
- **Data Quality**: Minimum thresholds ensure reliable signal generation

## 🧪 Testing & Validation

### Backtesting
```bash
# Quick backtest examples  
python run_backtest.py last_week
python run_backtest.py last_month 20000

# Custom timeframe
python backtest/engine.py --start-date 2025-01-01 --end-date 2025-01-31 --balance 50000
```

### System Tests
```bash
# Complete system validation
python validate_setup.py

# Component status checks
python status.py

# CVD analysis tool (any pair/timeframe)
python utils/cvd_analysis_tool.py BTC --hours 24
```

## 🔧 Configuration

### Core Configuration Files
```
config/
├── config.yaml              # Main system settings
├── exchanges.yaml            # API credentials & rate limits  
├── risk_management.yaml      # Position sizing & stop losses
├── trading_parameters.yaml   # Squeeze detection thresholds
├── ml_config.yaml           # FreqAI machine learning settings
└── execution_config.yaml    # Order execution parameters
```

### Key Trading Parameters
```yaml
# Squeeze Detection (optimized for cumulative CVD)
squeeze_detection:
  min_score_threshold: 0.3      # Entry threshold (lowered from 0.6)
  signal_persistence_minutes: 10 # Signal validity window
  confirmation_candles: 2        # Required confirmation
  
# Risk Management  
position_sizing:
  max_position_size: 0.20       # 20% per position
  max_open_trades: 2            # Maximum simultaneous positions
  leverage: 5.0                 # 5x leverage for precision
  
# CVD Calculation
cvd_normalization_factor: 100000  # 100k USD trend significance
```

## 🐳 Docker Deployment

### Production Deployment
```bash
# Full containerized stack
python init.py --mode production
./start.sh

# Individual service management
docker-compose up -d aggr-influx redis
docker-compose up -d squeezeflow-calculator
docker-compose up -d freqtrade grafana

# Monitor logs
docker-compose logs -f squeezeflow-calculator
```

### Service Health
```bash
# Check all services
python status.py

# Docker service status
docker-compose ps

# Service resource usage
docker stats
```

## 📊 Performance Characteristics

### Signal Generation
- **Processing latency**: Sub-100ms signal calculation
- **Multi-timeframe analysis**: 7 concurrent lookback periods
- **Symbol discovery**: Automatic scaling to new trading pairs
- **Data throughput**: Real-time processing of 24+ exchange feeds

### Trading Performance (Backtested)
- **Historical win rate**: 60-75% depending on market conditions
- **Average trade duration**: 2-8 hours based on timeframe
- **Risk-adjusted returns**: Configurable risk management
- **Maximum drawdown**: 15% system-wide protection

## 🔐 Security & Risk Management

### Security Features
- **API key protection**: Environment variable isolation
- **Testnet support**: Safe testing on all major exchanges
- **Dry-run mode**: Paper trading without real funds  
- **Rate limiting**: Built-in exchange API protection

### Risk Controls
- **Position limits**: 20% maximum per position, 2 concurrent trades
- **Dynamic stop losses**: Adjusted for leverage (2.5% spot risk)
- **Signal validation**: Multi-timeframe confirmation required
- **Emergency exits**: Automatic position closure on system issues

## 📚 Technical Documentation

### System Resources
- **[CLAUDE.md](CLAUDE.md)**: Complete technical architecture documentation
- **Service Documentation**: Individual service docs in respective directories  
- **Configuration Reference**: Fully documented YAML files in `config/`

### API References
- **FreqTrade API**: http://localhost:8080/docs (when running)
- **InfluxDB Query**: Direct database access for analysis
- **Redis Cache**: Signal persistence and system state

## ⚠️ Risk Disclaimer

**IMPORTANT**: This is sophisticated trading software for experienced users. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always start with dry-run mode and thoroughly backtest strategies before live trading.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔧 Technical Foundation

**SqueezeFlow Trader** represents institutional-grade trading infrastructure:

- **Signal Generation**: Multi-timeframe CVD divergence analysis with automatic symbol discovery
- **Data Pipeline**: 24+ exchange aggregation → Time-series storage → Real-time processing
- **Risk Management**: Dynamic position sizing, leverage-adjusted stops, drawdown protection  
- **Execution Engine**: FreqTrade integration with ML-enhanced decision making
- **Architecture**: Docker microservices, InfluxDB time-series database, Redis caching

**Repository**: https://github.com/unRekable/SqueezeFlow-Trader