# Strategy Runner Service - Phase 4 Live Trading

## Overview

The Strategy Runner Service is the live trading component of the SqueezeFlow Trader system. It bridges the gap between backtest strategy code and live trading by running the same SqueezeFlow strategy on real-time data and converting orders to signals for FreqTrade execution.

## Architecture

```
Real-time Data → Strategy Runner → Strategy Processing → Signal Generation → Redis/InfluxDB
                      ↓
FreqTrade ← Redis Signals ← Signal Publishing ← CVD Calculation ← Data Pipeline
```

### Key Components

1. **StrategyRunner**: Main service class that orchestrates the live trading pipeline
2. **ConfigManager**: Handles configuration from multiple sources (files, environment, defaults)  
3. **DataPipeline**: Loads real-time market data and calculates CVD
4. **SqueezeFlowStrategy**: The same strategy used in backtesting
5. **Signal Publishing**: Converts strategy orders to Redis signals for FreqTrade

## Features

### ✅ Core Functionality
- **Real-time CVD calculation** (unlike backtest which uses pre-calculated)
- **Same strategy code** as backtesting for consistency
- **Dynamic symbol discovery** from FreqTrade configuration
- **Signal publishing** to Redis with TTL
- **Signal storage** in InfluxDB for history
- **Signal cooldown** to prevent spam
- **Performance monitoring** with cycle tracking

### ✅ Configuration Management
- **Multi-source configuration**: YAML files, environment variables, defaults
- **FreqTrade integration**: Automatic pair discovery from FreqTrade config
- **Flexible timeframes**: Support for all major timeframes (1m, 5m, 15m, 30m, 1h, 4h)
- **Environment-specific settings**: Development vs production modes

### ✅ Error Handling & Monitoring
- **Graceful error handling**: Service continues running despite individual symbol failures
- **Connection health checks**: Redis and InfluxDB connectivity validation
- **Performance statistics**: Cycle timing, signal counts, error tracking
- **Comprehensive logging**: Configurable log levels with structured output

## Signal Format

Signals published to Redis for FreqTrade consumption:

```json
{
    "signal_id": "unique-uuid",
    "timestamp": "2025-01-01T00:00:00Z",
    "symbol": "BTC",
    "action": "LONG/SHORT/CLOSE",
    "score": 7.5,
    "position_size_factor": 1.0,
    "leverage": 3,
    "entry_price": 50000,
    "ttl": 300,
    "confidence": 0.85,
    "reasoning": "SqueezeFlow Phase 4 entry signal",
    "strategy": "SqueezeFlowStrategy",
    "service": "strategy_runner"
}
```

### Redis Keys
- **Individual signals**: `squeezeflow:signal:{SYMBOL}`
- **Signal channel**: `squeezeflow:signals` (pub/sub)

### InfluxDB Storage
- **Measurement**: `strategy_signals`
- **Tags**: symbol, action, strategy, service
- **Fields**: All signal data for historical analysis

## Installation & Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r services/requirements.txt

# Or install individually
pip install redis>=4.5.0 influxdb>=5.3.0 pyyaml>=6.0
```

### 2. Configuration

Create configuration file (optional):
```bash
python run_strategy_runner.py --create-config
```

Edit `services/config/service_config.yaml`:
```yaml
run_interval_seconds: 60
max_symbols_per_cycle: 10
data_lookback_hours: 48
default_timeframe: '5m'
redis_host: 'localhost'
redis_port: 6379
influx_host: 'localhost'
influx_port: 8086
log_level: 'INFO'
```

### 3. Environment Variables (Optional)

```bash
export SQUEEZEFLOW_RUN_INTERVAL=60
export SQUEEZEFLOW_MAX_SYMBOLS=10
export SQUEEZEFLOW_TIMEFRAME=5m
export REDIS_HOST=localhost
export INFLUX_HOST=localhost
export SQUEEZEFLOW_LOG_LEVEL=INFO
```

## Usage

### Start Service
```bash
# Default configuration
python run_strategy_runner.py

# Custom configuration directory
python run_strategy_runner.py --config /path/to/config

# Direct service execution
python services/strategy_runner.py
```

### Health Checks
```bash
# Check service health
python run_strategy_runner.py --health

# Test signal generation (without publishing)
python run_strategy_runner.py --test-signals
```

### Service Management
```bash
# Background execution
nohup python run_strategy_runner.py > strategy_runner.log 2>&1 &

# Using systemd (production)
sudo systemctl start squeezeflow-strategy-runner
sudo systemctl enable squeezeflow-strategy-runner
```

## Configuration Reference

### Core Settings
- `run_interval_seconds`: How often to run strategy cycles (default: 60)
- `max_symbols_per_cycle`: Maximum symbols to process per cycle (default: 10)
- `data_lookback_hours`: Hours of historical data to load (default: 48)
- `default_timeframe`: Primary analysis timeframe (default: '5m')

### Data Quality
- `min_data_points`: Minimum data points required for analysis (default: 500)
- Required data: OHLCV, spot volume, futures volume, spot CVD, futures CVD

### Signal Management
- `signal_cooldown_minutes`: Cooldown between signals for same symbol (default: 15)
- `redis_signal_ttl`: Redis signal expiration time in seconds (default: 300)
- `max_concurrent_signals`: Maximum active signals at once (default: 5)

### Performance
- `enable_parallel_processing`: Process symbols in parallel (default: true)
- `enable_performance_monitoring`: Track performance metrics (default: true)
- `health_check_interval`: Health check frequency in seconds (default: 30)

## Integration with FreqTrade

### 1. Symbol Discovery
The service automatically discovers trading pairs from FreqTrade configuration:
- Reads `freqtrade/user_data/config.json`
- Extracts `exchange.pair_whitelist`
- Converts to base symbols (e.g., "BTC/USDT:USDT" → "BTC")

### 2. Signal Consumption
FreqTrade should be configured to consume signals from Redis:
- Monitor Redis key pattern: `squeezeflow:signal:*`
- Subscribe to channel: `squeezeflow:signals`
- Implement signal validation and execution logic

### 3. Position Sizing & Leverage
Signals include dynamic position sizing based on SqueezeFlow scoring:
- **Score 8-10**: 1.5x position size, 5x leverage (high confidence)
- **Score 6-7**: 1.0x position size, 3x leverage (medium confidence)  
- **Score 4-5**: 0.5x position size, 2x leverage (low confidence)

## Monitoring & Observability

### Performance Metrics
- `cycles_completed`: Total strategy cycles executed
- `signals_generated`: Total signals published  
- `errors_encountered`: Error count for debugging
- `last_cycle_duration`: Time taken for last cycle
- `avg_cycle_duration`: Rolling average cycle time

### Health Status Endpoint
```python
# Get service health
runner = StrategyRunner()
health = runner.get_health_status()
print(json.dumps(health, indent=2))
```

### Logging
- **Structured logging** with timestamps and levels
- **Configurable verbosity**: DEBUG, INFO, WARNING, ERROR
- **Per-symbol logging** for detailed analysis
- **Performance logging** every 10 cycles

## Troubleshooting

### Common Issues

#### 1. Connection Errors
```bash
# Test Redis connection
redis-cli ping

# Test InfluxDB connection  
curl http://localhost:8086/ping

# Check service health
python run_strategy_runner.py --health
```

#### 2. No Signals Generated
- Verify data quality: Must have OHLCV, spot/futures volume, CVD data
- Check symbol configuration in FreqTrade config
- Review strategy logging for Phase 4 scoring results
- Ensure minimum data points threshold is met

#### 3. Data Quality Issues
```bash  
# Check available symbols in InfluxDB
# Query: SELECT DISTINCT("market") FROM "trades_5m" WHERE time > now() - 1h

# Verify CVD calculation
# Query: SELECT * FROM "trades_5m" WHERE market =~ /BTC/ AND time > now() - 1h LIMIT 10
```

#### 4. Performance Issues
- Reduce `max_symbols_per_cycle` for better performance
- Increase `run_interval_seconds` to reduce CPU usage
- Enable parallel processing for multiple symbols
- Monitor `avg_cycle_duration` in performance stats

### Debug Mode
```bash
# Enable debug logging
export SQUEEZEFLOW_LOG_LEVEL=DEBUG
python run_strategy_runner.py

# Test without publishing
python run_strategy_runner.py --test-signals
```

## Development

### Code Structure
```
services/
├── strategy_runner.py          # Main service implementation
├── config/
│   └── service_config.py      # Configuration management
├── requirements.txt           # Python dependencies
└── README.md                 # This documentation

run_strategy_runner.py         # Service launcher script
```

### Key Methods
- `StrategyRunner.start()`: Main service loop
- `StrategyRunner._process_symbol()`: Process individual symbol
- `StrategyRunner._convert_orders_to_signals()`: Order → Signal conversion
- `ConfigManager.get_freqtrade_pairs()`: Symbol discovery

### Testing
```bash
# Test configuration loading
python -c "from services.config.service_config import ConfigManager; print('OK')"

# Test FreqTrade integration
python -c "from services.config.service_config import ConfigManager; cm = ConfigManager(); print(cm.get_freqtrade_pairs())"

# Test signal generation
python run_strategy_runner.py --test-signals
```

## Production Deployment

### Docker Integration
Add to `docker-compose.yml`:
```yaml
strategy-runner:
  build: .
  container_name: strategy-runner
  command: python services/strategy_runner.py
  depends_on:
    - redis
    - aggr-influx
  environment:
    - REDIS_HOST=redis
    - INFLUX_HOST=aggr-influx
  restart: unless-stopped
```

### Systemd Service
Create `/etc/systemd/system/squeezeflow-strategy-runner.service`:
```ini
[Unit]
Description=SqueezeFlow Strategy Runner Service
After=network.target redis.service influxdb.service

[Service] 
Type=simple
User=squeezeflow
WorkingDirectory=/opt/squeezeflow-trader
ExecStart=/usr/bin/python3 services/strategy_runner.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Monitoring
- **Log rotation**: Configure logrotate for service logs
- **Alerting**: Monitor error rates and signal generation
- **Metrics**: Track performance stats in monitoring system
- **Health checks**: Automated health monitoring with alerting

## Signal Monitor Dashboard Integration

The Strategy Runner Service works seamlessly with the Signal Monitor Dashboard for real-time signal visualization.

**📋 Complete Dashboard Documentation**: [`/services/SIGNAL_MONITOR_DASHBOARD.md`](SIGNAL_MONITOR_DASHBOARD.md)

### Key Dashboard Features
- **Live Signal Feed**: Real-time display of signals published by Strategy Runner
- **Signal Metrics**: Signals per minute, unique symbols, average scores, Long/Short analysis  
- **Quality Monitoring**: High-quality signal tracking (≥7 score) with score distribution
- **Symbol Activity**: Bar charts showing most active trading symbols with signal counts
- **System Health**: Redis connection status and Strategy Runner health monitoring
- **Trading Control Tower**: Visual terminal dashboard with professional ASCII interface

### Quick Start Commands
```bash
# Start the signal monitor dashboard
docker-compose up -d signal-monitor-dashboard

# View the live interactive dashboard (Trading Control Tower)
docker attach squeezeflow-signal-dashboard

# Detach safely without stopping (Ctrl+P, then Ctrl+Q)
# Monitor dashboard logs
docker-compose logs -f signal-monitor-dashboard
```

### Integration Benefits
- **Real-time Validation**: Immediately see signals as they're generated by Strategy Runner
- **Quality Monitoring**: Track signal scores, distribution patterns, and success rates
- **System Visibility**: Monitor Strategy Runner health, Redis connectivity, and performance
- **Debug Support**: Visual feedback for troubleshooting signal generation and flow issues
- **Trading Focus**: Specialized monitoring for trading signals vs general system monitoring

## Future Enhancements

### Planned Features
- **WebSocket data feeds**: Real-time data streaming
- **Multi-strategy support**: Run multiple strategies simultaneously  
- **Advanced signal routing**: Route signals to multiple exchanges
- **ML model integration**: Enhanced signal scoring with ML models
- **Risk management**: Dynamic position sizing based on portfolio state

### API Interface
- **REST API**: Service control and status endpoints
- **WebSocket API**: Real-time signal streaming
- **Metrics API**: Performance and health metrics
- **Configuration API**: Dynamic configuration updates

---

*The Strategy Runner Service is a critical component of the SqueezeFlow Trader system, enabling live trading with the same proven strategy logic used in backtesting. Combined with the Signal Monitor Dashboard, it provides complete visibility into the signal generation pipeline.*