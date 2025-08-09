# Configuration Guide

## Environment Variables

The SqueezeFlow Trader system is configured through environment variables defined in `docker-compose.yml`. This document provides a comprehensive reference for all configuration options.

## Core Services Configuration

### Strategy Runner Service

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SQUEEZEFLOW_RUN_INTERVAL` | integer | `1` | Strategy execution interval in seconds |
| `SQUEEZEFLOW_MAX_SYMBOLS` | integer | `5` | Maximum number of symbols to process per cycle |
| `SQUEEZEFLOW_LOOKBACK_HOURS` | integer | `4` | Hours of historical data to load for analysis |
| `SQUEEZEFLOW_TIMEFRAME` | string | `5m` | Default data timeframe (1m, 5m, 15m, 30m, 1h, 4h) |
| `SQUEEZEFLOW_MIN_DATA_POINTS` | integer | `40` | Minimum required data points for strategy execution |
| `SQUEEZEFLOW_LOG_LEVEL` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### 🆕 Real-Time 1-Second Configuration

**NEW FEATURE**: The system now supports ultra-low latency 1-second data collection and processing, delivering 60x performance improvement.

| Variable | Type | **1s Default** | **Standard Default** | Description |
|----------|------|----------------|---------------------|-------------|
| `SQUEEZEFLOW_RUN_INTERVAL` | integer | **1** | 60 | Strategy execution interval in seconds (1s = real-time) |
| `SQUEEZEFLOW_DATA_INTERVAL` | integer | **1** | 60 | Data collection interval in seconds (1s = ultra-low latency) |
| `SQUEEZEFLOW_ENABLE_1S_MODE` | boolean | **true** | false | Enable 1-second optimizations and processing |
| `SQUEEZEFLOW_MAX_SYMBOLS` | integer | **3** | 5 | Max symbols for real-time (reduced for 1s performance) |
| `REDIS_MAXMEMORY` | string | **2gb** | 2gb | Redis memory limit (increased for 1s data buffering) |
| `INFLUX_RETENTION_1S` | string | **24h** | - | 1-second data retention policy (24h rolling window) |
| `SQUEEZEFLOW_1S_BUFFER_SIZE` | integer | **3600** | - | 1-second data buffer size (3600 = 1 hour buffer) |
| `SQUEEZEFLOW_1S_BATCH_SIZE` | integer | **100** | - | Batch size for 1s data processing optimization |
| `SQUEEZEFLOW_BACKTEST_STEP_SECONDS` | integer | **1** | - | Backtest step size for 1s mode (evaluates every second) |
| `SQUEEZEFLOW_BACKTEST_WINDOW_HOURS` | integer | **1** | 4 | Backtest window size (1h for 1s mode, 4h for regular) |

#### 📊 1-Second Performance Configuration

```yaml
# Real-time 1-second configuration for production
environment:
  # 🚀 ULTRA-LOW LATENCY SETTINGS
  - SQUEEZEFLOW_RUN_INTERVAL=1            # 1-second strategy execution
  - SQUEEZEFLOW_DATA_INTERVAL=1           # 1-second data collection  
  - SQUEEZEFLOW_ENABLE_1S_MODE=true       # Enable all 1s optimizations
  - SQUEEZEFLOW_MAX_SYMBOLS=3             # Reduced for real-time processing
  
  # 💾 MEMORY OPTIMIZATION FOR 1S DATA
  - REDIS_MAXMEMORY=2gb                   # Increased Redis memory
  - REDIS_MAXMEMORY_POLICY=allkeys-lru    # LRU eviction for 1s data
  - INFLUX_RETENTION_1S=24h               # 24-hour 1s data retention
  - SQUEEZEFLOW_1S_BUFFER_SIZE=3600       # 1-hour data buffer
  
  # ⚡ PROCESSING OPTIMIZATIONS  
  - SQUEEZEFLOW_1S_BATCH_SIZE=100         # Batch processing for efficiency
  - SQUEEZEFLOW_PARALLEL_PROCESSING=true  # Enable parallel CVD calculation
  - SQUEEZEFLOW_MEMORY_POOL_SIZE=512MB    # Memory pool for 1s processing
```

#### 🎯 System Requirements for 1-Second Mode

**Minimum Hardware Requirements:**
- **CPU**: 4 cores, 2.5GHz+ (8+ cores recommended for multiple symbols)
- **RAM**: 8GB minimum (16GB recommended for production)
- **Storage**: NVMe SSD mandatory (high IOPS for 1s writes)
- **Network**: <50ms exchange latency (critical for real-time performance)

**Performance Impact:**
- **Memory Usage**: 2-4x increase for 1s data buffering
- **CPU Load**: Significantly higher continuous processing
- **Storage I/O**: High write frequency requires fast storage
- **Network**: Critical dependency on low-latency connections

#### ⚠️ 1-Second Mode Warnings

```yaml
# Critical considerations for 1-second mode
production_warnings:
  memory: "System requires 2-4x more memory than 60s mode"
  cpu: "Continuous high CPU load - ensure adequate cooling"
  storage: "NVMe SSD mandatory - mechanical drives will fail"
  network: "Any network interruption affects real-time performance"
  symbols: "Reduce max symbols to 3-5 for optimal performance"
  monitoring: "Intensive system monitoring required in 1s mode"
```

### Database Connections

#### Redis Configuration

| Variable | Type | Default | **1s Mode** | Description |
|----------|------|---------|------------|-------------|
| `REDIS_HOST` | string | `redis` | `redis` | Redis server hostname |
| `REDIS_PORT` | integer | `6379` | `6379` | Redis server port |
| `REDIS_DB` | integer | `0` | `0` | Redis database index |
| `REDIS_MAXMEMORY` | string | `2gb` | **2gb** | Redis memory limit (increased for 1s buffering) |
| `REDIS_MAXMEMORY_POLICY` | string | `allkeys-lru` | **allkeys-lru** | Memory eviction policy for 1s data |
| `REDIS_TIMEOUT` | integer | `5` | **1** | Connection timeout (reduced for 1s latency) |

#### InfluxDB Configuration

| Variable | Type | Default | **1s Mode** | Description |
|----------|------|---------|------------|-------------|
| `INFLUX_HOST` | string | `aggr-influx` | `aggr-influx` | InfluxDB server hostname |
| `INFLUX_PORT` | integer | `8086` | `8086` | InfluxDB server port |
| `INFLUX_DATABASE` | string | `significant_trades` | `significant_trades` | Target database name |
| `INFLUX_USERNAME` | string | ` ` | ` ` | Authentication username (optional) |
| `INFLUX_PASSWORD` | string | ` ` | ` ` | Authentication password (optional) |
| `INFLUX_RETENTION_1S` | string | `-` | **24h** | 1-second data retention policy (NEW) |
| `INFLUX_BATCH_SIZE` | integer | `1000` | **100** | Batch size for 1s writes (optimized) |
| `INFLUX_FLUSH_INTERVAL` | integer | `10` | **1** | Flush interval seconds (real-time) |

### External Integrations

#### FreqTrade API Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FREQTRADE_API_URL` | string | `http://freqtrade:8080` | FreqTrade API endpoint |
| `FREQTRADE_API_USERNAME` | string | `0xGang` | API authentication username |
| `FREQTRADE_API_PASSWORD` | string | `0xGang` | API authentication password |
| `FREQTRADE_API_TIMEOUT` | integer | `10` | API request timeout in seconds |
| `FREQTRADE_ENABLE_INTEGRATION` | boolean | `true` | Enable/disable FreqTrade integration |

## Service Architecture

### Container Network

All services communicate through the Docker bridge network `squeezeflow_network`. Services reference each other by container name:

- `redis` - Redis cache and message broker
- `aggr-influx` - InfluxDB time-series database
- `freqtrade` - FreqTrade trading bot
- `strategy-runner` - Strategy execution service

### Port Mappings

| Service | Container Port | Host Port | Protocol |
|---------|---------------|-----------|----------|
| Redis | 6379 | 6379 | TCP |
| InfluxDB | 8086 | 8086 | HTTP |
| FreqTrade | 8080 | 8080 | HTTP |
| Health Monitor | 8080 | 8090 | HTTP |
| Chronograf | 8888 | 8885 | HTTP |

## Configuration Examples

### 🆕 Real-Time 1-Second Environment (Recommended)

```yaml
environment:
  # 🚀 ULTRA-LOW LATENCY 1-SECOND CONFIGURATION
  - SQUEEZEFLOW_RUN_INTERVAL=1            # 1-second strategy execution
  - SQUEEZEFLOW_DATA_INTERVAL=1           # 1-second data collection
  - SQUEEZEFLOW_ENABLE_1S_MODE=true       # Enable 1s optimizations
  - SQUEEZEFLOW_MAX_SYMBOLS=3             # Reduced for real-time processing
  - SQUEEZEFLOW_LOG_LEVEL=INFO            # Avoid DEBUG in 1s mode (performance)
  
  # 💾 OPTIMIZED FOR 1-SECOND DATA
  - REDIS_MAXMEMORY=2gb                   # Increased memory for 1s buffering
  - REDIS_MAXMEMORY_POLICY=allkeys-lru    # Efficient eviction policy
  - REDIS_TIMEOUT=1                       # Low-latency connections
  - INFLUX_RETENTION_1S=24h               # 24-hour 1s retention
  - INFLUX_BATCH_SIZE=100                 # Optimized batch size
  - INFLUX_FLUSH_INTERVAL=1               # Real-time flushing
  
  # 🎯 SERVICES CONFIGURATION
  - REDIS_HOST=redis
  - INFLUX_HOST=aggr-influx
  - FREQTRADE_ENABLE_INTEGRATION=true
```

### Development Environment (Standard 1-Second)

```yaml
environment:
  - SQUEEZEFLOW_RUN_INTERVAL=1
  - SQUEEZEFLOW_MAX_SYMBOLS=5
  - SQUEEZEFLOW_LOG_LEVEL=DEBUG
  - REDIS_HOST=redis
  - INFLUX_HOST=aggr-influx
  - FREQTRADE_ENABLE_INTEGRATION=true
```

### Production Environment (Standard)

```yaml
environment:
  - SQUEEZEFLOW_RUN_INTERVAL=300
  - SQUEEZEFLOW_MAX_SYMBOLS=10
  - SQUEEZEFLOW_LOG_LEVEL=INFO
  - REDIS_HOST=redis-cluster
  - INFLUX_HOST=influx-prod
  - FREQTRADE_ENABLE_INTEGRATION=true
```

### 🔧 Migration from 60s to 1s Mode

To migrate an existing system to 1-second mode:

```bash
# 1. Update docker-compose.yml with 1s configuration
# 2. Set up 1s retention policy
./scripts/setup_retention_policy.sh

# 3. Increase system resources
docker-compose down
docker-compose up -d --force-recreate

# 4. Monitor performance
./scripts/monitor_performance.sh

# 5. Verify 1s operation
./scripts/test_implementation.sh
```

## Implementation Details

### Configuration Loading

Services load configuration at startup through the `UnifiedConfig` class:

```python
from services.config.unified_config import get_config

config = get_config()
```

### Environment Variable Precedence

1. Docker Compose environment section
2. System environment variables
3. Default values in code

### Validation

Configuration validation occurs at service startup. Invalid values will prevent service initialization with descriptive error messages.

## Deployment Considerations

### Resource Limits

Each service has defined resource constraints in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
    reservations:
      memory: 512M
      cpus: '0.25'
```

### Health Checks

Services implement health check endpoints:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Logging

All services output structured logs to stdout/stderr, captured by Docker's logging driver. Configure log retention in Docker daemon settings or compose file.

## Security Considerations

### Credentials Management

- Never commit credentials to version control
- Use Docker secrets for sensitive values in production
- Rotate API keys regularly
- Limit network exposure through firewall rules

### Network Isolation

Services communicate through internal Docker networks. Only required ports are exposed to the host system.

## Troubleshooting

### Common Issues

**Service Connection Failures**
- Verify container network connectivity: `docker network inspect squeezeflow_network`
- Check DNS resolution: `docker exec <container> nslookup <service>`
- Validate environment variables: `docker exec <container> env`

**Configuration Not Applied**
- Rebuild containers after changes: `docker-compose build`
- Restart services: `docker-compose restart`
- Check for typos in variable names

**Performance Issues**
- Monitor resource usage: `docker stats`
- Adjust resource limits based on workload
- Scale services horizontally when needed

## Reference

### File Locations

- Configuration: `/docker-compose.yml`
- Service code: `/services/`
- Strategy implementation: `/strategies/`
- Data pipeline: `/data/`

### Related Documentation

- Docker Compose: https://docs.docker.com/compose/
- InfluxDB: https://docs.influxdata.com/influxdb/v1.8/
- Redis: https://redis.io/documentation
- FreqTrade: https://www.freqtrade.io/