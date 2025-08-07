# Configuration Guide

## Environment Variables

The SqueezeFlow Trader system is configured through environment variables defined in `docker-compose.yml`. This document provides a comprehensive reference for all configuration options.

## Core Services Configuration

### Strategy Runner Service

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SQUEEZEFLOW_RUN_INTERVAL` | integer | `60` | Strategy execution interval in seconds |
| `SQUEEZEFLOW_MAX_SYMBOLS` | integer | `5` | Maximum number of symbols to process per cycle |
| `SQUEEZEFLOW_LOOKBACK_HOURS` | integer | `4` | Hours of historical data to load for analysis |
| `SQUEEZEFLOW_TIMEFRAME` | string | `5m` | Default data timeframe (1m, 5m, 15m, 30m, 1h, 4h) |
| `SQUEEZEFLOW_MIN_DATA_POINTS` | integer | `40` | Minimum required data points for strategy execution |
| `SQUEEZEFLOW_LOG_LEVEL` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Database Connections

#### Redis Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_HOST` | string | `redis` | Redis server hostname |
| `REDIS_PORT` | integer | `6379` | Redis server port |
| `REDIS_DB` | integer | `0` | Redis database index |

#### InfluxDB Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `INFLUX_HOST` | string | `aggr-influx` | InfluxDB server hostname |
| `INFLUX_PORT` | integer | `8086` | InfluxDB server port |
| `INFLUX_DATABASE` | string | `significant_trades` | Target database name |
| `INFLUX_USERNAME` | string | ` ` | Authentication username (optional) |
| `INFLUX_PASSWORD` | string | ` ` | Authentication password (optional) |

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

### Development Environment

```yaml
environment:
  - SQUEEZEFLOW_RUN_INTERVAL=60
  - SQUEEZEFLOW_MAX_SYMBOLS=5
  - SQUEEZEFLOW_LOG_LEVEL=DEBUG
  - REDIS_HOST=redis
  - INFLUX_HOST=aggr-influx
  - FREQTRADE_ENABLE_INTEGRATION=true
```

### Production Environment

```yaml
environment:
  - SQUEEZEFLOW_RUN_INTERVAL=300
  - SQUEEZEFLOW_MAX_SYMBOLS=10
  - SQUEEZEFLOW_LOG_LEVEL=INFO
  - REDIS_HOST=redis-cluster
  - INFLUX_HOST=influx-prod
  - FREQTRADE_ENABLE_INTEGRATION=true
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