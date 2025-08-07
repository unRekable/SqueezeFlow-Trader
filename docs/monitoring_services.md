# Monitoring Services

The SqueezeFlow Trader includes comprehensive monitoring services for system health, performance tracking, and alerting.

## Services Overview

### Health Monitor Service (Enhanced)
Provides HTTP health check endpoints and monitors system components with auto-recovery capabilities.

**Features:**
- HTTP health endpoints for Docker health checks
- Service dependency monitoring (Redis, InfluxDB, Strategy Runner, Aggr-Server)
- **NEW: Data flow monitoring** - Detects when aggr-server stops writing to InfluxDB
- **NEW: Auto-restart capability** - Automatically restarts stuck services (max 3 times/hour)
- System resource monitoring (CPU, Memory, Disk)
- Real-time alerting system
- Prometheus-compatible metrics
- **NEW: Write success tracking** - Monitors InfluxDB write rates and data gaps

**Endpoints (Port 8090):**
- `GET /health` - Basic health check (200 if healthy, 503 if unhealthy)
- `GET /health/detailed` - Comprehensive health report with all services
- `GET /health/service/{name}` - Individual service health check
- `GET /metrics` - Prometheus-style metrics
- `GET /status` - System status summary

### Performance Monitor Service
Tracks detailed performance metrics and identifies bottlenecks.

**Features:**
- System metrics collection (CPU, memory, disk, network)
- Operation timing with context managers
- Custom performance alerts
- Memory profiling with tracemalloc
- Real-time dashboard in Redis
- Performance chart generation

## Quick Start

### Starting Monitoring Services

```bash
# Using the startup script
./scripts/start_monitoring.sh

# Or using docker-compose directly
docker-compose up -d health-monitor performance-monitor
```

### Checking System Health

```bash
# Basic health check
curl http://localhost:8090/health

# Detailed health report
curl http://localhost:8090/health/detailed | jq

# Check specific service
curl http://localhost:8090/health/service/redis
curl http://localhost:8090/health/service/influxdb
curl http://localhost:8090/health/service/strategy_runner

# Get Prometheus metrics
curl http://localhost:8090/metrics
```

### Monitoring via Redis

```bash
# Connect to Redis
redis-cli

# Get performance dashboard
GET squeezeflow:dashboard:performance

# Subscribe to alerts
SUBSCRIBE squeezeflow:alerts
SUBSCRIBE squeezeflow:performance_alerts

# View performance metrics
KEYS squeezeflow:metrics:*
```

## Data Flow Monitoring

The enhanced Health Monitor now tracks data flow from aggr-server to InfluxDB:

### Automatic Detection
- Monitors data write rates (expects >100 points per 5 minutes)
- Detects data gaps (alerts if >5 minutes without new data)
- Tracks last write timestamp for each service

### Auto-Recovery Features
When data flow issues are detected:
1. **Aggr-Server**: Automatically restarts if no data written for >10 minutes
2. **Strategy Runner**: Restarts if unhealthy or not generating signals
3. **Restart Limits**: Maximum 3 restarts per hour to prevent loops

### Monitoring Aggr-Server Health

```bash
# Check aggr-server specific health
curl http://localhost:8090/health/service/aggr_server | jq

# Response includes:
# - container_running: true/false
# - recent_data_count: points in last 5 min
# - data_flow_healthy: true/false
# - data_gap_minutes: time since last write
# - last_write_time: timestamp of last data
```

## Integration with Strategy Runner

The monitoring services automatically track the Strategy Runner:

1. **Health Checks**: Verifies Strategy Runner is publishing signals
2. **Performance Tracking**: Monitors signal generation time
3. **Resource Usage**: Tracks memory and CPU usage
4. **Error Tracking**: Logs and alerts on errors
5. **Data Validation**: Ensures sufficient data points for strategy execution

### Adding Custom Metrics

In your strategy or service code:

```python
from services.performance_monitor import PerformanceMonitor, PerformanceTimer

# Initialize monitor
perf_monitor = PerformanceMonitor()

# Use timer context manager
with PerformanceTimer(perf_monitor, "signal_generation"):
    # Your code here
    generate_signals()

# Record custom metric
perf_monitor.record_metric(
    metric_name="signals_generated",
    value=len(signals),
    unit="count",
    tags={"symbol": "BTC"}
)

# Record errors
perf_monitor.record_error(
    operation_name="signal_generation",
    error_message=str(e)
)
```

## Alert Configuration

### Default Alert Thresholds

| Metric | Threshold | Severity |
|--------|-----------|----------|
| CPU Usage | > 80% | Warning |
| Memory Usage | > 85% | Warning |
| Disk Usage | > 90% | Warning |
| Response Time | > 5000ms | Warning |
| Memory Usage | > 1GB | Critical |
| Error Rate | > 5% | Critical |

### Custom Alerts

Add custom alerts in your code:

```python
from services.health_monitor import HealthMonitor, PerformanceAlert

monitor = HealthMonitor()

# Add custom alert
alert = PerformanceAlert(
    metric_name='signal_backlog',
    threshold_value=100,
    comparison='gt',
    duration_seconds=60,
    severity='warning',
    message_template='Signal backlog high: {value} signals'
)

monitor.add_performance_alert(alert)
```

## Docker Health Checks

The health-monitor service includes Docker health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

This enables:
- Automatic container restart on failure
- Docker Swarm/Kubernetes readiness probes
- Load balancer health checks

## Monitoring Dashboard

### Viewing Performance Data

1. **HTTP Dashboard**: Access detailed metrics at `http://localhost:8090/health/detailed`

2. **Redis Dashboard**: Real-time metrics in Redis
   ```bash
   redis-cli GET squeezeflow:dashboard:performance | jq
   ```

3. **Logs**: View service logs
   ```bash
   docker-compose logs -f health-monitor
   docker-compose logs -f performance-monitor
   ```

### Key Metrics Tracked

**System Metrics:**
- CPU usage percentage
- Memory usage (MB/GB)
- Disk usage and free space
- Network I/O
- Process-specific resources

**Service Metrics:**
- Redis response time
- InfluxDB query time
- Strategy Runner signal rate
- API response times
- Error rates per service

**Performance Metrics:**
- Operation durations
- Signal processing time
- Database query performance
- Cache hit rates
- Queue depths

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs health-monitor
docker-compose logs performance-monitor

# Verify dependencies
docker-compose ps redis
docker-compose ps aggr-influx

# Check port availability
lsof -i :8090
```

### No Health Data

```bash
# Verify service is running
docker ps | grep health-monitor

# Check connectivity
curl -v http://localhost:8090/health

# Check Redis connection
redis-cli ping
```

### High Resource Usage Alerts

1. Check system resources: `docker stats`
2. Review service logs for errors
3. Check for memory leaks in performance dashboard
4. Restart services if needed: `docker-compose restart health-monitor`

## Advanced Configuration

### Environment Variables

```bash
# Health Monitor
REDIS_HOST=redis
REDIS_PORT=6379
INFLUX_HOST=aggr-influx
INFLUX_PORT=8086
INFLUX_DATABASE=significant_trades
LOG_LEVEL=INFO
HEALTH_CHECK_INTERVAL=30
HEALTH_MONITOR_PORT=8080

# Performance Monitor
ENABLE_MEMORY_TRACKING=true
DASHBOARD_UPDATE_INTERVAL=30
MAX_METRICS_DATA_POINTS=10000
```

### Scaling Considerations

For production environments:

1. **Separate Monitoring Stack**: Run monitoring on dedicated containers
2. **External Storage**: Store metrics in dedicated InfluxDB instance
3. **Alert Routing**: Integrate with PagerDuty, Slack, or email
4. **Grafana Integration**: Visualize metrics in Grafana dashboards
5. **Log Aggregation**: Send logs to ELK stack or similar

## Metrics Retention

- **Redis Metrics**: 1 hour (configurable)
- **Performance History**: Last 100 checks per service
- **Alert History**: 24 hours
- **Dashboard Data**: Real-time with 30-second updates

## Best Practices

1. **Regular Monitoring**: Check health endpoints periodically
2. **Alert Tuning**: Adjust thresholds based on your system
3. **Resource Limits**: Set Docker resource limits to prevent runaway processes
4. **Log Rotation**: Configure log rotation for long-running services
5. **Backup Metrics**: Periodically export metrics for analysis

## Integration with External Tools

### Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'squeezeflow'
    static_configs:
      - targets: ['localhost:8090']
    metrics_path: '/metrics'
```

### Grafana

Import metrics from Prometheus or directly from InfluxDB for visualization.

### Datadog/New Relic

Use the HTTP endpoints or Redis metrics for integration with APM tools.