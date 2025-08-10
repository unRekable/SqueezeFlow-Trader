# Distributed SqueezeFlow Trader Setup

## Overview

This document describes the distributed architecture setup where:
- **Server**: Runs 24/7 data collection (aggr-server + InfluxDB)
- **MacBook**: Development environment connecting to remote InfluxDB

## Architecture

```
┌─────────────────┐    ┌─────────────────────────┐
│     SERVER      │    │       MACBOOK           │
├─────────────────┤    ├─────────────────────────┤
│ aggr-server     │    │ Strategy Runner         │
│ InfluxDB        │ ◄──┤ FreqTrade              │
│ Chronograf      │    │ Health Monitor          │
│                 │    │ Performance Monitor     │
│ Data: 130k+/hr  │    │ Redis (local)          │
└─────────────────┘    └─────────────────────────┘
```

## Server Setup (VPS/Dedicated Server)

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/SqueezeFlow-Trader.git
cd SqueezeFlow-Trader
```

### 2. Setup aggr-server

```bash
# Clone aggr-server
git clone https://github.com/Tucsky/aggr-server.git aggr-server

# Configure for 1-second data collection
cd aggr-server
nano config.json
```

**Critical Configuration Changes in `aggr-server/config.json`:**
```json
{
  "influxMeasurement": "trades_1s",    # 1-second measurement
  "influxTimeframe": 1000,             # 1000ms = 1 second
  "influxHost": "aggr-influx",         # Docker service name
  "influxPort": 8086,
  "influxDatabase": "significant_trades"
}
```

### 3. Start Server Services

```bash
cd ..
docker-compose -f docker-compose.server.yml up -d
```

### 4. Verify Data Collection

```bash
# Check services
docker ps

# Verify InfluxDB connectivity
curl http://localhost:8086/ping

# Check data collection (should show 20k+ points)
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 10m" -database significant_trades
```

### 5. Setup 1-Minute Backup Aggregation (Recommended)

```bash
# Create continuous queries for 1s -> 1m aggregation
./scripts/setup_1m_continuous_queries.sh

# Monitor the aggregation process
./scripts/monitor_1m_aggregation.sh
```

### 5. Configure Firewall (if needed)

```bash
# Allow InfluxDB access
sudo ufw allow 8086/tcp

# Optional: Allow Chronograf UI
sudo ufw allow 8888/tcp
```

## MacBook Setup (Development Environment)

### 1. Configure Connection

```bash
# Copy environment template
cp .env.example .env

# Edit with your server IP
nano .env
```

**Set in `.env`:**
```bash
SERVER_IP=your.server.ip.here  # Your actual server IP
```

### 2. Test Remote Connection

```bash
# Test connectivity
./scripts/setup_distributed.sh test-connection

# Should output:
# ✓ InfluxDB is reachable
# ✓ Database 'significant_trades' exists
# ✓ Recent data found in database
```

### 3. Start Local Services

```bash
# Stop any existing local services
docker-compose down

# Start services with remote InfluxDB connection
docker-compose -f docker-compose.local.yml up -d
```

### 4. Verify Setup

```bash
# Check local services
docker ps

# Test direct Python connection
python3 -c "
from influxdb import InfluxDBClient
client = InfluxDBClient(host='YOUR_SERVER_IP', port=8086, database='significant_trades')
result = client.query('SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 10m')
data = list(result)
if data:
    count = data[0][0]['count_close']
    print(f'✅ Connected! {count} data points in last 10min')
"
```

## Data Storage Details

### Retention Policies

- **aggr_1s**: Infinite retention (0s duration = never expires)
- **Location**: `significant_trades.aggr_1s.trades_1s`
- **Data Rate**: 130,000+ points per hour (22k+ per 10 minutes)
- **Granularity**: 1-second intervals for ultra-low latency

### Query Examples

```bash
# Count recent data
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 1h" -database significant_trades

# Show latest trades
docker exec aggr-influx influx -execute "SELECT * FROM \"aggr_1s\".\"trades_1s\" ORDER BY time DESC LIMIT 5" -database significant_trades

# Check symbols being collected
docker exec aggr-influx influx -execute "SHOW TAG VALUES FROM \"aggr_1s\".\"trades_1s\" WITH KEY = market" -database significant_trades
```

## Performance Metrics

### Server Performance
- **Data Collection**: 130,000+ points/hour consistently
- **Latency**: Sub-second data aggregation
- **Storage**: ~10-20GB per symbol per week
- **Uptime**: 24/7 reliable data collection

### Network Performance
- **Ping Latency**: 70-140ms (MacBook ↔ Server)
- **Query Response**: <500ms for typical queries
- **Data Transfer**: Minimal overhead (compressed queries)

## Configuration Files

### docker-compose.server.yml
- Runs aggr-server and InfluxDB on server
- Optimized for 24/7 data collection
- Higher resource allocations for reliability

### docker-compose.local.yml
- Runs development services on MacBook
- Connects to remote InfluxDB via SERVER_IP
- Local Redis for caching and messaging

### Environment Variables

**Server Environment:**
```yaml
INFLUX_HOST: aggr-influx
INFLUX_PORT: 8086
INFLUXDB_HTTP_AUTH_ENABLED: false
```

**MacBook Environment:**
```yaml
INFLUX_HOST: ${SERVER_IP}  # Points to remote server
REDIS_HOST: redis          # Local Redis instance
```

## Troubleshooting

### Connection Issues

```bash
# Test basic connectivity
ping YOUR_SERVER_IP

# Test InfluxDB port
nc -zv YOUR_SERVER_IP 8086

# Check firewall on server
sudo ufw status
```

### Data Issues

```bash
# Check if using correct retention policy
docker exec aggr-influx influx -execute "SHOW RETENTION POLICIES ON significant_trades"

# Verify measurement exists
docker exec aggr-influx influx -execute "SHOW MEASUREMENTS" -database significant_trades
```

### Service Issues

```bash
# Server logs
docker logs aggr-server --tail 50
docker logs aggr-influx --tail 20

# MacBook logs
docker logs squeezeflow-strategy-runner --tail 20
```

## Security Considerations

### Current Setup (Development)
- Direct IP connection (your.server.ip:8086)
- No authentication on InfluxDB
- Firewall-protected server

### Production Recommendations
1. **VPN/Tailscale**: Private network connection
2. **InfluxDB Auth**: Enable username/password
3. **TLS/SSL**: Encrypt connections
4. **IP Restrictions**: Limit access to known IPs

## Monitoring

### Health Checks

```bash
# Server health
curl http://YOUR_SERVER_IP:8086/ping

# Data freshness (should be < 10 seconds)
docker exec aggr-influx influx -execute "SELECT * FROM \"aggr_1s\".\"trades_1s\" ORDER BY time DESC LIMIT 1" -database significant_trades

# Service status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Alerts Setup
- Monitor server disk space (InfluxDB grows ~10-20GB/week per symbol)
- Alert on data collection gaps (>60 seconds without new data)
- Monitor network connectivity between MacBook and server

## Benefits

✅ **Reliability**: 24/7 data collection on dedicated server  
✅ **Performance**: MacBook resources freed for development  
✅ **Flexibility**: Develop locally with production-grade data  
✅ **Scalability**: Easy to add more development machines  
✅ **Backup**: Natural disaster recovery (data on remote server)

## Commands Quick Reference

```bash
# Server Status
docker-compose -f docker-compose.server.yml ps

# MacBook Status  
docker-compose -f docker-compose.local.yml ps

# Connection Test
./scripts/setup_distributed.sh test-connection

# Data Check
curl -G "http://YOUR_SERVER_IP:8086/query" \
  --data-urlencode "db=significant_trades" \
  --data-urlencode 'q=SELECT COUNT(*) FROM "aggr_1s"."trades_1s" WHERE time > now() - 1m'
```

This distributed setup provides enterprise-grade reliability while maintaining development flexibility.