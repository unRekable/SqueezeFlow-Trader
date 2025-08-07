# Docker Microservices Architecture

## Overview

The SqueezeFlow Trader system operates as a containerized microservices architecture using Docker Compose. This document provides comprehensive documentation of all Docker services, their configurations, networking, and deployment patterns.

## Service Topology Diagram

```mermaid
graph TD
    subgraph "SqueezeFlow Network (Bridge Driver)"
        subgraph "Data Layer"
            A["• Redis<br/>• InfluxDB<br/>• aggr-server"]
        end
        
        subgraph "Processing Layer"
            B["• strategy-runner<br/>• signal-validator<br/>• influx-signal-manager"]
        end
        
        subgraph "Execution Layer"
            C["• freqtrade<br/>• freqtrade-ui"]
        end
        
        subgraph "Monitoring Layer"
            D["• system-monitor<br/>• health-monitor"]
        end
        
        subgraph "Support Services"
            E["• oi-tracker<br/>• chronograf"]
        end
        
        A <--> B
        B <--> C
        A --> D
        A --> E
    end
```

## Core Services Configuration

### 1. Redis Service

**Purpose:** Caching, message queue, and signal publishing  
**Image:** `redis:7-alpine`  
**Key Features:** Persistent storage, high-performance caching  

```yaml
redis:
  container_name: squeezeflow-redis
  image: redis:7-alpine
  restart: unless-stopped
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  command: redis-server --appendonly yes
  networks:
    - squeezeflow_network
  deploy:
    resources:
      limits:
        memory: 256M
        cpus: '0.3'
      reservations:
        memory: 128M
        cpus: '0.1'
```

**Configuration Details:**
- **Persistence:** AOF (Append Only File) enabled for data durability
- **Memory Management:** Limited to 256MB with LRU eviction policy
- **Network:** Connected to internal squeezeflow_network
- **Data Volume:** Persistent redis_data volume for data storage

**Redis Configuration (Internal):**
```bash
# Redis configuration applied via command
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

**Performance Tuning:**
- Connection pooling enabled (20 max connections)
- Pipeline operations for batch signal publishing
- Pub/sub channels for real-time communication
- TTL-based expiration for signal cleanup

### 2. InfluxDB Service

**Purpose:** Time-series database for market data and analytics  
**Image:** `influxdb:1.8.10`  
**Key Features:** Multi-timeframe continuous queries, 30-day retention  

```yaml
aggr-influx:
  container_name: aggr-influx
  image: influxdb:1.8.10
  restart: unless-stopped
  ports:
    - "8086:8086"
  volumes:
    - influxdb_data:/var/lib/influxdb
  networks:
    - squeezeflow_network
    - aggr_backend
  deploy:
    resources:
      limits:
        cpus: '0.5'
        memory: 4096M
      reservations:
        cpus: '0.3'
        memory: 2048M
  environment:
    - INFLUXDB_DB=significant_trades
    - INFLUXDB_ADMIN_USER=admin
    - INFLUXDB_ADMIN_PASSWORD=admin123
```

**Database Schema:**
```sql
-- Primary database for trading data
CREATE DATABASE significant_trades;

-- Retention policies
CREATE RETENTION POLICY "30_days" ON "significant_trades" 
DURATION 30d REPLICATION 1 DEFAULT;

-- Continuous queries for multi-timeframe aggregation
CREATE CONTINUOUS QUERY "cq_5m" ON "significant_trades"
BEGIN
  SELECT first(open) AS open, max(high) AS high, min(low) AS low, last(close) AS close,
         sum(vbuy) AS vbuy, sum(vsell) AS vsell
  INTO "aggr_5m"."trades_5m" FROM "aggr_1m"."trades_1m"
  GROUP BY time(5m), market
END;
```

**Key Measurements:**
- `aggr_1m.trades_1m` - Real-time 1-minute trading data
- `aggr_5m.trades_5m` - 5-minute aggregated data
- `aggr_15m.trades_15m` - 15-minute aggregated data
- `aggr_1h.trades_1h` - 1-hour aggregated data
- `aggr_4h.trades_4h` - 4-hour aggregated data
- `strategy_signals` - Signal tracking and analytics

### 3. aggr-server Service

**Purpose:** Real-time market data collection from 20+ exchanges  
**Build:** Custom Dockerfile from aggr-server directory  
**Key Features:** WebSocket connections, multi-exchange integration  

```yaml
aggr-server:
  container_name: aggr-server
  build:
    context: ./aggr-server
    dockerfile: Dockerfile
  restart: unless-stopped
  ports:
    - "3000:3000"
  environment:
    - PORT=3000
    - WORKDIR=/usr/src/app/
    - FILES_LOCATION=./data
    - INFLUX_HOST=aggr-influx
    - INFLUX_PORT=8086
  volumes:
    - ./aggr-server:/usr/src/app/
  networks:
    - squeezeflow_network
    - aggr_backend
  deploy:
    resources:
      limits:
        cpus: '0.5'
        memory: 1024M
      reservations:
        cpus: '0.3'
        memory: 512M
```

**Exchange Integrations:**
```javascript
// Supported exchanges (20+)
const exchanges = [
  // SPOT exchanges
  'BINANCE', 'COINBASE', 'KRAKEN', 'BITSTAMP', 'GEMINI',
  'KUCOIN', 'HUOBI', 'GATE', 'CRYPTOCOM', 'MEXC',
  
  // FUTURES exchanges  
  'BINANCE_FUTURES', 'BYBIT', 'OKX', 'DERIBIT', 'PHEMEX',
  'BITFINEX', 'BITGET', 'BITMEX', 'FTX', 'HUOBI_FUTURES'
];
```

**Data Flow:**
```mermaid
flowchart LR
    A[WebSocket APIs] --> B[aggr-server]
    B --> C[Data Processing]
    C --> D[InfluxDB Storage]
    
    A1["20+ Exchanges"] -.-> A
    B1["Trade Parsing"] -.-> B
    C1["Volume Calc"] -.-> C
    D1["Time-series DB"] -.-> D
```

**Performance Metrics:**
- Real-time latency: < 100ms
- Data throughput: ~1000 trades/second
- Memory usage: ~512MB baseline
- CPU usage: ~30% under normal load

### 4. Strategy Runner Service

**Purpose:** Core signal generation service running SqueezeFlow strategy  
**Build:** Custom Dockerfile.strategy-runner  
**Key Features:** Live CVD calculation, signal validation, Redis publishing  

```yaml
strategy-runner:
  container_name: squeezeflow-strategy-runner
  build:
    context: .
    dockerfile: docker/Dockerfile.strategy-runner
  restart: unless-stopped
  environment:
    - SQUEEZEFLOW_RUN_INTERVAL=60
    - SQUEEZEFLOW_MAX_SYMBOLS=5
    - SQUEEZEFLOW_LOOKBACK_HOURS=48
    - SQUEEZEFLOW_TIMEFRAME=5m
    - SQUEEZEFLOW_LOG_LEVEL=INFO
    - REDIS_HOST=redis
    - REDIS_PORT=6379
    - REDIS_DB=0
    - INFLUX_HOST=aggr-influx
    - INFLUX_PORT=8086
    - INFLUX_DATABASE=significant_trades
    - FREQTRADE_API_URL=http://freqtrade:8080
    - FREQTRADE_API_USERNAME=squeezeflow
    - FREQTRADE_API_PASSWORD=squeezeflow123
    - FREQTRADE_ENABLE_INTEGRATION=true
  depends_on:
    - redis
    - aggr-influx
  volumes:
    - ./freqtrade/user_data/config.json:/app/freqtrade/user_data/config.json:ro
  networks:
    - squeezeflow_network
    - aggr_backend
  healthcheck:
    test: ["CMD", "python", "-c", "import redis; r=redis.Redis(host='redis', port=6379); r.ping()"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s
  deploy:
    resources:
      limits:
        memory: 1G
        cpus: '0.5'
      reservations:
        memory: 512M
        cpus: '0.25'
```

**Service Architecture:**
```python
# Dockerfile.strategy-runner
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/logs data/session_stats

# Set Python path
ENV PYTHONPATH=/app

# Run strategy runner service
CMD ["python", "services/strategy_runner.py"]
```

**Environment Variables:**
- `SQUEEZEFLOW_RUN_INTERVAL`: Strategy execution frequency (60 seconds)
- `SQUEEZEFLOW_MAX_SYMBOLS`: Maximum symbols per cycle (5)
- `SQUEEZEFLOW_LOOKBACK_HOURS`: Historical data window (48 hours)
- `SQUEEZEFLOW_TIMEFRAME`: Primary timeframe (5m)
- Redis/InfluxDB connection parameters
- FreqTrade API integration settings

### 5. FreqTrade Service

**Purpose:** Trading execution engine with FreqAI support  
**Build:** Custom Dockerfile.freqtrade  
**Key Features:** Pure execution layer, API integration, web UI  

```yaml
freqtrade:
  container_name: squeezeflow-freqtrade
  build:
    context: .
    dockerfile: docker/Dockerfile.freqtrade
  restart: unless-stopped
  ports:
    - "8080:8080"
  environment:
    - FREQTRADE_UI_PASSWORD=0xGang
  volumes:
    - freqtrade_data:/freqtrade/user_data
    - ./freqtrade/user_data:/freqtrade/user_data
    - ./freqtrade/config/config.json:/freqtrade/config/config.json:ro
  depends_on:
    - redis
    - strategy-runner
  networks:
    - squeezeflow_network
  deploy:
    resources:
      limits:
        memory: 2G
      reservations:
        memory: 1G
```

**FreqTrade Configuration:**
```json
{
  "max_open_trades": 5,
  "stake_currency": "USDT",
  "stake_amount": 100,
  "tradable_balance_ratio": 0.99,
  "fiat_display_currency": "USD",
  "timeframe": "5m",
  "dry_run": true,
  "cancel_open_orders_on_exit": false,
  
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": "",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": [
      "BTC/USDT:USDT",
      "ETH/USDT:USDT"
    ],
    "pair_blacklist": []
  },
  
  "api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "verbosity": "info",
    "username": "squeezeflow",
    "password": "squeezeflow123"
  }
}
```

### 6. FreqTrade UI Service

**Purpose:** Web-based trading interface and monitoring dashboard  
**Image:** `freqtradeorg/frequi:latest`  
**Key Features:** Real-time trade monitoring, portfolio visualization  

```yaml
freqtrade-ui:
  container_name: squeezeflow-freqtrade-ui
  image: freqtradeorg/frequi:latest
  restart: unless-stopped
  ports:
    - "8081:8080"
  environment:
    - VITE_API_URL=http://localhost:8080
    - VITE_API_USERNAME=squeezeflow
    - VITE_API_PASSWORD=squeezeflow123
  depends_on:
    - freqtrade
  networks:
    - squeezeflow_network
```

**UI Features:**
- Real-time trade monitoring
- Portfolio performance analytics
- Strategy performance visualization
- Signal history and analysis
- System health monitoring
- Manual trade management

### 7. Open Interest Tracker

**Purpose:** Real-time open interest data collection and analysis  
**Build:** Custom Dockerfile.oi-tracker  
**Key Features:** OI data aggregation, futures market analysis  

```yaml
oi-tracker:
  container_name: squeezeflow-oi-tracker
  build:
    context: .
    dockerfile: docker/Dockerfile.oi-tracker
  restart: unless-stopped
  environment:
    - INFLUX_HOST=aggr-influx
    - INFLUX_PORT=8086
    - INFLUX_DATABASE=significant_trades
    - REDIS_URL=redis://redis:6379
  depends_on:
    - redis
    - aggr-influx
  networks:
    - squeezeflow_network
```

**OI Data Collection:**
```python
# OI tracking for major futures exchanges
oi_exchanges = [
    'BINANCE_FUTURES',
    'BYBIT', 
    'OKX',
    'DERIBIT',
    'PHEMEX'
]

# OI measurement schema
measurement = "open_interest"
tags = {
    "symbol": "BTCUSDT",
    "exchange": "BINANCE_FUTURES"
}
fields = {
    "open_interest": 1234567890.0,
    "oi_change_24h": 0.05,
    "oi_change_pct": 5.2
}
```

### 8. System Monitor Service

**Purpose:** System-wide health monitoring and alerting  
**Build:** Custom Dockerfile.monitor  
**Key Features:** Docker container monitoring, resource tracking, alerting  

```yaml
system-monitor:
  container_name: squeezeflow-monitor
  build:
    context: .
    dockerfile: docker/Dockerfile.monitor
  restart: unless-stopped
  environment:
    - INFLUX_HOST=aggr-influx
    - INFLUX_PORT=8086
    - INFLUX_DATABASE=significant_trades
  depends_on:
    - aggr-influx
  networks:
    - squeezeflow_network
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - /proc:/host/proc:ro
    - /sys:/host/sys:ro
```

**Monitoring Capabilities:**
- Docker container health checks
- Resource utilization tracking (CPU, Memory, Disk)
- Service availability monitoring
- Performance metrics collection
- Alert generation for system issues

### 9. Chronograf Service

**Purpose:** InfluxDB admin UI and data visualization  
**Image:** `chronograf:latest`  
**Key Features:** Database administration, query interface, dashboards  

```yaml
aggr-chronograf:
  container_name: aggr-chronograf
  image: chronograf:latest
  restart: unless-stopped
  volumes:
    - ./aggr-server/data/chronograf:/var/lib/chronograf
  ports:
    - '8885:8888'
  environment:
    - 'INFLUXDB_URL=http://aggr-influx:8086'
  depends_on:
    - aggr-influx
  networks:
    - squeezeflow_network
    - aggr_backend
```

## Network Configuration

### 1. Network Topology
```yaml
networks:
  squeezeflow_network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
  
  aggr_backend:
    external: true
    # Connects to existing aggr-server network
```

### 2. Service Communication Matrix
```mermaid
graph TD
    subgraph "Service Communication Matrix"
        subgraph "Services"
            SR[strategy-runner]
            FT[freqtrade]
            FTU[freqtrade-ui]
            OI[oi-tracker]
            SM[system-monitor]
            AS[aggr-server]
        end
        
        subgraph "Infrastructure"
            R[Redis]
            IDB[InfluxDB]
            AGS[aggr-srv]
            FTA[FreqTrade API]
        end
        
        SR -->|R/W| R
        SR -->|R/W| IDB
        SR -->|R| FTA
        
        FT -->|R| R
        
        FTU -->|API| FTA
        
        OI -->|R/W| R
        OI -->|W| IDB
        
        SM -->|R| R
        SM -->|W| IDB
        SM -->|API| FTA
        
        AS -->|W| IDB
    end
    
    subgraph "Legend"
        L1["R=Read, W=Write, API=REST API"]
    end
```

### 3. Port Mapping
```bash
# External port mappings
3000  → aggr-server (market data API)
6379  → redis (cache and messaging)
8080  → freqtrade (API server)
8081  → freqtrade-ui (web interface)
8086  → influxdb (database)
8885  → chronograf (database admin)

# Internal communication uses service names
redis:6379           # Internal Redis access
aggr-influx:8086     # Internal InfluxDB access
freqtrade:8080       # Internal FreqTrade API
```

## Volume Management

### 1. Persistent Volumes
```yaml
volumes:
  influxdb_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/squeezeflow/influxdb
  
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/squeezeflow/redis
  
  freqtrade_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/squeezeflow/freqtrade
```

### 2. Configuration Volumes
```yaml
# Read-only configuration mounts
- ./freqtrade/user_data/config.json:/freqtrade/config/config.json:ro
- ./freqtrade/user_data:/freqtrade/user_data
- ./aggr-server:/usr/src/app/
```

### 3. Backup Strategy
```bash
#!/bin/bash
# backup_volumes.sh

# InfluxDB backup
docker exec aggr-influx influxd backup -portable /backup
docker cp aggr-influx:/backup ./backups/influxdb-$(date +%Y%m%d)

# Redis backup  
docker exec squeezeflow-redis redis-cli BGSAVE
docker cp squeezeflow-redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb

# FreqTrade data backup
docker cp squeezeflow-freqtrade:/freqtrade/user_data ./backups/freqtrade-$(date +%Y%m%d)
```

## Service Health Checks

### 1. Application Health Checks
```yaml
# Strategy Runner health check
healthcheck:
  test: ["CMD", "python", "-c", "import redis; r=redis.Redis(host='redis', port=6379); r.ping()"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s

# FreqTrade health check  
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/api/v1/ping"]
  interval: 30s
  timeout: 10s
  retries: 3

# InfluxDB health check
healthcheck:
  test: ["CMD", "influx", "-execute", "SHOW DATABASES"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 2. Custom Health Check Scripts
```python
#!/usr/bin/env python3
# health_check.py

import redis
import requests
from influxdb import InfluxDBClient

def check_redis():
    try:
        r = redis.Redis(host='redis', port=6379)
        r.ping()
        return True
    except:
        return False

def check_influxdb():
    try:
        client = InfluxDBClient(host='aggr-influx', port=8086)
        client.ping()
        return True
    except:
        return False

def check_freqtrade():
    try:
        response = requests.get('http://freqtrade:8080/api/v1/ping', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_strategy_runner():
    try:
        r = redis.Redis(host='redis', port=6379)
        # Check for recent strategy runner activity
        last_signal = r.get('squeezeflow:last_activity')
        return last_signal is not None
    except:
        return False

if __name__ == "__main__":
    services = {
        'redis': check_redis(),
        'influxdb': check_influxdb(),
        'freqtrade': check_freqtrade(),
        'strategy_runner': check_strategy_runner()
    }
    
    all_healthy = all(services.values())
    print(f"System Health: {'HEALTHY' if all_healthy else 'DEGRADED'}")
    
    for service, status in services.items():
        print(f"{service}: {'UP' if status else 'DOWN'}")
```

## Resource Management

### 1. Resource Limits
```yaml
# Production resource allocation
deploy:
  resources:
    limits:
      cpus: '0.5'        # 50% of one CPU core
      memory: 1024M      # 1GB memory limit
    reservations:
      cpus: '0.25'       # 25% guaranteed CPU
      memory: 512M       # 512MB guaranteed memory
```

### 2. Performance Tuning
```bash
# Docker daemon optimization
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
```

### 3. Monitoring Resource Usage
```python
# Resource monitoring script
import docker
import psutil

client = docker.from_env()

def get_container_stats():
    containers = client.containers.list()
    
    for container in containers:
        if 'squeezeflow' in container.name:
            stats = container.stats(stream=False)
            
            # CPU usage
            cpu_percent = stats['cpu_stats']['cpu_usage']['total_usage']
            
            # Memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100
            
            print(f"{container.name}:")
            print(f"  Memory: {memory_usage/1024/1024:.1f}MB ({memory_percent:.1f}%)")
            print(f"  CPU: {cpu_percent:.1f}%")
```

## Deployment Strategies

### 1. Development Deployment
```bash
# Development mode with hot reloading
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Development overrides
version: '3.8'
services:
  strategy-runner:
    volumes:
      - .:/app
    environment:
      - SQUEEZEFLOW_LOG_LEVEL=DEBUG
      - SQUEEZEFLOW_RUN_INTERVAL=30
    command: python -u services/strategy_runner.py
```

### 2. Production Deployment
```bash
# Production deployment with optimizations
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Production overrides
services:
  strategy-runner:
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 3. Docker Swarm Deployment
```yaml
# docker-compose.swarm.yml
version: '3.8'
services:
  strategy-runner:
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.labels.role == compute
      update_config:
        parallelism: 1
        delay: 30s
      restart_policy:
        condition: on-failure
```

## Security Configuration

### 1. Network Security
```yaml
# Secure network configuration
networks:
  squeezeflow_internal:
    driver: bridge
    internal: true  # No external access
    
  squeezeflow_external:
    driver: bridge
    # Only expose necessary services
```

### 2. Container Security
```dockerfile
# Security-hardened Dockerfile
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r squeezeflow && useradd -r -g squeezeflow squeezeflow

# Install security updates
RUN apt-get update && apt-get upgrade -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Set secure permissions
COPY --chown=squeezeflow:squeezeflow . /app
USER squeezeflow

# Security-focused entrypoint
ENTRYPOINT ["python", "-u", "services/strategy_runner.py"]
```

### 3. Secrets Management
```bash
# Docker secrets for sensitive data
echo "api_key_here" | docker secret create freqtrade_api_key -
echo "password_here" | docker secret create redis_password -

# Use secrets in compose
services:
  freqtrade:
    secrets:
      - freqtrade_api_key
      - redis_password
    environment:
      - API_KEY_FILE=/run/secrets/freqtrade_api_key
```

## Troubleshooting Guide

### 1. Common Issues

**Service Won't Start:**
```bash
# Check service logs
docker-compose logs service-name

# Check resource constraints
docker stats

# Verify network connectivity
docker-compose exec service-name ping other-service
```

**Database Connection Issues:**
```bash
# Test InfluxDB connection
docker-compose exec strategy-runner python -c "
from influxdb import InfluxDBClient
client = InfluxDBClient(host='aggr-influx', port=8086)
print(client.ping())
"

# Test Redis connection
docker-compose exec strategy-runner python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(r.ping())
"
```

**Performance Issues:**
```bash
# Check container resource usage
docker stats --no-stream

# Check system resources
docker system df
docker system prune

# Analyze service performance
docker-compose exec strategy-runner python -c "
from services.performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
print(monitor.get_performance_analytics())
"
```

### 2. Recovery Procedures

**Service Recovery:**
```bash
# Restart individual service
docker-compose restart service-name

# Recreate service with latest image
docker-compose up -d --force-recreate service-name

# Full system restart
docker-compose down && docker-compose up -d
```

**Data Recovery:**
```bash
# Restore InfluxDB from backup
docker-compose exec aggr-influx influxd restore -portable /backup

# Restore Redis from backup
docker cp ./backups/redis-20250806.rdb squeezeflow-redis:/data/dump.rdb
docker-compose restart redis
```

This comprehensive Docker services architecture provides a robust, scalable, and maintainable foundation for the SqueezeFlow Trader system with proper containerization, networking, and deployment strategies.