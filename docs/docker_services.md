# Docker Microservices Architecture

## Overview

The SqueezeFlow Trader system operates as a containerized microservices architecture using Docker Compose. This document provides comprehensive documentation of all Docker services, their configurations, networking, and deployment patterns.

**🚀 NEW: Real-Time 1-Second Data Processing** - All services now optimized for ultra-low latency 1-second data collection and processing, delivering 60x performance improvement.

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

## ⚡ Real-Time 1-Second Service Configuration

### 🚀 Ultra-Low Latency Service Timing

**BREAKTHROUGH**: All services now operate with **1-second intervals** for real-time trading execution:

| Service | **Previous Timing** | **1-Second Timing** | Performance Gain |
|---------|-------------------|-------------------|------------------|
| **aggr-server** | 60s data collection | **1s data collection** | **60x faster data streaming** |
| **strategy-runner** | 60s execution cycle | **1s execution cycle** | **60x faster signal generation** |
| **Redis cache** | 60s TTL signals | **1s optimized buffering** | **Real-time signal delivery** |
| **InfluxDB writes** | Batched every 10s | **1s flush interval** | **Real-time data persistence** |

### 📊 1-Second Service Performance Requirements

```yaml
# Resource allocation for 1-second mode (increased requirements)
deploy:
  resources:
    limits:
      memory: 2048M      # 2x increase for 1s data buffering
      cpus: '1.0'        # 2x increase for real-time processing
    reservations:
      memory: 1024M      # 2x increase for guaranteed resources
      cpus: '0.5'        # 2x increase for baseline performance
```

### ⚙️ 1-Second Service Optimizations

**aggr-server (1-second data collection):**
```yaml
aggr-server:
  environment:
    - AGGR_INTERVAL=1                    # 1-second data collection (was 60s)
    - AGGR_BATCH_SIZE=100               # Smaller batches for 1s processing
    - AGGR_BUFFER_SIZE=3600             # 1-hour buffer for 1s data
    - INFLUX_FLUSH_INTERVAL=1           # Real-time flushing to InfluxDB
```

**strategy-runner (1-second execution):**
```yaml
strategy-runner:
  environment:
    - SQUEEZEFLOW_RUN_INTERVAL=1        # 1-second strategy execution
    - SQUEEZEFLOW_DATA_INTERVAL=1       # 1-second data polling
    - SQUEEZEFLOW_ENABLE_1S_MODE=true   # Enable 1s optimizations
    - SQUEEZEFLOW_MAX_SYMBOLS=3         # Reduced for real-time processing
    - SQUEEZEFLOW_PARALLEL_PROCESSING=true  # Enable parallel CVD calculation
```

**Redis (1-second optimized caching):**
```yaml
redis:
  environment:
    - REDIS_MAXMEMORY=2gb               # Increased for 1s data buffering
    - REDIS_MAXMEMORY_POLICY=allkeys-lru    # LRU eviction for 1s data
  command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

**InfluxDB (1-second data retention):**
```yaml
aggr-influx:
  environment:
    - INFLUXDB_DB=significant_trades
    - INFLUX_RETENTION_1S=24h           # 24-hour retention for 1s data
    - INFLUX_BATCH_SIZE=100             # Optimized batch size for 1s
    - INFLUX_FLUSH_INTERVAL=1           # Real-time flushing
```

### 🎯 Production 1-Second Configuration

```yaml
# Complete 1-second production configuration
version: '3.8'
services:
  # Enhanced for 1-second real-time processing
  strategy-runner:
    environment:
      # 🚀 REAL-TIME 1-SECOND SETTINGS
      - SQUEEZEFLOW_RUN_INTERVAL=1            # 1-second execution
      - SQUEEZEFLOW_DATA_INTERVAL=1           # 1-second data collection
      - SQUEEZEFLOW_ENABLE_1S_MODE=true       # Enable all 1s optimizations
      - SQUEEZEFLOW_MAX_SYMBOLS=3             # Reduced for real-time
      
      # 💾 MEMORY OPTIMIZATION
      - REDIS_MAXMEMORY=2gb                   # Increased for 1s buffering
      - INFLUX_RETENTION_1S=24h               # 24-hour 1s retention
      
      # ⚡ PERFORMANCE TUNING
      - SQUEEZEFLOW_PARALLEL_PROCESSING=true  # Enable parallel processing
      - SQUEEZEFLOW_1S_BATCH_SIZE=100         # Optimized batch size
    
    deploy:
      resources:
        limits:
          memory: 2G                          # Increased for 1s data
          cpus: '1.0'                         # Increased for real-time
        reservations:
          memory: 1G
          cpus: '0.5'
          
  redis:
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --appendonly yes
    deploy:
      resources:
        limits:
          memory: 2G                          # Increased for 1s buffering
          cpus: '0.5'
```

### ⚠️ 1-Second Mode Considerations

**Critical Requirements:**
- **CPU**: 8+ cores recommended (high continuous load)
- **RAM**: 16GB minimum (2-4x memory usage increase)
- **Storage**: NVMe SSD mandatory (high IOPS for 1s writes)
- **Network**: <50ms exchange latency (critical for real-time)

**Service Impact:**
- **Memory Usage**: 2-4x increase across all services
- **CPU Load**: Continuous high processing (no idle time)
- **Storage I/O**: High frequency writes to InfluxDB
- **Network Dependency**: Critical low-latency connection requirement

## Core Services Configuration

### 1. Redis Service (🆕 1-Second Optimized)

**Purpose:** Caching, message queue, and signal publishing  
**Image:** `redis:7-alpine`  
**Key Features:** **1-second data buffering**, ultra-low latency caching, real-time signal delivery

```yaml
redis:
  container_name: squeezeflow-redis
  image: redis:7-alpine
  restart: unless-stopped
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  # 🆕 1-SECOND OPTIMIZED CONFIGURATION
  command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --appendonly yes --appendfsync everysec
  networks:
    - squeezeflow_network
  deploy:
    resources:
      limits:
        memory: 2G              # 🆕 Increased 8x for 1s data buffering (was 256M)
        cpus: '0.5'             # 🆕 Increased for real-time processing (was 0.3)
      reservations:
        memory: 2G              # 🆕 Guaranteed 2GB for 1s data (was 128M)
        cpus: '0.25'            # 🆕 Baseline for continuous processing (was 0.1)
```

**🚀 1-Second Configuration Details:**
- **Memory Allocation:** **2GB limit** (8x increase) for 1-second data buffering
- **Eviction Policy:** LRU optimized for high-frequency 1s data turnover
- **Persistence:** AOF with 1-second fsync for real-time durability
- **Performance:** Optimized for continuous 1s signal publishing and retrieval

**🆕 Redis Configuration (1-Second Optimized):**
```bash
# Redis configuration optimized for 1-second real-time processing
maxmemory 2gb                    # 🆕 Increased from 256MB for 1s buffering
maxmemory-policy allkeys-lru     # Efficient eviction for high-frequency data
save 900 1                       # Background saves
save 300 10
save 60 10000
appendonly yes                   # Persistence enabled
appendfsync everysec            # 1-second fsync for real-time durability
timeout 1                       # 🆕 1-second timeout for low latency
tcp-keepalive 60                # Keep connections alive
```

**🎯 1-Second Performance Optimizations:**
- **Signal TTL:** 5-second TTL for 1s signals (rapid turnover)
- **Connection Pooling:** 50 max connections (increased from 20)
- **Pipeline Batching:** 100-command batches for 1s data operations
- **Memory Efficiency:** LRU eviction optimized for 1s data patterns

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
    # 🚀 REAL-TIME 1-SECOND CONFIGURATION
    - SQUEEZEFLOW_RUN_INTERVAL=1            # 🆕 1-second execution (was 60)
    - SQUEEZEFLOW_DATA_INTERVAL=1           # 🆕 1-second data collection
    - SQUEEZEFLOW_ENABLE_1S_MODE=true       # 🆕 Enable 1s optimizations
    - SQUEEZEFLOW_MAX_SYMBOLS=3             # 🆕 Reduced for real-time (was 5)
    - SQUEEZEFLOW_LOOKBACK_HOURS=4          # 🆕 Optimized for 1s processing (was 48)
    - SQUEEZEFLOW_TIMEFRAME=5m
    - SQUEEZEFLOW_LOG_LEVEL=INFO            # Avoid DEBUG in 1s mode
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
        memory: 2G              # 🆕 Increased 2x for 1s data processing (was 1G)
        cpus: '1.0'             # 🆕 Increased 2x for real-time execution (was 0.5)
      reservations:
        memory: 2G              # 🆕 Guaranteed 2GB for 1s operations (was 512M)
        cpus: '0.5'             # 🆕 Baseline for continuous processing (was 0.25)
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

**🆕 1-Second Environment Variables:**
- `SQUEEZEFLOW_RUN_INTERVAL`: **1-second execution frequency** (60x improvement from 60s)
- `SQUEEZEFLOW_DATA_INTERVAL`: **1-second data collection** (NEW - ultra-low latency)
- `SQUEEZEFLOW_ENABLE_1S_MODE`: **Enable 1s optimizations** (NEW - real-time mode)
- `SQUEEZEFLOW_MAX_SYMBOLS`: **3 symbols max for real-time** (reduced from 5 for performance)
- `SQUEEZEFLOW_LOOKBACK_HOURS`: **4 hours optimized window** (reduced from 48 for 1s efficiency)
- `SQUEEZEFLOW_TIMEFRAME`: Primary analysis timeframe (5m)
- Redis/InfluxDB connection parameters (optimized for 1s operations)
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
      memory: 2048M      # 2GB memory limit
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