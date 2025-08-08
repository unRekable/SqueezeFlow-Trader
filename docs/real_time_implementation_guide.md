# Real-Time 1-Second Data Collection Implementation Guide

## Executive Summary

This guide provides a streamlined implementation for transitioning the SqueezeFlow Trader system from 10-second batched data collection to 1-second real-time data collection. This change improves signal generation speed from ~60 seconds to ~5-10 seconds while maintaining system stability and reasonable resource usage.

**Key Point**: The system currently collects data every 10 seconds and runs strategy every 60 seconds. By changing to 1-second data collection, we enable faster response times and more accurate backtesting through dynamic timeframe aggregation.

## Table of Contents

1. [Prerequisites & Environment Verification](#prerequisites--environment-verification)
2. [aggr-server Configuration](#aggr-server-configuration)
3. [Database Optimization](#database-optimization)
4. [Strategy Runner Adjustments](#strategy-runner-adjustments)
5. [Docker Configuration Updates](#docker-configuration-updates)
6. [Backtesting with 1-Second Data](#backtesting-with-1-second-data)
7. [Testing & Validation](#testing--validation)
8. [Performance Monitoring](#performance-monitoring)
9. [Rollback Procedures](#rollback-procedures)

---

## Prerequisites & Environment Verification

### System Requirements
- Docker & Docker Compose installed
- InfluxDB 1.8 (currently running)
- Node.js 14+ for aggr-server
- Python 3.8+ for strategy runner
- Minimum 2GB RAM, 20GB storage

### Pre-Implementation Checklist

```bash
# 1. Verify InfluxDB is running and accessible
docker exec aggr-influx influx -execute "SHOW DATABASES"

# 2. Backup existing data (CRITICAL!)
docker exec aggr-influx influxd backup -portable /backup
docker cp aggr-influx:/backup ./influxdb_backup_$(date +%Y%m%d)

# 3. Check current data collection interval
docker exec aggr-server grep influxTimeframe /usr/src/app/src/config.js

# 4. Verify aggr-server is running
docker ps | grep aggr-server

# 5. Check available disk space
df -h | grep -E "/$|docker"
```

### Expected Storage Requirements
```
1-second bars (all configured pairs):
- Per day: ~200MB
- Per month: ~6GB
- Per year: ~72GB
```

---

## aggr-server Configuration

### Step 1: Update Configuration File

Edit the aggr-server configuration to collect 1-second data:

```javascript
// File: aggr-server/src/config.js
// Location: Line ~120

// Find this line and change:
influxTimeframe: 10000,  // Current: 10 seconds

// Change to:
influxTimeframe: 1000,   // New: 1 second

// Also update retention multiplier if needed (optional):
influxRetentionPerTimeframe: 10000,  // Keep 10,000 bars per timeframe
```

**Note**: This is the MAIN CHANGE that enables 1-second data collection. The aggr-server will automatically create `trades_1s` measurement in InfluxDB.

### Step 2: Update Backup Interval

To prevent data loss, align backup interval with new timeframe:

```javascript
// File: aggr-server/config.json

{
  // ... existing config ...
  "backupInterval": 1000,  // Changed from 5000ms to 1000ms
  // ... rest of config ...
}
```

### Step 3: Restart aggr-server

```bash
# Restart the service to apply changes
docker-compose restart aggr-server

# Verify the change took effect
docker logs aggr-server --tail 50 | grep "1s"
```

---

## Database Optimization

### Step 1: Create Optimized Retention Policy

```bash
# Connect to InfluxDB
docker exec -it aggr-influx influx

# Switch to the database
USE significant_trades;

# Create retention policy for 1-second data
CREATE RETENTION POLICY "rp_1s" 
  ON "significant_trades" 
  DURATION 30d 
  REPLICATION 1 
  SHARD DURATION 1d 
  DEFAULT;

# Verify creation
SHOW RETENTION POLICIES;

# Exit InfluxDB CLI
exit
```

### Step 2: Optimize InfluxDB Settings

```bash
# Check current memory usage
docker stats aggr-influx --no-stream

# If needed, increase cache size in docker-compose.yml:
# Under aggr-influx service, add environment variables:
environment:
  - INFLUXDB_DATA_CACHE_MAX_MEMORY_SIZE=1g
  - INFLUXDB_DATA_CACHE_SNAPSHOT_MEMORY_SIZE=25m
  - INFLUXDB_DATA_MAX_CONCURRENT_COMPACTIONS=2

# Restart to apply
docker-compose restart aggr-influx
```

---

## Strategy Runner Adjustments

### Step 1: Update Run Interval

Modify the strategy runner to process more frequently:

```yaml
# File: docker-compose.yml

strategy-runner:
  environment:
    # Change from 60 seconds to 5 seconds
    - SQUEEZEFLOW_RUN_INTERVAL=5
```

### Step 2: Optimize Data Queries

Update the strategy to query only recent data:

```python
# File: services/strategy_runner.py

# When fetching data, limit lookback period
def get_recent_data(self, symbol, lookback_minutes=30):
    """
    Fetch only recent data to reduce query load
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=lookback_minutes)
    
    # Query 1-second data
    query = f"""
    SELECT * FROM trades_1s 
    WHERE market = '{symbol}' 
    AND time >= '{start_time.isoformat()}Z'
    AND time <= '{end_time.isoformat()}Z'
    """
    return self.influx_client.query(query)
```

---

## Docker Configuration Updates

### Complete docker-compose.yml Updates

```yaml
version: '3.8'

services:
  aggr-influx:
    container_name: aggr-influx
    image: influxdb:1.8.10
    restart: unless-stopped
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb
    environment:
      - INFLUXDB_DB=significant_trades
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=admin123
      - INFLUXDB_DATA_CACHE_MAX_MEMORY_SIZE=1g
      - INFLUXDB_DATA_CACHE_SNAPSHOT_MEMORY_SIZE=25m
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4096M  # Reduced from 8GB

  aggr-server:
    container_name: aggr-server
    build: ./aggr-server
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - INFLUX_TIMEFRAME=1000  # 1 second
      - BACKUP_INTERVAL=1000
    volumes:
      - ./aggr-server:/usr/src/app/
    depends_on:
      - aggr-influx

  strategy-runner:
    container_name: squeezeflow-strategy-runner
    build: ./services
    restart: unless-stopped
    environment:
      - SQUEEZEFLOW_RUN_INTERVAL=5  # Run every 5 seconds
      - INFLUX_HOST=aggr-influx
      - INFLUX_PORT=8086
      - INFLUX_DATABASE=significant_trades
    depends_on:
      - aggr-influx
      - redis
```

---

## Backtesting with 1-Second Data

### Overview

Once you have collected 1-second data, the backtest system can aggregate it into any timeframe needed (1m, 5m, 15m, etc.) on-the-fly. This provides perfect accuracy for backtesting while maintaining flexibility.

### Step 1: Data Aggregation for Backtesting

The backtest engine should dynamically aggregate 1-second bars into required timeframes:

```python
# File: backtest/data_loader.py

import pandas as pd
from influxdb import InfluxDBClient

class BacktestDataLoader:
    def __init__(self):
        self.influx = InfluxDBClient(
            host='localhost',
            port=8086,
            database='significant_trades'
        )
    
    def load_1s_data(self, symbol, start_time, end_time):
        """Load raw 1-second data from InfluxDB"""
        query = f"""
        SELECT open, high, low, close, volume 
        FROM trades_1s 
        WHERE market = '{symbol}' 
        AND time >= '{start_time}' 
        AND time <= '{end_time}'
        ORDER BY time ASC
        """
        
        result = self.influx.query(query)
        if not result:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(result.get_points()))
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        return df
    
    def aggregate_to_timeframe(self, df_1s, timeframe):
        """
        Aggregate 1-second data to any timeframe
        
        Args:
            df_1s: DataFrame with 1-second OHLCV data
            timeframe: Target timeframe (e.g., '1T', '5T', '15T', '1H')
        
        Returns:
            DataFrame with aggregated OHLCV data
        """
        if df_1s.empty:
            return df_1s
        
        # Pandas resample rules:
        # '1T' = 1 minute, '5T' = 5 minutes, '15T' = 15 minutes
        # '30T' = 30 minutes, '1H' = 1 hour, '4H' = 4 hours
        
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_resampled = df_1s.resample(timeframe).agg(agg_rules)
        
        # Remove rows with NaN (gaps in data)
        df_resampled = df_resampled.dropna()
        
        return df_resampled
    
    def get_multi_timeframe_data(self, symbol, start_time, end_time):
        """
        Load 1s data and create all required timeframes for backtesting
        
        Returns dict with all timeframes needed by strategy
        """
        # Load base 1-second data once
        df_1s = self.load_1s_data(symbol, start_time, end_time)
        
        if df_1s.empty:
            print(f"No 1s data found for {symbol}")
            return {}
        
        # Generate all required timeframes from 1s data
        timeframes = {
            '1m': '1T',    # 1 minute
            '5m': '5T',    # 5 minutes  
            '15m': '15T',  # 15 minutes
            '30m': '30T',  # 30 minutes
            '1h': '1H',    # 1 hour
            '4h': '4H'     # 4 hours
        }
        
        result = {}
        for tf_label, tf_pandas in timeframes.items():
            result[tf_label] = self.aggregate_to_timeframe(df_1s, tf_pandas)
            print(f"  {tf_label}: {len(result[tf_label])} bars")
        
        return result
```

### Step 2: Integration with Backtest Engine

Update your backtest engine to use 1-second data as the source:

```python
# File: run_backtest.py

from backtest.data_loader import BacktestDataLoader
from datetime import datetime, timedelta

def run_backtest_with_1s_data():
    """Run backtest using 1-second data aggregated to required timeframes"""
    
    # Initialize data loader
    loader = BacktestDataLoader()
    
    # Define backtest period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)  # Last 7 days
    
    # Load and aggregate data
    print("Loading 1-second data and aggregating to timeframes...")
    data = loader.get_multi_timeframe_data(
        symbol='BINANCE:btcusdt',
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat()
    )
    
    # Now you have perfectly aligned multi-timeframe data
    # All generated from the same 1s source for consistency
    
    # Run your backtest with the aggregated data
    from backtest.engine import BacktestEngine
    engine = BacktestEngine()
    
    results = engine.run(
        strategy='SqueezeFlow',
        data=data,
        initial_capital=10000,
        position_size=0.1
    )
    
    return results
```

### Step 3: Verify Data Quality

Before running backtests, verify your 1-second data quality:

```bash
# Check data coverage for the last 7 days
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 7d GROUP BY time(1h), market"

# Check for gaps (should see ~3600 points per hour)
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) as bars_per_hour FROM trades_1s \
   WHERE market = 'BINANCE:btcusdt' AND time > now() - 24h \
   GROUP BY time(1h)"
```

### Benefits of 1-Second Source Data for Backtesting

1. **Perfect Timeframe Alignment**: All timeframes (1m, 5m, 15m, etc.) are derived from the same source
2. **No Data Conflicts**: Eliminates issues where different timeframes show conflicting signals
3. **Accurate Volume**: Volume aggregation is precise across all timeframes
4. **Flexible Testing**: Can test strategies at any timeframe without collecting new data
5. **Realistic Execution**: Can simulate order fills at 1-second precision

### Example: Complete Backtest Workflow

```python
# File: backtest_1s_example.py

import pandas as pd
from datetime import datetime, timedelta

# 1. Collect some days of 1s data (this happens automatically with new config)
print("Assuming 1s data collection is running...")

# 2. Wait for data to accumulate
print("After 2-3 days, you'll have enough data to backtest")

# 3. Run backtest with perfect timeframe generation
from backtest.data_loader import BacktestDataLoader

loader = BacktestDataLoader()

# Load last 3 days of 1s data
end = datetime.now()
start = end - timedelta(days=3)

# Get all timeframes from single 1s source
all_timeframes = loader.get_multi_timeframe_data(
    'BINANCE:btcusdt',
    start.isoformat(),
    end.isoformat()
)

print(f"Generated {len(all_timeframes)} timeframes from 1s data:")
for tf, data in all_timeframes.items():
    print(f"  {tf}: {len(data)} bars, {data.index[0]} to {data.index[-1]}")

# 4. Run strategy backtest with perfect data
# Your existing backtest code works unchanged, 
# but now with more accurate data
```

### Important Notes

- **Initial Data Collection**: After implementing 1s collection, wait 2-3 days to accumulate enough data for meaningful backtests
- **Storage Consideration**: 1s data uses ~200MB/day, plan storage accordingly
- **Query Performance**: When backtesting long periods, consider caching aggregated data to avoid repeated queries
- **Data Validation**: Always check for gaps in 1s data before running backtests

---

## Testing & Validation

### Step 1: Verify 1-Second Data Collection

```bash
# Check if 1-second data is being stored
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 1m"

# View sample 1-second bars
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT * FROM trades_1s WHERE market = 'BINANCE:btcusdt' ORDER BY time DESC LIMIT 10"

# Check data ingestion rate
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 1m GROUP BY time(10s)"
```

### Step 2: Verify Strategy Runner Performance

```bash
# Monitor strategy runner logs
docker logs squeezeflow-strategy-runner -f --tail 100

# Check processing time
docker exec squeezeflow-strategy-runner cat /tmp/performance.log | grep "Process time"

# Verify signal generation
docker exec redis redis-cli LRANGE signals 0 10
```

### Step 3: End-to-End Latency Test

```python
# File: test_latency.py

import time
import requests
from datetime import datetime

def test_signal_latency():
    """Test time from market event to signal generation"""
    
    # Monitor latest data timestamp
    influx_query = "SELECT last(close) FROM trades_1s WHERE market = 'BINANCE:btcusdt'"
    
    start = time.time()
    
    # Check data freshness
    response = requests.get(f"http://localhost:8086/query", 
                           params={'db': 'significant_trades', 'q': influx_query})
    
    data_time = response.json()['results'][0]['series'][0]['values'][0][0]
    current_time = datetime.utcnow()
    
    latency = (current_time - datetime.fromisoformat(data_time.replace('Z', ''))).total_seconds()
    
    print(f"Data latency: {latency:.2f} seconds")
    print(f"Expected total signal latency: {latency + 5:.2f} seconds")
    
    return latency < 10  # Should be under 10 seconds total

if __name__ == "__main__":
    test_signal_latency()
```

---

## Performance Monitoring

### Key Metrics to Monitor

```bash
# Create monitoring script
cat > monitor_performance.sh << 'EOF'
#!/bin/bash

echo "=== 1-Second Data Collection Performance ==="

# Check data freshness
echo -n "Latest data age: "
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT (now() - last(time)) / 1000000000 as age_seconds FROM trades_1s" \
  -format csv | tail -n 1

# Check ingestion rate
echo -n "Bars per minute: "
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 1m" \
  -format csv | tail -n 1

# Check strategy runner cycle time
echo -n "Strategy cycle time: "
docker logs squeezeflow-strategy-runner --tail 100 | \
  grep "Cycle completed" | tail -n 1

# Check memory usage
echo "Memory usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" \
  aggr-influx aggr-server squeezeflow-strategy-runner

EOF

chmod +x monitor_performance.sh
./monitor_performance.sh
```

### Resource Usage Expectations

| Component | CPU Usage | Memory | Disk I/O |
|-----------|-----------|---------|----------|
| aggr-server | 10-20% | 500MB | 10 MB/s |
| InfluxDB | 20-30% | 1-2GB | 20 MB/s |
| Strategy Runner | 5-10% | 200MB | Minimal |

---

## Rollback Procedures

### Emergency Rollback Steps

```bash
# 1. Stop affected services
docker-compose stop aggr-server strategy-runner

# 2. Restore original configuration
# In aggr-server/src/config.js, change back:
# influxTimeframe: 10000,

# 3. Restore original docker-compose settings
# SQUEEZEFLOW_RUN_INTERVAL=1

# 4. Restart services
docker-compose up -d aggr-server strategy-runner

# 5. Verify original functionality
docker logs aggr-server --tail 50
```

### Data Recovery

```bash
# If data was corrupted, restore from backup
docker exec aggr-influx influxd restore -portable /backup

# Verify data integrity
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) FROM trades_10s WHERE time > now() - 1h"
```

---

## Performance Comparison

### Before (10-second collection)
- Data collection: Every 10 seconds
- Strategy execution: Every 60 seconds
- Signal latency: 60-70 seconds
- Storage usage: ~20MB/day

### After (1-second collection)
- Data collection: Every 1 second
- Strategy execution: Every 5 seconds
- Signal latency: 5-10 seconds
- Storage usage: ~200MB/day

### Trade-offs
- ✅ 6-12x faster signal generation
- ✅ More accurate market representation
- ✅ Better stop-loss detection (5s vs 60s)
- ❌ 10x more storage usage
- ❌ 2x more CPU usage
- ❌ More frequent database writes

---

## Conclusion

This implementation provides a balanced approach to improving system responsiveness without the complexity of tick-level data. The 1-second granularity offers:

1. **Practical latency improvement** from ~60s to ~5-10s
2. **Manageable resource usage** (~200MB/day storage)
3. **Simple implementation** requiring only configuration changes
4. **Compatibility** with existing strategy and FreqTrade setup
5. **Perfect backtesting** with dynamic timeframe aggregation from 1s source

**Summary for New Claude Instance**:
- Current state: 10-second data collection, 60-second strategy runs
- Target state: 1-second data collection, 5-second strategy runs  
- Main change: Edit `influxTimeframe: 1000` in aggr-server/src/config.js
- Benefit: All timeframes (1m, 5m, 15m, etc.) generated from single 1s source for perfect alignment
- The system maintains realistic execution capabilities while providing faster market response times suitable for the actual trading infrastructure.

For support, check logs:
```bash
docker logs aggr-server --tail 100
docker logs squeezeflow-strategy-runner --tail 100
docker exec aggr-influx influx -execute "SHOW DIAGNOSTICS"
```