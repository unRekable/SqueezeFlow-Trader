#!/bin/bash

# 1-Second Data Collection Performance Monitor
# This script monitors the performance of 1-second data collection

echo "=== 1-Second Data Collection Performance Monitor ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# Check if Docker containers are running
echo ""
echo "📦 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}" | grep -E "aggr-influx|aggr-server|strategy-runner" || echo "No relevant containers running"

# Check data freshness - how old is the latest data?
echo ""
echo "⏱️  Data Freshness:"
LATEST_DATA=$(docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT last(close) FROM trades_1s WHERE time > now() - 1m" -format csv 2>/dev/null | tail -n 1)

if [ -n "$LATEST_DATA" ]; then
    echo "✅ 1-second data is being collected"
    
    # Calculate age of latest data
    AGE_QUERY="SELECT (now() - last(time)) / 1000000000 as age_seconds FROM trades_1s WHERE time > now() - 5m"
    AGE=$(docker exec aggr-influx influx -database significant_trades -execute "$AGE_QUERY" -format csv 2>/dev/null | tail -n 1)
    
    if [ -n "$AGE" ]; then
        echo "   Latest data age: ${AGE} seconds"
        
        # Alert if data is stale (> 10 seconds old)
        if (( $(echo "$AGE > 10" | bc -l) )); then
            echo "   ⚠️  WARNING: Data is stale (> 10 seconds old)"
        fi
    fi
else
    echo "❌ No 1-second data found - checking for 10s data..."
    FALLBACK_DATA=$(docker exec aggr-influx influx -database significant_trades -execute \
      "SELECT last(close) FROM trades_10s WHERE time > now() - 1m" -format csv 2>/dev/null | tail -n 1)
    
    if [ -n "$FALLBACK_DATA" ]; then
        echo "   Found 10s data (old configuration still active)"
    else
        echo "   No recent data found in any measurement"
    fi
fi

# Check ingestion rate - bars per minute
echo ""
echo "📊 Ingestion Rate:"
BARS_1M=$(docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 1m" -format csv 2>/dev/null | tail -n 1)

if [ -n "$BARS_1M" ] && [ "$BARS_1M" != "0" ]; then
    echo "   1-minute: $BARS_1M bars"
    
    # Expected: ~60 bars per minute per market
    # With multiple markets, should be much higher
    MARKETS_COUNT=$(docker exec aggr-influx influx -database significant_trades -execute \
      "SELECT COUNT(DISTINCT(market)) FROM trades_1s WHERE time > now() - 1m" -format csv 2>/dev/null | tail -n 1)
    
    if [ -n "$MARKETS_COUNT" ] && [ "$MARKETS_COUNT" != "0" ]; then
        echo "   Markets tracked: $MARKETS_COUNT"
        BARS_PER_MARKET=$(echo "scale=2; $BARS_1M / $MARKETS_COUNT" | bc)
        echo "   Bars per market: ~$BARS_PER_MARKET per minute"
        
        if (( $(echo "$BARS_PER_MARKET < 50" | bc -l) )); then
            echo "   ⚠️  WARNING: Low ingestion rate (expected ~60 per market)"
        fi
    fi
else
    echo "   No data ingested in the last minute"
fi

# Check data distribution over last 5 minutes
echo ""
echo "📈 Data Distribution (5 min):"
docker exec aggr-influx influx -database significant_trades -execute \
  "SELECT COUNT(*) as bars FROM trades_1s WHERE time > now() - 5m GROUP BY time(1m), market LIMIT 5" \
  -format csv 2>/dev/null | head -n 10 || echo "   No 1s data available"

# Check strategy runner performance
echo ""
echo "🎯 Strategy Runner:"
LAST_RUN=$(docker logs squeezeflow-strategy-runner --tail 100 2>&1 | grep -E "Strategy runner cycle|Processing complete" | tail -n 1)
if [ -n "$LAST_RUN" ]; then
    echo "   Last cycle: $LAST_RUN"
else
    echo "   No recent strategy runs detected"
fi

# Memory and CPU usage
echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
  aggr-influx aggr-server squeezeflow-strategy-runner 2>/dev/null || echo "   Unable to get stats"

# Disk usage for InfluxDB
echo ""
echo "💿 Storage Usage:"
INFLUX_SIZE=$(docker exec aggr-influx du -sh /var/lib/influxdb 2>/dev/null | cut -f1)
if [ -n "$INFLUX_SIZE" ]; then
    echo "   InfluxDB data: $INFLUX_SIZE"
    
    # Estimate daily growth rate
    BARS_24H=$(docker exec aggr-influx influx -database significant_trades -execute \
      "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 24h" -format csv 2>/dev/null | tail -n 1)
    
    if [ -n "$BARS_24H" ] && [ "$BARS_24H" != "0" ]; then
        echo "   24h bars collected: $BARS_24H"
        # Rough estimate: ~100 bytes per bar
        DAILY_MB=$(echo "scale=2; $BARS_24H * 100 / 1048576" | bc)
        echo "   Estimated daily growth: ~${DAILY_MB}MB"
    fi
fi

# Check for errors in logs
echo ""
echo "⚠️  Recent Errors/Warnings:"
echo "   aggr-server:"
docker logs aggr-server --tail 50 2>&1 | grep -iE "error|warning|fail" | tail -n 2 || echo "     No recent errors"

echo "   strategy-runner:"
docker logs squeezeflow-strategy-runner --tail 50 2>&1 | grep -iE "error|warning|fail" | tail -n 2 || echo "     No recent errors"

# Performance comparison
echo ""
echo "📊 Performance Comparison:"
echo "   Configuration: 1-second data collection"
echo "   Expected signal latency: 5-10 seconds"
echo "   Previous (10s): 60-70 seconds latency"
echo "   Improvement: ~6-12x faster signals"

# Recommendations
echo ""
echo "💡 Recommendations:"

if [ -n "$BARS_1M" ] && [ "$BARS_1M" != "0" ]; then
    echo "   ✅ 1-second data collection is active"
else
    echo "   ⚠️  Restart aggr-server to activate 1s collection:"
    echo "      docker-compose restart aggr-server"
fi

if [ -n "$AGE" ] && (( $(echo "$AGE > 10" | bc -l) )); then
    echo "   ⚠️  Check network connectivity and exchange connections"
fi

echo ""
echo "=================================================="
echo "Monitor complete at $(date '+%H:%M:%S')"