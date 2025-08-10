#!/bin/bash

# Setup 1-minute Continuous Queries for InfluxDB
# This creates backup 1-minute aggregated data from 1-second data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up 1-minute Continuous Queries${NC}"
echo "========================================"

# Function to execute InfluxDB commands
influx_exec() {
    local query="$1"
    echo -e "${YELLOW}Executing:${NC} $query"
    docker exec aggr-influx influx -execute "$query" -database significant_trades
}

echo -e "${YELLOW}1. Creating retention policies...${NC}"

# Create 1-minute retention policy (if it doesn't exist)
influx_exec "CREATE RETENTION POLICY rp_1m ON significant_trades DURATION INF REPLICATION 1" || true

echo -e "${YELLOW}2. Creating continuous query for 1-minute OHLCV aggregation...${NC}"

# Drop existing CQ if it exists
influx_exec "DROP CONTINUOUS QUERY cq_trades_1m ON significant_trades" || true

# Create 1-minute continuous query (single line format for InfluxDB)
CQ_QUERY='CREATE CONTINUOUS QUERY cq_trades_1m ON significant_trades BEGIN SELECT FIRST(open) AS open, MAX(high) AS high, MIN(low) AS low, LAST(close) AS close, SUM(vbuy) AS vbuy, SUM(vsell) AS vsell, SUM(cbuy) AS cbuy, SUM(csell) AS csell, SUM(lbuy) AS lbuy, SUM(lsell) AS lsell INTO "rp_1m"."trades_1m" FROM "aggr_1s"."trades_1s" GROUP BY time(1m), market END'

influx_exec "$CQ_QUERY"

echo -e "${YELLOW}3. Verifying continuous query...${NC}"
influx_exec "SHOW CONTINUOUS QUERIES ON significant_trades"

echo -e "${YELLOW}4. Checking if data will be aggregated (may take up to 1 minute)...${NC}"
sleep 5

# Check if 1-minute data is being created
echo -e "${YELLOW}Checking for 1-minute data...${NC}"
result=$(docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"rp_1m\".\"trades_1m\" WHERE time > now() - 5m" -database significant_trades 2>/dev/null || echo "No data yet")

if echo "$result" | grep -q "count"; then
    count=$(echo "$result" | grep -o '[0-9]\+' | head -1)
    echo -e "${GREEN}✅ 1-minute aggregation working! Found $count 1-minute bars${NC}"
else
    echo -e "${YELLOW}⏳ No 1-minute data yet (CQ runs every minute, wait 1-2 minutes)${NC}"
fi

echo ""
echo -e "${GREEN}Continuous Query Setup Complete!${NC}"
echo ""
echo "📊 What was created:"
echo "  • Retention Policy: rp_1m (infinite duration)"
echo "  • Measurement: trades_1m"
echo "  • Aggregation: OHLCV from 1s to 1m data"
echo "  • Schedule: Runs every minute automatically"
echo ""
echo "🔍 How to query 1-minute data:"
echo '  docker exec aggr-influx influx -execute "SELECT * FROM \"rp_1m\".\"trades_1m\" ORDER BY time DESC LIMIT 5" -database significant_trades'
echo ""
echo "📈 Benefits:"
echo "  ✅ Backup aggregated data (more storage efficient)"
echo "  ✅ Faster queries for longer timeframes"
echo "  ✅ Redundancy if 1s data retention changes"
echo "  ✅ Better performance for analysis tools"
echo ""
echo -e "${YELLOW}Note: CQ runs every minute. Check again in 1-2 minutes for data.${NC}"