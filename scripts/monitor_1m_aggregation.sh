#!/bin/bash

# Monitor 1-minute Continuous Query Status
# Check if 1s -> 1m aggregation is working properly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 1-Minute Aggregation Status Monitor${NC}"
echo "======================================"

# Check if continuous query exists
echo -e "${YELLOW}1. Checking Continuous Query Status...${NC}"
cq_status=$(docker exec aggr-influx influx -database significant_trades -execute "SHOW CONTINUOUS QUERIES" 2>/dev/null)

if echo "$cq_status" | grep -q "cq_trades_1m"; then
    echo -e "${GREEN}✅ Continuous Query 'cq_trades_1m' exists${NC}"
else
    echo -e "${RED}❌ Continuous Query 'cq_trades_1m' not found${NC}"
    echo "Run: ./scripts/setup_1m_continuous_queries.sh"
    exit 1
fi

# Check retention policies
echo -e "${YELLOW}2. Checking Retention Policies...${NC}"
rp_status=$(docker exec aggr-influx influx -execute "SHOW RETENTION POLICIES ON significant_trades" -database significant_trades 2>/dev/null)

if echo "$rp_status" | grep -q "rp_1m"; then
    echo -e "${GREEN}✅ Retention Policy 'rp_1m' exists${NC}"
else
    echo -e "${RED}❌ Retention Policy 'rp_1m' not found${NC}"
fi

# Check 1-second source data
echo -e "${YELLOW}3. Checking 1-second Source Data...${NC}"
source_count=$(docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 10m" -database significant_trades 2>/dev/null)

if echo "$source_count" | grep -q "count"; then
    count=$(echo "$source_count" | grep -o '[0-9]\+' | head -1)
    echo -e "${GREEN}✅ 1-second data available: $count points in last 10 minutes${NC}"
else
    echo -e "${RED}❌ No 1-second source data found${NC}"
fi

# Check 1-minute aggregated data
echo -e "${YELLOW}4. Checking 1-minute Aggregated Data...${NC}"
minute_count=$(docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"rp_1m\".\"trades_1m\" WHERE time > now() - 1h" -database significant_trades 2>/dev/null)

if echo "$minute_count" | grep -q "count"; then
    count=$(echo "$minute_count" | grep -o '[0-9]\+' | head -1)
    echo -e "${GREEN}✅ 1-minute data available: $count bars in last hour${NC}"
    
    # Show latest 1-minute data
    echo -e "${YELLOW}5. Latest 1-minute Data Sample:${NC}"
    docker exec aggr-influx influx -execute "SELECT * FROM \"rp_1m\".\"trades_1m\" ORDER BY time DESC LIMIT 3" -database significant_trades -precision rfc3339
else
    echo -e "${YELLOW}⏳ No 1-minute data yet (CQ may still be running)${NC}"
fi

# Performance comparison
echo -e "${YELLOW}6. Data Volume Comparison:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1-second data size (last hour)
source_1h=$(docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 1h" -database significant_trades 2>/dev/null)
if echo "$source_1h" | grep -q "count"; then
    count_1s=$(echo "$source_1h" | grep -o '[0-9]\+' | head -1)
    echo -e "📈 1-second data (1h):  ${BLUE}$count_1s${NC} points"
else
    count_1s=0
    echo -e "📈 1-second data (1h):  ${RED}0${NC} points"
fi

# 1-minute data size (last hour)
if echo "$minute_count" | grep -q "count"; then
    count_1m=$(echo "$minute_count" | grep -o '[0-9]\+' | head -1)
    echo -e "📊 1-minute data (1h):  ${BLUE}$count_1m${NC} bars"
    
    # Calculate compression ratio
    if [ "$count_1s" -gt 0 ] && [ "$count_1m" -gt 0 ]; then
        ratio=$((count_1s / count_1m))
        echo -e "🗜️  Compression ratio:  ${GREEN}${ratio}:1${NC} (${ratio}x smaller)"
    fi
else
    echo -e "📊 1-minute data (1h):  ${RED}0${NC} bars"
fi

echo ""
echo -e "${GREEN}Monitor Complete!${NC}"
echo ""
echo "🔧 Useful Commands:"
echo "  # View all 1-minute data"
echo '  docker exec aggr-influx influx -execute "SELECT * FROM \"rp_1m\".\"trades_1m\" ORDER BY time DESC LIMIT 10" -database significant_trades'
echo ""
echo "  # Count 1-minute bars by market"
echo '  docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"rp_1m\".\"trades_1m\" WHERE time > now() - 24h GROUP BY market" -database significant_trades'
echo ""
echo "  # Check CQ execution log"
echo '  docker exec aggr-influx influx -execute "SHOW CONTINUOUS QUERIES ON significant_trades" -database significant_trades'
echo ""

# Health assessment
if [ "$count_1s" -gt 0 ] && [ "$count_1m" -gt 0 ]; then
    echo -e "${GREEN}🎉 1-minute aggregation is working perfectly!${NC}"
elif [ "$count_1s" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  1-second data exists but no 1-minute aggregation yet. Wait 1-2 minutes.${NC}"
else
    echo -e "${RED}🚨 No source data - check aggr-server data collection${NC}"
fi