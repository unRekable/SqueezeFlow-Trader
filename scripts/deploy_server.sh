#!/bin/bash

# Deploy SqueezeFlow Server Configuration
# Run this on your server to start data collection services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying SqueezeFlow Server Services${NC}"
echo "=============================================="

# Check if we're on the server
if [ ! -f "docker-compose.server.yml" ]; then
    echo -e "${RED}❌ docker-compose.server.yml not found!${NC}"
    echo "Run this script from the project root directory on your server."
    exit 1
fi

echo -e "${YELLOW}1. Stopping any existing services...${NC}"
docker-compose -f docker-compose.server.yml down 2>/dev/null || true

echo -e "${YELLOW}2. Pulling latest images...${NC}"
docker-compose -f docker-compose.server.yml pull

echo -e "${YELLOW}3. Building custom images...${NC}"
docker-compose -f docker-compose.server.yml build

echo -e "${YELLOW}4. Starting server services...${NC}"
echo "  • InfluxDB (time-series database)"  
echo "  • aggr-server (market data collection)"
echo "  • OI Tracker (open interest tracking with dynamic symbol discovery)"
echo "  • Chronograf (InfluxDB admin UI)"

docker-compose -f docker-compose.server.yml up -d

echo -e "${YELLOW}5. Waiting for services to initialize...${NC}"
sleep 10

echo -e "${YELLOW}6. Setting up retention policies...${NC}"
./scripts/setup_retention_policies.sh

echo -e "${YELLOW}7. Setting up 1-minute continuous queries...${NC}"
./scripts/setup_1m_continuous_queries.sh

echo -e "${YELLOW}8. Checking service health...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check InfluxDB
if curl -sf http://localhost:8086/ping > /dev/null; then
    echo -e "${GREEN}✅ InfluxDB: Running${NC}"
else
    echo -e "${RED}❌ InfluxDB: Not responding${NC}"
fi

# Check aggr-server
if curl -sf http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ aggr-server: Running${NC}"
else
    echo -e "${YELLOW}⏳ aggr-server: Starting up...${NC}"
fi

# Check Redis
if docker exec squeezeflow-redis-server redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis: Running${NC}"
else
    echo -e "${RED}❌ Redis: Not responding${NC}"
fi

# Check Chronograf
if curl -sf http://localhost:8888 > /dev/null; then
    echo -e "${GREEN}✅ Chronograf: Running${NC}"
else
    echo -e "${YELLOW}⏳ Chronograf: Starting up...${NC}"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "${YELLOW}9. Monitoring initial data collection...${NC}"
echo "Waiting 30 seconds for data to start flowing..."
sleep 30

# Check if trades data is being collected
data_check=$(docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 2m" -database significant_trades 2>/dev/null || echo "0")

if echo "$data_check" | grep -q "[1-9]"; then
    count=$(echo "$data_check" | grep -o '[0-9]\+' | head -1)
    echo -e "${GREEN}🎉 Trades data collection started! $count data points in last 2 minutes${NC}"
else
    echo -e "${YELLOW}⏳ Trades data collection starting (may take 1-2 minutes)${NC}"
fi

# Check OI data collection (starts after symbol discovery)
echo -e "${YELLOW}10. Checking OI tracker initialization...${NC}"
sleep 30

oi_check=$(docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"open_interest\" WHERE time > now() - 5m" -database significant_trades 2>/dev/null || echo "0")

if echo "$oi_check" | grep -q "[1-9]"; then
    count=$(echo "$oi_check" | grep -o '[0-9]\+' | head -1)
    echo -e "${GREEN}🎉 OI data collection started! $count OI records in last 5 minutes${NC}"
else
    echo -e "${YELLOW}⏳ OI tracker discovering symbols and starting collection...${NC}"
fi

echo ""
echo -e "${GREEN}🚀 Server Deployment Complete!${NC}"
echo ""
echo "📊 Services Running:"
echo "  • InfluxDB:    http://localhost:8086"
echo "  • Chronograf:  http://localhost:8888"
echo "  • aggr-server: http://localhost:3000 (if enabled)"
echo "  • Redis:       localhost:6379"
echo ""
echo "🔧 Useful Commands:"
echo "  # Check all services"
echo "  docker-compose -f docker-compose.server.yml ps"
echo ""
echo "  # View logs"
echo "  docker-compose -f docker-compose.server.yml logs -f"
echo "  docker-compose -f docker-compose.server.yml logs -f oi-tracker"
echo ""
echo "  # Monitor data collection"
echo "  ./scripts/monitor_1m_aggregation.sh"
echo ""
echo "  # Check OI data collection"
echo '  docker exec aggr-influx influx -execute "SELECT * FROM open_interest ORDER BY time DESC LIMIT 5" -database significant_trades'
echo ""
echo "  # View discovered symbols for OI"
echo "  docker logs squeezeflow-oi-tracker | grep 'Discovered.*symbols'"
echo ""
echo "  # Stop all services"
echo "  docker-compose -f docker-compose.server.yml down"
echo ""
echo -e "${BLUE}📈 Server is now collecting 24/7 market data for your trading system!${NC}"