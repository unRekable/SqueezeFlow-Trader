#!/bin/bash
"""
SqueezeFlow Trader 2 - Shutdown Script
Safely stops all services and saves data
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Stopping SqueezeFlow Trader 2 System${NC}"
echo "=========================================="

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not installed.${NC}"
    exit 1
fi

# Stop services in reverse order
echo -e "${YELLOW}🤖 Stopping Freqtrade...${NC}"
docker-compose stop freqtrade freqtrade-ui

echo -e "${YELLOW}📊 Stopping monitoring services...${NC}"
docker-compose stop system-monitor

echo -e "${YELLOW}🎯 Stopping SqueezeFlow services...${NC}"
docker-compose stop squeezeflow-calculator oi-tracker

echo -e "${YELLOW}📡 Stopping aggr-server...${NC}"
docker-compose stop aggr-server

echo -e "${YELLOW}📈 Stopping Grafana...${NC}"
docker-compose stop grafana

echo -e "${YELLOW}🗄️  Stopping infrastructure services...${NC}"
docker-compose stop influxdb redis

echo -e "${GREEN}✅ All services stopped successfully!${NC}"

# Show final status
echo ""
echo "Final Status:"
docker-compose ps

echo ""
echo -e "${GREEN}💾 Data has been preserved in ./data/${NC}"
echo -e "${GREEN}📝 Logs are available in ./freqtrade/user_data/logs/${NC}"

echo ""
echo -e "${YELLOW}💡 Use './start.sh' to restart the system${NC}"
echo -e "${YELLOW}💡 Use 'docker-compose down' to remove containers${NC}"
echo -e "${YELLOW}💡 Use 'docker-compose down -v' to remove containers and volumes${NC}"

echo ""
echo -e "${GREEN}👋 SqueezeFlow Trader 2 stopped safely!${NC}"