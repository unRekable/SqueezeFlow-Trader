#!/bin/bash
"""
SqueezeFlow Trader 2 - Startup Script
Launches the complete trading system with all services
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting SqueezeFlow Trader 2 System${NC}"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not installed. Please install it first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data/{influxdb,redis,freqtrade,logs}
mkdir -p freqtrade/user_data/{logs,strategies,data}

# Set permissions
chmod +x start.sh stop.sh

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating default...${NC}"
    cat > .env << EOF
# InfluxDB Configuration
INFLUX_HOST=influxdb
INFLUX_PORT=8086
INFLUX_USER=squeezeflow
INFLUX_PASSWORD=password123
INFLUX_DATABASE=significant_trades

# Redis Configuration
REDIS_URL=redis://redis:6379

# Freqtrade Configuration
FREQTRADE_UI_PASSWORD=squeezeflow123

# System Configuration
NODE_ENV=production
TZ=UTC
EOF
fi

# Build and start services
echo -e "${YELLOW}🔧 Building and starting services...${NC}"

# Start infrastructure services first
echo -e "${YELLOW}📊 Starting infrastructure services...${NC}"
docker-compose up -d aggr-influx redis

# Wait for InfluxDB to be ready
echo -e "${YELLOW}⏳ Waiting for InfluxDB to be ready...${NC}"
sleep 10

# Initialize InfluxDB database with advanced setup
echo -e "${YELLOW}🗄️  Initializing InfluxDB with retention policies and continuous queries...${NC}"
python3 init.py --mode production --force

# Start aggr-server
echo -e "${YELLOW}📡 Starting aggr-server...${NC}"
docker-compose up -d aggr-server

# Wait for aggr-server to collect some data
echo -e "${YELLOW}⏳ Waiting for initial data collection...${NC}"
sleep 30

# Start SqueezeFlow services
echo -e "${YELLOW}🎯 Starting SqueezeFlow services...${NC}"
docker-compose up -d oi-tracker squeezeflow-calculator

# Wait for signals to be generated
echo -e "${YELLOW}⏳ Waiting for squeeze signals...${NC}"
sleep 30

# Start Freqtrade
echo -e "${YELLOW}🤖 Starting Freqtrade with FreqAI...${NC}"
docker-compose up -d freqtrade freqtrade-ui

# Start system monitor
echo -e "${YELLOW}📊 Starting system monitor...${NC}"
docker-compose up -d system-monitor

# Show status
echo -e "${GREEN}✅ All services started successfully!${NC}"
echo ""
echo "Service Status:"
docker-compose ps

echo ""
echo "Access URLs:"
echo -e "${GREEN}🤖 Freqtrade API: http://localhost:8080 (squeezeflow/squeezeflow123)${NC}"
echo -e "${GREEN}📊 Freqtrade UI: http://localhost:8081${NC}"
echo -e "${GREEN}📡 aggr-server: http://localhost:3000${NC}"
echo -e "${GREEN}📈 Chronograf (InfluxDB UI): http://localhost:8885${NC}"

echo ""
echo "System Information:"
echo -e "${YELLOW}📂 Data Location: ./data/${NC}"
echo -e "${YELLOW}📝 Logs Location: ./freqtrade/user_data/logs/${NC}"
echo -e "${YELLOW}⚙️  Configuration: ./freqtrade/config/${NC}"

echo ""
echo -e "${GREEN}🎉 SqueezeFlow Trader 2 is now running!${NC}"
echo -e "${YELLOW}💡 Use './stop.sh' to stop all services${NC}"
echo -e "${YELLOW}💡 Use 'docker-compose logs -f [service]' to view logs${NC}"

# Show initial system status
echo ""
echo "Initial System Check:"
sleep 5
docker-compose exec -T influxdb influx -execute "SHOW DATABASES" 2>/dev/null || echo "InfluxDB not ready yet"
echo ""
echo -e "${GREEN}🚀 Happy Trading!${NC}"