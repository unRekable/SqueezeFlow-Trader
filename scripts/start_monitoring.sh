#!/bin/bash

# SqueezeFlow Trader - Monitoring Services Startup Script
# Starts health-monitor and performance-monitor services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE} SqueezeFlow Monitoring Services${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to check if service is running
check_service_health() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Checking ${service_name} health...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps ${service_name} | grep -q "Up"; then
            echo -e "${GREEN}✓ ${service_name} is running${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ ${service_name} failed to start properly${NC}"
    return 1
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local service_name=$2
    local max_attempts=20
    local attempt=1
    
    echo -e "${YELLOW}Checking ${service_name} endpoint: ${url}${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "${url}" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ${service_name} endpoint is responding${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 3
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ ${service_name} endpoint not responding${NC}"
    return 1
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found. Please run from project root.${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Starting prerequisite services...${NC}"

# Start Redis and InfluxDB if not running
echo -e "${YELLOW}Starting Redis and InfluxDB...${NC}"
docker-compose up -d redis aggr-influx

# Wait for Redis and InfluxDB to be healthy
sleep 5

# Check Redis
if ! docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}Error: Redis is not responding${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Redis is ready${NC}"

# Check InfluxDB  
if ! curl -s http://localhost:8086/ping > /dev/null 2>&1; then
    echo -e "${RED}Error: InfluxDB is not responding${NC}"
    exit 1
fi
echo -e "${GREEN}✓ InfluxDB is ready${NC}"

echo ""
echo -e "${BLUE}Step 2: Starting monitoring services...${NC}"

# Start monitoring services
echo -e "${YELLOW}Starting Health Monitor...${NC}"
docker-compose up -d health-monitor

echo -e "${YELLOW}Starting Performance Monitor...${NC}" 
docker-compose up -d performance-monitor

echo -e "${YELLOW}Starting Signal Monitor Dashboard...${NC}"
docker-compose up -d signal-monitor-dashboard

echo ""
echo -e "${BLUE}Step 3: Verifying service health...${NC}"

# Check services are running
check_service_health "health-monitor"
check_service_health "performance-monitor"
check_service_health "signal-monitor-dashboard"

echo ""
echo -e "${BLUE}Step 4: Verifying endpoints...${NC}"

# Wait a bit for services to fully initialize
sleep 10

# Check Health Monitor HTTP endpoint
check_http_endpoint "http://localhost:8090/health" "Health Monitor"

echo ""
echo -e "${BLUE}Step 5: Service status summary...${NC}"

# Show service status
docker-compose ps health-monitor performance-monitor redis aggr-influx

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN} Monitoring Services Started! ${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

echo -e "${BLUE}Available Endpoints:${NC}"
echo -e "  • Health Check:      ${YELLOW}http://localhost:8090/health${NC}"
echo -e "  • Detailed Health:   ${YELLOW}http://localhost:8090/health/detailed${NC}" 
echo -e "  • Service Health:    ${YELLOW}http://localhost:8090/health/service/{name}${NC}"
echo -e "  • Prometheus Metrics: ${YELLOW}http://localhost:8090/metrics${NC}"
echo -e "  • System Status:     ${YELLOW}http://localhost:8090/status${NC}"
echo ""

echo -e "${BLUE}Signal Monitor Dashboard:${NC}"
echo -e "  • Live Dashboard:    ${YELLOW}docker attach squeezeflow-signal-dashboard${NC}"
echo -e "  • Dashboard Logs:    ${YELLOW}docker-compose logs -f signal-monitor-dashboard${NC}"
echo ""

echo -e "${BLUE}Monitoring Commands:${NC}"
echo -e "  • Health Monitor Logs:     ${YELLOW}docker-compose logs -f health-monitor${NC}"
echo -e "  • Performance Monitor Logs: ${YELLOW}docker-compose logs -f performance-monitor${NC}"
echo -e "  • Signal Dashboard Logs:   ${YELLOW}docker-compose logs -f signal-monitor-dashboard${NC}"
echo -e "  • View Metrics in Redis:   ${YELLOW}redis-cli KEYS 'squeezeflow:metrics:*'${NC}"
echo -e "  • View Active Alerts:      ${YELLOW}redis-cli KEYS 'squeezeflow:alerts'${NC}"
echo ""

echo -e "${BLUE}Testing Health Endpoint:${NC}"
echo -e "${YELLOW}curl http://localhost:8090/health | jq${NC}"
echo ""

# Test the endpoint and show result
if command -v jq >/dev/null 2>&1; then
    echo -e "${BLUE}Current Health Status:${NC}"
    curl -s http://localhost:8090/health | jq . || echo -e "${YELLOW}(jq not available - use: curl http://localhost:8090/health)${NC}"
else
    echo -e "${YELLOW}Install jq for pretty JSON output: curl http://localhost:8090/health${NC}"
fi

echo ""
echo -e "${GREEN}Monitoring services are ready!${NC}"
echo -e "${BLUE}Use 'docker-compose logs -f health-monitor performance-monitor' to view logs${NC}"