#!/bin/bash

# Setup script for distributed SqueezeFlow Trader deployment
# This script helps deploy data collection on server and development on local

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}SqueezeFlow Trader - Distributed Setup${NC}"
echo "========================================="

# Function to display usage
usage() {
    echo "Usage: $0 [server|local|test-connection|help]"
    echo ""
    echo "Commands:"
    echo "  server          - Setup instructions for server deployment"
    echo "  local           - Setup local MacBook environment"
    echo "  test-connection - Test connection to remote InfluxDB"
    echo "  help            - Show this help message"
    exit 1
}

# Server setup instructions
setup_server() {
    echo -e "${YELLOW}Server Setup Instructions:${NC}"
    echo ""
    echo "1. Clone the repository on your server:"
    echo "   git clone https://github.com/your-repo/squeezeflow-trader.git"
    echo "   cd squeezeflow-trader"
    echo ""
    echo "2. Clone the aggr-server submodule:"
    echo "   git submodule update --init --recursive"
    echo ""
    echo "3. Configure firewall (if needed):"
    echo "   # Allow InfluxDB port"
    echo "   sudo ufw allow 8086/tcp"
    echo "   # Allow Chronograf UI (optional)"
    echo "   sudo ufw allow 8888/tcp"
    echo ""
    echo "4. Start the services:"
    echo "   docker-compose -f docker-compose.server.yml up -d"
    echo ""
    echo "5. Verify services are running:"
    echo "   docker-compose -f docker-compose.server.yml ps"
    echo ""
    echo "6. Check InfluxDB is accessible:"
    echo "   curl http://localhost:8086/ping"
    echo ""
    echo -e "${GREEN}Server setup complete!${NC}"
}

# Local setup
setup_local() {
    echo -e "${YELLOW}Local MacBook Setup:${NC}"
    echo ""
    
    # Check if .env exists
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file...${NC}"
        cp .env.example .env
        echo -e "${RED}Please edit .env and set your SERVER_IP${NC}"
        echo "Run 'nano .env' or 'vim .env' to edit"
        exit 1
    fi
    
    # Load environment variables
    source .env
    
    if [ "$SERVER_IP" == "your.server.ip.here" ]; then
        echo -e "${RED}Please set SERVER_IP in .env file first!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Using server IP: $SERVER_IP${NC}"
    echo ""
    
    # Stop existing services
    echo "Stopping existing services..."
    docker-compose down 2>/dev/null || true
    
    # Start local services
    echo "Starting local services..."
    docker-compose -f docker-compose.local.yml up -d
    
    echo ""
    echo -e "${GREEN}Local setup complete!${NC}"
    echo ""
    echo "Services running locally:"
    docker-compose -f docker-compose.local.yml ps
}

# Test connection to remote InfluxDB
test_connection() {
    echo -e "${YELLOW}Testing Remote InfluxDB Connection:${NC}"
    echo ""
    
    # Load environment variables
    if [ -f .env ]; then
        source .env
    else
        echo -e "${RED}No .env file found. Please run setup first.${NC}"
        exit 1
    fi
    
    if [ -z "$SERVER_IP" ] || [ "$SERVER_IP" == "your.server.ip.here" ]; then
        echo -e "${RED}SERVER_IP not set in .env file!${NC}"
        exit 1
    fi
    
    echo "Testing connection to $SERVER_IP:8086..."
    
    # Test basic connectivity
    if curl -f -s "http://$SERVER_IP:8086/ping" > /dev/null; then
        echo -e "${GREEN}✓ InfluxDB is reachable${NC}"
    else
        echo -e "${RED}✗ Cannot reach InfluxDB at $SERVER_IP:8086${NC}"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check if InfluxDB is running on server:"
        echo "   ssh $SERVER_IP 'docker ps | grep influx'"
        echo "2. Check firewall rules:"
        echo "   ssh $SERVER_IP 'sudo ufw status'"
        echo "3. Try SSH tunnel:"
        echo "   ssh -L 8086:localhost:8086 $SERVER_IP"
        exit 1
    fi
    
    # Test database exists
    echo "Checking database 'significant_trades'..."
    response=$(curl -s "http://$SERVER_IP:8086/query?q=SHOW%20DATABASES" 2>/dev/null)
    if echo "$response" | grep -q "significant_trades"; then
        echo -e "${GREEN}✓ Database 'significant_trades' exists${NC}"
    else
        echo -e "${YELLOW}⚠ Database 'significant_trades' not found (will be created on first run)${NC}"
    fi
    
    # Test data availability
    echo "Checking for recent data..."
    query="SELECT%20COUNT(*)%20FROM%20trades_1s%20WHERE%20time%20>%20now()%20-%201h"
    response=$(curl -s "http://$SERVER_IP:8086/query?db=significant_trades&q=$query" 2>/dev/null)
    if echo "$response" | grep -q "values"; then
        echo -e "${GREEN}✓ Recent data found in database${NC}"
    else
        echo -e "${YELLOW}⚠ No recent data found (aggr-server may need to collect data)${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}Connection test complete!${NC}"
}

# SSH Tunnel setup (optional for secure connection)
setup_tunnel() {
    echo -e "${YELLOW}Setting up SSH Tunnel for Secure Connection:${NC}"
    echo ""
    
    if [ -f .env ]; then
        source .env
    fi
    
    if [ -z "$SERVER_IP" ]; then
        echo -e "${RED}SERVER_IP not set in .env file!${NC}"
        exit 1
    fi
    
    SSH_USER=${SSH_USER:-root}
    SSH_PORT=${SSH_PORT:-22}
    
    echo "Creating SSH tunnel to $SERVER_IP..."
    echo "Command: ssh -N -L 8086:localhost:8086 $SSH_USER@$SERVER_IP -p $SSH_PORT"
    echo ""
    echo "Keep this terminal open while using the tunnel."
    echo "In another terminal, set SERVER_IP=localhost in .env to use the tunnel."
    
    ssh -N -L 8086:localhost:8086 "$SSH_USER@$SERVER_IP" -p "$SSH_PORT"
}

# Main script logic
case "${1:-}" in
    server)
        setup_server
        ;;
    local)
        setup_local
        ;;
    test-connection|test)
        test_connection
        ;;
    tunnel)
        setup_tunnel
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        usage
        ;;
esac