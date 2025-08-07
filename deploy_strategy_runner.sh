#!/bin/bash
# Strategy Runner Service Deployment Script
# Safely deploys only the strategy runner service without disrupting existing services

set -e

echo "🚀 SqueezeFlow Strategy Runner Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'  
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Docker is running"

# Check if existing services are running
print_info "Checking existing service status..."

if docker ps | grep -q "aggr-influx"; then
    print_status "InfluxDB is running"
else
    print_warning "InfluxDB is not running. Starting required services first..."
    docker-compose up -d aggr-influx redis
    sleep 10
fi

if docker ps | grep -q "squeezeflow-redis"; then
    print_status "Redis is running"
else
    print_warning "Redis is not running. Starting Redis..."
    docker-compose up -d redis
    sleep 5
fi

# Build the updated Docker image
print_info "Building Strategy Runner Docker image..."
docker-compose build strategy-runner

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Deploy only the strategy runner service
print_info "Deploying Strategy Runner service..."
docker-compose up -d strategy-runner

if [ $? -eq 0 ]; then
    print_status "Strategy Runner service deployed successfully"
else
    print_error "Failed to deploy Strategy Runner service"
    exit 1
fi

# Wait for service to start
print_info "Waiting for service to initialize..."
sleep 15

# Check service health
if docker ps | grep -q "squeezeflow-strategy-runner"; then
    print_status "Strategy Runner container is running"
    
    # Show container logs
    print_info "Recent container logs:"
    echo "----------------------------------------"
    docker logs --tail 20 squeezeflow-strategy-runner
    echo "----------------------------------------"
    
else
    print_error "Strategy Runner container failed to start"
    print_info "Container logs:"
    docker logs squeezeflow-strategy-runner
    exit 1
fi

# Test Redis connection from strategy runner
print_info "Testing Redis connection..."
if docker exec squeezeflow-strategy-runner python -c "import redis; r=redis.Redis(host='redis', port=6379); r.ping(); print('Redis connection OK')" 2>/dev/null; then
    print_status "Redis connection successful"
else
    print_warning "Redis connection test failed - service may still be initializing"
fi

# Test InfluxDB connection from strategy runner  
print_info "Testing InfluxDB connection..."
if docker exec squeezeflow-strategy-runner python -c "from influxdb import InfluxDBClient; client=InfluxDBClient(host='aggr-influx', port=8086); client.ping(); print('InfluxDB connection OK')" 2>/dev/null; then
    print_status "InfluxDB connection successful"
else
    print_warning "InfluxDB connection test failed - service may still be initializing"
fi

print_status "Strategy Runner Service Deployment Complete!"
echo ""
echo "📊 Service Information:"
echo "  • Container Name: squeezeflow-strategy-runner"
echo "  • Service Mode: strategy_runner"  
echo "  • Run Interval: 60 seconds"
echo "  • Max Symbols per Cycle: 5"
echo "  • Timeframe: 5m"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs: docker logs -f squeezeflow-strategy-runner"
echo "  • Restart: docker-compose restart strategy-runner"
echo "  • Stop: docker-compose stop strategy-runner"
echo "  • Status: docker ps | grep strategy-runner"
echo ""
echo "📈 Signal Monitoring:"
echo "  • Redis signals: docker exec squeezeflow-redis redis-cli KEYS 'squeezeflow:*'"
echo "  • InfluxDB signals: Check 'strategy_signals' measurement"
echo ""
print_status "Deployment successful! Strategy Runner is now processing signals."