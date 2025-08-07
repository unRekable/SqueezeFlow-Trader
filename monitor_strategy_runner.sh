#!/bin/bash
# Strategy Runner Service Monitor
# Comprehensive health monitoring for the Strategy Runner Service

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}$1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if continuous monitoring was requested
CONTINUOUS=false
if [[ "$1" == "--continuous" || "$1" == "-c" ]]; then
    CONTINUOUS=true
    INTERVAL=${2:-30}  # Default 30 seconds
fi

monitor_once() {
    clear
    print_header "SqueezeFlow Strategy Runner - Health Monitor"
    echo "Timestamp: $(date)"
    echo ""

    # 1. Container Status
    print_header "Container Status"
    if docker ps | grep -q "squeezeflow-strategy-runner"; then
        container_status=$(docker inspect --format='{{.State.Status}}' squeezeflow-strategy-runner 2>/dev/null)
        container_health=$(docker inspect --format='{{.State.Health.Status}}' squeezeflow-strategy-runner 2>/dev/null)
        
        if [[ "$container_status" == "running" ]]; then
            print_status "Container is running"
            if [[ "$container_health" == "healthy" ]]; then
                print_status "Health check: healthy"
            elif [[ "$container_health" == "unhealthy" ]]; then
                print_error "Health check: unhealthy"
            else
                print_warning "Health check: $container_health"
            fi
        else
            print_error "Container status: $container_status"
        fi
        
        # Container resource usage
        container_stats=$(docker stats --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" squeezeflow-strategy-runner 2>/dev/null | tail -1)
        if [[ -n "$container_stats" ]]; then
            echo "  Resource usage: $container_stats"
        fi
    else
        print_error "Strategy Runner container is not running"
        return 1
    fi

    # 2. Service Dependencies
    print_header "Service Dependencies"
    
    # Check Redis
    if docker exec squeezeflow-strategy-runner python -c "import redis; r=redis.Redis(host='redis', port=6379); r.ping()" 2>/dev/null; then
        print_status "Redis connection: OK"
        
        # Check for recent signals
        signal_count=$(docker exec squeezeflow-redis redis-cli KEYS "squeezeflow:*" 2>/dev/null | wc -l)
        echo "  Active signals in Redis: $signal_count"
    else
        print_error "Redis connection: Failed"
    fi
    
    # Check InfluxDB
    if docker exec squeezeflow-strategy-runner python -c "from influxdb import InfluxDBClient; client=InfluxDBClient(host='aggr-influx', port=8086); client.ping()" 2>/dev/null; then
        print_status "InfluxDB connection: OK"
        
        # Check recent signals in InfluxDB
        recent_signals=$(docker exec squeezeflow-trader-aggr-influx-1 influx -execute "SELECT COUNT(*) FROM strategy_signals WHERE time > now() - 1h" -database="significant_trades" 2>/dev/null | tail -1 | awk '{print $2}')
        if [[ -n "$recent_signals" && "$recent_signals" != "0" ]]; then
            echo "  Signals in last hour: $recent_signals"
        else
            print_warning "No signals generated in the last hour"
        fi
    else
        print_error "InfluxDB connection: Failed"
    fi

    # 3. Service Performance
    print_header "Service Performance"
    
    # Get recent logs for performance info
    recent_logs=$(docker logs --tail 5 squeezeflow-strategy-runner 2>/dev/null | grep -E "(Performance|Cycle|Signal)" | tail -3)
    if [[ -n "$recent_logs" ]]; then
        echo "Recent performance logs:"
        echo "$recent_logs" | sed 's/^/  /'
    else
        print_warning "No recent performance logs found"
    fi

    # 4. Service Logs (Last 10 lines)
    print_header "Recent Service Logs"
    docker logs --tail 10 squeezeflow-strategy-runner 2>/dev/null | sed 's/^/  /'

    # 5. System Resources
    print_header "System Resources"
    
    # Docker system info
    docker_info=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}" 2>/dev/null)
    if [[ -n "$docker_info" ]]; then
        echo "Docker system usage:"
        echo "$docker_info" | sed 's/^/  /'
    fi
    
    echo ""
}

# Main execution
if [[ "$CONTINUOUS" == true ]]; then
    echo "Starting continuous monitoring (every ${INTERVAL}s). Press Ctrl+C to stop..."
    while true; do
        monitor_once
        echo ""
        echo -e "${BLUE}Next update in ${INTERVAL} seconds...${NC}"
        sleep "$INTERVAL"
    done
else
    monitor_once
fi