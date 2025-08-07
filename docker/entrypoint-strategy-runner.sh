#!/bin/bash
set -e

# Set timezone if provided
if [ ! -z "$TZ" ]; then
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
    echo "Timezone configured: $TZ"
fi

cd /app
export PYTHONPATH=/app

echo "=== SqueezeFlow Strategy Runner ==="
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Current time: $(date)"
echo "Timezone: $(date +%Z)"

# Wait for dependencies
echo "Waiting for Redis..."
timeout 30 sh -c 'until redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping > /dev/null 2>&1; do sleep 1; done' || echo "Warning: Redis connection timeout"

echo "Waiting for InfluxDB..."
timeout 30 sh -c 'until curl -f http://${INFLUX_HOST:-aggr-influx}:${INFLUX_PORT:-8086}/ping > /dev/null 2>&1; do sleep 2; done' || echo "Warning: InfluxDB connection timeout"

# Start the strategy runner service
echo "Starting Strategy Runner Service..."
exec python services/strategy_runner.py