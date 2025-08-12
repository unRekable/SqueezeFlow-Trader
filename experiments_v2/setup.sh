#!/bin/bash

# Setup script for Experiments V2 - Insight Engine

echo "=========================================="
echo "    Experiments V2 - Setup"
echo "=========================================="
echo ""

# Set required environment variables
export INFLUX_HOST=213.136.75.120
export INFLUX_PORT=8086
export INFLUX_DATABASE=significant_trades

echo "✅ Environment variables set:"
echo "   INFLUX_HOST=$INFLUX_HOST"
echo "   INFLUX_PORT=$INFLUX_PORT"
echo "   INFLUX_DATABASE=$INFLUX_DATABASE"
echo ""

# Create required directories if they don't exist
mkdir -p insights
mkdir -p adaptive_learning
echo "✅ Required directories created"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✅ Python version: $python_version"
echo ""

# Quick connectivity test
echo "🔍 Testing InfluxDB connection..."
python3 -c "
from influxdb import InfluxDBClient
try:
    client = InfluxDBClient(host='213.136.75.120', port=8086, database='significant_trades')
    result = client.query('SELECT COUNT(*) FROM aggr_1s.trades_1s WHERE time > now() - 1h')
    points = list(result.get_points())
    if points:
        print('✅ InfluxDB connection successful!')
        print(f'   Recent data points: {points[0].get(\"count_close\", 0)}')
    else:
        print('⚠️ Connected but no recent data found')
except Exception as e:
    print(f'❌ Connection failed: {e}')
" 2>/dev/null || echo "⚠️ InfluxDB client not installed - run: pip3 install influxdb"

echo ""
echo "=========================================="
echo "Setup complete! You can now use:"
echo ""
echo "  python3 run_insight_engine.py --mode analyze"
echo "  python3 run_insight_engine.py --mode test"
echo "  python3 run_insight_engine.py --mode optimize"
echo "  python3 run_insight_engine.py --mode report"
echo ""
echo "=========================================="