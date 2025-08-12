#!/bin/bash

# ETH Backtest Script - Tests all available 1-second data
# Data range: Aug 10-11, 2025 (about 2 days of 1s data)

cd "/Users/u/PycharmProjects/SqueezeFlow Trader"

echo "🚀 Starting ETH backtest with 1-second data..."
echo "📊 Processing ~172,800 data points (2 days)"
echo ""

INFLUX_HOST=213.136.75.120 python3 backtest/engine.py \
    --symbol ETH \
    --start-date 2025-08-10 \
    --end-date 2025-08-11 \
    --balance 10000 \
    --leverage 1.0 \
    --strategy SqueezeFlowStrategy

echo ""
echo "✅ Backtest complete! Check results/ folder for dashboard.html"