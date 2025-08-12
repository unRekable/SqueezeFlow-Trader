#!/bin/bash

# ETH Backtest Script - Uses FULL available data window
# Data available: Aug 10 09:54 to Aug 12 11:23 UTC (49.5 hours)
# Updated: 2025-08-12 to use 5m timeframe for reasonable performance

cd "/Users/u/PycharmProjects/SqueezeFlow Trader"

echo "🚀 Starting ETH backtest with FULL data window..."
echo "📊 Data range: 2025-08-10 09:54 to 2025-08-12 11:23 UTC"
echo "⏱️  Duration: 49.5 hours of market data"
echo "📈 Timeframe: 5-minute candles (594 data points)"
echo "💡 Note: System loads 1s data and aggregates to 5m internally"
echo ""

# Run backtest with 5m timeframe for reasonable performance
# The system will load 178k 1s data points and aggregate to ~594 5m candles
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py \
    --symbol ETH \
    --start-date 2025-08-10 \
    --end-date 2025-08-12 \
    --timeframe 5m \
    --balance 10000 \
    --leverage 1.0 \
    --strategy SqueezeFlowStrategy

echo ""
echo "✅ Backtest complete!"
echo "📁 Dashboard location: results/backtest_*/dashboard.html"
echo ""
echo "To view the dashboard:"
echo "  1. Look for the latest folder in results/"
echo "  2. Open dashboard.html in a browser"
echo "  3. Check all 3 tabs: Trading, Portfolio, Exchange"