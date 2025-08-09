#!/bin/bash
# Run 1s backtest with optimized settings

echo "========================================"
echo "1s BACKTEST WITH OPTIMIZED SETTINGS"
echo "========================================"
echo ""
echo "Strategy Configuration:"
echo "  Min Entry Score: 5.5 (raised from 4.0)"
echo "  CVD Movement Threshold: 10-20x stricter"
echo "  Expected trades: 20-40 per day (vs 160+)"
echo ""
echo "Backtest Configuration:"
echo "  Using 1-minute data (faster than 1s)"
echo "  4-hour windows, 5-minute steps"
echo "  Testing 4 hours of data (08:00-12:00)"
echo ""
echo "Starting backtest..."
echo "========================================"

cd /Users/u/PycharmProjects/SqueezeFlow\ Trader/backtest

# Run for 4 hours with 1-minute data (much faster)
python3 engine.py \
    --symbol BTC \
    --start-date 2025-08-09 \
    --start-time 08:00:00 \
    --end-date 2025-08-09 \
    --end-time 12:00:00 \
    --balance 10000 \
    --strategy SqueezeFlowStrategy \
    --disable-parallel \
    --timeframe 1m

echo ""
echo "========================================"
echo "Backtest complete!"
echo "========================================"