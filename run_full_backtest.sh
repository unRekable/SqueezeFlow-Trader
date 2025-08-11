#!/bin/bash
# Full backtest with all available data using TradingView implementation

cd "/Users/u/PycharmProjects/SqueezeFlow Trader"

USE_TRADINGVIEW_PANES=true INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol BTC --start-date 2025-08-10 --end-date 2025-08-11 --timeframe 5m --balance 10000 --leverage 1.0 --strategy SqueezeFlowStrategy