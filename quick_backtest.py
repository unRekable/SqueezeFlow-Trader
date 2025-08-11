#!/usr/bin/env python3
"""Quick backtest to generate interactive dashboard with real data"""

import os
import sys
os.environ['INFLUX_HOST'] = '213.136.75.120'
os.environ['INFLUX_PORT'] = '8086'

from datetime import datetime, timedelta
import pytz
from backtest.engine import BacktestEngine
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

# Create engine with performance monitoring disabled for speed
engine = BacktestEngine(
    initial_balance=10000,
    leverage=1.0,
    enable_1s_mode=True,
    enable_performance_monitoring=False
)

# Create strategy
strategy = SqueezeFlowStrategy()

# Run SHORT backtest (2 hours only) to get quick results
print("Running 2-hour backtest to generate interactive dashboard...")
result = engine.run(
    strategy=strategy,
    symbol='BTC',
    start_date='2025-08-10',
    end_date='2025-08-10',
    timeframe='1s'
)

print(f"\nBacktest complete!")
print(f"Final balance: ${result['final_balance']:,.2f}")
print(f"Total return: {result['total_return']:.2f}%")
print(f"Report generated: {result.get('visualization_path', 'N/A')}")