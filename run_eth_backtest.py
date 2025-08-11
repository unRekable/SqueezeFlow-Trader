#!/usr/bin/env python3
"""Run ETH backtest with 1-hour window to generate real data quickly"""

import os
os.environ['INFLUX_HOST'] = '213.136.75.120'

from datetime import datetime
import pytz
from backtest.engine import BacktestEngine
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

print("Running 1-hour ETH backtest from 19:30 to 20:30 UTC...")
print("This should complete in ~10 seconds...")

# Create engine optimized for speed
engine = BacktestEngine(
    initial_balance=10000,
    leverage=1.0,
    enable_1s_mode=True,
    enable_performance_monitoring=False,
    enable_parallel=False  # Disable parallel for faster completion
)

# Monkey patch to limit data range for speed
original_run = engine.run

def limited_run(strategy, symbol, start_date, end_date, timeframe='1s', balance=None, leverage=None):
    # Override dates to just 1 hour
    import pytz
    from datetime import datetime
    
    # Use specific 1-hour window
    engine.start_time_override = datetime(2025, 8, 10, 19, 30, 0, tzinfo=pytz.UTC)
    engine.end_time_override = datetime(2025, 8, 10, 20, 30, 0, tzinfo=pytz.UTC)
    
    return original_run(strategy, symbol, start_date, end_date, timeframe, balance, leverage)

engine.run = limited_run

# Run backtest
strategy = SqueezeFlowStrategy()
result = engine.run(
    strategy=strategy,
    symbol='ETH',
    start_date='2025-08-10',
    end_date='2025-08-10',
    timeframe='1s'
)

print(f"\n✅ Backtest complete!")
print(f"Final balance: ${result['final_balance']:,.2f}")
print(f"Total return: {result['total_return']:.2f}%")
print(f"Total trades: {result.get('total_trades', 0)}")
print(f"\n📊 Report generated: {result.get('visualization_path', 'N/A')}")