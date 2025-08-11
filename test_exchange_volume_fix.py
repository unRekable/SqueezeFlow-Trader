#!/usr/bin/env python3
"""Test exchange volume fix in dashboard"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Set environment variables
os.environ['INFLUX_HOST'] = '213.136.75.120'
os.environ['INFLUX_PORT'] = '8086'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

print("Testing exchange volume fix with 10 minutes of BTC data...")

# Create engine with correct parameters
engine = BacktestEngine(
    initial_balance=10000.0,
    leverage=1.0,
    enable_1s_mode=True
)

# Create strategy instance
strategy = SqueezeFlowStrategy()

# Run backtest with correct parameters
result = engine.run(
    strategy=strategy,
    symbol='BTC',
    start_date=datetime(2025, 8, 10, 20, 0),
    end_date=datetime(2025, 8, 10, 20, 10),
    timeframe='1s'
)

if result and 'visualization_path' in result:
    print(f"\n✅ Dashboard created: {result['visualization_path']}")
    print("\nCheck the Exchange tab to verify volume data is now displayed!")
else:
    print("\n❌ Failed to create dashboard")