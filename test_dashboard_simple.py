#!/usr/bin/env python3
"""Simple test of dashboard with exchange volume fix"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from backtest.reporting.tradingview_unified import TradingViewUnified

# Create mock data
results = {
    'symbol': 'BTC',
    'total_return': 1.5,
    'win_rate': 55.0,
    'sharpe_ratio': 1.2,
    'max_drawdown': -5.0,
    'executed_orders': [
        {'timestamp': datetime(2025, 8, 10, 12, 0), 'side': 'buy', 'price': 59000, 'pnl': 0},
        {'timestamp': datetime(2025, 8, 10, 13, 0), 'side': 'sell', 'price': 59100, 'pnl': 100}
    ]
}

# Create mock dataset with volume data
times = pd.date_range(start='2025-08-10 12:00', end='2025-08-10 14:00', freq='5min')
ohlcv = pd.DataFrame({
    'open': [59000] * len(times),
    'high': [59100] * len(times),
    'low': [58900] * len(times),
    'close': [59050] * len(times),
    'volume': [100] * len(times)
}, index=times)

# Create volume data with the columns that exist
spot_volume = pd.DataFrame({
    'total_vbuy_spot': [1000] * len(times),
    'total_vsell_spot': [900] * len(times),
    'total_cbuy_spot': [50] * len(times),
    'total_csell_spot': [45] * len(times)
}, index=times)

futures_volume = pd.DataFrame({
    'total_vbuy_futures': [2000] * len(times),
    'total_vsell_futures': [1800] * len(times),
    'total_cbuy_futures': [100] * len(times),
    'total_csell_futures': [90] * len(times)
}, index=times)

dataset = {
    'ohlcv': ohlcv,
    'spot_volume': spot_volume,
    'futures_volume': futures_volume,
    'spot_cvd': pd.Series([100] * len(times), index=times),
    'futures_cvd': pd.Series([200] * len(times), index=times)
}

# Create dashboard
output_dir = Path('results/test_dashboard')
output_dir.mkdir(parents=True, exist_ok=True)

tv_viz = TradingViewUnified()
dashboard_path = tv_viz.create_dashboard(results, dataset, str(output_dir))

print(f"✅ Dashboard created: {dashboard_path}")
print("\nOpen the dashboard and check the Exchange tab to see if volume data is displayed!")