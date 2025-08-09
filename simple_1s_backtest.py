#!/usr/bin/env python3
"""
Simple 1s backtest without unnecessary windowing complexity
"""

import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.pipeline import DataPipeline
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

def simple_1s_backtest(symbol='BTC', date='2025-08-09'):
    """
    Simple approach: Load all data, step through it second by second
    """
    print(f"\n{'='*60}")
    print(f"SIMPLE 1s BACKTEST - {date}")
    print(f"{'='*60}\n")
    
    # Load the day's data
    pipeline = DataPipeline()
    start_time = datetime.strptime(date, '%Y-%m-%d')
    end_time = start_time + timedelta(days=1)
    
    print("Loading data...")
    dataset = pipeline.get_complete_dataset(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        timeframe='1s'
    )
    
    if not dataset or 'ohlcv' not in dataset:
        print("❌ No data available")
        return
        
    ohlcv = dataset['ohlcv']
    print(f"✅ Loaded {len(ohlcv)} data points\n")
    
    # Initialize strategy
    strategy = SqueezeFlowStrategy()
    print(f"Strategy config:")
    print(f"  Min entry score: {strategy.config.min_entry_score}")
    print(f"  Timeframes: {strategy.config.reset_timeframes}\n")
    
    # Simple portfolio state
    portfolio = {
        'cash': 10000,
        'positions': {},
        'trades': []
    }
    
    # Minimum data requirements (30 minutes for largest timeframe)
    min_lookback = 30 * 60  # 30 minutes in seconds
    
    print(f"Starting backtest from position {min_lookback}...")
    start_backtest = time.time()
    
    trades = []
    signals_checked = 0
    
    # Step through each second (after we have enough history)
    for i in range(min_lookback, len(ohlcv)):
        current_time = ohlcv.index[i]
        
        # Get the data windows the strategy needs
        # Strategy wants different timeframe views, all ending at current_time
        window_30m = i - (30 * 60) if i >= (30 * 60) else 0
        window_15m = i - (15 * 60) if i >= (15 * 60) else 0
        window_5m = i - (5 * 60) if i >= (5 * 60) else 0
        
        # Create dataset for strategy (just the windows it needs)
        strategy_dataset = {
            'symbol': symbol,
            'ohlcv': ohlcv[window_30m:i+1],  # Up to current second
            'spot_cvd': dataset.get('spot_cvd', pd.Series())[window_30m:i+1],
            'futures_cvd': dataset.get('futures_cvd', pd.Series())[window_30m:i+1],
            'cvd_divergence': dataset.get('cvd_divergence', pd.Series())[window_30m:i+1],
            'timeframe': '1s'
        }
        
        # Run strategy
        result = strategy.process(strategy_dataset, portfolio)
        signals_checked += 1
        
        # Check for signals
        if result and 'orders' in result and result['orders']:
            for order in result['orders']:
                trades.append({
                    'time': current_time,
                    'type': order.get('side'),
                    'price': ohlcv.iloc[i]['close']
                })
                print(f"  📊 Trade at {current_time}: {order.get('side')}")
        
        # Progress update
        if signals_checked % 3600 == 0:  # Every hour
            elapsed = time.time() - start_backtest
            rate = signals_checked / elapsed
            print(f"  Progress: {signals_checked}/{len(ohlcv)-min_lookback} ({rate:.0f} signals/sec)")
    
    # Results
    elapsed = time.time() - start_backtest
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"⏱️  Time: {elapsed:.1f} seconds")
    print(f"📊 Signals checked: {signals_checked:,}")
    print(f"🚀 Rate: {signals_checked/elapsed:.0f} signals/second")
    print(f"📈 Trades generated: {len(trades)}")
    
    if trades:
        print(f"\nTrade distribution by hour:")
        hours = {}
        for t in trades:
            hour = t['time'].hour
            hours[hour] = hours.get(hour, 0) + 1
        for h in sorted(hours.keys()):
            print(f"  Hour {h:02d}: {'█' * hours[h]} ({hours[h]} trades)")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    simple_1s_backtest()