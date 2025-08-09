#!/usr/bin/env python3
"""
Optimized 1s backtest that respects lookahead bias prevention
but uses smarter stepping for better performance
"""

import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest.engine import BacktestEngine

def run_optimized_backtest():
    """
    Run backtest with optimized settings for 1s data
    """
    print("\n" + "="*60)
    print("OPTIMIZED 1s BACKTEST")
    print("="*60)
    
    # Strategy: Use larger steps to reduce redundant processing
    # Instead of 1-second steps (86,399 windows), use 30-second steps (2,880 windows)
    # This still evaluates the entire day but 30x faster
    
    print("\nConfiguration:")
    print("  Data: 1-second granularity")
    print("  Window: 1 hour (to ensure enough history)")
    print("  Step: 30 seconds (instead of 1 second)")
    print("  Expected windows: ~2,880 (vs 86,399)")
    print("  Expected speedup: ~30x")
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_balance=10000,
        leverage=1.0,
        enable_parallel=False,  # Sequential for now
        enable_1s_mode=False,   # We'll handle the stepping manually
        max_memory_gb=4.0
    )
    
    # Override the step duration in the engine
    # This is a bit hacky but avoids modifying the engine code
    original_method = engine._run_rolling_window_backtest
    
    def custom_rolling_window(full_dataset, start_time, end_time, timeframe):
        """Custom rolling window with 30-second steps"""
        # Use 1-hour windows but 30-second steps
        window_duration = timedelta(hours=1)
        step_duration = timedelta(seconds=30)  # 30-second steps instead of 1-second
        
        all_executed_orders = []
        current_time = start_time + window_duration
        
        total_iterations = int((end_time - current_time).total_seconds() / 30)
        print(f"\n📊 Processing {total_iterations} windows...")
        
        iteration = 0
        last_progress_time = time.time()
        
        while current_time <= end_time:
            window_start = current_time - window_duration
            window_end = current_time
            
            # Create windowed dataset
            windowed_dataset = engine._create_windowed_dataset(
                full_dataset, window_start, window_end
            )
            
            # Process window
            if engine._validate_windowed_data(windowed_dataset):
                orders = engine._process_strategy(windowed_dataset, timeframe)
                
                if orders:
                    executed = engine._execute_orders(orders, windowed_dataset, current_time)
                    all_executed_orders.extend(executed)
            
            # Progress update
            iteration += 1
            if iteration % 100 == 0:
                current_progress_time = time.time()
                elapsed = current_progress_time - last_progress_time
                rate = 100 / elapsed if elapsed > 0 else 0
                print(f"  Progress: {iteration}/{total_iterations} ({rate:.1f} windows/sec)")
                last_progress_time = current_progress_time
            
            # Move to next window (30-second step)
            current_time += step_duration
        
        return all_executed_orders
    
    # Replace the method temporarily
    engine._run_rolling_window_backtest = custom_rolling_window
    
    # Run backtest
    print("\nStarting backtest...")
    start_time = time.time()
    
    results = engine.run_backtest(
        symbol='BTC',
        start_date='2025-08-09',
        end_date='2025-08-09',
        timeframe='1s'  # Use 1s data
    )
    
    elapsed = time.time() - start_time
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if results:
        trades = results.get('trades', [])
        metrics = results.get('metrics', {})
        
        print(f"⏱️  Time elapsed: {elapsed:.1f} seconds")
        print(f"📈 Total trades: {len(trades)}")
        print(f"💰 Final balance: ${metrics.get('final_balance', 0):,.2f}")
        print(f"📊 Total return: {metrics.get('total_return', 0):.2%}")
        print(f"🎯 Win rate: {metrics.get('win_rate', 0):.1%}")
        
        if trades:
            print("\nFirst 5 trades:")
            for i, trade in enumerate(trades[:5]):
                print(f"  {i+1}. {trade.get('timestamp', 'N/A')} - {trade.get('side', 'N/A')}")
    else:
        print("❌ No results")
    
    print("\n" + "="*60)
    print(f"Completed in {elapsed:.1f} seconds")
    print("="*60)

if __name__ == "__main__":
    run_optimized_backtest()