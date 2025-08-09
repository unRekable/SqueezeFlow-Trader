#!/usr/bin/env python3
"""
Fast 1-second backtest with optimizations
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from backtest.engine import BacktestEngine
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

def main():
    print("\n" + "="*60)
    print("🚀 OPTIMIZED 1s BACKTEST")
    print("="*60)
    
    # Use today's date for testing
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create strategy with updated scoring thresholds
    strategy = SqueezeFlowStrategy()
    
    # Create backtest engine with optimizations
    engine = BacktestEngine(
        initial_balance=10000,
        leverage=1.0,
        strategy=strategy,
        enable_parallel=False,  # Disable parallel for now to reduce overhead
        enable_1s_mode=True,    # Enable 1s optimizations
        max_memory_gb=4.0,
        chunk_hours=4           # Load data in smaller chunks
    )
    
    # Run backtest for today with 1s timeframe
    print(f"\n📅 Running backtest for {today} with 1s timeframe...")
    print(f"⚙️  Min entry score: {strategy.config.min_entry_score}")
    print(f"🎯 Scoring weights: {strategy.config.scoring_weights}")
    
    start_time = time.time()
    
    results = engine.run_backtest(
        symbol='BTC',
        start_date=today,
        end_date=today,
        timeframe='1s'
    )
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    
    if results:
        trades = results.get('trades', [])
        metrics = results.get('metrics', {})
        
        print(f"⏱️  Time elapsed: {elapsed:.1f} seconds")
        print(f"📈 Total trades: {len(trades)}")
        print(f"💰 Final balance: ${metrics.get('final_balance', 0):,.2f}")
        print(f"📊 Total return: {metrics.get('total_return', 0):.2%}")
        print(f"🎯 Win rate: {metrics.get('win_rate', 0):.1%}")
        print(f"📉 Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        # Show trade distribution
        if trades:
            print(f"\n📊 Trade Distribution:")
            hours = {}
            for trade in trades:
                if 'timestamp' in trade:
                    hour = trade['timestamp'].hour
                    hours[hour] = hours.get(hour, 0) + 1
            
            for hour in sorted(hours.keys()):
                print(f"  Hour {hour:02d}: {'█' * hours[hour]} ({hours[hour]} trades)")
    else:
        print("❌ Backtest failed")
    
    print("\n" + "="*60)
    print(f"✅ Backtest completed in {elapsed:.1f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()