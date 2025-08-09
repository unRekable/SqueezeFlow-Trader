#!/usr/bin/env python3
"""
Run backtest starting from actual data availability
"""
import subprocess
import sys
import os
from datetime import datetime, timedelta
import pytz

def run_backtest_from_data():
    """Run backtest from where data actually exists"""
    
    print("=" * 70)
    print("🚀 RUNNING 1-SECOND BACKTEST FROM ACTUAL DATA")
    print("=" * 70)
    
    # First, find when data actually starts
    sys.path.insert(0, '/Users/u/PycharmProjects/SqueezeFlow Trader')
    from data.pipeline import DataPipeline
    
    pipeline = DataPipeline()
    check_date = datetime(2025, 8, 9, tzinfo=pytz.UTC)
    dataset = pipeline.get_complete_dataset(
        symbol='BTC',
        start_time=check_date,
        end_time=check_date + timedelta(days=1),
        timeframe='1s'
    )
    
    # Find actual data range
    if dataset and 'ohlcv' in dataset:
        df = dataset['ohlcv']
        non_zero = df[(df['close'] > 0) & (df['volume'] > 0)]
        if len(non_zero) > 0:
            actual_start = non_zero.index[0]
            actual_end = non_zero.index[-1]
            
            # Adjust start time to account for 1-hour window warmup
            backtest_start = actual_start - timedelta(hours=1)
            
            print(f"📊 Data Analysis:")
            print(f"  First data: {actual_start}")
            print(f"  Last data: {actual_end}")
            print(f"  Data points: {len(non_zero):,}")
            print(f"  Duration: {(actual_end - actual_start).total_seconds() / 3600:.2f} hours")
            print("")
            print(f"🎯 Backtest Configuration:")
            print(f"  Start: {backtest_start} (1h before data for warmup)")
            print(f"  End: {actual_end}")
            
            # We need to use dates only for the backtest engine
            # The engine will find the data within those dates
            cmd = [
                sys.executable, 'engine.py',
                '--symbol', 'BTC',
                '--start-date', '2025-08-09',
                '--end-date', '2025-08-09', 
                '--timeframe', '1s',
                '--balance', '10000',
                '--leverage', '1.0',
                '--strategy', 'SqueezeFlowStrategy',
                '--enable-1s-mode',
                '--max-memory-gb', '4.0'
            ]
            
            print("")
            print("⚡ Running with 1-second granularity...")
            print("  Mode: 1-hour windows, 1-second steps")
            print(f"  Expected evaluations: ~{len(non_zero) - 3600:,}")
            print("=" * 70)
            print("")
            
            env = os.environ.copy()
            env['SQUEEZEFLOW_ENABLE_1S_MODE'] = 'true'
            
            backtest_dir = os.path.join(os.getcwd(), 'backtest')
            
            start_time = datetime.now()
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=backtest_dir,
                    env=env,
                    text=True,
                    capture_output=False,
                    timeout=900
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                print("")
                print("=" * 70)
                if result.returncode == 0:
                    print(f"✅ BACKTEST COMPLETED!")
                    print(f"⏱️ Execution time: {duration:.1f} seconds")
                else:
                    print(f"⚠️ Backtest exited with code: {result.returncode}")
                print("=" * 70)
                    
            except subprocess.TimeoutExpired:
                print("\n⏰ Backtest timed out")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        else:
            print("❌ No non-zero data found")
    else:
        print("❌ Failed to load dataset")

if __name__ == "__main__":
    run_backtest_from_data()