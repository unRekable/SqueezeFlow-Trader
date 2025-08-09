#!/usr/bin/env python3
"""
Run full backtest with all available 1s data
"""
import subprocess
import sys
import os
from datetime import datetime

def run_full_1s_backtest():
    """Run backtest on all available 1s data"""
    
    print("=" * 70)
    print("🚀 RUNNING FULL 1-SECOND BACKTEST")
    print("=" * 70)
    
    # Use the full day to capture all available data
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
    
    print("📊 Backtest Configuration:")
    print("  Symbol: BTC")
    print("  Date: 2025-08-09")
    print("  Data Available: 08:28 to 14:28 UTC (6 hours)")
    print("  Timeframe: 1s")
    print("  Mode: 1-second stepping (adaptive windows)")
    print("  Expected Windows: ~18,000 (5 hours × 3,600 after 1h warmup)")
    print("=" * 70)
    print("")
    print("⚠️  IMPORTANT: This will evaluate the strategy EVERY SECOND")
    print("    Expect this to take 5-10 minutes to complete")
    print("")
    print("=" * 70)
    
    env = os.environ.copy()
    env['SQUEEZEFLOW_ENABLE_1S_MODE'] = 'true'
    env['SQUEEZEFLOW_RUN_INTERVAL'] = '1'
    env['SQUEEZEFLOW_DATA_INTERVAL'] = '1'
    
    backtest_dir = os.path.join(os.getcwd(), 'backtest')
    
    print("🔄 Starting backtest with 1-second granularity...")
    print("")
    
    start_time = datetime.now()
    
    try:
        # Run with real-time output
        result = subprocess.run(
            cmd,
            cwd=backtest_dir,
            env=env,
            text=True,
            capture_output=False,  # Show output in real-time
            timeout=900  # 15 minute timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("")
        print("=" * 70)
        if result.returncode == 0:
            print(f"✅ BACKTEST COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Execution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"📊 Check backtest/results for detailed HTML report")
            print(f"📈 Charts available in backtest/results/charts/")
        else:
            print(f"⚠️ Backtest exited with code: {result.returncode}")
            print(f"⏱️  Execution time: {duration:.1f} seconds")
        print("=" * 70)
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Backtest timed out after 15 minutes")
        print("This suggests an issue with 1s processing performance")
    except KeyboardInterrupt:
        print("\n⛔ Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")

if __name__ == "__main__":
    run_full_1s_backtest()