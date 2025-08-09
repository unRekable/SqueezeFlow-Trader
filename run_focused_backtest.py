#!/usr/bin/env python3
"""
Focused 1-second backtest for the available data window
"""

import subprocess
import sys
import os

def run_focused_1s_backtest():
    """Run backtest on the exact data we know is available"""
    
    # Parameters for known good data
    symbol = "BTC"
    start_date = "2025-08-09"
    end_date = "2025-08-09"  # Same day but backtest engine will find data
    timeframe = "1s"
    balance = 10000
    strategy = "SqueezeFlowStrategy"
    
    print(f"🚀 Running Focused 1s Backtest")
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Date: {start_date}")
    print(f"⏱️  Timeframe: {timeframe}")
    print(f"💰 Balance: ${balance:,}")
    print(f"⚙️  Strategy: {strategy}")
    print("=" * 60)
    
    # Change to backtest directory
    current_dir = os.getcwd()
    backtest_dir = os.path.join(current_dir, 'backtest')
    
    # Build command
    cmd = [
        sys.executable, 'engine.py',
        '--symbol', symbol,
        '--start-date', start_date,
        '--end-date', end_date,
        '--timeframe', timeframe,
        '--balance', str(balance),
        '--strategy', strategy,
        '--enable-1s-mode',
        '--max-memory-gb', '4.0'
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    print(f"📂 Working directory: {backtest_dir}")
    print("=" * 60)
    
    try:
        # Run backtest
        result = subprocess.run(
            cmd,
            cwd=backtest_dir,
            text=True,
            timeout=300,  # 5 minute timeout
            capture_output=False  # Show output in real-time
        )
        
        if result.returncode == 0:
            print("\n✅ Backtest completed successfully!")
        else:
            print(f"\n⚠️ Backtest completed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Backtest timed out after 5 minutes")
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")

if __name__ == "__main__":
    run_focused_1s_backtest()