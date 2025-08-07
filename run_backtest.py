#!/usr/bin/env python3
"""
SqueezeFlow Backtest Runner
Easy-to-use interface for running backtests with the enhanced modular engine
"""

import asyncio
import json
from datetime import datetime, timedelta
import sys
import os
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_date_range_suggestions():
    """Get suggested date ranges based on available data"""
    now = datetime.now()
    
    return {
        "yesterday": {
            "start": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d"),
            "description": "Yesterday"
        },
        "last_24hours": {
            "start": (now - timedelta(hours=24)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d"),
            "description": "Last 24 hours"
        },
        "last_week": {
            "start": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d"),
            "description": "Last 7 days"
        },
        "last_month": {
            "start": (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d"),
            "description": "Last 30 days"
        },
        "last_3_months": {
            "start": (now - timedelta(days=90)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d"),
            "description": "Last 3 months"
        },
        "january_2025": {
            "start": "2025-01-01",
            "end": "2025-01-31", 
            "description": "January 2025"
        },
        "december_2024": {
            "start": "2024-12-01",
            "end": "2024-12-31",
            "description": "December 2024"
        }
    }


async def run_quick_backtest(period_key: str = "last_week", balance: float = 10000, strategy: str = "production_enhanced_strategy", leverage: float = 1.0):
    """Run a quick backtest with predefined periods using the modular engine"""
    
    periods = get_date_range_suggestions()
    
    if period_key not in periods:
        print(f"❌ Invalid period. Available: {list(periods.keys())}")
        return
        
    period = periods[period_key]
    
    print(f"🚀 Running Enhanced SqueezeFlow Backtest: {period['description']}")
    print(f"📅 Period: {period['start']} to {period['end']}")
    print(f"💰 Initial Balance: ${balance:,.2f}")
    print(f"📈 Leverage: {leverage}x")
    print(f"⚙️ Strategy: {strategy}")
    print(f"{'='*60}")
    
    try:
        # Change to backtest directory for proper imports
        current_dir = os.getcwd()
        backtest_dir = os.path.join(current_dir, 'backtest')
        
        # Build command to run the modular engine
        cmd = [
            sys.executable, 'engine.py',
            '--start-date', period['start'],
            '--end-date', period['end'],
            '--balance', str(balance),
            '--leverage', str(leverage),
            '--strategy', strategy
        ]
        
        print(f"📋 Running command: {' '.join(cmd)}")
        print(f"📂 Working directory: {backtest_dir}")
        print(f"{'='*60}")
        
        # Run the backtest engine
        result = subprocess.run(
            cmd,
            cwd=backtest_dir,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print the output
        if result.stdout:
            print("📊 BACKTEST OUTPUT:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ WARNINGS/ERRORS:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Backtest completed successfully!")
            print(f"📁 Check backtest/ directory for generated charts and results")
        else:
            print(f"\n❌ Backtest failed with exit code: {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Backtest runner failed: {e}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return False


def print_backtest_results(report):
    """Print formatted backtest results"""
    
    if 'error' in report:
        print(f"❌ {report['error']}")
        return
        
    print(f"\n{'='*60}")
    print(f"🎯 SQUEEZEFLOW BACKTEST RESULTS")
    print(f"{'='*60}")
    
    # Performance metrics
    print(f"📊 PERFORMANCE SUMMARY")
    print(f"   Period: {report['backtest_period']}")
    print(f"   Initial Balance: ${report['initial_balance']:,.2f}")
    print(f"   Final Balance: ${report['final_balance']:,.2f}")
    
    return_color = "🟢" if report['total_return_pct'] > 0 else "🔴"
    print(f"   Total Return: {return_color} {report['total_return_pct']:.2f}% (${report['total_return_usd']:,.2f})")
    
    # Trading metrics
    print(f"\n📈 TRADING METRICS")
    print(f"   Total Trades: {report['total_trades']}")
    
    if report['total_trades'] > 0:
        print(f"   Win Rate: {report['win_rate_pct']:.1f}% ({report['winning_trades']}/{report['total_trades']})")
        print(f"   Average Trade: {report['avg_trade_return_pct']:.2f}%")
        print(f"   Best Trade: 🟢 {report['max_win_pct']:.2f}%")
        print(f"   Worst Trade: 🔴 {report['max_loss_pct']:.2f}%")
        print(f"   Average Duration: {report['avg_duration_minutes']:.1f} minutes")
        print(f"   Trades per Day: {report['trades_per_day']:.1f}")
        
        # Strategy config
        print(f"\n⚙️ STRATEGY CONFIG")
        config = report['strategy_config']
        print(f"   Squeeze Threshold: {config['squeeze_threshold']}")
        print(f"   Max Open Trades: {config['max_open_trades']}")
        print(f"   Symbols: {', '.join(config['symbols'])}")
        print(f"   Timeframes: {config['timeframes']}")
        
        # Recent trades summary
        if report['trade_history']:
            print(f"\n📋 RECENT TRADES (Last 5)")
            recent_trades = report['trade_history'][-5:]
            
            for i, trade in enumerate(recent_trades, 1):
                side_emoji = "🟢" if trade['side'] == 'long' else "🔴"
                pnl_emoji = "💚" if trade['pnl_pct'] > 0 else "💔"
                
                print(f"   {i}. {side_emoji} {trade['symbol']} {trade['side'].upper()}")
                print(f"      Entry: {trade['entry_price']:.2f} @ {trade['entry_time']}")
                print(f"      Exit: {trade['exit_price']:.2f} @ {trade['exit_time']}")
                print(f"      PnL: {pnl_emoji} {trade['pnl_pct']:.2f}% ({trade['duration_minutes']:.0f}min)")
                print(f"      Reason: {trade['exit_reason']}")
                print()
    
    print(f"{'='*60}")


async def main():
    """Main function with enhanced command line interface"""
    
    print("🚀 Enhanced SqueezeFlow Backtest Engine")
    print("Advanced Strategy Implementation with Multi-Timeframe Analysis")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("📖 USAGE:")
        print("  python run_backtest.py <period> [balance] [strategy] [leverage]")
        print()
        print("📅 AVAILABLE PERIODS:")
        
        periods = get_date_range_suggestions()
        for key, info in periods.items():
            print(f"   {key:<15} - {info['description']} ({info['start']} to {info['end']})")
            
        print()
        print("⚙️ AVAILABLE STRATEGIES:")
        print("   production_enhanced_strategy - OPTIMIZED implementation with:")
        print("     • Advanced State Machine System")
        print("     • Multi-Timeframe Analysis (5min-240min)")
        print("     • Dynamic Trailing Stops & ROI Management")
        print("     • Risk-Adjusted Leverage System")
        print("     • Real-Time Performance Validation")
        print()
        print("💡 EXAMPLES:")
        print("   python run_backtest.py last_week")
        print("   python run_backtest.py last_month 20000")
        print("   python run_backtest.py january_2025 50000")
        print("   python run_backtest.py last_week 15000 production_enhanced_strategy")
        print("   python run_backtest.py last_week 10000 production_enhanced_strategy 2.0  # 2x leverage")
        
        return
        
    period_key = sys.argv[1]
    balance = float(sys.argv[2]) if len(sys.argv) > 2 else 10000
    strategy = sys.argv[3] if len(sys.argv) > 3 else "production_enhanced_strategy"
    leverage = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    # Run backtest
    await run_quick_backtest(period_key, balance, strategy, leverage)


if __name__ == "__main__":
    asyncio.run(main())