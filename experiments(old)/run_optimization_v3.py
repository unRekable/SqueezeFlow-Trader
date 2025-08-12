#!/usr/bin/env python3
"""
Main runner for Optimization Framework V3
Integrates with visual validation and adaptive learning
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.optimization_framework_v3 import OptimizationFrameworkV3
from experiments.adaptive_learner import AdaptiveLearner


async def check_data_availability():
    """Check what data is available before optimization"""
    print("\n📊 Checking data availability...")
    
    import subprocess
    import os
    
    # Check ETH data
    env = os.environ.copy()
    env['INFLUX_HOST'] = '213.136.75.120'
    
    cmd = """python3 -c "
from influxdb import InfluxDBClient
client = InfluxDBClient(host='213.136.75.120', port=8086, database='significant_trades')

# Check ETH
result = client.query('SELECT COUNT(*) FROM aggr_1s.trades_1s WHERE market=\\'BINANCE:ethusdt\\' AND time > now() - 24h')
for point in result.get_points():
    print(f\\"ETH: {point.get('count', 0)} data points in last 24h\\")

# Check BTC  
result = client.query('SELECT COUNT(*) FROM aggr_1s.trades_1s WHERE market=\\'BINANCE:btcusdt\\' AND time > now() - 24h')
for point in result.get_points():
    print(f\\"BTC: {point.get('count', 0)} data points in last 24h\\")
"
"""
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout)
    
    return True


async def run_quick_optimization():
    """Quick optimization with minimal parameters"""
    
    framework = OptimizationFrameworkV3()
    
    print("\n🚀 Running quick optimization...")
    print("   Testing MIN_ENTRY_SCORE with 3 values")
    
    results = await framework.run_optimization_cycle(
        symbols=['ETH'],
        parameters={
            'MIN_ENTRY_SCORE': [2.0, 3.0, 4.0]
        }
    )
    
    # Show results
    print("\n📊 Quick Optimization Results:")
    for symbol, params in results.items():
        print(f"\n{symbol}:")
        for param, result in params.items():
            print(f"  {param}: Best={result['best_value']}, Score={result['best_score']:.1f}")
    
    return results


async def run_full_optimization():
    """Full optimization with multiple parameters"""
    
    framework = OptimizationFrameworkV3()
    learner = AdaptiveLearner()
    
    print("\n🚀 Running full optimization...")
    
    # Define parameters to test based on what's actually connected
    parameters = {
        'MIN_ENTRY_SCORE': [2.0, 3.0, 4.0, 5.0, 6.0],
        # Add more as we verify they're connected:
        # 'CVD_VOLUME_THRESHOLD': [1e5, 5e5, 1e6, 5e6],
        # 'MOMENTUM_LOOKBACK': [3, 5, 7, 10]
    }
    
    # Test on symbols we know have data
    symbols = ['ETH']  # Start with ETH, can add BTC later
    
    results = await framework.run_optimization_cycle(
        symbols=symbols,
        parameters=parameters
    )
    
    # Record learnings
    for symbol in symbols:
        for param in parameters:
            if symbol in results and param in results[symbol]:
                best_value = results[symbol][param]['best_value']
                best_score = results[symbol][param]['best_score']
                
                learner.record_learning(
                    symbol=symbol,
                    concept=f"{param}_optimization",
                    finding=f"Optimal {param}={best_value} gives score={best_score:.1f}",
                    confidence=0.7
                )
    
    # Show comprehensive results
    print("\n" + "=" * 80)
    print("FULL OPTIMIZATION RESULTS")
    print("=" * 80)
    
    for symbol, params in results.items():
        print(f"\n📈 {symbol} Optimization:")
        for param, result in params.items():
            print(f"\n  {param}:")
            print(f"    Best Value: {result['best_value']}")
            print(f"    Best Score: {result['best_score']:.1f}")
            print(f"    All Results:")
            for r in result['all_results']:
                print(f"      {r.parameter_value}: Score={r.optimization_score:.1f}, "
                     f"Trades={r.total_trades}, Win={r.win_rate:.1%}")
    
    return results


async def run_visual_analysis():
    """Analyze existing dashboards with visual validation"""
    
    from backtest.reporting.visual_validator import DashboardVisualValidator
    
    print("\n📸 Running visual analysis of recent backtests...")
    
    validator = DashboardVisualValidator("results")
    
    # Find all recent dashboards
    dashboards = list(Path("results").glob("backtest_*/dashboard.html"))
    dashboards.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not dashboards:
        print("❌ No dashboards found")
        return
    
    print(f"Found {len(dashboards)} dashboards")
    
    # Analyze up to 5 most recent
    for dashboard in dashboards[:5]:
        print(f"\n📊 Analyzing: {dashboard.parent.name}")
        
        result = validator.capture_dashboard(dashboard)
        
        if result.get('success'):
            print(f"   ✅ Screenshot: {Path(result['screenshot_path']).name}")
            print(f"   Size: {result.get('file_size', 0):,} bytes")
        else:
            print(f"   ❌ Failed: {result.get('error')}")
    
    print("\n💡 Claude can now analyze these screenshots to verify optimization results!")


async def main():
    """Main optimization runner"""
    
    parser = argparse.ArgumentParser(description='SqueezeFlow Optimization Framework V3')
    parser.add_argument('--mode', choices=['quick', 'full', 'visual', 'status'],
                       default='quick', help='Optimization mode')
    parser.add_argument('--symbols', nargs='+', default=['ETH'],
                       help='Symbols to optimize')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SQUEEZEFLOW OPTIMIZATION FRAMEWORK V3")
    print("Integrated with Visual Validation")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Always check data first
    await check_data_availability()
    
    try:
        if args.mode == 'status':
            # Show current status
            framework = OptimizationFrameworkV3()
            status = framework.get_status()
            
            print("\n📊 Optimization Status:")
            print(f"  Last run: {status['last_run']}")
            print(f"  Cycles completed: {status['cycles_completed']}")
            print(f"  Results collected: {status['results_collected']}")
            
            if status['best_parameters']:
                print("\n🏆 Best Parameters Found:")
                for symbol, params in status['best_parameters'].items():
                    print(f"\n  {symbol}:")
                    for param, info in params.items():
                        print(f"    {param}: {info['value']} (score: {info['score']:.1f})")
            
            print("\n📚 Learnings:")
            print(status['learnings'])
            
        elif args.mode == 'visual':
            # Run visual analysis only
            await run_visual_analysis()
            
        elif args.mode == 'quick':
            # Quick optimization
            await run_quick_optimization()
            
        elif args.mode == 'full':
            # Full optimization
            await run_full_optimization()
            
        print("\n✨ Done!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)