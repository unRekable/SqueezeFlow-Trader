#!/usr/bin/env python3
"""
Run the Insight Engine - Practical Optimization Tool

Usage:
    python3 run_insight_engine.py --mode analyze    # Analyze system issues
    python3 run_insight_engine.py --mode test       # Run test backtest
    python3 run_insight_engine.py --mode optimize   # Find optimal parameters
    python3 run_insight_engine.py --mode report     # Generate comprehensive report
"""

import asyncio
import argparse
import os
from pathlib import Path
from datetime import datetime

from insight_engine import InsightEngine


async def analyze_mode(engine: InsightEngine):
    """Analyze system for issues"""
    print("\n🔍 SYSTEM ANALYSIS")
    print("="*50)
    
    issues = engine.analyze_system_issues()
    
    if issues['critical']:
        print("\n🚨 Critical Issues:")
        for issue in issues['critical']:
            print(f"  - {issue}")
    else:
        print("\n✅ No critical issues found")
        
    if issues['warnings']:
        print("\n⚠️ Warnings:")
        for warning in issues['warnings']:
            print(f"  - {warning}")
    else:
        print("✅ No warnings")
        
    if issues['suggestions']:
        print("\n💡 Suggestions:")
        for suggestion in issues['suggestions']:
            print(f"  - {suggestion}")
            

async def test_mode(engine: InsightEngine):
    """Run a test backtest with current settings"""
    print("\n🧪 TEST BACKTEST")
    print("="*50)
    
    # Set environment variable
    os.environ['INFLUX_HOST'] = '213.136.75.120'
    
    print("Running backtest for ETH on 2025-08-10...")
    print("(This will take about 30 seconds)")
    
    insight = await engine.analyze_backtest(
        symbol='ETH',
        parameters={},  # Use defaults
        date_range=('2025-08-10', '2025-08-10')
    )
    
    print(f"\n📊 Results:")
    print(f"  Total trades: {insight.total_trades}")
    print(f"  Win rate: {insight.win_rate:.1%}")
    print(f"  Total return: {insight.total_return:.2%}")
    print(f"  Sharpe ratio: {insight.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {insight.max_drawdown:.2%}")
    print(f"  Quality score: {insight.quality_score:.1f}/100")
    
    if insight.structural_improvements:
        print(f"\n💡 Recommendations:")
        for improvement in insight.structural_improvements:
            print(f"  - {improvement}")
            
    if insight.parameter_adjustments:
        print(f"\n🔧 Suggested parameter adjustments:")
        for param, value in insight.parameter_adjustments.items():
            print(f"  {param} = {value}")
            

async def optimize_mode(engine: InsightEngine):
    """Find optimal parameters"""
    print("\n🎯 PARAMETER OPTIMIZATION")
    print("="*50)
    
    # Set environment variable
    os.environ['INFLUX_HOST'] = '213.136.75.120'
    
    print("Testing parameter combinations...")
    print("(This will run multiple backtests - about 2-3 minutes)")
    
    # Define search space
    parameter_ranges = {
        'MIN_ENTRY_SCORE': [3.0, 4.0, 5.0, 6.0]
    }
    
    results = await engine.find_optimal_parameters(
        symbol='ETH',
        parameter_ranges=parameter_ranges
    )
    
    print(f"\n🏆 Optimization Complete!")
    print(f"\nBaseline (no optimization):")
    baseline = results['baseline']
    print(f"  Quality score: {baseline.quality_score:.1f}")
    print(f"  Return: {baseline.total_return:.2%}")
    print(f"  Win rate: {baseline.win_rate:.1%}")
    
    print(f"\nOptimized configuration:")
    best = results['best_insight']
    print(f"  Parameters: {results['best_parameters']}")
    print(f"  Quality score: {best.quality_score:.1f}")
    print(f"  Return: {best.total_return:.2%}")
    print(f"  Win rate: {best.win_rate:.1%}")
    
    improvement = best.quality_score - baseline.quality_score
    if improvement > 0:
        print(f"\n✅ Improvement: +{improvement:.1f} points")
    else:
        print(f"\n⚠️ No improvement found - defaults are optimal")
        

async def report_mode(engine: InsightEngine):
    """Generate comprehensive report"""
    print("\n📊 COMPREHENSIVE REPORT")
    print("="*50)
    
    report = engine.generate_report()
    print(report)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path("insight_report_" + timestamp + ".txt")
    report_file.write_text(report)
    print(f"\n📁 Report saved to: {report_file}")


async def main():
    parser = argparse.ArgumentParser(description='Insight Engine - Practical Optimization')
    parser.add_argument(
        '--mode',
        choices=['analyze', 'test', 'optimize', 'report'],
        default='analyze',
        help='Operation mode'
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    INSIGHT ENGINE                            ║
║                                                              ║
║  Real backtests. Real analysis. Real improvements.           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create engine
    engine = InsightEngine()
    
    # Run selected mode
    if args.mode == 'analyze':
        await analyze_mode(engine)
    elif args.mode == 'test':
        await test_mode(engine)
    elif args.mode == 'optimize':
        await optimize_mode(engine)
    elif args.mode == 'report':
        await report_mode(engine)
        
    print("\n✅ Done!")
    

if __name__ == "__main__":
    asyncio.run(main())