#!/usr/bin/env python3
"""
Main runner for the Deep Optimization Framework

This script orchestrates the complete optimization process,
from bug fixing to validation.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.deep_optimizer import DeepOptimizer
from experiments.adaptive_learner import AdaptiveLearner


async def main():
    """Main optimization runner"""
    
    parser = argparse.ArgumentParser(description='SqueezeFlow Deep Optimization Framework')
    parser.add_argument('--mode', choices=['full', 'analyze', 'fix-bugs', 'learn', 'validate'],
                       default='full', help='Optimization mode to run')
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'AVAX'],
                       help='Symbols to test')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SQUEEZEFLOW DEEP OPTIMIZATION FRAMEWORK")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)
    print()
    
    # Initialize components
    optimizer = DeepOptimizer()
    learner = AdaptiveLearner()
    
    try:
        if args.mode == 'analyze':
            # Just analyze current system
            print("🔍 Analyzing system...")
            analysis = await optimizer.analyze_system()
            
            print("\n📊 Analysis Results:")
            print(f"  Bugs found: {len(analysis['bugs_found'])}")
            for bug in analysis['bugs_found']:
                print(f"    - {bug}")
            
            print(f"  Inefficiencies: {len(analysis['inefficiencies'])}")
            for ineff in analysis['inefficiencies'][:5]:
                print(f"    - {ineff['type']}: {ineff['likely_cause']}")
                
        elif args.mode == 'fix-bugs':
            # Fix known bugs
            print("🔧 Fixing bugs...")
            analysis = await optimizer.analyze_system()
            
            if analysis['bugs_found']:
                fixes = await optimizer.fix_bugs(analysis['bugs_found'])
                print(f"\n✅ Fixed {len(fixes)} bugs:")
                for bug, fix in fixes.items():
                    print(f"  - {bug}: {fix['status']}")
            else:
                print("✅ No bugs found!")
                
        elif args.mode == 'learn':
            # Run adaptive learning
            print("🧠 Running adaptive learning...")
            
            # Show current status
            print(learner.generate_status_report())
            
            # Get next investigation
            next_step = learner.get_next_investigation()
            if next_step:
                print(f"\n🎯 Next investigation: {next_step}")
            else:
                print("\n✅ No pending investigations")
                
        elif args.mode == 'validate':
            # Validate current performance
            print("✅ Validating performance...")
            
            # Run baseline first
            baseline = await optimizer._run_baseline_backtest()
            
            print("\n📊 Current Performance:")
            for symbol, metrics in baseline.items():
                if 'error' not in metrics:
                    print(f"\n  {symbol}:")
                    print(f"    Return: {metrics.get('total_return', 0):.2%}")
                    print(f"    Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"    Trades: {metrics.get('total_trades', 0)}")
                    print(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")
                    
        else:  # full mode
            # Run complete optimization cycle
            print("🚀 Running full optimization cycle...")
            
            # Step 1: Analyze
            print("\n[1/5] Analyzing system...")
            analysis = await optimizer.analyze_system()
            
            # Step 2: Fix bugs
            if analysis['bugs_found']:
                print(f"\n[2/5] Fixing {len(analysis['bugs_found'])} bugs...")
                fixes = await optimizer.fix_bugs(analysis['bugs_found'])
            else:
                print("\n[2/5] No bugs to fix ✅")
                
            # Step 3: Discover concepts
            print("\n[3/5] Discovering trading concepts...")
            discoveries = await optimizer.discover_concepts(analysis['current_performance'])
            print(f"  Found {len(discoveries['patterns'])} patterns")
            print(f"  Extracted {len(discoveries['principles'])} principles")
            
            # Step 4: Optimize
            print("\n[4/5] Optimizing strategy...")
            optimizations = await optimizer.optimize_strategy(analysis, discoveries)
            print(f"  Code changes: {len(optimizations['code_changes'])}")
            print(f"  Parameter updates: {len(optimizations['parameter_updates'])}")
            
            # Step 5: Validate
            if not args.skip_validation:
                print("\n[5/5] Validating improvements...")
                validation = await optimizer.validate_improvements(analysis['current_performance'])
                
                print(f"\n📈 Results:")
                print(f"  Improved: {validation.get('improved', [])}")
                print(f"  Degraded: {validation.get('degraded', [])}")
                print(f"  Unchanged: {validation.get('unchanged', [])}")
            else:
                print("\n[5/5] Skipping validation")
                
            # Generate report
            report = optimizer.generate_report(
                analysis, discoveries, optimizations, 
                validation if not args.skip_validation else {}
            )
            
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE")
            print("=" * 80)
            print(f"Report saved to: experiments/optimization_report.txt")
            
            # Update adaptive learner
            for pattern in discoveries['patterns']:
                learner.record_learning(
                    symbol='ALL',
                    concept=pattern['type'],
                    finding=pattern['insight'],
                    confidence=0.7
                )
            
            print(f"Recorded {len(discoveries['patterns'])} learnings")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n✨ Done!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)