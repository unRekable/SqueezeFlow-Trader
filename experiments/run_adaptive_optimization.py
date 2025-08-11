#!/usr/bin/env python3
"""
Main Entry Point for Adaptive Optimization

This orchestrates the entire learning → validation → modification cycle.
Run this to start the self-improving process.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from concept_validator import ConceptValidator
from adaptive_learner import AdaptiveLearner
from self_modifying_optimizer import SelfModifyingOptimizer


class AdaptiveOptimizationPipeline:
    """Orchestrates the complete optimization pipeline"""
    
    def __init__(self):
        self.validator = ConceptValidator()
        self.learner = AdaptiveLearner()
        self.modifier = SelfModifyingOptimizer()
        
        # Set remote InfluxDB
        os.environ['INFLUX_HOST'] = '213.136.75.120'
        os.environ['INFLUX_PORT'] = '8086'
        
        print("="*80)
        print("ADAPTIVE OPTIMIZATION PIPELINE")
        print("Learning → Understanding → Modifying → Improving")
        print("="*80)
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check which symbols have data available"""
        from influxdb import InfluxDBClient
        
        print("\n📊 Checking Data Availability...")
        print("-"*40)
        
        client = InfluxDBClient(
            host='213.136.75.120',
            port=8086,
            database='significant_trades'
        )
        
        symbols = ['BTC', 'ETH', 'TON', 'AVAX', 'SOL']
        available = {}
        
        for symbol in symbols:
            market = f'BINANCE:{symbol.lower()}usdt'
            query = f'''
            SELECT COUNT(*) FROM "aggr_1s"."trades_1s" 
            WHERE market = '{market}' 
            AND time > now() - 24h
            '''
            
            try:
                result = client.query(query)
                points = list(result.get_points())
                count = points[0]['count_close'] if points else 0
                available[symbol] = count > 0
                
                if count > 0:
                    print(f"✅ {symbol}: {count:,} data points available")
                else:
                    print(f"❌ {symbol}: No recent data")
            except:
                available[symbol] = False
                print(f"❌ {symbol}: Query failed")
        
        return available
    
    def run_cycle(self, target_symbol: Optional[str] = None):
        """
        Run one complete optimization cycle
        
        1. Check what to investigate (learner)
        2. Validate concepts (validator)
        3. Identify issues (validator)
        4. Modify code if needed (modifier)
        5. Test changes (modifier)
        6. Learn from results (learner)
        """
        
        print("\n" + "="*80)
        print("STARTING OPTIMIZATION CYCLE")
        print("="*80)
        
        # Step 1: Determine what to investigate
        print("\n📝 Step 1: Determining Focus...")
        next_investigation = self.learner.get_next_investigation()
        
        if target_symbol:
            # Override with user's target
            print(f"   User specified: Focus on {target_symbol}")
            symbol = target_symbol
        elif next_investigation:
            print(f"   Next investigation: {next_investigation}")
            symbol = next_investigation.get('symbol', 'BTC')
        else:
            print("   Starting with BTC as baseline")
            symbol = 'BTC'
        
        # Step 2: Validate concepts for this symbol
        print(f"\n🔍 Step 2: Validating Concepts for {symbol}...")
        print("-"*40)
        
        # Use most recent data
        date_range = ('2025-08-10', '2025-08-10')
        
        # Run validations
        divergence_validation = self.validator.analyze_divergence_detection(symbol, date_range)
        oi_validation = self.validator.analyze_oi_confirmation(symbol, date_range)
        
        print(f"Divergence Detection F1: {divergence_validation.f1_score:.2f}")
        print(f"OI Confirmation F1: {oi_validation.f1_score:.2f}")
        
        # Step 3: Learn from validation
        print(f"\n💡 Step 3: Recording Learnings...")
        
        # Record divergence learning
        if divergence_validation.failure_reasons:
            main_issue = divergence_validation.failure_reasons[0]
            self.learner.record_learning(
                symbol=symbol,
                concept="CVD_divergence_detection",
                finding=main_issue,
                confidence=1 - divergence_validation.noise_ratio
            )
            print(f"   Learned: {main_issue}")
        
        # Check if this is the hardcoded threshold issue
        validation_results = {}
        if any('hardcoded' in reason.lower() or '1m threshold' in reason.lower() 
               for reason in divergence_validation.failure_reasons):
            
            print(f"\n🚨 Critical Issue Detected: Hardcoded threshold blocks {symbol}")
            
            # Get the suggested threshold from validation
            for suggestion in divergence_validation.improvement_suggestions:
                if 'dynamic threshold' in suggestion:
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?(?:e[+-]?\d+)?)', suggestion)
                    if match:
                        suggested_threshold = float(match.group(1))
                        validation_results[symbol] = {
                            'blocked_by_threshold': True,
                            'typical_volume': suggested_threshold,
                            'required_threshold': 1e6
                        }
                        break
        
        # Step 4: Modify code if critical issues found
        if validation_results:
            print(f"\n🔧 Step 4: Modifying Code to Fix Issues...")
            print("-"*40)
            
            self.modifier.run_adaptive_modification(validation_results)
            
            # Record that we modified code
            self.learner.record_learning(
                symbol=symbol,
                concept="code_modification",
                finding=f"Modified hardcoded threshold for {symbol}",
                confidence=0.95
            )
        else:
            print(f"\n✅ Step 4: No code modifications needed")
        
        # Step 5: Generate reports
        print(f"\n📊 Step 5: Generating Reports...")
        print("-"*40)
        
        # Validation report
        self.validator.save_validations()
        validation_report = self.validator.generate_learning_report()
        
        # Learning status
        learning_status = self.learner.generate_status_report()
        
        # Save combined report
        report_dir = Path(__file__).parent / "optimization_reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"cycle_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ADAPTIVE OPTIMIZATION CYCLE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(validation_report)
            f.write("\n\n")
            f.write(learning_status)
        
        print(f"✅ Report saved to: {report_file}")
        
        # Step 6: Determine next steps
        print(f"\n🎯 Step 6: Planning Next Steps...")
        print("-"*40)
        
        next_step = self.learner.get_next_investigation()
        if next_step:
            print(f"Recommended next: {next_step}")
        else:
            print("Consider testing on different date ranges or market conditions")
        
        return {
            'symbol': symbol,
            'divergence_f1': divergence_validation.f1_score,
            'oi_f1': oi_validation.f1_score,
            'code_modified': bool(validation_results),
            'report_file': str(report_file)
        }
    
    def run_multi_symbol_optimization(self):
        """Run optimization for all symbols with issues"""
        
        print("\n🔄 Running Multi-Symbol Optimization")
        print("="*60)
        
        # Check data availability
        available = self.check_data_availability()
        
        # Focus on symbols with data
        symbols_to_test = [s for s, has_data in available.items() if has_data]
        
        if not symbols_to_test:
            print("❌ No symbols with available data")
            return
        
        print(f"\nWill test: {', '.join(symbols_to_test)}")
        
        results = []
        for symbol in symbols_to_test:
            print(f"\n{'='*60}")
            print(f"Testing {symbol}")
            print('='*60)
            
            result = self.run_cycle(target_symbol=symbol)
            results.append(result)
            
            # If we modified code, test if it helped
            if result['code_modified']:
                print(f"\n⚠️  Code was modified for {symbol}")
                print("Consider running backtests to verify improvements")
        
        # Summary
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        for r in results:
            print(f"\n{r['symbol']}:")
            print(f"  Divergence Detection: F1={r['divergence_f1']:.2f}")
            print(f"  Code Modified: {'Yes' if r['code_modified'] else 'No'}")
        
        # Save what we learned
        self.learner.save_all()
        
        print("\n✅ Optimization complete. Check optimization_reports/ for details")


def main():
    """Main entry point"""
    
    pipeline = AdaptiveOptimizationPipeline()
    
    print("\nOptions:")
    print("1. Run full multi-symbol optimization")
    print("2. Focus on specific symbol")
    print("3. Continue from last session")
    
    # For now, run multi-symbol by default
    print("\nRunning full optimization...")
    
    pipeline.run_multi_symbol_optimization()
    
    print("\n" + "="*80)
    print("ADAPTIVE OPTIMIZATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review code modifications in code_modifications/")
    print("2. Check learning journal in adaptive_learning/")
    print("3. Run backtests to verify improvements")
    print("4. Review reports in optimization_reports/")


if __name__ == "__main__":
    main()