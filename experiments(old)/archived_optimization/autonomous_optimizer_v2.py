#!/usr/bin/env python3
"""
Autonomous Optimizer v2.0

Self-driving optimization that learns and adapts.
Uses the optimization framework to automatically find the best parameters.

Key improvements:
- Actually connects to remote InfluxDB
- Handles dynamic thresholds per symbol
- Makes intelligent decisions based on market conditions
- Provides clear explanations for every decision
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from optimization_framework import (
    OptimizationFramework, 
    OptimizableParameter, 
    ExperimentResult
)


class AutonomousOptimizer:
    """Autonomous system that runs experiments and makes decisions"""
    
    def __init__(self):
        self.framework = OptimizationFramework()
        self.logger = self.framework.logger
        
        # Optimization strategy parameters
        self.exploration_rate = 0.3  # 30% exploration, 70% exploitation
        self.confidence_threshold = 0.8  # 80% confidence to adopt changes
        self.min_samples_per_param = 3  # Minimum tests before decision
        self.patience = 5  # Tests without improvement before moving on
        
        # Track optimization progress
        self.current_focus = None
        self.tests_without_improvement = 0
        self.baseline_scores = {}
        
        self.logger.info("Autonomous Optimizer v2.0 initialized")
    
    def analyze_symbol_characteristics(self, symbol: str) -> Dict[str, Any]:
        """Analyze a symbol's characteristics to inform optimization"""
        
        from influxdb import InfluxDBClient
        
        try:
            client = InfluxDBClient(
                host=self.framework.influx_host,
                port=self.framework.influx_port,
                database=self.framework.influx_database
            )
            
            # Map symbol to market
            market_map = {
                'BTC': 'BINANCE:btcusdt',
                'ETH': 'BINANCE:ethusdt',
                'TON': 'BINANCE:tonusdt',
                'AVAX': 'BINANCE:avaxusdt',
                'SOL': 'BINANCE:solusdt'
            }
            market = market_map.get(symbol, f'BINANCE:{symbol.lower()}usdt')
            
            # Get recent CVD data to understand volume patterns
            query = f'''
            SELECT 
                MEAN(buyvolume) as avg_buy_volume,
                MEAN(sellvolume) as avg_sell_volume,
                MAX(buyvolume) as max_buy_volume,
                MAX(sellvolume) as max_sell_volume,
                STDDEV(buyvolume) as volume_stddev
            FROM "aggr_1s"."trades_1s"
            WHERE market = '{market}'
            AND time > now() - 24h
            '''
            
            result = client.query(query)
            points = list(result.get_points())
            
            if points:
                data = points[0]
                
                # Calculate typical CVD changes
                avg_volume = (data.get('avg_buy_volume', 0) + data.get('avg_sell_volume', 0)) / 2
                max_volume = max(data.get('max_buy_volume', 0), data.get('max_sell_volume', 0))
                
                # Determine appropriate thresholds
                characteristics = {
                    'symbol': symbol,
                    'avg_volume': avg_volume,
                    'max_volume': max_volume,
                    'volume_stddev': data.get('volume_stddev', 0),
                    'suggested_cvd_threshold': max_volume * 0.1,  # 10% of max
                    'volume_category': self._categorize_volume(avg_volume),
                    'volatility': 'high' if data.get('volume_stddev', 0) > avg_volume * 0.5 else 'normal'
                }
                
                self.logger.info(f"Symbol {symbol} characteristics: {characteristics['volume_category']} volume, {characteristics['volatility']} volatility")
                return characteristics
                
            else:
                self.logger.warning(f"No data found for {symbol}")
                return {'symbol': symbol, 'volume_category': 'unknown'}
                
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {'symbol': symbol, 'volume_category': 'unknown'}
    
    def _categorize_volume(self, avg_volume: float) -> str:
        """Categorize volume levels"""
        if avg_volume > 1e7:  # > 10M
            return 'very_high'
        elif avg_volume > 1e6:  # > 1M
            return 'high'
        elif avg_volume > 1e5:  # > 100K
            return 'medium'
        elif avg_volume > 1e4:  # > 10K
            return 'low'
        else:
            return 'very_low'
    
    def decide_next_experiment(self) -> Tuple[str, str, Any, str]:
        """
        Decide what to test next using intelligent selection
        
        Returns:
            (symbol, parameter, value, reason)
        """
        
        # Priority 1: Fix critical issues (like hardcoded threshold)
        critical_params = [p for p, config in self.framework.parameters.items() 
                          if config.impact == 'critical']
        
        for param in critical_params:
            param_results = [r for r in self.framework.results_history 
                           if r.parameter == param]
            if len(param_results) < self.min_samples_per_param:
                # Test on affected symbols
                affected_symbols = self.framework.parameters[param].affects_symbols
                if affected_symbols:
                    symbol = affected_symbols[0]  # Start with first affected symbol
                    
                    # Get dynamic value for this symbol
                    characteristics = self.analyze_symbol_characteristics(symbol)
                    if param == 'CVD_VOLUME_THRESHOLD':
                        value = characteristics.get('suggested_cvd_threshold', 1e5)
                    else:
                        values = self.framework.parameters[param].get_test_values(symbol)
                        value = values[len(param_results) % len(values)]
                    
                    reason = f"Critical parameter {param} needs testing for {symbol}"
                    return symbol, param, value, reason
        
        # Priority 2: Complete baseline tests
        symbols = ['BTC', 'ETH', 'TON', 'AVAX', 'SOL']
        for symbol in symbols:
            if symbol not in self.baseline_scores:
                # Run baseline test
                return symbol, 'baseline', None, f"Establishing baseline for {symbol}"
        
        # Priority 3: Explore undertested parameters
        undertested = []
        for param_name, param_config in self.framework.parameters.items():
            param_results = [r for r in self.framework.results_history 
                           if r.parameter == param_name]
            if len(param_results) < self.min_samples_per_param:
                undertested.append((param_name, len(param_results)))
        
        if undertested:
            # Sort by least tested
            undertested.sort(key=lambda x: x[1])
            param = undertested[0][0]
            
            # Choose symbol based on parameter characteristics
            if self.framework.parameters[param].affects_symbols:
                symbol = self.framework.parameters[param].affects_symbols[0]
            else:
                # Use best performing symbol so far
                symbol_scores = {}
                for s in symbols:
                    s_results = [r for r in self.framework.results_history if r.symbol == s]
                    if s_results:
                        symbol_scores[s] = max(r.performance_score for r in s_results)
                
                symbol = max(symbol_scores.keys(), key=symbol_scores.get) if symbol_scores else 'BTC'
            
            values = self.framework.parameters[param].get_test_values(symbol)
            param_results = [r for r in self.framework.results_history if r.parameter == param]
            tested_values = [r.tested_value for r in param_results]
            untested = [v for v in values if v not in tested_values]
            
            value = untested[0] if untested else values[0]
            reason = f"Exploring parameter {param} (test {len(param_results)+1}/{self.min_samples_per_param})"
            
            return symbol, param, value, reason
        
        # Priority 4: Exploit promising parameters
        promising = self._identify_promising_parameters()
        
        if promising:
            param = promising[0]
            
            # Find best value so far
            param_results = [r for r in self.framework.results_history if r.parameter == param]
            best_result = max(param_results, key=lambda x: x.performance_score)
            
            # Test on different symbol or refine value
            tested_symbols = set(r.symbol for r in param_results)
            untested_symbols = [s for s in symbols if s not in tested_symbols]
            
            if untested_symbols:
                symbol = untested_symbols[0]
                value = best_result.tested_value
                reason = f"Testing promising parameter {param}={value} on new symbol {symbol}"
            else:
                # Refine the value
                symbol = best_result.symbol
                current_val = float(best_result.tested_value) if isinstance(best_result.tested_value, (int, float)) else best_result.tested_value
                
                # Try small variations
                if isinstance(current_val, (int, float)):
                    value = current_val * 1.1  # 10% increase
                else:
                    value = current_val
                
                reason = f"Refining promising parameter {param} for {symbol}"
            
            return symbol, param, value, reason
        
        # Default: Random exploration
        import random
        symbol = random.choice(symbols)
        param = random.choice(list(self.framework.parameters.keys()))
        values = self.framework.parameters[param].get_test_values(symbol)
        value = random.choice(values)
        
        reason = "Random exploration (all priorities exhausted)"
        return symbol, param, value, reason
    
    def _identify_promising_parameters(self) -> List[str]:
        """Identify parameters showing promise"""
        
        promising = []
        
        for param_name in self.framework.parameters.keys():
            param_results = [r for r in self.framework.results_history 
                           if r.parameter == param_name]
            
            if len(param_results) >= 2:
                # Check if any test beat baseline significantly
                scores = [r.performance_score for r in param_results]
                avg_score = np.mean(scores)
                
                # Get baseline comparison
                baseline_score = 50.0  # Default
                for symbol in set(r.symbol for r in param_results):
                    if symbol in self.baseline_scores:
                        baseline_score = self.baseline_scores[symbol]
                        break
                
                if avg_score > baseline_score * 1.1:  # 10% improvement
                    promising.append(param_name)
        
        # Sort by average improvement
        promising.sort(key=lambda p: np.mean([r.performance_score for r in self.framework.results_history if r.parameter == p]), reverse=True)
        
        return promising
    
    def evaluate_and_decide(self, result: ExperimentResult) -> Dict[str, Any]:
        """
        Evaluate experiment results and make decisions
        
        Returns decision dictionary with action and reasoning
        """
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': result.experiment_id,
            'parameter': result.parameter,
            'value': result.tested_value,
            'symbol': result.symbol,
            'score': result.performance_score,
            'action': 'pending',
            'confidence': 0.0,
            'reasoning': []
        }
        
        # Special handling for baseline
        if result.parameter == 'baseline':
            self.baseline_scores[result.symbol] = result.performance_score
            decision['action'] = 'record_baseline'
            decision['reasoning'].append(f"Baseline established for {result.symbol}: {result.performance_score:.1f}")
            return decision
        
        # Get baseline for comparison
        baseline_score = self.baseline_scores.get(result.symbol, 50.0)
        improvement = result.performance_score - baseline_score
        
        # Check if this is better than current best for this parameter
        param_results = [r for r in self.framework.results_history 
                        if r.parameter == result.parameter and r.symbol == result.symbol]
        
        is_best = True
        if param_results:
            best_previous = max(r.performance_score for r in param_results[:-1])  # Exclude current
            is_best = result.performance_score > best_previous
        
        # Decision logic
        if result.performance_score < 30:
            decision['action'] = 'reject'
            decision['confidence'] = 0.9
            decision['reasoning'].append(f"Poor performance (score={result.performance_score:.1f})")
            
        elif improvement < -10:
            decision['action'] = 'reject'
            decision['confidence'] = 0.8
            decision['reasoning'].append(f"Worse than baseline by {abs(improvement):.1f} points")
            
        elif improvement > 20 and is_best:
            decision['action'] = 'adopt'
            decision['confidence'] = 0.9
            decision['reasoning'].append(f"Significant improvement: +{improvement:.1f} points over baseline")
            
            # Update parameter
            self.framework.parameters[result.parameter].current_value = result.tested_value
            
        elif improvement > 10:
            decision['action'] = 'promising'
            decision['confidence'] = 0.7
            decision['reasoning'].append(f"Moderate improvement: +{improvement:.1f} points")
            decision['reasoning'].append("Needs more testing for confirmation")
            
        elif len(param_results) < self.min_samples_per_param:
            decision['action'] = 'continue_testing'
            decision['confidence'] = 0.5
            decision['reasoning'].append(f"Need {self.min_samples_per_param - len(param_results)} more tests")
            
        else:
            decision['action'] = 'no_improvement'
            decision['confidence'] = 0.6
            decision['reasoning'].append(f"No significant improvement (score={result.performance_score:.1f})")
        
        # Additional context
        if result.total_trades < 5:
            decision['reasoning'].append(f"Warning: Low trade count ({result.total_trades})")
            decision['confidence'] *= 0.8
        
        if abs(result.max_drawdown_pct) > 15:
            decision['reasoning'].append(f"Warning: High drawdown ({result.max_drawdown_pct:.1f}%)")
            decision['confidence'] *= 0.9
        
        # Save decision
        self.framework.decisions_history.append(decision)
        self.framework.save_all()
        
        return decision
    
    def run_optimization_cycle(self, 
                              max_experiments: int = 10,
                              target_symbols: Optional[List[str]] = None):
        """
        Run a complete optimization cycle
        
        Args:
            max_experiments: Maximum number of experiments to run
            target_symbols: Specific symbols to optimize for (None = all)
        """
        
        self.logger.info(f"Starting optimization cycle with {max_experiments} experiments")
        
        if target_symbols:
            self.logger.info(f"Targeting symbols: {target_symbols}")
        
        for i in range(max_experiments):
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {i+1}/{max_experiments}")
            print('='*60)
            
            # Decide what to test
            symbol, param, value, reason = self.decide_next_experiment()
            
            # Skip if targeting specific symbols
            if target_symbols and symbol not in target_symbols:
                continue
            
            print(f"Testing: {param} = {value} on {symbol}")
            print(f"Reason: {reason}")
            
            # Run the experiment
            if param == 'baseline':
                # Baseline test with current parameters
                result = self.framework.run_backtest(symbol, 'baseline', 'current')
            else:
                result = self.framework.run_backtest(symbol, param, value)
            
            if result:
                # Save result
                self.framework.results_history.append(result)
                self.framework.save_all()
                
                # Evaluate and decide
                decision = self.evaluate_and_decide(result)
                
                print(f"\nResults:")
                print(f"  Score: {result.performance_score:.1f}")
                print(f"  Trades: {result.total_trades}")
                print(f"  Win Rate: {result.win_rate:.1f}%")
                print(f"  Return: {result.total_return_pct:.2f}%")
                print(f"  Drawdown: {result.max_drawdown_pct:.1f}%")
                
                print(f"\nDecision: {decision['action']} (confidence: {decision['confidence']:.1%})")
                for reasoning in decision['reasoning']:
                    print(f"  - {reasoning}")
                
                # Track progress
                if decision['action'] in ['adopt', 'promising']:
                    self.tests_without_improvement = 0
                else:
                    self.tests_without_improvement += 1
                
                # Check if we should change focus
                if self.tests_without_improvement >= self.patience:
                    self.logger.info("Patience exhausted, changing focus")
                    self.current_focus = None
                    self.tests_without_improvement = 0
            
            else:
                print("ERROR: Experiment failed")
            
            # Brief pause between experiments
            time.sleep(2)
        
        # Generate final report
        print("\n" + "="*60)
        print("OPTIMIZATION CYCLE COMPLETE")
        print("="*60)
        
        self.framework.generate_report()
        
        # Save final state
        self.framework.save_all()
    
    def suggest_production_config(self) -> Dict[str, Any]:
        """
        Suggest production configuration based on all experiments
        
        Returns dictionary of recommended parameter values
        """
        
        recommendations = {}
        
        for param_name, param_config in self.framework.parameters.items():
            param_results = [r for r in self.framework.results_history 
                           if r.parameter == param_name]
            
            if param_results:
                # Group by value and average scores
                value_scores = {}
                for r in param_results:
                    val_str = str(r.tested_value)
                    if val_str not in value_scores:
                        value_scores[val_str] = []
                    value_scores[val_str].append(r.performance_score)
                
                # Find best average
                best_value = None
                best_avg = 0
                
                for val_str, scores in value_scores.items():
                    avg = np.mean(scores)
                    if avg > best_avg:
                        best_avg = avg
                        best_value = val_str
                
                # Parse back to original type
                if param_config.param_type == 'float':
                    best_value = float(best_value)
                elif param_config.param_type == 'int':
                    best_value = int(float(best_value))
                elif param_config.param_type == 'list':
                    best_value = json.loads(best_value.replace("'", '"'))
                
                recommendations[param_name] = {
                    'recommended_value': best_value,
                    'current_value': param_config.current_value,
                    'avg_score': best_avg,
                    'confidence': min(len(param_results) / self.min_samples_per_param, 1.0),
                    'env_var': param_config.env_var
                }
        
        return recommendations


def main():
    """Main entry point"""
    
    optimizer = AutonomousOptimizer()
    
    print("="*80)
    print("AUTONOMOUS OPTIMIZER v2.0")
    print("="*80)
    print("\nSelf-driving parameter optimization for SqueezeFlow strategy")
    print(f"Using InfluxDB at {optimizer.framework.influx_host}")
    print(f"Parameters tracked: {len(optimizer.framework.parameters)}")
    print(f"Historical results: {len(optimizer.framework.results_history)}")
    
    # Run optimization
    print("\nStarting optimization cycle...")
    print("This will automatically:")
    print("1. Fix the hardcoded threshold bug with dynamic values")
    print("2. Test parameters on appropriate symbols")
    print("3. Make data-driven decisions")
    print("4. Generate comprehensive reports")
    
    # You can specify target symbols or let it choose
    # optimizer.run_optimization_cycle(max_experiments=5, target_symbols=['TON', 'AVAX'])
    optimizer.run_optimization_cycle(max_experiments=5)
    
    # Get recommendations
    print("\n" + "="*60)
    print("PRODUCTION RECOMMENDATIONS")
    print("="*60)
    
    recommendations = optimizer.suggest_production_config()
    
    for param, rec in recommendations.items():
        if rec['recommended_value'] != rec['current_value']:
            print(f"\n{param}:")
            print(f"  Current: {rec['current_value']}")
            print(f"  Recommended: {rec['recommended_value']}")
            print(f"  Confidence: {rec['confidence']:.0%}")
            print(f"  Set via: export {rec['env_var']}={rec['recommended_value']}")


if __name__ == "__main__":
    main()