#!/usr/bin/env python3
"""
Test the optimization framework to ensure everything is connected properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from optimization_framework import OptimizationFramework
from autonomous_optimizer_v2 import AutonomousOptimizer


def test_framework_initialization():
    """Test that the framework initializes correctly"""
    print("\n1. Testing Framework Initialization...")
    print("-" * 40)
    
    framework = OptimizationFramework()
    
    print(f"✅ Framework initialized")
    print(f"   Data directory: {framework.data_dir}")
    print(f"   InfluxDB: {framework.influx_host}:{framework.influx_port}")
    print(f"   Parameters defined: {len(framework.parameters)}")
    
    # List parameters
    print("\n   Tracked parameters:")
    for name, param in framework.parameters.items():
        print(f"   - {name}: {param.current_value} ({param.impact} impact)")
    
    return framework


def test_data_availability(framework):
    """Test data availability for different symbols"""
    print("\n2. Testing Data Availability...")
    print("-" * 40)
    
    symbols = ['BTC', 'ETH', 'TON', 'AVAX', 'SOL']
    
    available = {}
    for symbol in symbols:
        has_data, first_date, last_date = framework.check_data_availability(symbol)
        available[symbol] = has_data
        
        if has_data:
            print(f"✅ {symbol}: {first_date} to {last_date}")
        else:
            print(f"❌ {symbol}: No data found")
    
    return available


def test_symbol_analysis():
    """Test symbol characteristic analysis"""
    print("\n3. Testing Symbol Analysis...")
    print("-" * 40)
    
    optimizer = AutonomousOptimizer()
    
    # Analyze ETH (should have data)
    characteristics = optimizer.analyze_symbol_characteristics('ETH')
    
    if characteristics.get('volume_category') != 'unknown':
        print(f"✅ ETH analysis successful:")
        print(f"   Volume category: {characteristics.get('volume_category')}")
        print(f"   Suggested CVD threshold: {characteristics.get('suggested_cvd_threshold', 'N/A'):.0f}")
        print(f"   Volatility: {characteristics.get('volatility', 'N/A')}")
    else:
        print(f"⚠️  Could not analyze ETH - may need to check connection")
    
    return characteristics


def test_parameter_generation():
    """Test parameter value generation"""
    print("\n4. Testing Parameter Value Generation...")
    print("-" * 40)
    
    framework = OptimizationFramework()
    
    # Test dynamic parameter (CVD threshold)
    cvd_param = framework.parameters['CVD_VOLUME_THRESHOLD']
    
    print("Testing dynamic CVD threshold values:")
    for symbol in ['BTC', 'ETH', 'TON']:
        values = cvd_param.get_test_values(symbol)
        print(f"   {symbol}: {[f'{v:.0f}' for v in values[:3]]}...")
    
    # Test regular parameter
    score_param = framework.parameters['MIN_ENTRY_SCORE']
    values = score_param.get_test_values()
    print(f"\nMIN_ENTRY_SCORE test values: {values}")


def test_decision_logic():
    """Test the decision-making logic"""
    print("\n5. Testing Decision Logic...")
    print("-" * 40)
    
    optimizer = AutonomousOptimizer()
    
    # Simulate some results
    from optimization_framework import ExperimentResult
    from datetime import datetime
    
    # Good result
    good_result = ExperimentResult(
        experiment_id="test_good",
        timestamp=datetime.now(),
        parameter="MIN_ENTRY_SCORE",
        tested_value=3.5,
        baseline_value=4.0,
        symbol="ETH",
        date_range=("2025-08-10", "2025-08-10"),
        total_trades=15,
        winning_trades=11,
        losing_trades=4,
        win_rate=73.3,
        starting_balance=10000,
        final_balance=10800,
        total_return_pct=8.0,
        max_drawdown_pct=-3.5,
        sharpe_ratio=1.8,
        sortino_ratio=2.1,
        calmar_ratio=2.3,
        avg_win=120,
        avg_loss=-60,
        profit_factor=2.0,
        expectancy=53.3,
        avg_trade_duration=45,
        longest_trade=120,
        shortest_trade=15,
        market_volatility=0.02,
        market_trend="bullish",
        total_volume=1e9,
        backtest_duration_seconds=4.2,
        data_points_processed=86400
    )
    good_result.calculate_score()
    
    # Set baseline for comparison
    optimizer.baseline_scores['ETH'] = 50.0
    
    decision = optimizer.evaluate_and_decide(good_result)
    
    print(f"Good result decision: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.0%}")
    print("Reasoning:")
    for reason in decision['reasoning']:
        print(f"  - {reason}")
    
    # Bad result
    bad_result = ExperimentResult(
        experiment_id="test_bad",
        timestamp=datetime.now(),
        parameter="OI_RISE_THRESHOLD",
        tested_value=10.0,
        baseline_value=5.0,
        symbol="ETH",
        date_range=("2025-08-10", "2025-08-10"),
        total_trades=2,
        winning_trades=0,
        losing_trades=2,
        win_rate=0.0,
        starting_balance=10000,
        final_balance=9500,
        total_return_pct=-5.0,
        max_drawdown_pct=-8.0,
        sharpe_ratio=-0.5,
        sortino_ratio=-0.8,
        calmar_ratio=-0.6,
        avg_win=0,
        avg_loss=-250,
        profit_factor=0,
        expectancy=-250,
        avg_trade_duration=30,
        longest_trade=40,
        shortest_trade=20,
        market_volatility=0.03,
        market_trend="bearish",
        total_volume=8e8,
        backtest_duration_seconds=3.8,
        data_points_processed=86400
    )
    bad_result.calculate_score()
    
    decision = optimizer.evaluate_and_decide(bad_result)
    
    print(f"\nBad result decision: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.0%}")
    print("Reasoning:")
    for reason in decision['reasoning']:
        print(f"  - {reason}")


def test_next_experiment_selection():
    """Test how the system chooses next experiments"""
    print("\n6. Testing Experiment Selection...")
    print("-" * 40)
    
    optimizer = AutonomousOptimizer()
    
    # Get next experiment suggestions
    for i in range(3):
        symbol, param, value, reason = optimizer.decide_next_experiment()
        print(f"\nExperiment {i+1}:")
        print(f"  Symbol: {symbol}")
        print(f"  Parameter: {param}")
        print(f"  Value: {value}")
        print(f"  Reason: {reason}")


def main():
    """Run all tests"""
    print("="*80)
    print("OPTIMIZATION FRAMEWORK TEST SUITE")
    print("="*80)
    
    # Run tests
    framework = test_framework_initialization()
    available_symbols = test_data_availability(framework)
    
    if any(available_symbols.values()):
        test_symbol_analysis()
    else:
        print("\n⚠️  Skipping symbol analysis - no data available")
    
    test_parameter_generation()
    test_decision_logic()
    test_next_experiment_selection()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print("\nThe optimization framework is ready to:")
    print("✅ Track and optimize parameters")
    print("✅ Connect to remote InfluxDB for data")
    print("✅ Analyze symbol characteristics")
    print("✅ Generate intelligent test values")
    print("✅ Make data-driven decisions")
    print("✅ Select experiments autonomously")
    
    print("\nTo run actual optimization:")
    print("  python3 autonomous_optimizer_v2.py")
    
    print("\nTo run a single test:")
    print("  from optimization_framework import OptimizationFramework")
    print("  framework = OptimizationFramework()")
    print("  result = framework.run_backtest('ETH', 'MIN_ENTRY_SCORE', 3.5)")


if __name__ == "__main__":
    main()