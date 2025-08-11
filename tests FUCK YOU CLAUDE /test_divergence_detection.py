#!/usr/bin/env python3
"""Test divergence detection improvements - PROPERLY PLACED TEST"""

import os
os.environ['INFLUX_HOST'] = '213.136.75.120'

from datetime import datetime
import pytz
from data.pipeline import DataPipeline
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

def test_divergence_with_relative_patterns():
    """Test that the improved divergence detection finds relative patterns"""
    
    print("🔍 Testing Improved Divergence Detection\n")
    
    # Load data
    pipeline = DataPipeline()
    strategy = SqueezeFlowStrategy()
    
    # Use Aug 10 with plenty of data
    end_time = datetime(2025, 8, 10, 16, 0, 0, tzinfo=pytz.UTC)
    start_time = datetime(2025, 8, 10, 12, 0, 0, tzinfo=pytz.UTC)
    
    print(f"Loading BTC data: {start_time} to {end_time}")
    
    dataset = pipeline.get_complete_dataset(
        symbol='BTC',
        start_time=start_time,
        end_time=end_time,
        timeframe='1s'
    )
    
    if not dataset:
        print("❌ Failed to load dataset")
        return False
    
    print(f"✅ Dataset loaded: {len(dataset.get('ohlcv', []))} data points")
    
    # Process through strategy
    portfolio = {
        'positions': [],
        'cash': 10000,
        'total_value': 10000
    }
    
    print("\n🔄 Processing through strategy...")
    result = strategy.process(dataset, portfolio)
    
    # Check phase results
    phase_results = result.get('phase_results', {})
    phase2 = phase_results.get('phase2_divergence', {})
    
    print(f"\n📈 Divergence Detection Results:")
    print(f"  Has divergence: {phase2.get('has_divergence', False)}")
    print(f"  Pattern: {phase2.get('cvd_patterns', {}).get('pattern', 'N/A')}")
    print(f"  Setup type: {phase2.get('setup_type', 'N/A')}")
    
    # Check scoring
    phase4 = phase_results.get('phase4_scoring', {})
    total_score = phase4.get('total_score', 0)
    print(f"\n📊 Strategy Score: {total_score:.2f}/10")
    
    # Check orders
    orders = result.get('orders', [])
    print(f"\n📈 Orders Generated: {len(orders)}")
    
    # Test passes if we detect divergence and generate score
    success = phase2.get('has_divergence', False) and total_score > 0
    
    if success:
        print("\n✅ TEST PASSED: Divergence detected and score generated")
    else:
        print("\n❌ TEST FAILED: No divergence or score")
    
    return success

if __name__ == "__main__":
    test_divergence_with_relative_patterns()