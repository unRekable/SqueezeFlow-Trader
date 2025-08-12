"""Test script for Veloce SqueezeFlow strategy"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_strategy_components():
    """Test individual strategy components"""
    print("\n" + "="*60)
    print("Testing Veloce SqueezeFlow Strategy Components")
    print("="*60)
    
    # Test imports
    try:
        from veloce.core import CONFIG
        from veloce.data import DATA
        print("✅ Core modules imported")
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    try:
        from veloce.strategy.squeezeflow import (
            SqueezeFlowStrategy,
            Indicators,
            FivePhaseAnalyzer,
            SignalGenerator
        )
        print("✅ Strategy modules imported")
    except Exception as e:
        print(f"❌ Strategy import failed: {e}")
        return False
    
    # Test configuration
    print(f"\n📊 Configuration:")
    print(f"  - System: Veloce")
    print(f"  - Primary timeframe: {CONFIG.primary_timeframe}")
    print(f"  - Squeeze period: {CONFIG.squeeze_period}")
    print(f"  - OI enabled: {CONFIG.oi_enabled}")
    
    # Test indicators
    print(f"\n🎯 Testing Indicators:")
    indicators = Indicators(CONFIG)
    print(f"  - Indicators initialized")
    print(f"  - Methods available: {len([m for m in dir(indicators) if not m.startswith('_')])}")
    
    # Test phase analyzer
    print(f"\n🔄 Testing Phase Analyzer:")
    analyzer = FivePhaseAnalyzer(CONFIG)
    print(f"  - Phase analyzer initialized")
    print(f"  - Phases configured: {list(analyzer.phase_config.keys())}")
    
    # Test signal generator
    print(f"\n📡 Testing Signal Generator:")
    signals = SignalGenerator(CONFIG)
    print(f"  - Signal generator initialized")
    print(f"  - Active signals: {len(signals.active_signals)}")
    
    return True

def test_strategy_initialization():
    """Test strategy initialization"""
    print("\n" + "="*60)
    print("Testing Strategy Initialization")
    print("="*60)
    
    try:
        from veloce.strategy.squeezeflow import SqueezeFlowStrategy
        from veloce.core import CONFIG
        from veloce.data import DATA
        
        # Initialize strategy
        strategy = SqueezeFlowStrategy(CONFIG, DATA)
        print("✅ Strategy initialized successfully")
        
        # Check state
        state = strategy.get_state()
        print(f"\n📊 Strategy State:")
        print(f"  - Mode: {state['mode']}")
        print(f"  - Positions: {len(state['positions'])}")
        print(f"  - Active signals: {state['active_signals']}")
        print(f"  - Total signals: {state['total_signals']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_analysis():
    """Test strategy analysis with mock data"""
    print("\n" + "="*60)
    print("Testing Mock Analysis")
    print("="*60)
    
    try:
        from veloce.strategy.squeezeflow import SqueezeFlowStrategy
        from veloce.core import CONFIG
        from veloce.data import DATA
        import pandas as pd
        import numpy as np
        
        # Create strategy
        strategy = SqueezeFlowStrategy(CONFIG, DATA)
        
        # Create mock data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.rand(100) * 1000000
        }, index=dates)
        
        print("✅ Mock data created")
        
        # Test indicator calculation
        indicators = strategy.indicators
        df_with_indicators = indicators.calculate_all(mock_df)
        
        print(f"\n📊 Indicators Calculated:")
        indicator_cols = [c for c in df_with_indicators.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        for col in indicator_cols[:5]:  # Show first 5
            if not df_with_indicators[col].isna().all():
                print(f"  - {col}: {df_with_indicators[col].iloc[-1]:.4f}")
        
        # Test signal extraction
        signals = indicators.get_indicator_signals(df_with_indicators)
        print(f"\n📡 Extracted Signals:")
        for key, value in signals.items():
            print(f"  - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VELOCE SQUEEZEFLOW STRATEGY TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Component Test", test_strategy_components()))
    results.append(("Initialization Test", test_strategy_initialization()))
    results.append(("Mock Analysis Test", test_mock_analysis()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Strategy implementation is working.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)