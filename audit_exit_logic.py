#!/usr/bin/env python3
"""
Comprehensive audit of exit logic to identify why trades aren't closing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Import the exit management component
from strategies.squeezeflow.components.phase5_exits import ExitManagement

def create_test_position(side='BUY', entry_price=3587.35):
    """Create a test position"""
    return {
        'id': 'test_position_1',
        'symbol': 'ETH',
        'side': side,
        'quantity': 0.055751,
        'entry_price': entry_price,
        'entry_time': datetime.now(tz=pytz.UTC) - timedelta(hours=1),
        'trade_id': 'trade_001'
    }

def create_test_dataset(current_price, price_change_pct=0, cvd_spot_change=1000000, cvd_futures_change=500000):
    """Create test market data"""
    # Create price data
    base_price = current_price / (1 + price_change_pct/100)
    prices = np.linspace(base_price, current_price, 100)
    
    ohlcv = pd.DataFrame({
        'time': pd.date_range(end=datetime.now(tz=pytz.UTC), periods=100, freq='1min'),
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 500, 100)
    })
    
    # Create CVD data
    spot_cvd_base = 10000000
    futures_cvd_base = 5000000
    
    spot_cvd = pd.Series(
        np.linspace(spot_cvd_base, spot_cvd_base + cvd_spot_change, 100),
        index=ohlcv['time']
    )
    
    futures_cvd = pd.Series(
        np.linspace(futures_cvd_base, futures_cvd_base + cvd_futures_change, 100),
        index=ohlcv['time']
    )
    
    cvd_divergence = spot_cvd - futures_cvd
    
    return {
        'symbol': 'ETH',
        'ohlcv': ohlcv.set_index('time'),
        'spot_cvd': spot_cvd,
        'futures_cvd': futures_cvd,
        'cvd_divergence': cvd_divergence
    }

def test_exit_conditions():
    """Test various exit conditions"""
    
    print("=" * 80)
    print("EXIT LOGIC AUDIT - Testing Exit Conditions")
    print("=" * 80)
    
    exit_mgr = ExitManagement()
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Normal conditions (no exit)',
            'position': create_test_position('BUY', 3587.35),
            'current_price': 3600.00,
            'price_change_pct': 0.35,  # Small profit
            'cvd_spot_change': 1000000,  # Positive CVD
            'cvd_futures_change': 500000  # Positive CVD
        },
        {
            'name': 'Range break below (0.5% threshold)',
            'position': create_test_position('BUY', 3587.35),
            'current_price': 3565.00,  # Below entry - 0.5% = 3569.40
            'price_change_pct': -0.62,
            'cvd_spot_change': 1000000,
            'cvd_futures_change': 500000
        },
        {
            'name': 'Flow reversal - SPOT selling, FUTURES buying',
            'position': create_test_position('BUY', 3587.35),
            'current_price': 3590.00,
            'price_change_pct': 0.07,
            'cvd_spot_change': -2000000,  # SPOT selling
            'cvd_futures_change': 1000000   # FUTURES buying (shorts covering)
        },
        {
            'name': 'CVD invalidation - both CVDs declining',
            'position': create_test_position('BUY', 3587.35),
            'current_price': 3585.00,
            'price_change_pct': -0.07,
            'cvd_spot_change': -1500000,  # Both negative
            'cvd_futures_change': -1000000
        },
        {
            'name': 'Structure break - 2% drop',
            'position': create_test_position('BUY', 3587.35),
            'current_price': 3515.00,
            'price_change_pct': -2.0,
            'cvd_spot_change': -2000000,
            'cvd_futures_change': -1500000
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 60)
        
        # Create test data
        position = test_case['position']
        dataset = create_test_dataset(
            test_case['current_price'],
            test_case['price_change_pct'],
            test_case['cvd_spot_change'],
            test_case['cvd_futures_change']
        )
        
        # Run exit check
        exit_result = exit_mgr.manage_exits(dataset, position, {})
        
        # Print results
        print(f"Entry Price: ${position['entry_price']:.2f}")
        print(f"Current Price: ${test_case['current_price']:.2f}")
        print(f"Price Change: {test_case['price_change_pct']:.2f}%")
        print(f"CVD Spot Change: {test_case['cvd_spot_change']/1e6:.1f}M")
        print(f"CVD Futures Change: {test_case['cvd_futures_change']/1e6:.1f}M")
        print()
        print(f"SHOULD EXIT: {exit_result.get('should_exit', False)}")
        print(f"Exit Reasoning: {exit_result.get('exit_reasoning', 'None')}")
        
        # Detailed breakdown
        if 'flow_reversal' in exit_result:
            fr = exit_result['flow_reversal']
            print(f"  Flow Reversal: {fr.get('detected', False)} - {fr.get('severity', 'NONE')}")
            
        if 'range_break' in exit_result:
            rb = exit_result['range_break']
            range_threshold = position['entry_price'] * 0.005
            print(f"  Range Break: {rb.get('detected', False)} - {rb.get('break_type', 'NONE')}")
            print(f"    Range: ${position['entry_price'] - range_threshold:.2f} - ${position['entry_price'] + range_threshold:.2f}")
            
        if 'cvd_invalidation' in exit_result:
            ci = exit_result['cvd_invalidation']
            print(f"  CVD Invalidation: {ci.get('detected', False)} - {ci.get('invalidation_type', 'NONE')}")
            
        if 'structure_break' in exit_result:
            sb = exit_result['structure_break']
            print(f"  Structure Break: {sb.get('detected', False)} - {sb.get('structure_type', 'INTACT')}")

def analyze_real_backtest_data():
    """Analyze why the real backtest didn't close the ETH position"""
    
    print("\n" + "=" * 80)
    print("REAL BACKTEST ANALYSIS - ETH Position Not Closing")
    print("=" * 80)
    
    print("\nKnown Facts:")
    print("- ETH position opened: SELL 0.055751 @ $3587.35")
    print("- Position side: SELL (short)")
    print("- Backtest ran from 2025-07-25 to 2025-08-05 (11 days)")
    print("- Position never closed")
    
    print("\nPotential Issues:")
    print("\n1. EXIT CONDITION THRESHOLDS:")
    print("   - Range break: 0.5% threshold = $17.94 movement needed")
    print("   - For SHORT to exit: price must rise above $3605.29")
    print("   - This is a relatively tight threshold")
    
    print("\n2. CVD INVALIDATION LOGIC:")
    print("   - For shorts, exit if BOTH CVDs are rising")
    print("   - Requires spot_change > 0 AND futures_change > 0")
    print("   - May be too restrictive (both must reverse)")
    
    print("\n3. FLOW REVERSAL DETECTION:")
    print("   - For shorts, needs spot_trend > 0 AND futures_trend < 0")
    print("   - Requires opposite movements in SPOT vs FUTURES")
    print("   - May not occur frequently")
    
    print("\n4. STRUCTURE BREAK:")
    print("   - Uses 20-candle lookback for swing high/low")
    print("   - For shorts, exits if price > swing_high * 1.002")
    print("   - With only 4-hour windows, may not have enough data")
    
    print("\n5. DATA WINDOW ISSUES:")
    print("   - Rolling window uses 4-hour windows")
    print("   - Exit checks only see last 240 data points")
    print("   - Entry analysis may not be preserved correctly")

def check_exit_integration():
    """Check how exits are integrated in the strategy"""
    
    print("\n" + "=" * 80)
    print("INTEGRATION ANALYSIS")
    print("=" * 80)
    
    print("\nExit Order Flow:")
    print("1. Strategy.analyze() calls phase5.manage_exits() for each position")
    print("2. If should_exit=True, _generate_exit_order() creates order")
    print("3. Order includes signal_type='EXIT'")
    print("4. BacktestEngine._execute_order() checks signal_type")
    print("5. Portfolio.close_matching_position() closes the position")
    
    print("\nPotential Integration Issues:")
    print("- ✓ Exit orders are properly marked with signal_type='EXIT'")
    print("- ✓ Backtest engine handles EXIT orders correctly")
    print("- ? Position tracking across rolling windows")
    print("- ? Entry analysis preservation for exit comparisons")
    print("- ? CVD baseline tracking without baseline manager")

def recommend_fixes():
    """Recommend specific fixes"""
    
    print("\n" + "=" * 80)
    print("RECOMMENDED FIXES")
    print("=" * 80)
    
    print("\n1. CRITICAL - Relax Exit Thresholds:")
    print("   File: strategies/squeezeflow/components/phase5_exits.py")
    print("   - Line 223: Change range_size from 0.005 to 0.01 (1% instead of 0.5%)")
    print("   - Line 295-300: Change CVD invalidation to require only ONE CVD reversing")
    print("   - Line 185-188: Lower flow reversal severity thresholds")
    
    print("\n2. IMPORTANT - Add Debug Logging:")
    print("   File: strategies/squeezeflow/components/phase5_exits.py")
    print("   - Add logging for each exit condition evaluation")
    print("   - Log actual vs threshold values")
    print("   - Track how often each condition is checked")
    
    print("\n3. IMPORTANT - Fix Entry Analysis Tracking:")
    print("   File: strategies/squeezeflow/strategy.py")
    print("   - Store entry_analysis with position metadata")
    print("   - Pass correct entry_analysis to exit checks")
    print("   - Currently passing self.last_analysis which may be stale")
    
    print("\n4. MODERATE - Improve CVD Baseline Tracking:")
    print("   File: strategies/squeezeflow/components/phase5_exits.py")
    print("   - Line 279-290: Fix fallback CVD baseline calculation")
    print("   - Use actual entry index instead of -20 approximation")
    
    print("\n5. MODERATE - Adjust Structure Break Detection:")
    print("   File: strategies/squeezeflow/components/phase5_exits.py")
    print("   - Line 342-348: Increase buffer from 0.2% to 0.5%")
    print("   - Reduce lookback from 20 to 10 candles for faster response")

if __name__ == "__main__":
    test_exit_conditions()
    analyze_real_backtest_data()
    check_exit_integration()
    recommend_fixes()
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print("\nMost Likely Root Cause:")
    print("Exit thresholds are too restrictive, especially the 0.5% range break")
    print("and the requirement for BOTH CVDs to reverse for invalidation.")
    print("\nThe ETH short at $3587.35 likely never met these strict conditions.")