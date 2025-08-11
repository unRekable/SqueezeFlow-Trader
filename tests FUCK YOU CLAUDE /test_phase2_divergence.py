"""
Phase 2 Divergence Detection Tests
Test Phase 2 divergence detection accuracy and fix Direction: NONE issues
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategies.squeezeflow.components.phase2_divergence import DivergenceDetection


class TestPhase2DivergenceDetection:
    """Test Phase 2 divergence detection accuracy and eliminate Direction: NONE issues"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.phase2 = DivergenceDetection()
    
    def test_long_setup_detection_accuracy(self):
        """Test accurate long setup detection (not NONE)"""
        # Arrange: Clear long squeeze scenario
        context = {'market_bias': 'NEUTRAL'}
        
        # Create explicit long setup conditions:
        # - Price stable/rising
        # - Spot CVD up (buying pressure in spot)  
        # - Futures CVD down (selling pressure in futures)
        scenario = self._create_long_squeeze_scenario()
        
        # Act: Detect divergence
        result = self.phase2.detect_divergence(scenario, context)
        
        # Assert: Should detect LONG_SETUP, not NONE
        assert result['setup_type'] == 'LONG_SETUP', \
            f"Expected LONG_SETUP, got {result['setup_type']}. Error: {result.get('error', 'None')}"
        
        assert result['is_significant'] == True, \
            f"Should be significant divergence. Volume significance: {result.get('volume_significance', {})}"
        
        assert 'error' not in result, f"Unexpected error: {result.get('error', 'None')}"
        
        # Detailed pattern validation
        cvd_patterns = result['cvd_patterns']
        assert cvd_patterns['pattern'] == 'SPOT_LEADING_UP', \
            f"Wrong CVD pattern: {cvd_patterns['pattern']}. Expected SPOT_LEADING_UP"
        
        assert cvd_patterns['spot_direction'] > 0, \
            f"Spot should be up (positive): {cvd_patterns['spot_direction']}"
        
        assert cvd_patterns['futures_direction'] < 0, \
            f"Futures should be down (negative): {cvd_patterns['futures_direction']}"
        
        # Price pattern validation
        price_pattern = result['price_pattern']
        assert price_pattern['movement'] in ['STABLE', 'RISING'], \
            f"Price should be stable/rising for long setup: {price_pattern['movement']}"
    
    def test_short_setup_detection_accuracy(self):
        """Test accurate short setup detection (not NONE)"""
        # Arrange: Clear short squeeze scenario
        context = {'market_bias': 'NEUTRAL'}
        
        # Create explicit short setup conditions:
        # - Price stable/falling
        # - Spot CVD down (selling pressure in spot)
        # - Futures CVD up (buying pressure in futures)
        scenario = self._create_short_squeeze_scenario()
        
        # Act: Detect divergence
        result = self.phase2.detect_divergence(scenario, context)
        
        # Assert: Should detect SHORT_SETUP, not NONE
        assert result['setup_type'] == 'SHORT_SETUP', \
            f"Expected SHORT_SETUP, got {result['setup_type']}. Error: {result.get('error', 'None')}"
        
        assert result['is_significant'] == True, \
            "Should be significant divergence"
        
        # Detailed pattern validation
        cvd_patterns = result['cvd_patterns']
        assert cvd_patterns['pattern'] == 'FUTURES_LEADING_UP', \
            f"Wrong CVD pattern: {cvd_patterns['pattern']}. Expected FUTURES_LEADING_UP"
        
        assert cvd_patterns['spot_direction'] < 0, \
            f"Spot should be down (negative): {cvd_patterns['spot_direction']}"
        
        assert cvd_patterns['futures_direction'] > 0, \
            f"Futures should be up (positive): {cvd_patterns['futures_direction']}"
        
        # Price pattern validation
        price_pattern = result['price_pattern']
        assert price_pattern['movement'] in ['STABLE', 'FALLING'], \
            f"Price should be stable/falling for short setup: {price_pattern['movement']}"
    
    def test_no_setup_scenarios_valid(self):
        """Test NO_SETUP scenarios are properly identified (not NONE)"""
        
        test_scenarios = [
            ('neutral_market', self._create_neutral_market_scenario()),
            ('insufficient_data', self._create_insufficient_data_scenario()),
            ('conflicting_signals', self._create_conflicting_signals_scenario()),
            ('weak_divergence', self._create_weak_divergence_scenario())
        ]
        
        context = {'market_bias': 'NEUTRAL'}
        
        for scenario_name, scenario_data in test_scenarios:
            result = self.phase2.detect_divergence(scenario_data, context)
            
            # Should have valid setup_type (not NONE unless explicitly NO_SETUP)
            valid_setups = ['LONG_SETUP', 'SHORT_SETUP', 'NO_SETUP']
            assert result['setup_type'] in valid_setups, \
                f"Scenario '{scenario_name}': Invalid setup_type: {result['setup_type']}. Valid: {valid_setups}"
            
            # Should not have NONE as direction anywhere
            if 'direction' in result:
                assert result['direction'] != 'NONE', \
                    f"Scenario '{scenario_name}': Direction should not be NONE"
            
            # Should have proper error handling if insufficient data
            if result['setup_type'] == 'NO_SETUP':
                # This is valid - some scenarios genuinely have no setup
                pass
            else:
                # Should not have errors for valid scenarios
                assert 'error' not in result or 'insufficient' in result.get('error', '').lower(), \
                    f"Scenario '{scenario_name}': Unexpected error: {result.get('error', 'None')}"
    
    def test_direction_none_edge_cases_fixed(self):
        """Test edge cases that previously caused Direction: NONE are now fixed"""
        
        # Edge case 1: Very small data set
        small_data_case = self._create_edge_case_small_data()
        result = self.phase2.detect_divergence(small_data_case, {'market_bias': 'NEUTRAL'})
        
        assert result['setup_type'] in ['LONG_SETUP', 'SHORT_SETUP', 'NO_SETUP'], \
            f"Small data case: Invalid setup_type: {result['setup_type']}"
        
        # Edge case 2: All zero CVD changes
        zero_cvd_case = self._create_edge_case_zero_cvd()
        result = self.phase2.detect_divergence(zero_cvd_case, {'market_bias': 'NEUTRAL'})
        
        assert result['setup_type'] in ['LONG_SETUP', 'SHORT_SETUP', 'NO_SETUP'], \
            f"Zero CVD case: Invalid setup_type: {result['setup_type']}"
        
        # Edge case 3: Extreme price volatility
        volatile_case = self._create_edge_case_high_volatility()
        result = self.phase2.detect_divergence(volatile_case, {'market_bias': 'NEUTRAL'})
        
        assert result['setup_type'] in ['LONG_SETUP', 'SHORT_SETUP', 'NO_SETUP'], \
            f"High volatility case: Invalid setup_type: {result['setup_type']}"
        
        # Edge case 4: Missing or NaN data
        missing_data_case = self._create_edge_case_missing_data()
        result = self.phase2.detect_divergence(missing_data_case, {'market_bias': 'NEUTRAL'})
        
        assert result['setup_type'] in ['LONG_SETUP', 'SHORT_SETUP', 'NO_SETUP'], \
            f"Missing data case: Invalid setup_type: {result['setup_type']}"
    
    def test_market_bias_influence_on_setup_detection(self):
        """Test market bias correctly influences setup detection"""
        
        # Same divergence scenario with different market biases
        base_scenario = self._create_long_squeeze_scenario()
        
        test_cases = [
            ('BULLISH', 'LONG_SETUP'),    # Bullish bias should favor long setups
            ('NEUTRAL', 'LONG_SETUP'),    # Neutral should allow both
            ('BEARISH', 'NO_SETUP')       # Bearish bias should reject long setups
        ]
        
        for market_bias, expected_result in test_cases:
            context = {'market_bias': market_bias}
            result = self.phase2.detect_divergence(base_scenario, context)
            
            if market_bias == 'BEARISH':
                # Bearish bias should reject long setups
                assert result['setup_type'] != 'LONG_SETUP', \
                    f"Bearish bias should reject LONG_SETUP, got {result['setup_type']}"
            else:
                # Should detect long setup with bullish/neutral bias
                assert result['setup_type'] == expected_result, \
                    f"Market bias {market_bias}: Expected {expected_result}, got {result['setup_type']}"
    
    def test_volume_significance_calculation(self):
        """Test volume significance calculation for divergence validation"""
        
        # Create scenario with known volume significance
        scenario = self._create_volume_significance_test_scenario()
        result = self.phase2.detect_divergence(scenario, {'market_bias': 'NEUTRAL'})
        
        volume_significance = result['volume_significance']
        
        # Should have all required fields
        required_fields = ['is_significant', 'multiplier', 'recent_avg', 'current']
        for field in required_fields:
            assert field in volume_significance, f"Missing volume significance field: {field}"
        
        # Multiplier should be reasonable (>0, <100)
        multiplier = volume_significance['multiplier']
        assert 0 <= multiplier <= 100, f"Unreasonable multiplier: {multiplier}"
        
        # Current divergence should be positive (absolute value)
        current = volume_significance['current']
        assert current >= 0, f"Current divergence should be absolute value: {current}"
    
    def test_pattern_recognition_accuracy(self):
        """Test CVD pattern recognition accuracy"""
        
        # Test specific patterns
        pattern_tests = [
            {
                'name': 'spot_leading_up',
                'spot_change': 5000,      # Spot CVD up
                'futures_change': -3000,  # Futures CVD down
                'expected_pattern': 'SPOT_LEADING_UP'
            },
            {
                'name': 'futures_leading_up', 
                'spot_change': -4000,     # Spot CVD down
                'futures_change': 2000,   # Futures CVD up
                'expected_pattern': 'FUTURES_LEADING_UP'
            },
            {
                'name': 'both_up',
                'spot_change': 3000,      # Both up
                'futures_change': 2000,
                'expected_pattern': 'BOTH_UP'
            },
            {
                'name': 'both_down',
                'spot_change': -2000,     # Both down
                'futures_change': -3000,
                'expected_pattern': 'BOTH_DOWN'
            }
        ]
        
        for test_case in pattern_tests:
            scenario = self._create_pattern_test_scenario(
                test_case['spot_change'], 
                test_case['futures_change']
            )
            
            result = self.phase2.detect_divergence(scenario, {'market_bias': 'NEUTRAL'})
            
            cvd_patterns = result['cvd_patterns']
            assert cvd_patterns['pattern'] == test_case['expected_pattern'], \
                f"Pattern {test_case['name']}: Expected {test_case['expected_pattern']}, got {cvd_patterns['pattern']}"
    
    @pytest.mark.parametrize("scenario_type,expected_setup", [
        ("strong_long_divergence", "LONG_SETUP"),
        ("strong_short_divergence", "SHORT_SETUP"),
        ("weak_divergence", "NO_SETUP"),
        ("no_divergence", "NO_SETUP")
    ])
    def test_divergence_strength_thresholds(self, scenario_type, expected_setup):
        """Test divergence strength affects setup detection"""
        
        scenario = self._create_divergence_strength_scenario(scenario_type)
        result = self.phase2.detect_divergence(scenario, {'market_bias': 'NEUTRAL'})
        
        if expected_setup == "NO_SETUP":
            assert result['setup_type'] == expected_setup, \
                f"Scenario {scenario_type}: Expected {expected_setup}, got {result['setup_type']}"
            assert not result['is_significant'], \
                f"Scenario {scenario_type}: Should not be significant"
        else:
            assert result['setup_type'] == expected_setup, \
                f"Scenario {scenario_type}: Expected {expected_setup}, got {result['setup_type']}"
            assert result['is_significant'], \
                f"Scenario {scenario_type}: Should be significant"
    
    # Helper methods for creating test scenarios
    
    def _create_long_squeeze_scenario(self):
        """Create clear long squeeze scenario"""
        dates = pd.date_range(start='2024-08-01', periods=50, freq='5min')
        
        # Price stable to slightly rising
        close_prices = np.linspace(50000, 50200, 50)  # 0.4% rise (stable)
        
        # Spot CVD trending up (buying pressure)
        spot_cvd = pd.Series(np.linspace(0, 8000, 50), index=dates)
        
        # Futures CVD trending down (selling pressure) 
        futures_cvd = pd.Series(np.linspace(0, -5000, 50), index=dates)
        
        # CVD divergence (large and growing)
        cvd_divergence = spot_cvd - futures_cvd
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'open': close_prices * 0.999,  # Slightly lower open
            'high': close_prices * 1.002,  # Slightly higher high
            'low': close_prices * 0.998,   # Slightly lower low
            'close': close_prices,
            'volume': np.random.uniform(1000, 2000, 50)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': cvd_divergence,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_short_squeeze_scenario(self):
        """Create clear short squeeze scenario"""
        dates = pd.date_range(start='2024-08-01', periods=50, freq='5min')
        
        # Price stable to slightly falling
        close_prices = np.linspace(50000, 49800, 50)  # 0.4% fall (stable)
        
        # Spot CVD trending down (selling pressure)
        spot_cvd = pd.Series(np.linspace(0, -6000, 50), index=dates)
        
        # Futures CVD trending up (buying pressure)
        futures_cvd = pd.Series(np.linspace(0, 4000, 50), index=dates)
        
        cvd_divergence = spot_cvd - futures_cvd
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices * 1.001,
            'high': close_prices * 1.002,
            'low': close_prices * 0.998,
            'volume': np.random.uniform(1000, 2000, 50)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': cvd_divergence,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_neutral_market_scenario(self):
        """Create neutral market scenario (no clear setup)"""
        dates = pd.date_range(start='2024-08-01', periods=50, freq='5min')
        
        # Price sideways
        close_prices = np.full(50, 50000) + np.random.normal(0, 50, 50)
        
        # CVD both moving in same direction (no divergence)
        base_cvd_trend = np.linspace(0, 1000, 50)
        spot_cvd = pd.Series(base_cvd_trend + np.random.normal(0, 200, 50), index=dates)
        futures_cvd = pd.Series(base_cvd_trend + np.random.normal(0, 200, 50), index=dates)
        
        cvd_divergence = spot_cvd - futures_cvd
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices,
            'high': close_prices * 1.001,
            'low': close_prices * 0.999,
            'volume': np.random.uniform(1000, 2000, 50)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': cvd_divergence,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_insufficient_data_scenario(self):
        """Create scenario with insufficient data"""
        dates = pd.date_range(start='2024-08-01', periods=5, freq='5min')  # Only 5 data points
        
        close_prices = [50000, 50100, 50050, 50150, 50100]
        spot_cvd = pd.Series([0, 100, 200, 150, 300], index=dates)
        futures_cvd = pd.Series([0, 50, 100, 200, 100], index=dates)
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices,
            'high': [p * 1.001 for p in close_prices],
            'low': [p * 0.999 for p in close_prices],
            'volume': [1000] * 5
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_conflicting_signals_scenario(self):
        """Create scenario with conflicting signals"""
        dates = pd.date_range(start='2024-08-01', periods=50, freq='5min')
        
        # Price rising (bullish signal)
        close_prices = np.linspace(50000, 52000, 50)  # 4% rise (strong bullish)
        
        # But CVD shows bearish divergence
        spot_cvd = pd.Series(np.linspace(0, -3000, 50), index=dates)  # Selling pressure
        futures_cvd = pd.Series(np.linspace(0, 2000, 50), index=dates)   # Buying pressure
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices * 0.998,
            'high': close_prices * 1.005,
            'low': close_prices * 0.995,
            'volume': np.random.uniform(1500, 3000, 50)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_weak_divergence_scenario(self):
        """Create scenario with weak divergence (below significance threshold)"""
        dates = pd.date_range(start='2024-08-01', periods=50, freq='5min')
        
        # Price stable
        close_prices = np.full(50, 50000) + np.random.normal(0, 25, 50)
        
        # Small CVD divergence (not significant)
        spot_cvd = pd.Series(np.linspace(0, 500, 50), index=dates)
        futures_cvd = pd.Series(np.linspace(0, 300, 50), index=dates)
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices,
            'high': close_prices * 1.0005,
            'low': close_prices * 0.9995,
            'volume': np.random.uniform(1000, 1500, 50)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_edge_case_small_data(self):
        """Create edge case with very small dataset"""
        dates = pd.date_range(start='2024-08-01', periods=3, freq='5min')
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': [50000, 50100, 50200],
            'open': [49950, 50050, 50150],
            'high': [50050, 50150, 50250],
            'low': [49900, 50000, 50100],
            'volume': [1000, 1100, 1200]
        })
        
        spot_cvd = pd.Series([0, 100, 150], index=dates)
        futures_cvd = pd.Series([0, -50, -100], index=dates)
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_edge_case_zero_cvd(self):
        """Create edge case with zero CVD changes"""
        dates = pd.date_range(start='2024-08-01', periods=30, freq='5min')
        
        # All CVD values are zero (no volume delta)
        spot_cvd = pd.Series(np.zeros(30), index=dates)
        futures_cvd = pd.Series(np.zeros(30), index=dates)
        
        close_prices = np.full(30, 50000)
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices,
            'high': close_prices,
            'low': close_prices,
            'volume': np.full(30, 1000)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_edge_case_high_volatility(self):
        """Create edge case with extreme price volatility"""
        dates = pd.date_range(start='2024-08-01', periods=30, freq='5min')
        
        # Highly volatile prices
        base_price = 50000
        price_changes = np.random.normal(0, 1000, 30)  # High volatility
        close_prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = close_prices[-1] + change
            close_prices.append(max(new_price, 1000))  # Prevent negative prices
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': [p * 0.99 for p in close_prices],
            'high': [p * 1.02 for p in close_prices],
            'low': [p * 0.98 for p in close_prices],
            'volume': np.random.uniform(2000, 5000, 30)
        })
        
        # Moderate CVD divergence
        spot_cvd = pd.Series(np.linspace(0, 2000, 30), index=dates)
        futures_cvd = pd.Series(np.linspace(0, -1500, 30), index=dates)
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_edge_case_missing_data(self):
        """Create edge case with missing/NaN data"""
        dates = pd.date_range(start='2024-08-01', periods=30, freq='5min')
        
        close_prices = np.full(30, 50000)
        close_prices[10:15] = np.nan  # Missing data in middle
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': close_prices,
            'open': close_prices,
            'high': close_prices,
            'low': close_prices,
            'volume': np.full(30, 1000)
        })
        
        # CVD with some missing values
        spot_cvd_values = np.linspace(0, 1000, 30)
        spot_cvd_values[5:10] = np.nan
        spot_cvd = pd.Series(spot_cvd_values, index=dates)
        
        futures_cvd_values = np.linspace(0, -800, 30)
        futures_cvd = pd.Series(futures_cvd_values, index=dates)
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_pattern_test_scenario(self, spot_change, futures_change):
        """Create scenario to test specific CVD patterns"""
        dates = pd.date_range(start='2024-08-01', periods=20, freq='5min')
        
        # Create CVD with specified changes
        spot_cvd = pd.Series(np.linspace(0, spot_change, 20), index=dates)
        futures_cvd = pd.Series(np.linspace(0, futures_change, 20), index=dates)
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': np.full(20, 50000),
            'open': np.full(20, 50000),
            'high': np.full(20, 50100),
            'low': np.full(20, 49900),
            'volume': np.full(20, 1000)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_volume_significance_test_scenario(self):
        """Create scenario to test volume significance calculation"""
        dates = pd.date_range(start='2024-08-01', periods=50, freq='5min')
        
        # Create recent activity baseline (first 30 points)
        recent_divergence = np.random.uniform(-500, 500, 30)
        
        # Create significant divergence (last 20 points)
        significant_divergence = np.random.uniform(-2000, 2000, 20)  # 4x larger
        
        all_divergence = np.concatenate([recent_divergence, significant_divergence])
        
        # Build CVDs from divergence
        spot_cvd = pd.Series(np.cumsum(np.random.uniform(-100, 100, 50)), index=dates)
        futures_cvd = spot_cvd - all_divergence
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': np.full(50, 50000),
            'open': np.full(50, 50000),
            'high': np.full(50, 50100),
            'low': np.full(50, 49900),
            'volume': np.full(50, 1000)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': pd.Series(all_divergence, index=dates),
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }
    
    def _create_divergence_strength_scenario(self, scenario_type):
        """Create scenarios with different divergence strengths"""
        dates = pd.date_range(start='2024-08-01', periods=40, freq='5min')
        
        if scenario_type == "strong_long_divergence":
            # Strong long setup
            spot_cvd = pd.Series(np.linspace(0, 10000, 40), index=dates)  # Strong up
            futures_cvd = pd.Series(np.linspace(0, -6000, 40), index=dates)  # Strong down
            
        elif scenario_type == "strong_short_divergence":
            # Strong short setup
            spot_cvd = pd.Series(np.linspace(0, -8000, 40), index=dates)  # Strong down
            futures_cvd = pd.Series(np.linspace(0, 5000, 40), index=dates)  # Strong up
            
        elif scenario_type == "weak_divergence":
            # Weak divergence
            spot_cvd = pd.Series(np.linspace(0, 500, 40), index=dates)   # Weak up
            futures_cvd = pd.Series(np.linspace(0, -300, 40), index=dates)  # Weak down
            
        else:  # no_divergence
            # No meaningful divergence
            base_trend = np.linspace(0, 1000, 40)
            spot_cvd = pd.Series(base_trend + np.random.normal(0, 50, 40), index=dates)
            futures_cvd = pd.Series(base_trend + np.random.normal(0, 50, 40), index=dates)
        
        ohlcv = pd.DataFrame({
            'time': dates,
            'close': np.full(40, 50000) + np.random.normal(0, 25, 40),
            'open': np.full(40, 50000),
            'high': np.full(40, 50100),
            'low': np.full(40, 49900),
            'volume': np.random.uniform(1000, 2000, 40)
        })
        
        return {
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'ohlcv': ohlcv,
            'symbol': 'BTCUSDT'
        }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=long'])