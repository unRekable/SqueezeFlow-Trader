"""
Property-Based Tests for SqueezeFlow Trader
Robust validation using hypothesis for edge cases and invariants
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Import components to test
from strategies.squeezeflow.config import SqueezeFlowConfig
from data.processors.cvd_calculator import CVDCalculator
from backtest.core.portfolio import Portfolio


class TestCVDCalculationProperties:
    """Property-based tests for CVD calculation invariants"""
    
    @given(
        vbuy=st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False), min_size=10, max_size=100),
        vsell=st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False), min_size=10, max_size=100)
    )
    @pytest.mark.property_based
    def test_cvd_calculation_properties(self, vbuy, vsell):
        """Test CVD calculation properties with random data"""
        # Ensure lists are same length
        min_len = min(len(vbuy), len(vsell))
        vbuy = vbuy[:min_len]
        vsell = vsell[:min_len]
        
        # Create test dataframe
        test_data = pd.DataFrame({
            'total_vbuy': vbuy,
            'total_vsell': vsell,
            'time': pd.date_range(start='2024-01-01', periods=min_len, freq='5min')
        })
        
        calculator = CVDCalculator()
        cvd = calculator.calculate_spot_cvd(test_data)
        
        # Properties that should always hold
        assert len(cvd) == len(vbuy), "CVD length should match input length"
        assert isinstance(cvd, pd.Series), "CVD should be pandas Series"
        
        # CVD should be cumulative sum of volume delta
        volume_delta = np.array(vbuy) - np.array(vsell)
        expected_cvd = np.cumsum(volume_delta)
        
        # Allow for small floating point differences
        np.testing.assert_allclose(cvd.values, expected_cvd, rtol=1e-10)
        
        # CVD monotonicity property: if all deltas are positive, CVD should be increasing
        if all(d >= 0 for d in volume_delta):
            assert cvd.is_monotonic_increasing or cvd.nunique() == 1, "CVD should be monotonic when all deltas >= 0"
        
        # First CVD value should equal first volume delta
        if len(cvd) > 0:
            assert abs(cvd.iloc[0] - volume_delta[0]) < 1e-10, "First CVD should equal first volume delta"
    
    @given(
        spot_cvd=st.lists(st.floats(min_value=-1000000, max_value=1000000, allow_nan=False), min_size=10, max_size=50),
        futures_cvd=st.lists(st.floats(min_value=-1000000, max_value=1000000, allow_nan=False), min_size=10, max_size=50)
    )
    @pytest.mark.property_based
    def test_cvd_divergence_properties(self, spot_cvd, futures_cvd):
        """Test CVD divergence calculation properties"""
        # Ensure same length
        min_len = min(len(spot_cvd), len(futures_cvd))
        spot_cvd = spot_cvd[:min_len]
        futures_cvd = futures_cvd[:min_len]
        
        dates = pd.date_range(start='2024-01-01', periods=min_len, freq='5min')
        spot_series = pd.Series(spot_cvd, index=dates)
        futures_series = pd.Series(futures_cvd, index=dates)
        
        calculator = CVDCalculator()
        divergence = calculator.calculate_cvd_divergence(spot_series, futures_series)
        
        # Divergence properties
        assert len(divergence) == min_len, "Divergence length should match input"
        assert isinstance(divergence, pd.Series), "Divergence should be pandas Series"
        
        # Divergence should be the difference
        expected_divergence = np.array(spot_cvd) - np.array(futures_cvd)
        np.testing.assert_allclose(divergence.values, expected_divergence, rtol=1e-10)
        
        # If spot and futures are identical, divergence should be zero
        if np.allclose(spot_cvd, futures_cvd):
            assert np.allclose(divergence.values, 0), "Divergence should be zero when CVDs are identical"


class TestPortfolioProperties:
    """Property-based tests for portfolio management invariants"""
    
    @given(
        initial_balance=st.floats(min_value=1000, max_value=100000),
        quantity=st.floats(min_value=0.001, max_value=1.0),
        price=st.floats(min_value=1000, max_value=100000)
    )
    @pytest.mark.property_based
    def test_portfolio_balance_invariants(self, initial_balance, quantity, price):
        """Test portfolio balance invariants"""
        portfolio = Portfolio(initial_balance)
        
        initial_state = portfolio.get_state()
        assert initial_state['total_value'] == initial_balance
        assert initial_state['available_balance'] == initial_balance
        assert len(initial_state['positions']) == 0
        
        # Try to open position (might fail due to insufficient balance)
        position_value = quantity * price
        
        if position_value <= initial_balance * 0.9:  # Leave some buffer for fees
            success = portfolio.open_long_position(
                symbol='TESTUSDT',
                quantity=quantity,
                price=price,
                timestamp=datetime.now()
            )
            
            if success:
                state_after = portfolio.get_state()
                
                # Balance should decrease
                assert state_after['available_balance'] < initial_balance
                
                # Should have one position
                assert len(state_after['positions']) == 1
                
                # Position should have correct properties
                position = state_after['positions'][0]
                assert position['symbol'] == 'TESTUSDT'
                assert position['side'] == 'BUY'
                assert abs(position['quantity'] - quantity) < 1e-10
                assert abs(position['entry_price'] - price) < 1e-10
                
                # Total value should be conserved (balance + position value)
                position_current_value = position['quantity'] * position.get('current_price', price)
                total_value = state_after['available_balance'] + position_current_value
                
                # Should be close to initial balance (minus fees)
                assert abs(total_value - initial_balance) <= initial_balance * 0.01  # Allow 1% for fees
    
    @given(
        balance=st.floats(min_value=5000, max_value=50000),
        num_positions=st.integers(min_value=1, max_value=5),
        base_price=st.floats(min_value=10000, max_value=80000)
    )
    @pytest.mark.property_based
    def test_portfolio_multiple_positions(self, balance, num_positions, base_price):
        """Test portfolio with multiple positions"""
        portfolio = Portfolio(balance)
        
        opened_positions = 0
        position_value_sum = 0
        
        for i in range(num_positions):
            quantity = 0.001 * (i + 1)  # Small varying quantities
            price = base_price + i * 100  # Varying prices
            position_value = quantity * price
            
            # Only open position if we have enough balance
            if position_value_sum + position_value <= balance * 0.8:  # Leave buffer
                success = portfolio.open_long_position(
                    symbol=f'TEST{i}USDT',
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now()
                )
                
                if success:
                    opened_positions += 1
                    position_value_sum += position_value
        
        state = portfolio.get_state()
        
        # Should have correct number of positions
        assert len(state['positions']) == opened_positions
        
        # Each position should be unique
        symbols = [pos['symbol'] for pos in state['positions']]
        assert len(set(symbols)) == len(symbols), "All positions should have unique symbols"
        
        # Available balance should be reduced
        if opened_positions > 0:
            assert state['available_balance'] < balance


class TestStrategyConfigProperties:
    """Property-based tests for strategy configuration"""
    
    @given(
        score=st.floats(min_value=0, max_value=10)
    )
    @pytest.mark.property_based
    def test_position_size_factor_properties(self, score):
        """Test position size factor calculation properties"""
        config = SqueezeFlowConfig()
        
        factor = config.get_position_size_factor(score)
        
        # Factor should be non-negative
        assert factor >= 0
        
        # Factor should increase with score
        if score >= 8:
            assert factor == config.position_size_by_score["8+"]
        elif score >= 6:
            assert factor == config.position_size_by_score["6-7"]
        elif score >= 4:
            assert factor == config.position_size_by_score["4-5"]
        else:
            assert factor == config.position_size_by_score["0-3"]
            
        # Higher scores should have higher or equal factors
        higher_score = min(score + 1, 10)
        higher_factor = config.get_position_size_factor(higher_score)
        assert higher_factor >= factor
    
    @given(
        score=st.floats(min_value=0, max_value=10)
    )
    @pytest.mark.property_based
    def test_leverage_properties(self, score):
        """Test leverage calculation properties"""
        config = SqueezeFlowConfig()
        
        leverage = config.get_leverage(score)
        
        # Leverage should be non-negative integer
        assert leverage >= 0
        assert isinstance(leverage, int)
        
        # Leverage should be reasonable (not too high)
        assert leverage <= 10
        
        # Higher scores should have higher or equal leverage
        higher_score = min(score + 1, 10)
        higher_leverage = config.get_leverage(higher_score)
        assert higher_leverage >= leverage


class TestDataValidationProperties:
    """Property-based tests for data validation"""
    
    @given(
        num_points=st.integers(min_value=10, max_value=1000),
        base_price=st.floats(min_value=1000, max_value=100000),
        volatility=st.floats(min_value=0.001, max_value=0.1)
    )
    @pytest.mark.property_based
    def test_ohlcv_data_properties(self, num_points, base_price, volatility):
        """Test OHLCV data validation properties"""
        
        # Generate realistic OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=num_points, freq='5min')
        
        prices = []
        current_price = base_price
        
        for _ in range(num_points):
            # Random walk
            change = np.random.normal(0, base_price * volatility)
            current_price = max(current_price + change, base_price * 0.1)  # Prevent negative prices
            prices.append(current_price)
        
        # Create OHLCV data with proper constraints
        ohlcv_data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else close_price
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high_base = max(open_price, close_price)
            low_base = min(open_price, close_price)
            
            high = high_base + abs(np.random.normal(0, high_base * volatility * 0.1))
            low = low_base - abs(np.random.normal(0, low_base * volatility * 0.1))
            
            # Ensure low >= 0
            low = max(low, base_price * 0.01)
            
            ohlcv_data.append({
                'time': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': abs(np.random.normal(1000, 500))
            })
        
        df = pd.DataFrame(ohlcv_data)
        
        # OHLCV invariants
        assert len(df) == num_points
        
        # High should be >= open and close
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        
        # Low should be <= open and close  
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        
        # All prices should be positive
        assert (df['open'] > 0).all()
        assert (df['high'] > 0).all()
        assert (df['low'] > 0).all()
        assert (df['close'] > 0).all()
        
        # Volume should be non-negative
        assert (df['volume'] >= 0).all()
        
        # Time should be monotonic
        assert df['time'].is_monotonic_increasing
    
    @given(
        data_points=st.integers(min_value=50, max_value=500),
        missing_ratio=st.floats(min_value=0, max_value=0.3)
    )
    @pytest.mark.property_based
    def test_data_completeness_properties(self, data_points, missing_ratio):
        """Test data completeness validation"""
        
        # Create dataset with some missing data
        dates = pd.date_range(start='2024-01-01', periods=data_points, freq='5min')
        
        # Randomly remove some data points
        num_missing = int(data_points * missing_ratio)
        if num_missing > 0:
            missing_indices = np.random.choice(data_points, num_missing, replace=False)
            valid_indices = [i for i in range(data_points) if i not in missing_indices]
        else:
            valid_indices = list(range(data_points))
        
        # Create sparse data
        ohlcv = pd.DataFrame({
            'time': [dates[i] for i in valid_indices],
            'open': np.random.uniform(45000, 55000, len(valid_indices)),
            'high': np.random.uniform(50000, 60000, len(valid_indices)),
            'low': np.random.uniform(40000, 50000, len(valid_indices)),
            'close': np.random.uniform(45000, 55000, len(valid_indices)),
            'volume': np.random.uniform(1000, 5000, len(valid_indices))
        })
        
        spot_cvd = pd.Series(
            np.random.uniform(-10000, 10000, len(valid_indices)),
            index=[dates[i] for i in valid_indices]
        )
        
        futures_cvd = pd.Series(
            np.random.uniform(-15000, 15000, len(valid_indices)),
            index=[dates[i] for i in valid_indices]
        )
        
        dataset = {
            'ohlcv': ohlcv,
            'spot_cvd': spot_cvd,
            'futures_cvd': futures_cvd,
            'cvd_divergence': spot_cvd - futures_cvd,
            'metadata': {
                'data_points': len(ohlcv),
                'spot_markets_count': 2,
                'futures_markets_count': 1
            }
        }
        
        # Data completeness properties
        actual_completeness = len(valid_indices) / data_points
        expected_completeness = 1.0 - missing_ratio
        
        # Should be close to expected (within some tolerance due to randomness)
        assert abs(actual_completeness - expected_completeness) < 0.05
        
        # All data should have same length
        assert len(dataset['ohlcv']) == len(dataset['spot_cvd'])
        assert len(dataset['spot_cvd']) == len(dataset['futures_cvd'])
        assert len(dataset['futures_cvd']) == len(dataset['cvd_divergence'])


class TestOrderGenerationProperties:
    """Property-based tests for order generation"""
    
    @given(
        quantity=st.floats(min_value=0.001, max_value=10.0),
        price=st.floats(min_value=100, max_value=200000),
        leverage=st.integers(min_value=1, max_value=10)
    )
    @pytest.mark.property_based
    def test_order_properties(self, quantity, price, leverage):
        """Test order generation properties"""
        
        # Create a mock order
        order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': quantity,
            'price': price,
            'leverage': leverage,
            'timestamp': datetime.now()
        }
        
        # Order invariants
        assert order['quantity'] > 0
        assert order['price'] > 0
        assert order['leverage'] >= 1
        assert order['side'] in ['BUY', 'SELL']
        
        # Position value calculation
        position_value = order['quantity'] * order['price']
        leveraged_value = position_value * order['leverage']
        
        assert position_value > 0
        assert leveraged_value >= position_value
        
        # Leverage amplification
        assert leveraged_value == position_value * leverage
    
    @given(
        scores=st.lists(st.floats(min_value=0, max_value=10), min_size=5, max_size=20)
    )
    @pytest.mark.property_based
    def test_scoring_consistency_properties(self, scores):
        """Test scoring system consistency"""
        config = SqueezeFlowConfig()
        
        for score in scores:
            # Test consistency between position sizing and leverage
            position_factor = config.get_position_size_factor(score)
            leverage = config.get_leverage(score)
            
            # Higher scores should generally have higher position factors and leverage
            if score >= 6.0:
                assert position_factor >= 1.0, f"High score {score} should have position factor >= 1.0"
                assert leverage >= 3, f"High score {score} should have leverage >= 3"
            
            if score < 4.0:
                assert position_factor == 0.0, f"Low score {score} should have zero position factor"
                assert leverage == 0, f"Low score {score} should have zero leverage"


# Configuration for property-based tests
class TestConfiguration:
    """Configuration for property-based testing"""
    
    @settings(max_examples=50, deadline=5000)  # Reduce examples for faster testing
    @given(st.data())
    @pytest.mark.property_based
    def test_hypothesis_configuration(self, data):
        """Test that hypothesis configuration works"""
        # Simple test to verify hypothesis is working
        value = data.draw(st.integers(min_value=1, max_value=100))
        assert 1 <= value <= 100