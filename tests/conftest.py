"""
Pytest Configuration for SqueezeFlow Trader Tests
Comprehensive fixtures and test configuration
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_influxdb():
    """Mock InfluxDB client for testing"""
    mock_client = MagicMock()
    
    # Mock query results with realistic data structure
    mock_result = MagicMock()
    mock_result.raw = {
        'series': [{
            'name': 'trades_5m',
            'columns': ['time', 'open', 'high', 'low', 'close', 'volume', 'vbuy', 'vsell'],
            'values': [
                ['2024-08-01T00:00:00Z', 50000, 50500, 49800, 50200, 1000, 600, 400],
                ['2024-08-01T00:05:00Z', 50200, 50700, 49900, 50400, 1200, 700, 500],
                ['2024-08-01T00:10:00Z', 50400, 50900, 50000, 50600, 1100, 650, 450]
            ]
        }]
    }
    mock_client.query.return_value = mock_result
    return mock_client


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-08-01', periods=100, freq='5min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 50000
    
    data = {
        'time': dates,
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': np.random.uniform(800, 1500, 100)
    }
    
    # Generate realistic price movements
    close_price = base_price
    for _ in range(100):
        # Random walk with slight upward bias
        change = np.random.normal(0, 100)  # $100 std dev
        close_price += change
        
        open_price = close_price + np.random.normal(0, 50)
        high_price = max(open_price, close_price) + np.random.uniform(0, 200)
        low_price = min(open_price, close_price) - np.random.uniform(0, 200)
        
        data['open'].append(open_price)
        data['high'].append(high_price)
        data['low'].append(low_price)
        data['close'].append(close_price)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_volume_data():
    """Sample volume data with buy/sell split"""
    dates = pd.date_range(start='2024-08-01', periods=100, freq='5min')
    
    np.random.seed(42)
    
    data = {
        'time': dates,
        'total_volume': np.random.uniform(1000, 3000, 100),
        'total_vbuy': [],
        'total_vsell': [],
        'total_cbuy': [],
        'total_csell': [],
        'total_lbuy': [],
        'total_lsell': []
    }
    
    # Generate buy/sell splits
    for total_vol in data['total_volume']:
        # Random buy/sell ratio (50-80% buy typical)
        buy_ratio = np.random.uniform(0.4, 0.8)
        vbuy = total_vol * buy_ratio
        vsell = total_vol * (1 - buy_ratio)
        
        data['total_vbuy'].append(vbuy)
        data['total_vsell'].append(vsell)
        
        # Distribute across market types (approximate)
        data['total_cbuy'].append(vbuy * 0.6)  # 60% retail
        data['total_csell'].append(vsell * 0.6)
        data['total_lbuy'].append(vbuy * 0.4)   # 40% large orders
        data['total_lsell'].append(vsell * 0.4)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_cvd_data():
    """Sample CVD data for testing"""
    dates = pd.date_range(start='2024-08-01', periods=100, freq='5min')
    
    np.random.seed(42)
    
    # Generate CVD with some trend
    volume_deltas = np.random.normal(100, 500, 100)  # Random deltas
    spot_cvd = np.cumsum(volume_deltas)  # Cumulative sum
    
    # Futures CVD with some divergence
    futures_deltas = volume_deltas + np.random.normal(0, 200, 100)
    futures_cvd = np.cumsum(futures_deltas)
    
    return {
        'spot_cvd': pd.Series(spot_cvd, index=dates),
        'futures_cvd': pd.Series(futures_cvd, index=dates),
        'cvd_divergence': pd.Series(spot_cvd - futures_cvd, index=dates)
    }


@pytest.fixture
def sample_dataset(sample_ohlcv_data, sample_volume_data, sample_cvd_data):
    """Complete dataset for strategy testing"""
    return {
        'symbol': 'BTCUSDT',
        'timeframe': '5m',
        'start_time': datetime(2024, 8, 1),
        'end_time': datetime(2024, 8, 2),
        'ohlcv': sample_ohlcv_data,
        'spot_volume': sample_volume_data.copy(),
        'futures_volume': sample_volume_data.copy(),
        'spot_cvd': sample_cvd_data['spot_cvd'],
        'futures_cvd': sample_cvd_data['futures_cvd'],
        'cvd_divergence': sample_cvd_data['cvd_divergence'],
        'markets': {
            'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD'],
            'perp': ['BINANCE_FUTURES:btcusdt', 'BYBIT:BTCUSDT']
        },
        'metadata': {
            'spot_markets_count': 2,
            'futures_markets_count': 2,
            'data_points': 100
        }
    }


@pytest.fixture
def sample_portfolio_state():
    """Sample portfolio state for testing"""
    return {
        'total_value': 10000.0,
        'available_balance': 8000.0,
        'positions': [],
        'open_orders': [],
        'pnl': 0.0,
        'available_leverage': 3.0
    }


@pytest.fixture
def sample_portfolio_with_position():
    """Portfolio state with existing position"""
    return {
        'total_value': 10000.0,
        'available_balance': 7000.0,
        'positions': [{
            'id': 'pos_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.02,
            'entry_price': 50000.0,
            'current_price': 50500.0,
            'pnl': 10.0,
            'timestamp': datetime.now()
        }],
        'open_orders': [],
        'pnl': 10.0,
        'available_leverage': 3.0
    }


@pytest.fixture
def mock_data_pipeline():
    """Mock data pipeline for testing"""
    mock_pipeline = MagicMock()
    
    # Mock discovery methods
    mock_pipeline.discover_available_symbols.return_value = ['BTC', 'ETH']
    mock_pipeline.discover_markets_for_symbol.return_value = {
        'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD'],
        'perp': ['BINANCE_FUTURES:btcusdt', 'BYBIT:BTCUSDT']
    }
    
    # Mock data quality validation
    mock_pipeline.validate_data_quality.return_value = {
        'has_price_data': True,
        'has_spot_volume': True,
        'has_futures_volume': True,
        'has_spot_cvd': True,
        'has_futures_cvd': True,
        'has_cvd_divergence': True,
        'sufficient_data_points': True,
        'multiple_spot_markets': True,
        'multiple_futures_markets': True,
        'overall_quality': True
    }
    
    return mock_pipeline


@pytest.fixture
def squeeze_flow_config():
    """SqueezeFlow strategy configuration for testing"""
    from strategies.squeezeflow.config import SqueezeFlowConfig
    return SqueezeFlowConfig()


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    return MagicMock()


# Test scenarios for different market conditions
@pytest.fixture
def long_squeeze_scenario(sample_dataset):
    """Market data representing a long squeeze setup"""
    dataset = sample_dataset.copy()
    
    # Modify CVD to show long squeeze pattern
    # Spot selling while futures relatively stable
    spot_pressure = np.linspace(0, -5000, 100)  # Increasing spot selling
    futures_stable = np.linspace(0, -1000, 100)  # Less futures selling
    
    dataset['spot_cvd'] = pd.Series(spot_pressure, index=dataset['spot_cvd'].index)
    dataset['futures_cvd'] = pd.Series(futures_stable, index=dataset['futures_cvd'].index)
    dataset['cvd_divergence'] = dataset['spot_cvd'] - dataset['futures_cvd']
    
    return dataset


@pytest.fixture
def short_squeeze_scenario(sample_dataset):
    """Market data representing a short squeeze setup"""
    dataset = sample_dataset.copy()
    
    # Modify CVD to show short squeeze pattern
    # Futures selling while spot relatively stable
    spot_stable = np.linspace(0, -1000, 100)    # Less spot selling
    futures_pressure = np.linspace(0, -8000, 100)  # Heavy futures selling
    
    dataset['spot_cvd'] = pd.Series(spot_stable, index=dataset['spot_cvd'].index)
    dataset['futures_cvd'] = pd.Series(futures_pressure, index=dataset['futures_cvd'].index)
    dataset['cvd_divergence'] = dataset['spot_cvd'] - dataset['futures_cvd']
    
    return dataset


@pytest.fixture
def neutral_market_scenario(sample_dataset):
    """Market data with no clear squeeze pattern"""
    dataset = sample_dataset.copy()
    
    # Similar CVD patterns (no divergence)
    base_cvd = np.cumsum(np.random.normal(0, 100, 100))
    noise = np.random.normal(0, 50, 100)
    
    dataset['spot_cvd'] = pd.Series(base_cvd + noise, index=dataset['spot_cvd'].index)
    dataset['futures_cvd'] = pd.Series(base_cvd - noise, index=dataset['futures_cvd'].index)
    dataset['cvd_divergence'] = dataset['spot_cvd'] - dataset['futures_cvd']
    
    return dataset


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "property_based: mark test as property-based test"
    )


# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_mock_order(symbol='BTCUSDT', side='BUY', quantity=0.001, price=50000):
        """Create a mock order for testing"""
        return {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'signal_type': 'TEST',
            'confidence': 0.8
        }
    
    @staticmethod
    def assert_valid_order(order):
        """Assert that an order has valid structure"""
        required_fields = ['symbol', 'side', 'quantity', 'price']
        for field in required_fields:
            assert field in order, f"Missing required field: {field}"
        
        assert order['side'] in ['BUY', 'SELL'], f"Invalid side: {order['side']}"
        assert order['quantity'] > 0, f"Invalid quantity: {order['quantity']}"
        assert order['price'] > 0, f"Invalid price: {order['price']}"
    
    @staticmethod
    def assert_valid_strategy_result(result):
        """Assert that strategy result has valid structure"""
        assert 'orders' in result, "Missing orders in strategy result"
        assert 'phase_results' in result, "Missing phase_results in strategy result"
        assert isinstance(result['orders'], list), "Orders must be a list"
        assert isinstance(result['phase_results'], dict), "Phase results must be a dict"


@pytest.fixture
def test_utils():
    """Test utilities fixture"""
    return TestUtils