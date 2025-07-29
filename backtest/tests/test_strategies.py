#!/usr/bin/env python3
"""
Unit Tests for Strategy System
Tests strategy interfaces, signal generation, and strategy loading
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Add backtest directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import (
    BaseStrategy, TradingSignal, SignalStrength, load_strategy, get_available_strategies
)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing"""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.signal_count = 0
    
    def setup_strategy(self):
        """Setup strategy parameters and configuration"""
        # Mock strategy setup
        pass
    
    def generate_signal(self, symbol: str, timestamp: datetime,
                       lookback_data: dict) -> TradingSignal:
        """Generate a mock signal matching the actual API"""
        self.signal_count += 1
        
        price_data = lookback_data.get('price', pd.Series())
        
        if len(price_data) < 10:
            return TradingSignal(
                symbol=symbol,
                signal_type="NONE",
                strength=SignalStrength.NONE,
                confidence=0.0,
                price=price_data.iloc[-1] if len(price_data) > 0 else 50000.0,
                timestamp=timestamp
            )
        
        # Generate alternating signals for testing
        signal_type = "LONG" if self.signal_count % 2 == 1 else "SHORT"
        strength = SignalStrength.MODERATE
        confidence = 0.7
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=price_data.iloc[-1],
            timestamp=timestamp,
            metadata={'test': True}
        )


class TestTradingSignal(unittest.TestCase):
    """Test TradingSignal data class"""
    
    def test_signal_creation(self):
        """Test trading signal creation"""
        timestamp = datetime.now(timezone.utc)
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type="LONG",
            strength=SignalStrength.STRONG,
            confidence=0.85,
            price=50000.0,
            timestamp=timestamp,
            metadata={"source": "test"}
        )
        
        self.assertEqual(signal.symbol, "BTCUSDT")
        self.assertEqual(signal.signal_type, "LONG")
        self.assertEqual(signal.strength, SignalStrength.STRONG)
        self.assertEqual(signal.confidence, 0.85)
        self.assertEqual(signal.price, 50000.0)
        self.assertEqual(signal.timestamp, timestamp)
        self.assertEqual(signal.metadata["source"], "test")
    
    def test_signal_strength_enum(self):
        """Test SignalStrength enum values"""
        self.assertEqual(SignalStrength.NONE.value, 0)
        self.assertEqual(SignalStrength.WEAK.value, 1)
        self.assertEqual(SignalStrength.MODERATE.value, 2)
        self.assertEqual(SignalStrength.STRONG.value, 3)
    
    def test_signal_with_none_metadata(self):
        """Test signal creation with None metadata"""
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type="SHORT",
            strength=SignalStrength.WEAK,
            confidence=0.3,
            price=49000.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.assertIsNone(signal.metadata)


class TestBaseStrategy(unittest.TestCase):
    """Test BaseStrategy abstract class"""
    
    def test_base_strategy_cannot_be_instantiated(self):
        """Test that BaseStrategy cannot be instantiated directly"""
        from abc import ABC
        self.assertTrue(issubclass(BaseStrategy, ABC))
        
        with self.assertRaises(TypeError):
            BaseStrategy({})
    
    def test_mock_strategy_initialization(self):
        """Test mock strategy initialization"""
        config = {"test_param": "test_value"}
        strategy = MockStrategy(config)
        
        self.assertEqual(strategy.config, config)
        self.assertEqual(strategy.signal_count, 0)
    
    def test_mock_strategy_signal_generation(self):
        """Test mock strategy signal generation"""
        strategy = MockStrategy()
        
        # Create test data using correct API format
        lookback_data = {
            'price': pd.Series([50000, 50100, 50200, 50300, 50400] * 3),  # 15 points
            'spot_cvd': pd.Series([1000, 1100, 1200, 1300, 1400] * 3),
            'perp_cvd': pd.Series([2000, 2100, 2200, 2300, 2400] * 3)
        }
        timestamp = datetime.now(timezone.utc)
        
        signal = strategy.generate_signal("BTCUSDT", timestamp, lookback_data)
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertEqual(signal.symbol, "BTCUSDT")
        self.assertIn(signal.signal_type, ["LONG", "SHORT"])
        self.assertEqual(signal.strength, SignalStrength.MODERATE)
        self.assertEqual(signal.confidence, 0.7)
        self.assertEqual(strategy.signal_count, 1)
    
    def test_insufficient_data_handling(self):
        """Test strategy behavior with insufficient data"""
        strategy = MockStrategy()
        
        # Create insufficient data (less than 10 points)
        lookback_data = {
            'price': pd.Series([50000, 50100, 50200]),
            'spot_cvd': pd.Series([1000, 1100, 1200]),
            'perp_cvd': pd.Series([2000, 2100, 2200])
        }
        timestamp = datetime.now(timezone.utc)
        
        signal = strategy.generate_signal("BTCUSDT", timestamp, lookback_data)
        
        self.assertEqual(signal.signal_type, "NONE")
        self.assertEqual(signal.strength, SignalStrength.NONE)
        self.assertEqual(signal.confidence, 0.0)
    
    def test_strategy_with_metadata(self):
        """Test strategy signal generation with metadata"""
        strategy = MockStrategy()
        
        lookback_data = {
            'price': pd.Series([50000] * 15),
            'spot_cvd': pd.Series([1000] * 15),
            'perp_cvd': pd.Series([2000] * 15)
        }
        timestamp = datetime.now(timezone.utc)
        
        signal = strategy.generate_signal("BTCUSDT", timestamp, lookback_data)
        
        # Mock strategy sets test=True in metadata
        self.assertEqual(signal.metadata['test'], True)


class TestStrategyLoading(unittest.TestCase):
    """Test strategy loading and discovery system"""
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies"""
        strategies = get_available_strategies()
        
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)  # Should have at least some strategies
        
        # Should have basic strategies
        self.assertIn('squeezeflow', strategies)
        self.assertIn('ma_crossover', strategies)
    
    def test_load_strategy_success(self):
        """Test successful strategy loading"""
        # Test loading basic strategy
        strategy = load_strategy('squeezeflow')
        
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, BaseStrategy)
        self.assertEqual(strategy.get_strategy_name(), 'SqueezeFlowStrategy')
    
    def test_load_strategy_failure(self):
        """Test strategy loading failure handling"""
        # Try to load a non-existent strategy
        with self.assertRaises(ValueError):
            load_strategy('non_existent_strategy')
    
    def test_load_strategy_with_config(self):
        """Test strategy loading with configuration"""
        config = {"signal_threshold": 0.8}
        
        strategy = load_strategy('squeezeflow', config)
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.get_config(), config)


class TestStrategyValidation(unittest.TestCase):
    """Test strategy validation and error handling"""
    
    def test_empty_data_handling(self):
        """Test strategy behavior with empty data"""
        strategy = MockStrategy()
        
        lookback_data = {
            'price': pd.Series([], dtype=float),
            'spot_cvd': pd.Series([], dtype=float),
            'perp_cvd': pd.Series([], dtype=float)
        }
        timestamp = datetime.now(timezone.utc)
        
        # Should not raise exception
        signal = strategy.generate_signal("BTCUSDT", timestamp, lookback_data)
        
        self.assertEqual(signal.signal_type, "NONE")
        self.assertEqual(signal.strength, SignalStrength.NONE)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)