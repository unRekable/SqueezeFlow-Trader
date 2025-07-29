#!/usr/bin/env python3
"""
Trading Strategy Interface and Default Implementation
Provides base class for trading strategies and default SqueezeFlow implementation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class SignalStrength(Enum):
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3


@dataclass
class TradingSignal:
    """Standardized trading signal structure"""
    symbol: str
    signal_type: str        # 'LONG', 'SHORT', 'NONE'
    strength: SignalStrength
    confidence: float       # 0.0 to 1.0
    price: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    All strategies must implement the signal generation method
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize strategy with configuration
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_strategy()
    
    @abstractmethod
    def setup_strategy(self):
        """Setup strategy parameters and configuration"""
        pass
    
    @abstractmethod
    def generate_signal(self, symbol: str, timestamp: datetime,
                       lookback_data: Dict[str, pd.Series]) -> TradingSignal:
        """
        Generate trading signal for given symbol and data
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            lookback_data: Dict with 'price', 'spot_cvd', 'perp_cvd' series
            
        Returns:
            TradingSignal object
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.config


class SqueezeFlowStrategy(BaseStrategy):
    """
    Default SqueezeFlow Strategy Implementation
    Detects squeeze signals based on CVD divergence analysis
    """
    
    def setup_strategy(self):
        """Setup SqueezeFlow strategy parameters"""
        # Default strategy configuration
        self.strategy_config = {
            'signal_threshold': self.config.get('signal_threshold', 0.6),
            'lookback_periods': self.config.get('lookback_periods', [5, 10, 15, 30, 60, 120, 240]),
            'cvd_threshold': self.config.get('cvd_threshold', 50_000_000),
            'price_threshold': self.config.get('price_threshold', 0.5),
            'min_data_points': self.config.get('min_data_points', 240),  # Minimum data required
        }
        
        self.logger.info(f"✅ {self.get_strategy_name()} initialized with config: {self.strategy_config}")
    
    def generate_signal(self, symbol: str, timestamp: datetime,
                       lookback_data: Dict[str, pd.Series]) -> TradingSignal:
        """
        Generate SqueezeFlow trading signals based on CVD divergence
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            lookback_data: Dict with 'price', 'spot_cvd', 'perp_cvd' series
            
        Returns:
            TradingSignal object
        """
        try:
            # Extract data
            prices = lookback_data['price']
            spot_cvd = lookback_data['spot_cvd']
            perp_cvd = lookback_data['perp_cvd']
            
            # Check data availability
            min_required_length = max(self.strategy_config['lookback_periods'])
            if (len(prices) < min_required_length or 
                len(spot_cvd) < min_required_length or 
                len(perp_cvd) < min_required_length):
                return TradingSignal(
                    symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                    confidence=0.0, price=prices.iloc[-1] if len(prices) > 0 else 0.0, 
                    timestamp=timestamp
                )
            
            current_price = prices.iloc[-1]
            
            # Calculate price momentum (use available data, minimum 30 or less)
            lookback_idx = min(30, len(prices) - 1)
            if lookback_idx <= 0:
                return TradingSignal(
                    symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                    confidence=0.0, price=current_price, timestamp=timestamp
                )
                
            price_change_pct = ((current_price - prices.iloc[-lookback_idx]) / prices.iloc[-lookback_idx]) * 100
            
            # Calculate CVD changes
            spot_change = spot_cvd.iloc[-1] - spot_cvd.iloc[-lookback_idx]
            perp_change = perp_cvd.iloc[-1] - perp_cvd.iloc[-lookback_idx]
            cvd_divergence = abs(spot_change - perp_change)
            
            # Signal detection logic
            signal_type = 'NONE'
            strength = SignalStrength.NONE
            confidence = 0.0
            
            # Check thresholds
            price_threshold_met = abs(price_change_pct) > self.strategy_config['price_threshold']
            cvd_threshold_met = cvd_divergence > self.strategy_config['cvd_threshold']
            
            if price_threshold_met and cvd_threshold_met:
                # Determine signal direction
                if price_change_pct > 0 and spot_change > perp_change:
                    signal_type = 'LONG'
                elif price_change_pct < 0 and spot_change < perp_change:
                    signal_type = 'SHORT'
                
                # Calculate strength and confidence
                if signal_type != 'NONE':
                    strength_score = min(abs(price_change_pct) / 2.0, 1.0)  # Normalize to 0-1
                    cvd_score = min(cvd_divergence / (self.strategy_config['cvd_threshold'] * 2), 1.0)
                    
                    confidence = (strength_score + cvd_score) / 2
                    
                    if confidence > 0.8:
                        strength = SignalStrength.STRONG
                    elif confidence > 0.6:
                        strength = SignalStrength.MODERATE
                    else:
                        strength = SignalStrength.WEAK
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price=current_price,
                timestamp=timestamp,
                metadata={
                    'price_change_pct': price_change_pct,
                    'spot_change': spot_change,
                    'perp_change': perp_change,
                    'cvd_divergence': cvd_divergence,
                    'lookback_idx': lookback_idx,
                    'strategy': self.get_strategy_name()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                confidence=0.0, price=0.0, timestamp=timestamp
            )


class SimpleMovingAverageStrategy(BaseStrategy):
    """
    Example alternative strategy using moving averages
    Demonstrates how to create custom strategies
    """
    
    def setup_strategy(self):
        """Setup moving average strategy parameters"""
        self.strategy_config = {
            'short_ma': self.config.get('short_ma', 20),
            'long_ma': self.config.get('long_ma', 50),
            'signal_threshold': self.config.get('signal_threshold', 0.5),
            'min_data_points': self.config.get('min_data_points', 60),
        }
        
        self.logger.info(f"✅ {self.get_strategy_name()} initialized with config: {self.strategy_config}")
    
    def generate_signal(self, symbol: str, timestamp: datetime,
                       lookback_data: Dict[str, pd.Series]) -> TradingSignal:
        """
        Generate signals based on moving average crossover
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            lookback_data: Dict with 'price', 'spot_cvd', 'perp_cvd' series
            
        Returns:
            TradingSignal object
        """
        try:
            prices = lookback_data['price']
            
            # Check minimum data requirement
            min_required = max(self.strategy_config['long_ma'], self.strategy_config['min_data_points'])
            if len(prices) < min_required:
                return TradingSignal(
                    symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                    confidence=0.0, price=prices.iloc[-1] if len(prices) > 0 else 0.0, 
                    timestamp=timestamp
                )
            
            current_price = prices.iloc[-1]
            
            # Calculate moving averages
            short_ma = prices.rolling(window=self.strategy_config['short_ma']).mean()
            long_ma = prices.rolling(window=self.strategy_config['long_ma']).mean()
            
            # Current and previous MA values
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            prev_short_ma = short_ma.iloc[-2] if len(short_ma) > 1 else current_short_ma
            prev_long_ma = long_ma.iloc[-2] if len(long_ma) > 1 else current_long_ma
            
            # Signal generation
            signal_type = 'NONE'
            strength = SignalStrength.NONE
            confidence = 0.0
            
            # Crossover detection
            if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
                # Bullish crossover
                signal_type = 'LONG'
                confidence = min(abs(current_short_ma - current_long_ma) / current_long_ma, 1.0)
            elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
                # Bearish crossover
                signal_type = 'SHORT'
                confidence = min(abs(current_short_ma - current_long_ma) / current_long_ma, 1.0)
            
            # Determine strength based on confidence
            if confidence > 0.8:
                strength = SignalStrength.STRONG
            elif confidence > 0.5:
                strength = SignalStrength.MODERATE
            elif confidence > 0.2:
                strength = SignalStrength.WEAK
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price=current_price,
                timestamp=timestamp,
                metadata={
                    'short_ma': current_short_ma,
                    'long_ma': current_long_ma,
                    'ma_diff': current_short_ma - current_long_ma,
                    'strategy': self.get_strategy_name()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating MA signal for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                confidence=0.0, price=0.0, timestamp=timestamp
            )


# Import enhanced strategies
try:
    from .strategies.enhanced_squeezeflow_strategy import EnhancedSqueezeFlowStrategy
    ENHANCED_STRATEGY_AVAILABLE = True
except ImportError:
    try:
        from strategies.enhanced_squeezeflow_strategy import EnhancedSqueezeFlowStrategy
        ENHANCED_STRATEGY_AVAILABLE = True
    except ImportError:
        ENHANCED_STRATEGY_AVAILABLE = False
        EnhancedSqueezeFlowStrategy = None

# Import production enhanced strategy
try:
    from .strategies.production_enhanced_strategy import ProductionEnhancedStrategy
    PRODUCTION_STRATEGY_AVAILABLE = True
except ImportError:
    try:
        from strategies.production_enhanced_strategy import ProductionEnhancedStrategy
        PRODUCTION_STRATEGY_AVAILABLE = True
    except ImportError:
        PRODUCTION_STRATEGY_AVAILABLE = False
        ProductionEnhancedStrategy = None


# Strategy registry for dynamic loading
AVAILABLE_STRATEGIES = {
    'squeezeflow': SqueezeFlowStrategy,
    'ma_crossover': SimpleMovingAverageStrategy,
}

# Add enhanced strategy if available
if ENHANCED_STRATEGY_AVAILABLE:
    AVAILABLE_STRATEGIES['enhanced_squeezeflow_strategy'] = EnhancedSqueezeFlowStrategy

# Add production enhanced strategy if available
if PRODUCTION_STRATEGY_AVAILABLE:
    AVAILABLE_STRATEGIES['production_enhanced_strategy'] = ProductionEnhancedStrategy

# Import debug strategy
try:
    from .strategies.debug_strategy import DebugStrategy
    DEBUG_STRATEGY_AVAILABLE = True
except ImportError:
    try:
        from strategies.debug_strategy import DebugStrategy
        DEBUG_STRATEGY_AVAILABLE = True
    except ImportError:
        DEBUG_STRATEGY_AVAILABLE = False
        DebugStrategy = None

# Add debug strategy if available
if DEBUG_STRATEGY_AVAILABLE:
    AVAILABLE_STRATEGIES['debug_strategy'] = DebugStrategy

# Import simple squeeze strategy
try:
    from .strategies.simple_squeeze_strategy import SimpleSqueezeStrategy
    SIMPLE_SQUEEZE_AVAILABLE = True
except ImportError:
    try:
        from strategies.simple_squeeze_strategy import SimpleSqueezeStrategy
        SIMPLE_SQUEEZE_AVAILABLE = True
    except ImportError:
        SIMPLE_SQUEEZE_AVAILABLE = False
        SimpleSqueezeStrategy = None

# Add simple squeeze strategy if available
if SIMPLE_SQUEEZE_AVAILABLE:
    AVAILABLE_STRATEGIES['simple_squeeze_strategy'] = SimpleSqueezeStrategy

# Import working squeeze strategy
try:
    from .strategies.working_squeeze_strategy import WorkingSqueezeStrategy
    WORKING_SQUEEZE_AVAILABLE = True
except ImportError:
    try:
        from strategies.working_squeeze_strategy import WorkingSqueezeStrategy
        WORKING_SQUEEZE_AVAILABLE = True
    except ImportError:
        WORKING_SQUEEZE_AVAILABLE = False
        WorkingSqueezeStrategy = None

# Add working squeeze strategy if available
if WORKING_SQUEEZE_AVAILABLE:
    AVAILABLE_STRATEGIES['working_squeeze_strategy'] = WorkingSqueezeStrategy

# Import new comprehensive SqueezeFlow strategy
try:
    from .strategies.squeezeflow_strategy import SqueezeFlowStrategy as NewSqueezeFlowStrategy
    NEW_SQUEEZEFLOW_AVAILABLE = True
except ImportError:
    try:
        from strategies.squeezeflow_strategy import SqueezeFlowStrategy as NewSqueezeFlowStrategy
        NEW_SQUEEZEFLOW_AVAILABLE = True
    except ImportError:
        NEW_SQUEEZEFLOW_AVAILABLE = False
        NewSqueezeFlowStrategy = None

# Add new comprehensive SqueezeFlow strategy if available
if NEW_SQUEEZEFLOW_AVAILABLE:
    AVAILABLE_STRATEGIES['squeezeflow_strategy'] = NewSqueezeFlowStrategy


def load_strategy(strategy_name: str, config: Dict[str, Any] = None) -> BaseStrategy:
    """
    Load a trading strategy by name
    
    Args:
        strategy_name: Name of the strategy to load
        config: Strategy configuration
        
    Returns:
        Initialized strategy instance
        
    Raises:
        ValueError: If strategy name not found
    """
    if strategy_name not in AVAILABLE_STRATEGIES:
        available = ', '.join(AVAILABLE_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
    
    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
    return strategy_class(config)


def get_available_strategies() -> List[str]:
    """Get list of available strategy names"""
    return list(AVAILABLE_STRATEGIES.keys())


if __name__ == "__main__":
    # Test strategy loading
    print("🧠 Trading Strategy System Testing")
    print("=" * 50)
    
    # Test available strategies
    print(f"Available strategies: {get_available_strategies()}")
    
    # Test SqueezeFlow strategy
    print("\n📈 Testing SqueezeFlow Strategy:")
    strategy = load_strategy('squeezeflow', {
        'signal_threshold': 0.7,
        'cvd_threshold': 100_000_000
    })
    print(f"Strategy: {strategy.get_strategy_name()}")
    print(f"Config: {strategy.get_config()}")
    
    # Test MA strategy
    print("\n📊 Testing MA Crossover Strategy:")
    ma_strategy = load_strategy('ma_crossover', {
        'short_ma': 10,
        'long_ma': 30
    })
    print(f"Strategy: {ma_strategy.get_strategy_name()}")
    print(f"Config: {ma_strategy.get_config()}")
    
    print("\n✅ Strategy system ready for dynamic loading!")