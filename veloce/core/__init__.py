"""Core module for Veloce system"""

# Configuration
from veloce.core.config import VeloceConfig, CONFIG

# Protocols
from veloce.core.protocols import (
    DataProvider,
    Strategy,
    Dashboard,
    Executor,
    Monitor,
    Signal,
    Position
)

# Types
from veloce.core.types import (
    OrderSide,
    OrderType,
    PositionSide,
    SignalAction,
    StrategyMode,
    TimeFrame,
    PhaseResult,
    IndicatorData,
    MarketData,
    CVDData,
    OIData,
    TradeResult,
    PerformanceMetrics,
    SystemHealth
)

# Constants
from veloce.core.constants import (
    VERSION,
    SYSTEM_NAME,
    SUPPORTED_EXCHANGES,
    TIMEFRAME_SECONDS,
    PHASE_NAMES,
    ERROR_MESSAGES
)

# Exceptions
from veloce.core.exceptions import (
    VeloceException,
    ConfigurationError,
    DataProviderError,
    ConnectionError,
    NoDataError,
    StrategyError,
    SignalValidationError,
    IndicatorError,
    ExecutionError,
    InsufficientBalanceError,
    PositionError,
    BacktestError,
    DashboardError,
    APIError,
    AuthenticationError,
    RateLimitError
)

__all__ = [
    # Config
    'VeloceConfig',
    'CONFIG',
    
    # Protocols
    'DataProvider',
    'Strategy',
    'Dashboard',
    'Executor',
    'Monitor',
    'Signal',
    'Position',
    
    # Types
    'OrderSide',
    'OrderType',
    'PositionSide',
    'SignalAction',
    'StrategyMode',
    'TimeFrame',
    'PhaseResult',
    'IndicatorData',
    'MarketData',
    'CVDData',
    'OIData',
    'TradeResult',
    'PerformanceMetrics',
    'SystemHealth',
    
    # Constants
    'VERSION',
    'SYSTEM_NAME',
    'SUPPORTED_EXCHANGES',
    'TIMEFRAME_SECONDS',
    'PHASE_NAMES',
    'ERROR_MESSAGES',
    
    # Exceptions
    'VeloceException',
    'ConfigurationError',
    'DataProviderError',
    'ConnectionError',
    'NoDataError',
    'StrategyError',
    'SignalValidationError',
    'IndicatorError',
    'ExecutionError',
    'InsufficientBalanceError',
    'PositionError',
    'BacktestError',
    'DashboardError',
    'APIError',
    'AuthenticationError',
    'RateLimitError'
]