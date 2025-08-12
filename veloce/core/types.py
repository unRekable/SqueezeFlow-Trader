"""Shared type definitions for Veloce system"""
from typing import TypedDict, Literal, Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    
    
class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class PositionSide(str, Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"


class SignalAction(str, Enum):
    """Signal action enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE = "CLOSE"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class StrategyMode(str, Enum):
    """Strategy mode enumeration"""
    PRODUCTION = "production"
    PAPER = "paper"
    BACKTEST = "backtest"
    OPTIMIZE = "optimize"


class TimeFrame(str, Enum):
    """Supported timeframes"""
    ONE_SECOND = "1s"
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


class PhaseResult(TypedDict):
    """Result from a strategy phase"""
    passed: bool
    score: float
    reason: str
    data: Dict[str, Any]


class IndicatorData(TypedDict):
    """Indicator calculation result"""
    value: float
    signal: Optional[str]
    metadata: Dict[str, Any]


class MarketData(TypedDict):
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: Optional[float]
    sell_volume: Optional[float]


class CVDData(TypedDict):
    """CVD data structure"""
    spot_cvd: float
    perp_cvd: float
    spot_cumulative: float
    perp_cumulative: float
    divergence: float


class OIData(TypedDict):
    """Open Interest data structure"""
    value: float
    change: float
    exchange: str


class TradeResult(TypedDict):
    """Trade execution result"""
    id: str
    symbol: str
    side: str
    price: float
    quantity: float
    timestamp: datetime
    pnl: Optional[float]
    pnl_percent: Optional[float]


class PerformanceMetrics(TypedDict):
    """Performance metrics structure"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float


class SystemHealth(TypedDict):
    """System health status"""
    status: Literal["healthy", "degraded", "unhealthy"]
    influx_connected: bool
    redis_connected: bool
    api_running: bool
    strategy_running: bool
    last_signal: Optional[datetime]
    error_count: int
    warnings: List[str]