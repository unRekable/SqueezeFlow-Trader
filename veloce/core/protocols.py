"""Protocol interfaces for Veloce system - all contracts defined here"""
from typing import Protocol, Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Signal:
    """Trading signal data structure"""
    id: str
    symbol: str
    timestamp: datetime
    action: Any  # SignalAction enum
    side: Any  # OrderSide enum
    order_type: Any  # OrderType enum
    price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # 'LONG', 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    id: Optional[str] = None
    current_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataProvider(Protocol):
    """Protocol for data access - ALL data must go through this interface"""
    
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str,
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get OHLCV data for symbol and timeframe"""
        ...
    
    def get_cvd(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get CVD (Cumulative Volume Delta) data"""
        ...
    
    def get_oi(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get Open Interest data"""
        ...
    
    def get_multi_timeframe(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get data for all configured timeframes"""
        ...
    
    def clear_cache(self) -> None:
        """Clear data cache"""
        ...


class Strategy(Protocol):
    """Protocol for trading strategies"""
    
    def analyze(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Analyze market at given timestamp"""
        ...
    
    def generate_signal(
        self,
        analysis: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate trading signal from analysis"""
        ...
    
    def validate_signal(
        self,
        signal: Signal
    ) -> Tuple[bool, str]:
        """Validate signal before execution"""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        ...


class Dashboard(Protocol):
    """Protocol for visualization/dashboard"""
    
    def create_chart(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> str:
        """Create chart and return chart ID"""
        ...
    
    def update_chart(
        self,
        chart_id: str,
        data: pd.DataFrame
    ) -> None:
        """Update existing chart with new data"""
        ...
    
    def generate_dashboard(
        self,
        analysis: Dict[str, Any],
        trades: List[Dict[str, Any]]
    ) -> str:
        """Generate complete dashboard HTML"""
        ...
    
    def export(
        self,
        format: str = 'html'
    ) -> bytes:
        """Export dashboard in specified format"""
        ...


class Executor(Protocol):
    """Protocol for order execution"""
    
    def execute_signal(
        self,
        signal: Signal
    ) -> Dict[str, Any]:
        """Execute trading signal"""
        ...
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        ...
    
    def close_position(
        self,
        position_id: str,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """Close specific position"""
        ...
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        ...
    
    def get_order_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history"""
        ...


class Monitor(Protocol):
    """Protocol for system monitoring"""
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health"""
        ...
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        ...
    
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Log system event"""
        ...
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        ...