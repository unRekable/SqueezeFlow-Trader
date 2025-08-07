"""
Base Strategy Interface

All strategies must implement the BaseStrategy interface.
This ensures compatibility with both the backtest engine and live trading services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseStrategy(ABC):
    """
    Base strategy interface - all strategies must implement process()
    
    This interface ensures that strategies can be used in both:
    - Backtest mode: Engine provides pre-calculated CVD data
    - Live mode: Strategy runner calculates CVD in real-time
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name for identification
        """
        self.name = name
    
    @abstractmethod
    def process(self, dataset: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data and return trading orders.
        
        This is the ONLY required method that strategies must implement.
        The strategy receives complete market data and returns executable orders.
        
        Args:
            dataset: Complete dataset from backtest engine or live service with:
                - 'symbol': str - Trading symbol (e.g., 'BTCUSDT')
                - 'ohlcv': pd.DataFrame - Price data with columns: open, high, low, close, volume
                - 'spot_cvd': pd.Series - Pre-calculated spot CVD (cumulative volume delta)
                - 'futures_cvd': pd.Series - Pre-calculated futures CVD
                - 'cvd_divergence': pd.Series - CVD divergence (futures - spot)
                - 'metadata': Dict - Additional market information
                
            portfolio_state: Current portfolio state with:
                - 'total_value': float - Total portfolio value
                - 'cash': float - Available cash
                - 'positions': List[Dict] - Open positions
                - 'equity': float - Total equity including positions
                
        Returns:
            Dict with required keys:
                - 'orders': List[Dict] - List of orders to execute
                    Each order must contain:
                    - 'symbol': str - Trading symbol
                    - 'side': str - 'BUY' or 'SELL'
                    - 'quantity': float - Base quantity (before leverage)
                    - 'price': float - Entry price
                    - 'timestamp': datetime - Order timestamp
                    - 'leverage': float (optional) - Position leverage
                    - 'stop_loss': float (optional) - Stop loss price
                    - 'take_profit': float (optional) - Take profit price
                    - 'signal_type': str (optional) - Signal description
                    - 'confidence': float (optional) - Signal confidence 0-1
                    
                - 'phase_results': Dict (optional) - Phase analysis results
                - 'metadata': Dict (optional) - Strategy metadata
                - 'error': str (optional) - Error message if processing failed
        """
        pass
    
    def calculate_position_size(self, balance: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk management.
        
        This is a helper method that strategies can use but are not required to.
        Strategies are free to implement their own position sizing logic.
        
        Args:
            balance: Current account balance
            risk_per_trade: Risk percentage per trade (default: 2%)
            
        Returns:
            Position size in quote currency
        """
        return balance * risk_per_trade