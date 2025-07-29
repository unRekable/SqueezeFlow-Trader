#!/usr/bin/env python3
"""
Portfolio Management - Position and Risk Management for Backtest Engine
Extracted from monolithic engine.py for clean modular architecture
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class Position:
    """Trading position with complete lifecycle tracking"""
    id: str
    symbol: str
    position_type: PositionType
    entry_time: datetime
    entry_price: float
    size: float
    status: PositionStatus = PositionStatus.OPEN
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    fees_paid: float = 0.0
    exit_reason: str = ""
    
    def duration_hours(self, current_time: datetime = None) -> float:
        """Position duration in hours"""
        end_time = self.exit_time or current_time or datetime.now(timezone.utc)
        
        # Handle timezone-aware/naive datetime mixing
        if self.entry_time.tzinfo is not None and end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=self.entry_time.tzinfo)
        elif self.entry_time.tzinfo is None and end_time.tzinfo is not None:
            entry_time = self.entry_time.replace(tzinfo=end_time.tzinfo)
            return (end_time - entry_time).total_seconds() / 3600
        
        return (end_time - self.entry_time).total_seconds() / 3600
    
    @property
    def pnl_absolute(self) -> float:
        """Absolute P&L in base currency"""
        if self.status != PositionStatus.CLOSED or self.exit_price is None:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.exit_price - self.entry_price) * self.size - self.fees_paid
        else:
            return (self.entry_price - self.exit_price) * self.size - self.fees_paid
    
    @property
    def pnl_percentage(self) -> float:
        """P&L as percentage of invested capital"""
        if self.status != PositionStatus.CLOSED:
            return 0.0
        invested_capital = self.entry_price * self.size
        return (self.pnl_absolute / invested_capital) * 100 if invested_capital > 0 else 0.0


@dataclass
class RiskLimits:
    """Risk management configuration"""
    max_position_size: float = 0.02        # 2% max per position
    max_total_exposure: float = 0.1        # 10% total exposure
    max_open_positions: int = 2            # Max concurrent positions
    max_daily_loss: float = 0.05          # 5% max daily loss
    max_drawdown: float = 0.15            # 15% max drawdown
    min_position_size: float = 0.001      # 0.1% minimum position
    stop_loss_percentage: float = 0.025   # 2.5% stop loss
    take_profit_percentage: float = 0.04  # 4% take profit


class PortfolioManager:
    """
    Comprehensive portfolio and risk management system
    Handles position lifecycle, risk controls, and performance tracking
    """
    
    def __init__(self, initial_balance: float, risk_limits: RiskLimits = None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_limits = risk_limits or RiskLimits()
        
        # Position tracking
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.position_counter = 0
        
        # Performance tracking
        self.daily_pnl: Dict[str, float] = {}
        self.peak_balance = initial_balance
        self.max_drawdown_reached = 0.0
        
        self.logger = logging.getLogger('PortfolioManager')
        
    def can_open_position(self, symbol: str, position_size_percentage: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened based on risk limits
        
        Returns:
            (can_open, reason)
        """
        # Check if position already exists for this symbol (NO DCA - only 1 position per symbol)
        for position in self.open_positions.values():
            if position.symbol == symbol:
                return False, f"Position already exists for {symbol}: {position.id}"
        
        # Check max open positions
        if len(self.open_positions) >= self.risk_limits.max_open_positions:
            return False, f"Max open positions reached ({self.risk_limits.max_open_positions})"
        
        # Check position size limits
        if position_size_percentage > self.risk_limits.max_position_size:
            return False, f"Position size too large: {position_size_percentage:.3f} > {self.risk_limits.max_position_size}"
        
        if position_size_percentage < self.risk_limits.min_position_size:
            return False, f"Position size too small: {position_size_percentage:.3f} < {self.risk_limits.min_position_size}"
        
        # Check total exposure
        current_exposure = self.get_total_exposure()
        if current_exposure + position_size_percentage > self.risk_limits.max_total_exposure:
            return False, f"Total exposure limit exceeded: {current_exposure + position_size_percentage:.3f} > {self.risk_limits.max_total_exposure}"
        
        # Check daily loss limit
        today = datetime.utcnow().date().isoformat()
        daily_loss = abs(self.daily_pnl.get(today, 0.0)) / self.initial_balance
        if daily_loss > self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit reached: {daily_loss:.3f} > {self.risk_limits.max_daily_loss}"
        
        # Check drawdown limit
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.risk_limits.max_drawdown:
            return False, f"Drawdown limit reached: {current_drawdown:.3f} > {self.risk_limits.max_drawdown}"
        
        return True, "Position allowed"
    
    def open_position(self, symbol: str, position_type: PositionType, entry_price: float,
                     position_size_percentage: float, timestamp: datetime,
                     stop_loss: float = None, take_profit: float = None) -> Optional[Position]:
        """
        Open a new trading position
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            position_type: LONG or SHORT
            entry_price: Entry price
            position_size_percentage: Position size as percentage of balance
            timestamp: Entry timestamp
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Position object if successful, None if rejected
        """
        # Check risk limits
        can_open, reason = self.can_open_position(symbol, position_size_percentage)
        if not can_open:
            self.logger.warning(f"Position rejected for {symbol}: {reason}")
            return None
        
        # Calculate position size in units (with division by zero protection)
        if entry_price <= 0 or pd.isna(entry_price):
            self.logger.error(f"Cannot open position for {symbol}: invalid entry_price {entry_price}")
            return None
            
        position_value = self.current_balance * position_size_percentage
        position_size = position_value / entry_price
        
        # Deduct invested capital from balance when opening position
        self.current_balance -= position_value
        
        # Generate position ID
        self.position_counter += 1
        position_id = f"{symbol}_{position_type.value}_{self.position_counter:04d}"
        
        # Set default stop loss and take profit if not provided
        if stop_loss is None and position_type == PositionType.LONG:
            stop_loss = entry_price * (1 - self.risk_limits.stop_loss_percentage)
        elif stop_loss is None and position_type == PositionType.SHORT:
            stop_loss = entry_price * (1 + self.risk_limits.stop_loss_percentage)
            
        if take_profit is None and position_type == PositionType.LONG:
            take_profit = entry_price * (1 + self.risk_limits.take_profit_percentage)
        elif take_profit is None and position_type == PositionType.SHORT:
            take_profit = entry_price * (1 - self.risk_limits.take_profit_percentage)
        
        # Create position
        position = Position(
            id=position_id,
            symbol=symbol,
            position_type=position_type,
            entry_time=timestamp,
            entry_price=entry_price,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Store position
        self.open_positions[position_id] = position
        
        self.logger.info(f"Opened {position_type.value} position: {position_id} at {entry_price:.2f} "
                        f"(size: {position_size:.6f}, value: ${position_value:.2f})")
        
        return position
    
    def close_position(self, position_id: str, exit_price: float, timestamp: datetime,
                      exit_reason: str = "Manual close", trading_fees: float = 0.0) -> Optional[Position]:
        """
        Close an existing position
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            timestamp: Exit timestamp
            exit_reason: Reason for closing
            trading_fees: Total fees paid for this trade
            
        Returns:
            Closed position or None if not found
        """
        if position_id not in self.open_positions:
            self.logger.error(f"Position {position_id} not found")
            return None
        
        position = self.open_positions[position_id]
        
        # Update position
        position.exit_time = timestamp
        position.exit_price = exit_price
        position.status = PositionStatus.CLOSED
        position.exit_reason = exit_reason
        position.fees_paid = trading_fees
        
        # Calculate P&L and update balance
        pnl = position.pnl_absolute  # Already includes fees deduction
        # Only add the P&L to balance (position_value was already deducted at entry)
        self.current_balance += pnl  # Add P&L only (position value was already deducted)
        
        # Update daily P&L tracking
        day_key = timestamp.date().isoformat()
        self.daily_pnl[day_key] = self.daily_pnl.get(day_key, 0.0) + pnl
        
        # Update peak balance and drawdown tracking
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.max_drawdown_reached:
            self.max_drawdown_reached = current_drawdown
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[position_id]
        
        self.logger.info(f"Closed position {position_id}: {exit_reason} at {exit_price:.2f} "
                        f"(P&L: ${pnl:.2f}, {position.pnl_percentage:.2f}%)")
        
        return position
    
    def update_position_metrics(self, current_prices: Dict[str, float], timestamp: datetime):
        """Update max drawdown and profit for all open positions"""
        for position in self.open_positions.values():
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                
                # Calculate unrealized P&L
                if position.position_type == PositionType.LONG:
                    unrealized_pnl = (current_price - position.entry_price) * position.size
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.size
                
                # Update max profit and drawdown
                if unrealized_pnl > position.max_profit:
                    position.max_profit = unrealized_pnl
                
                if unrealized_pnl < position.max_drawdown:
                    position.max_drawdown = unrealized_pnl
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float], 
                                   timestamp: datetime) -> List[str]:
        """
        Check all open positions for stop loss and take profit triggers
        
        Returns:
            List of position IDs that should be closed
        """
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            if position.symbol not in current_prices:
                continue
                
            current_price = current_prices[position.symbol]
            should_close = False
            exit_reason = ""
            
            # Check stop loss
            if position.stop_loss is not None:
                if position.position_type == PositionType.LONG and current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif position.position_type == PositionType.SHORT and current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
            
            # Check take profit
            if not should_close and position.take_profit is not None:
                if position.position_type == PositionType.LONG and current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
                elif position.position_type == PositionType.SHORT and current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            
            if should_close:
                positions_to_close.append(position_id)
                self.logger.info(f"Position {position_id} triggered {exit_reason} at {current_price:.2f}")
        
        return positions_to_close
    
    def get_total_exposure(self) -> float:
        """Calculate total exposure as percentage of balance"""
        total_exposure = 0.0
        for position in self.open_positions.values():
            position_value = position.entry_price * position.size
            exposure_pct = position_value / self.current_balance
            total_exposure += exposure_pct
        return total_exposure
    
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win_pct': 0.0,
                'largest_loss_pct': 0.0
            }
        
        # Basic metrics
        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p.pnl_absolute > 0]
        losing_trades = [p for p in self.closed_positions if p.pnl_absolute < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0.0
        total_pnl = sum(p.pnl_absolute for p in self.closed_positions)
        
        # Trade returns for distribution plotting
        trade_returns = [p.pnl_percentage * 100 for p in self.closed_positions]  # Convert to percentages
        
        # Protect against invalid balance calculations
        if np.isnan(self.current_balance) or np.isinf(self.current_balance) or self.current_balance <= 0:
            self.logger.warning(f"Invalid current balance: {self.current_balance}, resetting to initial balance")
            self.current_balance = max(self.initial_balance + total_pnl, 0.01)  # Ensure positive balance
            
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        # Win/Loss statistics
        avg_win = np.mean([p.pnl_absolute for p in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([p.pnl_absolute for p in losing_trades]) if losing_trades else 0.0
        
        # FIXED: Use percentage returns for largest_win/loss, not absolute P&L
        largest_win_pct = max([p.pnl_percentage for p in winning_trades]) if winning_trades else 0.0
        largest_loss_pct = min([p.pnl_percentage for p in losing_trades]) if losing_trades else 0.0
        
        # Profit factor - handle edge cases properly
        gross_profit = sum(p.pnl_absolute for p in winning_trades)
        gross_loss = abs(sum(p.pnl_absolute for p in losing_trades))
        
        if gross_loss > 0 and gross_profit > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0:
            profit_factor = float('inf')  # All winning trades
        else:
            profit_factor = 0.0  # No profit or all losing trades
        
        # Sharpe ratio (simplified - assumes daily returns)
        daily_returns = list(self.daily_pnl.values())
        if len(daily_returns) > 1:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown_reached * 100,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win_pct': largest_win_pct,
            'largest_loss_pct': largest_loss_pct,
            'avg_trade_duration': np.mean([p.duration_hours() for p in self.closed_positions]),
            'total_fees': sum(p.fees_paid for p in self.closed_positions),
            'trade_returns': trade_returns  # Add trade returns for plotting
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Generate equity curve data for plotting"""
        if not self.closed_positions:
            return pd.DataFrame({'timestamp': [datetime.utcnow()], 'balance': [self.initial_balance], 'trade_pnl': [0.0]})
        
        # Sort positions by exit time
        sorted_positions = sorted(self.closed_positions, key=lambda p: p.exit_time)
        
        equity_data = []
        running_balance = self.initial_balance
        
        # FIXED: Add initial balance point at the start
        first_trade_time = sorted_positions[0].entry_time if sorted_positions else datetime.utcnow()
        equity_data.append({
            'timestamp': first_trade_time,
            'balance': self.initial_balance,
            'trade_pnl': 0.0,
            'position_id': 'INITIAL'
        })
        
        for position in sorted_positions:
            running_balance += position.pnl_absolute
            equity_data.append({
                'timestamp': position.exit_time,
                'balance': running_balance,
                'trade_pnl': position.pnl_absolute,
                'position_id': position.id
            })
        
        return pd.DataFrame(equity_data)