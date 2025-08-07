#!/usr/bin/env python3
"""
Portfolio Management - Position tracking and risk management
Clean state management with realistic fee calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Position:
    """Individual position tracking with CVD baseline support"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    trade_id: Optional[int] = None  # Added for CVD baseline tracking
    signal_id: Optional[str] = None  # Added for signal correlation
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fees_paid: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    closed: bool = False  # Track if position is closed


class Portfolio:
    """Portfolio management with risk controls"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.transaction_history: List[Dict] = []
        
        # Risk parameters
        self.max_position_size = 0.1  # 10% max per position
        self.max_total_exposure = 0.5  # 50% max total exposure
        self.trading_fee = 0.001  # 0.1% trading fee
        
    def get_state(self) -> Dict:
        """Get current portfolio state"""
        total_value = self.cash_balance
        total_exposure = 0.0
        unrealized_pnl = 0.0
        
        # Calculate current position values
        for position in self.positions.values():
            position_value = position.quantity * position.current_price
            total_value += position_value
            total_exposure += position_value
            unrealized_pnl += position.unrealized_pnl
        
        realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        total_pnl = realized_pnl + unrealized_pnl
        
        return {
            'initial_balance': self.initial_balance,
            'cash_balance': self.cash_balance,
            'total_value': total_value,
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / total_value if total_value > 0 else 0,
            'open_positions': len(self.positions),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'total_return': (total_value - self.initial_balance) / self.initial_balance * 100,
            'positions': [self._position_to_dict(v) for v in self.positions.values()]
        }
    
    def _position_to_dict(self, position: Position) -> Dict:
        """Convert position to dictionary"""
        return {
            'id': position.trade_id or f"{position.symbol}_{position.side}",  # Use trade_id as id
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'entry_time': position.entry_time,
            'trade_id': position.trade_id,
            'signal_id': position.signal_id,
            'unrealized_pnl': position.unrealized_pnl,
            'fees_paid': position.fees_paid
        }
    
    def can_open_position(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if position can be opened within risk limits"""
        position_value = quantity * price
        
        # Check maximum position size
        if position_value > self.initial_balance * self.max_position_size:
            return False
        
        # Check total exposure
        current_exposure = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        if (current_exposure + position_value) > self.initial_balance * self.max_total_exposure:
            return False
        
        # Check available cash (including fees)
        required_cash = position_value * (1 + self.trading_fee)
        if required_cash > self.cash_balance:
            return False
        
        return True
    
    def open_long_position(self, symbol: str, quantity: float, price: float, 
                          timestamp: datetime, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None, trade_id: Optional[int] = None,
                          signal_id: Optional[str] = None) -> bool:
        """Open a long position"""
        
        if not self.can_open_position(symbol, quantity, price):
            return False
        
        # Calculate fees
        position_value = quantity * price
        fees = position_value * self.trading_fee
        total_cost = position_value + fees
        
        # Update cash balance
        self.cash_balance -= total_cost
        
        # Create position
        position = Position(
            symbol=symbol,
            side='LONG',
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=timestamp,
            trade_id=trade_id,
            signal_id=signal_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees_paid=fees
        )
        
        # Store position
        position_key = f"{symbol}_LONG_{timestamp.isoformat()}"
        self.positions[position_key] = position
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': timestamp,
            'type': 'OPEN_LONG',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'fees': fees,
            'cash_after': self.cash_balance
        })
        
        return True
    
    def open_short_position(self, symbol: str, quantity: float, price: float,
                           timestamp: datetime, stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None, trade_id: Optional[int] = None,
                           signal_id: Optional[str] = None) -> bool:
        """Open a short position"""
        
        if not self.can_open_position(symbol, quantity, price):
            return False
        
        # Calculate fees and margin
        position_value = quantity * price
        fees = position_value * self.trading_fee
        margin_required = position_value  # Simple 1:1 margin
        total_cost = margin_required + fees
        
        # Update cash balance
        self.cash_balance -= total_cost
        
        # Create position
        position = Position(
            symbol=symbol,
            side='SHORT',
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=timestamp,
            trade_id=trade_id,
            signal_id=signal_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees_paid=fees
        )
        
        # Store position
        position_key = f"{symbol}_SHORT_{timestamp.isoformat()}"
        self.positions[position_key] = position
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': timestamp,
            'type': 'OPEN_SHORT',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'fees': fees,
            'cash_after': self.cash_balance
        })
        
        return True
    
    def close_position(self, position_key: str, price: float, timestamp: datetime) -> bool:
        """Close a specific position"""
        
        if position_key not in self.positions:
            return False
        
        position = self.positions[position_key]
        
        # Calculate PnL
        if position.side == 'LONG':
            pnl = (price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - price) * position.quantity
        
        # Calculate closing fees
        position_value = position.quantity * price
        closing_fees = position_value * self.trading_fee
        net_pnl = pnl - closing_fees
        
        # Update cash balance
        if position.side == 'LONG':
            self.cash_balance += position_value - closing_fees
        else:  # SHORT
            # Return margin + PnL
            self.cash_balance += (position.quantity * position.entry_price) + net_pnl
        
        # Update position for final record
        position.current_price = price
        position.realized_pnl = net_pnl
        position.fees_paid += closing_fees
        position.closed = True  # Mark as closed
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_key]
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': timestamp,
            'type': 'CLOSE_POSITION',
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': price,
            'pnl': net_pnl,
            'fees': closing_fees,
            'cash_after': self.cash_balance
        })
        
        return True
    
    def update_position_prices(self, symbol_prices: Dict[str, float]):
        """Update current prices for all positions"""
        
        for position in self.positions.values():
            if position.symbol in symbol_prices:
                new_price = symbol_prices[position.symbol]
                position.current_price = new_price
                
                # Calculate unrealized PnL
                if position.side == 'LONG':
                    position.unrealized_pnl = (new_price - position.entry_price) * position.quantity
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - new_price) * position.quantity
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not self.closed_positions and not self.positions:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Closed positions analysis
        closed_pnls = [pos.realized_pnl for pos in self.closed_positions]
        winning_trades = [pnl for pnl in closed_pnls if pnl > 0]
        losing_trades = [pnl for pnl in closed_pnls if pnl < 0]
        
        # Calculate metrics
        total_trades = len(closed_pnls)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown from transaction history
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'total_fees_paid': sum(pos.fees_paid for pos in self.closed_positions + list(self.positions.values()))
        }
    
    def close_position_by_trade_id(self, trade_id: int, price: float, timestamp: datetime) -> Optional[Dict]:
        """Close position by trade_id"""
        position_key = None
        for key, position in self.positions.items():
            if position.trade_id == trade_id:
                position_key = key
                break
        
        if position_key and self.close_position(position_key, price, timestamp):
            return {
                'status': 'closed',
                'position_key': position_key,
                'trade_id': trade_id,
                'exit_price': price,
                'timestamp': timestamp
            }
        
        return None
    
    def close_position_by_signal_id(self, signal_id: str, price: float, timestamp: datetime) -> Optional[Dict]:
        """Close position by signal_id"""
        position_key = None
        for key, position in self.positions.items():
            if position.signal_id == signal_id:
                position_key = key
                break
        
        if position_key and self.close_position(position_key, price, timestamp):
            return {
                'status': 'closed',
                'position_key': position_key,
                'signal_id': signal_id,
                'exit_price': price,
                'timestamp': timestamp
            }
        
        return None
    
    def close_matching_position(self, symbol: str, side: str, price: float, timestamp: datetime) -> Optional[Dict]:
        """Close the most recent matching position by symbol and side"""
        matching_positions = []
        
        # Find all matching positions
        for key, position in self.positions.items():
            if position.symbol == symbol and position.side.upper() == side.upper():
                matching_positions.append((key, position))
        
        if not matching_positions:
            return None
        
        # Sort by entry time (most recent first)
        matching_positions.sort(key=lambda x: x[1].entry_time, reverse=True)
        position_key = matching_positions[0][0]
        
        if self.close_position(position_key, price, timestamp):
            return {
                'status': 'closed',
                'position_key': position_key,
                'symbol': symbol,
                'side': side,
                'exit_price': price,
                'timestamp': timestamp
            }
        
        return None
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from transaction history"""
        
        if not self.transaction_history:
            return 0.0
        
        # Build equity curve
        equity_curve = [self.initial_balance]
        running_balance = self.initial_balance
        
        for transaction in self.transaction_history:
            if transaction['type'] == 'CLOSE_POSITION':
                running_balance += transaction['pnl']
                equity_curve.append(running_balance)
        
        # Calculate drawdown
        peak = equity_curve[0]
        max_drawdown = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown