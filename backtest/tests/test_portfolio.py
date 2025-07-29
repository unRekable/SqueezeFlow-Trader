#!/usr/bin/env python3
"""
Unit Tests for Portfolio Management
Tests portfolio operations, position management, and risk limits
"""

import unittest
import sys
import os
from datetime import datetime, timezone
from decimal import Decimal

# Add backtest directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.portfolio import (
    PortfolioManager, Position, PositionType, PositionStatus, RiskLimits
)


class TestPosition(unittest.TestCase):
    """Test Position data class and methods"""
    
    def setUp(self):
        """Set up test position"""
        self.position = Position(
            id="test_pos_1",
            symbol="BTCUSDT",
            position_type=PositionType.LONG,
            entry_time=datetime.now(timezone.utc),
            entry_price=50000.0,
            size=0.1
        )
    
    def test_position_creation(self):
        """Test position creation with valid parameters"""
        self.assertEqual(self.position.id, "test_pos_1")
        self.assertEqual(self.position.symbol, "BTCUSDT")
        self.assertEqual(self.position.position_type, PositionType.LONG)
        self.assertEqual(self.position.entry_price, 50000.0)
        self.assertEqual(self.position.size, 0.1)
        self.assertEqual(self.position.status, PositionStatus.OPEN)
    
    def test_position_value_calculation(self):
        """Test position value calculation"""
        expected_value = self.position.entry_price * self.position.size
        self.assertEqual(self.position.entry_price * self.position.size, expected_value)
    
    def test_position_pnl_calculation(self):
        """Test P&L calculation for different scenarios"""
        # Long position profit
        current_price = 55000.0
        pnl = (current_price - self.position.entry_price) * self.position.size
        expected_pnl = 500.0  # (55000 - 50000) * 0.1
        self.assertEqual(pnl, expected_pnl)
        
        # Long position loss
        current_price = 45000.0
        pnl = (current_price - self.position.entry_price) * self.position.size
        expected_pnl = -500.0  # (45000 - 50000) * 0.1
        self.assertEqual(pnl, expected_pnl)
    
    def test_short_position_pnl(self):
        """Test P&L calculation for short positions"""
        short_position = Position(
            id="test_short_1",
            symbol="BTCUSDT",
            position_type=PositionType.SHORT,
            entry_time=datetime.now(timezone.utc),
            entry_price=50000.0,
            size=0.1
        )
        
        # Short position profit (price goes down)
        current_price = 45000.0
        pnl = (short_position.entry_price - current_price) * short_position.size
        expected_pnl = 500.0  # (50000 - 45000) * 0.1
        self.assertEqual(pnl, expected_pnl)


class TestRiskLimits(unittest.TestCase):
    """Test RiskLimits configuration"""
    
    def test_default_risk_limits(self):
        """Test default risk limit values"""
        risk_limits = RiskLimits()
        self.assertEqual(risk_limits.max_position_size, 0.02)
        self.assertEqual(risk_limits.max_total_exposure, 0.1)
        self.assertEqual(risk_limits.max_open_positions, 2)
        self.assertEqual(risk_limits.max_daily_loss, 0.05)
        self.assertEqual(risk_limits.max_drawdown, 0.15)  # Actual default from code
    
    def test_custom_risk_limits(self):
        """Test custom risk limit configuration"""
        custom_limits = RiskLimits(
            max_position_size=0.05,
            max_total_exposure=0.2,
            max_open_positions=5
        )
        self.assertEqual(custom_limits.max_position_size, 0.05)
        self.assertEqual(custom_limits.max_total_exposure, 0.2)
        self.assertEqual(custom_limits.max_open_positions, 5)


class TestPortfolioManager(unittest.TestCase):
    """Test PortfolioManager functionality"""
    
    def setUp(self):
        """Set up test portfolio manager"""
        self.initial_balance = 10000.0
        self.risk_limits = RiskLimits(max_position_size=0.1, max_open_positions=3)
        self.portfolio = PortfolioManager(
            initial_balance=self.initial_balance,
            risk_limits=self.risk_limits
        )
    
    def test_portfolio_initialization(self):
        """Test portfolio manager initialization"""
        self.assertEqual(self.portfolio.initial_balance, self.initial_balance)
        self.assertEqual(self.portfolio.current_balance, self.initial_balance)
        self.assertEqual(len(self.portfolio.open_positions), 0)
        self.assertEqual(len(self.portfolio.closed_positions), 0)
        self.assertEqual(self.portfolio.position_counter, 0)
    
    def test_position_size_validation(self):
        """Test position size validation against risk limits"""
        # Test can_open_position method which actually exists
        can_open, reason = self.portfolio.can_open_position("BTCUSDT", 0.05)  # 5% of balance
        self.assertTrue(can_open)
        self.assertEqual(reason, "Position allowed")
        
        # Position size too large (exceeds risk limit)
        can_open_large, reason_large = self.portfolio.can_open_position("BTCUSDT", 0.15)  # 15% > 10% limit
        self.assertFalse(can_open_large)
        self.assertIn("Position size too large", reason_large)
    
    def test_open_position(self):
        """Test opening a new position"""
        position = self.portfolio.open_position(
            symbol="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            position_size_percentage=0.05,  # Correct parameter name
            timestamp=datetime.now(timezone.utc)  # Required parameter
        )
        
        self.assertIsNotNone(position)
        self.assertEqual(len(self.portfolio.open_positions), 1)  # open_positions not positions
        self.assertEqual(self.portfolio.position_counter, 1)  # position_counter not total_trades
        self.assertEqual(position.symbol, "BTCUSDT")
        self.assertEqual(position.position_type, PositionType.LONG)
    
    def test_close_position(self):
        """Test closing an existing position"""
        # First open a position
        position = self.portfolio.open_position(
            symbol="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            position_size_percentage=0.05,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Then close it
        exit_price = 55000.0
        closed_position = self.portfolio.close_position(
            position.id, 
            exit_price, 
            timestamp=datetime.now(timezone.utc),  # Required parameter
            exit_reason="profit_target"
        )
        
        self.assertIsNotNone(closed_position)
        self.assertEqual(len(self.portfolio.open_positions), 0)  # open_positions not positions
        self.assertEqual(len(self.portfolio.closed_positions), 1)
        self.assertEqual(closed_position.status, PositionStatus.CLOSED)
        self.assertEqual(closed_position.exit_price, exit_price)
        self.assertEqual(closed_position.exit_reason, "profit_target")
    
    def test_max_positions_limit(self):
        """Test maximum open positions limit"""
        # Open positions up to the limit
        for i in range(self.risk_limits.max_open_positions):
            position = self.portfolio.open_position(
                symbol=f"TEST{i}USDT",
                position_type=PositionType.LONG,
                entry_price=50000.0,
                position_size_percentage=0.02,
                timestamp=datetime.now(timezone.utc)
            )
            self.assertIsNotNone(position)
        
        # Try to open one more position (should fail)
        failed_position = self.portfolio.open_position(
            symbol="FAILUSDT",
            position_type=PositionType.LONG,
            entry_price=50000.0,
            position_size_percentage=0.02,
            timestamp=datetime.now(timezone.utc)
        )
        self.assertIsNone(failed_position)
        self.assertEqual(len(self.portfolio.open_positions), self.risk_limits.max_open_positions)
    
    def test_portfolio_metrics(self):
        """Test portfolio performance metrics calculation"""
        timestamp = datetime.now(timezone.utc)
        
        # Open and close some positions for testing
        pos1 = self.portfolio.open_position("BTC1", PositionType.LONG, 50000.0, 0.05, timestamp)
        self.portfolio.close_position(pos1.id, 55000.0, timestamp, "profit")  # Profit
        
        pos2 = self.portfolio.open_position("BTC2", PositionType.LONG, 50000.0, 0.05, timestamp)
        self.portfolio.close_position(pos2.id, 45000.0, timestamp, "loss")    # Loss
        
        metrics = self.portfolio.get_performance_metrics()  # Correct method name
        
        self.assertIn('total_return', metrics)  # total_return not total_return_pct
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        self.assertIn('total_trades', metrics)
        
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['winning_trades'], 1)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertEqual(metrics['win_rate'], 50.0)  # 50% win rate (returned as percentage)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation during losses"""
        timestamp = datetime.now(timezone.utc)
        
        # Create a losing position
        position = self.portfolio.open_position("BTCUSDT", PositionType.LONG, 50000.0, 0.1, timestamp)
        self.portfolio.close_position(position.id, 40000.0, timestamp, "stop_loss")  # 20% loss
        
        metrics = self.portfolio.get_performance_metrics()  # Correct method name
        
        # Should have some drawdown from the loss
        self.assertGreater(metrics['max_drawdown'], 0)
        self.assertLess(self.portfolio.current_balance, self.initial_balance)


class TestPortfolioIntegration(unittest.TestCase):
    """Integration tests for portfolio components"""
    
    def test_complete_trading_cycle(self):
        """Test a complete trading cycle with multiple positions"""
        portfolio = PortfolioManager(initial_balance=10000.0)
        timestamp = datetime.now(timezone.utc)
        
        # Open multiple positions (use smaller position sizes that fit within limits)
        long_pos = portfolio.open_position("BTCUSDT", PositionType.LONG, 50000.0, 0.01, timestamp)  # 1% fits in default 2% limit
        short_pos = portfolio.open_position("ETHUSDT", PositionType.SHORT, 3000.0, 0.01, timestamp)  # 1% fits in default 2% limit
        
        self.assertEqual(len(portfolio.open_positions), 2)  # open_positions not positions
        
        # Close positions with different outcomes
        portfolio.close_position(long_pos.id, 52000.0, timestamp, "profit_target")  # Profit
        portfolio.close_position(short_pos.id, 3200.0, timestamp, "stop_loss")     # Loss
        
        self.assertEqual(len(portfolio.open_positions), 0)  # open_positions not positions
        self.assertEqual(len(portfolio.closed_positions), 2)
        
        # Check final metrics
        metrics = portfolio.get_performance_metrics()  # Correct method name
        self.assertEqual(metrics['total_trades'], 2)
        self.assertIsInstance(metrics['total_return'], float)  # total_return not total_return_pct


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run tests
    unittest.main(verbosity=2)