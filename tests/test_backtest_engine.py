"""
Test Backtest Engine Integration
Comprehensive tests for backtest engine and portfolio management
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import backtest components
from backtest.engine import BacktestEngine
from backtest.core.portfolio import Portfolio
from strategies.squeezeflow.strategy import SqueezeFlowStrategy


class TestBacktestEngine:
    """Test the main backtest engine"""
    
    def test_backtest_engine_initialization(self):
        """Test backtest engine initializes correctly"""
        engine = BacktestEngine(initial_balance=10000.0, leverage=2.0)
        
        assert engine.initial_balance == 10000.0
        assert engine.leverage == 2.0
        assert isinstance(engine.portfolio, Portfolio)
        assert engine.data_pipeline is not None
        assert engine.logger is not None
        assert engine.visualizer is not None
        
    def test_backtest_engine_default_initialization(self):
        """Test backtest engine with default parameters"""
        engine = BacktestEngine()
        
        assert engine.initial_balance == 10000.0
        assert engine.leverage == 1.0
        
    @pytest.mark.integration
    def test_run_backtest_basic(self, sample_dataset, mock_data_pipeline):
        """Test basic backtest execution"""
        engine = BacktestEngine(initial_balance=5000.0)
        
        # Create a simple mock strategy
        mock_strategy = MagicMock()
        mock_strategy.process.return_value = {
            'orders': [{
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000.0,
                'timestamp': datetime.now()
            }],
            'phase_results': {
                'phase4_scoring': {'total_score': 6.5, 'should_trade': True}
            }
        }
        
        # Mock data pipeline
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02',
                timeframe='5m'
            )
            
            # Verify result structure
            assert 'symbol' in result
            assert 'total_return' in result
            assert 'total_trades' in result
            assert 'final_balance' in result
            assert 'executed_orders' in result
            
            assert result['symbol'] == 'BTCUSDT'
            assert result['initial_balance'] == 5000.0
            assert isinstance(result['total_return'], (int, float))
            
    @pytest.mark.integration
    def test_run_backtest_with_real_strategy(self, sample_dataset, mock_data_pipeline):
        """Test backtest with real SqueezeFlow strategy"""
        engine = BacktestEngine(initial_balance=10000.0, leverage=3.0)
        strategy = SqueezeFlowStrategy()
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            result = engine.run(
                strategy=strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02',
                timeframe='5m',
                leverage=3.0
            )
            
            # Should execute without errors
            assert 'total_return' in result
            assert 'executed_orders' in result
            assert result['initial_balance'] == 10000.0
            
    def test_run_backtest_data_quality_failure(self, mock_data_pipeline):
        """Test backtest handles data quality failures"""
        engine = BacktestEngine()
        mock_strategy = MagicMock()
        
        # Mock poor data quality
        poor_dataset = {'ohlcv': pd.DataFrame(), 'metadata': {'data_points': 0}}
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = poor_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': False}
            
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02'
            )
            
            # Should return failed result
            assert result['success'] == False
            assert 'error' in result
            assert result['total_trades'] == 0
            
    def test_run_backtest_strategy_error(self, sample_dataset, mock_data_pipeline):
        """Test backtest handles strategy processing errors"""
        engine = BacktestEngine()
        
        # Mock strategy that raises exception
        mock_strategy = MagicMock()
        mock_strategy.process.side_effect = Exception("Strategy processing error")
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02'
            )
            
            # Should handle error gracefully
            assert result['success'] == False
            assert 'Strategy error' in result['error']
            
    @pytest.mark.unit
    def test_execute_order_buy(self, sample_dataset):
        """Test buy order execution"""
        engine = BacktestEngine(initial_balance=10000.0, leverage=2.0)
        
        buy_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000.0,
            'timestamp': datetime.now()
        }
        
        result = engine._execute_order(buy_order, sample_dataset)
        
        assert result is not None
        assert result['symbol'] == 'BTCUSDT'
        assert result['side'] == 'BUY'
        assert result['leverage'] == 2.0
        assert result['effective_quantity'] == 0.001 * 2.0  # With leverage
        
    @pytest.mark.unit
    def test_execute_order_sell(self, sample_dataset):
        """Test sell order execution"""
        engine = BacktestEngine(initial_balance=10000.0, leverage=1.5)
        
        sell_order = {
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'quantity': 0.002,
            'price': 49500.0,
            'timestamp': datetime.now(),
            'leverage': 3.0  # Order-specific leverage
        }
        
        result = engine._execute_order(sell_order, sample_dataset)
        
        assert result is not None
        assert result['side'] == 'SELL'
        assert result['leverage'] == 3.0  # Should use order-specific leverage
        assert result['effective_quantity'] == 0.002 * 3.0
        
    @pytest.mark.unit
    def test_execute_order_invalid(self):
        """Test invalid order handling"""
        engine = BacktestEngine()
        
        # Missing required fields
        invalid_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY'
            # Missing quantity and price
        }
        
        result = engine._execute_order(invalid_order, {})
        
        assert result is None
        
    @pytest.mark.unit
    def test_execute_order_unknown_side(self, sample_dataset):
        """Test order with unknown side"""
        engine = BacktestEngine()
        
        invalid_side_order = {
            'symbol': 'BTCUSDT',
            'side': 'INVALID',
            'quantity': 0.001,
            'price': 50000.0
        }
        
        result = engine._execute_order(invalid_side_order, sample_dataset)
        
        assert result is None
        
    def test_create_result(self, sample_dataset):
        """Test result creation"""
        engine = BacktestEngine(initial_balance=10000.0)
        
        executed_orders = [
            {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000.0,
                'pnl': 100.0
            },
            {
                'symbol': 'BTCUSDT',
                'side': 'SELL',
                'quantity': 0.001,
                'price': 50500.0,
                'pnl': -50.0
            }
        ]
        
        result = engine._create_result(sample_dataset, executed_orders)
        
        assert result['symbol'] == sample_dataset['symbol']
        assert result['initial_balance'] == 10000.0
        assert result['total_trades'] == 2
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 1
        assert result['win_rate'] == 50.0
        
    def test_create_failed_result(self):
        """Test failed result creation"""
        engine = BacktestEngine(initial_balance=5000.0)
        
        result = engine._create_failed_result("Test error message")
        
        assert result['success'] == False
        assert result['error'] == "Test error message"
        assert result['initial_balance'] == 5000.0
        assert result['final_balance'] == 5000.0
        assert result['total_return'] == 0.0
        assert result['total_trades'] == 0


class TestPortfolioIntegration:
    """Test portfolio integration within backtest engine"""
    
    @pytest.mark.integration
    def test_portfolio_long_position_workflow(self):
        """Test complete long position workflow"""
        engine = BacktestEngine(initial_balance=10000.0, leverage=2.0)
        
        # Open long position
        success = engine.portfolio.open_long_position(
            symbol='BTCUSDT',
            quantity=0.002,  # Will be multiplied by leverage
            price=50000.0,
            timestamp=datetime.now()
        )
        
        assert success == True
        
        # Check portfolio state
        state = engine.portfolio.get_state()
        assert len(state['positions']) == 1
        assert state['positions'][0]['side'] == 'BUY'
        assert state['positions'][0]['quantity'] == 0.002
        
    @pytest.mark.integration
    def test_portfolio_short_position_workflow(self):
        """Test complete short position workflow"""
        engine = BacktestEngine(initial_balance=10000.0, leverage=1.5)
        
        # Open short position
        success = engine.portfolio.open_short_position(
            symbol='BTCUSDT',
            quantity=0.001,
            price=49000.0,
            timestamp=datetime.now()
        )
        
        assert success == True
        
        # Check portfolio state
        state = engine.portfolio.get_state()
        assert len(state['positions']) == 1
        assert state['positions'][0]['side'] == 'SELL'
        
    @pytest.mark.integration
    def test_portfolio_risk_limits(self):
        """Test portfolio respects risk limits"""
        small_balance = 1000.0
        engine = BacktestEngine(initial_balance=small_balance, leverage=1.0)
        
        # Try to open position larger than available balance
        success = engine.portfolio.open_long_position(
            symbol='BTCUSDT',
            quantity=1.0,  # Very large position
            price=50000.0,  # Would require $50,000
            timestamp=datetime.now()
        )
        
        # Should fail due to insufficient balance
        assert success == False
        
        state = engine.portfolio.get_state()
        assert len(state['positions']) == 0


class TestBacktestEnginePerformance:
    """Performance tests for backtest engine"""
    
    @pytest.mark.performance
    def test_backtest_execution_speed(self, sample_dataset, mock_data_pipeline):
        """Test backtest execution speed"""
        import time
        
        engine = BacktestEngine()
        
        # Create strategy that generates multiple orders
        mock_strategy = MagicMock()
        orders = []
        for i in range(10):  # Generate 10 orders
            orders.append({
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.001,
                'price': 50000.0 + i * 100,
                'timestamp': datetime.now()
            })
        
        mock_strategy.process.return_value = {
            'orders': orders,
            'phase_results': {}
        }
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            start_time = time.time()
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02'
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should execute within reasonable time
            assert execution_time < 2.0, f"Backtest too slow: {execution_time:.3f}s"
            assert result['total_trades'] == 10
            
    @pytest.mark.performance
    def test_order_execution_speed(self, sample_dataset):
        """Test individual order execution speed"""
        import time
        
        engine = BacktestEngine()
        
        order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000.0,
            'timestamp': datetime.now()
        }
        
        # Execute multiple orders and measure average time
        times = []
        for _ in range(100):
            start_time = time.time()
            result = engine._execute_order(order, sample_dataset)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Order execution should be very fast
        assert avg_time < 0.001, f"Order execution too slow: {avg_time:.6f}s"


class TestBacktestEngineEdgeCases:
    """Edge case tests for backtest engine"""
    
    def test_backtest_with_empty_orders(self, sample_dataset, mock_data_pipeline):
        """Test backtest when strategy generates no orders"""
        engine = BacktestEngine()
        
        # Strategy returns no orders
        mock_strategy = MagicMock()
        mock_strategy.process.return_value = {
            'orders': [],
            'phase_results': {'phase4_scoring': {'total_score': 2.0, 'should_trade': False}}
        }
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02'
            )
            
            assert result['total_trades'] == 0
            assert result['total_return'] == 0.0
            assert result['final_balance'] == result['initial_balance']
            
    def test_backtest_with_malformed_orders(self, sample_dataset, mock_data_pipeline):
        """Test backtest handles malformed orders gracefully"""
        engine = BacktestEngine()
        
        # Strategy returns malformed orders
        mock_strategy = MagicMock()
        mock_strategy.process.return_value = {
            'orders': [
                {'symbol': 'BTCUSDT', 'side': 'BUY'},  # Missing quantity, price
                {'side': 'SELL', 'quantity': 0.001, 'price': 50000},  # Missing symbol
                None,  # Null order
                {'symbol': 'BTCUSDT', 'side': 'BUY', 'quantity': 0.001, 'price': 50000}  # Valid
            ],
            'phase_results': {}
        }
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02'
            )
            
            # Should only execute the one valid order
            assert result['total_trades'] == 1
            
    def test_backtest_balance_override(self, sample_dataset, mock_data_pipeline):
        """Test backtest balance override functionality"""
        engine = BacktestEngine(initial_balance=5000.0)
        
        mock_strategy = MagicMock()
        mock_strategy.process.return_value = {'orders': [], 'phase_results': {}}
        
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-08-01',
                end_date='2024-08-02',
                balance=15000.0  # Override balance
            )
            
            # Should use overridden balance
            assert result['initial_balance'] == 15000.0
            assert result['final_balance'] == 15000.0  # No trades
            
    def test_backtest_leverage_override(self, sample_dataset, mock_data_pipeline):
        """Test backtest leverage override functionality"""
        engine = BacktestEngine(leverage=1.0)
        
        mock_strategy = MagicMock()
        mock_strategy.process.return_value = {
            'orders': [{
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000.0,
                'timestamp': datetime.now()
            }],
            'phase_results': {}
        }
        
        # Mock successful portfolio execution
        with patch.object(engine.portfolio, 'open_long_position') as mock_open_long:
            mock_open_long.return_value = True
            
            with patch.object(engine, 'data_pipeline', mock_data_pipeline):
                mock_data_pipeline.get_complete_dataset.return_value = sample_dataset
                mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
                
                result = engine.run(
                    strategy=mock_strategy,
                    symbol='BTCUSDT',
                    start_date='2024-08-01',
                    end_date='2024-08-02',
                    leverage=4.0  # Override leverage
                )
                
                # Verify leverage was applied
                mock_open_long.assert_called_once()
                call_args = mock_open_long.call_args
                # The effective quantity should be original * leverage
                assert call_args[1]['quantity'] == 0.001 * 4.0