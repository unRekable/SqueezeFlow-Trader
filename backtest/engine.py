#!/usr/bin/env python3
"""
Clean Backtest Engine - Pure Orchestration
NO calculations, NO trading logic - just data loading and order execution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

from data.pipeline import create_data_pipeline

# Handle imports for both package and direct script execution
try:
    from .core.portfolio import Portfolio
    from .core.fake_redis import FakeRedis
    from .reporting.logger import BacktestLogger
    from .reporting.visualizer import BacktestVisualizer
except ImportError:
    # Direct script execution
    from core.portfolio import Portfolio
    from core.fake_redis import FakeRedis
    from reporting.logger import BacktestLogger
    from reporting.visualizer import BacktestVisualizer

# Optional import for CVD baseline manager
try:
    from strategies.squeezeflow.baseline_manager import CVDBaselineManager
except ImportError:
    CVDBaselineManager = None


class BacktestEngine:
    """
    Pure orchestration engine - loads data and executes orders
    Strategy has COMPLETE authority over all calculations and decisions
    """
    
    def __init__(self, initial_balance: float = 10000.0, leverage: float = 1.0):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.portfolio = Portfolio(initial_balance)
        self.data_pipeline = create_data_pipeline()
        self.logger = BacktestLogger()
        self.visualizer = BacktestVisualizer()
        
        # CVD baseline tracking for backtests
        self.fake_redis = FakeRedis()
        self.cvd_baseline_manager = None
        if CVDBaselineManager:
            self.cvd_baseline_manager = CVDBaselineManager(
                redis_client=self.fake_redis, 
                key_prefix="backtest"
            )
        
        # State tracking
        self.current_time = None
        self.current_data = None
        self.strategy = None
        self.next_trade_id = 1  # For generating trade IDs in backtest
        
    def run(self, strategy, symbol: str, start_date: str, end_date: str, 
            timeframe: str = '5m', balance: Optional[float] = None, leverage: Optional[float] = None) -> Dict:
        """
        Run backtest with given strategy
        
        Args:
            strategy: Strategy instance with process() method
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            timeframe: Timeframe for analysis
            balance: Override initial balance
            
        Returns:
            Dict with backtest results
        """
        # Setup
        if balance:
            self.portfolio = Portfolio(balance)
            self.initial_balance = balance
        
        if leverage:
            self.leverage = leverage
            
        self.strategy = strategy
        # Pass leverage to strategy if it supports it
        if hasattr(strategy, 'set_leverage'):
            strategy.set_leverage(self.leverage)
        
        # Pass CVD baseline manager to strategy if it supports it
        if hasattr(strategy, 'set_cvd_baseline_manager') and self.cvd_baseline_manager:
            strategy.set_cvd_baseline_manager(self.cvd_baseline_manager)
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.logger.info(f"🚀 Starting backtest: {symbol} from {start_date} to {end_date}")
        self.logger.info(f"💰 Initial balance: ${self.initial_balance:,.2f}")
        
        # STEP 1: LOAD RAW DATA ONLY (no calculations)
        self.logger.info("📊 Loading raw market data...")
        dataset = self.data_pipeline.get_complete_dataset(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )
        
        # Validate data quality
        quality = self.data_pipeline.validate_data_quality(dataset)
        if not quality['overall_quality']:
            self.logger.error(f"❌ Data quality check failed: {quality}")
            return self._create_failed_result("Data quality insufficient")
        
        self.logger.info(f"✅ Data loaded: {dataset['metadata']['data_points']} data points")
        self.logger.info(f"📈 Markets: {dataset['metadata']['spot_markets_count']} SPOT, {dataset['metadata']['futures_markets_count']} FUTURES")
        
        # STEP 2: STRATEGY PROCESSING (strategy calculates everything)
        self.logger.info("🧠 Running strategy analysis...")
        
        try:
            # Strategy processes ALL data and returns trading decisions
            portfolio_state = self.portfolio.get_state()
            portfolio_state['available_leverage'] = self.leverage  # Pass leverage to strategy
            strategy_result = self.strategy.process(dataset, portfolio_state)
            
            if not strategy_result or 'orders' not in strategy_result:
                self.logger.warning("⚠️ Strategy returned no orders")
                return self._create_result(dataset, [])
            
            orders = strategy_result['orders']
            self.logger.info(f"📋 Strategy generated {len(orders)} orders")
            
        except Exception as e:
            self.logger.error(f"❌ Strategy processing failed: {e}")
            return self._create_failed_result(f"Strategy error: {e}")
        
        # STEP 3: EXECUTE ORDERS (pure orchestration)
        self.logger.info("⚡ Executing trading orders...")
        executed_orders = []
        
        for order in orders:
            try:
                execution_result = self._execute_order(order, dataset)
                if execution_result:
                    executed_orders.append(execution_result)
                    self.logger.info(f"✅ Order executed: {execution_result['side']} {execution_result['quantity']} @ ${execution_result['price']:.2f}")
                
            except Exception as e:
                self.logger.error(f"❌ Order execution failed: {order} - {e}")
        
        # STEP 4: GENERATE RESULTS
        result = self._create_result(dataset, executed_orders)
        
        # STEP 5: CREATE VISUALIZATIONS
        self.logger.info("📊 Generating visualizations...")
        visualization_path = self.visualizer.create_backtest_report(
            result, dataset, executed_orders
        )
        result['visualization_path'] = visualization_path
        
        self.logger.info(f"🎯 Backtest completed - Final balance: ${result['final_balance']:,.2f}")
        self.logger.info(f"📈 Total return: {result['total_return']:.2f}%")
        
        return result
    
    def _execute_order(self, order: Dict, dataset: Dict) -> Optional[Dict]:
        """
        Execute a single order (pure execution, no logic)
        
        Args:
            order: Order dict from strategy
            dataset: Market dataset for execution
            
        Returns:
            Execution result or None if failed
        """
        required_fields = ['symbol', 'side', 'quantity', 'price']
        if not all(field in order for field in required_fields):
            self.logger.error(f"❌ Invalid order format: {order}")
            return None
        
        symbol = order['symbol']
        side = order['side'].upper()
        quantity = float(order['quantity'])
        price = float(order['price'])
        timestamp = order.get('timestamp', datetime.now())
        
        # Check if this is an exit order first
        signal_type = order.get('signal_type', 'ENTRY')
        
        if signal_type == 'EXIT':
            # This is an exit order - close the matching position
            self.logger.info(f"🚪 Processing EXIT order for {symbol}")
            
            # Try different methods to find and close position
            close_result = None
            
            # Method 1: Close by specific trade_id if provided
            if 'trade_id' in order:
                close_result = self.portfolio.close_position_by_trade_id(
                    trade_id=order['trade_id'],
                    price=price,
                    timestamp=timestamp
                )
                
            # Method 2: Close by signal_id if provided
            elif 'signal_id' in order:
                close_result = self.portfolio.close_position_by_signal_id(
                    signal_id=order['signal_id'],
                    price=price,
                    timestamp=timestamp
                )
                
            # Method 3: Close most recent matching position by symbol and side
            else:
                # For exit orders, we need to close the opposite position
                # If exit order is 'BUY', we're closing a SHORT position
                # If exit order is 'SELL', we're closing a LONG position
                position_side = 'SHORT' if side == 'BUY' else 'LONG'
                close_result = self.portfolio.close_matching_position(
                    symbol=symbol,
                    side=position_side,
                    price=price,
                    timestamp=timestamp
                )
            
            if close_result:
                self.logger.info(f"✅ Position closed successfully: {close_result}")
                return {
                    'symbol': symbol,
                    'side': f"EXIT_{side}",
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp,
                    'signal_type': 'EXIT',
                    'close_result': close_result,
                    'fees': order.get('fees', 0),
                    'slippage': order.get('slippage', 0)
                }
            else:
                self.logger.warning(f"⚠️ No matching position found to close for EXIT order: {symbol}")
                return None
        
        else:
            # Handle regular entry orders (existing logic)
            
            # Apply leverage (strategy can override with 'leverage' field in order)
            order_leverage = order.get('leverage', self.leverage)
            effective_quantity = quantity * order_leverage
            
            # Generate trade ID and store CVD baseline if available
            trade_id = self.next_trade_id
            self.next_trade_id += 1
            
            signal_id = order.get('signal_id', f"backtest_{trade_id}")
            
            # Store CVD baseline if manager available and order has CVD data
            if self.cvd_baseline_manager and 'spot_cvd' in order and 'futures_cvd' in order:
                self.cvd_baseline_manager.store_baseline(
                    signal_id=signal_id,
                    trade_id=trade_id,
                    symbol=symbol,
                    side=side.lower(),
                    entry_price=price,
                    spot_cvd=order['spot_cvd'],
                    futures_cvd=order['futures_cvd']
                )
            
            # Execute through portfolio with leverage and tracking info
            success = False
            if side == 'BUY':
                success = self.portfolio.open_long_position(
                    symbol=symbol,
                    quantity=effective_quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_id=trade_id,
                    signal_id=signal_id
                )
            elif side == 'SELL':
                success = self.portfolio.open_short_position(
                    symbol=symbol,
                    quantity=effective_quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_id=trade_id,
                    signal_id=signal_id
                )
            else:
                self.logger.error(f"❌ Unknown order side: {side}")
                return None
            
            if success:
                return {
                    'symbol': symbol,
                    'side': side,
                    'quantity': effective_quantity,
                    'original_quantity': quantity,
                    'leverage': order_leverage,
                    'price': price,
                    'trade_id': trade_id,
                    'signal_id': signal_id,
                    'timestamp': timestamp,
                    'signal_type': signal_type,
                    'fees': order.get('fees', 0),
                    'slippage': order.get('slippage', 0)
                }
        
        return None
    
    def _create_result(self, dataset: Dict, executed_orders: List[Dict]) -> Dict:
        """Create backtest result summary"""
        portfolio_state = self.portfolio.get_state()
        
        return {
            'symbol': dataset['symbol'],
            'timeframe': dataset['timeframe'],
            'start_time': dataset['start_time'],
            'end_time': dataset['end_time'],
            'initial_balance': self.initial_balance,
            'final_balance': portfolio_state['total_value'],
            'total_return': ((portfolio_state['total_value'] - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': len(executed_orders),
            'winning_trades': len([o for o in executed_orders if o.get('pnl', 0) > 0]),
            'losing_trades': len([o for o in executed_orders if o.get('pnl', 0) < 0]),
            'win_rate': (len([o for o in executed_orders if o.get('pnl', 0) > 0]) / max(len(executed_orders), 1)) * 100,
            'executed_orders': executed_orders,
            'portfolio_state': portfolio_state,
            'data_quality': self.data_pipeline.validate_data_quality(dataset),
            'metadata': dataset['metadata']
        }
    
    def _create_failed_result(self, error_message: str) -> Dict:
        """Create failed result"""
        return {
            'success': False,
            'error': error_message,
            'initial_balance': self.initial_balance,
            'final_balance': self.initial_balance,
            'total_return': 0.0,
            'total_trades': 0
        }


if __name__ == "__main__":
    # CLI interface with leverage support
    import argparse
    
    # Handle import for both package and direct script execution
    try:
        # Import from new modular strategy location
        from strategies.squeezeflow.strategy import SqueezeFlowStrategy
        from strategies.base import BaseStrategy
    except ImportError:
        # Direct script execution - add parent directory to path for strategy imports
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from strategies.squeezeflow.strategy import SqueezeFlowStrategy
        from strategies.base import BaseStrategy
        
    # Simple debug strategy for testing
    class SimpleDebugStrategy(BaseStrategy):
        """Simple debug strategy for testing backtest engine functionality including exits"""
        
        def __init__(self):
            super().__init__(name="SimpleDebugStrategy")
            self.position_opened = False
            self.entry_price = None
            self.entry_trade_id = None
            
        def process(self, dataset, portfolio_state):
            """Generate test orders for debugging both entry and exit functionality"""
            import pandas as pd
            from datetime import datetime
            
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            if ohlcv.empty:
                return {'orders': [], 'metadata': {'strategy': self.name}}
                
            symbol = dataset.get('symbol', 'UNKNOWN')
            current_price = ohlcv.iloc[-1]['close'] if 'close' in ohlcv.columns else ohlcv.iloc[-1].iloc[3]
            existing_positions = portfolio_state.get('positions', {})
            
            orders = []
            
            # If we have no positions and haven't opened one yet, open a position
            if not existing_positions and not self.position_opened:
                orders.append({
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': 0.001,  # Small test quantity
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'signal_type': 'ENTRY',
                    'signal_id': f"debug_entry_{int(datetime.now().timestamp())}"
                })
                self.position_opened = True
                self.entry_price = current_price
                
            # If we have positions and price moved significantly, create exit order
            elif existing_positions and self.entry_price:
                # Simple exit logic: exit if price moved 1% in either direction
                price_change = abs(current_price - self.entry_price) / self.entry_price
                
                if price_change > 0.01:  # 1% move
                    # Create exit order
                    orders.append({
                        'symbol': symbol,
                        'side': 'SELL',  # Exit long position
                        'quantity': 0.001,
                        'price': current_price,
                        'timestamp': datetime.now(),
                        'signal_type': 'EXIT',
                        'reason': f'Price moved {price_change:.2%} from entry'
                    })
                    self.position_opened = False  # Reset for potential re-entry
                    self.entry_price = None
            
            return {
                'orders': orders,
                'metadata': {'strategy': self.name, 'position_opened': self.position_opened}
            }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SqueezeFlow Backtest Engine')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--start-date', default='2024-08-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-08-04', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='5m', help='Timeframe (1m, 5m, 15m, etc.)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--leverage', type=float, default=1.0, help='Trading leverage (default: 1.0)')
    parser.add_argument('--strategy', default='SqueezeFlowStrategy', help='Strategy class name')
    
    args = parser.parse_args()
    
    # Create engine with leverage support
    engine = BacktestEngine(initial_balance=args.balance, leverage=args.leverage)
    
    # Load strategy
    if args.strategy == 'SqueezeFlowStrategy':
        strategy = SqueezeFlowStrategy()
    elif args.strategy == 'SimpleDebugStrategy':
        strategy = SimpleDebugStrategy()
    else:
        strategy = SqueezeFlowStrategy()  # Default fallback
    
    # Run backtest
    result = engine.run(
        strategy=strategy,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        balance=args.balance,
        leverage=args.leverage
    )
    
    print(f"Backtest Result: {result['total_return']:.2f}% return")
    print(f"Leverage Used: {args.leverage}x")
    print(f"Total Trades: {result['total_trades']}")