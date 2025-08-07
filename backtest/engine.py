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
import pytz
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
        Run backtest with rolling window processing to eliminate lookahead bias
        
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
        # Parse dates and make them timezone-aware (UTC) to match InfluxDB data
        start_time = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        end_time = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        
        self.logger.info(f"🚀 Starting rolling window backtest: {symbol} from {start_date} to {end_date}")
        self.logger.info(f"💰 Initial balance: ${self.initial_balance:,.2f}")
        self.logger.info("🔄 Using 4-hour rolling windows with 5-minute steps")
        
        # STEP 1: LOAD COMPLETE RAW DATA FIRST
        self.logger.info("📊 Loading complete raw market data...")
        full_dataset = self.data_pipeline.get_complete_dataset(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )
        
        # Validate data quality
        quality = self.data_pipeline.validate_data_quality(full_dataset)
        if not quality['overall_quality']:
            self.logger.error(f"❌ Data quality check failed: {quality}")
            return self._create_failed_result("Data quality insufficient")
        
        self.logger.info(f"✅ Data loaded: {full_dataset['metadata']['data_points']} data points")
        self.logger.info(f"📈 Markets: {full_dataset['metadata']['spot_markets_count']} SPOT, {full_dataset['metadata']['futures_markets_count']} FUTURES")
        
        # STEP 2: ROLLING WINDOW STRATEGY PROCESSING
        self.logger.info("🧠 Running rolling window strategy analysis...")
        
        try:
            executed_orders = self._run_rolling_window_backtest(
                full_dataset, start_time, end_time, timeframe
            )
            
        except Exception as e:
            self.logger.error(f"❌ Rolling window processing failed: {e}")
            return self._create_failed_result(f"Strategy error: {e}")
        
        # STEP 3: GENERATE RESULTS
        result = self._create_result(full_dataset, executed_orders)
        
        # STEP 4: CREATE VISUALIZATIONS
        self.logger.info("📊 Generating visualizations...")
        visualization_path = self.visualizer.create_backtest_report(
            result, full_dataset, executed_orders
        )
        result['visualization_path'] = visualization_path
        
        self.logger.info(f"🎯 Backtest completed - Final balance: ${result['final_balance']:,.2f}")
        self.logger.info(f"📈 Total return: {result['total_return']:.2f}%")
        
        return result
    
    def _run_rolling_window_backtest(self, full_dataset: Dict, start_time: datetime, 
                                   end_time: datetime, timeframe: str) -> List[Dict]:
        """
        Process backtest using rolling 4-hour windows stepping forward 5 minutes at a time
        This eliminates lookahead bias by only providing data up to "current time"
        
        Args:
            full_dataset: Complete dataset loaded from data pipeline
            start_time: Backtest start time
            end_time: Backtest end time  
            timeframe: Trading timeframe (e.g., '5m')
            
        Returns:
            List of executed orders from all windows
        """
        # Rolling window parameters (matching live trading)
        window_hours = 4  # 4-hour rolling windows
        step_minutes = 5  # Step forward 5 minutes each iteration
        
        window_duration = timedelta(hours=window_hours)
        step_duration = timedelta(minutes=step_minutes)
        
        all_executed_orders = []
        
        # Get data timeframes for indexing
        ohlcv = full_dataset.get('ohlcv', pd.DataFrame())
        if ohlcv.empty:
            self.logger.error("No OHLCV data available for rolling window processing")
            return all_executed_orders
        
        # Ensure datetime index for proper slicing
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            self.logger.error("OHLCV data must have datetime index for rolling windows")
            return all_executed_orders
        
        # Calculate total iterations for progress tracking
        total_duration = end_time - start_time
        total_iterations = int(total_duration.total_seconds() / step_duration.total_seconds())
        
        self.logger.info(f"🔄 Processing {total_iterations:,} rolling windows ({window_hours}h windows, {step_minutes}m steps)")
        
        # Start with first window (4 hours from start_time)
        # Ensure current_time is timezone-aware (UTC) to match data timestamps
        current_time = start_time + window_duration
        iteration = 0
        
        while current_time <= end_time:
            iteration += 1
            
            # Define window boundaries
            window_start = current_time - window_duration
            window_end = current_time
            
            # Progress logging every 100 iterations or at milestones
            if iteration % 100 == 0 or iteration == total_iterations:
                progress_pct = (iteration / total_iterations) * 100
                self.logger.info(f"🔄 Progress: {iteration:,}/{total_iterations:,} ({progress_pct:.1f}%) - Window: {window_end.strftime('%Y-%m-%d %H:%M')}")
            
            try:
                # Create windowed dataset (only data up to current_time)
                windowed_dataset = self._create_windowed_dataset(
                    full_dataset, window_start, window_end
                )
                
                # Skip if insufficient data in window
                if not self._validate_windowed_data(windowed_dataset):
                    current_time += step_duration
                    continue
                
                # Get current portfolio state
                portfolio_state = self.portfolio.get_state()
                portfolio_state['available_leverage'] = self.leverage
                portfolio_state['current_time'] = current_time  # Provide current time to strategy
                
                # Process strategy with windowed data (no future visibility)
                strategy_result = self.strategy.process(windowed_dataset, portfolio_state)
                
                if strategy_result and 'orders' in strategy_result:
                    orders = strategy_result['orders']
                    
                    # Execute any orders generated for this window
                    for order in orders:
                        # Ensure order timestamp doesn't exceed current_time and is timezone-aware
                        if 'timestamp' not in order or order['timestamp'] > current_time:
                            order['timestamp'] = current_time
                        elif 'timestamp' in order and order['timestamp'].tzinfo is None and current_time.tzinfo is not None:
                            # Make order timestamp timezone-aware if current_time is timezone-aware
                            order['timestamp'] = order['timestamp'].replace(tzinfo=current_time.tzinfo)
                        
                        execution_result = self._execute_order(order, windowed_dataset)
                        if execution_result:
                            all_executed_orders.append(execution_result)
                            
                            # Log significant trades
                            side = execution_result.get('side', 'UNKNOWN')
                            qty = execution_result.get('quantity', 0)
                            price = execution_result.get('price', 0)
                            signal_type = execution_result.get('signal_type', 'UNKNOWN')
                            
                            self.logger.info(f"✅ {window_end.strftime('%Y-%m-%d %H:%M')} - {side} {qty:.6f} @ ${price:.2f} ({signal_type})")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing window {window_end}: {e}")
            
            # Move to next window
            current_time += step_duration
        
        self.logger.info(f"🎯 Rolling window processing completed: {len(all_executed_orders)} total orders executed")
        return all_executed_orders
    
    def _create_windowed_dataset(self, full_dataset: Dict, window_start: datetime, 
                               window_end: datetime) -> Dict:
        """
        Create a dataset containing only data from window_start to window_end
        This prevents lookahead bias by limiting data visibility
        
        Args:
            full_dataset: Complete dataset from data pipeline
            window_start: Start of current window (timezone-aware UTC)
            window_end: End of current window (current_time, timezone-aware UTC)
            
        Returns:
            Dict with windowed data matching full_dataset structure
        """
        windowed_dataset = {
            'symbol': full_dataset.get('symbol'),
            'timeframe': full_dataset.get('timeframe'),
            'start_time': window_start,
            'end_time': window_end,
            'markets': full_dataset.get('markets', {}),
            'metadata': full_dataset.get('metadata', {})
        }
        
        # Slice time-series data to window boundaries
        for data_key in ['ohlcv', 'spot_volume', 'futures_volume', 'spot_cvd', 'futures_cvd', 'cvd_divergence']:
            full_data = full_dataset.get(data_key, pd.DataFrame() if data_key in ['ohlcv', 'spot_volume', 'futures_volume'] else pd.Series())
            
            if isinstance(full_data, (pd.DataFrame, pd.Series)) and not full_data.empty:
                if isinstance(full_data.index, pd.DatetimeIndex):
                    # Ensure window boundaries are timezone-aware to match data timestamps
                    window_start_tz = window_start
                    window_end_tz = window_end
                    
                    # If data is timezone-aware but window boundaries aren't, make them UTC
                    if full_data.index.tz is not None:
                        if window_start.tzinfo is None:
                            window_start_tz = window_start.replace(tzinfo=pytz.UTC)
                        if window_end.tzinfo is None:
                            window_end_tz = window_end.replace(tzinfo=pytz.UTC)
                    # If data is timezone-naive but window boundaries are timezone-aware, convert data
                    elif full_data.index.tz is None and window_start.tzinfo is not None:
                        # Convert data index to UTC timezone-aware
                        full_data.index = full_data.index.tz_localize('UTC')
                    
                    # Slice data to window - only data up to window_end (current_time)
                    try:
                        windowed_data = full_data.loc[window_start_tz:window_end_tz]
                        windowed_dataset[data_key] = windowed_data
                    except Exception as e:
                        # Fallback for timezone comparison issues
                        self.logger.warning(f"Timezone comparison failed for {data_key}, using fallback filtering: {e}")
                        # Manual filtering as fallback
                        mask = (full_data.index >= window_start_tz) & (full_data.index <= window_end_tz)
                        windowed_dataset[data_key] = full_data[mask]
                else:
                    # Fallback: use full data if no datetime index
                    windowed_dataset[data_key] = full_data
            else:
                # Empty data
                windowed_dataset[data_key] = full_data
        
        # Update metadata for windowed dataset
        if isinstance(windowed_dataset.get('ohlcv'), pd.DataFrame):
            windowed_dataset['metadata']['window_data_points'] = len(windowed_dataset['ohlcv'])
        
        return windowed_dataset
    
    def _validate_windowed_data(self, windowed_dataset: Dict) -> bool:
        """
        Validate that windowed dataset has sufficient data for strategy processing
        
        Args:
            windowed_dataset: Windowed dataset to validate
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        ohlcv = windowed_dataset.get('ohlcv', pd.DataFrame())
        spot_cvd = windowed_dataset.get('spot_cvd', pd.Series())
        futures_cvd = windowed_dataset.get('futures_cvd', pd.Series())
        
        # Require minimum data points for analysis (at least 30 points for 4h window at 5m intervals)
        min_points = 30
        
        if ohlcv.empty or len(ohlcv) < min_points:
            return False
        
        if spot_cvd.empty or len(spot_cvd) < min_points:
            return False
            
        if futures_cvd.empty or len(futures_cvd) < min_points:
            return False
        
        return True
    
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
        timestamp = order.get('timestamp', datetime.now(tz=pytz.UTC))
        
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
                    'timestamp': datetime.now(tz=pytz.UTC),
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
                        'timestamp': datetime.now(tz=pytz.UTC),
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