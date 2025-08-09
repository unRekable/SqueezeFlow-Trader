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
import gc
import psutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing
import queue

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

# Import performance monitoring
try:
    from utils.performance_monitor import PerformanceMonitorIntegration, time_operation
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    print("Warning: Performance monitoring not available")


class BacktestEngine:
    """
    Pure orchestration engine - loads data and executes orders
    Strategy has COMPLETE authority over all calculations and decisions
    
    1s Data Memory Management:
    - Streaming data pipeline prevents OOM errors
    - Rolling window system with LRU cache
    - Explicit memory cleanup with garbage collection
    - Memory monitoring with auto-pause threshold
    
    Phase 2.1 Parallel Processing:
    - Multi-threaded window processing for independent windows
    - Configurable worker count (default 4, max CPU cores)
    - Thread-safe data structures and calculations
    - Progress tracking across parallel workers
    - Error handling for failed workers
    - Maintains result ordering despite parallel execution
    """
    
    def __init__(self, initial_balance: float = 10000.0, leverage: float = 1.0, 
                 enable_1s_mode: bool = False, max_memory_gb: float = 8.0, 
                 max_workers: int = None, enable_parallel: bool = True,
                 enable_performance_monitoring: bool = True):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.portfolio = Portfolio(initial_balance)
        self.data_pipeline = create_data_pipeline()
        self.logger = BacktestLogger()
        self.visualizer = BacktestVisualizer()
        
        # Initialize performance monitoring
        self.performance_monitor = None
        if enable_performance_monitoring and PERFORMANCE_MONITORING_AVAILABLE:
            self.performance_monitor = PerformanceMonitorIntegration()
            self.logger.info("🔍 Performance monitoring enabled")
        else:
            self.logger.info("⚠️ Performance monitoring disabled")
        
        # Parallel processing configuration
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.thread_pool = None
        self.parallel_lock = Lock()  # Thread safety for portfolio operations
        
        self.logger.info(f"🔧 Parallel Processing: {'Enabled' if enable_parallel else 'Disabled'} (Workers: {self.max_workers})")
        
        # 1s Data Memory Management Configuration
        self.enable_1s_mode = enable_1s_mode
        self.max_memory_gb = max_memory_gb
        self.chunk_hours = 2 if enable_1s_mode else 72  # 2h chunks for 1s, 3d for regular
        self.rolling_window_size = 7200 if enable_1s_mode else 288  # 2h of 1s data vs 24h of 5m data
        self.memory_check_interval = 100  # Check memory every N windows
        
        # Rolling window cache for memory efficiency (LRU cache)
        self.data_window_cache = deque(maxlen=self.rolling_window_size)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory_mb = 0
        self.memory_warnings = 0
        
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
        
        # Parallel processing state
        self.window_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.parallel_stats = {
            'parallel_windows_processed': 0,
            'sequential_windows_processed': 0,
            'parallel_speedup_ratio': 1.0,
            'thread_pool_errors': 0
        }
        
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
        
        # STEP 1: MEMORY-EFFICIENT DATA LOADING
        if self.enable_1s_mode:
            self.logger.info("🚨 1s MODE: Using streaming data pipeline to prevent OOM errors")
            self.logger.info(f"⚙️  Memory limit: {self.max_memory_gb}GB, Chunk size: {self.chunk_hours}h")
            # For 1s mode, we'll use streaming approach - no upfront full dataset loading
            full_dataset = self._create_minimal_dataset_structure(symbol, timeframe, start_time, end_time)
        else:
            self.logger.info("📊 Loading complete raw market data...")
            
            # Time data loading with performance monitoring
            data_load_timer = None
            if self.performance_monitor:
                data_load_timer = self.performance_monitor.timer("data_loading", {
                    "symbol": symbol, "timeframe": timeframe, "mode": "regular"
                })
                data_load_timer.__enter__()
                
            try:
                full_dataset = self.data_pipeline.get_complete_dataset(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
                
                # Record data loading metrics
                if self.performance_monitor:
                    data_points = full_dataset.get('metadata', {}).get('data_points', 0)
                    self.performance_monitor.record_data_loading(
                        data_load_timer.start_time and (time.perf_counter() - data_load_timer.start_time) * 1000 or 0,
                        data_points
                    )
                    
            finally:
                if data_load_timer:
                    data_load_timer.__exit__(None, None, None)
            
            # Validate data quality for regular mode
            quality = self.data_pipeline.validate_data_quality(full_dataset)
            if not quality['overall_quality']:
                self.logger.error(f"❌ Data quality check failed: {quality}")
                return self._create_failed_result("Data quality insufficient")
            
            self.logger.info(f"✅ Data loaded: {full_dataset['metadata']['data_points']} data points")
            self.logger.info(f"📈 Markets: {full_dataset['metadata']['spot_markets_count']} SPOT, {full_dataset['metadata']['futures_markets_count']} FUTURES")
        
        # STEP 2: MEMORY-EFFICIENT ROLLING WINDOW STRATEGY PROCESSING
        if self.enable_1s_mode:
            self.logger.info("🧠 Running memory-efficient streaming strategy analysis...")
        else:
            self.logger.info("🧠 Running rolling window strategy analysis...")
        
        # Time strategy processing
        strategy_timer = None
        if self.performance_monitor:
            strategy_timer = self.performance_monitor.timer("strategy_processing", {
                "symbol": symbol, "mode": "1s" if self.enable_1s_mode else "regular",
                "parallel": self.enable_parallel
            })
            strategy_timer.__enter__()
        
        try:
            if self.enable_1s_mode:
                executed_orders = self._run_streaming_backtest(
                    symbol, start_time, end_time, timeframe
                )
            else:
                executed_orders = self._run_rolling_window_backtest(
                    full_dataset, start_time, end_time, timeframe
                )
            
        except Exception as e:
            self.logger.error(f"❌ Strategy processing failed: {e}")
            return self._create_failed_result(f"Strategy error: {e}")
        finally:
            if strategy_timer:
                strategy_timer.__exit__(None, None, None)
        
        # Final memory cleanup
        self._cleanup_memory()
        
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
        
        # Log final memory statistics
        if self.enable_1s_mode:
            self._log_memory_statistics()
            
        # Log parallel processing statistics
        if self.enable_parallel:
            self._log_final_parallel_stats()
            
        # Generate performance report
        if self.performance_monitor:
            self._generate_performance_report(result)
        
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
        # Rolling window parameters (adaptive for 1s mode)
        if self.enable_1s_mode or timeframe == '1s':
            window_hours = 1  # 1-hour window for 1s mode (3600 data points)
            step_seconds = 1  # Step forward 1 SECOND for true granularity
            step_duration = timedelta(seconds=step_seconds)
            self.logger.info(f"📊 1s mode: Using {window_hours}h windows with {step_seconds}s steps")
        else:
            window_hours = 4  # 4-hour rolling windows
            step_minutes = 5  # Step forward 5 minutes each iteration
            step_duration = timedelta(minutes=step_minutes)
            self.logger.info(f"📊 Regular mode: Using {window_hours}h windows with {step_minutes}m steps")
        
        window_duration = timedelta(hours=window_hours)
        
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
        
        # Determine if parallel processing is beneficial
        use_parallel = (self.enable_parallel and 
                       total_iterations > 100 and  # Only for substantial workloads
                       self.max_workers > 1)
        
        if use_parallel:
            if self.enable_1s_mode or timeframe == '1s':
                self.logger.info(f"🚀 PARALLEL Processing {total_iterations:,} rolling windows ({window_hours}h windows, 1s steps)")
            else:
                self.logger.info(f"🚀 PARALLEL Processing {total_iterations:,} rolling windows ({window_hours}h windows, 5m steps)")
            self.logger.info(f"⚡ Using {self.max_workers} parallel workers for optimal performance")
            return self._run_parallel_rolling_windows(full_dataset, start_time, end_time, timeframe, 
                                                    window_duration, step_duration, total_iterations)
        else:
            if self.enable_1s_mode or timeframe == '1s':
                self.logger.info(f"🔄 SEQUENTIAL Processing {total_iterations:,} rolling windows ({window_hours}h windows, 1s steps)")
            else:
                self.logger.info(f"🔄 SEQUENTIAL Processing {total_iterations:,} rolling windows ({window_hours}h windows, 5m steps)")
            self.logger.info(f"📌 Using sequential processing (parallel={'disabled' if not self.enable_parallel else 'not beneficial'})")
        
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
                # Time window processing
                window_start_time = time.perf_counter()
                
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
                
                # Record window processing time
                window_duration_ms = (time.perf_counter() - window_start_time) * 1000
                if self.performance_monitor:
                    self.performance_monitor.record_window_processing_time(window_duration_ms)
                
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
    
    def _run_parallel_rolling_windows(self, full_dataset: Dict, start_time: datetime, 
                                     end_time: datetime, timeframe: str,
                                     window_duration: timedelta, step_duration: timedelta,
                                     total_iterations: int) -> List[Dict]:
        """
        Process backtest using parallel rolling windows for optimal performance
        
        Args:
            full_dataset: Complete dataset loaded from data pipeline
            start_time: Backtest start time
            end_time: Backtest end time
            timeframe: Trading timeframe
            window_duration: Duration of each window
            step_duration: Step between windows
            total_iterations: Total number of windows to process
            
        Returns:
            List of executed orders from all windows
        """
        all_executed_orders = []
        
        # Get data timeframes for indexing
        ohlcv = full_dataset.get('ohlcv', pd.DataFrame())
        if ohlcv.empty:
            self.logger.error("No OHLCV data available for parallel window processing")
            return all_executed_orders
        
        # Ensure datetime index for proper slicing
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            self.logger.error("OHLCV data must have datetime index for parallel windows")
            return all_executed_orders
        
        try:
            # Create thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='backtest_worker') as executor:
                self.thread_pool = executor
                
                # Generate all window tasks
                window_tasks = self._generate_window_tasks(
                    full_dataset, start_time, end_time, window_duration, 
                    step_duration, total_iterations
                )
                
                # Submit all tasks for parallel execution
                future_to_window = {}
                for i, task in enumerate(window_tasks):
                    future = executor.submit(self._process_single_window, task, i)
                    future_to_window[future] = task
                
                # Collect results as they complete
                completed_windows = 0
                failed_windows = 0
                
                for future in as_completed(future_to_window):
                    completed_windows += 1
                    task = future_to_window[future]
                    
                    try:
                        window_result = future.result()
                        if window_result and window_result.get('orders'):
                            # Thread-safe order collection
                            with self.parallel_lock:
                                all_executed_orders.extend(window_result['orders'])
                        
                        # Progress logging every 50 completions or at milestones
                        if completed_windows % 50 == 0 or completed_windows == total_iterations:
                            progress_pct = (completed_windows / total_iterations) * 100
                            self.logger.info(
                                f"⚡ Parallel Progress: {completed_windows:,}/{total_iterations:,} "
                                f"({progress_pct:.1f}%) - Window: {task['window_end'].strftime('%Y-%m-%d %H:%M')}"
                            )
                        
                    except Exception as e:
                        failed_windows += 1
                        self.parallel_stats['thread_pool_errors'] += 1
                        self.logger.error(f"❌ Parallel window processing error: {e}")
                        continue
                
                # Update parallel processing stats
                self.parallel_stats['parallel_windows_processed'] = completed_windows - failed_windows
                
        except Exception as e:
            self.logger.error(f"❌ Parallel processing setup failed: {e}")
            self.logger.info("🔄 Falling back to sequential processing...")
            # Fallback to sequential processing
            return self._run_sequential_fallback(full_dataset, start_time, end_time, timeframe)
        finally:
            self.thread_pool = None
        
        # Log parallel processing results
        self._log_parallel_performance(completed_windows, failed_windows, total_iterations)
        
        self.logger.info(f"🎯 Parallel processing completed: {len(all_executed_orders)} total orders executed")
        return all_executed_orders
    
    def _generate_window_tasks(self, full_dataset: Dict, start_time: datetime, end_time: datetime,
                              window_duration: timedelta, step_duration: timedelta, 
                              total_iterations: int) -> List[Dict]:
        """
        Generate all window processing tasks for parallel execution
        
        Args:
            full_dataset: Complete dataset
            start_time: Backtest start time
            end_time: Backtest end time  
            window_duration: Duration of each window
            step_duration: Step between windows
            total_iterations: Total iterations expected
            
        Returns:
            List of window task dictionaries
        """
        tasks = []
        current_time = start_time + window_duration
        iteration = 0
        
        while current_time <= end_time and iteration < total_iterations:
            iteration += 1
            
            # Define window boundaries
            window_start = current_time - window_duration
            window_end = current_time
            
            # Create task dict (avoiding heavy data copying)
            task = {
                'iteration': iteration,
                'window_start': window_start,
                'window_end': window_end,
                'dataset_ref': id(full_dataset),  # Reference only, actual data passed separately
                'timeframe': full_dataset.get('timeframe'),
                'symbol': full_dataset.get('symbol')
            }
            
            tasks.append(task)
            current_time += step_duration
        
        self.logger.info(f"📅 Generated {len(tasks)} window tasks for parallel processing")
        return tasks
    
    def _process_single_window(self, task: Dict, task_index: int) -> Optional[Dict]:
        """
        Process a single window task in parallel worker thread
        
        This method is thread-safe and processes one window independently
        
        Args:
            task: Window task dictionary
            task_index: Task index for tracking
            
        Returns:
            Dict with window processing results or None if failed
        """
        try:
            window_start = task['window_start']
            window_end = task['window_end']
            iteration = task['iteration']
            
            # Create windowed dataset (each thread gets its own copy)
            # Note: This assumes self.current_data is set before parallel processing
            windowed_dataset = self._create_windowed_dataset(
                self.current_data, window_start, window_end
            )
            
            # Skip if insufficient data in window
            if not self._validate_windowed_data(windowed_dataset):
                return None
            
            # Get current portfolio state (thread-safe snapshot)
            with self.parallel_lock:
                portfolio_state = self.portfolio.get_state().copy()
                portfolio_state['available_leverage'] = self.leverage
                portfolio_state['current_time'] = window_end
            
            # Process strategy with windowed data (thread-safe)
            strategy_result = self.strategy.process(windowed_dataset, portfolio_state)
            
            if strategy_result and 'orders' in strategy_result:
                orders = strategy_result['orders']
                executed_orders = []
                
                # Execute orders with thread safety
                for order in orders:
                    # Ensure order timestamp doesn't exceed current_time and is timezone-aware
                    if 'timestamp' not in order or order['timestamp'] > window_end:
                        order['timestamp'] = window_end
                    elif 'timestamp' in order and order['timestamp'].tzinfo is None and window_end.tzinfo is not None:
                        order['timestamp'] = order['timestamp'].replace(tzinfo=window_end.tzinfo)
                    
                    # Thread-safe order execution
                    with self.parallel_lock:
                        execution_result = self._execute_order(order, windowed_dataset)
                        if execution_result:
                            executed_orders.append(execution_result)
                            
                            # Log significant trades (with thread info)
                            side = execution_result.get('side', 'UNKNOWN')
                            qty = execution_result.get('quantity', 0)
                            price = execution_result.get('price', 0)
                            signal_type = execution_result.get('signal_type', 'UNKNOWN')
                            
                            self.logger.info(
                                f"✅ [{task_index:03d}] {window_end.strftime('%Y-%m-%d %H:%M')} - "
                                f"{side} {qty:.6f} @ ${price:.2f} ({signal_type})"
                            )
                
                return {
                    'iteration': iteration,
                    'window_end': window_end,
                    'orders': executed_orders,
                    'task_index': task_index
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Window processing error (task {task_index}): {e}")
            return None
    
    def _run_sequential_fallback(self, full_dataset: Dict, start_time: datetime, 
                               end_time: datetime, timeframe: str) -> List[Dict]:
        """
        Fallback to sequential processing if parallel processing fails
        
        Args:
            full_dataset: Complete dataset
            start_time: Backtest start time
            end_time: Backtest end time
            timeframe: Trading timeframe
            
        Returns:
            List of executed orders
        """
        # Store current data for window processing
        self.current_data = full_dataset
        
        # Use original sequential logic with adaptive parameters
        if self.enable_1s_mode or timeframe == '1s':
            window_hours = 1  # 1-hour window for 1s mode
            step_seconds = 1  # 1-second steps
            step_duration = timedelta(seconds=step_seconds)
        else:
            window_hours = 4
            step_minutes = 5
            step_duration = timedelta(minutes=step_minutes)
        
        window_duration = timedelta(hours=window_hours)
        
        return self._run_rolling_window_backtest_sequential(
            full_dataset, start_time, end_time, timeframe,
            window_duration, step_duration
        )
    
    def _run_rolling_window_backtest_sequential(self, full_dataset: Dict, start_time: datetime,
                                              end_time: datetime, timeframe: str,
                                              window_duration: timedelta, step_duration: timedelta) -> List[Dict]:
        """
        Original sequential rolling window processing (renamed for clarity)
        This is the fallback when parallel processing fails
        """
        all_executed_orders = []
        
        # Get data timeframes for indexing
        ohlcv = full_dataset.get('ohlcv', pd.DataFrame())
        if ohlcv.empty:
            self.logger.error("No OHLCV data available for sequential window processing")
            return all_executed_orders
        
        # Ensure datetime index for proper slicing
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            self.logger.error("OHLCV data must have datetime index for rolling windows")
            return all_executed_orders
        
        # Calculate total iterations for progress tracking
        total_duration = end_time - start_time
        total_iterations = int(total_duration.total_seconds() / step_duration.total_seconds())
        
        # Start with first window
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
                self.logger.info(
                    f"🔄 Sequential Progress: {iteration:,}/{total_iterations:,} "
                    f"({progress_pct:.1f}%) - Window: {window_end.strftime('%Y-%m-%d %H:%M')}"
                )
            
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
                portfolio_state['current_time'] = current_time
                
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
                            order['timestamp'] = order['timestamp'].replace(tzinfo=current_time.tzinfo)
                        
                        execution_result = self._execute_order(order, windowed_dataset)
                        if execution_result:
                            all_executed_orders.append(execution_result)
                            
                            # Log significant trades
                            side = execution_result.get('side', 'UNKNOWN')
                            qty = execution_result.get('quantity', 0)
                            price = execution_result.get('price', 0)
                            signal_type = execution_result.get('signal_type', 'UNKNOWN')
                            
                            self.logger.info(
                                f"✅ {window_end.strftime('%Y-%m-%d %H:%M')} - "
                                f"{side} {qty:.6f} @ ${price:.2f} ({signal_type})"
                            )
                
            except Exception as e:
                self.logger.error(f"❌ Error processing window {window_end}: {e}")
            
            # Move to next window
            current_time += step_duration
        
        # Update stats
        self.parallel_stats['sequential_windows_processed'] = iteration
        
        self.logger.info(f"🎯 Sequential processing completed: {len(all_executed_orders)} total orders executed")
        return all_executed_orders
    
    def _log_parallel_performance(self, completed: int, failed: int, total: int):
        """
        Log parallel processing performance metrics
        
        Args:
            completed: Number of completed windows
            failed: Number of failed windows
            total: Total windows attempted
        """
        success_rate = (completed / total * 100) if total > 0 else 0
        
        # Estimate speedup (conservative estimate based on worker count)
        theoretical_speedup = min(self.max_workers, total / max(1, self.max_workers))
        actual_speedup = theoretical_speedup * (success_rate / 100) * 0.8  # Account for overhead
        
        self.parallel_stats['parallel_speedup_ratio'] = actual_speedup
        
        self.logger.info(f"📊 Parallel Performance Summary:")
        self.logger.info(f"   ✅ Completed: {completed:,} windows ({success_rate:.1f}% success)")
        self.logger.info(f"   ❌ Failed: {failed:,} windows")
        self.logger.info(f"   ⚡ Workers: {self.max_workers} threads")
        self.logger.info(f"   🚀 Estimated Speedup: {actual_speedup:.1f}x vs sequential")
        
        if failed > 0:
            self.logger.warning(f"   ⚠️ {failed} windows failed - check logs for details")
    
    def _run_streaming_backtest(self, symbol: str, start_time: datetime, 
                               end_time: datetime, timeframe: str) -> List[Dict]:
        """
        Memory-efficient streaming backtest for 1s data
        Processes data in small chunks to prevent OOM errors
        
        Args:
            symbol: Trading symbol
            start_time: Backtest start time
            end_time: Backtest end time  
            timeframe: Trading timeframe
            
        Returns:
            List of executed orders from all chunks
        """
        # Rolling window parameters optimized for 1s data
        if timeframe == '1s':
            window_hours = 1  # Smaller windows for 1s data (3600 data points)
            step_seconds = 1  # Step forward 1 SECOND for true 1s granularity
            step_duration = timedelta(seconds=step_seconds)
            self.logger.info(f"📊 1s mode: Using {window_hours}h windows with {step_seconds}s steps")
        else:
            window_hours = 2  # 2-hour windows for other timeframes
            step_minutes = 5  # Step forward 5 minutes
            step_duration = timedelta(minutes=step_minutes)
            self.logger.info(f"📊 Regular mode: Using {window_hours}h windows with {step_minutes}m steps")
        
        window_duration = timedelta(hours=window_hours)
        
        all_executed_orders = []
        
        # Calculate total iterations for progress tracking
        total_duration = end_time - start_time
        total_iterations = int(total_duration.total_seconds() / step_duration.total_seconds())
        
        if timeframe == '1s':
            self.logger.info(f"🔄 Streaming processing: {total_iterations:,} windows ({window_hours}h windows, 1s steps)")
        else:
            self.logger.info(f"🔄 Streaming processing: {total_iterations:,} windows ({window_hours}h windows, 5m steps)")
        self.logger.info(f"🎯 Target chunk size: {self.chunk_hours}h for optimal memory usage")
        
        # Start processing
        current_time = start_time + window_duration
        iteration = 0
        
        while current_time <= end_time:
            iteration += 1
            
            # Define window boundaries
            window_start = current_time - window_duration
            window_end = current_time
            
            # Memory monitoring every N iterations
            if iteration % self.memory_check_interval == 0:
                if not self._check_memory_usage(iteration, total_iterations):
                    self.logger.error("❌ Memory limit exceeded, stopping backtest")
                    break
            
            # Progress logging
            if iteration % 100 == 0 or iteration == total_iterations:
                progress_pct = (iteration / total_iterations) * 100
                self.logger.info(f"🔄 Progress: {iteration:,}/{total_iterations:,} ({progress_pct:.1f}%) - Window: {window_end.strftime('%Y-%m-%d %H:%M')}")
            
            try:
                # Load data for current window using streaming approach
                windowed_dataset = self._load_streaming_window_data(
                    symbol, window_start, window_end, timeframe
                )
                
                # Skip if insufficient data in window
                if not self._validate_windowed_data(windowed_dataset):
                    current_time += step_duration
                    continue
                
                # Add to rolling cache and remove old data
                self._manage_rolling_cache(windowed_dataset, window_end)
                
                # Get current portfolio state
                portfolio_state = self.portfolio.get_state()
                portfolio_state['available_leverage'] = self.leverage
                portfolio_state['current_time'] = current_time
                
                # Process strategy with windowed data
                strategy_result = self.strategy.process(windowed_dataset, portfolio_state)
                
                if strategy_result and 'orders' in strategy_result:
                    orders = strategy_result['orders']
                    
                    # Execute any orders generated for this window
                    for order in orders:
                        # Ensure order timestamp doesn't exceed current_time and is timezone-aware
                        if 'timestamp' not in order or order['timestamp'] > current_time:
                            order['timestamp'] = current_time
                        elif 'timestamp' in order and order['timestamp'].tzinfo is None and current_time.tzinfo is not None:
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
                
                # Explicit memory cleanup for 1s mode
                del windowed_dataset
                if iteration % 50 == 0:  # Force GC every 50 iterations
                    collected = gc.collect()
                    if collected > 0:
                        self.logger.debug(f"🗑️ Garbage collected {collected} objects")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing streaming window {window_end}: {e}")
                # Continue processing despite errors
            
            # Move to next window
            current_time += step_duration
        
        self.logger.info(f"🎯 Streaming processing completed: {len(all_executed_orders)} total orders executed")
        return all_executed_orders
    
    def _load_streaming_window_data(self, symbol: str, window_start: datetime, 
                                   window_end: datetime, timeframe: str) -> Dict:
        """
        Load data for a single window using streaming approach
        
        Args:
            symbol: Trading symbol
            window_start: Start of window
            window_end: End of window
            timeframe: Timeframe
            
        Returns:
            Dataset for current window
        """
        try:
            # Check cache first (LRU behavior)
            cache_key = f"{symbol}_{window_start.isoformat()}_{window_end.isoformat()}_{timeframe}"
            
            # Simple cache check (basic implementation)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                self.cache_hits += 1
                return cached_data
            
            self.cache_misses += 1
            
            # Load fresh data for this window
            dataset = self.data_pipeline.get_complete_dataset(
                symbol=symbol,
                start_time=window_start,
                end_time=window_end,
                timeframe=timeframe
            )
            
            # Add to cache
            self._add_to_cache(cache_key, dataset)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load streaming window data: {e}")
            return self._create_minimal_dataset_structure(symbol, timeframe, window_start, window_end)
    
    def _manage_rolling_cache(self, dataset: Dict, timestamp: datetime):
        """
        Manage rolling window cache with LRU eviction
        
        Args:
            dataset: Dataset to cache
            timestamp: Window timestamp
        """
        cache_entry = {
            'timestamp': timestamp,
            'data_points': dataset.get('metadata', {}).get('data_points', 0),
            'size_estimate': self._estimate_dataset_size(dataset)
        }
        
        # Add to rolling cache (deque automatically handles max size)
        self.data_window_cache.append(cache_entry)
    
    def _estimate_dataset_size(self, dataset: Dict) -> int:
        """
        Estimate memory size of dataset in MB
        
        Args:
            dataset: Dataset to estimate
            
        Returns:
            Estimated size in MB
        """
        size_mb = 0
        
        # Estimate DataFrame sizes
        for key in ['ohlcv', 'spot_volume', 'futures_volume']:
            df = dataset.get(key, pd.DataFrame())
            if not df.empty:
                size_mb += df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Estimate Series sizes
        for key in ['spot_cvd', 'futures_cvd', 'cvd_divergence']:
            series = dataset.get(key, pd.Series())
            if not series.empty:
                size_mb += series.memory_usage(deep=True) / (1024 * 1024)
        
        return int(size_mb)
    
    def _check_memory_usage(self, iteration: int, total_iterations: int) -> bool:
        """
        Check current memory usage and warn/pause if approaching limits
        
        Args:
            iteration: Current iteration number
            total_iterations: Total iterations
            
        Returns:
            True if safe to continue, False if should stop
        """
        try:
            # Get current memory usage
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / (1024 * 1024)
            current_memory_gb = current_memory_mb / 1024
            
            # Update peak memory tracking
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb
            
            # Check against limit
            memory_usage_pct = (current_memory_gb / self.max_memory_gb) * 100
            
            if current_memory_gb > self.max_memory_gb * 0.9:  # 90% warning threshold
                self.memory_warnings += 1
                self.logger.warning(f"⚠️ Memory usage high: {current_memory_gb:.1f}GB ({memory_usage_pct:.1f}%) - Iteration {iteration}")
                
                # Force aggressive garbage collection
                collected = gc.collect()
                self.logger.info(f"🗑️ Emergency GC collected {collected} objects")
                
                # Check again after GC
                memory_info_after = self.process.memory_info()
                memory_after_gb = (memory_info_after.rss / (1024 * 1024)) / 1024
                
                if memory_after_gb > self.max_memory_gb:
                    self.logger.error(f"❌ Memory limit exceeded: {memory_after_gb:.1f}GB > {self.max_memory_gb}GB")
                    return False
            
            # Log memory status periodically
            if iteration % (self.memory_check_interval * 2) == 0:
                progress_pct = (iteration / total_iterations) * 100
                self.logger.info(f"💾 Memory: {current_memory_gb:.1f}GB ({memory_usage_pct:.1f}%) - Cache: {self.cache_hits}/{self.cache_hits + self.cache_misses} hits")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")
            return True  # Continue on memory check errors
    
    def _cleanup_memory(self):
        """
        Final memory cleanup after backtest completion
        """
        self.logger.info("🧹 Performing final memory cleanup...")
        
        # Clear caches
        self.data_window_cache.clear()
        
        # Clear pipeline caches if available
        if hasattr(self.data_pipeline, 'clear_cache'):
            self.data_pipeline.clear_cache()
        
        # Force garbage collection
        collected = gc.collect()
        self.logger.info(f"🗑️ Final cleanup collected {collected} objects")
    
    def _log_memory_statistics(self):
        """
        Log final memory usage statistics for 1s mode
        """
        try:
            final_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            final_memory_gb = final_memory_mb / 1024
            
            cache_hit_rate = (self.cache_hits / max(self.cache_hits + self.cache_misses, 1)) * 100
            
            self.logger.info("📊 Memory Statistics Summary:")
            self.logger.info(f"   Peak Memory: {self.peak_memory_mb:.1f}MB ({self.peak_memory_mb/1024:.1f}GB)")
            self.logger.info(f"   Final Memory: {final_memory_mb:.1f}MB ({final_memory_gb:.1f}GB)")
            self.logger.info(f"   Memory Limit: {self.max_memory_gb}GB")
            self.logger.info(f"   Memory Warnings: {self.memory_warnings}")
            self.logger.info(f"   Cache Hit Rate: {cache_hit_rate:.1f}% ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
            self.logger.info(f"   Rolling Cache Size: {len(self.data_window_cache)}/{self.rolling_window_size}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log memory statistics: {e}")
    
    def _log_final_parallel_stats(self):
        """
        Log final parallel processing statistics
        """
        try:
            stats = self.parallel_stats
            parallel_processed = stats.get('parallel_windows_processed', 0)
            sequential_processed = stats.get('sequential_windows_processed', 0)
            total_processed = parallel_processed + sequential_processed
            speedup = stats.get('parallel_speedup_ratio', 1.0)
            errors = stats.get('thread_pool_errors', 0)
            
            self.logger.info("🚀 Parallel Processing Final Statistics:")
            self.logger.info(f"   Total Windows: {total_processed:,}")
            self.logger.info(f"   Parallel: {parallel_processed:,} ({parallel_processed/max(total_processed,1)*100:.1f}%)")
            self.logger.info(f"   Sequential: {sequential_processed:,} ({sequential_processed/max(total_processed,1)*100:.1f}%)")
            self.logger.info(f"   Speedup Achieved: {speedup:.1f}x")
            self.logger.info(f"   Thread Errors: {errors}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log parallel statistics: {e}")
    
    def _generate_performance_report(self, result: Dict):
        """Generate comprehensive performance report"""
        try:
            self.logger.info("📊 Generating performance report...")
            
            # Generate performance report
            report = self.performance_monitor.analyzer.generate_performance_report()
            
            # Log key performance metrics
            summary = report.get('summary', {})
            self.logger.info(f"🔍 Performance Summary:")
            self.logger.info(f"   Total operations tracked: {summary.get('total_operations_tracked', 0)}")
            self.logger.info(f"   Total measurements: {summary.get('total_measurements', 0)}")
            self.logger.info(f"   Slow operations detected: {summary.get('slow_operations_detected', 0)}")
            
            # Log system performance
            system_perf = report.get('system_performance', {})
            if not system_perf.get('error'):
                cpu_stats = system_perf.get('cpu_usage_percent', {})
                memory_stats = system_perf.get('memory_usage_mb', {})
                self.logger.info(f"   CPU usage: avg={cpu_stats.get('mean', 0):.1f}%, max={cpu_stats.get('max', 0):.1f}%")
                self.logger.info(f"   Memory usage: avg={memory_stats.get('mean', 0):.0f}MB, max={memory_stats.get('max', 0):.0f}MB")
            
            # Log window processing performance
            window_perf = report.get('window_processing', {})
            if not window_perf.get('error'):
                processing_stats = window_perf.get('processing_time_stats_ms', {})
                self.logger.info(f"   Window processing: avg={processing_stats.get('mean', 0):.1f}ms, p95={processing_stats.get('p95', 0):.1f}ms")
                self.logger.info(f"   Windows processed: {window_perf.get('total_windows_processed', 0)}")
                self.logger.info(f"   Throughput: {window_perf.get('throughput_windows_per_minute', 0):.1f} windows/min")
            
            # Log bottlenecks
            bottlenecks = report.get('bottlenecks', [])
            if bottlenecks:
                self.logger.warning(f"⚠️ Performance bottlenecks detected:")
                for bottleneck in bottlenecks[:3]:  # Top 3
                    self.logger.warning(f"   - {bottleneck['type']}: {bottleneck.get('operation_name', 'System')} ({bottleneck.get('severity', 'unknown')})")
            else:
                self.logger.info("✅ No significant performance bottlenecks detected")
            
            # Log recommendations
            recommendations = report.get('recommendations', [])
            if recommendations:
                self.logger.info("💡 Performance recommendations:")
                for rec in recommendations[:3]:  # Top 3
                    self.logger.info(f"   - {rec}")
            
            # Save detailed report
            try:
                import json
                from pathlib import Path
                
                report_dir = Path("data/performance_reports")
                report_dir.mkdir(parents=True, exist_ok=True)
                
                report_file = report_dir / f"backtest_performance_{int(time.time())}.json"
                
                # Add backtest metadata to report
                report['backtest_metadata'] = {
                    'symbol': result.get('symbol'),
                    'timeframe': result.get('timeframe'),
                    'initial_balance': result.get('initial_balance'),
                    'final_balance': result.get('final_balance'),
                    'total_return': result.get('total_return'),
                    'total_trades': result.get('total_trades'),
                    'enable_1s_mode': self.enable_1s_mode,
                    'enable_parallel': self.enable_parallel,
                    'max_workers': self.max_workers
                }
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                self.logger.info(f"📄 Detailed performance report saved: {report_file}")
                
                # Generate performance charts if possible
                chart_path = self.performance_monitor.generate_performance_chart("memory", str(report_dir / f"performance_chart_{int(time.time())}.png"))
                if chart_path:
                    self.logger.info(f"📊 Performance chart saved: {chart_path}")
                
            except Exception as e:
                self.logger.warning(f"Could not save detailed performance report: {e}")
            
            # Add 1s vs regular mode comparison if applicable
            comparison = self.performance_monitor.get_1s_vs_regular_comparison()
            if comparison.get('data_available'):
                self.logger.info("🔄 1s vs Regular Mode Performance Comparison:")
                for mode, stats in comparison.items():
                    if isinstance(stats, dict) and 'avg_duration_ms' in stats:
                        self.logger.info(f"   {mode}: avg={stats['avg_duration_ms']:.1f}ms, throughput={stats['throughput_ops_per_min']:.1f} ops/min")
                
                for rec in comparison.get('recommendations', []):
                    self.logger.info(f"   💡 {rec}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Simple cache lookup (basic implementation)
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached dataset or None
        """
        # Basic implementation - in production this would be more sophisticated
        return None
    
    def _add_to_cache(self, cache_key: str, dataset: Dict):
        """
        Add dataset to cache (basic implementation)
        
        Args:
            cache_key: Cache key
            dataset: Dataset to cache
        """
        # Basic implementation - in production this would implement actual caching
        pass
    
    def _create_minimal_dataset_structure(self, symbol: str, timeframe: str, 
                                         start_time: datetime, end_time: datetime) -> Dict:
        """
        Create minimal dataset structure for streaming mode
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            
        Returns:
            Minimal dataset structure
        """
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': end_time,
            'ohlcv': pd.DataFrame(),
            'spot_volume': pd.DataFrame(),
            'futures_volume': pd.DataFrame(),
            'spot_cvd': pd.Series(dtype=float),
            'futures_cvd': pd.Series(dtype=float),
            'cvd_divergence': pd.Series(dtype=float),
            'markets': {'spot': [], 'perp': []},
            'data_source': 'streaming',
            'metadata': {
                'spot_markets_count': 0,
                'futures_markets_count': 0,
                'data_points': 0,
                'streaming_mode': True
            }
        }
    
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
            position_id = order.get('position_id', 'unknown')
            self.logger.info(f"🚪 Processing EXIT order for {symbol} (position_id: {position_id})")
            
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
                
                # Debug: Check current positions before trying to close
                current_positions = self.portfolio.get_state()['positions']
                matching = [p for p in current_positions if p['symbol'] == symbol]
                if not matching:
                    self.logger.debug(f"No positions exist for {symbol} to close")
                else:
                    self.logger.debug(f"Found {len(matching)} positions for {symbol}:")
                    for pos in matching:
                        self.logger.debug(f"  Position: side={pos.get('side')}, qty={pos.get('quantity')}, id={pos.get('id')}")
                
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
            
            # Get CVD baselines and entry analysis for position
            spot_cvd_entry = None
            futures_cvd_entry = None
            if 'spot_cvd' in dataset and 'futures_cvd' in dataset:
                spot_cvd = dataset['spot_cvd']
                futures_cvd = dataset['futures_cvd']
                if not spot_cvd.empty and not futures_cvd.empty:
                    spot_cvd_entry = float(spot_cvd.iloc[-1])
                    futures_cvd_entry = float(futures_cvd.iloc[-1])
            
            # Get entry analysis from order (scoring results)
            entry_analysis = order.get('scoring_result', {})
            
            # Execute through portfolio with leverage and tracking info
            success = False
            if side == 'BUY':
                success = self.portfolio.open_long_position(
                    symbol=symbol,
                    quantity=effective_quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_id=trade_id,
                    signal_id=signal_id,
                    entry_analysis=entry_analysis,
                    spot_cvd_entry=spot_cvd_entry,
                    futures_cvd_entry=futures_cvd_entry
                )
            elif side == 'SELL':
                success = self.portfolio.open_short_position(
                    symbol=symbol,
                    quantity=effective_quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_id=trade_id,
                    signal_id=signal_id,
                    entry_analysis=entry_analysis,
                    spot_cvd_entry=spot_cvd_entry,
                    futures_cvd_entry=futures_cvd_entry
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
    parser = argparse.ArgumentParser(description='SqueezeFlow Backtest Engine with 1s Data Support')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--start-date', default='2024-08-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-08-04', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='5m', help='Timeframe (1s, 1m, 5m, 15m, etc.)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--leverage', type=float, default=1.0, help='Trading leverage (default: 1.0)')
    parser.add_argument('--strategy', default='SqueezeFlowStrategy', help='Strategy class name')
    parser.add_argument('--enable-1s-mode', action='store_true', help='Enable 1s mode for memory-efficient processing')
    parser.add_argument('--max-memory-gb', type=float, default=8.0, help='Maximum memory limit in GB (default: 8.0)')
    parser.add_argument('--max-workers', type=int, help='Maximum parallel workers (auto-detect if not specified)')
    parser.add_argument('--disable-parallel', action='store_true', help='Disable parallel processing (force sequential)')
    
    args = parser.parse_args()
    
    # Create engine with 1s mode and parallel processing support
    engine = BacktestEngine(
        initial_balance=args.balance, 
        leverage=args.leverage,
        enable_1s_mode=args.enable_1s_mode,
        max_memory_gb=args.max_memory_gb,
        max_workers=args.max_workers,
        enable_parallel=not args.disable_parallel
    )
    
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
    if args.enable_1s_mode:
        print(f"1s Mode: Enabled (Memory Limit: {args.max_memory_gb}GB)")
    if not args.disable_parallel:
        print(f"Parallel Processing: Enabled (Workers: {engine.max_workers})")
    else:
        print(f"Parallel Processing: Disabled (Sequential Mode)")