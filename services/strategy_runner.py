#!/usr/bin/env python3
"""
Strategy Runner Service - Phase 4 Live Trading Service

Runs the SqueezeFlow strategy on real-time data and converts orders to signals
for FreqTrade execution. This service bridges backtest strategy code with live trading.

Architecture:
- Uses existing SqueezeFlowStrategy from /strategies/squeezeflow/strategy.py
- Uses existing DataPipeline from /data/pipeline.py for data loading
- Uses existing CVDCalculator from /data/processors/cvd_calculator.py
- Calculates CVD in real-time (unlike backtest which provides pre-calculated)
- Publishes signals to Redis for FreqTrade consumption
- Stores signals in InfluxDB for history and analysis
"""

import asyncio
import redis
import json
import uuid
import logging
import pandas as pd
import hashlib
import threading
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from influxdb import InfluxDBClient
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeezeflow.strategy import SqueezeFlowStrategy
from data.pipeline import DataPipeline
from data.processors.cvd_calculator import CVDCalculator
from services.config.unified_config import ConfigManager
from services.signal_validator import SignalValidator, BatchSignalValidator, ValidationResult
from services.influx_signal_manager import InfluxSignalManager, create_signal_manager_from_config
from services.freqtrade_client import FreqTradeAPIClient, create_freqtrade_client_from_config
from strategies.squeezeflow.baseline_manager import CVDBaselineManager, create_cvd_baseline_manager_from_config


class StrategyRunner:
    """Live trading service that runs SqueezeFlow strategy on real-time data"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize Strategy Runner Service
        
        Args:
            config_manager: Configuration manager (creates default if None)
        """
        # Initialize configuration
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize core components
        self.strategy = SqueezeFlowStrategy()
        self.data_pipeline = DataPipeline()
        self.cvd_calculator = CVDCalculator()
        
        # Initialize connections (lazy loading)
        self._redis_client: Optional[redis.Redis] = None
        self._influx_client: Optional[InfluxDBClient] = None
        self._signal_manager: Optional[InfluxSignalManager] = None
        self._freqtrade_client: Optional[FreqTradeAPIClient] = None
        self._cvd_baseline_manager: Optional[CVDBaselineManager] = None
        
        # Set up CVD baseline manager integration with strategy (after attribute initialization)
        self._setup_strategy_cvd_integration()
        
        # Service state
        self.is_running = False
        self.last_signals: Dict[str, datetime] = {}  # Symbol -> last signal time
        self.cycle_count = 0
        
        # Add periodic cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Position state tracking for logging spam reduction
        self._previous_portfolio_states: Dict[str, str] = {}
        self._previous_position_states: Dict[str, str] = {}
        
        # Performance monitoring
        self.performance_stats = {
            'cycles_completed': 0,
            'signals_generated': 0,
            'signals_published': 0,
            'signals_failed': 0,
            'signals_duplicate': 0,
            'signals_rate_limited': 0,
            'batch_published': 0,
            'errors_encountered': 0,
            'last_cycle_duration': 0.0,
            'avg_cycle_duration': 0.0,
            'redis_publish_time': 0.0,
            'validation_time': 0.0
        }
        
        # Signal validation and deduplication
        validator_config = {
            'max_signals_per_minute': self.config.max_concurrent_signals * 2,
            'max_signals_per_symbol_per_hour': 10,
            'signal_cooldown_minutes': self.config.signal_cooldown_minutes,
            'cleanup_interval_hours': 12
        }
        self.signal_validator = SignalValidator(validator_config)
        self.batch_validator = BatchSignalValidator(self.signal_validator)
        
        # Redis pub/sub for real-time signal distribution
        self._pubsub_client: Optional[redis.Redis] = None
        self._signal_subscribers: Dict[str, List] = {}  # Channel -> callbacks
        
        # Batch publishing configuration
        self.batch_config = {
            'enabled': getattr(self.config, 'enable_batch_publishing', True),
            'max_batch_size': getattr(self.config, 'max_batch_size', 10),
            'batch_timeout_seconds': getattr(self.config, 'batch_timeout_seconds', 5),
            'flush_on_shutdown': True
        }
        
        # Signal batching
        self._signal_batch: List[Dict] = []
        self._batch_lock = threading.Lock()
        self._last_batch_flush = datetime.now()
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='signal_publisher')
        
        # Signal expiry management
        self._signal_expiry_tracker: Dict[str, datetime] = {}
        
        # Monitoring metrics
        self._redis_metrics = {
            'total_publishes': 0,
            'successful_publishes': 0,
            'failed_publishes': 0,
            'avg_publish_time': 0.0,
            'pubsub_subscribers': 0,
            'expired_signals_cleaned': 0
        }
        
        self.logger.info("Strategy Runner Service initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('strategy_runner')
        self.logger.info(f"Logging configured at level: {self.config.log_level}")
    
    def _setup_strategy_cvd_integration(self):
        """Setup CVD baseline manager integration with strategy"""
        # This will be called lazily when CVD baseline manager is available
        try:
            if hasattr(self, '_strategy_cvd_setup_done'):
                return
            
            # Check if CVD baseline tracking is enabled
            if not self.config.enable_cvd_baseline_tracking:
                self.logger.info("CVD baseline tracking disabled in configuration")
                return
                
            # Try to get CVD manager (will create if config allows)
            cvd_manager = self.cvd_baseline_manager
            if cvd_manager and hasattr(self.strategy, 'set_cvd_baseline_manager'):
                self.strategy.set_cvd_baseline_manager(cvd_manager)
                self.logger.info("✅ CVD baseline manager successfully integrated with strategy")
                self._strategy_cvd_setup_done = True
            elif not cvd_manager:
                self.logger.debug("CVD baseline manager not yet initialized, will retry during first property access")
            else:
                self.logger.warning("Strategy does not support CVD baseline manager integration")
        except Exception as e:
            self.logger.warning(f"CVD baseline integration setup delayed: {e}")
            # This is expected during initialization - the manager will be created lazily
    
    @property
    def redis_client(self) -> redis.Redis:
        """Lazy loading Redis client"""
        if self._redis_client is None:
            redis_config = self.config_manager.get_redis_config()
            # Add connection pool for better performance
            from redis.connection import ConnectionPool
            pool = ConnectionPool(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                max_connections=20,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self._redis_client = redis.Redis(connection_pool=pool, decode_responses=True)
            self.logger.info("Redis client initialized with connection pooling")
        return self._redis_client
    
    @property
    def pubsub_client(self) -> redis.Redis:
        """Separate Redis client for pub/sub operations"""
        if self._pubsub_client is None:
            redis_config = self.config_manager.get_redis_config()
            self._pubsub_client = redis.Redis(**redis_config)
            self.logger.info("Redis pub/sub client initialized")
        return self._pubsub_client
    
    @property
    def influx_client(self) -> InfluxDBClient:
        """Lazy loading InfluxDB client"""
        if self._influx_client is None:
            influx_config = self.config_manager.get_influx_config()
            self._influx_client = InfluxDBClient(**influx_config)
            self.logger.info("InfluxDB client initialized")
        return self._influx_client
    
    @property
    def signal_manager(self) -> InfluxSignalManager:
        """Lazy loading Enhanced InfluxDB Signal Manager"""
        if self._signal_manager is None:
            self._signal_manager = create_signal_manager_from_config(self.config_manager)
            self.logger.info("Enhanced InfluxDB Signal Manager initialized")
        return self._signal_manager
    
    @property
    def freqtrade_client(self) -> Optional[FreqTradeAPIClient]:
        """Lazy loading FreqTrade API client"""
        if self._freqtrade_client is None and self.config.enable_freqtrade_integration:
            try:
                self._freqtrade_client = create_freqtrade_client_from_config(self.config_manager)
                self.logger.info("FreqTrade API client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize FreqTrade client: {e}")
                return None
        return self._freqtrade_client
    
    @property
    def cvd_baseline_manager(self) -> Optional[CVDBaselineManager]:
        """Lazy loading CVD baseline manager"""
        if self._cvd_baseline_manager is None and self.config.enable_cvd_baseline_tracking:
            try:
                self._cvd_baseline_manager = create_cvd_baseline_manager_from_config(self.config_manager)
                self.logger.info("✅ CVD Baseline Manager initialized successfully")
                
                # Setup strategy integration now that manager is available
                # Reset the setup flag to allow retry
                if hasattr(self, '_strategy_cvd_setup_done'):
                    delattr(self, '_strategy_cvd_setup_done')
                self._setup_strategy_cvd_integration()
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize CVD baseline manager: {e}")
                # Don't store None - allow retries on next access
                return None
        return self._cvd_baseline_manager
    
    async def start(self):
        """Start the strategy runner service"""
        
        self.logger.info("Starting Strategy Runner Service...")
        
        # Test connections
        if not await self._test_connections():
            raise RuntimeError("Failed to establish required connections")
        
        # Get trading symbols from FreqTrade config
        symbols = self.config_manager.get_freqtrade_pairs()
        self.logger.info(f"Trading symbols loaded: {symbols}")
        
        # Start periodic cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Start main execution loop
        self.is_running = True
        self.logger.info(f"Service started - Running every {self.config.run_interval_seconds}s")
        
        try:
            while self.is_running:
                cycle_start = datetime.now()
                
                # Run strategy cycle
                await self._run_strategy_cycle(symbols)
                
                # Update performance stats
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                self._update_performance_stats(cycle_duration)
                
                # Log progress
                if self.cycle_count % 10 == 0:  # Every 10 cycles
                    self._log_performance_stats()
                
                # Periodic batch flush check
                if self.batch_config['enabled']:
                    await self._check_batch_flush()
                
                # Sleep until next cycle
                await asyncio.sleep(self.config.run_interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the strategy runner service"""
        
        self.logger.info("Stopping Strategy Runner Service...")
        self.is_running = False
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Flush any pending batch signals
        if self._signal_batch:
            await self._flush_signal_batch(force=True)
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Close connections
        if self._redis_client:
            self._redis_client.close()
        if self._pubsub_client:
            self._pubsub_client.close()
        if self._influx_client:
            self._influx_client.close()
        if self._freqtrade_client:
            self._freqtrade_client.close()
        # Note: signal_manager uses the same influx_client, so no separate close needed
        # Note: cvd_baseline_manager uses the same redis_client, so no separate close needed
        
        self.logger.info("Strategy Runner Service stopped")
    
    async def _test_connections(self) -> bool:
        """Test all required connections"""
        
        try:
            # Test Redis main client
            self.redis_client.ping()
            self.logger.info("Redis main client connection successful")
            
            # Test Redis pub/sub client
            self.pubsub_client.ping()
            self.logger.info("Redis pub/sub client connection successful")
            
            # Test InfluxDB
            self.influx_client.ping()
            self.logger.info("InfluxDB connection successful")
            
            # Test FreqTrade API (optional)
            if self.config.enable_freqtrade_integration and self.freqtrade_client:
                connection_ok, message = self.freqtrade_client.test_connection()
                if connection_ok:
                    self.logger.info("FreqTrade API connection successful")
                else:
                    self.logger.warning(f"FreqTrade API connection failed: {message}")
                    # Don't fail overall test - FreqTrade might not be running yet
            
            # Initialize pub/sub channels
            await self._initialize_pubsub_channels()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _run_strategy_cycle(self, symbols: List[str]):
        """Run one complete strategy cycle for all symbols"""
        
        self.cycle_count += 1
        self.logger.debug(f"Starting cycle #{self.cycle_count} for {len(symbols)} symbols")
        
        # Process symbols (limit to max per cycle)
        symbols_to_process = symbols[:self.config.max_symbols_per_cycle]
        
        if self.config.enable_parallel_processing:
            # Process symbols in parallel
            tasks = [self._process_symbol(symbol) for symbol in symbols_to_process]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any exceptions from parallel processing
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing {symbols_to_process[i]}: {result}")
        else:
            # Process symbols sequentially
            for symbol in symbols_to_process:
                await self._process_symbol(symbol)
        
        self.logger.debug(f"Completed cycle #{self.cycle_count}")
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol through the complete strategy pipeline"""
        
        try:
            # Check signal cooldown
            if self._is_symbol_in_cooldown(symbol):
                self.logger.debug(f"{symbol}: Skipping due to cooldown")
                return
            
            # Load real-time data with 1s data support
            dataset = await self._load_symbol_data(symbol)
            if not dataset or not self._validate_dataset(dataset):
                data_source = dataset.get('data_source', 'unknown') if dataset else 'no_data'
                self.logger.warning(f"{symbol}: Invalid or insufficient data (source: {data_source})")
                return
            
            # Get real portfolio state from FreqTrade or use mock data
            portfolio_state = await self._get_portfolio_state()
            
            # CRITICAL POSITION AWARENESS FIX: Validate portfolio data reliability
            if not self._has_valid_position_data(portfolio_state):
                self.logger.error(f"{symbol}: POSITION AWARENESS FAILED - Cannot execute strategy without real position data")
                return
            
            # Get portfolio state information
            total_value = portfolio_state.get('total_value', 0)
            positions = portfolio_state.get('positions', [])
            source = portfolio_state.get('source', 'freqtrade')
            balance_source = portfolio_state.get('balance_source', 'unknown')
            
            # CRITICAL: Get symbol-specific positions for entry/exit mode switching
            existing_positions = self._get_symbol_positions(portfolio_state, symbol)
            
            # Store previous state for change detection
            current_portfolio_info = f"{len(positions)}_${total_value:.2f}_{source}_{balance_source}"
            previous_key = f"portfolio_info_{symbol}"
            
            # Log only on changes or every 20 cycles to reduce noise
            should_log = (
                self.cycle_count % 20 == 0 or  # Every 20 cycles
                not hasattr(self, '_previous_portfolio_states') or
                self._previous_portfolio_states.get(previous_key) != current_portfolio_info
            )
            
            if should_log:
                if not hasattr(self, '_previous_portfolio_states'):
                    self._previous_portfolio_states = {}
                    
                self._previous_portfolio_states[previous_key] = current_portfolio_info
                self.logger.info(f"{symbol}: Portfolio - {len(positions)} positions, ${total_value:.2f} total (source: {source}, balance: {balance_source})")
            
            # Critical check: If total_value is 0, this will cause position size to be 0
            if total_value <= 0:
                self.logger.error(f"{symbol}: CRITICAL - Portfolio total_value is {total_value}, this will cause 0 position size! (balance_source: {balance_source})")
            
            # Log mode awareness only when positions change
            mode_key = f"mode_{symbol}"
            current_mode = f"EXIT_{len(existing_positions)}" if existing_positions else "ENTRY_0"
            previous_mode = self._previous_position_states.get(mode_key)
            
            if previous_mode != current_mode:
                self._previous_position_states[mode_key] = current_mode
                if existing_positions:
                    self.logger.info(f"{symbol}: Mode changed to EXIT - {len(existing_positions)} open positions detected")
                else:
                    self.logger.info(f"{symbol}: Mode changed to ENTRY - no open positions, seeking entry opportunities")
            else:
                # Mode unchanged - minimal logging
                if existing_positions:
                    self.logger.debug(f"{symbol}: EXIT MODE (no changes)")
                else:
                    self.logger.debug(f"{symbol}: ENTRY MODE (no changes)")
            
            # Call strategy - it handles entry/exit mode internally based on portfolio_state
            strategy_result = self.strategy.process(dataset, portfolio_state)
            
            # Process any orders generated
            orders = strategy_result.get('orders', [])
            if orders:
                signals_processed = await self._convert_orders_to_signals(symbol, orders, strategy_result, dataset)
                mode = "EXIT" if existing_positions else "ENTRY"
                self.logger.info(f"{symbol}: {mode} MODE - Generated {len(orders)} orders, processed {signals_processed} signals")
            else:
                mode = "EXIT" if existing_positions else "ENTRY"
                self.logger.debug(f"{symbol}: {mode} MODE - No signals generated")
                
            # Check if we need to flush batched signals
            if self.batch_config['enabled']:
                await self._check_batch_flush()
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            self.performance_stats['errors_encountered'] += 1
    
    async def _load_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Load real-time data for symbol with CVD calculations using efficient 1s data"""
        
        try:
            # Calculate time range with limited lookback for real-time efficiency
            end_time = datetime.now()
            
            # For real-time trading, use sufficient lookback for multi-timeframe analysis
            # 4 hours provides enough data for all timeframes (1m, 5m, 15m, 30m, 1h, 4h)
            # We have 24+ hours of 1s data available in aggr_1s retention policy
            max_lookback_minutes = 240  # 4 hours for proper multi-timeframe analysis
            start_time = end_time - timedelta(minutes=max_lookback_minutes)
            
            # Use async data loading with 1-second data preference
            dataset = await self.data_pipeline.get_complete_dataset_async(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                timeframe=self.config.default_timeframe,
                prefer_1s_data=True,
                max_lookback_minutes=max_lookback_minutes
            )
            
            if not dataset or dataset.get('ohlcv', pd.DataFrame()).empty:
                self.logger.warning(f"No data loaded for {symbol}")
                return None
            
            # Log data source and efficiency info
            data_source = dataset.get('data_source', 'unknown')
            data_points = dataset.get('metadata', {}).get('data_points', 0)
            
            if data_source == '1s_aggregated':
                self.logger.debug(f"{symbol}: Using 1s aggregated data ({data_points} bars, {max_lookback_minutes}min lookback)")
            else:
                self.logger.debug(f"{symbol}: Using {data_source} data ({data_points} bars)")
            
            # The data pipeline automatically calculates CVD using CVDCalculator
            # This is different from backtest where CVD is pre-calculated
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _validate_dataset(self, dataset: Dict) -> bool:
        """Validate dataset quality for trading"""
        
        validation = self.data_pipeline.validate_data_quality(dataset)
        
        # Require high quality data for live trading
        required_validations = [
            'has_price_data',
            'has_spot_cvd',
            'has_futures_cvd',
            'sufficient_data_points'
        ]
        
        for validation_key in required_validations:
            if not validation.get(validation_key, False):
                self.logger.warning(f"Dataset validation failed: {validation_key}")
                return False
        
        return True
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state from FreqTrade or fallback to mock data
        
        Returns:
            Portfolio state dict compatible with strategy interface
        """
        try:
            if self.config.enable_freqtrade_integration and self.freqtrade_client:
                # Test FreqTrade connection first
                connection_ok, message = self.freqtrade_client.test_connection()
                if not connection_ok:
                    self.logger.error(f"FreqTrade connection failed: {message}")
                    return self._get_fallback_portfolio_state()
                
                # Get real portfolio state from FreqTrade
                portfolio_state = self.freqtrade_client.get_portfolio_state()
                
                # Check for error in response
                if portfolio_state and 'error' in portfolio_state:
                    self.logger.error(f"FreqTrade API error: {portfolio_state['error']}")
                    return self._get_fallback_portfolio_state()
                
                # Check if we have valid portfolio state with non-zero total_value
                total_value = portfolio_state.get('total_value', 0) if portfolio_state else 0
                positions = portfolio_state.get('positions', []) if portfolio_state else []
                
                if portfolio_state and total_value > 0:
                    # Only log detailed success info on first success or errors
                    balance_source = portfolio_state.get('balance_source', 'unknown')
                    self.logger.debug(f"✅ FreqTrade portfolio retrieved: {len(positions)} positions, ${total_value:.2f} total value (source: {balance_source})")
                    return portfolio_state
                else:
                    self.logger.warning(f"❌ FreqTrade returned invalid portfolio (total_value: ${total_value})")
                    self.logger.debug(f"Raw FreqTrade response: {portfolio_state}")
                    return self._get_fallback_portfolio_state()
                    
            else:
                self.logger.warning("FreqTrade integration disabled or client unavailable")
                return self._get_fallback_portfolio_state()
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio state: {e}")
            return self._get_fallback_portfolio_state()
    
    def _get_fallback_portfolio_state(self) -> Dict[str, Any]:
        """
        Get fallback portfolio state for testing/demo purposes
        
        Returns:
            Mock portfolio state
        """
        return {
            'positions': [], 
            'total_value': 100000, 
            'cash': 100000,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback_mock',
            'balance_source': 'fallback_mock'
        }
    
    def _has_valid_position_data(self, portfolio_state: Dict[str, Any]) -> bool:
        """
        Validate that portfolio state contains reliable position data
        
        Args:
            portfolio_state: Portfolio state dict
            
        Returns:
            bool: True if portfolio data is reliable for trading decisions
        """
        source = portfolio_state.get('source', 'unknown')
        balance_source = portfolio_state.get('balance_source', 'unknown')
        total_value = portfolio_state.get('total_value', 0)
        
        # Check if using fallback mock data
        if source == 'fallback_mock' or balance_source == 'fallback_mock':
            self.logger.warning("POSITION AWARENESS: Using fallback mock portfolio - no position awareness")
            return False
        
        # Check for error indicators
        if 'error' in portfolio_state:
            self.logger.error(f"POSITION AWARENESS: Portfolio state contains error: {portfolio_state['error']}")
            return False
        
        # Additional reliability checks
        if total_value <= 0 and balance_source not in ['real_zero_balance']:
            self.logger.warning(f"POSITION AWARENESS: Suspicious zero balance (source: {balance_source})")
            return False
            
        return True
    
    def _get_symbol_positions(self, portfolio_state: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific symbol from portfolio state
        
        Args:
            portfolio_state: Portfolio state dict
            symbol: Base symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            List of positions for the symbol
        """
        positions = portfolio_state.get('positions', [])
        
        # Convert symbol format for matching (BTC -> BTCUSDT, BTCUSD, etc.)
        symbol_variations = [
            symbol,                    # Base symbol: BTC
            f"{symbol}USDT",          # Trading pair: BTCUSDT
            f"{symbol}USD",           # Alternative: BTCUSD
            f"{symbol}/USDT",         # FreqTrade format: BTC/USDT
            f"{symbol}/USDT:USDT",    # FreqTrade futures: BTC/USDT:USDT
            f"{symbol}/USD:USD"       # FreqTrade futures alternative
        ]
        
        symbol_positions = []
        for position in positions:
            pos_symbol = position.get('symbol', '')
            pos_quantity = position.get('quantity', 0)
            
            # Check if position matches any symbol variation and has non-zero quantity
            if any(variation.upper() in pos_symbol.upper() for variation in symbol_variations) and pos_quantity != 0:
                symbol_positions.append(position)
        
        # Create position state summary for change detection
        if symbol_positions:
            position_summary = ';'.join([
                f"{pos.get('symbol')}_{pos.get('side')}_{pos.get('quantity')}_{pos.get('entry_price')}"
                for pos in symbol_positions
            ])
            position_key = f"positions_{symbol}"
            
            # Only log if positions have changed
            previous_positions = self._previous_position_states.get(position_key)
            if previous_positions != position_summary:
                self._previous_position_states[position_key] = position_summary
                self.logger.info(f"{symbol}: Position changed - Found {len(symbol_positions)} existing positions")
                for pos in symbol_positions:
                    self.logger.info(f"  Position: {pos.get('symbol')} {pos.get('side')} {pos.get('quantity')} @ {pos.get('entry_price')}")
            else:
                # Positions unchanged - minimal logging
                self.logger.debug(f"{symbol}: {len(symbol_positions)} positions unchanged")
        else:
            position_key = f"positions_{symbol}"
            previous_positions = self._previous_position_states.get(position_key)
            if previous_positions is not None:  # Had positions before, now has none
                self._previous_position_states[position_key] = None
                self.logger.info(f"{symbol}: Position changed - No positions (previously had positions)")
            else:
                self.logger.debug(f"{symbol}: No existing positions found")
        
        return symbol_positions
    
    async def _convert_orders_to_signals(self, symbol: str, orders: List[Dict], 
                                       strategy_result: Dict, dataset: Dict) -> int:
        """Convert strategy orders to Redis signals with comprehensive validation"""
        
        signals_processed = 0
        signals_to_process = []
        
        # Create signals from orders
        for order in orders:
            try:
                signal = self._create_signal_from_order(symbol, order, strategy_result)
                signals_to_process.append(signal)
            except Exception as e:
                self.logger.error(f"Error creating signal from order: {e}")
                self.performance_stats['errors_encountered'] += 1
        
        if not signals_to_process:
            return 0
        
        # Batch validate signals
        validation_start = datetime.now()
        if len(signals_to_process) > 1:
            batch_result = self.batch_validator.validate_batch(signals_to_process)
            valid_signals = batch_result['valid_signals']
            invalid_signals = batch_result['invalid_signals']
            
                # Log batch validation results
            batch_stats = batch_result['batch_stats']
            if batch_stats['invalid_count'] > 0:
                self.logger.warning(
                    f"Batch validation: {batch_stats['valid_count']}/{batch_stats['total_processed']} valid, "
                    f"{batch_stats['duplicate_count']} duplicates, {batch_stats['rate_limited_count']} rate limited, "
                    f"{batch_stats['expired_count']} expired"
                )
            else:
                self.logger.debug(
                    f"Batch validation: {batch_stats['valid_count']}/{batch_stats['total_processed']} valid"
                )
        else:
            # Single signal validation
            signal = signals_to_process[0]
            result, errors = self.signal_validator.validate_signal(signal)
            
            if result == ValidationResult.VALID:
                valid_signals = [signal]
                invalid_signals = []
                self.logger.info(f"✅ Signal validation PASSED for {symbol}")
            else:
                valid_signals = []
                invalid_signals = [{
                    'signal': signal,
                    'result': result.value,
                    'errors': [{'code': e.code, 'message': e.message} for e in errors]
                }]
                self.logger.warning(f"❌ Signal validation FAILED for {symbol}: {result.value} - {[e.message for e in errors]}")
        
        validation_time = (datetime.now() - validation_start).total_seconds()
        self.performance_stats['validation_time'] += validation_time
        
        # Update statistics for invalid signals
        for invalid in invalid_signals:
            result = invalid['result']
            if result == 'duplicate':
                self.performance_stats['signals_duplicate'] += 1
            elif result == 'rate_limited':
                self.performance_stats['signals_rate_limited'] += 1
            else:
                self.performance_stats['signals_failed'] += 1
        
        # Process valid signals
        for signal in valid_signals:
            try:
                # Choose publishing method based on configuration
                if self.batch_config['enabled'] and len(valid_signals) > 1:
                    # Add to batch for later publishing
                    await self._add_signal_to_batch(signal)
                else:
                    # Publish immediately
                    await self._publish_signal_immediately(signal)
                
                # Store CVD baseline for position tracking
                await self._store_cvd_baseline_for_signal(signal, dataset)
                
                # Store in InfluxDB using enhanced signal manager
                if self.config.store_in_influxdb:
                    await self._store_enhanced_signal_in_influxdb(signal)
                
                # Update tracking
                self.last_signals[symbol] = datetime.now()
                self.performance_stats['signals_generated'] += 1
                signals_processed += 1
                
            except Exception as e:
                self.logger.error(f"Error processing valid signal: {e}")
                self.performance_stats['errors_encountered'] += 1
        
        return signals_processed
    
    def _convert_symbol_to_trading_pair(self, base_symbol: str) -> str:
        """Convert base symbol to trading pair format for Redis keys
        
        Args:
            base_symbol: Base symbol like 'BTC' or 'ETH'
            
        Returns:
            Trading pair format like 'BTCUSDT' or 'ETHUSDT'
        """
        # Map base symbols to trading pairs that match FreqTrade format
        symbol_mapping = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT',
            'SOL': 'SOLUSDT',
            'ADA': 'ADAUSDT',
            'DOT': 'DOTUSDT',
        }
        
        return symbol_mapping.get(base_symbol, f"{base_symbol}USDT")

    def _create_signal_from_order(self, symbol: str, order: Dict, 
                                 strategy_result: Dict) -> Dict:
        """Create Redis signal from strategy order"""
        
        # Check signal_type first for exit orders
        signal_type = order.get('signal_type', 'ENTRY')
        
        if signal_type == 'EXIT':
            # Exit orders always generate CLOSE signals
            action = 'CLOSE'
            total_score = 10.0  # Exit signals are high priority
            position_size_factor = 1.0  # Close entire position
            leverage = 1  # Not applicable for exits
            
            self.logger.info(f"{symbol}: Exit signal detected - converting to CLOSE action (score: {total_score})")
        else:
            # Entry orders - determine LONG/SHORT based on side
            side = order.get('side', 'BUY')
            action = 'LONG' if side == 'BUY' else 'SHORT'
            
            # Get scoring information
            phase4_result = strategy_result.get('phase_results', {}).get('phase4_scoring', {})
            total_score = phase4_result.get('total_score', 5.0)
            
            # Calculate position sizing and leverage based on score
            if total_score >= 8.0:
                position_size_factor = 1.5
                leverage = 5
            elif total_score >= 6.0:
                position_size_factor = 1.0
                leverage = 3
            else:
                position_size_factor = 0.5
                leverage = 2
        
        # Convert base symbol to trading pair for Redis key format
        trading_pair_symbol = self._convert_symbol_to_trading_pair(symbol)
        
        # Create signal
        signal = {
            'signal_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'symbol': trading_pair_symbol,  # Use trading pair format for Redis key compatibility
            'base_symbol': symbol,          # Keep base symbol for reference
            'action': action,
            'score': total_score,
            'position_size_factor': position_size_factor,
            'leverage': leverage,
            'entry_price': order.get('price', 0),
            'ttl': self.config.redis_signal_ttl,
            'confidence': order.get('confidence', 0),
            'reasoning': order.get('reasoning', 'SqueezeFlow signal'),
            'strategy': 'SqueezeFlowStrategy',
            'service': 'strategy_runner'
        }
        
        return signal
    
    async def _store_cvd_baseline_for_signal(self, signal: Dict, dataset: Dict):
        """
        Store CVD baseline data when signal is published
        This will be used later when FreqTrade executes the trade to track CVD flow changes
        
        Args:
            signal: Signal data with signal_id
            dataset: Market dataset with current CVD values
        """
        try:
            # Check if CVD baseline tracking is enabled
            if not self.config.enable_cvd_baseline_tracking:
                self.logger.debug("CVD baseline tracking disabled in config")
                return
                
            # Check if CVD baseline manager is available
            cvd_manager = self.cvd_baseline_manager
            if not cvd_manager:
                self.logger.warning("⚠️  CVD baseline manager not initialized - cannot store baseline")
                return
            
            # Get current CVD values from dataset
            spot_cvd_series = dataset.get('spot_cvd')
            futures_cvd_series = dataset.get('futures_cvd')
            
            if spot_cvd_series is None or futures_cvd_series is None:
                self.logger.warning("Cannot store CVD baseline: CVD data not available in dataset")
                return
            
            # Get latest CVD values
            current_spot_cvd = float(spot_cvd_series.iloc[-1]) if not spot_cvd_series.empty else 0.0
            current_futures_cvd = float(futures_cvd_series.iloc[-1]) if not futures_cvd_series.empty else 0.0
            
            # Store baseline temporarily with signal_id as key until we get trade_id from FreqTrade
            baseline_key = f"signal_cvd:{signal['signal_id']}"
            baseline_data = {
                'signal_id': signal['signal_id'],
                'symbol': signal['base_symbol'],  # Use base symbol
                'side': signal['action'].lower(),
                'entry_price': signal['entry_price'],
                'spot_cvd': current_spot_cvd,
                'futures_cvd': current_futures_cvd,
                'cvd_divergence': current_futures_cvd - current_spot_cvd,
                'timestamp': signal['timestamp']
            }
            
            # Store in Redis with TTL (24 hours) until trade execution
            self.redis_client.setex(
                baseline_key, 
                86400,  # 24 hour TTL - Extended to allow more time for trade execution
                json.dumps(baseline_data, default=str)
            )
            
            self.logger.info(f"✅ CVD baseline stored for signal {signal['signal_id']} ({signal['base_symbol']}): "
                           f"spot={current_spot_cvd:.2f}, futures={current_futures_cvd:.2f}, "
                           f"divergence={current_futures_cvd - current_spot_cvd:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing CVD baseline: {e}")
    
    async def _publish_signal_immediately(self, signal: Dict):
        """Publish single signal immediately to Redis with monitoring"""
        
        publish_start = datetime.now()
        
        try:
            # Prepare Redis operations
            redis_key = f"{self.config.redis_key_prefix}:signal:{signal['symbol']}"
            signal_json = json.dumps(signal, default=str)
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # 1. Set signal with TTL
            pipe.setex(redis_key, signal['ttl'], signal_json)
            
            # 2. Publish to real-time channel
            channel = f"{self.config.redis_key_prefix}:signals"
            pipe.publish(channel, signal_json)
            
            # 3. Add to signal history (limited retention)
            history_key = f"{self.config.redis_key_prefix}:history:{signal['symbol']}"
            pipe.lpush(history_key, signal_json)
            pipe.ltrim(history_key, 0, 99)  # Keep last 100 signals
            pipe.expire(history_key, 86400)  # 24 hour expiry
            
            # 4. Update signal counters
            counter_key = f"{self.config.redis_key_prefix}:stats:signals_published"
            pipe.incr(counter_key)
            pipe.expire(counter_key, 86400)
            
            # Execute pipeline
            results = pipe.execute()
            
            # Check if publish was successful (number of subscribers)
            subscribers_notified = results[1] if len(results) > 1 else 0
            
            # Update metrics
            publish_time = (datetime.now() - publish_start).total_seconds()
            self._update_redis_metrics(publish_time, True, subscribers_notified)
            
            # Track signal expiry
            expiry_time = datetime.now() + timedelta(seconds=signal['ttl'])
            self._signal_expiry_tracker[signal['signal_id']] = expiry_time
            
            self.performance_stats['signals_published'] += 1
            self.logger.debug(
                f"Published signal {signal['signal_id']} to Redis: {redis_key} "
                f"({subscribers_notified} subscribers notified, {publish_time:.3f}s)"
            )
            
        except Exception as e:
            publish_time = (datetime.now() - publish_start).total_seconds()
            self._update_redis_metrics(publish_time, False, 0)
            self.performance_stats['signals_failed'] += 1
            self.logger.error(f"Error publishing signal to Redis: {e}")
            raise
    
    async def _store_enhanced_signal_in_influxdb(self, signal: Dict):
        """Store signal using enhanced InfluxDB signal manager"""
        
        try:
            # Use enhanced signal manager for storage
            success = self.signal_manager.store_enhanced_signal(signal)
            
            if success:
                self.logger.debug(f"Enhanced signal stored: {signal['signal_id']}")
            else:
                self.logger.warning(f"Failed to store enhanced signal: {signal['signal_id']}")
                
        except Exception as e:
            self.logger.error(f"Error storing enhanced signal: {e}")
    
    async def _store_signal_in_influxdb(self, signal: Dict):
        """Store signal in InfluxDB for history (legacy method - kept for compatibility)"""
        
        try:
            # Create InfluxDB point
            point = {
                'measurement': 'strategy_signals',
                'tags': {
                    'symbol': signal['symbol'],
                    'action': signal['action'],
                    'strategy': signal['strategy'],
                    'service': signal['service']
                },
                'fields': {
                    'signal_id': signal['signal_id'],
                    'score': signal['score'],
                    'position_size_factor': signal['position_size_factor'],
                    'leverage': signal['leverage'],
                    'entry_price': signal['entry_price'],
                    'confidence': signal['confidence'],
                    'reasoning': signal['reasoning']
                },
                'time': signal['timestamp']
            }
            
            # Write to InfluxDB
            self.influx_client.write_points([point])
            
            self.logger.debug(f"Stored signal in InfluxDB: {signal['signal_id']}")
            
        except Exception as e:
            self.logger.error(f"Error storing signal in InfluxDB: {e}")
    
    def _is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in signal cooldown period using validator"""
        
        # Use the validator's more sophisticated cooldown check
        return self.signal_validator.is_symbol_in_cooldown(symbol)
    
    def get_symbol_cooldown_status(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get cooldown status for multiple symbols"""
        
        status = {}
        for symbol in symbols:
            in_cooldown = self.signal_validator.is_symbol_in_cooldown(symbol)
            remaining_seconds = self.signal_validator.get_symbol_cooldown_remaining(symbol)
            
            status[symbol] = {
                'in_cooldown': in_cooldown,
                'remaining_seconds': remaining_seconds,
                'next_available': (
                    datetime.now() + timedelta(seconds=remaining_seconds)
                ).isoformat() if in_cooldown else 'available_now'
            }
        
        return status
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task for expired signals and metrics"""
        
        while self.is_running:
            try:
                # Wait 5 minutes between cleanups
                await asyncio.sleep(300)
                
                if not self.is_running:
                    break
                
                # Clean up expired signals
                await self._cleanup_expired_signals()
                
                # Force batch flush if timeout exceeded
                if self.batch_config['enabled']:
                    time_since_flush = datetime.now() - self._last_batch_flush
                    max_timeout = timedelta(seconds=self.batch_config['batch_timeout_seconds'] * 2)
                    
                    if time_since_flush > max_timeout and self._signal_batch:
                        self.logger.warning("Forcing batch flush due to extended timeout")
                        await self._flush_signal_batch(force=True)
                
                self.logger.debug("Periodic cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
    
    def _update_performance_stats(self, cycle_duration: float):
        """Update performance statistics"""
        
        self.performance_stats['cycles_completed'] += 1
        self.performance_stats['last_cycle_duration'] = cycle_duration
        
        # Calculate rolling average
        cycles = self.performance_stats['cycles_completed']
        avg_duration = self.performance_stats['avg_cycle_duration']
        self.performance_stats['avg_cycle_duration'] = (
            (avg_duration * (cycles - 1) + cycle_duration) / cycles
        )
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        
        stats = self.performance_stats
        redis_stats = self._redis_metrics
        
        self.logger.info(
            f"Performance: Cycles={stats['cycles_completed']}, "
            f"Signals={stats['signals_generated']}, "
            f"Published={stats['signals_published']}, "
            f"Failed={stats['signals_failed']}, "
            f"Duplicates={stats['signals_duplicate']}, "
            f"RateLimited={stats['signals_rate_limited']}, "
            f"Batched={stats['batch_published']}, "
            f"Errors={stats['errors_encountered']}, "
            f"LastCycle={stats['last_cycle_duration']:.2f}s, "
            f"AvgCycle={stats['avg_cycle_duration']:.2f}s, "
            f"RedisPublishes={redis_stats['successful_publishes']}/{redis_stats['total_publishes']}, "
            f"AvgPublishTime={redis_stats['avg_publish_time']:.3f}s"
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive service health status with enhanced metrics"""
        
        # Get validation metrics
        validation_metrics = self.signal_validator.get_validation_metrics()
        
        # Get signal manager metrics if available
        signal_manager_metrics = {}
        try:
            if self._signal_manager is not None:
                signal_manager_metrics = self.signal_manager.get_operation_metrics()
        except Exception as e:
            self.logger.warning(f"Could not get signal manager metrics: {e}")
        
        # Get CVD baseline manager metrics if available
        cvd_baseline_metrics = {}
        try:
            if self._cvd_baseline_manager is not None:
                cvd_baseline_metrics = self._cvd_baseline_manager.get_metrics()
        except Exception as e:
            self.logger.warning(f"Could not get CVD baseline manager metrics: {e}")
        
        return {
            'service': 'strategy_runner',
            'status': 'running' if self.is_running else 'stopped',
            'uptime_cycles': self.cycle_count,
            'performance': self.performance_stats,
            'redis_metrics': self._redis_metrics,
            'validation_metrics': validation_metrics,
            'signal_manager_metrics': signal_manager_metrics,
            'cvd_baseline_metrics': cvd_baseline_metrics,
            'batch_status': {
                'enabled': self.batch_config['enabled'],
                'pending_signals': len(self._signal_batch),
                'last_flush': self._last_batch_flush.isoformat(),
                'config': self.batch_config
            },
            'signal_tracking': {
                'tracked_signals': len(self._signal_expiry_tracker),
                'symbols_with_recent_signals': len(self.last_signals)
            },
            'config': {
                'run_interval': self.config.run_interval_seconds,
                'max_symbols': self.config.max_symbols_per_cycle,
                'timeframe': self.config.default_timeframe,
                'signal_ttl': self.config.redis_signal_ttl,
                'cooldown_minutes': self.config.signal_cooldown_minutes
            },
            'connections': {
                'redis_main': self._redis_client is not None,
                'redis_pubsub': self._pubsub_client is not None,
                'influxdb': self._influx_client is not None,
                'signal_manager': self._signal_manager is not None,
                'freqtrade_api': self._freqtrade_client is not None,
                'cvd_baseline_manager': self._cvd_baseline_manager is not None
            },
            'pubsub_status': {
                'subscribers': self._redis_metrics['pubsub_subscribers'],
                'channels': list(self._signal_subscribers.keys())
            },
            'timestamp': datetime.now().isoformat()
        }


    # Additional methods for enhanced Redis functionality
    
    async def _add_signal_to_batch(self, signal: Dict):
        """Add signal to batch for later publishing"""
        
        with self._batch_lock:
            self._signal_batch.append(signal)
            
            # Auto-flush if batch is full
            if len(self._signal_batch) >= self.batch_config['max_batch_size']:
                await self._flush_signal_batch()
    
    async def _check_batch_flush(self):
        """Check if batch should be flushed due to timeout"""
        
        if not self._signal_batch:
            return
        
        time_since_last_flush = datetime.now() - self._last_batch_flush
        timeout = timedelta(seconds=self.batch_config['batch_timeout_seconds'])
        
        if time_since_last_flush >= timeout:
            await self._flush_signal_batch()
    
    async def _flush_signal_batch(self, force: bool = False):
        """Flush batched signals to Redis"""
        
        if not self._signal_batch and not force:
            return
        
        with self._batch_lock:
            if not self._signal_batch:
                return
            
            signals_to_publish = self._signal_batch.copy()
            self._signal_batch.clear()
            self._last_batch_flush = datetime.now()
        
        if not signals_to_publish:
            return
        
        batch_start = datetime.now()
        published_count = 0
        
        try:
            # Use pipeline for efficient batch publishing
            pipe = self.redis_client.pipeline()
            
            for signal in signals_to_publish:
                redis_key = f"{self.config.redis_key_prefix}:signal:{signal['symbol']}"
                signal_json = json.dumps(signal, default=str)
                channel = f"{self.config.redis_key_prefix}:signals"
                
                # Add to pipeline
                pipe.setex(redis_key, signal['ttl'], signal_json)
                pipe.publish(channel, signal_json)
                
                # Add to history
                history_key = f"{self.config.redis_key_prefix}:history:{signal['symbol']}"
                pipe.lpush(history_key, signal_json)
                pipe.ltrim(history_key, 0, 99)
                pipe.expire(history_key, 86400)
            
            # Execute batch
            results = pipe.execute()
            published_count = len(signals_to_publish)
            
            # Update metrics
            batch_time = (datetime.now() - batch_start).total_seconds()
            self.performance_stats['batch_published'] += published_count
            self.performance_stats['redis_publish_time'] += batch_time
            
            # Update individual metrics
            for signal in signals_to_publish:
                self.performance_stats['signals_published'] += 1
                expiry_time = datetime.now() + timedelta(seconds=signal['ttl'])
                self._signal_expiry_tracker[signal['signal_id']] = expiry_time
            
            self.logger.info(
                f"Batch published {published_count} signals in {batch_time:.3f}s "
                f"(avg {batch_time/published_count:.3f}s per signal)"
            )
            
        except Exception as e:
            batch_time = (datetime.now() - batch_start).total_seconds()
            self.performance_stats['redis_publish_time'] += batch_time
            self.performance_stats['signals_failed'] += len(signals_to_publish)
            self.logger.error(f"Error in batch publishing: {e}")
            raise
    
    async def _initialize_pubsub_channels(self):
        """Initialize Redis pub/sub channels for real-time signal distribution"""
        
        try:
            # Create standard channels
            channels = [
                f"{self.config.redis_key_prefix}:signals",  # All signals
                f"{self.config.redis_key_prefix}:alerts",   # System alerts
                f"{self.config.redis_key_prefix}:status"    # Status updates
            ]
            
            for channel in channels:
                self._signal_subscribers[channel] = []
            
            self.logger.info(f"Initialized {len(channels)} pub/sub channels")
            
        except Exception as e:
            self.logger.error(f"Error initializing pub/sub channels: {e}")
    
    def subscribe_to_signals(self, callback_function, channels: List[str] = None) -> str:
        """Subscribe to signal channels with callback function"""
        
        if channels is None:
            channels = [f"{self.config.redis_key_prefix}:signals"]
        
        subscriber_id = f"sub_{uuid.uuid4().hex[:8]}"
        
        try:
            # Add callback to channels
            for channel in channels:
                if channel not in self._signal_subscribers:
                    self._signal_subscribers[channel] = []
                
                self._signal_subscribers[channel].append({
                    'id': subscriber_id,
                    'callback': callback_function,
                    'subscribed_at': datetime.now()
                })
            
            self._redis_metrics['pubsub_subscribers'] += 1
            self.logger.info(f"Added subscriber {subscriber_id} to channels: {channels}")
            
            return subscriber_id
            
        except Exception as e:
            self.logger.error(f"Error subscribing to signals: {e}")
            return None
    
    def unsubscribe_from_signals(self, subscriber_id: str):
        """Remove subscriber from all channels"""
        
        removed_count = 0
        
        for channel, subscribers in self._signal_subscribers.items():
            original_count = len(subscribers)
            subscribers[:] = [sub for sub in subscribers if sub['id'] != subscriber_id]
            removed_count += original_count - len(subscribers)
        
        if removed_count > 0:
            self._redis_metrics['pubsub_subscribers'] = max(0, self._redis_metrics['pubsub_subscribers'] - 1)
            self.logger.info(f"Removed subscriber {subscriber_id} from {removed_count} channels")
    
    async def _cleanup_expired_signals(self):
        """Clean up expired signals from tracking"""
        
        now = datetime.now()
        expired_signals = [
            signal_id for signal_id, expiry_time in self._signal_expiry_tracker.items()
            if now > expiry_time
        ]
        
        for signal_id in expired_signals:
            del self._signal_expiry_tracker[signal_id]
        
        if expired_signals:
            self._redis_metrics['expired_signals_cleaned'] += len(expired_signals)
            self.logger.debug(f"Cleaned up {len(expired_signals)} expired signal records")
    
    def _update_redis_metrics(self, publish_time: float, success: bool, subscribers_notified: int):
        """Update Redis publishing metrics"""
        
        self._redis_metrics['total_publishes'] += 1
        
        if success:
            self._redis_metrics['successful_publishes'] += 1
        else:
            self._redis_metrics['failed_publishes'] += 1
        
        # Update average publish time
        total_publishes = self._redis_metrics['total_publishes']
        current_avg = self._redis_metrics['avg_publish_time']
        self._redis_metrics['avg_publish_time'] = (
            (current_avg * (total_publishes - 1) + publish_time) / total_publishes
        )
        
        # Update subscriber metrics
        if subscribers_notified > self._redis_metrics['pubsub_subscribers']:
            self._redis_metrics['pubsub_subscribers'] = subscribers_notified
    
    def get_signal_publishing_metrics(self) -> Dict:
        """Get detailed signal publishing metrics"""
        
        # Calculate success rates
        total_attempts = (self.performance_stats['signals_published'] + 
                         self.performance_stats['signals_failed'])
        success_rate = (self.performance_stats['signals_published'] / max(1, total_attempts)) * 100
        
        redis_success_rate = (
            self._redis_metrics['successful_publishes'] / 
            max(1, self._redis_metrics['total_publishes'])
        ) * 100
        
        return {
            'signal_stats': {
                'total_generated': self.performance_stats['signals_generated'],
                'total_published': self.performance_stats['signals_published'],
                'total_failed': self.performance_stats['signals_failed'],
                'duplicates_filtered': self.performance_stats['signals_duplicate'],
                'rate_limited': self.performance_stats['signals_rate_limited'],
                'batch_published': self.performance_stats['batch_published'],
                'success_rate_percent': round(success_rate, 2)
            },
            'redis_metrics': {
                **self._redis_metrics,
                'success_rate_percent': round(redis_success_rate, 2)
            },
            'validation_metrics': self.signal_validator.get_validation_metrics(),
            'batch_status': {
                'enabled': self.batch_config['enabled'],
                'pending_signals': len(self._signal_batch),
                'last_flush': self._last_batch_flush.isoformat(),
                'max_batch_size': self.batch_config['max_batch_size'],
                'timeout_seconds': self.batch_config['batch_timeout_seconds'],
                'auto_flush_threshold': self.batch_config['max_batch_size'],
                'time_since_last_flush': (datetime.now() - self._last_batch_flush).total_seconds()
            },
            'tracking': {
                'active_signals': len(self._signal_expiry_tracker),
                'symbols_tracked': len(self.last_signals),
                'expired_cleaned': self._redis_metrics['expired_signals_cleaned']
            },
            'pubsub': {
                'channels': len(self._signal_subscribers),
                'total_subscribers': sum(len(subs) for subs in self._signal_subscribers.values()),
                'active_channels': list(self._signal_subscribers.keys())
            },
            'performance': {
                'avg_validation_time_ms': round(
                    (self.performance_stats['validation_time'] / max(1, self.performance_stats['cycles_completed'])) * 1000, 3
                ),
                'avg_redis_publish_time_ms': round(self._redis_metrics['avg_publish_time'] * 1000, 3),
                'total_redis_time_seconds': round(self.performance_stats['redis_publish_time'], 2)
            },
            'health_indicators': {
                'high_success_rate': success_rate >= 95,
                'low_error_rate': self.performance_stats['errors_encountered'] < self.performance_stats['cycles_completed'] * 0.1,
                'reasonable_response_times': self._redis_metrics['avg_publish_time'] < 0.1,
                'active_monitoring': len(self._signal_subscribers) > 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_publishing_metrics(self):
        """Reset all publishing metrics"""
        
        self._redis_metrics = {
            'total_publishes': 0,
            'successful_publishes': 0,
            'failed_publishes': 0,
            'avg_publish_time': 0.0,
            'pubsub_subscribers': 0,
            'expired_signals_cleaned': 0
        }
        
        # Reset relevant performance stats
        signal_stats = [
            'signals_published', 'signals_failed', 'signals_duplicate',
            'signals_rate_limited', 'batch_published', 'redis_publish_time', 'validation_time'
        ]
        
        for stat in signal_stats:
            if stat in self.performance_stats:
                self.performance_stats[stat] = 0
        
        # Reset validator metrics
        self.signal_validator.reset_metrics()
        
        # Clear tracking data
        self._signal_expiry_tracker.clear()
        self._signal_batch.clear()
        self._last_batch_flush = datetime.now()
        
        self.logger.info("Publishing metrics and tracking data reset")
    
    # Enhanced Signal Analytics Methods
    
    def get_signal_analytics(self, symbol: str = None, hours_back: int = 24) -> Dict:
        """
        Get comprehensive signal analytics using enhanced signal manager
        
        Args:
            symbol: Filter by symbol (optional)
            hours_back: Hours to analyze
            
        Returns:
            Dict: Analytics data with additional metadata
        """
        try:
            analytics = self.signal_manager.get_signal_analytics(symbol, hours_back)
            
            # Convert to dict format for API compatibility
            return {
                'analytics': {
                    'total_signals': analytics.total_signals,
                    'profitable_signals': analytics.profitable_signals,
                    'unprofitable_signals': analytics.unprofitable_signals,
                    'pending_signals': analytics.pending_signals,
                    'expired_signals': analytics.expired_signals,
                    'win_rate_percent': round(analytics.win_rate, 2),
                    'average_pnl': analytics.average_pnl,
                    'average_holding_time_minutes': analytics.average_holding_time,
                    'total_pnl': analytics.total_pnl,
                    'max_profit': analytics.max_profit,
                    'max_loss': analytics.max_loss,
                    'profit_factor': analytics.profit_factor,
                    'avg_score': analytics.avg_score,
                    'avg_confidence': analytics.avg_confidence
                },
                'filters': {
                    'symbol': symbol,
                    'hours_back': hours_back
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal analytics: {e}")
            return {'error': str(e)}
    
    def get_signal_performance_by_score(self, hours_back: int = 168) -> Dict:
        """
        Get signal performance breakdown by score tiers
        
        Args:
            hours_back: Hours to analyze (default: 1 week)
            
        Returns:
            Dict: Performance by score tier
        """
        try:
            performance = self.signal_manager.get_signal_performance_by_score(hours_back)
            
            return {
                'performance_by_score': performance,
                'analysis_period_hours': hours_back,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance by score: {e}")
            return {'error': str(e)}
    
    def get_signal_quality_report(self, hours_back: int = 24) -> Dict:
        """
        Get comprehensive signal quality report
        
        Args:
            hours_back: Hours to analyze
            
        Returns:
            Dict: Quality metrics and health indicators
        """
        try:
            quality_metrics = self.signal_manager.get_signal_quality_metrics(hours_back)
            signal_manager_metrics = self.signal_manager.get_operation_metrics()
            
            return {
                'signal_quality': quality_metrics,
                'signal_manager_performance': signal_manager_metrics,
                'combined_health_score': (
                    quality_metrics.get('overall_health_score', 0) * 0.7 +
                    (100 if signal_manager_metrics['health_status']['low_error_rate'] else 0) * 0.3
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal quality report: {e}")
            return {'error': str(e)}
    
    def get_recent_signals_with_outcomes(self, symbol: str = None, limit: int = 50) -> Dict:
        """
        Get recent signals with their outcomes
        
        Args:
            symbol: Filter by symbol
            limit: Maximum signals to return
            
        Returns:
            Dict: Recent signals with enhanced data
        """
        try:
            signals = self.signal_manager.get_recent_signals(symbol, limit)
            
            return {
                'recent_signals': signals,
                'filters': {
                    'symbol': symbol,
                    'limit': limit
                },
                'count': len(signals),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {e}")
            return {'error': str(e)}
    
    def update_signal_outcome_from_trade(self, signal_id: str, trade_result: Dict) -> bool:
        """
        Update signal outcome based on trade execution result
        
        Args:
            signal_id: Signal identifier
            trade_result: Trade execution result with PnL data
            
        Returns:
            bool: Success status
        """
        try:
            from services.influx_signal_manager import SignalOutcome
            
            # Determine outcome based on PnL
            pnl = trade_result.get('pnl', 0)
            if pnl > 0:
                outcome = SignalOutcome.PROFITABLE
            elif pnl < 0:
                outcome = SignalOutcome.UNPROFITABLE
            else:
                outcome = SignalOutcome.EXPIRED  # No trade executed
            
            # Update signal outcome
            success = self.signal_manager.update_signal_outcome(
                signal_id=signal_id,
                outcome=outcome,
                exit_price=trade_result.get('exit_price'),
                pnl=pnl,
                pnl_percentage=trade_result.get('pnl_percentage')
            )
            
            if success:
                self.logger.info(f"Signal outcome updated: {signal_id} -> {outcome.value}")
            else:
                self.logger.warning(f"Failed to update signal outcome: {signal_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating signal outcome: {e}")
            return False
    
    def create_signal_aggregation_views(self) -> Dict:
        """
        Create continuous queries for signal aggregation
        
        Returns:
            Dict: Creation results for different timeframes
        """
        try:
            results = {}
            timeframes = ['1h', '4h', '1d']
            
            for timeframe in timeframes:
                success = self.signal_manager.create_signal_aggregation_view(timeframe)
                results[timeframe] = success
                
                if success:
                    self.logger.info(f"Created signal aggregation view for {timeframe}")
                else:
                    self.logger.warning(f"Failed to create aggregation view for {timeframe}")
            
            return {
                'aggregation_views': results,
                'total_created': sum(results.values()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating aggregation views: {e}")
            return {'error': str(e)}
    
    def archive_old_signals(self, days_to_keep: int = 30) -> Dict:
        """
        Archive old signals to reduce database size
        
        Args:
            days_to_keep: Days of signals to keep
            
        Returns:
            Dict: Archive operation results
        """
        try:
            result = self.signal_manager.archive_old_signals(days_to_keep)
            
            if result.get('success', False):
                self.logger.info(f"Archived {result['archived_signals']} old signals")
            else:
                self.logger.warning(f"Signal archival failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error archiving signals: {e}")
            return {'error': str(e)}
    
    def get_database_retention_info(self) -> Dict:
        """
        Get database retention policy information
        
        Returns:
            Dict: Retention policy details
        """
        try:
            policies = self.signal_manager.get_retention_policy_info()
            
            return {
                'retention_policies': policies,
                'database': self.signal_manager.database,
                'total_policies': len(policies),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting retention info: {e}")
            return {'error': str(e)}
    
    def link_signal_to_trade(self, signal_id: str, trade_id: int) -> bool:
        """
        Link a signal to a FreqTrade execution and move CVD baseline to permanent storage
        
        This method can be called by external systems (like FreqTrade webhook) 
        when a trade is executed based on a signal.
        
        Args:
            signal_id: Original signal ID
            trade_id: FreqTrade trade ID
            
        Returns:
            bool: Success status
        """
        try:
            if not self.config.enable_cvd_baseline_tracking or not self.cvd_baseline_manager:
                return False
            
            # Get temporary baseline data
            baseline_key = f"signal_cvd:{signal_id}"
            baseline_data_str = self.redis_client.get(baseline_key)
            
            if not baseline_data_str:
                self.logger.warning(f"No CVD baseline found for signal {signal_id}")
                return False
            
            baseline_data = json.loads(baseline_data_str)
            
            # Store in permanent CVD baseline manager
            success = self.cvd_baseline_manager.store_baseline(
                signal_id=signal_id,
                trade_id=trade_id,
                symbol=baseline_data['symbol'],
                side=baseline_data['side'],
                entry_price=baseline_data['entry_price'],
                spot_cvd=baseline_data['spot_cvd'],
                futures_cvd=baseline_data['futures_cvd']
            )
            
            if success:
                # Remove temporary data
                self.redis_client.delete(baseline_key)
                self.logger.info(f"Linked signal {signal_id} to trade {trade_id} with CVD baseline")
                return True
            else:
                self.logger.error(f"Failed to store permanent CVD baseline for signal {signal_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error linking signal to trade: {e}")
            return False
    
    def get_trade_cvd_analysis(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Get CVD flow analysis for an open trade
        
        Args:
            trade_id: FreqTrade trade ID
            
        Returns:
            CVD analysis dict or None if not available
        """
        try:
            if not self.config.enable_cvd_baseline_tracking or not self.cvd_baseline_manager:
                return None
            
            # Get baseline
            baseline = self.cvd_baseline_manager.get_baseline(trade_id)
            if not baseline:
                return None
            
            # Get current CVD data for the symbol
            dataset = self.data_pipeline.get_complete_dataset(
                symbol=baseline.symbol,
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                timeframe=self.config.default_timeframe
            )
            
            if not dataset:
                return None
            
            # Get current CVD values
            spot_cvd_series = dataset.get('spot_cvd')
            futures_cvd_series = dataset.get('futures_cvd')
            
            if spot_cvd_series is None or futures_cvd_series is None:
                return None
            
            current_spot_cvd = float(spot_cvd_series.iloc[-1]) if not spot_cvd_series.empty else 0.0
            current_futures_cvd = float(futures_cvd_series.iloc[-1]) if not futures_cvd_series.empty else 0.0
            
            # Calculate flow changes
            flow_metrics = self.cvd_baseline_manager.calculate_cvd_flow_change(
                trade_id, current_spot_cvd, current_futures_cvd
            )
            
            if flow_metrics:
                flow_metrics.update({
                    'baseline_data': baseline.to_dict(),
                    'current_spot_cvd': current_spot_cvd,
                    'current_futures_cvd': current_futures_cvd,
                    'timestamp': datetime.now().isoformat()
                })
            
            return flow_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting trade CVD analysis: {e}")
            return None


async def main():
    """Main entry point for running the service"""
    
    # Initialize service
    config_manager = ConfigManager()
    runner = StrategyRunner(config_manager)
    
    try:
        # Start service
        await runner.start()
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    except Exception as e:
        print(f"Service error: {e}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    # Enhanced startup with logging
    import sys
    
    # Setup basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('data/logs/strategy_runner_startup.log', mode='a')
        ]
    )
    
    logger = logging.getLogger('strategy_runner_main')
    logger.info("Starting Enhanced Strategy Runner with Redis Signal Publishing")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.fatal(f"Failed to start Strategy Runner: {e}")
        sys.exit(1)