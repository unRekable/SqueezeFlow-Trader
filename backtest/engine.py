#!/usr/bin/env python3
"""
SqueezeFlow Backtest Engine - Clean Orchestrator
Modular architecture using existing infrastructure and specialized components

Architecture:
- Uses existing utils/ (exchange_mapper, market_discovery, influxdb_handler)
- Integrates with fees.py for realistic trading costs
- Uses portfolio.py for position & risk management  
- Uses plotter.py for comprehensive visualization
- Clean separation of concerns and best practices
"""

import asyncio
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing architecture
from data.storage.influxdb_handler import InfluxDBHandler
from utils.exchange_mapper import exchange_mapper
from utils.market_discovery import market_discovery

# Import modular components from new structure
try:
    # Try relative imports first (when used as module)
    from .core.portfolio import PortfolioManager, Position, PositionType, RiskLimits
    from .core.fees import FeeCalculator, TradingCosts
    from .visualization.plotter import BacktestPlotter
    from .core.strategy import BaseStrategy, TradingSignal, SignalStrength, load_strategy
except ImportError:
    # Fall back to absolute imports (when run directly)
    from core.portfolio import PortfolioManager, Position, PositionType, RiskLimits
    from core.fees import FeeCalculator, TradingCosts
    from visualization.plotter import BacktestPlotter
    from core.strategy import BaseStrategy, TradingSignal, SignalStrength, load_strategy


class SqueezeFlowBacktestEngine:
    """
    Clean, modular backtest engine for SqueezeFlow trading strategy
    Orchestrates data loading, signal generation, trade execution, and analysis
    """
    
    def __init__(self, start_date: str, end_date: str, initial_balance: float = 10000,
                 symbols: List[str] = None, risk_limits: RiskLimits = None,
                 strategy: BaseStrategy = None, strategy_config: Dict[str, Any] = None,
                 debug_mode: bool = False, show_full_range: bool = False):
        """
        Initialize backtest engine with modular components
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            initial_balance: Starting portfolio balance
            symbols: List of symbols to trade (default: ['BTC', 'ETH'])
            risk_limits: Risk management configuration
            strategy: Trading strategy instance (auto-loaded if None)
            strategy_config: Strategy configuration dict
        """
        self.start_date = pd.to_datetime(start_date).tz_localize('UTC')
        self.end_date = pd.to_datetime(end_date).tz_localize('UTC')
        self.initial_balance = initial_balance
        self.symbols = symbols or ['BTC', 'ETH']
        self.debug_mode = debug_mode
        self.show_full_range = show_full_range
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.setup_database()
        self.setup_portfolio(risk_limits)
        self.setup_fee_calculator()
        self.setup_plotter()
        self.setup_strategy(strategy, strategy_config)
        
        # Data storage
        self.historical_data = {}
        self.execution_log = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SqueezeFlowBacktest')
        
    def setup_database(self):
        """Setup database connection using existing infrastructure"""
        try:
            self.db_handler = InfluxDBHandler(
                host=os.getenv('INFLUX_HOST', 'localhost'),
                port=int(os.getenv('INFLUX_PORT', 8086)),
                username=os.getenv('INFLUX_USER', ''),
                password=os.getenv('INFLUX_PASSWORD', ''),
                database='significant_trades'
            )
            
            # Test connection
            if self.db_handler.test_connection():
                self.logger.info("✅ Database connection established")
            else:
                raise ConnectionError("Failed to connect to InfluxDB")
                
        except Exception as e:
            self.logger.error(f"❌ Database setup failed: {e}")
            raise
    
    def setup_portfolio(self, risk_limits: RiskLimits = None):
        """Setup portfolio manager with risk controls"""
        self.portfolio = PortfolioManager(
            initial_balance=self.initial_balance,
            risk_limits=risk_limits or RiskLimits()
        )
        self.logger.info(f"💼 Portfolio initialized: ${self.initial_balance:,.2f}")
        
    def setup_fee_calculator(self):
        """Setup fee calculator using exchange mapper"""
        self.fee_calculator = FeeCalculator()
        self.logger.info("💰 Fee calculator initialized with real market fees")
        
    def setup_plotter(self):
        """Setup plotting system"""
        self.plotter = BacktestPlotter(save_directory="backtest")
        self.logger.info("📊 Plotting system initialized")
        
    def setup_strategy(self, strategy: BaseStrategy = None, strategy_config: Dict[str, Any] = None):
        """Setup trading strategy"""
        if strategy is not None:
            self.strategy = strategy
            self.logger.info(f"🧠 Using provided strategy: {strategy.get_strategy_name()}")
        else:
            # Load default SqueezeFlow strategy
            default_config = strategy_config or {
                'signal_threshold': 0.6,
                'lookback_periods': [5, 10, 15, 30, 60, 120, 240],
                'cvd_threshold': 50_000_000,
                'price_threshold': 0.5,
            }
            self.strategy = load_strategy('squeezeflow_strategy', default_config)
            self.logger.info(f"🧠 Loaded default strategy: {self.strategy.get_strategy_name()}")
        
        # Engine configuration (separate from strategy)
        self.engine_config = {
            'check_interval': 5,            # Check signals every N minutes
            'position_size_pct': 0.02,      # 2% position size
            'max_holding_hours': 24         # Maximum holding period
        }
        
    async def load_historical_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load historical data for all symbols using existing infrastructure
        
        Returns:
            Dict with symbol -> {price, spot_cvd, perp_cvd} DataFrames
        """
        self.logger.info(f"📥 Loading historical data for {len(self.symbols)} symbols...")
        
        if self.debug_mode:
            self.logger.info(f"🔍 DEBUG: Requested date range: {self.start_date.date()} to {self.end_date.date()}")
            
            # Check actual data availability first
            for symbol in self.symbols:
                await self._debug_data_coverage(symbol)
        
        for symbol in self.symbols:
            try:
                self.logger.info(f"Loading {symbol} data...")
                
                # Get markets for symbol using market discovery
                markets_by_type = market_discovery.get_markets_by_type(symbol)
                
                if not markets_by_type['spot'] and not markets_by_type['perp']:
                    self.logger.warning(f"No markets found for {symbol}")
                    continue
                
                # Load price data
                price_data = await self._load_price_data(symbol, markets_by_type['spot'])
                
                # Load CVD data  
                spot_cvd = await self._load_cvd_data(symbol, markets_by_type['spot'], 'spot')
                perp_cvd = await self._load_cvd_data(symbol, markets_by_type['perp'], 'perp')
                
                if not price_data.empty and not spot_cvd.empty and not perp_cvd.empty:
                    self.historical_data[symbol] = {
                        'price': price_data,
                        'spot_cvd': spot_cvd,
                        'perp_cvd': perp_cvd,
                        'markets': markets_by_type
                    }
                    self.logger.info(f"✅ {symbol}: {len(price_data)} price points, "
                                   f"{len(spot_cvd)} spot CVD, {len(perp_cvd)} perp CVD")
                else:
                    self.logger.warning(f"❌ {symbol}: Insufficient data (price: {len(price_data)}, "
                                       f"spot CVD: {len(spot_cvd)}, perp CVD: {len(perp_cvd)})")
                    
            except Exception as e:
                self.logger.error(f"❌ Error loading {symbol} data: {e}")
                continue
        
        if not self.historical_data:
            raise ValueError("No historical data loaded for any symbol")
            
        self.logger.info(f"✅ Historical data loaded for {len(self.historical_data)} symbols")
        return self.historical_data
    
    async def _load_price_data(self, symbol: str, spot_markets: List[str]) -> pd.DataFrame:
        """Load price data using major spot markets with 1-minute resolution"""
        try:
            # Use only major exchanges for reliable pricing
            major_exchanges = ['BINANCE:', 'COINBASE:', 'KRAKEN:', 'BITSTAMP:']
            major_markets = [m for m in spot_markets if any(ex in m.upper() for ex in major_exchanges)]
            
            if not major_markets:
                major_markets = spot_markets[:5]  # Fallback to first 5 markets
            
            # Build market filter for InfluxDB query
            market_filter = ' OR '.join([f"market = '{market}'" for market in major_markets])
            
            # Convert timestamps to nanoseconds for InfluxDB
            start_time_ns = int(self.start_date.timestamp() * 1_000_000_000)
            end_time_ns = int(self.end_date.timestamp() * 1_000_000_000)
            
            query = f"""
            SELECT median(close) as price
            FROM "aggr_1m"."trades_1m"
            WHERE ({market_filter})
            AND time >= {start_time_ns} AND time <= {end_time_ns}
            AND close > 0
            GROUP BY time(1m)
            ORDER BY time ASC
            """
            
            result = self.db_handler.client.query(query)
            points = list(result.get_points())
            
            if not points:
                self.logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(points)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            self.logger.info(f"✅ Loaded {len(df)} price data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading price data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _load_cvd_data(self, symbol: str, markets: List[str], market_type: str) -> pd.Series:
        """Load CVD data for specific market type with 1-minute resolution"""
        try:
            if not markets:
                self.logger.warning(f"No {market_type} markets provided for {symbol}")
                return pd.Series()
            
            # Build market filter
            market_filter = ' OR '.join([f"market = '{market}'" for market in markets])
            
            # Convert timestamps to nanoseconds for InfluxDB
            start_time_ns = int(self.start_date.timestamp() * 1_000_000_000)
            end_time_ns = int(self.end_date.timestamp() * 1_000_000_000)
            
            query = f"""
            SELECT sum(vbuy) - sum(vsell) as cvd_delta
            FROM "aggr_1m"."trades_1m" 
            WHERE ({market_filter})
            AND time >= {start_time_ns} AND time <= {end_time_ns}
            GROUP BY time(1m)
            ORDER BY time ASC
            """
            
            result = self.db_handler.client.query(query)
            points = list(result.get_points())
            
            if not points:
                self.logger.warning(f"No {market_type} CVD data found for {symbol}")
                return pd.Series()
            
            df = pd.DataFrame(points)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Calculate cumulative CVD (industry standard)
            cvd_cumulative = df['cvd_delta'].fillna(0).cumsum()
            
            self.logger.info(f"✅ Loaded {len(cvd_cumulative)} {market_type} CVD data points for {symbol}")
            return cvd_cumulative
            
        except Exception as e:
            self.logger.error(f"Error loading {market_type} CVD for {symbol}: {e}")
            return pd.Series()
    
    async def _debug_data_coverage(self, symbol: str):
        """Debug function to check data coverage for a symbol"""
        try:
            # Get basic coverage info
            # Convert timestamps to nanoseconds for InfluxDB
            start_time_ns = int(self.start_date.timestamp() * 1_000_000_000)
            end_time_ns = int(self.end_date.timestamp() * 1_000_000_000)
            
            query = f"""
            SELECT count(close) as data_points,
                   min(time) as first_data,
                   max(time) as last_data
            FROM "aggr_1m"."trades_1m"
            WHERE market =~ /{symbol.lower()}/
            AND time >= {start_time_ns} AND time <= {end_time_ns}
            """
            
            result = self.db_handler.client.query(query)
            points = list(result.get_points())
            
            if points and points[0]['data_points'] > 0:
                point = points[0]
                self.logger.info(f"🔍 DEBUG {symbol}: {point['data_points']} total points from {point['first_data']} to {point['last_data']}")
                
                # Daily breakdown
                daily_query = f"""
                SELECT count(close) as daily_points
                FROM "aggr_1m"."trades_1m"
                WHERE market =~ /{symbol.lower()}/
                AND time >= {start_time_ns} AND time <= {end_time_ns}
                GROUP BY time(1d)
                ORDER BY time ASC
                """
                
                daily_result = self.db_handler.client.query(daily_query)
                daily_points = list(daily_result.get_points())
                
                self.logger.info(f"🔍 DEBUG {symbol} daily breakdown:")
                for i, point in enumerate(daily_points):
                    date_str = point['time'][:10] if point['time'] else 'Unknown'
                    self.logger.info(f"    {date_str}: {point['daily_points']} data points")
                    
            else:
                self.logger.warning(f"🔍 DEBUG {symbol}: NO DATA FOUND in requested range!")
                
        except Exception as e:
            self.logger.error(f"🔍 DEBUG {symbol}: Error checking coverage: {e}")
    
    def generate_trading_signals(self, symbol: str, timestamp: datetime,
                                lookback_data: Dict[str, pd.Series]) -> TradingSignal:
        """
        Generate trading signals using the loaded strategy
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            lookback_data: Dict with 'price', 'spot_cvd', 'perp_cvd' series
            
        Returns:
            TradingSignal object
        """
        return self.strategy.generate_signal(symbol, timestamp, lookback_data)
    
    def execute_trade_logic(self, signal: TradingSignal) -> Optional[Position]:
        """
        Execute trading logic based on signal
        
        Args:
            signal: Trading signal to process
            
        Returns:
            Position if trade executed, None otherwise
        """
        try:
            # Skip weak signals
            strategy_threshold = self.strategy.config.get('signal_threshold', 0.6)
            
            # Debug logging
            if signal.signal_type != 'NONE':
                self.logger.info(f"🔍 Signal for {signal.symbol}: {signal.signal_type}, "
                               f"strength: {signal.strength}, confidence: {signal.confidence:.3f}, "
                               f"threshold: {strategy_threshold}")
            
            if signal.strength == SignalStrength.NONE or signal.confidence < strategy_threshold:
                if signal.signal_type != 'NONE':
                    self.logger.info(f"❌ Signal rejected: confidence {signal.confidence:.3f} < {strategy_threshold}")
                return None
            
            # Check if we can open a position
            base_position_size_pct = self.engine_config['position_size_pct']
            
            # Adjust position size based on signal strength, but respect risk limits
            max_allowed_size = self.portfolio.risk_limits.max_position_size
            
            if signal.strength == SignalStrength.STRONG:
                position_size_pct = min(base_position_size_pct * 1.5, max_allowed_size)
            elif signal.strength == SignalStrength.WEAK:
                position_size_pct = base_position_size_pct * 0.5
            else:
                position_size_pct = base_position_size_pct
            
            # Ensure we don't exceed risk limits
            position_size_pct = min(position_size_pct, max_allowed_size)
            
            # Calculate trading costs
            trade_value = self.portfolio.current_balance * position_size_pct
            trading_costs = self.fee_calculator.calculate_trading_costs(
                symbol=signal.symbol,
                trade_value=trade_value,
                holding_hours=self.engine_config['max_holding_hours']
            )
            
            # Adjust position size for fees
            net_position_size_pct = position_size_pct - (trading_costs.cost_percentage / 100)
            
            if net_position_size_pct <= 0:
                self.logger.warning(f"Position size too small after fees for {signal.symbol}")
                return None
            
            # Determine position type
            position_type = PositionType.LONG if signal.signal_type == 'LONG' else PositionType.SHORT
            
            # Attempt to open position
            position = self.portfolio.open_position(
                symbol=signal.symbol,
                position_type=position_type,
                entry_price=signal.price,
                position_size_percentage=net_position_size_pct,
                timestamp=signal.timestamp
            )
            
            if position:
                # CRITICAL: Notify strategy that position was opened
                if hasattr(self.strategy, 'confirm_position_opened'):
                    try:
                        self.strategy.confirm_position_opened(signal.symbol, position)
                    except Exception as e:
                        self.logger.error(f"Error notifying strategy of position open: {e}")
                
                # Log trade execution
                self.execution_log.append({
                    'timestamp': signal.timestamp,
                    'action': 'OPEN',
                    'symbol': signal.symbol,
                    'side': signal.signal_type,
                    'price': signal.price,
                    'size_pct': net_position_size_pct,
                    'confidence': signal.confidence,
                    'strength': signal.strength.name,
                    'trading_costs': trading_costs.total_cost,
                    'position_id': position.id
                })
                
                self.logger.info(f"🚀 Opened {position.position_type.value} position: "
                               f"{position.symbol} at ${position.entry_price:.2f} "
                               f"(size: {net_position_size_pct:.2f}%, confidence: {signal.confidence:.2f})")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing trade logic for {signal.symbol}: {e}")
            return None
    
    def manage_open_positions(self, current_prices: Dict[str, float], timestamp: datetime):
        """
        Manage open positions (stop loss, take profit, time-based exits)
        
        Args:
            current_prices: Current prices for all symbols
            timestamp: Current timestamp
        """
        try:
            # Update position metrics
            self.portfolio.update_position_metrics(current_prices, timestamp)
            
            # Check strategy-specific exit conditions first
            strategy_exits = []
            for position_id, position in self.portfolio.open_positions.items():
                if position.symbol in current_prices:
                    # Prepare current data for strategy
                    if position.symbol in self.historical_data:
                        data = self.historical_data[position.symbol]
                        current_idx = None
                        for idx, ts in enumerate(data['price'].index):
                            if ts <= timestamp:
                                current_idx = idx
                        
                        if current_idx is not None and current_idx >= 240:  # Ensure enough lookback
                            start_idx = max(0, current_idx - 240)
                            current_data = {
                                'price': data['price'].iloc[start_idx:current_idx+1]['price'],
                                'spot_cvd': data['spot_cvd'].iloc[start_idx:current_idx+1],
                                'perp_cvd': data['perp_cvd'].iloc[start_idx:current_idx+1]
                            }
                            
                            # Check strategy exit conditions
                            should_exit, exit_reason, confidence = self.strategy.should_exit_position(
                                position, current_data, timestamp
                            )
                            
                            if should_exit:
                                strategy_exits.append((position_id, exit_reason))
                                print(f"    Strategy exit: {position_id} - {exit_reason}")
            
            # Close positions based on strategy exits
            for position_id, exit_reason in strategy_exits:
                if position_id in self.portfolio.open_positions:
                    position = self.portfolio.open_positions[position_id]
                    exit_price = current_prices.get(position.symbol, position.entry_price)
                    
                    trade_value = position.entry_price * position.size
                    exit_costs = self.fee_calculator.calculate_trading_costs(
                        symbol=position.symbol,
                        trade_value=trade_value,
                        holding_hours=position.duration_hours(timestamp)
                    )
                    
                    closed_position = self.portfolio.close_position(
                        position_id=position_id,
                        exit_price=exit_price,
                        timestamp=timestamp,
                        exit_reason=exit_reason,
                        trading_fees=exit_costs.total_cost
                    )
                    
                    if closed_position:
                        # CRITICAL: Notify strategy that position was closed
                        if hasattr(self.strategy, 'confirm_position_closed'):
                            try:
                                self.strategy.confirm_position_closed(closed_position.symbol, closed_position, exit_reason)
                            except Exception as e:
                                self.logger.error(f"Error notifying strategy of position close: {e}")
                        
                        self.execution_log.append({
                            'timestamp': timestamp,
                            'action': 'CLOSE',
                            'symbol': closed_position.symbol,
                            'side': closed_position.position_type.value,
                            'price': exit_price,
                            'pnl_pct': closed_position.pnl_percentage,
                            'pnl_abs': closed_position.pnl_absolute,
                            'duration_hours': closed_position.duration_hours(),
                            'exit_reason': closed_position.exit_reason,
                            'trading_costs': exit_costs.total_cost,
                            'position_id': closed_position.id
                        })
            
            # Check stop loss and take profit
            positions_to_close = self.portfolio.check_stop_loss_take_profit(current_prices, timestamp)
            
            # Close triggered positions
            for position_id in positions_to_close:
                if position_id in self.portfolio.open_positions:
                    position = self.portfolio.open_positions[position_id]
                    exit_price = current_prices.get(position.symbol, position.entry_price)
                    
                    # Calculate final trading costs
                    trade_value = position.entry_price * position.size
                    exit_costs = self.fee_calculator.calculate_trading_costs(
                        symbol=position.symbol,
                        trade_value=trade_value,
                        holding_hours=position.duration_hours(timestamp)
                    )
                    
                    # Close position
                    closed_position = self.portfolio.close_position(
                        position_id=position_id,
                        exit_price=exit_price,
                        timestamp=timestamp,
                        exit_reason="Stop/Target triggered",
                        trading_fees=exit_costs.total_cost
                    )
                    
                    if closed_position:
                        # Log trade closure
                        self.execution_log.append({
                            'timestamp': timestamp,
                            'action': 'CLOSE',
                            'symbol': closed_position.symbol,
                            'side': closed_position.position_type.value,
                            'price': exit_price,
                            'pnl_pct': closed_position.pnl_percentage,
                            'pnl_abs': closed_position.pnl_absolute,
                            'duration_hours': closed_position.duration_hours(),
                            'exit_reason': closed_position.exit_reason,
                            'trading_costs': exit_costs.total_cost,
                            'position_id': closed_position.id
                        })
            
            # Check for time-based exits
            for position_id, position in list(self.portfolio.open_positions.items()):
                duration_hours = position.duration_hours(timestamp)
                print(f"    Checking time exit for {position_id}: {duration_hours:.2f}h vs {self.engine_config['max_holding_hours']}h limit")
                if duration_hours > self.engine_config['max_holding_hours']:
                    exit_price = current_prices.get(position.symbol, position.entry_price)
                    
                    trade_value = position.entry_price * position.size
                    exit_costs = self.fee_calculator.calculate_trading_costs(
                        symbol=position.symbol,
                        trade_value=trade_value,
                        holding_hours=position.duration_hours(timestamp)
                    )
                    
                    closed_position = self.portfolio.close_position(
                        position_id=position_id,
                        exit_price=exit_price,
                        timestamp=timestamp,
                        exit_reason="Time-based exit",
                        trading_fees=exit_costs.total_cost
                    )
                    
                    if closed_position:
                        # CRITICAL: Notify strategy that position was closed
                        if hasattr(self.strategy, 'confirm_position_closed'):
                            try:
                                self.strategy.confirm_position_closed(closed_position.symbol, closed_position, exit_reason)
                            except Exception as e:
                                self.logger.error(f"Error notifying strategy of position close: {e}")
                        
                        self.execution_log.append({
                            'timestamp': timestamp,
                            'action': 'CLOSE',
                            'symbol': closed_position.symbol,
                            'side': closed_position.position_type.value,
                            'price': exit_price,
                            'pnl_pct': closed_position.pnl_percentage,
                            'pnl_abs': closed_position.pnl_absolute,
                            'duration_hours': closed_position.duration_hours(),
                            'exit_reason': closed_position.exit_reason,
                            'trading_costs': exit_costs.total_cost,
                            'position_id': closed_position.id
                        })
                        
                        self.logger.info(f"⏰ Time-based exit: {closed_position.symbol} "
                                       f"(P&L: {closed_position.pnl_percentage:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
    
    async def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtest simulation
        
        Returns:
            Comprehensive backtest results
        """
        self.logger.info(f"🚀 Starting SqueezeFlow backtest: {self.start_date.date()} to {self.end_date.date()}")
        
        try:
            # Load historical data
            await self.load_historical_data()
            
            # Get all timestamps for simulation
            all_timestamps = set()
            for data in self.historical_data.values():
                all_timestamps.update(data['price'].index)
            all_timestamps = sorted(all_timestamps)
            
            # Find valid simulation range
            strategy_lookbacks = self.strategy.config.get('lookback_periods', [240])
            max_lookback = max(strategy_lookbacks) if strategy_lookbacks else 240
            start_idx = max_lookback + 10  # Ensure sufficient lookback data
            simulation_timestamps = all_timestamps[start_idx::self.engine_config['check_interval']]
            
            self.logger.info(f"📊 Simulating {len(simulation_timestamps)} time periods...")
            
            # Main simulation loop
            daily_stats = {}
            current_day = None
            
            for i, timestamp in enumerate(simulation_timestamps):
                try:
                    # Track daily progress for debug mode
                    if self.debug_mode:
                        day_str = timestamp.strftime('%Y-%m-%d')
                        if day_str != current_day:
                            if current_day and current_day in daily_stats:
                                self.logger.info(f"🔍 DEBUG {current_day}: {daily_stats[current_day]['signals']} signals, {daily_stats[current_day]['trades']} trades, Balance: ${self.portfolio.current_balance:,.2f}")
                            current_day = day_str
                            daily_stats[day_str] = {'signals': 0, 'trades': 0, 'balance': self.portfolio.current_balance}
                    # Get current prices
                    current_prices = {}
                    for symbol, data in self.historical_data.items():
                        if timestamp in data['price'].index:
                            price = data['price'].loc[timestamp, 'price']
                            if not pd.isna(price):
                                current_prices[symbol] = price
                    
                    # Manage existing positions first
                    if current_prices:
                        self.manage_open_positions(current_prices, timestamp)
                    
                    # Generate new signals (only if we have room for new positions)
                    if len(self.portfolio.open_positions) < self.portfolio.risk_limits.max_open_positions:
                        for symbol in self.symbols:
                            if symbol in current_prices and symbol in self.historical_data:
                                # CRITICAL FIX: Check if symbol already has an open position
                                has_open_position = any(pos.symbol == symbol for pos in self.portfolio.open_positions.values())
                                if has_open_position:
                                    continue  # Skip if already have position in this symbol
                                
                                # Prepare lookback data
                                data = self.historical_data[symbol]
                                end_idx = data['price'].index.get_loc(timestamp)
                                start_idx = max(0, end_idx - max_lookback)
                                
                                lookback_data = {
                                    'price': data['price'].iloc[start_idx:end_idx+1]['price'],
                                    'spot_cvd': data['spot_cvd'].iloc[start_idx:end_idx+1],
                                    'perp_cvd': data['perp_cvd'].iloc[start_idx:end_idx+1]
                                }
                                
                                # Generate signal
                                signal = self.generate_trading_signals(symbol, timestamp, lookback_data)
                                
                                # Execute trade if signal is strong enough
                                if signal.signal_type != 'NONE':
                                    if self.debug_mode and current_day:
                                        daily_stats[current_day]['signals'] += 1
                                    
                                    position = self.execute_trade_logic(signal)
                                    if position and self.debug_mode and current_day:
                                        daily_stats[current_day]['trades'] += 1
                    
                    # Progress logging
                    if i % 100 == 0 and i > 0:
                        progress_pct = (i / len(simulation_timestamps)) * 100
                        self.logger.info(f"📈 Progress: {progress_pct:.1f}% "
                                       f"(Balance: ${self.portfolio.current_balance:,.2f}, "
                                       f"Open: {len(self.portfolio.open_positions)})")
                        
                except Exception as e:
                    self.logger.error(f"Error at timestamp {timestamp}: {e}")
                    continue
            
            # Log final day stats if in debug mode
            if self.debug_mode and current_day and current_day in daily_stats:
                self.logger.info(f"🔍 DEBUG {current_day}: {daily_stats[current_day]['signals']} signals, {daily_stats[current_day]['trades']} trades, Balance: ${self.portfolio.current_balance:,.2f}")
                
                # Summary of all days
                self.logger.info(f"🔍 DEBUG SUMMARY:")
                total_signals = sum(stats['signals'] for stats in daily_stats.values())
                total_trades = sum(stats['trades'] for stats in daily_stats.values())
                self.logger.info(f"    Total simulation days: {len(daily_stats)}")
                self.logger.info(f"    Total signals generated: {total_signals}")
                self.logger.info(f"    Total trades executed: {total_trades}")
                self.logger.info(f"    Days with trading activity: {sum(1 for stats in daily_stats.values() if stats['trades'] > 0)}")
                self.logger.info(f"    Days with signals only: {sum(1 for stats in daily_stats.values() if stats['signals'] > 0 and stats['trades'] == 0)}")
                self.logger.info(f"    Days with no activity: {sum(1 for stats in daily_stats.values() if stats['signals'] == 0)}")
            
            # Close any remaining positions
            if self.portfolio.open_positions:
                final_timestamp = simulation_timestamps[-1]
                final_prices = {}
                for symbol, data in self.historical_data.items():
                    if final_timestamp in data['price'].index:
                        final_prices[symbol] = data['price'].loc[final_timestamp, 'price']
                
                for position_id in list(self.portfolio.open_positions.keys()):
                    position = self.portfolio.open_positions[position_id]
                    exit_price = final_prices.get(position.symbol, position.entry_price)
                    
                    trade_value = position.entry_price * position.size
                    exit_costs = self.fee_calculator.calculate_trading_costs(
                        symbol=position.symbol,
                        trade_value=trade_value,
                        holding_hours=position.duration_hours(timestamp)
                    )
                    
                    closed_position = self.portfolio.close_position(
                        position_id=position_id,
                        exit_price=exit_price,
                        timestamp=final_timestamp,
                        exit_reason="Backtest end",
                        trading_fees=exit_costs.total_cost
                    )
                    
                    # CRITICAL: Notify strategy that position was closed at backtest end
                    if closed_position and hasattr(self.strategy, 'confirm_position_closed'):
                        try:
                            self.strategy.confirm_position_closed(closed_position.symbol, closed_position, "Backtest end")
                        except Exception as e:
                            self.logger.error(f"Error notifying strategy of final position close: {e}")
            
            # Generate results and plots
            results = self.generate_results()
            self.create_plots()
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        try:
            # Get portfolio metrics
            portfolio_metrics = self.portfolio.get_performance_metrics()
            
            # Convert closed positions to trade history
            trade_history = []
            for position in self.portfolio.closed_positions:
                trade_history.append({
                    'symbol': position.symbol,
                    'side': position.position_type.value.lower(),
                    'entry_time': position.entry_time,
                    'exit_time': position.exit_time,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'size': position.size,
                    'pnl_pct': position.pnl_percentage,
                    'pnl_abs': position.pnl_absolute,
                    'duration_hours': position.duration_hours(),
                    'exit_reason': position.exit_reason,
                    'fees_paid': position.fees_paid
                })
            
            # Calculate additional metrics
            total_trading_costs = sum(pos.fees_paid for pos in self.portfolio.closed_positions)
            avg_trade_duration = portfolio_metrics.get('avg_trade_duration', 0)
            
            # Safe access to portfolio metrics with defaults and NaN protection
            current_balance = portfolio_metrics.get('current_balance', self.portfolio.current_balance)
            total_return = portfolio_metrics.get('total_return', 0.0)
            # Use actual balance change, not just position PnL
            total_pnl = current_balance - self.initial_balance
            
            # Protect against NaN values in results
            def safe_float(value, default=0.0):
                """Convert value to safe float, replacing NaN/inf with default"""
                if pd.isna(value) or np.isinf(value):
                    return default
                return float(value)
            
            current_balance = safe_float(current_balance, self.initial_balance)
            total_return = safe_float(total_return, 0.0)
            total_pnl = safe_float(total_pnl, 0.0)
            
            results = {
                'strategy_name': self.strategy.get_strategy_name() if self.strategy else 'Unknown',
                'backtest_type': 'MODULAR_SQUEEZEFLOW_BACKTEST',
                'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
                'symbols': self.symbols,
                
                # Portfolio performance
                'initial_balance': self.initial_balance,
                'final_balance': current_balance,
                'total_return_pct': total_return,
                'total_return_usd': total_pnl,
                
                # Trading metrics
                'total_trades': portfolio_metrics.get('total_trades', 0),
                'winning_trades': portfolio_metrics.get('winning_trades', 0),
                'losing_trades': portfolio_metrics.get('losing_trades', 0),
                'win_rate_pct': safe_float(portfolio_metrics.get('win_rate', 0.0)),
                'avg_trade_return_pct': safe_float(total_return / max(portfolio_metrics.get('total_trades', 1), 1)),
                'best_trade_pct': safe_float(portfolio_metrics.get('largest_win_pct', 0.0)),
                'worst_trade_pct': safe_float(portfolio_metrics.get('largest_loss_pct', 0.0)),
                'avg_duration_hours': safe_float(avg_trade_duration),
                
                # Risk metrics
                'max_drawdown_pct': safe_float(portfolio_metrics.get('max_drawdown', 0.0)),
                'sharpe_ratio': safe_float(portfolio_metrics.get('sharpe_ratio', 0.0)),
                'profit_factor': safe_float(portfolio_metrics.get('profit_factor', 0.0)),
                
                # Cost analysis
                'total_trading_costs_usd': total_trading_costs,
                'trading_costs_pct': (total_trading_costs / self.initial_balance) * 100,
                'avg_cost_per_trade': total_trading_costs / max(portfolio_metrics['total_trades'], 1),
                
                # Strategy configuration
                'strategy_name': self.strategy.get_strategy_name(),
                'strategy_config': self.strategy.get_config(),
                'engine_config': self.engine_config,
                'risk_limits': {
                    'max_position_size': self.portfolio.risk_limits.max_position_size,
                    'max_total_exposure': self.portfolio.risk_limits.max_total_exposure,
                    'max_open_positions': self.portfolio.risk_limits.max_open_positions
                },
                
                # Detailed data
                'trade_history': trade_history,
                'execution_log': self.execution_log,
                'portfolio_metrics': portfolio_metrics
            }
            
            self.logger.info(f"✅ Backtest completed: {results['total_return_pct']:.2f}% return "
                           f"({results['total_trades']} trades, {results['win_rate_pct']:.1f}% win rate)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating results: {e}")
            return {'error': str(e)}
    
    def create_plots(self):
        """Create comprehensive plots for backtest results"""
        try:
            self.logger.info("📊 Generating plots...")
            
            # Pass additional parameters to plotter
            self.plotter.show_full_range = self.show_full_range
            self.plotter.requested_start_date = self.start_date
            self.plotter.requested_end_date = self.end_date
            self.plotter.debug_mode = self.debug_mode
            
            # Create comprehensive plot for each symbol
            for symbol in self.symbols:
                if symbol in self.historical_data:
                    symbol_trades = [t for t in self.portfolio.closed_positions 
                                   if t.symbol == symbol]
                    
                    if symbol_trades:
                        # Convert positions to trade format for plotting
                        plot_trades = []
                        for pos in symbol_trades:
                            plot_trades.append({
                                'symbol': pos.symbol,
                                'side': pos.position_type.value.lower(),
                                'entry_time': pos.entry_time,
                                'exit_time': pos.exit_time,
                                'entry_price': pos.entry_price,
                                'exit_price': pos.exit_price,
                                'pnl_pct': pos.pnl_percentage,
                                'duration_minutes': pos.duration_hours() * 60,
                                'strategy_type': 'MODULAR'
                            })
                        
                        portfolio_metrics = self.portfolio.get_performance_metrics()
                        
                        filename = f'modular_backtest_{symbol.lower()}_results.png'
                        self.plotter.create_comprehensive_plot(
                            symbol=symbol,
                            historical_data=self.historical_data[symbol],
                            trades=plot_trades,
                            filename=filename,
                            portfolio_metrics=portfolio_metrics
                        )
                    else:
                        self.logger.info(f"No trades for {symbol}, skipping plot")
            
            # Create equity curve
            if self.portfolio.closed_positions:
                self.plotter.create_equity_curve_plot(
                    portfolio_manager=self.portfolio,
                    filename='modular_equity_curve.png'
                )
            
            # Create performance summary
            portfolio_metrics = self.portfolio.get_performance_metrics()
            self.plotter.create_performance_summary_plot(
                metrics=portfolio_metrics,
                filename='modular_performance_summary.png'
            )
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")


async def main():
    """Run backtest with command line arguments"""
    import argparse
    try:
        from .strategies import get_available_strategies
    except ImportError:
        from strategies import get_available_strategies
    
    parser = argparse.ArgumentParser(description='SqueezeFlow Modular Backtest Engine')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH'], help='Symbols to trade')
    parser.add_argument('--strategy', default='squeezeflow', choices=get_available_strategies(),
                       help='Trading strategy to use (default: squeezeflow)')
    parser.add_argument('--strategy-config', help='Strategy config JSON string')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--show-full-range', action='store_true', help='Show full requested date range in plots, not just trading periods')
    
    args = parser.parse_args()
    
    # Parse strategy configuration if provided
    strategy_config = {}
    if args.strategy_config:
        try:
            strategy_config = json.loads(args.strategy_config)
        except json.JSONDecodeError as e:
            print(f"Error parsing strategy config JSON: {e}")
            return
    
    # Load strategy
    try:
        strategy = load_strategy(args.strategy, strategy_config)
        print(f"🧠 Loaded strategy: {strategy.get_strategy_name()}")
        if strategy_config:
            print(f"📋 Strategy config: {strategy_config}")
    except ValueError as e:
        print(f"❌ Strategy loading error: {e}")
        return
    
    # Create risk limits
    risk_limits = RiskLimits(
        max_position_size=0.02,     # 2% per position
        max_total_exposure=0.1,     # 10% total exposure
        max_open_positions=2,       # Max 2 positions
        max_daily_loss=0.05,        # 5% daily loss limit
        max_drawdown=0.99          # 99% drawdown limit (disabled for backtesting)
    )
    
    # Run backtest
    engine = SqueezeFlowBacktestEngine(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.balance,
        symbols=args.symbols,
        risk_limits=risk_limits,
        strategy=strategy,
        debug_mode=args.debug,
        show_full_range=args.show_full_range
    )
    
    results = await engine.run_backtest()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"🚀 MODULAR SQUEEZEFLOW BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Strategy: {results['strategy_name']}")
    print(f"Period: {results['backtest_period']}")
    print(f"Symbols: {', '.join(results['symbols'])}")
    print(f"Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}% (${results['total_return_usd']:,.2f})")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.1f}% ({results['winning_trades']}/{results['total_trades']})")
    print(f"Average Trade: {results['avg_trade_return_pct']:.2f}%")
    print(f"Best Trade: {results['best_trade_pct']:.2f}%")
    print(f"Worst Trade: {results['worst_trade_pct']:.2f}%")
    print(f"Average Duration: {results['avg_duration_hours']:.1f} hours")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"\n💰 COST ANALYSIS:")
    print(f"Total Trading Costs: ${results['total_trading_costs_usd']:.2f} ({results['trading_costs_pct']:.3f}%)")
    print(f"Average Cost per Trade: ${results['avg_cost_per_trade']:.2f}")
    print(f"{'='*80}")
    
    # Save detailed results
    output_file = args.output
    if not output_file:
        # Auto-generate output file in logs directory
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(logs_dir, f"backtest_results_{timestamp}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save results to {output_file}: {e}")
    
    # Show usage instructions for new debug features
    if not args.debug and not args.show_full_range:
        print(f"\n💡 TIP: Use --debug to see detailed daily analysis")
        print(f"💡 TIP: Use --show-full-range to display full requested date range in plots")


if __name__ == "__main__":
    asyncio.run(main())