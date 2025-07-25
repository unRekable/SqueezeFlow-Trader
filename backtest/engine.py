#!/usr/bin/env python3
"""
SqueezeFlow Custom Backtest Engine
Tests the exact strategy implementation with historical InfluxDB data
"""

import asyncio
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage.influxdb_handler import InfluxDBHandler
from indicators.squeeze_score_calculator import SqueezeScoreCalculator
from utils.exchange_mapper import exchange_mapper


class SqueezeFlowBacktestEngine:
    """
    Custom Backtest Engine that replays the exact SqueezeFlow strategy
    using historical InfluxDB data and regenerated squeeze signals
    """
    
    def __init__(self, start_date: str, end_date: str, initial_balance: float = 10000):
        self.start_date = pd.to_datetime(start_date).tz_localize('UTC')
        self.end_date = pd.to_datetime(end_date).tz_localize('UTC')
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        self.setup_logging()
        self.setup_database()
        self.setup_calculator()
        
        # Strategy configuration - base symbols for robust market discovery
        self.symbols = ['BTC', 'ETH']
        self.lookback_periods = [5, 10, 15, 20, 30, 60, 120, 240]  # Include 20-minute (primary signal)
        self.squeeze_threshold = 0.3  # Lowered for cumulative CVD (was 0.5 for per-minute deltas)
        self.max_open_trades = 2
        
        # Trading state
        self.open_positions = {}
        self.trade_history = []
        
        # Signal Persistence System (EXACT copy from FreqTrade strategy)
        self.signal_cache = {}  # Per-pair signal cache with timing
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SqueezeFlowBacktest')
        
    def setup_database(self):
        """Setup InfluxDB connection to historical data"""
        from influxdb import InfluxDBClient
        
        self.influx_client = InfluxDBClient(
            host=os.getenv('INFLUX_HOST', 'localhost'),
            port=int(os.getenv('INFLUX_PORT', 8086)),
            database='significant_trades'  # Same database as live system
        )
        
        try:
            databases = self.influx_client.get_list_database()
            self.logger.info(f"Connected to InfluxDB: {[db['name'] for db in databases]}")
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            raise
            
    def setup_calculator(self):
        """Setup squeeze score calculator (exact same as live)"""
        self.calculator = SqueezeScoreCalculator(
            price_weight=0.3,
            spot_cvd_weight=0.35,
            futures_cvd_weight=0.35,
            smoothing_period=5
        )
        
    async def get_historical_price_data(self, symbol: str, start_time: datetime, 
                                       end_time: datetime) -> pd.DataFrame:
        """
        Get historical price data from InfluxDB
        """
        try:
            # Get all markets for price calculation using robust Market Discovery
            from utils.market_discovery import market_discovery
            markets_by_type = market_discovery.get_markets_by_type(symbol)
            market_list = markets_by_type['spot'] + markets_by_type['perp']
            
            if not market_list:
                self.logger.warning(f"No markets found for {symbol}")
                return pd.DataFrame()
            
            market_filter = ' OR '.join([f"market = '{market}'" for market in market_list])
            
            query = f"""
            SELECT mean(close) as close
            FROM "aggr_1m"."trades_1m"
            WHERE ({market_filter})
            AND time >= '{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
            AND time <= '{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
            GROUP BY time(1m)
            ORDER BY time ASC
            """
            
            result = self.influx_client.query(query)
            points = list(result.get_points())
            
            if not points:
                return pd.DataFrame()
                
            df = pd.DataFrame(points)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Rename close to price for compatibility with squeeze calculator
            df = df.rename(columns={'close': 'price'})
            
            # DEBUG: Check what we actually got
            if not df.empty:
                if not df['price'].isna().all():
                    sample_price = df['price'].iloc[-1]
                    self.logger.info(f"🔍 Price data loaded for {symbol}: {len(df)} points, sample: {sample_price:.2f}")
                else:
                    self.logger.info(f"🔍 Price data loaded for {symbol}: {len(df)} points, but ALL NaN")
            else:
                self.logger.warning(f"🔍 Price data EMPTY for {symbol}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def calculate_historical_cvd(self, symbol: str, start_time: datetime, 
                                     end_time: datetime) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate historical CVD data from trades (exact same logic as live)
        """
        try:
            # Get spot and perp markets using robust Market Discovery
            from utils.market_discovery import market_discovery
            markets_by_type = market_discovery.get_markets_by_type(symbol)
            
            spot_markets = markets_by_type['spot']
            perp_markets = markets_by_type['perp']
            
            if not spot_markets or not perp_markets:
                self.logger.warning(f"Missing markets for {symbol}: spot={len(spot_markets)}, perp={len(perp_markets)}")
                return pd.Series(dtype=float), pd.Series(dtype=float)
            
            # Build spot CVD query (exact same as live calculator)
            spot_market_filter = ' OR '.join([f"market = '{market}'" for market in spot_markets])
            spot_query = f"""
            SELECT sum(vbuy) as total_vbuy_spot, sum(vsell) as total_vsell_spot
            FROM "aggr_1m"."trades_1m"
            WHERE ({spot_market_filter})
            AND time >= '{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
            AND time <= '{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
            GROUP BY time(1m)
            ORDER BY time ASC
            """
            
            # Build perp CVD query (exact same as live calculator)
            perp_market_filter = ' OR '.join([f"market = '{market}'" for market in perp_markets])
            perp_query = f"""
            SELECT sum(vbuy) as total_vbuy_perp, sum(vsell) as total_vsell_perp
            FROM "aggr_1m"."trades_1m"
            WHERE ({perp_market_filter})
            AND time >= '{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
            AND time <= '{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
            GROUP BY time(1m)
            ORDER BY time ASC
            """
            
            spot_result = self.influx_client.query(spot_query)
            perp_result = self.influx_client.query(perp_query)
            
            spot_points = list(spot_result.get_points())
            perp_points = list(perp_result.get_points())
            
            # Process spot CVD (VERIFIED METHODOLOGY - same as debug tool)
            if spot_points:
                spot_df = pd.DataFrame(spot_points)
                spot_df['time'] = pd.to_datetime(spot_df['time'])
                spot_df['total_vbuy_spot'] = spot_df['total_vbuy_spot'].fillna(0)
                spot_df['total_vsell_spot'] = spot_df['total_vsell_spot'].fillna(0)
                # Step 1: Calculate per-minute volume delta (Buy Volume - Sell Volume)
                spot_df['total_cvd_spot'] = spot_df['total_vbuy_spot'] - spot_df['total_vsell_spot']
                spot_df = spot_df.set_index('time').sort_index()
                # Step 2: Calculate CUMULATIVE Volume Delta (running total) - EXACT same logic as debug tool
                spot_df['total_cvd_spot_cumulative'] = spot_df['total_cvd_spot'].cumsum()
                spot_cvd = spot_df['total_cvd_spot_cumulative']  # Use cumulative CVD, not per-minute delta
            else:
                spot_cvd = pd.Series(dtype=float)
                
            # Process perp CVD (VERIFIED METHODOLOGY - same as debug tool)
            if perp_points:
                perp_df = pd.DataFrame(perp_points)
                perp_df['time'] = pd.to_datetime(perp_df['time'])
                perp_df['total_vbuy_perp'] = perp_df['total_vbuy_perp'].fillna(0)
                perp_df['total_vsell_perp'] = perp_df['total_vsell_perp'].fillna(0)
                # Step 1: Calculate per-minute volume delta (Buy Volume - Sell Volume)
                perp_df['total_cvd_perp'] = perp_df['total_vbuy_perp'] - perp_df['total_vsell_perp']
                perp_df = perp_df.set_index('time').sort_index()
                # Step 2: Calculate CUMULATIVE Volume Delta (running total) - EXACT same logic as debug tool
                perp_df['total_cvd_perp_cumulative'] = perp_df['total_cvd_perp'].cumsum()
                perp_cvd = perp_df['total_cvd_perp_cumulative']  # Use cumulative CVD, not per-minute delta
            else:
                perp_cvd = pd.Series(dtype=float)
                
            self.logger.info(f"Historical CVD calculated for {symbol}: spot={len(spot_cvd)}, perp={len(perp_cvd)} points")
            
            return spot_cvd, perp_cvd
            
        except Exception as e:
            self.logger.error(f"Error calculating historical CVD for {symbol}: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
    def get_external_squeeze_signals(self, symbol: str, current_time: datetime, 
                                   signals_by_timeframe: Dict) -> Dict:
        """
        EXACT copy of FreqTrade strategy's get_external_squeeze_signals method
        Including signal persistence logic with timing
        """
        try:
            # Build current signals dict (same format as Redis)
            current_signals = {}
            signal_count = 0
            
            for lookback in [5, 10, 15, 20, 30, 60, 120, 240]:  # Extended timeframes (exact match with calculation)
                if lookback in signals_by_timeframe:
                    signal = signals_by_timeframe[lookback]
                    current_signals[f'squeeze_score_{lookback}'] = signal.get('squeeze_score', 0)
                    current_signals[f'signal_strength_{lookback}'] = signal.get('signal_strength', 0)
                    current_signals[f'signal_type_{lookback}'] = signal.get('signal_type', 'NEUTRAL')
                    signal_count += 1
                else:
                    current_signals[f'squeeze_score_{lookback}'] = 0
                    current_signals[f'signal_strength_{lookback}'] = 0
                    current_signals[f'signal_type_{lookback}'] = 'NEUTRAL'
            
            # SIGNAL PERSISTENCE LOGIC (EXACT copy from FreqTrade)
            current_score = current_signals.get('squeeze_score_20', 0)
            
            # Initialize cache for this symbol
            if symbol not in self.signal_cache:
                self.signal_cache[symbol] = {
                    'score': 0, 
                    'timestamp': current_time, 
                    'active': False,
                    'strength': 0,
                    'type': 'NEUTRAL'
                }
            
            cache = self.signal_cache[symbol]
            
            # Check for new strong signal
            if abs(current_score) >= 0.3:  # Strong signal threshold
                # Update cache with new signal
                cache['score'] = current_score
                cache['timestamp'] = current_time
                cache['active'] = True
                cache['strength'] = current_signals.get('signal_strength_20', 0)
                cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
                
                self.logger.info(f"🟢 NEW SIGNAL {symbol}: {current_score:.3f} - Cached for persistence")
            
            # Check if cached signal is still valid (5-10 minute persistence)
            time_diff = (current_time - cache['timestamp']).total_seconds() / 60  # minutes
            
            if cache['active'] and time_diff <= 10:  # Signal valid for 10 minutes
                # Use cached signal
                return_signals = current_signals.copy()
                return_signals['squeeze_score_20'] = cache['score']
                return_signals['signal_strength_20'] = cache['strength']
                return_signals['signal_type_20'] = cache['type']
                return_signals['signal_active'] = True
                return_signals['signal_age_minutes'] = time_diff
                
                self.logger.debug(f"🔄 USING CACHED SIGNAL {symbol}: {cache['score']:.3f} (age: {time_diff:.1f}min)")
            else:
                # Signal expired or weak - deactivate
                if cache['active'] and time_diff > 10:
                    cache['active'] = False
                    self.logger.info(f"⏰ SIGNAL EXPIRED {symbol}: {cache['score']:.3f} (age: {time_diff:.1f}min)")
                
                # Use current weak values
                return_signals = current_signals.copy()
                return_signals['signal_active'] = False
                return_signals['signal_age_minutes'] = 0
            
            return return_signals
            
        except Exception as e:
            self.logger.error(f"Error getting squeeze signals for {symbol}: {e}")
            return {}
            
    async def regenerate_squeeze_signals(self, symbol: str, price_data: pd.Series,
                                       spot_cvd: pd.Series, perp_cvd: pd.Series,
                                       current_time: datetime) -> Dict:
        """
        Regenerate squeeze signals using exact same calculator as live system
        """
        signals_by_timeframe = {}
        
        # Calculate for all timeframes (exact same as live)
        for lookback in self.lookback_periods:
            try:
                result = self.calculator.calculate_squeeze_score(
                    price_data=price_data,
                    spot_cvd_data=spot_cvd,
                    futures_cvd_data=perp_cvd,
                    lookback=lookback
                )
                
                signals_by_timeframe[lookback] = result
                
            except Exception as e:
                self.logger.error(f"Error calculating squeeze for {symbol} lookback {lookback}: {e}")
                signals_by_timeframe[lookback] = {
                    'squeeze_score': 0.0,
                    'signal_type': 'neutral',
                    'signal_strength': 0.0
                }
        
        # Apply signal persistence logic (EXACT same as FreqTrade)
        persistent_signals = self.get_external_squeeze_signals(symbol, current_time, signals_by_timeframe)
        
        return persistent_signals
        
    def simulate_strategy_entry(self, symbol: str, current_time: datetime, 
                               signals: Dict, price: float) -> Optional[Dict]:
        """
        EXACT copy of FreqTrade strategy's populate_entry_trend logic
        """
        # Check if we already have a position for this symbol
        if symbol in self.open_positions:
            return None
            
        # Check if we have reached max open trades
        if len(self.open_positions) >= self.max_open_trades:
            return None
            
        # Get persistent signals with timing info (EXACT same as FreqTrade)
        signal_active = signals.get('signal_active', False)
        signal_age = signals.get('signal_age_minutes', 0)
        squeeze_score = signals.get('squeeze_score_20', 0)
        
        # DEBUG: Log current signal status (same as FreqTrade)
        self.logger.info(f"🔍 ENTRY CHECK {symbol}: score={squeeze_score:.3f}, active={signal_active}, age={signal_age:.1f}min")
        
        # LONG ENTRY CONDITIONS (EXACT copy from FreqTrade strategy lines 482-496)
        long_conditions = [
            # PRIMARY: Signal active and fresh (< 5 minutes)
            (squeeze_score <= -self.squeeze_threshold),
            (signal_active == True),  # Signal must be active
            (signal_age < 5),  # Signal max 5 minutes old
            # CONFIRMATION: Multiple timeframe alignment with extended timeframes
            # Entry timing: 10min or 30min confirmation
            ((signals.get('squeeze_score_10', 0) <= -0.15) or (signals.get('squeeze_score_30', 0) <= -0.2)),
            # PRIMARY timeframes: 60min (1h), 120min (2h) für echte Squeeze-Signale
            ((signals.get('squeeze_score_60', 0) <= -0.3) or (signals.get('squeeze_score_120', 0) <= -0.4)),
        ]
        
        # SHORT ENTRY CONDITIONS (EXACT copy from FreqTrade strategy lines 498-512)
        short_conditions = [
            # PRIMARY: Signal active and fresh (< 5 minutes)
            (squeeze_score >= self.squeeze_threshold),
            (signal_active == True),  # Signal must be active
            (signal_age < 5),  # Signal max 5 minutes old  
            # CONFIRMATION: Multiple timeframe alignment
            # Entry timing: 10min or 30min confirmation
            ((signals.get('squeeze_score_10', 0) >= 0.15) or (signals.get('squeeze_score_30', 0) >= 0.2)),
            # PRIMARY timeframes: 60min (1h), 120min (2h) für echte Squeeze-Signale
            ((signals.get('squeeze_score_60', 0) >= 0.3) or (signals.get('squeeze_score_120', 0) >= 0.4)),
        ]
        
        # Check entry conditions
        if all(long_conditions):
            side = 'long'
            entry_reason = f"Long squeeze: score={squeeze_score:.3f}, age={signal_age:.1f}min"
        elif all(short_conditions):
            side = 'short'
            entry_reason = f"Short squeeze: score={squeeze_score:.3f}, age={signal_age:.1f}min"
        else:
            return None
            
        # Calculate position size (same as FreqTrade: 45% per position)
        position_size = self.current_balance * 0.45  # 45% per position (90% total for 2 positions)
        
        trade = {
            'symbol': symbol,
            'side': side,
            'entry_time': current_time,
            'entry_price': price,
            'size': position_size,
            'entry_reason': entry_reason,
            'entry_signals': signals.copy()
        }
        
        self.open_positions[symbol] = trade
        self.logger.info(f"🟢 ENTRY {side.upper()} {symbol} @ {price:.2f} - {entry_reason}")
        
        return trade
        
    def simulate_strategy_exit(self, symbol: str, current_time: datetime,
                              signals: Dict, price: float) -> Optional[Dict]:
        """
        IMPROVED: Reverse Exit Logic - Exit on opposite squeeze signal
        """
        if symbol not in self.open_positions:
            return None
            
        position = self.open_positions[symbol]
        side = position['side']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Calculate current PnL
        if side == 'long':
            pnl_pct = (price - entry_price) / entry_price * 100
        else:  # short
            pnl_pct = (entry_price - price) / entry_price * 100
            
        # Get squeeze score for exit decision
        position_age_minutes = (current_time - entry_time).total_seconds() / 60
        squeeze_score = signals.get('squeeze_score_20', 0)
        
        # REVERSE EXIT LOGIC - Simple and effective!
        exit_reason = None
        
        if side == 'long':
            # Exit LONG position when SHORT squeeze signal appears
            if squeeze_score >= self.squeeze_threshold:  # +0.5
                exit_reason = f"Reverse signal (Short squeeze): {squeeze_score:.3f}"
                
        elif side == 'short':
            # Exit SHORT position when LONG squeeze signal appears  
            if squeeze_score <= -self.squeeze_threshold:  # -0.5
                exit_reason = f"Reverse signal (Long squeeze): {squeeze_score:.3f}"
        
        # Emergency exits (keep minimal safety nets)
        if not exit_reason:
            # Hard stop loss
            if pnl_pct <= -2.5:
                exit_reason = f"Stop loss: {pnl_pct:.2f}%"
            # Maximum position age (24 hours)
            elif position_age_minutes > 1440:  # 24 hours
                exit_reason = f"Max age exceeded: {position_age_minutes:.0f}min"
            
        # Log exit check
        self.logger.debug(f"🔍 EXIT CHECK {symbol}: age={position_age_minutes:.1f}min, score={squeeze_score:.3f}, side={side}")
            
        if exit_reason:
            # Execute exit
            trade_result = {
                'symbol': symbol,
                'side': side,
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'size': position['size'],
                'pnl_pct': pnl_pct,
                'pnl_usd': position['size'] * pnl_pct / 100,
                'exit_reason': exit_reason,
                'duration_minutes': position_age_minutes
            }
            
            # Update balance
            self.current_balance += trade_result['pnl_usd']
            
            # Remove position
            del self.open_positions[symbol]
            
            # Add to history
            self.trade_history.append(trade_result)
            
            self.logger.info(f"🔴 EXIT {side.upper()} {symbol} @ {price:.2f} - {exit_reason} - PnL: {pnl_pct:.2f}%")
            
            return trade_result
            
        return None
        
    async def run_backtest(self) -> Dict:
        """
        Run the complete backtest simulation
        """
        self.logger.info(f"🚀 Starting backtest: {self.start_date} to {self.end_date}")
        
        # Get historical data for all symbols
        historical_data = {}
        for symbol in self.symbols:
            self.logger.info(f"Loading historical data for {symbol}...")
            
            price_data = await self.get_historical_price_data(symbol, self.start_date, self.end_date)
            spot_cvd, perp_cvd = await self.calculate_historical_cvd(symbol, self.start_date, self.end_date)
            
            if price_data.empty or spot_cvd.empty or perp_cvd.empty:
                self.logger.warning(f"Insufficient data for {symbol}, skipping...")
                continue
                
            historical_data[symbol] = {
                'price': price_data,
                'spot_cvd': spot_cvd,
                'perp_cvd': perp_cvd
            }
            
        if not historical_data:
            raise ValueError("No historical data available for any symbol")
            
        # Get all timestamps (union of all symbols)
        all_timestamps = set()
        for data in historical_data.values():
            all_timestamps.update(data['price'].index)
        all_timestamps = sorted(all_timestamps)
        
        # Analyze valid data range for each symbol
        valid_ranges = {}
        for symbol, data in historical_data.items():
            valid_mask = ~data['price']['price'].isna()
            if valid_mask.any():
                first_valid_idx = valid_mask.argmax()  # First True index
                first_valid_time = data['price'].index[first_valid_idx]
                last_valid_time = data['price']['price'].last_valid_index()
                valid_count = valid_mask.sum()
                valid_ranges[symbol] = {
                    'first_idx': first_valid_idx,
                    'first_time': first_valid_time,
                    'last_time': last_valid_time,
                    'valid_count': valid_count,
                    'total_count': len(data['price'])
                }
                self.logger.info(f"📊 {symbol} valid data: {first_valid_time} to {last_valid_time} ({valid_count}/{len(data['price'])} points)")
            else:
                self.logger.error(f"❌ {symbol}: No valid data found!")
                
        if not valid_ranges:
            raise ValueError("No symbols have valid data for backtesting")
            
        # Find the latest start time (intersection of valid periods)
        latest_start_time = max(ranges['first_time'] for ranges in valid_ranges.values())
        earliest_end_time = min(ranges['last_time'] for ranges in valid_ranges.values())
        
        # Find index in all_timestamps
        start_idx = next(i for i, ts in enumerate(all_timestamps) if ts >= latest_start_time)
        start_idx = max(start_idx, max(self.lookback_periods) + 10)  # Ensure enough lookback data
        valid_timestamps = all_timestamps[start_idx:]
        
        self.logger.info(f"Backtesting {len(valid_timestamps)} time periods (skipped {start_idx} initial periods with insufficient data)...")
        
        # Simulate trading minute by minute (sample every 5 minutes for speed)
        sampled_timestamps = valid_timestamps[::5]  # Every 5th timestamp
        for i, timestamp in enumerate(sampled_timestamps):
            # Generate signals for all symbols at this timestamp
            current_signals = {}
            current_prices = {}
            
            for symbol, data in historical_data.items():
                if timestamp in data['price'].index:
                    current_prices[symbol] = data['price'].loc[timestamp, 'price']
                    
                    # Get data up to current timestamp for signal calculation (only valid data range)
                    valid_start_time = valid_ranges[symbol]['first_time']
                    price_slice = data['price'].loc[valid_start_time:timestamp, 'price']  # Only valid data range
                    spot_slice = data['spot_cvd'].loc[valid_start_time:timestamp]  
                    perp_slice = data['perp_cvd'].loc[valid_start_time:timestamp]
                    
                    if len(price_slice) >= max(self.lookback_periods) and not price_slice.isna().all():
                            
                        signals = await self.regenerate_squeeze_signals(
                            symbol, price_slice, spot_slice, perp_slice, timestamp
                        )
                        current_signals[symbol] = signals
                        
            # Show progress every 100 steps
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{len(sampled_timestamps)} ({i/len(sampled_timestamps)*100:.1f}%)")
                        
            # Process trading logic for each symbol
            for symbol in current_signals:
                if symbol in current_prices:
                    price = current_prices[symbol]
                    signals = current_signals[symbol]
                    
                    # Check for exits first
                    self.simulate_strategy_exit(symbol, timestamp, signals, price)
                    
                    # Then check for entries
                    self.simulate_strategy_entry(symbol, timestamp, signals, price)
                    
        # Close any remaining positions at end
        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]
            final_price = current_prices.get(symbol, position['entry_price'])
            
            if position['side'] == 'long':
                pnl_pct = (final_price - position['entry_price']) / position['entry_price'] * 100
            else:
                pnl_pct = (position['entry_price'] - final_price) / position['entry_price'] * 100
                
            trade_result = {
                'symbol': symbol,
                'side': position['side'],
                'entry_time': position['entry_time'],
                'exit_time': self.end_date,
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'size': position['size'],
                'pnl_pct': pnl_pct,
                'pnl_usd': position['size'] * pnl_pct / 100,
                'exit_reason': 'Backtest end',
                'duration_minutes': (self.end_date - position['entry_time']).total_seconds() / 60
            }
            
            self.current_balance += trade_result['pnl_usd']
            self.trade_history.append(trade_result)
            
        return self.generate_report()
        
    def generate_report(self) -> Dict:
        """
        Generate detailed backtest report
        """
        if not self.trade_history:
            return {
                'error': 'No trades executed during backtest period',
                'total_trades': 0,
                'total_return': 0.0
            }
            
        df = pd.DataFrame(self.trade_history)
        
        # Calculate metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl_pct'] > 0])
        losing_trades = len(df[df['pnl_pct'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_return_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        avg_trade_return = df['pnl_pct'].mean()
        max_win = df['pnl_pct'].max()
        max_loss = df['pnl_pct'].min()
        
        avg_duration = df['duration_minutes'].mean()
        
        report = {
            'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return_pct': total_return_pct,
            'total_return_usd': self.current_balance - self.initial_balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_trade_return,
            'max_win_pct': max_win,
            'max_loss_pct': max_loss,
            'avg_duration_minutes': avg_duration,
            'trades_per_day': total_trades / (self.end_date - self.start_date).days if (self.end_date - self.start_date).days > 0 else total_trades,
            'strategy_config': {
                'squeeze_threshold': self.squeeze_threshold,
                'max_open_trades': self.max_open_trades,
                'symbols': self.symbols,
                'timeframes': self.lookback_periods
            },
            'trade_history': self.trade_history
        }
        
        return report


async def main():
    """Run backtest with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SqueezeFlow Custom Backtest Engine')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Run backtest
    engine = SqueezeFlowBacktestEngine(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.balance
    )
    
    report = await engine.run_backtest()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SQUEEZEFLOW BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Period: {report['backtest_period']}")
    print(f"Initial Balance: ${report['initial_balance']:,.2f}")
    print(f"Final Balance: ${report['final_balance']:,.2f}")
    print(f"Total Return: {report['total_return_pct']:.2f}% (${report['total_return_usd']:,.2f})")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate_pct']:.1f}% ({report['winning_trades']}/{report['total_trades']})")
    print(f"Average Trade: {report['avg_trade_return_pct']:.2f}%")
    print(f"Best Trade: {report['max_win_pct']:.2f}%")
    print(f"Worst Trade: {report['max_loss_pct']:.2f}%")
    print(f"Average Duration: {report['avg_duration_minutes']:.1f} minutes")
    print(f"{'='*60}")
    
    # Save detailed report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Detailed report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())