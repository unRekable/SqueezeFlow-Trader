#!/usr/bin/env python3
"""
SqueezeFlow FreqAI Strategy - Phase 5 Pure Execution Layer
ARCHITECTURE: strategy_runner → Redis → FreqTrade (pure execution)
SIMPLIFIED: Only reads Redis signals and executes trades - NO strategy logic
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import redis
import json
import uuid
from datetime import datetime, timezone

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class SqueezeFlowFreqAI(IStrategy):
    """
    SqueezeFlow strategy - Phase 5 Pure Execution Layer
    Only executes trades based on Redis signals from strategy_runner
    NO strategy calculations or decision making
    """
    
    # Strategy information
    STRATEGY_VERSION = "Phase5-Pure-Execution"
    
    # Basic FreqTrade settings
    minimal_roi = {"0": 100}  # Disable ROI, rely on signals
    stoploss = -0.08  # Basic stop loss
    timeframe = '5m'
    
    # FreqTrade settings
    can_short = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    startup_candle_count = 30  # Minimal startup
    
    def __init__(self, config: Dict) -> None:
        """Initialize strategy with Redis connection only"""
        super().__init__(config)
        
        # Redis connection for signal consumption
        self.redis_client = None
        self.setup_redis_connection()
        
        # Signal tracking to prevent duplicates
        self.executed_signals = set()  # Track executed signal IDs
        self.signal_cache = {}  # Cache for signal validation
        
        logger.info("🚀 SqueezeFlow Phase 5 Pure Execution Layer initialized")
        
    def setup_redis_connection(self):
        """Setup Redis connection for signal reading"""
        try:
            # Try Docker hostname first, then fallback to localhost
            redis_hosts = ['redis', 'localhost']
            connected_host = None
            
            for redis_host in redis_hosts:
                try:
                    redis_url = f'redis://{redis_host}:6379'
                    
                    pool = redis.ConnectionPool.from_url(
                        redis_url, 
                        decode_responses=True,
                        max_connections=5,
                        socket_connect_timeout=2,
                        socket_timeout=2,
                        retry_on_timeout=True
                    )
                    test_client = redis.Redis(connection_pool=pool)
                    test_client.ping()
                    
                    # Connection successful
                    self.redis_client = test_client
                    connected_host = redis_host
                    logger.info(f"✅ Redis connected: {redis_host}:6379")
                    return
                    
                except Exception as e:
                    logger.debug(f"Failed to connect to {redis_host}:6379: {e}")
                    continue
            
            # If we get here, no connection worked
            raise Exception(f"Could not connect to Redis on any of: {redis_hosts}")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def get_redis_signals(self, pair: str) -> Dict:
        """
        Get trading signals from Redis
        Expected format from strategy_runner:
        {
            "signal_id": "unique-uuid",
            "timestamp": "2025-01-01T00:00:00Z",
            "symbol": "BTCUSDT", 
            "action": "LONG",              // LONG/SHORT/CLOSE
            "score": 7.5,                  // Total score from 10-point system
            "position_size_factor": 1.0,   // 0.5/1.0/1.5 based on score
            "leverage": 3,                 // 2/3/5 based on score
            "entry_price": 50000,
            "ttl": 300                     // Expires in 5 minutes
        }
        """
        if not self.redis_client:
            return {}
            
        try:
            # Convert pair to symbol (BTC/USDT:USDT -> BTCUSDT)  
            # Handle futures format: BTC/USDT:USDT -> BTCUSDT
            if ':' in pair:
                symbol = pair.split(':')[0].replace('/', '')  # BTC/USDT:USDT -> BTC/USDT -> BTCUSDT
            else:
                symbol = pair.replace('/', '')  # BTC/USDT -> BTCUSDT
            
            # Get latest signal for this symbol
            # Use the correct key format that matches strategy_runner
            signal_key = f"squeezeflow:signal:{symbol}"
            logger.info(f"🔎 Looking for Redis key: {signal_key} (pair: {pair})")
            signal_data = self.redis_client.get(signal_key)
            
            if not signal_data:
                # Debug: Check what keys actually exist
                all_keys = self.redis_client.keys("squeezeflow:signal:*")
                logger.info(f"🗝️ Key {signal_key} not found. Available keys: {all_keys}")
                return {}
            
            signal = json.loads(signal_data)
            
            # Validate signal timestamp (check TTL)
            # Handle both naive and aware timestamps
            timestamp_str = signal['timestamp']
            if 'T' in timestamp_str and not ('+' in timestamp_str or 'Z' in timestamp_str):
                # Naive timestamp, assume UTC
                signal_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            else:
                # Aware timestamp
                signal_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            age_seconds = (datetime.now(timezone.utc) - signal_time).total_seconds()
            
            if age_seconds > signal.get('ttl', 300):  # Default 5 minutes TTL
                logger.debug(f"⏰ Signal expired for {symbol}: {age_seconds:.1f}s old")
                return {}
            
            # Check if already executed
            signal_id = signal.get('signal_id', '')
            if signal_id in self.executed_signals:
                return {}
            
            logger.info(f"📡 Signal received for {pair}: {signal['action']} "
                       f"score={signal.get('score', 0):.1f} "
                       f"age={age_seconds:.1f}s")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error reading Redis signal for {pair}: {e}")
            return {}
    
    def mark_signal_executed(self, signal_id: str):
        """Mark signal as executed to prevent duplicates"""
        if signal_id:
            self.executed_signals.add(signal_id)
            
            # Keep only last 100 executed signals to prevent memory growth
            if len(self.executed_signals) > 100:
                # Remove oldest 50 signals
                executed_list = list(self.executed_signals)
                self.executed_signals = set(executed_list[-50:])
    
    def bot_loop_start(self, **kwargs) -> None:
        """
        Called on every bot iteration (every few seconds).
        Check for new Redis signals in real-time and cache them.
        """
        # Cache latest signals for all pairs to ensure real-time execution
        try:
            # Get all pairs from config
            pairs = self.dp.current_whitelist() if hasattr(self, 'dp') else []
            
            for pair in pairs:
                # Get fresh signal from Redis
                signal = self.get_redis_signals(pair)
                
                if signal and signal.get('action') != 'NONE':
                    # Store in instance cache for immediate use
                    if not hasattr(self, '_signal_cache'):
                        self._signal_cache = {}
                    
                    self._signal_cache[pair] = {
                        'signal': signal,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    
                    # Log new signals
                    if signal.get('signal_id') not in self.executed_signals:
                        logger.info(f"🚨 NEW SIGNAL DETECTED for {pair}: {signal.get('action')} score={signal.get('score'):.1f}")
                        
        except Exception as e:
            logger.error(f"Error in bot_loop_start: {e}")
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Minimal indicators - only what's needed for signal execution"""
        
        # Debug logging
        pair = metadata['pair']
        logger.info(f"🔍 populate_indicators called for {pair}, dataframe shape: {dataframe.shape}")
        
        # Initialize signal columns with default values
        dataframe['signal_action'] = 'NONE'
        dataframe['signal_score'] = 0.0
        dataframe['signal_id'] = ''
        dataframe['position_size_factor'] = 1.0
        dataframe['signal_leverage'] = 3.0
        
        # Basic price validation
        dataframe['price_valid'] = (dataframe['close'] > 0) & (dataframe['volume'] > 0)
        
        # Get signal from cache (populated by bot_loop_start) or Redis
        if not dataframe.empty:
            pair = metadata['pair']
            latest_idx = dataframe.index[-1]
            
            # First check cached signal from bot_loop_start
            signal = None
            if hasattr(self, '_signal_cache') and pair in self._signal_cache:
                cached = self._signal_cache[pair]
                # Use cached signal if it's fresh (less than 5 minutes old)
                if (datetime.now(timezone.utc) - cached['timestamp']).total_seconds() < 300:
                    signal = cached['signal']
                    logger.debug(f"📦 Using cached signal for {pair}")
            
            # Fall back to direct Redis check if no cached signal
            if not signal:
                signal = self.get_redis_signals(pair)
            
            if signal:
                # Populate signal data in dataframe for latest candle
                dataframe.loc[latest_idx, 'signal_action'] = signal.get('action', 'NONE')
                dataframe.loc[latest_idx, 'signal_score'] = signal.get('score', 0.0)
                dataframe.loc[latest_idx, 'signal_id'] = signal.get('signal_id', '')
                dataframe.loc[latest_idx, 'position_size_factor'] = signal.get('position_size_factor', 1.0)
                dataframe.loc[latest_idx, 'signal_leverage'] = signal.get('leverage', 3.0)
                
                logger.info(f"📈 {pair} signal populated: {signal.get('action')} score={signal.get('score')}")
            else:
                logger.debug(f"⚠️ {pair} no signal found")
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Pure signal execution - read signals from populated dataframe"""
        
        pair = metadata['pair']
        logger.info(f"🎯 populate_entry_trend called for {pair}, dataframe shape: {dataframe.shape}")
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        # Only check signals for the latest candle (live trading)
        # In backtesting, this will check each historical candle
        latest_idx = dataframe.index[-1]
        
        # Read signal data from populated indicators
        signal_action = dataframe.loc[latest_idx, 'signal_action']
        signal_id = dataframe.loc[latest_idx, 'signal_id']
        signal_score = dataframe.loc[latest_idx, 'signal_score']
        
        logger.info(f"📊 {pair} latest signal: action={signal_action}, score={signal_score}, id={signal_id[:8] if signal_id else 'N/A'}")
        
        if signal_action == 'NONE' or signal_score <= 0:
            # Log every 10th attempt to avoid spam
            if hasattr(self, '_no_signal_count'):
                self._no_signal_count += 1
            else:
                self._no_signal_count = 1
            
            if self._no_signal_count % 10 == 0:
                logger.debug(f"🔍 No valid signal found for {pair} (attempt {self._no_signal_count})")
            return dataframe
        
        # Validate price data
        if not dataframe.loc[latest_idx, 'price_valid']:
            logger.warning(f"⚠️ Invalid price data for {pair}, skipping signal")
            return dataframe
        
        # Check if signal already executed
        if signal_id and signal_id in self.executed_signals:
            return dataframe
        
        # Execute LONG signals
        if signal_action == 'LONG':
            dataframe.loc[latest_idx, 'enter_long'] = 1
            # Add entry tag for tracking
            dataframe.loc[latest_idx, 'enter_tag'] = f"squeeze_long_s{signal_score:.1f}"
            # Don't mark as executed yet - wait for trade confirmation
            
            logger.info(f"🟢 LONG ENTRY SIGNAL: {pair} "
                       f"score={signal_score:.1f} "
                       f"id={signal_id[:8] if signal_id else 'N/A'}...")
        
        # Execute SHORT signals  
        elif signal_action == 'SHORT':
            dataframe.loc[latest_idx, 'enter_short'] = 1
            # Add entry tag for tracking
            dataframe.loc[latest_idx, 'enter_tag'] = f"squeeze_short_s{signal_score:.1f}"
            # Don't mark as executed yet - wait for trade confirmation
            
            logger.info(f"🔴 SHORT ENTRY SIGNAL: {pair} "
                       f"score={signal_score:.1f} "
                       f"id={signal_id[:8] if signal_id else 'N/A'}...")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Pure signal execution for exits - read signals from populated dataframe"""
        
        pair = metadata['pair']
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        latest_idx = dataframe.index[-1]
        
        # Read signal data from populated indicators
        signal_action = dataframe.loc[latest_idx, 'signal_action']
        signal_id = dataframe.loc[latest_idx, 'signal_id']
        
        # Execute CLOSE signals
        if signal_action == 'CLOSE' and (not signal_id or signal_id not in self.executed_signals):
            # Close both long and short positions
            dataframe.loc[latest_idx, 'exit_long'] = 1
            dataframe.loc[latest_idx, 'exit_short'] = 1
            # Add exit tag for tracking
            dataframe.loc[latest_idx, 'exit_tag'] = 'squeeze_exit_signal'
            # Don't mark as executed yet - wait for trade confirmation
            
            logger.info(f"⚪ CLOSE SIGNAL: {pair} id={signal_id[:8] if signal_id else 'N/A'}...")
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Dynamic position sizing based on signal metadata"""
        
        try:
            # Get the most recent signal for position sizing
            signal = self.get_redis_signals(pair)
            position_size_factor = signal.get('position_size_factor', 1.0) if signal else 1.0
            
            # Apply position size factor to proposed stake
            adjusted_stake = proposed_stake * position_size_factor
            
            # Respect limits
            if min_stake and adjusted_stake < min_stake:
                adjusted_stake = min_stake
            if adjusted_stake > max_stake:
                adjusted_stake = max_stake
            
            logger.debug(f"💰 Position sizing {pair}: "
                        f"{proposed_stake:.2f} → {adjusted_stake:.2f} "
                        f"factor={position_size_factor:.1f}")
            
            return adjusted_stake
            
        except Exception as e:
            logger.error(f"❌ Error in position sizing for {pair}: {e}")
            return proposed_stake
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str | None, 
                side: str, **kwargs) -> float:
        """Dynamic leverage based on signal metadata"""
        
        try:
            # Get the most recent signal for leverage calculation
            signal = self.get_redis_signals(pair)
            signal_leverage = signal.get('leverage', 3) if signal else 3
            
            # Use signal leverage but respect max_leverage
            final_leverage = min(signal_leverage, max_leverage)
            
            logger.debug(f"🚀 Leverage {pair}: {final_leverage:.1f}x "
                        f"(signal={signal_leverage}, max={max_leverage})")
            
            return final_leverage
            
        except Exception as e:
            logger.error(f"❌ Error calculating leverage for {pair}: {e}")
            return min(3.0, max_leverage)  # Fallback to 3x
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Confirm trade entry - validate signal freshness"""
        
        try:
            # Final validation before trade execution - get fresh signal
            signal = self.get_redis_signals(pair)
            
            if not signal:
                logger.warning(f"⚠️ No valid signal for {pair} entry, rejecting")
                return False
            
            # Check signal direction matches trade side
            signal_action = signal.get('action', 'NONE')
            if (side == 'long' and signal_action != 'LONG') or \
               (side == 'short' and signal_action != 'SHORT'):
                logger.warning(f"⚠️ Signal direction mismatch {pair}: "
                              f"signal={signal_action}, trade={side}")
                return False
            
            # Mark signal as executed after successful confirmation
            signal_id = signal.get('signal_id', '')
            if signal_id:
                self.mark_signal_executed(signal_id)
                logger.info(f"✅ Trade entry confirmed {pair}: {side} "
                           f"amount={amount:.4f} rate={rate:.2f} signal_id={signal_id[:8]}...")
            else:
                logger.info(f"✅ Trade entry confirmed {pair}: {side} "
                           f"amount={amount:.4f} rate={rate:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error confirming trade entry {pair}: {e}")
            return False
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, exit_reason: str,
                         current_time: datetime, **kwargs) -> bool:
        """Confirm trade exit"""
        
        logger.info(f"✅ Trade exit confirmed {pair}: reason={exit_reason} "
                   f"amount={amount:.4f} rate={rate:.2f}")
        
        return True
    
    def bot_start(self, **kwargs) -> None:
        """Called once when bot starts - test Redis connection and log status"""
        logger.info("🚀 SqueezeFlow FreqAI Strategy Starting...")
        
        # Test Redis connection
        if self.redis_client:
            try:
                self.redis_client.ping()
                logger.info("✅ Redis connection verified")
                
                # Log current signals for debugging
                try:
                    keys = self.redis_client.keys("squeeze_signal:*")
                    logger.info(f"📊 Found {len(keys)} Redis signal keys: {keys}")
                    
                    # Show sample signal data
                    if keys:
                        sample_key = keys[0]
                        sample_data = self.redis_client.get(sample_key)
                        logger.info(f"📝 Sample signal data for {sample_key}: {sample_data}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not read Redis signal keys: {e}")
                    
            except Exception as e:
                logger.error(f"❌ Redis connection test failed: {e}")
        else:
            logger.error("❌ No Redis connection available")
        
        # Log strategy configuration
        logger.info(f"⚙️ Strategy config: timeframe={self.timeframe}, "
                   f"can_short={self.can_short}, use_exit_signal={self.use_exit_signal}")

    def informative_pairs(self) -> list:
        """No informative pairs needed for pure execution"""
        return []