#!/usr/bin/env python3
"""
SqueezeFlow FreqAI Strategy for Freqtrade 2025.6 - OPTION A IMPLEMENTATION
PRIMARY DATA SOURCE: SqueezeFlow Calculator via Redis (no more direct CVD calculations)
ARCHITECTURE: SqueezeFlow Calculator → Redis → Freqtrade (clean separation)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
import pandas_ta as ta
import redis
import json
from datetime import datetime, timezone
from influxdb import InfluxDBClient
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

# Import state machine components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# State machine imports removed


logger = logging.getLogger(__name__)


class SqueezeFlowFreqAI(IStrategy):
    """
    SqueezeFlow strategy using FreqAI for ML-enhanced trading decisions
    Integrates external squeeze signals from InfluxDB with local technical indicators
    """
    
    # Strategy information
    STRATEGY_VERSION = "2025.6"
    
    # Strategy parameters - No minimal ROI, pure reversal detection
    minimal_roi = {
        "0": 100  # Disable ROI, rely on exit signals
    }
    
    stoploss = -0.08  # Enhanced: Dynamic stop loss - adjusted based on enhanced system (4-6x leverage)
    
    timeframe = '15m'
    
    # Freqtrade settings - OPTIMIZED FOR REAL-TIME
    can_short = True
    use_exit_signal = True  # Enable automatic signal-based trading
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # FreqAI settings - REAL-TIME OPTIMIZED
    process_only_new_candles = True   # Only process new candles for performance
    use_custom_stoploss = True        # Enable custom stoploss for real-time adjustments
    
    stoploss_on_exchange = False
    startup_candle_count = 120
    
    # Strategy parameters - REAL-TIME OPTIMIZED THRESHOLDS
    squeeze_threshold = DecimalParameter(0.1, 0.3, default=0.15, space="buy")  # Real-time: Lower threshold for faster reactions
    rsi_oversold = IntParameter(20, 40, default=30, space="buy")
    rsi_overbought = IntParameter(60, 80, default=70, space="sell")
    volume_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy")    # Real-time: Lower volume requirement
    
    # FreqAI parameters - ML as supportive, not blocking
    use_freqai = BooleanParameter(default=True, space="buy")
    freqai_confidence_threshold = DecimalParameter(0.3, 0.7, default=0.5, space="buy")  # Lower threshold
    ml_weight = DecimalParameter(0.1, 0.5, default=0.3, space="buy")  # ML influence weight
    
    def __init__(self, config: Dict) -> None:
        """Initialize strategy"""
        super().__init__(config)
        
        # Only connect to significant_trades for OI data (CVD now comes from Redis signals)
        influx_host = 'localhost' if not os.path.exists('/.dockerenv') else 'aggr-influx'
        self.squeezeflow_client = InfluxDBClient(
            host=influx_host,  # Use localhost for local testing, aggr-influx for Docker
            port=8086,
            database='significant_trades'
        )
        
        # Redis connection for external signals
        self.redis_client = None
        self.setup_redis_connection()
        
        # SIGNAL PERSISTENCE SYSTEM
        self.signal_cache = {}  # Persistent signal storage
        self.entry_signals = {}  # Entry tracking for timing and direction
        self.last_signal_update = {}
        self.active_positions = {}  # Track which position direction is active
        
        # State machine integration removed
        
    def setup_redis_connection(self):
        """Setup Redis connection with connection pooling for real-time performance"""
        try:
            redis_url = self.config.get('redis_url', 'redis://redis:6379')
            
            # Optimized Redis connection with connection pooling for real-time performance
            pool = redis.ConnectionPool.from_url(
                redis_url, 
                decode_responses=True,
                max_connections=10,        # Connection pool for better performance
                socket_connect_timeout=1,  # Fast connection timeout
                socket_timeout=1,          # Fast socket timeout
                retry_on_timeout=True      # Retry on timeout
            )
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info("🚀 Redis connection established with connection pooling for real-time signals")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            
    def get_external_squeeze_signals(self, pair: str, timeframe: str = '15m') -> Dict:
        """
        Get external squeeze signals with REAL-TIME OPTIMIZED PERFORMANCE
        """
        if not self.redis_client:
            logger.warning(f"🔴 Redis client not available for {pair}")
            return {}
            
        try:
            # Convert pair to base symbol (e.g., BTC/USDT:USDT -> BTC)
            symbol = pair.split('/')[0]
            
            # REAL-TIME OPTIMIZED: Use Redis pipeline for batch operations
            pipe = self.redis_client.pipeline()
            
            # Focus on key timeframes for real-time trading
            priority_lookbacks = [20, 60, 240]  # 20min, 1h, 4h - most important for entries
            
            # Batch all Redis operations for minimal latency
            for lookback in priority_lookbacks:
                key = f"squeeze_signal:{symbol}:{lookback}"
                pipe.get(key)
            
            # Execute all Redis operations in one round-trip
            results = pipe.execute()
            
            current_signals = {}
            signal_count = 0
            
            for i, lookback in enumerate(priority_lookbacks):
                signal_data = results[i]
                
                if signal_data:
                    signal = json.loads(signal_data)
                    current_signals[f'squeeze_score_{lookback}'] = signal.get('squeeze_score', 0)
                    current_signals[f'signal_strength_{lookback}'] = signal.get('signal_strength', 0)
                    current_signals[f'signal_type_{lookback}'] = signal.get('signal_type', 'NEUTRAL')
                    # Enhanced metrics from updated calculator
                    current_signals[f'alignment_quality_{lookback}'] = signal.get('alignment_quality', 'NONE')
                    current_signals[f'signal_multiplier_{lookback}'] = signal.get('signal_multiplier', 1.0)
                    signal_count += 1
                else:
                    current_signals[f'squeeze_score_{lookback}'] = 0
                    current_signals[f'signal_strength_{lookback}'] = 0
                    current_signals[f'signal_type_{lookback}'] = 'NEUTRAL'
                    current_signals[f'alignment_quality_{lookback}'] = 'NONE'
                    current_signals[f'signal_multiplier_{lookback}'] = 1.0
            
            # ENHANCED SIGNAL PERSISTENCE LOGIC
            current_score = current_signals.get('squeeze_score_20', 0)
            current_alignment = current_signals.get('alignment_quality_20', 'NONE')
            current_multiplier = current_signals.get('signal_multiplier_20', 1.0)
            
            # Initialize enhanced cache for this pair
            if pair not in self.signal_cache:
                self.signal_cache[pair] = {
                    'score': 0, 
                    'timestamp': datetime.now(), 
                    'active': False,
                    'strength': 0,
                    'type': 'NEUTRAL',
                    'alignment_quality': 'NONE',
                    'signal_multiplier': 1.0,
                    'higher_tf_conflict': False
                }
            
            cache = self.signal_cache[pair]
            
            # HIGHER TIMEFRAME CONFLICT DETECTION (from research)
            higher_tf_conflict = self._detect_higher_tf_conflict(current_signals)
            cache['higher_tf_conflict'] = higher_tf_conflict
            
            # ENHANCED ENTRY LOGIC with adaptive thresholds
            activation_threshold = 0.15
            if current_alignment == 'TRIPLE':
                activation_threshold = 0.1  # Lower threshold for triple alignment
            elif current_alignment == 'DOUBLE':
                activation_threshold = 0.12  # Slightly lower for double alignment
            
            if abs(current_score) > activation_threshold:
                if not cache['active']:
                    # Completely new signal
                    cache['score'] = current_score
                    cache['timestamp'] = datetime.now()
                    cache['active'] = True
                    cache['strength'] = current_signals.get('signal_strength_20', 0)
                    cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
                    cache['alignment_quality'] = current_alignment
                    cache['signal_multiplier'] = current_multiplier
                    
                    # Save entry signal for timing
                    self.entry_signals[pair] = cache.copy()
                    
                    conflict_msg = " [HT_CONFLICT]" if higher_tf_conflict else ""
                    logger.info(f"🎯 NEW SIGNAL ACTIVATED {symbol}: score={current_score:.3f}, "
                               f"type={cache['type']}, quality={current_alignment}{conflict_msg}")
                    
                elif abs(current_score) > abs(cache['score']) * 1.3:  # Signal 30% stronger (more responsive)
                    # Signal refresh: Strong new signal overrides old one
                    cache['score'] = current_score
                    cache['timestamp'] = datetime.now()  # REFRESH TIMESTAMP!
                    cache['strength'] = current_signals.get('signal_strength_20', 0)
                    cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
                    cache['alignment_quality'] = current_alignment
                    cache['signal_multiplier'] = current_multiplier
                    
                    # Refresh entry signal for timing
                    self.entry_signals[pair] = cache.copy()
                    
                    logger.info(f"🔄 SIGNAL REFRESHED {symbol}: score={current_score:.3f} "
                               f"(was {cache['score']:.3f}), type={cache['type']}, quality={current_alignment}")
                
            # EXIT LOGIC: Signal weakens significantly
            elif abs(current_score) < 0.1 and cache['active']:
                cache['active'] = False
                logger.info(f"🔄 SIGNAL DEACTIVATED {symbol}: score dropped to {current_score:.3f}")
            
            # UPDATE: Update active signals with new values and enhanced metrics
            elif cache['active'] and abs(current_score) > 0.1:
                cache['score'] = current_score  # Update with current score
                cache['strength'] = current_signals.get('signal_strength_20', 0)
                cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
                cache['alignment_quality'] = current_alignment
                cache['signal_multiplier'] = current_multiplier
            
            # RETURN: Enhanced signal data with all metrics
            if cache['active']:
                # Active signal: Use cached values with enhancements
                return_signals = current_signals.copy()
                return_signals['squeeze_score_20'] = cache['score']
                return_signals['signal_strength_20'] = cache['strength']
                return_signals['signal_type_20'] = cache['type']
                return_signals['alignment_quality_20'] = cache['alignment_quality']
                return_signals['signal_multiplier_20'] = cache['signal_multiplier']
                return_signals['signal_active'] = True
                return_signals['signal_age_minutes'] = (datetime.now() - cache['timestamp']).total_seconds() / 60
                return_signals['higher_tf_conflict'] = cache['higher_tf_conflict']
            else:
                # No active signal: Use current weak values
                return_signals = current_signals.copy()
                return_signals['signal_active'] = False
                return_signals['signal_age_minutes'] = 0
                return_signals['higher_tf_conflict'] = higher_tf_conflict
            
            # REAL-TIME MONITORING LOG (every 10th call to avoid spam)
            if not hasattr(self, '_signal_call_count'):
                self._signal_call_count = 0
            self._signal_call_count += 1
            
            if self._signal_call_count % 10 == 0 or return_signals.get('signal_active', False):
                logger.info(f"🚀 REAL-TIME SIGNAL {pair}: Score={return_signals.get('squeeze_score_20', 0):.3f} | "
                           f"Active={return_signals.get('signal_active', False)} | "
                           f"Type={return_signals.get('signal_type_20', 'NEUTRAL')} | "
                           f"Age={return_signals.get('signal_age_minutes', 0):.1f}min | "
                           f"Redis Latency: <1ms")
            
            return return_signals
            
        except Exception as e:
            logger.error(f"❌ Error getting squeeze signals for {pair}: {e}")
            return {}
    
    def _detect_higher_tf_conflict(self, signals: Dict) -> bool:
        """
        Detect higher timeframe conflicts based on research findings
        """
        try:
            # Check 2-4 hour timeframes for strong opposing trends
            score_120 = signals.get('squeeze_score_120', 0)  # 2 hours
            score_240 = signals.get('squeeze_score_240', 0)  # 4 hours
            score_20 = signals.get('squeeze_score_20', 0)    # 20 minutes (primary)
            
            # Strong higher timeframe threshold
            strong_ht_threshold = 0.4
            
            # Check for opposing directions with strong signals
            if abs(score_120) > strong_ht_threshold or abs(score_240) > strong_ht_threshold:
                ht_direction = np.sign(score_120) if abs(score_120) > abs(score_240) else np.sign(score_240)
                primary_direction = np.sign(score_20)
                
                # Conflict detected if directions oppose and primary signal is weaker
                if (ht_direction != 0 and primary_direction != 0 and 
                    ht_direction != primary_direction and abs(score_20) < 0.8):
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detecting higher TF conflict: {e}")
            return False
            
    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int,
                                     metadata: Dict, **kwargs) -> pd.DataFrame:
        """
        Add all features for FreqAI training
        This function is called for each pair and timeframe
        """
        
        # Get external squeeze signals
        external_signals = self.get_external_squeeze_signals(metadata['pair'])
        
        # Add external squeeze signals as features
        for signal_name, signal_value in external_signals.items():
            if isinstance(signal_value, (int, float)):
                dataframe[f'&-{signal_name}'] = signal_value
            else:
                # Convert signal types to numeric
                signal_map = {
                    'STRONG_LONG_SQUEEZE': -1.0,
                    'LONG_SQUEEZE': -0.5,
                    'NEUTRAL': 0.0,
                    'SHORT_SQUEEZE': 0.5,
                    'STRONG_SHORT_SQUEEZE': 1.0
                }
                dataframe[f'&-{signal_name}'] = signal_map.get(signal_value, 0.0)
        
        # Technical indicators
        dataframe['&-rsi'] = ta.rsi(dataframe['close'], length=14)
        macd = ta.macd(dataframe['close'])
        dataframe['&-macd'] = macd['MACD_12_26_9']
        dataframe['&-macd_signal'] = macd['MACDs_12_26_9']
        dataframe['&-macd_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(dataframe['close'], length=20)
        dataframe['&-bb_upper'] = bb['BBU_20_2.0']
        dataframe['&-bb_middle'] = bb['BBM_20_2.0']
        dataframe['&-bb_lower'] = bb['BBL_20_2.0']
        dataframe['&-bb_percent'] = (dataframe['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])
        
        # Volume indicators
        dataframe['&-volume_sma'] = ta.sma(dataframe['volume'], length=20)
        dataframe['&-volume_ratio'] = dataframe['volume'] / dataframe['&-volume_sma']
        
        # Price action features
        dataframe['&-price_change'] = dataframe['close'].pct_change()
        dataframe['&-high_low_ratio'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        dataframe['&-body_ratio'] = abs(dataframe['close'] - dataframe['open']) / (dataframe['high'] - dataframe['low'])
        
        # ATR for volatility
        dataframe['&-atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        dataframe['&-atr_ratio'] = dataframe['&-atr'] / dataframe['close']
        
        # Stochastic
        stoch = ta.stoch(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['&-stoch_k'] = stoch['STOCHk_14_3_3']
        dataframe['&-stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Williams %R
        dataframe['&-williams_r'] = ta.willr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        
        # Commodity Channel Index
        dataframe['&-cci'] = ta.cci(dataframe['high'], dataframe['low'], dataframe['close'], length=20)
        
        # Support and resistance levels
        dataframe['&-support'] = dataframe['low'].rolling(window=20).min()
        dataframe['&-resistance'] = dataframe['high'].rolling(window=20).max()
        dataframe['&-support_distance'] = (dataframe['close'] - dataframe['&-support']) / dataframe['close']
        dataframe['&-resistance_distance'] = (dataframe['&-resistance'] - dataframe['close']) / dataframe['close']
        
        return dataframe
        
    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs) -> pd.DataFrame:
        """
        Add basic features that are always present
        """
        # Always include current squeeze score
        external_signals = self.get_external_squeeze_signals(metadata['pair'])
        dataframe['squeeze_score'] = external_signals.get('squeeze_score_20', 0)
        dataframe['signal_strength'] = external_signals.get('signal_strength_20', 0)
        
        return dataframe
        
    # CVD data now comes from SqueezeFlow Calculator via Redis - no need for direct InfluxDB queries

    def get_oi_data(self, pair: str, timeframe: str = '1m') -> pd.DataFrame:
        """
        Get Open Interest data from InfluxDB using database-driven symbol discovery
        """
        try:
            # Convert pair to base symbol for OI discovery (BTC/USDT:USDT -> BTC)
            base_symbol = pair.split('/')[0]  # Extract base symbol
            
            # Discover available OI symbols from database instead of hardcoded variants
            from utils.symbol_discovery import symbol_discovery
            available_oi_symbols = symbol_discovery.discover_oi_symbols_for_base(
                base_symbol, 
                min_data_points=50,  # Lower threshold for OI data
                hours_lookback=24
            )
            
            if not available_oi_symbols:
                # Fallback to basic variants if discovery fails
                logger.warning(f"🔄 No OI symbols discovered for {base_symbol}, using fallback")
                available_oi_symbols = [
                    base_symbol.upper(),                    # BTC
                    base_symbol.upper() + 'USD',           # BTCUSD
                    base_symbol.upper() + 'USDT',          # BTCUSDT
                    base_symbol.upper() + 'USDC'           # BTCUSDC
                ]
            else:
                logger.info(f"✅ Discovered OI symbols for {base_symbol}: {available_oi_symbols}")
            
            # Build OI query with discovered symbols
            symbol_filter = ' OR '.join([f"symbol = '{symbol}'" for symbol in available_oi_symbols])
            oi_query = f"""
            SELECT time, exchange, open_interest_usd, open_interest_change_24h
            FROM open_interest 
            WHERE ({symbol_filter})
            AND time >= now() - 1h 
            ORDER BY time DESC LIMIT 60
            """
            
            oi_result = self.squeezeflow_client.query(oi_query)
            oi_df = pd.DataFrame(list(oi_result.get_points()))
            
            if not oi_df.empty:
                oi_df['time'] = pd.to_datetime(oi_df['time'])
                oi_df.set_index('time', inplace=True)
                
                # Aggregate OI across exchanges (sum total OI)
                oi_agg = oi_df.groupby('time').agg({
                    'open_interest_usd': 'sum',
                    'open_interest_change_24h': 'mean'
                }).fillna(0)
                
                logger.info(f"OI data loaded for {pair}: {len(oi_agg)} points")
                return oi_agg
                
            else:
                logger.warning(f"No OI data found for {pair}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting OI data: {e}")
            return pd.DataFrame()

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs) -> pd.DataFrame:
        """
        Add standard features for prediction - CVD now comes from Redis signals
        """
        # Standard technical indicators for FreqAI
        dataframe['%-rsi'] = ta.rsi(dataframe['close'], length=14)
        macd = ta.macd(dataframe['close'])
        dataframe['%-macd'] = macd['MACD_12_26_9']
        bb = ta.bbands(dataframe['close'], length=20)
        dataframe['%-bb_percent'] = (dataframe['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])
        
        # Get squeeze signals from Redis (replaces direct CVD calculation)
        external_signals = self.get_external_squeeze_signals(metadata['pair'])
        
        # Add squeeze signals as ML features
        dataframe['%-squeeze_score'] = external_signals.get('squeeze_score_20', 0)
        dataframe['%-signal_strength'] = external_signals.get('signal_strength_20', 0)
        
        # Add different timeframe signals for ML
        for lookback in [5, 10, 30]:
            dataframe[f'%-squeeze_score_{lookback}'] = external_signals.get(f'squeeze_score_{lookback}', 0)
            dataframe[f'%-signal_strength_{lookback}'] = external_signals.get(f'signal_strength_{lookback}', 0)
        
        # Get OI data from InfluxDB - CRITICAL for squeeze strategy
        oi_data = self.get_oi_data(metadata['pair'])
        
        # Add OI features if data is available
        if not oi_data.empty:
            # Resample to match strategy timeframe
            oi_resampled = oi_data.resample('1min').last().ffill()
            
            # Merge with dataframe
            dataframe = dataframe.merge(oi_resampled[['open_interest_usd', 'open_interest_change_24h']], 
                                      left_on='date', right_index=True, how='left')
            
            # Calculate OI features for ML
            dataframe['%-oi_normalized'] = dataframe['open_interest_usd'] / dataframe['open_interest_usd'].rolling(100).mean()
            dataframe['%-oi_change_24h'] = dataframe['open_interest_change_24h'] / 100  # Convert to percentage
            
            # OI momentum (rate of change)
            dataframe['%-oi_momentum'] = dataframe['open_interest_usd'].pct_change(periods=5)
            
            # Fill NaN values
            dataframe['%-oi_normalized'] = dataframe['%-oi_normalized'].fillna(1)
            dataframe['%-oi_change_24h'] = dataframe['%-oi_change_24h'].fillna(0)
            dataframe['%-oi_momentum'] = dataframe['%-oi_momentum'].fillna(0)
            dataframe['open_interest_usd'] = dataframe['open_interest_usd'].fillna(0)
            dataframe['open_interest_change_24h'] = dataframe['open_interest_change_24h'].fillna(0)
        else:
            dataframe['%-oi_normalized'] = 1
            dataframe['%-oi_change_24h'] = 0
            dataframe['%-oi_momentum'] = 0
            dataframe['open_interest_usd'] = 0
            dataframe['open_interest_change_24h'] = 0
        
        return dataframe
        
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Add indicators to dataframe
        """
        # Technical indicators
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        macd = ta.macd(dataframe['close'])
        dataframe['macd'] = macd['MACD_12_26_9']
        dataframe['macd_signal'] = macd['MACDs_12_26_9']
        
        # Volume
        dataframe['volume_sma'] = ta.sma(dataframe['volume'], length=20)
        
        # Get external squeeze signals
        external_signals = self.get_external_squeeze_signals(metadata['pair'])
        
        # Add to dataframe
        for signal_name, signal_value in external_signals.items():
            if isinstance(signal_value, (int, float)):
                dataframe[signal_name] = signal_value
            else:
                # Convert to numeric
                signal_map = {
                    'STRONG_LONG_SQUEEZE': -1.0,
                    'LONG_SQUEEZE': -0.5,
                    'NEUTRAL': 0.0,
                    'SHORT_SQUEEZE': 0.5,
                    'STRONG_SHORT_SQUEEZE': 1.0
                }
                dataframe[signal_name] = signal_map.get(signal_value, 0.0)
        
        # Primary squeeze score (20-period lookback) - now the MAIN signal source
        dataframe['squeeze_score'] = external_signals.get('squeeze_score_20', 0)
        dataframe['signal_strength'] = external_signals.get('signal_strength_20', 0)
        dataframe['signal_type'] = external_signals.get('signal_type_20', 'NEUTRAL')
        
        # Add different timeframe signals for confirmation
        dataframe['squeeze_score_5'] = external_signals.get('squeeze_score_5', 0)
        dataframe['squeeze_score_10'] = external_signals.get('squeeze_score_10', 0)
        dataframe['squeeze_score_30'] = external_signals.get('squeeze_score_30', 0)
        
        # Add higher timeframes for trend confirmation (1h, 2h, 4h)
        dataframe['squeeze_score_60'] = external_signals.get('squeeze_score_60', 0)   # 1h
        dataframe['squeeze_score_120'] = external_signals.get('squeeze_score_120', 0) # 2h  
        dataframe['squeeze_score_240'] = external_signals.get('squeeze_score_240', 0) # 4h
        
        # Get OI data for additional confirmation
        oi_data = self.get_oi_data(metadata['pair'])
        
        # Add OI data if available - CRITICAL for squeeze strategy
        if not oi_data.empty:
            # Resample to match strategy timeframe
            oi_resampled = oi_data.resample('1min').last().ffill()
            
            # Merge with dataframe
            dataframe = dataframe.merge(oi_resampled[['open_interest_usd', 'open_interest_change_24h']], 
                                      left_on='date', right_index=True, how='left')
            
            # Calculate OI features
            dataframe['oi_normalized'] = dataframe['open_interest_usd'] / dataframe['open_interest_usd'].rolling(100).mean()
            dataframe['oi_momentum'] = dataframe['open_interest_usd'].pct_change(periods=5)
            
            # Fill NaN values
            dataframe['oi_normalized'] = dataframe['oi_normalized'].fillna(1)
            dataframe['oi_momentum'] = dataframe['oi_momentum'].fillna(0)
            dataframe['open_interest_usd'] = dataframe['open_interest_usd'].fillna(0)
            dataframe['open_interest_change_24h'] = dataframe['open_interest_change_24h'].fillna(0)
        else:
            dataframe['oi_normalized'] = 1
            dataframe['oi_momentum'] = 0
            dataframe['open_interest_usd'] = 0
            dataframe['open_interest_change_24h'] = 0
        
        return dataframe
        
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Populate entry signals with OPTIMIZED LOGIC from research
        """
        pair = metadata['pair']
        
        # Get enhanced signals with all metrics
        external_signals = self.get_external_squeeze_signals(pair)
        signal_active = external_signals.get('signal_active', False)
        signal_age = external_signals.get('signal_age_minutes', 0)
        alignment_quality = external_signals.get('alignment_quality_20', 'NONE')
        higher_tf_conflict = external_signals.get('higher_tf_conflict', False)
        
        # DEBUG: Enhanced logging
        logger.info(f"🔍 ENTRY CHECK {pair}: score={external_signals.get('squeeze_score_20', 0):.3f}, "
                   f"active={signal_active}, age={signal_age:.1f}min, quality={alignment_quality}, "
                   f"ht_conflict={higher_tf_conflict}")
        
        # ADAPTIVE THRESHOLDS based on signal quality (from research)
        base_threshold = self.squeeze_threshold.value  # 0.5 default
        
        if alignment_quality == 'TRIPLE':
            entry_threshold = base_threshold * 0.7  # Lower threshold for triple alignment
            logger.debug(f"Triple alignment detected, lowered threshold to {entry_threshold:.2f}")
        elif alignment_quality == 'DOUBLE':
            entry_threshold = base_threshold * 0.8  # Slightly lower for double alignment
        else:
            entry_threshold = base_threshold  # Standard threshold for divergence
        
        # HIGHER TIMEFRAME CONFLICT PENALTY (from research)
        if higher_tf_conflict:
            entry_threshold *= 1.4  # Require stronger signal if fighting higher TF
            logger.debug(f"Higher TF conflict detected, raised threshold to {entry_threshold:.2f}")
        
        # OPTIMIZED LONG ENTRY CONDITIONS
        long_conditions = [
            # PRIMARY: Enhanced signal requirements
            (dataframe['squeeze_score'] <= -entry_threshold),
            (signal_active == True),  # Signal must be active
            (signal_age < 5),  # Signal max 5 minutes old
            
            # MOMENTUM ALIGNMENT PRIORITY (from research)
            # For momentum alignment, require less strict timeframe confirmation
            ((alignment_quality in ['TRIPLE', 'DOUBLE']) |
             # For divergence, require stronger multi-timeframe confirmation
             ((dataframe['squeeze_score_10'] <= -0.15) | (dataframe['squeeze_score_30'] <= -0.2))),
            
            # RELAXED HIGHER TIMEFRAME for momentum alignment
            ((alignment_quality == 'TRIPLE') |
             (alignment_quality == 'DOUBLE' and 
              ((dataframe['squeeze_score_60'] <= -0.1) | (dataframe['squeeze_score_120'] <= -0.05))) |
             # Strict requirements for divergence
             (alignment_quality == 'DIVERGENCE' and
              ((dataframe['squeeze_score_60'] <= -0.15) | (dataframe['squeeze_score_120'] <= -0.1) | (dataframe['squeeze_score_240'] <= -0.05)))),
            
            # TECHNICAL FILTERS (relaxed for high-quality signals)
            (dataframe['rsi'] < (self.rsi_overbought.value + 5 if alignment_quality == 'TRIPLE' else self.rsi_overbought.value)),
            (dataframe['volume'] > dataframe['volume_sma'] * 
             (self.volume_threshold.value * 0.8 if alignment_quality in ['TRIPLE', 'DOUBLE'] else self.volume_threshold.value)),
            
            # OI confirmation (more lenient for quality signals)
            (dataframe['oi_normalized'] > (0.6 if alignment_quality == 'TRIPLE' else 0.8)),
            (dataframe['oi_momentum'] > -0.15)
        ]
        
        # Add FreqAI prediction as supportive signal (not blocking)
        if self.use_freqai.value and '&-s-up_or_down' in dataframe.columns:
            # ML provides additional confidence, but doesn't block trades
            ml_long_signal = (
                (dataframe['&-s-up_or_down'] == 'up') & 
                (dataframe['&-s-up_or_down_confidence'] >= self.freqai_confidence_threshold.value)
            )
            # ML adds weight but doesn't block - create ML-enhanced conditions
            dataframe['ml_long_boost'] = np.where(ml_long_signal, self.ml_weight.value, 0)
        
        # OPTIMIZED SHORT ENTRY CONDITIONS
        short_conditions = [
            # PRIMARY: Enhanced signal requirements
            (dataframe['squeeze_score'] >= entry_threshold),
            (signal_active == True),  # Signal must be active  
            (signal_age < 5),  # Signal max 5 minutes old
            
            # MOMENTUM ALIGNMENT PRIORITY (from research)
            # For momentum alignment, require less strict timeframe confirmation
            ((alignment_quality in ['TRIPLE', 'DOUBLE']) |
             # For divergence, require stronger multi-timeframe confirmation
             ((dataframe['squeeze_score_10'] >= 0.15) | (dataframe['squeeze_score_30'] >= 0.2))),
            
            # RELAXED HIGHER TIMEFRAME for momentum alignment
            ((alignment_quality == 'TRIPLE') |
             (alignment_quality == 'DOUBLE' and 
              ((dataframe['squeeze_score_60'] >= 0.1) | (dataframe['squeeze_score_120'] >= 0.05))) |
             # Strict requirements for divergence
             (alignment_quality == 'DIVERGENCE' and
              ((dataframe['squeeze_score_60'] >= 0.15) | (dataframe['squeeze_score_120'] >= 0.1) | (dataframe['squeeze_score_240'] >= 0.05)))),
            
            # TECHNICAL FILTERS (relaxed for high-quality signals)
            (dataframe['rsi'] > (self.rsi_oversold.value - 5 if alignment_quality == 'TRIPLE' else self.rsi_oversold.value)),
            (dataframe['volume'] > dataframe['volume_sma'] * 
             (self.volume_threshold.value * 0.8 if alignment_quality in ['TRIPLE', 'DOUBLE'] else self.volume_threshold.value)),
            
            # OI confirmation (more lenient for quality signals)
            (dataframe['oi_normalized'] > (0.6 if alignment_quality == 'TRIPLE' else 0.8)),
            (dataframe['oi_momentum'] > -0.15)
        ]
        
        # Add FreqAI prediction as supportive signal (not blocking)
        if self.use_freqai.value and '&-s-up_or_down' in dataframe.columns:
            # ML provides additional confidence, but doesn't block trades
            ml_short_signal = (
                (dataframe['&-s-up_or_down'] == 'down') & 
                (dataframe['&-s-up_or_down_confidence'] >= self.freqai_confidence_threshold.value)
            )
            # ML adds weight but doesn't block
            dataframe['ml_short_boost'] = np.where(ml_short_signal, self.ml_weight.value, 0)
        
        # ENHANCED ENTRY LOGIC with quality-based decisions
        if signal_active and signal_age < 5:  # Only for fresh active signals
            
            # LONG ENTRY with enhanced logging
            if len(long_conditions) > 0:
                try:
                    latest_idx = dataframe.index[-1]
                    long_check = all([
                        condition.iloc[-1] if hasattr(condition, 'iloc') else condition 
                        for condition in long_conditions
                    ])
                    
                    if long_check:
                        dataframe.loc[latest_idx, 'enter_long'] = 1
                        # Enhanced entry tracking
                        if pair not in self.entry_signals:
                            self.entry_signals[pair] = {}
                        self.entry_signals[pair].update({
                            'entry_time': datetime.now(),
                            'direction': 'long',
                            'alignment_quality': alignment_quality,
                            'signal_strength': external_signals.get('squeeze_score_20', 0),
                            'higher_tf_conflict': higher_tf_conflict,
                            'entry_threshold_used': entry_threshold
                        })
                        self.active_positions[pair] = 'long'
                        
                        conflict_msg = " [FIGHTING_HT]" if higher_tf_conflict else ""
                        logger.info(f"🟢 LONG ENTRY: {pair} at ${dataframe['close'].iloc[-1]:.2f} | "
                                   f"Quality: {alignment_quality} | Threshold: {entry_threshold:.2f}{conflict_msg}")
                except Exception as e:
                    logger.error(f"Error in long entry logic: {e}")
            
            # SHORT ENTRY with enhanced logging
            if len(short_conditions) > 0:
                try:
                    latest_idx = dataframe.index[-1] 
                    short_check = all([
                        condition.iloc[-1] if hasattr(condition, 'iloc') else condition
                        for condition in short_conditions
                    ])
                    
                    if short_check:
                        dataframe.loc[latest_idx, 'enter_short'] = 1
                        # Enhanced entry tracking
                        if pair not in self.entry_signals:
                            self.entry_signals[pair] = {}
                        self.entry_signals[pair].update({
                            'entry_time': datetime.now(),
                            'direction': 'short',
                            'alignment_quality': alignment_quality,
                            'signal_strength': external_signals.get('squeeze_score_20', 0),
                            'higher_tf_conflict': higher_tf_conflict,
                            'entry_threshold_used': entry_threshold
                        })
                        self.active_positions[pair] = 'short'
                        
                        conflict_msg = " [FIGHTING_HT]" if higher_tf_conflict else ""
                        logger.info(f"🔴 SHORT ENTRY: {pair} at ${dataframe['close'].iloc[-1]:.2f} | "
                                   f"Quality: {alignment_quality} | Threshold: {entry_threshold:.2f}{conflict_msg}")
                except Exception as e:
                    logger.error(f"Error in short entry logic: {e}")
        else:
            # Enhanced debug information
            if not signal_active:
                logger.info(f"⚪ No entry for {pair}: No active signal "
                           f"(score={external_signals.get('squeeze_score_20', 0):.3f}, "
                           f"quality={alignment_quality})")
            elif signal_age >= 5:
                logger.info(f"⚪ No entry for {pair}: Signal too old ({signal_age:.1f} min, "
                           f"quality={alignment_quality})")
            else:
                logger.info(f"⚪ No entry for {pair}: Conditions not met "
                           f"(age={signal_age:.1f}min, active={signal_active}, "
                           f"quality={alignment_quality}, threshold={entry_threshold:.2f}, "
                           f"ht_conflict={higher_tf_conflict})")
        
        return dataframe
        
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Populate exit signals with TIMING-based logic
        """
        pair = metadata['pair']
        
        # AUTO-RECOVERY: Rebuild active_positions from FreqTrade trade data after restart
        if not hasattr(self, 'active_positions'):
            self.active_positions = {}
        if not hasattr(self, 'entry_signals'):  
            self.entry_signals = {}
        if not hasattr(self, 'signal_inactive_start'):
            self.signal_inactive_start = {}
            
        # Check if FreqTrade has open trades we don't know about (after restart)
        if pair not in self.active_positions:
            # This would need trade info from FreqTrade - for now set from pair pattern
            # We can detect from exit_long/exit_short signals in dataframe
            pass  # Will be handled in exit logic
        
        # Get persistent signals and check if we have an active position
        external_signals = self.get_external_squeeze_signals(pair)
        signal_active = external_signals.get('signal_active', False)
        
        # Check if we have position timing info - SAFE CHECK for race condition
        has_position_timing = (pair in self.entry_signals and 
                             'entry_time' in self.entry_signals[pair])
        
        if has_position_timing:
            try:
                entry_time = self.entry_signals[pair]['entry_time']
                position_age_minutes = (datetime.now() - entry_time).total_seconds() / 60
            except (KeyError, TypeError) as e:
                logger.warning(f"Entry time not ready for {pair}: {e}, skipping exit logic")
                return dataframe  # Exit early - no cleanup, as entry is running
            
            # EXIT CONDITIONS - IMMEDIATE EXITS possible on signal reversal
            # MINIMUM HOLD TIME ENTFERNT - Squeeze-Reversals brauchen sofortige Reaktion!
            
            # Check if signal has been inactive for a longer time (less nervous)
            signal_inactive_duration = 0
            if pair in self.signal_inactive_start:
                signal_inactive_duration = (datetime.now() - self.signal_inactive_start[pair]).total_seconds() / 60
            elif not signal_active:
                # Signal gerade erst inaktiv geworden - Timer starten
                if pair not in self.signal_inactive_start:
                    self.signal_inactive_start = getattr(self, 'signal_inactive_start', {})
                    self.signal_inactive_start[pair] = datetime.now()
                    signal_inactive_duration = 0
            else:
                # Signal active - reset timer
                if hasattr(self, 'signal_inactive_start') and pair in self.signal_inactive_start:
                    del self.signal_inactive_start[pair]
                signal_inactive_duration = 0
            
            # Get enhanced exit data
            entry_quality = self.entry_signals[pair].get('alignment_quality', 'NONE')
            entry_conflict = self.entry_signals[pair].get('higher_tf_conflict', False)
            current_signals = self.get_external_squeeze_signals(pair)
            current_quality = current_signals.get('alignment_quality_20', 'NONE')
            current_score = dataframe['squeeze_score'].iloc[-1]
            
            # ADAPTIVE EXIT THRESHOLDS based on entry quality
            if entry_quality == 'TRIPLE':
                reversal_threshold = 1.2  # Higher threshold for quality entries
                weak_signal_threshold = 0.03
                max_hold_time = 180  # 3 hours for quality signals
            elif entry_quality == 'DOUBLE':
                reversal_threshold = 1.0
                weak_signal_threshold = 0.04
                max_hold_time = 150  # 2.5 hours
            else:
                reversal_threshold = 0.8  # Lower threshold for divergence
                weak_signal_threshold = 0.05
                max_hold_time = 120  # 2 hours
            
            # CONFLICT-BASED EARLY EXIT
            if entry_conflict and position_age_minutes > 60:  # Early exit for conflict trades
                max_hold_time = min(max_hold_time, 90)  # Cap at 1.5 hours
                weak_signal_threshold *= 1.5  # More aggressive exit
            
            # OPTIMIZED LONG EXIT CONDITIONS
            long_exit_conditions = [
                # 1. ADAPTIVE SIGNAL REVERSAL
                current_score > reversal_threshold,
                
                # 2. SIGNAL PERMANENTLY INACTIVE with quality-based filter
                (signal_inactive_duration > 30 and abs(current_score) < weak_signal_threshold),
                
                # 3. QUALITY-BASED TIME EXIT
                (position_age_minutes > max_hold_time and abs(current_score) < 0.1),
                
                # 4. MOMENTUM LOSS (for momentum trades)
                (entry_quality in ['TRIPLE', 'DOUBLE'] and 
                 current_quality == 'NONE' and 
                 position_age_minutes > 30 and 
                 abs(current_score) < 0.2),
                
                # 5. EARLY CONFLICT EXIT
                (entry_conflict and position_age_minutes > 45 and current_score > -0.1)
            ]
            
            # OPTIMIZED SHORT EXIT CONDITIONS  
            short_exit_conditions = [
                # 1. ADAPTIVE SIGNAL REVERSAL
                current_score < -reversal_threshold,
                
                # 2. SIGNAL PERMANENTLY INACTIVE with quality-based filter
                (signal_inactive_duration > 30 and abs(current_score) < weak_signal_threshold),
                
                # 3. QUALITY-BASED TIME EXIT
                (position_age_minutes > max_hold_time and abs(current_score) < 0.1),
                
                # 4. MOMENTUM LOSS (for momentum trades)
                (entry_quality in ['TRIPLE', 'DOUBLE'] and 
                 current_quality == 'NONE' and 
                 position_age_minutes > 30 and 
                 abs(current_score) < 0.2),
                
                # 5. EARLY CONFLICT EXIT
                (entry_conflict and position_age_minutes > 45 and current_score < 0.1)
            ]
                
            logger.info(f"🔍 EXIT CONDITIONS {pair}: age={position_age_minutes:.1f}min, "
                       f"score={current_score:.3f}, entry_quality={entry_quality}, "
                       f"current_quality={current_quality}, inactive_duration={signal_inactive_duration:.1f}min, "
                       f"reversal_threshold=±{reversal_threshold:.1f}, max_hold={max_hold_time}min")
        else:
            # FALLBACK: EMERGENCY EXIT without position timing (after restart)
            # When entry time is lost, use signal status for exit decision
            logger.warning(f"🚨 EMERGENCY EXIT {pair}: No position timing available, using signal-based fallback")
            
            # After restart: Exit only if signal completely inactive or very weak
            emergency_exit_conditions = [
                # 1. Signal completely inactive - IMMEDIATE EXIT
                (not signal_active and abs(dataframe['squeeze_score'].iloc[-1]) < 0.1),
                # 2. Very weak signal indicates no clear direction
                (signal_active and abs(dataframe['squeeze_score'].iloc[-1]) < 0.05)
            ]
            
            # Apply same basic emergency conditions to both directions
            # We avoid directional exits since we don't know the actual position direction
            long_exit_conditions = emergency_exit_conditions
            short_exit_conditions = emergency_exit_conditions
            
            logger.warning(f"⚠️ EMERGENCY EXIT {pair}: signal_active={signal_active}, score={dataframe['squeeze_score'].iloc[-1]:.3f}, emergency_conditions={len(emergency_exit_conditions)}")
        
        # Add FreqAI exit signals if enabled - Fixed for pandas Series
        if self.use_freqai.value and '&-s-up_or_down' in dataframe.columns:
            try:
                freqai_long_exit = (
                    (dataframe['&-s-up_or_down'].iloc[-1] == 'down') and 
                    (dataframe['&-s-up_or_down_confidence'].iloc[-1] >= self.freqai_confidence_threshold.value)
                )
                freqai_short_exit = (
                    (dataframe['&-s-up_or_down'].iloc[-1] == 'up') and 
                    (dataframe['&-s-up_or_down_confidence'].iloc[-1] >= self.freqai_confidence_threshold.value)
                )
                long_exit_conditions.append(freqai_long_exit)
                short_exit_conditions.append(freqai_short_exit)
            except Exception as e:
                logger.debug(f"FreqAI exit conditions not available: {e}")
        
        # Set exit signals - ONLY FOR ACTIVE POSITION DIRECTION
        try:
            latest_idx = dataframe.index[-1]
            active_direction = self.active_positions.get(pair, None)
            
            logger.info(f"🔍 EXIT CHECK {pair}: active_direction={active_direction}, signal_active={signal_active}")
            
            # EMERGENCY EXIT: Wenn keine Direction bekannt ist, aber Exit-Bedingungen existieren
            # Passiert nach Restart wenn Position noch offen ist
            if active_direction is None and len(long_exit_conditions) > 0:
                # Versuche beide Exit-Richtungen - eine wird funktionieren
                emergency_exit_triggered = any(long_exit_conditions) or any(short_exit_conditions)
                if emergency_exit_triggered:
                    logger.warning(f"🚨 EMERGENCY EXIT TRIGGERED {pair}: trying both long and short exits")
                    dataframe.loc[latest_idx, 'exit_long'] = 1   # Versuche Long Exit
                    dataframe.loc[latest_idx, 'exit_short'] = 1  # Versuche Short Exit
                    # Eine wird funktionieren, die andere wird ignoriert
                    logger.warning(f"🚨 EMERGENCY EXIT {pair}: Both exit signals set (signal_active={signal_active}, score={dataframe['squeeze_score'].iloc[-1]:.3f})")
                    return dataframe  # Early return nach Emergency Exit
            
            # ENHANCED LONG EXIT with detailed reasoning
            elif active_direction == 'long' and len(long_exit_conditions) > 0:
                long_exit_triggered = any(long_exit_conditions)
                
                # Determine exit reason for logging
                exit_reason = 'unknown'
                if long_exit_conditions[0]:  # Signal reversal
                    exit_reason = f'signal_reversal (>{reversal_threshold:.1f})'
                elif long_exit_conditions[1]:  # Permanently inactive
                    exit_reason = f'signal_inactive ({signal_inactive_duration:.1f}min)'
                elif long_exit_conditions[2]:  # Time exit
                    exit_reason = f'time_exit ({position_age_minutes:.1f}min>{max_hold_time}min)'
                elif long_exit_conditions[3]:  # Momentum loss
                    exit_reason = f'momentum_loss ({entry_quality}→{current_quality})'
                elif long_exit_conditions[4]:  # Conflict exit
                    exit_reason = 'early_conflict_exit'
                
                logger.info(f"🔍 LONG EXIT CHECK {pair}: triggered={long_exit_triggered}, reason={exit_reason}")
                
                if long_exit_triggered:
                    dataframe.loc[latest_idx, 'exit_long'] = 1
                    # Enhanced cleanup with position tracking
                    if pair in self.entry_signals:
                        entry_data = self.entry_signals[pair].copy()  # Save for logging
                        del self.entry_signals[pair]
                    if pair in self.active_positions:
                        del self.active_positions[pair]
                    
                    logger.info(f"🟢 LONG EXIT: {pair} after {position_age_minutes:.1f}min | "
                               f"Reason: {exit_reason} | Entry: {entry_quality} | Score: {current_score:.3f}")
            
            # ENHANCED SHORT EXIT with detailed reasoning
            elif active_direction == 'short' and len(short_exit_conditions) > 0:
                short_exit_triggered = any(short_exit_conditions)
                
                # Determine exit reason for logging
                exit_reason = 'unknown'
                if short_exit_conditions[0]:  # Signal reversal
                    exit_reason = f'signal_reversal (<-{reversal_threshold:.1f})'
                elif short_exit_conditions[1]:  # Permanently inactive
                    exit_reason = f'signal_inactive ({signal_inactive_duration:.1f}min)'
                elif short_exit_conditions[2]:  # Time exit
                    exit_reason = f'time_exit ({position_age_minutes:.1f}min>{max_hold_time}min)'
                elif short_exit_conditions[3]:  # Momentum loss
                    exit_reason = f'momentum_loss ({entry_quality}→{current_quality})'
                elif short_exit_conditions[4]:  # Conflict exit
                    exit_reason = 'early_conflict_exit'
                
                logger.info(f"🔍 SHORT EXIT CHECK {pair}: triggered={short_exit_triggered}, reason={exit_reason}")
                
                if short_exit_triggered:
                    dataframe.loc[latest_idx, 'exit_short'] = 1
                    # Enhanced cleanup with position tracking
                    if pair in self.entry_signals:
                        entry_data = self.entry_signals[pair].copy()  # Save for logging
                        del self.entry_signals[pair]
                    if pair in self.active_positions:
                        del self.active_positions[pair]
                    
                    logger.info(f"🔴 SHORT EXIT: {pair} after {position_age_minutes:.1f}min | "
                               f"Reason: {exit_reason} | Entry: {entry_quality} | Score: {current_score:.3f}")
            elif active_direction:
                # Enhanced holding status
                logger.info(f"⚪ HOLDING {pair}: {active_direction} position active "
                           f"(age={position_age_minutes:.1f}min, score={current_score:.3f}, "
                           f"entry_quality={entry_quality}, current_quality={current_quality})")
                        
        except Exception as e:
            logger.error(f"Error in exit logic: {e}")
        
        return dataframe
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ADAPTIVE STOPLOSS based on signal quality and trade characteristics
        """
        try:
            # Get current leverage and trade data
            leverage = getattr(trade, 'leverage', 1.0)
            
            # Get enhanced signal data
            external_signals = self.get_external_squeeze_signals(pair)
            current_score = external_signals.get('squeeze_score_20', 0)
            signal_active = external_signals.get('signal_active', False)
            current_quality = external_signals.get('alignment_quality_20', 'NONE')
            
            # Get entry data if available
            entry_quality = 'NONE'
            entry_conflict = False
            if pair in self.entry_signals:
                entry_quality = self.entry_signals[pair].get('alignment_quality', 'NONE')
                entry_conflict = self.entry_signals[pair].get('higher_tf_conflict', False)
            
            # QUALITY-BASED BASE STOPLOSS
            if entry_quality == 'TRIPLE':
                base_risk = 0.035  # 3.5% risk for high-quality signals
            elif entry_quality == 'DOUBLE':
                base_risk = 0.030  # 3.0% risk for good signals
            elif entry_quality == 'DIVERGENCE':
                base_risk = 0.025  # 2.5% risk for divergence signals
            else:
                base_risk = 0.020  # 2.0% risk for unknown quality
            
            # Calculate leverage-adjusted stoploss
            base_stoploss = -(base_risk * leverage)
            
            # CONFLICT PENALTY
            if entry_conflict:
                base_stoploss *= 0.8  # Tighter stop for conflict trades
            
            # SIGNAL WEAKENING ADJUSTMENT
            if not signal_active and abs(current_score) < 0.1:
                # Signal gone - tighten significantly
                base_stoploss *= 0.7
            elif current_quality == 'NONE' and entry_quality in ['TRIPLE', 'DOUBLE']:
                # Quality degraded - moderate tightening
                base_stoploss *= 0.85
            
            # PROFIT-BASED TRAILING (simple implementation)
            if current_profit > 0.02:  # 2%+ profit
                # Trail stop loss to break-even + small buffer
                trailing_stop = -0.005  # 0.5% buffer
                base_stoploss = max(base_stoploss, trailing_stop)
            
            # Cap the stoploss (don't make it too tight)
            final_stoploss = max(base_stoploss, -0.15)  # Max 15% stop
            
            logger.debug(f"💡 ADAPTIVE STOPLOSS {pair}: leverage={leverage:.1f}x, "
                        f"entry_quality={entry_quality}, current_quality={current_quality}, "
                        f"signal_active={signal_active}, conflict={entry_conflict}, "
                        f"profit={current_profit:.3f}, final_stop={final_stoploss:.3f}")
            
            return final_stoploss
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stoploss: {e}")
            # Fallback to conservative stoploss
            fallback_stop = -0.025 * getattr(trade, 'leverage', 1.0)
            return max(fallback_stop, -0.10)  # Cap at 10%
        
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Confirm trade entry based on current conditions
        """
        # Always confirm for now
        return True
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, exit_reason: str,
                         current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit
        """
        # Always confirm exits
        return True
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        ADAPTIVE POSITION SIZING based on signal quality (from research)
        """
        try:
            # Get total balance
            total_balance = self.wallets.get_total(self.config['stake_currency'])
            
            # Base stake percentage
            base_percentage = 0.20  # 20% base
            
            # Get signal quality for adaptive sizing
            external_signals = self.get_external_squeeze_signals(pair)
            alignment_quality = external_signals.get('alignment_quality_20', 'NONE')
            signal_multiplier = external_signals.get('signal_multiplier_20', 1.0)
            higher_tf_conflict = external_signals.get('higher_tf_conflict', False)
            signal_strength = abs(external_signals.get('squeeze_score_20', 0))
            
            # ADAPTIVE SIZING MULTIPLIERS (from research)
            if alignment_quality == 'TRIPLE':
                quality_multiplier = 1.5  # 50% larger for triple alignment
            elif alignment_quality == 'DOUBLE':
                quality_multiplier = 1.2  # 20% larger for double alignment
            elif alignment_quality == 'DIVERGENCE':
                quality_multiplier = 0.8  # 20% smaller for divergence
            else:
                quality_multiplier = 1.0  # Standard sizing
            
            # CONFLICT PENALTY (from research)
            conflict_penalty = 0.6 if higher_tf_conflict else 1.0
            
            # SIGNAL STRENGTH BONUS
            strength_bonus = min(1.0 + (signal_strength - 0.5) * 0.4, 1.8)  # Cap at 1.8x
            
            # Calculate final sizing
            final_percentage = (base_percentage * quality_multiplier * 
                               conflict_penalty * strength_bonus * signal_multiplier)
            
            # Cap between 10% and 35% of balance
            final_percentage = max(0.10, min(final_percentage, 0.35))
            
            desired_stake = total_balance * final_percentage
            
            # Respect limits
            if min_stake and desired_stake < min_stake:
                logger.debug(f"⚠️ STAKE TOO LOW {pair}: {desired_stake:.2f} < min_stake {min_stake:.2f}")
                return min_stake
                
            if desired_stake > max_stake:
                logger.debug(f"⚠️ STAKE TOO HIGH {pair}: {desired_stake:.2f} > max_stake {max_stake:.2f}")
                return max_stake
            
            logger.info(f"✅ ADAPTIVE STAKE {pair}: {desired_stake:.2f} USDT ({final_percentage*100:.1f}%) | "
                       f"Quality: {alignment_quality} | Conflict: {higher_tf_conflict} | "
                       f"Strength: {signal_strength:.2f} | Multipliers: Q={quality_multiplier:.1f}, "
                       f"C={conflict_penalty:.1f}, S={strength_bonus:.1f}")
            
            return desired_stake
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stake amount: {e}")
            # Fallback to base 20%
            fallback_stake = total_balance * 0.20 if 'total_balance' in locals() else proposed_stake
            logger.warning(f"📉 FALLBACK STAKE {pair}: Using {fallback_stake:.2f}")
            return fallback_stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str | None, 
                side: str, **kwargs) -> float:
        """
        ADAPTIVE LEVERAGE based on signal quality and conflict detection
        """
        try:
            # Get signal quality for adaptive leverage
            external_signals = self.get_external_squeeze_signals(pair)
            alignment_quality = external_signals.get('alignment_quality_20', 'NONE')
            higher_tf_conflict = external_signals.get('higher_tf_conflict', False)
            signal_strength = abs(external_signals.get('squeeze_score_20', 0))
            
            # BASE LEVERAGE based on signal quality (from research)
            if alignment_quality == 'TRIPLE':
                base_leverage = 5.0  # Highest leverage for triple alignment
            elif alignment_quality == 'DOUBLE':
                base_leverage = 4.0  # High leverage for double alignment
            elif alignment_quality == 'DIVERGENCE':
                base_leverage = 3.0  # Moderate leverage for divergence
            else:
                base_leverage = 2.0  # Conservative for weak signals
            
            # CONFLICT PENALTY (from research)
            if higher_tf_conflict:
                final_leverage = base_leverage * 0.6  # Reduce leverage when fighting higher TF
                logger.info(f"⚠️ LEVERAGE REDUCED {pair}: {base_leverage:.1f}x → {final_leverage:.1f}x (HT conflict)")
            else:
                final_leverage = base_leverage
            
            # STRENGTH ADJUSTMENT
            if signal_strength > 1.0:
                final_leverage = min(final_leverage * 1.2, max_leverage)  # Boost for very strong signals
            
            # Ensure we don't exceed max leverage
            final_leverage = min(final_leverage, max_leverage)
            
            logger.info(f"🚀 ADAPTIVE LEVERAGE {pair}: {final_leverage:.1f}x | "
                       f"Quality: {alignment_quality} | Conflict: {higher_tf_conflict} | "
                       f"Strength: {signal_strength:.2f}")
            
            return final_leverage
            
        except Exception as e:
            logger.error(f"Error calculating adaptive leverage: {e}")
            # Fallback to conservative leverage
            fallback_leverage = 3.0
            logger.warning(f"📉 FALLBACK LEVERAGE {pair}: Using {fallback_leverage:.1f}x")
            return fallback_leverage
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced dynamic stop loss based on signal quality and leverage
        """
        try:
            # Get current squeeze signals for quality assessment
            external_signals = self.get_external_squeeze_signals(pair)
            current_score = external_signals.get('squeeze_score_20', 0)
            current_quality = external_signals.get('alignment_quality_20', 'NONE')
            
            # Calculate trade age
            trade_age_minutes = (current_time - trade.open_date).total_seconds() / 60
            
            # Base stop loss (enhanced system values)
            base_stoploss = -0.08  # Base 8% for 4x leverage
            
            # Quality-based adjustment
            if current_quality == 'TRIPLE':
                # High quality trades get tighter stops after profit
                if current_profit > 0.02:  # 2% profit
                    return base_stoploss * 0.6  # Tighten to ~5%
                else:
                    return base_stoploss * 1.25  # Give more room initially (~10%)
            elif current_quality == 'DOUBLE':
                # Medium quality trades
                if current_profit > 0.015:  # 1.5% profit
                    return base_stoploss * 0.75  # Tighten to ~6%
                else:
                    return base_stoploss
            else:
                # Lower quality or weakening signals - tighter stops
                if abs(current_score) < 0.1:  # Weak signal
                    return base_stoploss * 0.7  # Tighter stop ~5.6%
                else:
                    return base_stoploss
            
            # Time-based tightening for old positions
            if trade_age_minutes > 120:  # After 2 hours
                return base_stoploss * 0.8  # Tighten stop
            
            return base_stoploss
            
        except Exception as e:
            logger.error(f"Error in custom_stoploss: {e}")
            return -0.08  # Fallback to base stop loss
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        STATE MACHINE ENHANCED POSITION SIZING
        Dynamically adjust position size based on trading mode and signal quality
        """
        try:
            # Get signal strength for sizing calculation
            external_signals = self.get_external_squeeze_signals(pair)
            signal_strength = abs(external_signals.get('squeeze_score_20', 0))
            signal_quality_score = external_signals.get('signal_multiplier_20', 1.0)
            
            # Simple stake sizing without state machine
            final_stake = max(min_stake if min_stake else 0, min(proposed_stake, max_stake))
            
            logger.info(f"💰 STAKE SIZING {pair}: "
                       f"Proposed: {proposed_stake:.2f} → Final: {final_stake:.2f} | "
                       f"Signal: {signal_strength:.3f}")
            
            return final_stake
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount for {pair}: {e}")
            return proposed_stake  # Fallback to proposed stake
    
    # State machine integration methods removed

    def informative_pairs(self) -> list:
        """
        Additional pairs for context
        """
        return []