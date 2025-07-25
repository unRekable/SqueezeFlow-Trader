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

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade


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
    
    stoploss = -0.10  # Dynamic stop loss - will be adjusted based on leverage in custom_stoploss()
    
    timeframe = '1m'
    
    # Freqtrade settings
    can_short = True
    use_exit_signal = True  # Enable automatic signal-based trading
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # FreqAI settings
    process_only_new_candles = True
    
    stoploss_on_exchange = False
    startup_candle_count = 120
    
    # Strategy parameters
    squeeze_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="buy")  # Increased from 0.2 to 0.5 for fewer noise trades
    rsi_oversold = IntParameter(20, 40, default=30, space="buy")
    rsi_overbought = IntParameter(60, 80, default=70, space="sell")
    volume_threshold = DecimalParameter(1.1, 2.0, default=1.5, space="buy")
    
    # FreqAI parameters - ML as supportive, not blocking
    use_freqai = BooleanParameter(default=True, space="buy")
    freqai_confidence_threshold = DecimalParameter(0.3, 0.7, default=0.5, space="buy")  # Lower threshold
    ml_weight = DecimalParameter(0.1, 0.5, default=0.3, space="buy")  # ML influence weight
    
    def __init__(self, config: Dict) -> None:
        """Initialize strategy"""
        super().__init__(config)
        
        # Only connect to significant_trades for OI data (CVD now comes from Redis signals)
        self.squeezeflow_client = InfluxDBClient(
            host='aggr-influx',  # Use migrated InfluxDB
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
        
    def setup_redis_connection(self):
        """Setup Redis connection for external signals"""
        try:
            redis_url = self.config.get('redis_url', 'redis://redis:6379')  # Use container name
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for SqueezeFlow signals")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            
    def get_external_squeeze_signals(self, pair: str, timeframe: str = '1m') -> Dict:
        """
        Get external squeeze signals with PERSISTENCE - Solution for timing problem
        """
        if not self.redis_client:
            logger.warning(f"🔴 Redis client not available for {pair}")
            return {}
            
        try:
            # Convert pair to base symbol for Market Discovery (e.g., BTC/USDT:USDT -> BTC)
            symbol = pair.split('/')[0]  # Extract base symbol for robust market discovery
            
            # AKTUELLE Redis-Signale holen
            current_signals = {}
            signal_count = 0
            
            for lookback in [5, 10, 20, 30, 60, 120, 240]:  # Erweiterte Timeframes: 1h, 2h, 4h
                key = f"squeeze_signal:{symbol}:{lookback}"
                signal_data = self.redis_client.get(key)
                
                if signal_data:
                    signal = json.loads(signal_data)
                    current_signals[f'squeeze_score_{lookback}'] = signal.get('squeeze_score', 0)
                    current_signals[f'signal_strength_{lookback}'] = signal.get('signal_strength', 0)
                    current_signals[f'signal_type_{lookback}'] = signal.get('signal_type', 'NEUTRAL')
                    signal_count += 1
                else:
                    current_signals[f'squeeze_score_{lookback}'] = 0
                    current_signals[f'signal_strength_{lookback}'] = 0
                    current_signals[f'signal_type_{lookback}'] = 'NEUTRAL'
            
            # SIGNAL PERSISTENCE LOGIC
            current_score = current_signals.get('squeeze_score_20', 0)
            
            # Initialize cache for this pair
            if pair not in self.signal_cache:
                self.signal_cache[pair] = {
                    'score': 0, 
                    'timestamp': datetime.now(), 
                    'active': False,
                    'strength': 0,
                    'type': 'NEUTRAL'
                }
            
            cache = self.signal_cache[pair]
            
            # ENTRY LOGIC: New strong signal detected OR signal refresh
            if abs(current_score) > 0.15:
                if not cache['active']:
                    # Komplett neues Signal
                    cache['score'] = current_score
                    cache['timestamp'] = datetime.now()
                    cache['active'] = True
                    cache['strength'] = current_signals.get('signal_strength_20', 0)
                    cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
                    
                    # Save entry signal for timing
                    self.entry_signals[pair] = cache.copy()
                    
                    logger.info(f"🎯 NEW SIGNAL ACTIVATED {symbol}: score={current_score:.3f}, type={cache['type']}")
                    
                elif abs(current_score) > abs(cache['score']) * 1.5:  # Signal 50% stronger
                    # Signal refresh: Strong new signal overrides old one
                    cache['score'] = current_score
                    cache['timestamp'] = datetime.now()  # REFRESH TIMESTAMP!
                    cache['strength'] = current_signals.get('signal_strength_20', 0)
                    cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
                    
                    # Refresh entry signal for timing
                    self.entry_signals[pair] = cache.copy()
                    
                    logger.info(f"🔄 SIGNAL REFRESHED {symbol}: score={current_score:.3f} (was {cache['score']:.3f}), type={cache['type']}")
                
            # EXIT LOGIC: Signal weakens significantly
            elif abs(current_score) < 0.1 and cache['active']:
                cache['active'] = False
                logger.info(f"🔄 SIGNAL DEACTIVATED {symbol}: score dropped to {current_score:.3f}")
            
            # UPDATE: Aktualisiere aktive Signale mit neuen Werten
            elif cache['active'] and abs(current_score) > 0.1:
                cache['score'] = current_score  # Update mit aktuellem Score
                cache['strength'] = current_signals.get('signal_strength_20', 0)
                cache['type'] = current_signals.get('signal_type_20', 'NEUTRAL')
            
            # RETURN: Use persisted signals for active ones, otherwise current
            if cache['active']:
                # Aktives Signal: Verwende Cache-Werte
                return_signals = current_signals.copy()
                return_signals['squeeze_score_20'] = cache['score']
                return_signals['signal_strength_20'] = cache['strength']
                return_signals['signal_type_20'] = cache['type']
                return_signals['signal_active'] = True
                return_signals['signal_age_minutes'] = (datetime.now() - cache['timestamp']).total_seconds() / 60
            else:
                # Kein aktives Signal: Verwende aktuelle schwache Werte
                return_signals = current_signals.copy()
                return_signals['signal_active'] = False
                return_signals['signal_age_minutes'] = 0
            
            return return_signals
            
        except Exception as e:
            logger.error(f"❌ Error getting squeeze signals for {pair}: {e}")
            return {}
            
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
        Populate entry signals with PERSISTENT SIGNAL TIMING
        """
        pair = metadata['pair']
        
        # Get persistent signals with timing info
        external_signals = self.get_external_squeeze_signals(pair)
        signal_active = external_signals.get('signal_active', False)
        signal_age = external_signals.get('signal_age_minutes', 0)
        
        # DEBUG: Log current signal status
        logger.info(f"🔍 ENTRY CHECK {pair}: score={external_signals.get('squeeze_score_20', 0):.3f}, active={signal_active}, age={signal_age:.1f}min")
        
        # LONG ENTRY CONDITIONS with TIMING
        long_conditions = [
            # PRIMARY: Signal aktiv und frisch (< 5 Minuten)
            (dataframe['squeeze_score'] <= -self.squeeze_threshold.value),
            (signal_active == True),  # Signal muss aktiv sein
            (signal_age < 5),  # Signal max 5 Minuten alt
            # CONFIRMATION: Multiple timeframe alignment mit erweiterten Timeframes
            # Entry timing: 10min or 30min confirmation
            ((dataframe['squeeze_score_10'] <= -0.15) | (dataframe['squeeze_score_30'] <= -0.2)),  # Entry timing
            # Primary timeframes: 1h+ signal for real squeeze confirmation - ACTIVATED!
            ((dataframe['squeeze_score_60'] <= -0.15) | (dataframe['squeeze_score_120'] <= -0.1) | (dataframe['squeeze_score_240'] <= -0.05)),  # 1h,2h,4h trend confirmation
            # FILTER: Basic technical conditions
            (dataframe['rsi'] < self.rsi_overbought.value),
            (dataframe['volume'] > dataframe['volume_sma'] * self.volume_threshold.value),
            # OI confirmation - if available
            (dataframe['oi_normalized'] > 0.8),  # Less restrictive
            (dataframe['oi_momentum'] > -0.1)   # Less restrictive
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
        
        # SHORT ENTRY CONDITIONS with TIMING
        short_conditions = [
            # PRIMARY: Signal aktiv und frisch (< 5 Minuten)
            (dataframe['squeeze_score'] >= self.squeeze_threshold.value),
            (signal_active == True),  # Signal muss aktiv sein  
            (signal_age < 5),  # Signal max 5 Minuten alt
            # CONFIRMATION: Multiple timeframe alignment mit erweiterten Timeframes  
            # Entry timing: 10min or 30min confirmation
            ((dataframe['squeeze_score_10'] >= 0.15) | (dataframe['squeeze_score_30'] >= 0.2)),  # Entry timing
            # Primary timeframes: 1h+ signal for real squeeze confirmation - ACTIVATED!
            ((dataframe['squeeze_score_60'] >= 0.15) | (dataframe['squeeze_score_120'] >= 0.1) | (dataframe['squeeze_score_240'] >= 0.05)),  # 1h,2h,4h trend confirmation
            # FILTER: Basic technical conditions
            (dataframe['rsi'] > self.rsi_oversold.value),
            (dataframe['volume'] > dataframe['volume_sma'] * self.volume_threshold.value),
            # OI confirmation - if available
            (dataframe['oi_normalized'] > 0.8),  # Less restrictive
            (dataframe['oi_momentum'] > -0.1)   # Less restrictive
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
        
        # Set entry signals with TIMING-based logic
        if signal_active and signal_age < 5:  # Nur bei frischen aktiven Signalen
            
            # LONG ENTRY: Check conditions on latest candle only  
            if len(long_conditions) > 0:
                # Check all conditions for latest candle
                try:
                    latest_idx = dataframe.index[-1]
                    long_check = all([
                        condition.iloc[-1] if hasattr(condition, 'iloc') else condition 
                        for condition in long_conditions
                    ])
                    
                    if long_check:
                        dataframe.loc[latest_idx, 'enter_long'] = 1
                        # Entry-Zeit und Direction speichern - SAFE INIT
                        if pair not in self.entry_signals:
                            self.entry_signals[pair] = {}
                        self.entry_signals[pair]['entry_time'] = datetime.now()
                        self.entry_signals[pair]['direction'] = 'long'
                        self.active_positions[pair] = 'long'
                        logger.info(f"🟢 LONG ENTRY TRIGGERED: {pair} at {dataframe['close'].iloc[-1]:.2f}")
                except Exception as e:
                    logger.error(f"Error in long entry logic: {e}")
            
            # SHORT ENTRY: Check conditions on latest candle only
            if len(short_conditions) > 0:
                try:
                    latest_idx = dataframe.index[-1] 
                    short_check = all([
                        condition.iloc[-1] if hasattr(condition, 'iloc') else condition
                        for condition in short_conditions
                    ])
                    
                    if short_check:
                        dataframe.loc[latest_idx, 'enter_short'] = 1
                        # Entry-Zeit und Direction speichern - SAFE INIT
                        if pair not in self.entry_signals:
                            self.entry_signals[pair] = {}
                        self.entry_signals[pair]['entry_time'] = datetime.now()
                        self.entry_signals[pair]['direction'] = 'short'
                        self.active_positions[pair] = 'short'
                        logger.info(f"🔴 SHORT ENTRY TRIGGERED: {pair} at {dataframe['close'].iloc[-1]:.2f}")
                except Exception as e:
                    logger.error(f"Error in short entry logic: {e}")
        else:
            # Debug: Warum kein Entry?
            if not signal_active:
                logger.info(f"⚪ No entry for {pair}: No active signal (score={external_signals.get('squeeze_score_20', 0):.3f})")
            elif signal_age >= 5:
                logger.info(f"⚪ No entry for {pair}: Signal too old ({signal_age:.1f} min)")
            else:
                logger.info(f"⚪ No entry for {pair}: Other conditions not met (age={signal_age:.1f}min, active={signal_active})")
        
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
            
            # LONG EXIT CONDITIONS - IMMEDIATE EXITS possible
            long_exit_conditions = [
                # 1. Signal reversal: Strong opposite signal (Short Squeeze) - stricter threshold
                dataframe['squeeze_score'].iloc[-1] > 1.0,  # Increased from 0.7 to 1.0 for real reversals
                # 2. Signal PERMANENTLY inactive with stronger filter
                (signal_inactive_duration > 30 and abs(dataframe['squeeze_score'].iloc[-1]) < 0.02),  # 30min + score < 0.02
                # 3. Lange Position mit schwachem Signal (Emergency Exit)
                (position_age_minutes > 120 and abs(dataframe['squeeze_score'].iloc[-1]) < 0.05)  # 2h + sehr schwaches Signal
            ]
            
            # SHORT EXIT CONDITIONS - IMMEDIATE EXITS possible  
            short_exit_conditions = [
                # 1. Signal reversal: Strong opposite signal (Long Squeeze) - stricter threshold
                dataframe['squeeze_score'].iloc[-1] < -1.0,  # Increased from -0.7 to -1.0 for real reversals
                # 2. Signal PERMANENTLY inactive with stronger filter
                (signal_inactive_duration > 30 and abs(dataframe['squeeze_score'].iloc[-1]) < 0.02),  # 30min + score < 0.02
                # 3. Lange Position mit schwachem Signal (Emergency Exit)
                (position_age_minutes > 120 and abs(dataframe['squeeze_score'].iloc[-1]) < 0.05)  # 2h + sehr schwaches Signal
            ]
                
            logger.info(f"🔍 EXIT CONDITIONS {pair}: age={position_age_minutes:.1f}min, score={dataframe['squeeze_score'].iloc[-1]:.3f}, inactive_duration={signal_inactive_duration:.1f}min, conditions_count=long:{len(long_exit_conditions)}/short:{len(short_exit_conditions)}")
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
            
            # LONG EXIT: Nur wenn LONG-Position aktiv ist UND Exit-Bedingungen vorhanden
            elif active_direction == 'long' and len(long_exit_conditions) > 0:
                long_exit_triggered = any(long_exit_conditions)
                logger.info(f"🔍 LONG EXIT CHECK {pair}: triggered={long_exit_triggered}, conditions={long_exit_conditions}")
                if long_exit_triggered:
                    dataframe.loc[latest_idx, 'exit_long'] = 1
                    # Cleanup position - SAFE DELETE
                    if pair in self.entry_signals:
                        if 'entry_time' in self.entry_signals[pair]:
                            del self.entry_signals[pair]['entry_time']
                        if 'direction' in self.entry_signals[pair]:
                            del self.entry_signals[pair]['direction']
                        if not self.entry_signals[pair]:  # Dict leer?
                            del self.entry_signals[pair]
                    if pair in self.active_positions:
                        del self.active_positions[pair]
                    position_age = (datetime.now() - self.entry_signals[pair].get('entry_time', datetime.now())).total_seconds() / 60 if pair in self.entry_signals and 'entry_time' in self.entry_signals[pair] else 0
                    logger.info(f"🟢 LONG EXIT TRIGGERED: {pair} after {position_age:.1f}min")
            
            # SHORT EXIT: Nur wenn SHORT-Position aktiv ist UND Exit-Bedingungen vorhanden
            elif active_direction == 'short' and len(short_exit_conditions) > 0:
                short_exit_triggered = any(short_exit_conditions)
                logger.info(f"🔍 SHORT EXIT CHECK {pair}: triggered={short_exit_triggered}, conditions={short_exit_conditions}")
                if short_exit_triggered:
                    dataframe.loc[latest_idx, 'exit_short'] = 1
                    # Cleanup position - SAFE DELETE
                    if pair in self.entry_signals:
                        if 'entry_time' in self.entry_signals[pair]:
                            del self.entry_signals[pair]['entry_time']
                        if 'direction' in self.entry_signals[pair]:
                            del self.entry_signals[pair]['direction']
                        if not self.entry_signals[pair]:  # Dict leer?
                            del self.entry_signals[pair]
                    if pair in self.active_positions:
                        del self.active_positions[pair]
                    position_age = (datetime.now() - self.entry_signals[pair].get('entry_time', datetime.now())).total_seconds() / 60 if pair in self.entry_signals and 'entry_time' in self.entry_signals[pair] else 0
                    logger.info(f"🔴 SHORT EXIT TRIGGERED: {pair} after {position_age:.1f}min")
            elif active_direction:
                position_age = (datetime.now() - self.entry_signals[pair].get('entry_time', datetime.now())).total_seconds() / 60 if pair in self.entry_signals and 'entry_time' in self.entry_signals[pair] else 0
                logger.info(f"⚪ HOLDING {pair}: Position {active_direction} active, no exit conditions met (age={position_age:.1f}min)")
                        
        except Exception as e:
            logger.error(f"Error in exit logic: {e}")
        
        return dataframe
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stoploss based on leverage - maintains 2-3% spot price movement risk
        Mit 5x Leverage: 12.5% stoploss = 2.5% Spot-Bewegung
        """
        
        # Get current leverage from trade
        leverage = getattr(trade, 'leverage', 1.0)
        
        # Calculate dynamic stoploss based on leverage
        # Target: 2.5% spot price movement regardless of leverage
        if leverage >= 5.0:
            # 5x Leverage: 12.5% stoploss = 2.5% Spot risk
            base_stoploss = -0.125  # -12.5%
        elif leverage >= 3.0:
            # 3x Leverage: 7.5% stoploss = 2.5% Spot risk  
            base_stoploss = -0.075  # -7.5%
        elif leverage >= 2.0:
            # 2x Leverage: 5% stoploss = 2.5% Spot risk
            base_stoploss = -0.05   # -5%
        else:
            # 1x Leverage: 2.5% stoploss = 2.5% Spot risk
            base_stoploss = -0.025  # -2.5%
        
        # Squeeze-Signal-basierte Anpassung
        external_signals = self.get_external_squeeze_signals(pair)
        current_score = external_signals.get('squeeze_score', 0)
        signal_active = external_signals.get('signal_active', False)
        
        stoploss = base_stoploss
        
        # Tighten stoploss if squeeze signal is weakening significantly
        if not signal_active and abs(current_score) < 0.1:
            # Signal komplett weg - tighten stoploss um 20%
            stoploss = stoploss * 0.8  # 20% enger
            
        logger.debug(f"💡 DYNAMIC STOPLOSS {pair}: leverage={leverage}x, base={base_stoploss:.3f}, final={stoploss:.3f}")
        
        return stoploss
        
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
        Custom stake amount: 20% of total balance per trade
        Used with stake_amount: "unlimited" for proper dynamic staking
        """
        try:
            # Get total balance from wallets
            total_balance = self.wallets.get_total(self.config['stake_currency'])
            
            # Calculate 20% of total balance for each trade
            stake_percentage = 0.20
            desired_stake = total_balance * stake_percentage
            
            # Log for debugging
            logger.debug(f"💰 CUSTOM STAKE {pair}: total_balance={total_balance:.2f}, desired={desired_stake:.2f}, proposed={proposed_stake:.2f}")
            
            # Ensure we respect min/max limits
            if min_stake and desired_stake < min_stake:
                logger.debug(f"⚠️ STAKE TOO LOW {pair}: {desired_stake:.2f} < min_stake {min_stake:.2f}")
                return min_stake
                
            if desired_stake > max_stake:
                logger.debug(f"⚠️ STAKE TOO HIGH {pair}: {desired_stake:.2f} > max_stake {max_stake:.2f}")
                return max_stake
                
            logger.info(f"✅ CUSTOM STAKE {pair}: Using {desired_stake:.2f} USDT (20% of {total_balance:.2f})")
            return desired_stake
            
        except Exception as e:
            logger.error(f"Error calculating custom stake amount: {e}")
            # Fallback to proposed stake from unlimited mode
            logger.warning(f"📉 FALLBACK STAKE {pair}: Using proposed {proposed_stake:.2f}")
            return proposed_stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str | None, 
                side: str, **kwargs) -> float:
        """
        Set leverage for futures trading - Squeeze strategy with higher leverage
        """
        # Debug logging
        logger.info(f"🔍 LEVERAGE CALL {pair}: proposed={proposed_leverage}, max={max_leverage}, tag={entry_tag}, side={side}")
        
        # Squeeze signals are precise → higher leverage justified
        if entry_tag == "force_entry":
            leverage_value = 5.0  # Higher leverage for manual tests
        else:
            leverage_value = 5.0  # Aggressive leverage for squeeze trades (due to high precision)
            
        logger.info(f"🚀 LEVERAGE SET {pair}: returning {leverage_value}x (was proposed: {proposed_leverage}x)")
        return leverage_value
        
    def informative_pairs(self) -> list:
        """
        Additional pairs for context
        """
        return []