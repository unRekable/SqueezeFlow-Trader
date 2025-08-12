"""All indicator calculations for SqueezeFlow strategy"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from veloce.core.config import VeloceConfig, CONFIG
from veloce.core.types import IndicatorData

logger = logging.getLogger(__name__)


class Indicators:
    """All indicator calculations in one place"""
    
    def __init__(self, config: VeloceConfig = CONFIG):
        """Initialize with configuration"""
        self.config = config
        self.indicator_config = config.get_indicator_config()
        
    def calculate_squeeze_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate squeeze momentum indicator (TTM Squeeze)"""
        if df.empty or len(df) < self.config.squeeze_period:
            return df
        
        try:
            # Bollinger Bands
            bb_period = self.config.squeeze_period
            bb_mult = self.config.squeeze_bb_mult
            
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_mult * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_mult * bb_std)
            
            # Keltner Channels
            kc_period = self.config.squeeze_period
            kc_mult = self.config.squeeze_kc_mult
            
            # True Range
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # ATR
            df['atr'] = df['true_range'].rolling(kc_period).mean()
            
            # Keltner Channels
            df['kc_middle'] = df['close'].rolling(kc_period).mean()
            df['kc_upper'] = df['kc_middle'] + (kc_mult * df['atr'])
            df['kc_lower'] = df['kc_middle'] - (kc_mult * df['atr'])
            
            # Squeeze detection (BB inside KC)
            df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
            df['squeeze'] = df['squeeze'].astype(float)
            
            # Momentum calculation
            momentum_length = self.config.squeeze_momentum_length
            
            # Linear regression for momentum
            def linreg(series, length):
                """Calculate linear regression value"""
                if len(series) < length:
                    return np.nan
                x = np.arange(length)
                y = series.iloc[-length:].values
                # Check for valid values
                if np.isnan(y).any():
                    return np.nan
                # Calculate linear regression
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                return m * (length - 1) + c
            
            # Calculate momentum
            highest_high = df['high'].rolling(momentum_length).max()
            lowest_low = df['low'].rolling(momentum_length).min()
            df['hl_avg'] = (highest_high + lowest_low) / 2
            df['close_hl_avg'] = df['close'] - df['hl_avg']
            
            # Apply linear regression
            df['momentum'] = df['close_hl_avg'].rolling(momentum_length).apply(
                lambda x: linreg(x, len(x)) if len(x) == momentum_length else np.nan
            )
            
            # Smooth momentum if configured
            if self.config.squeeze_momentum_smooth > 1:
                df['momentum'] = df['momentum'].rolling(self.config.squeeze_momentum_smooth).mean()
            
            # Clean up temporary columns
            temp_cols = ['high_low', 'high_close', 'low_close', 'true_range', 'hl_avg', 'close_hl_avg']
            df.drop(columns=temp_cols, inplace=True, errors='ignore')
            
            logger.debug(f"Calculated squeeze momentum, squeeze active: {df['squeeze'].iloc[-1] if not df.empty else False}")
            
        except Exception as e:
            logger.error(f"Error calculating squeeze momentum: {e}")
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator"""
        if df.empty or len(df) < period:
            return df
        
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            logger.debug(f"Calculated RSI: {df['rsi'].iloc[-1]:.2f}" if not df.empty else "No RSI")
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        if df.empty or len(df) < slow:
            return df
        
        try:
            df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Clean up
            df.drop(columns=['ema_fast', 'ema_slow'], inplace=True, errors='ignore')
            
            logger.debug(f"Calculated MACD: {df['macd'].iloc[-1]:.4f}" if not df.empty else "No MACD")
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        return df
    
    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """Calculate volume profile"""
        if df.empty:
            return {}
        
        try:
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # Calculate volume at each price level
            volume_profile = {}
            
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                
                # Find candles that touched this price range
                mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
                bin_volume = df.loc[mask, 'volume'].sum()
                
                volume_profile[f"{bin_low:.2f}-{bin_high:.2f}"] = {
                    'volume': float(bin_volume),
                    'price': float((bin_low + bin_high) / 2)
                }
            
            # Find POC (Point of Control - highest volume price)
            if volume_profile:
                poc = max(volume_profile.items(), key=lambda x: x[1]['volume'])
                return {
                    'profile': volume_profile,
                    'poc_price': poc[1]['price'],
                    'poc_volume': poc[1]['volume']
                }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
        
        return {}
    
    def detect_cvd_divergence(self, cvd_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect CVD divergence between spot and perp"""
        if cvd_df.empty or price_df.empty:
            return {'divergence': False, 'type': None, 'strength': 0.0}
        
        try:
            # Ensure we have required columns
            if 'spot_cvd_cumulative' not in cvd_df.columns or 'perp_cvd_cumulative' not in cvd_df.columns:
                return {'divergence': False, 'type': None, 'strength': 0.0}
            
            # Get recent data
            lookback = min(self.config.cvd_lookback, len(cvd_df))
            recent_cvd = cvd_df.tail(lookback)
            recent_price = price_df.tail(lookback)
            
            if len(recent_cvd) < 10:  # Need minimum data
                return {'divergence': False, 'type': None, 'strength': 0.0}
            
            # Calculate divergence
            spot_cvd_change = recent_cvd['spot_cvd_cumulative'].iloc[-1] - recent_cvd['spot_cvd_cumulative'].iloc[0]
            perp_cvd_change = recent_cvd['perp_cvd_cumulative'].iloc[-1] - recent_cvd['perp_cvd_cumulative'].iloc[0]
            price_change = recent_price['close'].iloc[-1] - recent_price['close'].iloc[0]
            
            # Normalize changes
            price_pct = price_change / recent_price['close'].iloc[0] if recent_price['close'].iloc[0] != 0 else 0
            
            # Detect divergence types
            divergence_type = None
            strength = 0.0
            
            # Bullish divergence: Price down but CVD up
            if price_pct < -self.config.cvd_threshold and spot_cvd_change > 0:
                divergence_type = 'bullish'
                strength = abs(spot_cvd_change) / (abs(price_pct) + 1)
            
            # Bearish divergence: Price up but CVD down
            elif price_pct > self.config.cvd_threshold and spot_cvd_change < 0:
                divergence_type = 'bearish'
                strength = abs(spot_cvd_change) / (abs(price_pct) + 1)
            
            # Spot/Perp divergence
            cvd_diff = abs(spot_cvd_change - perp_cvd_change)
            if cvd_diff > self.config.cvd_divergence_min_strength:
                if divergence_type is None:
                    divergence_type = 'spot_perp'
                strength = max(strength, cvd_diff)
            
            result = {
                'divergence': divergence_type is not None,
                'type': divergence_type,
                'strength': float(strength),
                'spot_cvd_change': float(spot_cvd_change),
                'perp_cvd_change': float(perp_cvd_change),
                'price_change_pct': float(price_pct)
            }
            
            if divergence_type:
                logger.info(f"CVD divergence detected: {divergence_type} with strength {strength:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting CVD divergence: {e}")
            return {'divergence': False, 'type': None, 'strength': 0.0}
    
    def calculate_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market structure (higher highs/lows, support/resistance)"""
        if df.empty or len(df) < 20:
            return {}
        
        try:
            # Find swing points
            window = 5
            
            # Swing highs and lows
            df['swing_high'] = df['high'].rolling(window=window*2+1, center=True).apply(
                lambda x: x[window] if x[window] == x.max() else np.nan
            )
            df['swing_low'] = df['low'].rolling(window=window*2+1, center=True).apply(
                lambda x: x[window] if x[window] == x.min() else np.nan
            )
            
            # Get valid swing points
            swing_highs = df['swing_high'].dropna()
            swing_lows = df['swing_low'].dropna()
            
            # Determine trend
            trend = 'neutral'
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Check for higher highs and higher lows (uptrend)
                if swing_highs.iloc[-1] > swing_highs.iloc[-2] and swing_lows.iloc[-1] > swing_lows.iloc[-2]:
                    trend = 'bullish'
                # Check for lower highs and lower lows (downtrend)
                elif swing_highs.iloc[-1] < swing_highs.iloc[-2] and swing_lows.iloc[-1] < swing_lows.iloc[-2]:
                    trend = 'bearish'
            
            # Find support and resistance levels
            recent_highs = df['high'].tail(50).nlargest(3).mean() if len(df) >= 50 else df['high'].max()
            recent_lows = df['low'].tail(50).nsmallest(3).mean() if len(df) >= 50 else df['low'].min()
            
            # Clean up
            df.drop(columns=['swing_high', 'swing_low'], inplace=True, errors='ignore')
            
            return {
                'trend': trend,
                'resistance': float(recent_highs),
                'support': float(recent_lows),
                'last_swing_high': float(swing_highs.iloc[-1]) if len(swing_highs) > 0 else None,
                'last_swing_low': float(swing_lows.iloc[-1]) if len(swing_lows) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating market structure: {e}")
            return {}
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators at once"""
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Calculate indicators in order
        df = self.calculate_squeeze_momentum(df)
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)
        
        logger.info(f"Calculated all indicators for {len(df)} candles")
        return df
    
    def get_indicator_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract indicator signals from calculated data"""
        if df.empty:
            return {}
        
        try:
            latest = df.iloc[-1]
            signals = {}
            
            # Squeeze signal
            if 'squeeze' in df.columns:
                signals['squeeze'] = {
                    'active': bool(latest['squeeze']),
                    'momentum': float(latest.get('momentum', 0)),
                    'momentum_increasing': float(latest.get('momentum', 0)) > float(df['momentum'].iloc[-2]) if len(df) > 1 and 'momentum' in df.columns else False
                }
            
            # RSI signal
            if 'rsi' in df.columns:
                rsi_value = float(latest['rsi'])
                signals['rsi'] = {
                    'value': rsi_value,
                    'oversold': rsi_value < 30,
                    'overbought': rsi_value > 70
                }
            
            # MACD signal
            if 'macd' in df.columns:
                signals['macd'] = {
                    'value': float(latest['macd']),
                    'signal': float(latest.get('macd_signal', 0)),
                    'histogram': float(latest.get('macd_histogram', 0)),
                    'bullish_cross': False,
                    'bearish_cross': False
                }
                
                # Check for crossovers
                if len(df) > 1 and 'macd_signal' in df.columns:
                    prev = df.iloc[-2]
                    if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                        signals['macd']['bullish_cross'] = True
                    elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                        signals['macd']['bearish_cross'] = True
            
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting indicator signals: {e}")
            return {}