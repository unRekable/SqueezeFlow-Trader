"""
Squeeze Score Calculator with Futures/Spot CVD Divergence
Score range: -1.0 (strong long squeeze) to +1.0 (strong short squeeze)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SqueezeScoreCalculator:
    """
    Calculate squeeze score based on:
    - Price movement
    - Spot CVD trend
    - Futures CVD trend
    - Divergence between futures and spot
    """
    
    def __init__(self, 
                 price_weight: float = 0.3,
                 spot_cvd_weight: float = 0.35,
                 futures_cvd_weight: float = 0.35,
                 smoothing_period: int = 5,
                 enable_simplified_mode: bool = True):
        """
        Initialize calculator with component weights
        
        Args:
            price_weight: Weight for price component (0-1)
            spot_cvd_weight: Weight for spot CVD component (0-1)
            futures_cvd_weight: Weight for futures CVD component (0-1)
            smoothing_period: Period for smoothing noisy signals
        """
        self.price_weight = price_weight
        self.spot_cvd_weight = spot_cvd_weight
        self.futures_cvd_weight = futures_cvd_weight
        self.smoothing_period = smoothing_period
        self.enable_simplified_mode = enable_simplified_mode
        
        # Simplified mode parameters
        if enable_simplified_mode:
            self.divergence_threshold = 5.0      # 5M USD minimum divergence
            self.price_momentum_threshold = 0.2  # 0.2% price movement
            self.trend_window = 15               # 15 minute trend window
        
        # Ensure weights sum to 1
        total_weight = price_weight + spot_cvd_weight + futures_cvd_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            self.price_weight /= total_weight
            self.spot_cvd_weight /= total_weight
            self.futures_cvd_weight /= total_weight
            
    def calculate_price_component(self, price_data: pd.Series, lookback: int = 20) -> float:
        """
        Calculate price component of squeeze score
        
        Returns:
            Score between -1 and 1
            Positive = price rising (bearish for long squeeze)
            Negative = price falling (bearish for short squeeze)
        """
        if len(price_data) < lookback:
            return 0.0
            
        # Calculate price change percentage
        price_change = (price_data.iloc[-1] - price_data.iloc[-lookback]) / price_data.iloc[-lookback]
        
        # Normalize to -1 to 1 range (cap at ±4% for enhanced sensitivity)
        normalized_change = np.clip(price_change / 0.04, -1, 1)
        
        return normalized_change
        
    def calculate_cvd_trend(self, cvd_data: pd.Series, lookback: int = 20) -> Tuple[float, float]:
        """
        Calculate CVD trend and acceleration
        
        Returns:
            (trend_score, acceleration) both between -1 and 1
        """
        if len(cvd_data) < lookback:
            return 0.0, 0.0
            
        # Calculate trend using linear regression
        x = np.arange(len(cvd_data[-lookback:]))
        y = cvd_data[-lookback:].values
        
        # Handle NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0, 0.0
            
        x = x[mask]
        y = y[mask]
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Normalize slope for CUMULATIVE CVD (millions USD scale)
        # New scale: 100,000 USD per period is significant (adjusted for cumulative values)
        cvd_normalization_factor = 100_000  # 100k USD per period is significant
        trend_score = np.clip(slope / cvd_normalization_factor, -1, 1)
        
        # Calculate acceleration (change in slope)
        if len(cvd_data) >= lookback * 2:
            # First half slope
            y1 = cvd_data[-(lookback*2):-lookback].values
            mask1 = ~np.isnan(y1)
            if np.sum(mask1) >= 2:
                x1 = np.arange(len(y1))[mask1]
                y1 = y1[mask1]
                slope1 = np.polyfit(x1, y1, 1)[0]
                
                # Acceleration is change in slope (scaled for cumulative CVD)
                acceleration = (slope - slope1) / cvd_normalization_factor
                acceleration = np.clip(acceleration, -1, 1)
            else:
                acceleration = 0.0
        else:
            acceleration = 0.0
            
        return trend_score, acceleration
        
    def detect_simplified_divergence(self,
                                    price_data: pd.Series,
                                    spot_cvd_data: pd.Series,
                                    futures_cvd_data: pd.Series) -> Dict[str, float]:
        """
        Detect divergence using simplified approach from user's visual analysis:
        Price positive + spot volume trend positive + futures CVD trends down = Long divergence
        """
        if len(price_data) < self.trend_window:
            return {'signal_type': 'NEUTRAL', 'score': 0.0}
        
        # Calculate price change percentage over trend window
        price_start = price_data.iloc[-self.trend_window]
        price_end = price_data.iloc[-1]
        price_change_pct = ((price_end - price_start) / price_start) * 100
        
        # Calculate CVD trend directions
        spot_start = spot_cvd_data.iloc[-self.trend_window]
        spot_end = spot_cvd_data.iloc[-1]
        spot_change = spot_end - spot_start
        
        futures_start = futures_cvd_data.iloc[-self.trend_window]
        futures_end = futures_cvd_data.iloc[-1]
        futures_change = futures_end - futures_start
        
        # Determine trend directions
        spot_direction = "up" if spot_change > 1.0 else ("down" if spot_change < -1.0 else "neutral")
        futures_direction = "up" if futures_change > 1.0 else ("down" if futures_change < -1.0 else "neutral")
        
        # Calculate divergence magnitude
        divergence_amount = abs(spot_change - futures_change)
        
        # Check minimum requirements
        if abs(price_change_pct) < self.price_momentum_threshold:
            return {'signal_type': 'NEUTRAL', 'score': 0.0}
            
        if divergence_amount < self.divergence_threshold:
            return {'signal_type': 'NEUTRAL', 'score': 0.0}
        
        # LONG DIVERGENCE DETECTION (from image)
        if (price_change_pct > 0 and          # Price positive (going up)
            spot_direction == "up" and        # Spot volume trend positive
            futures_direction == "down"):     # Futures CVD trends down
            
            logger.info(f"🟢 SIMPLIFIED LONG DIVERGENCE: Price +{price_change_pct:.2f}%, "
                       f"Spot ↑{spot_change:.1f}M, Futures ↓{abs(futures_change):.1f}M, "
                       f"Divergence: {divergence_amount:.1f}M")
            
            return {
                'signal_type': 'LONG_DIVERGENCE',
                'score': -divergence_amount / 10.0,  # Negative for long signal
                'price_change_pct': price_change_pct,
                'spot_direction': spot_direction,
                'futures_direction': futures_direction,
                'divergence_amount': divergence_amount,
                'alignment_quality': 'DIVERGENCE'
            }
        
        # SHORT DIVERGENCE DETECTION (inverse pattern)
        elif (price_change_pct < 0 and        # Price negative (going down)
              spot_direction == "down" and    # Spot trend negative
              futures_direction == "up"):     # Futures CVD trends up
            
            logger.info(f"🔴 SIMPLIFIED SHORT DIVERGENCE: Price {price_change_pct:.2f}%, "
                       f"Spot ↓{abs(spot_change):.1f}M, Futures ↑{futures_change:.1f}M, "
                       f"Divergence: {divergence_amount:.1f}M")
            
            return {
                'signal_type': 'SHORT_DIVERGENCE',
                'score': divergence_amount / 10.0,   # Positive for short signal
                'price_change_pct': price_change_pct,
                'spot_direction': spot_direction,
                'futures_direction': futures_direction,
                'divergence_amount': divergence_amount,
                'alignment_quality': 'DIVERGENCE'
            }
        
        # No divergence pattern detected
        return {'signal_type': 'NEUTRAL', 'score': 0.0}
        
    def calculate_squeeze_score(self,
                              price_data: pd.Series,
                              spot_cvd_data: pd.Series,
                              futures_cvd_data: pd.Series,
                              lookback: int = 20) -> Dict[str, float]:
        """
        Calculate overall squeeze score with optional simplified mode
        
        If simplified mode is enabled, uses visual divergence detection approach.
        Otherwise uses the complex momentum alignment algorithm.
        
        Returns:
            Dictionary with score components and final score
        """
        
        # Use simplified divergence detection if enabled
        if self.enable_simplified_mode:
            simplified_result = self.detect_simplified_divergence(price_data, spot_cvd_data, futures_cvd_data)
            
            if simplified_result['signal_type'] != 'NEUTRAL':
                # Convert to full result format
                result = {
                    'squeeze_score': simplified_result['score'],
                    'raw_score': simplified_result['score'],
                    'price_component': simplified_result.get('price_change_pct', 0) / 100,
                    'spot_cvd_trend': 1.0 if simplified_result.get('spot_direction') == 'up' else -1.0,
                    'futures_cvd_trend': 1.0 if simplified_result.get('futures_direction') == 'up' else -1.0,
                    'cvd_divergence': simplified_result.get('divergence_amount', 0) / 100,
                    'spot_cvd_acceleration': 0.0,
                    'futures_cvd_acceleration': 0.0,
                    'signal_type': self._classify_signal(simplified_result['score']),
                    'signal_strength': abs(simplified_result['score']),
                    'alignment_quality': simplified_result.get('alignment_quality', 'DIVERGENCE'),
                    'signal_multiplier': 1.0,
                    'price_direction': 1 if simplified_result.get('price_change_pct', 0) > 0 else -1,
                    'spot_direction': 1 if simplified_result.get('spot_direction') == 'up' else -1,
                    'futures_direction': 1 if simplified_result.get('futures_direction') == 'up' else -1,
                    'timestamp': datetime.now()
                }
                return result
        # Fall back to complex algorithm if simplified mode is off or no signal
        # Calculate individual components
        price_component = self.calculate_price_component(price_data, lookback)
        spot_trend, spot_accel = self.calculate_cvd_trend(spot_cvd_data, lookback)
        futures_trend, futures_accel = self.calculate_cvd_trend(futures_cvd_data, lookback)
        
        # Calculate divergence between futures and spot
        cvd_divergence = futures_trend - spot_trend
        
        # DEBUG: Log calculation details
        logger.info(f"🔍 DEBUG Components: price={price_component:.4f}, spot_trend={spot_trend:.4f}, "
                   f"futures_trend={futures_trend:.4f}, divergence={cvd_divergence:.4f}")
        
        # OPTIMIZED HYBRID ALGORITHM - Momentum Alignment + Divergence
        # Based on research findings: momentum alignment works better than pure divergence
        
        # Determine directional alignment
        price_direction = np.sign(price_component)
        spot_direction = np.sign(spot_trend)
        futures_direction = np.sign(futures_trend)
        
        # Calculate alignment quality and signal strength
        alignment_quality = 'NONE'
        signal_strength_multiplier = 1.0
        
        # MOMENTUM ALIGNMENT STRATEGY (primary - proven to work)
        if price_direction == spot_direction and price_direction != 0:
            # Price and spot CVD aligned = strong momentum signal
            alignment_quality = 'DOUBLE'
            signal_strength_multiplier = 1.3
            
            if futures_direction == price_direction:
                # Triple alignment = strongest signal
                alignment_quality = 'TRIPLE'
                signal_strength_multiplier = 1.8
                
            # Base score from momentum alignment
            momentum_score = (
                abs(price_component) * 0.4 +     # 40% price momentum strength
                abs(spot_trend) * 0.3 +          # 30% spot CVD strength
                abs(futures_trend) * 0.3         # 30% futures CVD strength
            )
            
            # Apply directional sign (negative = long signal, positive = short signal)
            raw_score = momentum_score * (-1 if price_direction > 0 else 1)
            signal_type = 'MOMENTUM_ALIGNMENT'
            
        # DIVERGENCE STRATEGY (secondary - for when momentum isn't clear)
        elif abs(cvd_divergence) > 0.1:  # Significant divergence threshold
            # Traditional squeeze detection with lower weight
            alignment_quality = 'DIVERGENCE'
            signal_strength_multiplier = 0.8
            
            # Calculate divergence-based score
            divergence_strength = min(abs(cvd_divergence), 1.0)  # Cap at 1.0
            price_strength = min(abs(price_component), 1.0)
            
            # Combined divergence score
            base_strength = (divergence_strength * 0.6 + price_strength * 0.4)
            
            # Apply squeeze logic: price up + cvd divergence negative = long squeeze
            if price_component > 0.01 and cvd_divergence < -0.05:
                raw_score = -base_strength  # Long squeeze signal
                signal_type = 'LONG_SQUEEZE'
            elif price_component < -0.01 and cvd_divergence > 0.05:
                raw_score = base_strength   # Short squeeze signal
                signal_type = 'SHORT_SQUEEZE'
            else:
                raw_score = 0.0
                signal_type = 'WEAK_DIVERGENCE'
                
        else:
            # No clear signal
            raw_score = 0.0
            signal_type = 'NEUTRAL'
            alignment_quality = 'NONE'
        
        # Apply signal strength multiplier
        raw_score *= signal_strength_multiplier
        
        # WINNING PATTERN BOOST (from research)
        # Boost signals that match the 4.15% winning pattern
        spot_change_30min = spot_trend * 30 * 100_000  # Estimate 30min change
        futures_change_30min = futures_trend * 30 * 100_000
        
        # Check if matches winning pattern ranges
        winning_spot_range = (-20_000_000 <= spot_change_30min <= -5_000_000)
        winning_futures_range = (80_000_000 <= futures_change_30min <= 150_000_000)
        winning_divergence = abs(spot_change_30min - futures_change_30min) >= 100_000_000
        
        if winning_spot_range and winning_futures_range and winning_divergence:
            raw_score *= 1.5  # 50% boost for winning pattern
            signal_type += '_WINNING_PATTERN'
            
        # Enhanced logging
        if abs(raw_score) > 0.1:
            emoji = '🟢' if raw_score < 0 else '🔴'
            logger.info(f"{emoji} {signal_type} | Quality: {alignment_quality} | "
                       f"Score: {raw_score:.3f} | Price: {price_component:.3f} | "
                       f"Spot: {spot_trend:.3f} | Futures: {futures_trend:.3f} | "
                       f"Divergence: {cvd_divergence:.3f}")
        else:
            logger.debug(f"⚪ NO SIGNAL: {signal_type} | Price: {price_component:.3f} | "
                        f"Divergence: {cvd_divergence:.3f}")
            
        # Apply smoothing
        if hasattr(self, '_score_history'):
            self._score_history.append(raw_score)
            if len(self._score_history) > self.smoothing_period:
                self._score_history.pop(0)
            smoothed_score = np.mean(self._score_history)
        else:
            self._score_history = [raw_score]
            smoothed_score = raw_score
            
        # Prepare detailed results with enhanced metrics
        result = {
            'squeeze_score': smoothed_score,
            'raw_score': raw_score,
            'price_component': price_component,
            'spot_cvd_trend': spot_trend,
            'futures_cvd_trend': futures_trend,
            'cvd_divergence': cvd_divergence,
            'spot_cvd_acceleration': spot_accel,
            'futures_cvd_acceleration': futures_accel,
            'signal_type': self._classify_signal(smoothed_score),
            'signal_strength': abs(smoothed_score),
            'alignment_quality': alignment_quality,
            'signal_multiplier': signal_strength_multiplier,
            'price_direction': price_direction,
            'spot_direction': spot_direction,
            'futures_direction': futures_direction,
            'timestamp': datetime.now()
        }
        
        return result
        
    def _classify_signal(self, score: float) -> str:
        """Classify the signal based on enhanced system thresholds"""
        # Enhanced thresholds for better signal classification
        if score <= -0.35:
            return "STRONG_LONG_SQUEEZE"
        elif score <= -0.15:
            return "LONG_SQUEEZE"
        elif score <= -0.08:
            return "WEAK_LONG_SQUEEZE"
        elif score >= 0.35:
            return "STRONG_SHORT_SQUEEZE"
        elif score >= 0.15:
            return "SHORT_SQUEEZE"
        elif score >= 0.08:
            return "WEAK_SHORT_SQUEEZE"
        else:
            return "NEUTRAL"
            
    def should_enter_position(self, score_data: Dict[str, float], 
                            min_score_threshold: float = 0.2) -> Tuple[bool, str]:
        """
        Determine if we should enter a position based on score
        
        Returns:
            (should_enter, direction)
        """
        score = abs(score_data['squeeze_score'])
        
        if score < min_score_threshold:
            return False, "NONE"
            
        if score_data['squeeze_score'] < 0:
            # Long squeeze - we go long
            return True, "LONG"
        else:
            # Short squeeze - we go short
            return True, "SHORT"
            
    def should_exit_position(self, 
                           current_score: Dict[str, float],
                           position_direction: str,
                           entry_score: float,
                           entry_price: float = None,
                           current_price: float = None,
                           hard_stop_pct: float = 2.0) -> Tuple[bool, str]:
        """
        Determine if we should exit a position
        
        Args:
            current_score: Current score data
            position_direction: "LONG" or "SHORT"
            entry_score: Score when position was entered
            entry_price: Entry price (for hard stop calculation)
            current_price: Current price (for hard stop calculation)
            hard_stop_pct: Hard stop loss percentage (default 2%)
            
        Returns:
            (should_exit, reason)
        """
        score = current_score['squeeze_score']
        
        # Check hard stop loss first
        if entry_price and current_price:
            if position_direction == "LONG":
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                if loss_pct <= -hard_stop_pct:
                    return True, f"Hard stop loss hit ({loss_pct:.1f}%)"
            elif position_direction == "SHORT":
                loss_pct = ((entry_price - current_price) / entry_price) * 100
                if loss_pct <= -hard_stop_pct:
                    return True, f"Hard stop loss hit ({loss_pct:.1f}%)"
        
        # Original exit conditions (reverse signals)
        if position_direction == "LONG":
            # We're long, exit if:
            # 1. Score turns significantly positive (short squeeze)
            if score > 0.3:
                return True, "Reversal to short squeeze"
            # 2. Score returns to neutral from negative
            if score > -0.1 and entry_score < -0.6:
                return True, "Squeeze exhausted"
            # 3. Divergence disappears
            if abs(current_score['cvd_divergence']) < 0.1:
                return True, "No more divergence"
                
        elif position_direction == "SHORT":
            # We're short, exit if:
            # 1. Score turns significantly negative (long squeeze)
            if score < -0.3:
                return True, "Reversal to long squeeze"
            # 2. Score returns to neutral from positive
            if score < 0.1 and entry_score > 0.6:
                return True, "Squeeze exhausted"
            # 3. Divergence disappears
            if abs(current_score['cvd_divergence']) < 0.1:
                return True, "No more divergence"
                
        return False, ""
        
    def get_score_description(self, score_data: Dict[str, float]) -> str:
        """Get human-readable description of current conditions"""
        score = score_data['squeeze_score']
        
        if abs(score) < 0.1:
            return "Neutral - No clear squeeze signal"
            
        direction = "LONG SQUEEZE" if score < 0 else "SHORT SQUEEZE"
        strength = "Strong" if abs(score) > 0.6 else "Moderate"
        
        desc = f"{strength} {direction} (Score: {score:.2f})\n"
        desc += f"  Price: {'UP' if score_data['price_component'] > 0 else 'DOWN'} ({score_data['price_component']:.2f})\n"
        desc += f"  Spot CVD: {'UP' if score_data['spot_cvd_trend'] > 0 else 'DOWN'} ({score_data['spot_cvd_trend']:.2f})\n"
        desc += f"  Futures CVD: {'UP' if score_data['futures_cvd_trend'] > 0 else 'DOWN'} ({score_data['futures_cvd_trend']:.2f})\n"
        desc += f"  Divergence: {score_data['cvd_divergence']:.2f}"
        
        return desc