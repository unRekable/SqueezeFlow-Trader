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
                 smoothing_period: int = 5):
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
        
        # Normalize to -1 to 1 range (cap at ±5% for full signal)
        normalized_change = np.clip(price_change / 0.05, -1, 1)
        
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
        
    def calculate_squeeze_score(self,
                              price_data: pd.Series,
                              spot_cvd_data: pd.Series,
                              futures_cvd_data: pd.Series,
                              lookback: int = 20) -> Dict[str, float]:
        """
        Calculate overall squeeze score
        
        Long Squeeze (negative score):
        - Price UP + Spot CVD UP + Futures CVD DOWN
        
        Short Squeeze (positive score):
        - Price DOWN + Spot CVD DOWN + Futures CVD UP
        
        Returns:
            Dictionary with score components and final score
        """
        # Calculate individual components
        price_component = self.calculate_price_component(price_data, lookback)
        spot_trend, spot_accel = self.calculate_cvd_trend(spot_cvd_data, lookback)
        futures_trend, futures_accel = self.calculate_cvd_trend(futures_cvd_data, lookback)
        
        # Calculate divergence between futures and spot
        cvd_divergence = futures_trend - spot_trend
        
        # DEBUG: Log calculation details
        logger.info(f"🔍 DEBUG Components: price={price_component:.4f}, spot_trend={spot_trend:.4f}, "
                   f"futures_trend={futures_trend:.4f}, divergence={cvd_divergence:.4f}")
        
        # MUCH MORE LENIENT SQUEEZE DETECTION
        
        # Kombiniere alle Indikatoren für eine gewichtete Score-Berechnung
        # Keine perfekten Bedingungen mehr erforderlich
        
        # Preis-Komponente: Normalisiert auf 0-1
        price_factor = abs(price_component) if abs(price_component) < 1.0 else 1.0
        
        # CVD-Divergenz: Normalisiert auf 0-1
        divergence_factor = min(abs(cvd_divergence), 1.0)
        
        # CVD-Trend-Stärke: Kombiniere beide Trends
        trend_factor = (abs(spot_trend) + abs(futures_trend)) / 2
        trend_factor = min(trend_factor, 1.0)
        
        # NEUE SQUEEZE-DETECTION: Viel weniger restriktiv
        
        # LONG SQUEEZE: Preis steigt ODER CVD-Divergenz negativ (nicht UND!)
        if price_component > 0.01 or cvd_divergence < -0.05:  # Viel niedrigere Schwelle
            long_score = (
                price_factor * 0.3 +          # Preis-Komponente
                divergence_factor * 0.4 +     # Divergenz-Stärke  
                trend_factor * 0.3            # CVD-Trend-Stärke
            )
            if cvd_divergence < 0 or price_component > 0:  # Zumindest eine Bedingung
                raw_score = -long_score
                logger.info(f"🟢 LONG SQUEEZE: price={price_component:.3f}, div={cvd_divergence:.3f}, score={raw_score:.3f}")
            else:
                raw_score = 0.0
                
        # SHORT SQUEEZE: Preis fällt ODER CVD-Divergenz positiv (nicht UND!)
        elif price_component < -0.01 or cvd_divergence > 0.05:  # Viel niedrigere Schwelle
            short_score = (
                price_factor * 0.3 +          # Preis-Komponente
                divergence_factor * 0.4 +     # Divergenz-Stärke
                trend_factor * 0.3            # CVD-Trend-Stärke
            )
            if cvd_divergence > 0 or price_component < 0:  # Zumindest eine Bedingung
                raw_score = short_score
                logger.info(f"🔴 SHORT SQUEEZE: price={price_component:.3f}, div={cvd_divergence:.3f}, score={raw_score:.3f}")
            else:
                raw_score = 0.0
                
        # SCHWACHE SIGNALE: Auch bei kleinen Bewegungen
        elif abs(cvd_divergence) > 0.02 or abs(price_component) > 0.005:  # Sehr niedrige Schwelle
            weak_strength = max(abs(cvd_divergence), abs(price_component)) * 0.1
            if cvd_divergence < 0 or price_component > 0:
                raw_score = -weak_strength
                logger.info(f"🟡 WEAK LONG: price={price_component:.3f}, div={cvd_divergence:.3f}, score={raw_score:.3f}")
            else:
                raw_score = weak_strength
                logger.info(f"🟡 WEAK SHORT: price={price_component:.3f}, div={cvd_divergence:.3f}, score={raw_score:.3f}")
        else:
            raw_score = 0.0
            logger.debug(f"⚪ NO SQUEEZE: price={price_component:.3f}, div={cvd_divergence:.3f}")
            
        # Apply smoothing
        if hasattr(self, '_score_history'):
            self._score_history.append(raw_score)
            if len(self._score_history) > self.smoothing_period:
                self._score_history.pop(0)
            smoothed_score = np.mean(self._score_history)
        else:
            self._score_history = [raw_score]
            smoothed_score = raw_score
            
        # Prepare detailed results
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
            'timestamp': datetime.now()
        }
        
        return result
        
    def _classify_signal(self, score: float) -> str:
        """Classify the signal based on score (adjusted for cumulative CVD)"""
        if score <= -0.4:
            return "STRONG_LONG_SQUEEZE"
        elif score <= -0.2:
            return "LONG_SQUEEZE"
        elif score >= 0.4:
            return "STRONG_SHORT_SQUEEZE"
        elif score >= 0.2:
            return "SHORT_SQUEEZE"
        else:
            return "NEUTRAL"
            
    def should_enter_position(self, score_data: Dict[str, float], 
                            min_score_threshold: float = 0.3) -> Tuple[bool, str]:
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