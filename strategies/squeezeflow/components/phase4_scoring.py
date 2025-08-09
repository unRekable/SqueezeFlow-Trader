"""
Phase 4: 10-Point Scoring System Component

Consolidates all intelligence from Phases 1-3 into a scored decision.
This is where ALL trading decisions are made based on a 0-10 point system.

Based on SqueezeFlow.md lines 135-188
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import pytz
import os


class ScoringSystem:
    """
    Phase 4: Scoring Decision & Entry Signal
    
    Objective: Consolidate all market intelligence into a single scored decision
    Total Score: 0-10 points determines trade quality and confidence
    
    Scoring thresholds:
    - 8-10 points: Premium quality signals
    - 6-7 points: High quality signals  
    - 4-5 points: Medium quality signals
    - 0-3 points: No signal (insufficient conditions)
    """
    
    def __init__(self, scoring_weights: Dict[str, float] = None, min_entry_score: float = None):
        """
        Initialize scoring system with configurable weights and threshold
        
        Args:
            scoring_weights: Custom weights for scoring criteria
            min_entry_score: Minimum score required to trade
        """
        # 1s mode awareness
        self.enable_1s_mode = os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true'
        
        self.scoring_weights = scoring_weights or {
            "cvd_reset_deceleration": 3.5,     # Critical
            "absorption_candle": 2.5,          # High priority
            "failed_breakdown": 2.0,           # Medium priority
            "directional_bias": 2.0            # Supporting
        }
        
        self.min_entry_score = min_entry_score if min_entry_score is not None else 1.5
        
        # Log 1s mode status
        if self.enable_1s_mode:
            print(f"Phase 4 Scoring: 1s mode enabled, statistical adjustment applied")
        
    def calculate_score(self, context: Dict[str, Any], divergence: Dict[str, Any], 
                       reset: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate 10-point score based on all phase intelligence
        
        Four scored criteria:
        1. CVD Reset Deceleration (3.5 points) - validates reset quality
        2. Absorption Candle Confirmation (2.5 points) - price action validation
        3. Failed Breakdown Pattern (2.0 points) - pattern strength
        4. Directional Bias Confirmation (2.0 points) - new trend direction
        
        Args:
            context: Phase 1 context assessment results
            divergence: Phase 2 divergence detection results
            reset: Phase 3 reset detection results
            dataset: Market data for additional analysis
            
        Returns:
            Dict containing score breakdown and trading decision
        """
        try:
            # Extract market data
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            spot_cvd = dataset.get('spot_cvd', pd.Series())
            futures_cvd = dataset.get('futures_cvd', pd.Series())
            
            if ohlcv.empty or spot_cvd.empty:
                return self._empty_score()
            
            # Initialize score components
            scores = {
                'cvd_reset_deceleration': 0,
                'absorption_candle': 0,
                'failed_breakdown': 0,
                'directional_bias': 0
            }
            
            # Score components (modified to be less dependent on strict reset detection)
            reset_detected = reset.get('reset_detected', False)
            
            # 1. CVD Reset Deceleration (3.5 points)
            # Allow partial scoring even without strict reset detection
            scores['cvd_reset_deceleration'] = self._score_cvd_deceleration(
                spot_cvd, futures_cvd, reset
            )
            
            # 2. Absorption Candle Confirmation (2.5 points)
            # This can be scored independently of reset detection
            scores['absorption_candle'] = self._score_absorption_candles(
                ohlcv, context, divergence
            )
            
            # 3. Failed Breakdown Pattern (2.0 points)
            # This can be scored independently of reset detection
            scores['failed_breakdown'] = self._score_failed_breakdown(
                ohlcv, spot_cvd, futures_cvd
            )
            
            # 4. Directional Bias Confirmation (2.0 points)
            # This can be scored independently of reset detection
            scores['directional_bias'] = self._score_directional_bias(
                ohlcv, spot_cvd, futures_cvd, context
            )
            
            # Apply reset bonus: if reset detected, multiply total score by 1.5
            if reset_detected:
                # Award bonus for having proper reset confirmation
                for key in scores:
                    scores[key] *= 1.2  # 20% bonus for confirmed reset
            
            # Calculate total score
            total_score = sum(scores.values())
            
            # Determine signal type and quality
            signal_info = self._determine_signal_quality(total_score, context, divergence)
            
            # Determine if we should trade (UPDATED: Use dynamic threshold from config)
            should_trade = total_score >= self.min_entry_score
            
            return {
                'phase': 'SCORING_DECISION',
                'total_score': total_score,
                'score_breakdown': scores,
                'signal_quality': signal_info['quality'],
                'signal_type': signal_info['type'],
                'confidence': signal_info['confidence'],
                'should_trade': should_trade,
                'direction': signal_info['direction'],
                'reasoning': self._generate_reasoning(scores, reset),
                'timestamp': datetime.now(tz=pytz.UTC),
                # Include reset and divergence data for exit logic
                'reset': reset,
                'divergence': divergence,
                'context': context
            }
            
        except Exception as e:
            return {
                'phase': 'SCORING_DECISION',
                'error': f'Scoring error: {str(e)}',
                'total_score': 0,
                'should_trade': False,
                # Empty structures for consistency
                'reset': {},
                'divergence': {},
                'context': {}
            }
    
    def _score_cvd_deceleration(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                               reset: Dict[str, Any]) -> float:
        """
        Score CVD Reset Deceleration (3.5 points max)
        
        Criteria:
        - CVD stops moving as much during reset attempts
        - Second test shows less CVD movement than first
        - Both CVD and price showing "compression"
        """
        
        score = 0.0
        max_score = self.scoring_weights['cvd_reset_deceleration']
        
        # Check if convergence was detected (more lenient approach)
        convergence = reset.get('convergence', {})
        convergence_strength = convergence.get('strength', 0)
        
        # Score based on convergence strength (tightened for 1s data)
        if convergence_strength > 0.6:  # Strong convergence (was 0.5)
            score += max_score * 0.4  # Reduced multiplier
        elif convergence_strength > 0.4:  # Moderate convergence (was 0.3)
            score += max_score * 0.25  # Reduced multiplier
        elif convergence_strength > 0.25:  # Weak convergence (was 0.1)
            score += max_score * 0.15  # Reduced multiplier
        elif len(spot_cvd) >= 10:
            # Even without convergence, look for CVD relationship patterns
            spot_recent_change = abs(spot_cvd.iloc[-5:].pct_change(fill_method=None).mean())
            futures_recent_change = abs(futures_cvd.iloc[-5:].pct_change(fill_method=None).mean())
            
            # If both CVDs are showing similar patterns, award some points (stricter for 1s)
            if abs(spot_recent_change - futures_recent_change) < max(spot_recent_change, futures_recent_change) * 0.3:
                score += max_score * 0.05  # Smaller reward for CVD alignment
            
        # Check for deceleration pattern
        if len(spot_cvd) >= 20:
            # Compare recent vs earlier CVD movement
            recent_movement = abs(spot_cvd.iloc[-5:].pct_change(fill_method=None).mean())
            earlier_movement = abs(spot_cvd.iloc[-20:-10].pct_change(fill_method=None).mean())
            
            if earlier_movement > 0.001 and recent_movement < earlier_movement * 0.4:  # Stricter thresholds
                # Strong deceleration detected
                score += max_score * 0.25  # Reduced multiplier
            elif earlier_movement > 0.0005 and recent_movement < earlier_movement * 0.6:  # Stricter
                # Moderate deceleration
                score += max_score * 0.15  # Reduced multiplier
                
        # Check momentum exhaustion from reset
        if reset.get('momentum_exhaustion', {}).get('exhausted', False):
            score += max_score * 0.2
            
        return min(score, max_score)
    
    def _score_absorption_candles(self, ohlcv: pd.DataFrame, context: Dict[str, Any], 
                                divergence: Dict[str, Any]) -> float:
        """
        Score Absorption Candle Confirmation (2.5 points max)
        
        Criteria:
        - Candles that don't close under reset low
        - Close higher than open (buyers stepping in)
        - Wicks below but close above (selling absorbed)
        - Volume confirmation on wick rejection
        """
        
        score = 0.0
        max_score = self.scoring_weights['absorption_candle']
        
        if len(ohlcv) < 5:
            return score
            
        # Get OHLC columns
        open_col = 'open' if 'open' in ohlcv.columns else ohlcv.columns[0]
        high_col = 'high' if 'high' in ohlcv.columns else ohlcv.columns[1]
        low_col = 'low' if 'low' in ohlcv.columns else ohlcv.columns[2]
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        volume_col = 'volume' if 'volume' in ohlcv.columns else ohlcv.columns[4]
        
        # Analyze recent candles for absorption patterns
        recent_candles = ohlcv.iloc[-5:]
        
        # Find recent low (potential reset low)
        reset_low = recent_candles[low_col].min()
        
        # Count absorption patterns
        absorption_count = 0
        for idx in range(len(recent_candles)):
            candle = recent_candles.iloc[idx]
            
            # Check for buyers stepping in (require meaningful move for 1s data)
            if candle[close_col] > candle[open_col] * 1.0002:  # 0.02% minimum move
                absorption_count += 1
                score += max_score * 0.1  # Reduced multiplier
                
            # Check for wick rejection (wick below but close above)
            wick_size = (candle[close_col] - candle[low_col]) / (candle[high_col] - candle[low_col] + 0.0001)
            if wick_size > 0.6 and candle[close_col] > reset_low:
                score += max_score * 0.2
                
        # Volume confirmation
        if len(ohlcv) >= 20:
            recent_volume = recent_candles[volume_col].mean()
            avg_volume = ohlcv[volume_col].iloc[-20:].mean()
            
            if recent_volume > avg_volume * 1.5:  # High volume on absorption
                score += max_score * 0.15
                
        return min(score, max_score)
    
    def _score_failed_breakdown(self, ohlcv: pd.DataFrame, spot_cvd: pd.Series, 
                              futures_cvd: pd.Series) -> float:
        """
        Score Failed Breakdown Pattern (2.0 points max)
        
        Criteria:
        - Multiple candles try to break range but fail
        - Each attempt shows decreasing CVD movement
        - Market equilibrium forming after chaos
        """
        
        score = 0.0
        max_score = self.scoring_weights['failed_breakdown']
        
        if len(ohlcv) < 10:
            return score
            
        # Get price columns
        high_col = 'high' if 'high' in ohlcv.columns else ohlcv.columns[1]
        low_col = 'low' if 'low' in ohlcv.columns else ohlcv.columns[2]
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        
        # Define range from recent price action
        recent_prices = ohlcv.iloc[-10:]
        range_high = recent_prices[high_col].max()
        range_low = recent_prices[low_col].min()
        range_size = range_high - range_low
        
        if range_size <= 0:
            return score
            
        # Count failed breakout attempts
        failed_breaks = 0
        cvd_decreasing = False
        
        for i in range(5, len(recent_prices)):
            candle = recent_prices.iloc[i]
            
            # Check for failed break attempts (wicks outside range)
            if candle[high_col] > range_high and candle[close_col] < range_high:
                failed_breaks += 1
            if candle[low_col] < range_low and candle[close_col] > range_low:
                failed_breaks += 1
                
        # Score based on number of failed attempts
        if failed_breaks >= 3:
            score += max_score * 0.6
        elif failed_breaks >= 2:
            score += max_score * 0.4
        elif failed_breaks >= 1:
            score += max_score * 0.2
            
        # Check for decreasing CVD movement on attempts
        if len(spot_cvd) >= 10:
            recent_cvd_volatility = spot_cvd.iloc[-5:].std()
            earlier_cvd_volatility = spot_cvd.iloc[-10:-5].std()
            
            if earlier_cvd_volatility > 0 and recent_cvd_volatility < earlier_cvd_volatility * 0.7:
                cvd_decreasing = True
                score += max_score * 0.4
                
        return min(score, max_score)
    
    def _score_directional_bias(self, ohlcv: pd.DataFrame, spot_cvd: pd.Series, 
                              futures_cvd: pd.Series, context: Dict[str, Any]) -> float:
        """
        Score Directional Bias Confirmation (2.0 points max)
        
        Criteria:
        - CVD must start going up "a bit"
        - Price must follow CVD direction
        - Both CVDs moving in same direction acceptable
        """
        
        score = 0.0
        max_score = self.scoring_weights['directional_bias']
        
        if len(spot_cvd) < 5 or len(ohlcv) < 5:
            return score
            
        # Get price column
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        
        # Check recent CVD movement
        spot_recent = spot_cvd.iloc[-5:].pct_change(fill_method=None).mean()
        futures_recent = futures_cvd.iloc[-5:].pct_change(fill_method=None).mean()
        
        # Handle NaN values
        if pd.isna(spot_recent):
            spot_recent = 0
        if pd.isna(futures_recent):
            futures_recent = 0
            
        # Check if both CVDs moving in same direction (much stricter for 1s data)
        cvd_aligned = np.sign(spot_recent) == np.sign(futures_recent)
        if cvd_aligned and abs(spot_recent) > 0.002:  # 20x stricter threshold
            score += max_score * 0.4  # Reduced multiplier
        elif abs(spot_recent) > 0.001:  # 10x stricter threshold
            score += max_score * 0.2  # Reduced multiplier
            
        # Check if price follows CVD
        price_recent = ohlcv[close_col].iloc[-5:].pct_change(fill_method=None).mean()
        if pd.isna(price_recent):
            price_recent = 0
            
        price_follows_cvd = np.sign(price_recent) == np.sign(spot_recent) if spot_recent != 0 else False
        
        if price_follows_cvd and abs(price_recent) > 0.001:  # 10x stricter
            score += max_score * 0.25  # Reduced multiplier
        elif abs(price_recent) > 0.0005:  # 5x stricter
            score += max_score * 0.05  # Much smaller reward
            
        # Consider market context (always award some points for context alignment)
        market_bias = context.get('market_bias', 'NEUTRAL')
        if market_bias == 'BULLISH':
            score += max_score * 0.2
        elif market_bias == 'BEARISH':
            score += max_score * 0.2
        else:  # NEUTRAL
            score += max_score * 0.1  # Small reward for having context
            
        return min(score, max_score)
    
    def _determine_signal_quality(self, total_score: float, context: Dict[str, Any], 
                                divergence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine signal quality and type based on total score
        
        Enhanced direction logic with multiple fallback methods:
        1. Phase 2 setup type (primary)
        2. Market context bias (secondary) 
        3. CVD pattern analysis (tertiary)
        4. Score-based inference (quaternary)
        """
        
        # Determine quality tier
        if total_score >= 8:
            quality = 'PREMIUM'
            confidence = 0.9
        elif total_score >= 6:
            quality = 'HIGH'
            confidence = 0.75
        elif total_score >= 4:
            quality = 'MEDIUM'
            confidence = 0.6
        else:
            quality = 'INSUFFICIENT'
            confidence = 0
            
        # PRIMARY: Phase 2 setup type
        setup_type = divergence.get('setup_type', 'NO_SETUP')
        market_bias = context.get('market_bias', 'NEUTRAL')
        
        if setup_type == 'LONG_SETUP':
            signal_type = 'LONG_SQUEEZE'
            direction = 'LONG'
        elif setup_type == 'SHORT_SETUP':
            signal_type = 'SHORT_SQUEEZE'
            direction = 'SHORT'
        
        # SECONDARY: Market context bias (if setup_type failed)
        elif market_bias == 'BULLISH' and total_score >= self.min_entry_score:
            signal_type = 'LONG_SQUEEZE'
            direction = 'LONG'
        elif market_bias == 'BEARISH' and total_score >= self.min_entry_score:
            signal_type = 'SHORT_SQUEEZE'
            direction = 'SHORT'
        
        # TERTIARY: CVD pattern analysis
        elif total_score >= self.min_entry_score:
            cvd_patterns = divergence.get('cvd_patterns', {})
            pattern = cvd_patterns.get('pattern', 'NEUTRAL')
            spot_direction = cvd_patterns.get('spot_direction', 0)
            futures_direction = cvd_patterns.get('futures_direction', 0)
            
            # Based on strategy doc:
            # LONG: Spot CVD up (buying pressure from spot)
            # SHORT: Futures CVD up (perp longs accumulating)
            if pattern == 'SPOT_LEADING_UP' or (pattern == 'BOTH_UP' and spot_direction > futures_direction):
                signal_type = 'LONG_SQUEEZE'
                direction = 'LONG'
            elif pattern == 'FUTURES_LEADING_UP' or (pattern == 'BOTH_DOWN' and futures_direction < spot_direction):
                signal_type = 'SHORT_SQUEEZE' 
                direction = 'SHORT'
            elif spot_direction > 0 and spot_direction > abs(futures_direction):
                # Spot buying dominates
                signal_type = 'LONG_SQUEEZE'
                direction = 'LONG'
            elif futures_direction > 0 and futures_direction > abs(spot_direction):
                # Futures buying dominates (shorts will be squeezed)
                signal_type = 'SHORT_SQUEEZE'
                direction = 'SHORT'
            else:
                # QUATERNARY: Default to LONG for positive scores (common in bull markets)
                signal_type = 'LONG_SQUEEZE'
                direction = 'LONG'
        
        else:
            signal_type = 'NEUTRAL'
            direction = 'NONE'
            
        return {
            'quality': quality,
            'confidence': confidence,
            'type': signal_type,
            'direction': direction
        }
    
    def _generate_reasoning(self, scores: Dict[str, float], reset: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the score"""
        
        reasons = []
        
        if scores['cvd_reset_deceleration'] > 0:
            reasons.append(f"CVD deceleration detected ({scores['cvd_reset_deceleration']:.1f} pts)")
            
        if scores['absorption_candle'] > 0:
            reasons.append(f"Absorption candles present ({scores['absorption_candle']:.1f} pts)")
            
        if scores['failed_breakdown'] > 0:
            reasons.append(f"Failed breakdown pattern ({scores['failed_breakdown']:.1f} pts)")
            
        if scores['directional_bias'] > 0:
            reasons.append(f"Directional bias confirmed ({scores['directional_bias']:.1f} pts)")
            
        if not reasons:
            if not reset.get('reset_detected', False):
                return "No reset detected - waiting for better conditions"
            else:
                return "Reset detected but scoring criteria not met"
                
        return " | ".join(reasons)
    
    def _empty_score(self) -> Dict[str, Any]:
        """Return empty score when data is insufficient"""
        return {
            'phase': 'SCORING_DECISION',
            'total_score': 0,
            'score_breakdown': {
                'cvd_reset_deceleration': 0,
                'absorption_candle': 0,
                'failed_breakdown': 0,
                'directional_bias': 0
            },
            'signal_quality': 'INSUFFICIENT',
            'should_trade': False,
            'direction': 'NONE',
            'error': 'Insufficient data for scoring',
            # Empty structures for consistency
            'reset': {},
            'divergence': {},
            'context': {}
        }