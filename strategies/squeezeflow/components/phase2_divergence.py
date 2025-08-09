"""
Phase 2: Divergence Detection Component

Identifies price-CVD imbalances that create trading opportunities.
This phase provides market intelligence without scoring points.

Based on SqueezeFlow.md lines 88-98
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import os


class DivergenceDetection:
    """
    Phase 2: Divergence Detection
    
    Objective: Identify price-CVD imbalances that create trading opportunities
    Critical Insight: Looking for moments when "price is unbalanced relative to what CVD is saying"
    """
    
    def __init__(self, divergence_timeframes: List[str] = None):
        """
        Initialize divergence detection component
        
        Args:
            divergence_timeframes: Timeframes to analyze (1s-aware)
        """
        # 1s mode awareness
        self.enable_1s_mode = os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true'
        
        if divergence_timeframes is None:
            if self.enable_1s_mode:
                self.divergence_timeframes = ["5m", "15m"]  # Shorter for 1s mode
            else:
                self.divergence_timeframes = ["15m", "30m"]  # Original
        else:
            self.divergence_timeframes = divergence_timeframes
            
        # Log 1s mode status
        if self.enable_1s_mode:
            print(f"Phase 2 Divergence: 1s mode enabled, using timeframes: {self.divergence_timeframes}")
        
    def detect_divergence(self, dataset: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify CVD leadership patterns between spot and futures
        
        Pattern Recognition:
        - Long Setup: Price stable/rising + Spot CVD up + Perp CVD down (shorts accumulating, price not falling)
        - Short Setup: Price stable/falling + Spot CVD down + Perp CVD up (longs accumulating, price not rising)
        - Volume Pattern: Look for divergences notably larger than recent activity
        
        Args:
            dataset: Market data including spot_cvd, futures_cvd, ohlcv
            context: Results from Phase 1 context assessment
            
        Returns:
            Dict containing divergence analysis results
        """
        try:
            # Extract data
            spot_cvd = dataset.get('spot_cvd', pd.Series())
            futures_cvd = dataset.get('futures_cvd', pd.Series())
            cvd_divergence = dataset.get('cvd_divergence', pd.Series())
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            
            if spot_cvd.empty or futures_cvd.empty or ohlcv.empty:
                return self._empty_divergence()
            
            # Calculate price movement patterns
            price_pattern = self._analyze_price_movement(ohlcv)
            
            # Detect CVD leadership patterns
            cvd_patterns = self._detect_cvd_patterns(spot_cvd, futures_cvd, cvd_divergence)
            
            # Analyze volume significance
            volume_significance = self._analyze_volume_significance(cvd_divergence)
            
            # Identify setup type based on patterns
            setup_type = self._identify_setup_type(price_pattern, cvd_patterns, context)
            
            # Detect if divergence is significant
            is_significant = self._is_divergence_significant(volume_significance, cvd_patterns)
            
            return {
                'phase': 'DIVERGENCE_DETECTION',
                'setup_type': setup_type,
                'price_pattern': price_pattern,
                'cvd_patterns': cvd_patterns,
                'volume_significance': volume_significance,
                'is_significant': is_significant,
                'market_imbalance': self._assess_market_imbalance(cvd_patterns),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'phase': 'DIVERGENCE_DETECTION',
                'error': f'Divergence detection error: {str(e)}',
                'setup_type': 'NONE',
                'is_significant': False
            }
    
    def _analyze_price_movement(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if price is stable, rising, or falling"""
        
        if len(ohlcv) < 20:
            return {'movement': 'UNKNOWN', 'stability': 0, 'trend': 0}
        
        # Get close prices
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        closes = ohlcv[close_col]
        
        # Recent price change (last 10 periods)
        if len(closes) >= 10 and closes.iloc[-10] != 0:
            recent_change = (closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10]
        else:
            recent_change = 0
        
        # Price volatility (stability measure)
        price_std = closes.iloc[-20:].pct_change(fill_method=None).std()
        stability = 1 - min(price_std * 10, 1)  # Higher stability = lower volatility
        
        # Determine movement type
        if abs(recent_change) < 0.01:  # Less than 1% change
            movement = 'STABLE'
        elif recent_change > 0.01:
            movement = 'RISING'
        else:
            movement = 'FALLING'
            
        return {
            'movement': movement,
            'stability': stability,
            'trend': recent_change,
            'volatility': price_std
        }
    
    def _detect_cvd_patterns(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                           cvd_divergence: pd.Series) -> Dict[str, Any]:
        """Detect CVD leadership patterns"""
        
        if len(spot_cvd) < 10:
            return {'pattern': 'INSUFFICIENT_DATA', 'spot_direction': 0, 'futures_direction': 0}
        
        # Recent CVD movements with 1s mode adjustment
        base_lookback = 10 if not self.enable_1s_mode else 600  # 10min in 1s mode
        lookback = min(base_lookback, len(spot_cvd) // 2)
        
        if len(spot_cvd) < lookback or len(futures_cvd) < lookback or lookback < 1:
            return {'pattern': 'INSUFFICIENT_DATA', 'spot_direction': 0, 'futures_direction': 0}
        
        spot_change = spot_cvd.iloc[-1] - spot_cvd.iloc[-lookback]
        futures_change = futures_cvd.iloc[-1] - futures_cvd.iloc[-lookback]
        
        # Direction analysis
        spot_direction = np.sign(spot_change)
        futures_direction = np.sign(futures_change)
        
        # Enhanced pattern identification with magnitude consideration
        spot_magnitude = abs(spot_change)
        futures_magnitude = abs(futures_change)
        
        if spot_direction > 0 and futures_direction < 0:
            pattern = 'SPOT_LEADING_UP'  # Potential long squeeze setup
        elif spot_direction < 0 and futures_direction > 0:
            pattern = 'FUTURES_LEADING_UP'  # Potential short squeeze setup
        elif spot_direction > 0 and futures_direction > 0:
            pattern = 'BOTH_UP'
        elif spot_direction < 0 and futures_direction < 0:
            pattern = 'BOTH_DOWN'
        elif spot_magnitude > futures_magnitude * 2 and spot_direction != 0:
            # Spot dominant pattern (even if both same direction)
            pattern = 'SPOT_LEADING_UP' if spot_direction > 0 else 'SPOT_LEADING_DOWN'
        elif futures_magnitude > spot_magnitude * 2 and futures_direction != 0:
            # Futures dominant pattern
            pattern = 'FUTURES_LEADING_UP' if futures_direction > 0 else 'FUTURES_LEADING_DOWN'
        else:
            pattern = 'NEUTRAL'
            
        # Calculate divergence strength
        divergence_strength = abs(spot_change - futures_change)
        
        return {
            'pattern': pattern,
            'spot_direction': spot_direction,
            'futures_direction': futures_direction,
            'spot_change': spot_change,
            'futures_change': futures_change,
            'divergence_strength': divergence_strength,
            'recent_divergence': cvd_divergence.iloc[-1] if not cvd_divergence.empty else 0
        }
    
    def _analyze_volume_significance(self, cvd_divergence: pd.Series) -> Dict[str, Any]:
        """
        Analyze if current divergence is significantly larger than recent activity
        
        From SqueezeFlow.md: "if recent swings were 200M, a 400M+ divergence is significant"
        """
        
        # Adjust lookback for 1s mode
        volume_lookback = 20 if not self.enable_1s_mode else 1200  # 20min in 1s mode
        
        if len(cvd_divergence) < volume_lookback:
            return {'is_significant': False, 'multiplier': 0, 'recent_avg': 0}
        
        # Recent divergence swings (absolute values)
        recent_swings = cvd_divergence.iloc[-volume_lookback:-1].abs()
        recent_avg = recent_swings.mean()
        recent_max = recent_swings.max()
        
        # Current divergence
        current_divergence = abs(cvd_divergence.iloc[-1])
        
        # Calculate significance multiplier
        if recent_avg > 0:
            multiplier = current_divergence / recent_avg
        else:
            multiplier = 0
            
        # Significant if current is 2x+ recent average or larger than recent max
        is_significant = multiplier >= 2.0 or current_divergence > recent_max * 1.5
        
        return {
            'is_significant': is_significant,
            'multiplier': multiplier,
            'recent_avg': recent_avg,
            'recent_max': recent_max,
            'current': current_divergence
        }
    
    def _identify_setup_type(self, price_pattern: Dict[str, Any], 
                           cvd_patterns: Dict[str, Any], 
                           context: Dict[str, Any]) -> str:
        """
        Identify trading setup type based on patterns
        
        Enhanced logic with multiple fallback scenarios:
        1. Primary: Exact CVD divergence patterns
        2. Secondary: CVD direction alignment patterns  
        3. Tertiary: Market context-based patterns
        """
        
        movement = price_pattern['movement']
        cvd_pattern = cvd_patterns['pattern']
        market_bias = context.get('market_bias', 'NEUTRAL')
        spot_direction = cvd_patterns.get('spot_direction', 0)
        futures_direction = cvd_patterns.get('futures_direction', 0)
        
        # PRIMARY: Exact CVD divergence patterns (existing logic)
        if (movement in ['STABLE', 'RISING'] and 
            cvd_pattern == 'SPOT_LEADING_UP' and
            market_bias != 'BEARISH'):
            return 'LONG_SETUP'
            
        elif (movement in ['STABLE', 'FALLING'] and 
              cvd_pattern == 'FUTURES_LEADING_UP' and
              market_bias != 'BULLISH'):
            return 'SHORT_SETUP'
        
        # SECONDARY: CVD alignment patterns (less restrictive)
        elif (spot_direction > 0 and movement != 'FALLING' and market_bias != 'BEARISH'):
            # Spot CVD positive + non-falling price + not bearish context
            return 'LONG_SETUP'
            
        elif (spot_direction < 0 and movement != 'RISING' and market_bias != 'BULLISH'):
            # Spot CVD negative + non-rising price + not bullish context
            return 'SHORT_SETUP'
            
        # TERTIARY: Market context patterns (when CVD is mixed)
        elif market_bias == 'BULLISH' and movement in ['STABLE', 'RISING']:
            return 'LONG_SETUP'
            
        elif market_bias == 'BEARISH' and movement in ['STABLE', 'FALLING']:
            return 'SHORT_SETUP'
            
        # QUATERNARY: CVD momentum patterns (very permissive)
        elif (abs(spot_direction) > abs(futures_direction) and spot_direction > 0):
            # Spot CVD dominance upward
            return 'LONG_SETUP'
            
        elif (abs(futures_direction) > abs(spot_direction) and futures_direction < 0):
            # Futures CVD dominance downward (short squeeze potential)
            return 'SHORT_SETUP'
        
        else:
            return 'NO_SETUP'
    
    def _is_divergence_significant(self, volume_significance: Dict[str, Any], 
                                  cvd_patterns: Dict[str, Any]) -> bool:
        """Determine if divergence is significant enough to trade"""
        
        # Must have significant volume
        if not volume_significance['is_significant']:
            return False
            
        # Must have clear pattern
        if cvd_patterns['pattern'] in ['NEUTRAL', 'INSUFFICIENT_DATA']:
            return False
            
        # Must have meaningful divergence strength
        if cvd_patterns['divergence_strength'] < volume_significance['recent_avg'] * 0.5:
            return False
            
        return True
    
    def _assess_market_imbalance(self, cvd_patterns: Dict[str, Any]) -> str:
        """Assess the degree of market imbalance"""
        
        pattern = cvd_patterns['pattern']
        strength = cvd_patterns['divergence_strength']
        
        if pattern in ['SPOT_LEADING_UP', 'FUTURES_LEADING_UP']:
            if strength > cvd_patterns.get('recent_avg', 1) * 3:
                return 'EXTREME_IMBALANCE'
            elif strength > cvd_patterns.get('recent_avg', 1) * 2:
                return 'HIGH_IMBALANCE'
            else:
                return 'MODERATE_IMBALANCE'
        else:
            return 'BALANCED'
    
    def _empty_divergence(self) -> Dict[str, Any]:
        """Return empty divergence when data is insufficient"""
        return {
            'phase': 'DIVERGENCE_DETECTION',
            'setup_type': 'NO_SETUP',
            'price_pattern': {'movement': 'UNKNOWN', 'stability': 0, 'trend': 0},
            'cvd_patterns': {'pattern': 'INSUFFICIENT_DATA'},
            'volume_significance': {'is_significant': False},
            'is_significant': False,
            'market_imbalance': 'UNKNOWN',
            'error': 'Insufficient data for divergence detection'
        }