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
import sys

# Import optimized statistics functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.statistics import (
    rolling_divergence_analysis,
    vectorized_momentum_analysis,
    adaptive_significance_threshold,
    set_1s_mode
)


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
        
        # Configure statistics processor for 1s mode
        set_1s_mode(self.enable_1s_mode)
        
        # Performance optimization: pre-calculate parameters
        self.density_factor = 60 if self.enable_1s_mode else 1
        self.pattern_lookback = 600 if self.enable_1s_mode else 10    # 10 minutes equivalent
        self.volume_lookback = 1200 if self.enable_1s_mode else 20    # 20 minutes equivalent
        self.price_stability_periods = 600 if self.enable_1s_mode else 10  # 10 minutes equivalent
        
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
        """Analyze if price is stable, rising, or falling (optimized for 1s density)"""
        
        if ohlcv.empty:
            return {'movement': 'UNKNOWN', 'stability': 0, 'trend': 0}
        
        # Use vectorized momentum analysis
        momentum_analysis = vectorized_momentum_analysis(ohlcv, periods=[self.price_stability_periods // self.density_factor])
        
        recent_change = momentum_analysis.get('momentum_score', 0)
        
        # Use adaptive thresholds for movement classification
        stable_threshold = adaptive_significance_threshold(pd.Series([recent_change]), 0.01)
        rising_threshold = stable_threshold
        
        # Determine movement type with adaptive thresholds
        if abs(recent_change) < stable_threshold:
            movement = 'STABLE'
        elif recent_change > rising_threshold:
            movement = 'RISING'
        else:
            movement = 'FALLING'
        
        # Calculate stability from momentum analysis
        stability = 1.0 - min(recent_change * 5, 1.0) if recent_change > 0 else 1.0
        
        return {
            'movement': movement,
            'stability': max(0.0, stability),
            'trend': recent_change,
            'exhausted': momentum_analysis.get('exhausted', False),
            'momentum_trend': momentum_analysis.get('trend', 'UNKNOWN')
        }
    
    def _detect_cvd_patterns(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                           cvd_divergence: pd.Series) -> Dict[str, Any]:
        """Detect CVD leadership patterns (optimized for 1s density)"""
        
        if spot_cvd.empty or futures_cvd.empty:
            return {'pattern': 'INSUFFICIENT_DATA', 'spot_direction': 0, 'futures_direction': 0}
        
        # Use optimized rolling divergence analysis
        divergence_analysis = rolling_divergence_analysis(
            spot_cvd, futures_cvd, window_size=self.pattern_lookback
        )
        
        if divergence_analysis['pattern'] == 'INSUFFICIENT_DATA':
            return {
                'pattern': 'INSUFFICIENT_DATA',
                'spot_direction': 0,
                'futures_direction': 0,
                'divergence_strength': 0,
                'recent_divergence': 0
            }
        
        # Extract pattern information
        pattern = divergence_analysis['pattern']
        current_divergence = divergence_analysis['current_divergence']
        z_score = divergence_analysis['z_score']
        
        # Calculate direction indicators
        spot_direction = 1 if current_divergence > 0 else -1 if current_divergence < 0 else 0
        futures_direction = -spot_direction  # Opposite by definition of divergence
        
        # Use lookback for change calculations with safety checks
        lookback = min(self.pattern_lookback, len(spot_cvd) - 1)
        if lookback < 1:
            lookback = 1
        
        spot_change = spot_cvd.iloc[-1] - spot_cvd.iloc[-lookback-1] if len(spot_cvd) > lookback else 0
        futures_change = futures_cvd.iloc[-1] - futures_cvd.iloc[-lookback-1] if len(futures_cvd) > lookback else 0
        
        return {
            'pattern': pattern,
            'spot_direction': spot_direction,
            'futures_direction': futures_direction,
            'spot_change': spot_change,
            'futures_change': futures_change,
            'divergence_strength': abs(current_divergence),
            'recent_divergence': current_divergence,
            'significance': divergence_analysis['significance'],
            'z_score': z_score
        }
    
    def _analyze_volume_significance(self, cvd_divergence: pd.Series) -> Dict[str, Any]:
        """
        Analyze if current divergence is significantly larger than recent activity
        (optimized vectorized version)
        
        From SqueezeFlow.md: "if recent swings were 200M, a 400M+ divergence is significant"
        """
        
        if cvd_divergence.empty:
            return {'is_significant': False, 'multiplier': 0, 'recent_avg': 0}
        
        # Use pre-calculated lookback
        volume_lookback = min(self.volume_lookback, len(cvd_divergence) - 1)
        if volume_lookback < 1:
            return {'is_significant': False, 'multiplier': 0, 'recent_avg': 0}
        
        # Vectorized calculations using numpy
        recent_values = cvd_divergence.iloc[-volume_lookback-1:-1].values
        current_divergence = abs(cvd_divergence.iloc[-1])
        
        if len(recent_values) == 0:
            return {'is_significant': False, 'multiplier': 0, 'recent_avg': 0}
        
        # Vectorized statistical calculations
        recent_abs = np.abs(recent_values)
        recent_avg = np.mean(recent_abs)
        recent_max = np.max(recent_abs)
        recent_std = np.std(recent_abs)
        
        # Calculate significance using multiple methods
        if recent_avg > 0:
            multiplier = current_divergence / recent_avg
        else:
            multiplier = 0
        
        # Z-score based significance (more robust for 1s data)
        if recent_std > 0:
            z_score = (current_divergence - recent_avg) / recent_std
        else:
            z_score = 0
        
        # Adaptive significance threshold
        base_threshold = adaptive_significance_threshold(cvd_divergence, 2.0)
        
        # Multiple significance criteria
        is_significant = (
            multiplier >= base_threshold or 
            current_divergence > recent_max * 1.5 or
            abs(z_score) > 2.0  # Statistical significance
        )
        
        return {
            'is_significant': is_significant,
            'multiplier': multiplier,
            'recent_avg': recent_avg,
            'recent_max': recent_max,
            'current': current_divergence,
            'z_score': z_score,
            'threshold_used': base_threshold
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