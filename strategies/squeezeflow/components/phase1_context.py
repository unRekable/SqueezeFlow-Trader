"""
Phase 1: Context Assessment Component

Analyzes larger timeframes (30m, 1h, 4h) to determine the dominant squeeze environment.
This phase provides market intelligence without scoring points.

Based on SqueezeFlow.md lines 76-87
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
    vectorized_trend_analysis,
    vectorized_momentum_analysis,
    set_1s_mode,
    adaptive_significance_threshold
)


class ContextAssessment:
    """
    Phase 1: Larger Context Assessment ("Zoom Out")
    
    Objective: Determine the dominant squeeze environment
    Key Question: "Is this a short squeeze environment or long squeeze environment?"
    """
    
    def __init__(self, context_timeframes: List[str] = None):
        """
        Initialize context assessment component
        
        Args:
            context_timeframes: Timeframes to analyze (1s-aware)
        """
        # 1s mode awareness
        self.enable_1s_mode = os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true'
        
        if context_timeframes is None:
            if self.enable_1s_mode:
                self.context_timeframes = ["15m", "30m", "1h"]  # Shorter for 1s mode
            else:
                self.context_timeframes = ["30m", "1h", "4h"]  # Original
        else:
            self.context_timeframes = context_timeframes
        
        # Configure statistics processor for 1s mode
        set_1s_mode(self.enable_1s_mode)
        
        # Performance optimization: pre-calculate density factors
        self.density_factor = 60 if self.enable_1s_mode else 1
        self.volume_lookback = 6000 if self.enable_1s_mode else 100  # 100 minutes equivalent
        self.trend_lookback = 3000 if self.enable_1s_mode else 50   # 50 minutes equivalent
        self.divergence_lookback = 1200 if self.enable_1s_mode else 20  # 20 minutes equivalent
        
    def assess_context(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze larger timeframes to identify overall market bias
        
        Process:
        - Examine longer timeframes to identify overall market bias
        - Determine if market is in SHORT_SQUEEZE or LONG_SQUEEZE environment
        - Look for sustained volume accumulation patterns in one direction
        - Larger volume imbalances = longer squeeze duration potential
        
        Args:
            dataset: Market data including spot_cvd, futures_cvd, ohlcv
            
        Returns:
            Dict containing context analysis results
        """
        try:
            # Extract data
            spot_cvd = dataset.get('spot_cvd', pd.Series())
            futures_cvd = dataset.get('futures_cvd', pd.Series())
            cvd_divergence = dataset.get('cvd_divergence', pd.Series())
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            
            if spot_cvd.empty or futures_cvd.empty or ohlcv.empty:
                return self._empty_context()
                
            # Analyze volume accumulation patterns
            volume_analysis = self._analyze_volume_accumulation(spot_cvd, futures_cvd)
            
            # Determine dominant squeeze environment
            squeeze_environment = self._determine_squeeze_environment(
                spot_cvd, futures_cvd, cvd_divergence, ohlcv
            )
            
            # Assess squeeze duration potential
            duration_potential = self._assess_duration_potential(volume_analysis)
            
            return {
                'phase': 'CONTEXT_ASSESSMENT',
                'squeeze_environment': squeeze_environment,
                'volume_analysis': volume_analysis,
                'duration_potential': duration_potential,
                'market_bias': self._determine_market_bias(squeeze_environment),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'phase': 'CONTEXT_ASSESSMENT',
                'error': f'Context assessment error: {str(e)}',
                'squeeze_environment': 'NEUTRAL',
                'market_bias': 'NEUTRAL'
            }
    
    def _analyze_volume_accumulation(self, spot_cvd: pd.Series, futures_cvd: pd.Series) -> Dict[str, Any]:
        """Analyze sustained volume accumulation patterns (optimized for 1s density)"""
        
        if spot_cvd.empty or futures_cvd.empty:
            return {'trend': 'INSUFFICIENT_DATA', 'strength': 0}
        
        # Use optimized lookback with safety checks
        lookback = min(self.volume_lookback, len(spot_cvd) // 4)
        if lookback < 2:
            lookback = min(len(spot_cvd), len(futures_cvd)) - 1
            if lookback < 1:
                return {'trend': 'INSUFFICIENT_DATA', 'strength': 0}
        
        # Vectorized trend analysis for both series
        spot_analysis = vectorized_trend_analysis(spot_cvd, periods=[lookback])
        futures_analysis = vectorized_trend_analysis(futures_cvd, periods=[lookback])
        
        if not spot_analysis['trends'] or not futures_analysis['trends']:
            return {'trend': 'INSUFFICIENT_DATA', 'strength': 0}
        
        # Extract trend metrics
        spot_key = list(spot_analysis['trends'].keys())[0]
        futures_key = list(futures_analysis['trends'].keys())[0]
        
        spot_trend_data = spot_analysis['trends'][spot_key]
        futures_trend_data = futures_analysis['trends'][futures_key]
        
        spot_change = spot_trend_data['pct_change'] * spot_cvd.iloc[-1] if spot_cvd.iloc[-1] != 0 else 0
        futures_change = futures_trend_data['pct_change'] * futures_cvd.iloc[-1] if futures_cvd.iloc[-1] != 0 else 0
        
        # Vectorized dominance calculation
        total_volume_change = abs(spot_change) + abs(futures_change)
        if total_volume_change > 0:
            spot_dominance = abs(spot_change) / total_volume_change
            futures_dominance = abs(futures_change) / total_volume_change
        else:
            spot_dominance = 0.5
            futures_dominance = 0.5
        
        # Use adaptive thresholds for 1s mode
        dominance_threshold = adaptive_significance_threshold(spot_cvd, 0.6)
        
        return {
            'spot_trend': spot_change,
            'futures_trend': futures_change,
            'spot_dominance': spot_dominance,
            'futures_dominance': futures_dominance,
            'trend': 'SPOT_DOMINATED' if spot_dominance > dominance_threshold else 
                    'FUTURES_DOMINATED' if futures_dominance > dominance_threshold else 'BALANCED',
            'strength': max(spot_dominance, futures_dominance),
            'trend_consistency': (spot_analysis['trend_consistency'] + futures_analysis['trend_consistency']) / 2
        }
    
    def _determine_squeeze_environment(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                                     cvd_divergence: pd.Series, ohlcv: pd.DataFrame) -> str:
        """
        Determine if market is in SHORT_SQUEEZE or LONG_SQUEEZE environment
        (optimized for 1s density)
        
        LONG_SQUEEZE: Shorts are being squeezed (price rising, spot leading)
        SHORT_SQUEEZE: Longs are being squeezed (price falling, futures heavy)
        """
        
        # Vectorized price trend analysis
        price_analysis = vectorized_momentum_analysis(ohlcv, periods=[self.trend_lookback // self.density_factor])
        price_trend_score = price_analysis.get('momentum_score', 0)
        
        # Use pre-calculated lookback periods
        lookback = min(self.trend_lookback, len(spot_cvd) // 2)
        if lookback < 1:
            return 'NEUTRAL'
        
        # Vectorized momentum analysis for both CVD series
        spot_analysis = vectorized_trend_analysis(spot_cvd, periods=[lookback])
        futures_analysis = vectorized_trend_analysis(futures_cvd, periods=[lookback])
        
        if not spot_analysis['trends'] or not futures_analysis['trends']:
            return 'NEUTRAL'
        
        # Extract momentum values
        spot_key = list(spot_analysis['trends'].keys())[0]
        futures_key = list(futures_analysis['trends'].keys())[0]
        
        spot_momentum = spot_analysis['trends'][spot_key]['pct_change']
        futures_momentum = futures_analysis['trends'][futures_key]['pct_change']
        
        # Vectorized recent divergence calculation
        divergence_lookback = min(self.divergence_lookback, len(cvd_divergence))
        if divergence_lookback > 0:
            recent_divergence = np.mean(cvd_divergence.iloc[-divergence_lookback:].values)
        else:
            recent_divergence = 0
        
        # Use adaptive thresholds for squeeze detection
        momentum_threshold = adaptive_significance_threshold(spot_cvd, 0.01)
        
        # Determine squeeze type with improved logic
        price_rising = price_trend_score > momentum_threshold
        price_falling = price_trend_score < -momentum_threshold
        
        if price_rising and spot_momentum > futures_momentum and recent_divergence > 0:
            return 'LONG_SQUEEZE'  # Shorts being squeezed
        elif price_falling and futures_momentum > spot_momentum and recent_divergence < 0:
            return 'SHORT_SQUEEZE'  # Longs being squeezed
        else:
            return 'NEUTRAL'
    
    def _calculate_price_trend(self, ohlcv: pd.DataFrame) -> float:
        """Calculate overall price trend (optimized vectorized version)"""
        if ohlcv.empty:
            return 0
        
        # Use vectorized momentum analysis
        analysis = vectorized_momentum_analysis(ohlcv)
        return analysis.get('momentum_score', 0)
    
    def _assess_duration_potential(self, volume_analysis: Dict[str, Any]) -> str:
        """
        Assess squeeze duration potential based on volume imbalances
        Larger volume imbalances = longer squeeze duration potential
        """
        strength = volume_analysis.get('strength', 0)
        
        if strength > 0.8:
            return 'HIGH_DURATION'  # Strong imbalance, likely to persist
        elif strength > 0.65:
            return 'MEDIUM_DURATION'
        else:
            return 'LOW_DURATION'
    
    def _determine_market_bias(self, squeeze_environment: str) -> str:
        """Determine overall market bias based on squeeze environment"""
        if squeeze_environment == 'LONG_SQUEEZE':
            return 'BULLISH'  # Shorts squeezed = bullish
        elif squeeze_environment == 'SHORT_SQUEEZE':
            return 'BEARISH'  # Longs squeezed = bearish
        else:
            return 'NEUTRAL'
    
    def _empty_context(self) -> Dict[str, Any]:
        """Return empty context when data is insufficient"""
        return {
            'phase': 'CONTEXT_ASSESSMENT',
            'squeeze_environment': 'NEUTRAL',
            'volume_analysis': {'trend': 'INSUFFICIENT_DATA', 'strength': 0},
            'duration_potential': 'UNKNOWN',
            'market_bias': 'NEUTRAL',
            'error': 'Insufficient data for context assessment'
        }