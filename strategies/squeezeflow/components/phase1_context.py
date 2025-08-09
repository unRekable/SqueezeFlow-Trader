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
        """Analyze sustained volume accumulation patterns (1s-aware)"""
        
        # Calculate recent trends with 1s mode adjustment
        base_lookback = 100 if not self.enable_1s_mode else 6000  # 100min in 1s mode
        lookback = min(base_lookback, len(spot_cvd) // 4)  # Use 25% of data or adjusted periods
        
        if len(spot_cvd) < lookback or len(futures_cvd) < lookback or lookback < 2:
            return {'trend': 'INSUFFICIENT_DATA', 'strength': 0}
            
        # Recent CVD changes - safe array access
        spot_recent_slice = spot_cvd.iloc[-lookback:]
        futures_recent_slice = futures_cvd.iloc[-lookback:]
        
        if len(spot_recent_slice) < 2 or len(futures_recent_slice) < 2:
            return {'trend': 'INSUFFICIENT_DATA', 'strength': 0}
            
        spot_trend = spot_recent_slice.iloc[-1] - spot_recent_slice.iloc[0]
        futures_trend = futures_recent_slice.iloc[-1] - futures_recent_slice.iloc[0]
        
        # Volume imbalance analysis
        total_volume_change = abs(spot_trend) + abs(futures_trend)
        if total_volume_change > 0:
            spot_dominance = abs(spot_trend) / total_volume_change
            futures_dominance = abs(futures_trend) / total_volume_change
        else:
            spot_dominance = 0.5
            futures_dominance = 0.5
            
        return {
            'spot_trend': spot_trend,
            'futures_trend': futures_trend,
            'spot_dominance': spot_dominance,
            'futures_dominance': futures_dominance,
            'trend': 'SPOT_DOMINATED' if spot_dominance > 0.6 else 'FUTURES_DOMINATED' if futures_dominance > 0.6 else 'BALANCED',
            'strength': max(spot_dominance, futures_dominance)
        }
    
    def _determine_squeeze_environment(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                                     cvd_divergence: pd.Series, ohlcv: pd.DataFrame) -> str:
        """
        Determine if market is in SHORT_SQUEEZE or LONG_SQUEEZE environment
        
        LONG_SQUEEZE: Shorts are being squeezed (price rising, spot leading)
        SHORT_SQUEEZE: Longs are being squeezed (price falling, futures heavy)
        """
        
        # Price trend analysis
        price_trend = self._calculate_price_trend(ohlcv)
        
        # CVD trend analysis with 1s mode adjustment
        base_lookback = 50 if not self.enable_1s_mode else 3000  # 50min in 1s mode
        lookback = min(base_lookback, len(spot_cvd) // 2)
        
        if len(spot_cvd) > lookback and lookback > 0:
            spot_pct_change = spot_cvd.pct_change(lookback).iloc[-1]
            spot_momentum = spot_pct_change if not pd.isna(spot_pct_change) else 0
        else:
            spot_momentum = 0
            
        if len(futures_cvd) > lookback and lookback > 0:
            futures_pct_change = futures_cvd.pct_change(lookback).iloc[-1]
            futures_momentum = futures_pct_change if not pd.isna(futures_pct_change) else 0
        else:
            futures_momentum = 0
        
        # Recent divergence trend with 1s mode adjustment
        divergence_lookback = 20 if not self.enable_1s_mode else 1200  # 20min in 1s mode
        recent_divergence = cvd_divergence.iloc[-divergence_lookback:].mean() if len(cvd_divergence) > divergence_lookback else 0
        
        # Determine squeeze type based on patterns
        if price_trend > 0 and spot_momentum > futures_momentum and recent_divergence > 0:
            return 'LONG_SQUEEZE'  # Shorts being squeezed
        elif price_trend < 0 and futures_momentum > spot_momentum and recent_divergence < 0:
            return 'SHORT_SQUEEZE'  # Longs being squeezed
        else:
            return 'NEUTRAL'
    
    def _calculate_price_trend(self, ohlcv: pd.DataFrame) -> float:
        """Calculate overall price trend"""
        if ohlcv.empty or len(ohlcv) < 20:
            return 0
            
        # Get close prices
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        closes = ohlcv[close_col]
        
        # Simple trend: current vs 20 periods ago
        if len(closes) < 20 or closes.iloc[-20] == 0:
            return 0
        return (closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20]
    
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