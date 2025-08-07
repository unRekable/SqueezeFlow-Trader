"""
Phase 3: Reset Detection Component

Identifies market exhaustion through convergence patterns and equilibrium restoration.
This phase provides timing signals without scoring points.

Based on SqueezeFlow.md lines 99-134
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class ResetDetection:
    """
    Phase 3: Reset Detection - Convergence-Based Exhaustion System
    
    Objective: Identify market exhaustion through convergence patterns
    Two reset types:
    - Type A: Convergence exhaustion (CVD convergence + price stagnation)
    - Type B: Explosive confirmation (large price move after convergence)
    """
    
    def __init__(self, reset_timeframes: List[str] = None):
        """
        Initialize reset detection component
        
        Args:
            reset_timeframes: Timeframes to analyze (default: ["5m", "15m", "30m"])
        """
        self.reset_timeframes = reset_timeframes or ["5m", "15m", "30m"]
        
    def detect_reset(self, dataset: Dict[str, Any], context: Dict[str, Any], divergence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify convergence exhaustion patterns (Reset Type A & B)
        
        Reset Type A - Convergence Exhaustion Pattern:
        - Price movement driven by unbalanced CVD
        - SPOT and PERP CVD begin converging
        - Price stagnation despite active convergence
        - Market seeking equilibrium
        
        Reset Type B - Explosive Confirmation:
        - Large price movement following convergence
        - Major move with supporting CVD
        - Equilibrium restoration through volatility
        
        Args:
            dataset: Market data including spot_cvd, futures_cvd, ohlcv
            context: Results from Phase 1 context assessment
            divergence: Results from Phase 2 divergence detection
            
        Returns:
            Dict containing reset detection results
        """
        try:
            # Extract data
            spot_cvd = dataset.get('spot_cvd', pd.Series())
            futures_cvd = dataset.get('futures_cvd', pd.Series())
            cvd_divergence = dataset.get('cvd_divergence', pd.Series())
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            
            if spot_cvd.empty or futures_cvd.empty or ohlcv.empty:
                return self._empty_reset()
            
            # Detect unbalanced movement patterns
            unbalanced_pattern = self._detect_unbalanced_movement(spot_cvd, futures_cvd, divergence)
            
            # Check for convergence patterns
            convergence = self._detect_convergence(spot_cvd, futures_cvd, cvd_divergence)
            
            # Check for price momentum exhaustion
            momentum_exhaustion = self._detect_momentum_exhaustion(ohlcv)
            
            # Detect Reset Type A (convergence exhaustion)
            reset_type_a = self._detect_reset_type_a(convergence, momentum_exhaustion)
            
            # Detect Reset Type B (explosive confirmation)
            reset_type_b = self._detect_reset_type_b(ohlcv, cvd_divergence)
            
            # Multi-timeframe alignment check
            timeframe_alignment = self._check_timeframe_alignment(convergence)
            
            # Determine if reset is detected
            reset_detected = reset_type_a['detected'] or reset_type_b['detected']
            reset_type = self._determine_reset_type(reset_type_a, reset_type_b)
            
            return {
                'phase': 'RESET_DETECTION',
                'reset_detected': reset_detected,
                'reset_type': reset_type,
                'unbalanced_pattern': unbalanced_pattern,
                'convergence': convergence,
                'momentum_exhaustion': momentum_exhaustion,
                'reset_type_a': reset_type_a,
                'reset_type_b': reset_type_b,
                'timeframe_alignment': timeframe_alignment,
                'equilibrium_status': self._assess_equilibrium_status(convergence, reset_detected),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'phase': 'RESET_DETECTION',
                'error': f'Reset detection error: {str(e)}',
                'reset_detected': False,
                'reset_type': 'NONE'
            }
    
    def _detect_unbalanced_movement(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                                  divergence: Dict[str, Any]) -> Dict[str, Any]:
        """Identify price trends driven primarily by one CVD type"""
        
        if len(spot_cvd) < 20:
            return {'pattern': 'INSUFFICIENT_DATA', 'dominant_side': 'NONE'}
        
        # Recent movement analysis
        lookback = 20
        if len(spot_cvd) >= lookback and len(futures_cvd) >= lookback:
            spot_pct_changes = spot_cvd.iloc[-lookback:].pct_change(fill_method=None)
            futures_pct_changes = futures_cvd.iloc[-lookback:].pct_change(fill_method=None)
            
            # Handle NaN values from pct_change
            spot_momentum = spot_pct_changes.dropna().mean() if not spot_pct_changes.dropna().empty else 0
            futures_momentum = futures_pct_changes.dropna().mean() if not futures_pct_changes.dropna().empty else 0
        else:
            spot_momentum = 0
            futures_momentum = 0
        
        # Determine dominance
        total_momentum = abs(spot_momentum) + abs(futures_momentum)
        if total_momentum > 0:
            spot_dominance = abs(spot_momentum) / total_momentum
            futures_dominance = abs(futures_momentum) / total_momentum
        else:
            spot_dominance = 0.5
            futures_dominance = 0.5
        
        # Identify dominant pattern
        if spot_dominance > 0.65:
            pattern = 'SPOT_DOMINATED'
            dominant_side = 'SPOT'
        elif futures_dominance > 0.65:
            pattern = 'PERP_DOMINATED'
            dominant_side = 'PERP'
        else:
            pattern = 'BALANCED'
            dominant_side = 'NONE'
            
        return {
            'pattern': pattern,
            'dominant_side': dominant_side,
            'spot_dominance': spot_dominance,
            'futures_dominance': futures_dominance,
            'is_unbalanced': pattern != 'BALANCED'
        }
    
    def _detect_convergence(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                          cvd_divergence: pd.Series) -> Dict[str, Any]:
        """Detect if SPOT and PERP CVD are converging"""
        
        if len(cvd_divergence) < 10:
            return {'converging': False, 'rate': 0, 'strength': 0}
        
        # Calculate convergence rate
        recent_divergence = cvd_divergence.iloc[-10:]
        divergence_change = recent_divergence.diff()
        convergence_rate = -divergence_change.mean()
        
        # Check if converging (divergence decreasing)
        is_converging = convergence_rate > 0 and divergence_change.iloc[-3:].mean() < 0
        
        # Calculate convergence strength
        initial_divergence = abs(cvd_divergence.iloc[-10])
        current_divergence = abs(cvd_divergence.iloc[-1])
        
        if initial_divergence > 0:
            convergence_strength = (initial_divergence - current_divergence) / initial_divergence
        else:
            convergence_strength = 0
        
        # Check threshold
        convergence_strength_threshold = 0.2
        
        return {
            'converging': is_converging and convergence_strength > convergence_strength_threshold,
            'rate': convergence_rate,
            'strength': convergence_strength,
            'initial_divergence': initial_divergence,
            'current_divergence': current_divergence,
            'trend': 'CONVERGING' if is_converging else 'DIVERGING'
        }
    
    def _detect_momentum_exhaustion(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Detect if price momentum is exhausting"""
        
        if len(ohlcv) < 20:
            return {'exhausted': False, 'deceleration': 0}
        
        # Get close prices
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        closes = ohlcv[close_col]
        
        # Calculate momentum over different periods
        momentum_5 = closes.pct_change(5, fill_method=None).iloc[-1]
        momentum_10 = closes.pct_change(10, fill_method=None).iloc[-1]
        momentum_20 = closes.pct_change(20, fill_method=None).iloc[-1]
        
        # Check for deceleration
        deceleration = abs(momentum_5) < abs(momentum_10) < abs(momentum_20)
        
        # Check for stagnation
        recent_range = closes.iloc[-10:].max() - closes.iloc[-10:].min()
        avg_range = closes.pct_change(fill_method=None).abs().mean() * closes.mean() * 10
        is_stagnant = recent_range < avg_range * 0.5
        
        return {
            'exhausted': deceleration or is_stagnant,
            'deceleration': deceleration,
            'stagnant': is_stagnant,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'momentum_20': momentum_20
        }
    
    def _detect_reset_type_a(self, convergence: Dict[str, Any], 
                           momentum_exhaustion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect Reset Type A - Convergence Exhaustion Pattern
        
        Criteria:
        - CVD convergence detected 
        - Price momentum exhausting
        """
        
        detected = (
            convergence['converging'] and
            convergence['strength'] > 0.2 and
            momentum_exhaustion['exhausted']
        )
        
        return {
            'detected': detected,
            'convergence_strength': convergence['strength'],
            'has_momentum_exhaustion': momentum_exhaustion['exhausted'],
            'pattern': 'CONVERGENCE_EXHAUSTION' if detected else 'NONE'
        }
    
    def _detect_reset_type_b(self, ohlcv: pd.DataFrame, cvd_divergence: pd.Series) -> Dict[str, Any]:
        """
        Detect Reset Type B - Explosive Confirmation
        
        Criteria:
        - Large price movement detected
        - Supporting CVD movement
        - Volatility spike
        """
        
        if len(ohlcv) < 10:
            return {'detected': False, 'spike_magnitude': 0}
        
        # Get price data
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        high_col = 'high' if 'high' in ohlcv.columns else ohlcv.columns[1]
        low_col = 'low' if 'low' in ohlcv.columns else ohlcv.columns[2]
        
        closes = ohlcv[close_col]
        highs = ohlcv[high_col]
        lows = ohlcv[low_col]
        
        # Calculate recent volatility spike
        if len(closes) >= 5:
            recent_high = highs.iloc[-5:].max()
            recent_low = lows.iloc[-5:].min()
            recent_close_mean = closes.iloc[-5:].mean()
            
            if recent_close_mean != 0:
                recent_range = (recent_high - recent_low) / recent_close_mean
            else:
                recent_range = 0
        else:
            recent_range = 0
            
        if len(closes) >= 20:
            avg_high_low_diff = (highs - lows).iloc[-20:-5].mean()
            avg_close_mean = closes.iloc[-20:-5].mean()
            
            if avg_close_mean != 0:
                avg_range = avg_high_low_diff / avg_close_mean
            else:
                avg_range = 0.001  # Small non-zero value to prevent division by zero
        else:
            avg_range = 0.001
        
        volatility_spike = recent_range > avg_range * 2.5
        
        # Check for large price movement
        price_spike = abs(closes.pct_change(5, fill_method=None).iloc[-1]) > 0.03  # 3% move
        
        # Check for supporting CVD
        if len(cvd_divergence) >= 5:
            cvd_support = abs(cvd_divergence.iloc[-1]) > abs(cvd_divergence.iloc[-5]) * 1.5
        else:
            cvd_support = False
        
        detected = volatility_spike and (price_spike or cvd_support)
        
        return {
            'detected': detected,
            'spike_magnitude': recent_range / avg_range if avg_range > 0 else 0,
            'has_volatility_spike': volatility_spike,
            'has_price_spike': price_spike,
            'has_cvd_support': cvd_support,
            'pattern': 'EXPLOSIVE_CONFIRMATION' if detected else 'NONE'
        }
    
    def _check_timeframe_alignment(self, convergence: Dict[str, Any]) -> Dict[str, Any]:
        """Check if multiple timeframes show aligned convergence"""
        
        # In real implementation, this would check multiple timeframes
        # For now, we use the primary timeframe data
        alignment_score = convergence['strength'] if convergence['converging'] else 0
        
        return {
            'aligned': alignment_score > 0.3,
            'score': alignment_score,
            'timeframes_checked': self.reset_timeframes
        }
    
    def _determine_reset_type(self, reset_type_a: Dict[str, Any], 
                            reset_type_b: Dict[str, Any]) -> str:
        """Determine which reset type was detected"""
        
        if reset_type_a['detected'] and reset_type_b['detected']:
            return 'BOTH'
        elif reset_type_a['detected']:
            return 'TYPE_A'
        elif reset_type_b['detected']:
            return 'TYPE_B'
        else:
            return 'NONE'
    
    def _assess_equilibrium_status(self, convergence: Dict[str, Any], 
                                 reset_detected: bool) -> str:
        """Assess market equilibrium restoration status"""
        
        if reset_detected:
            return 'EQUILIBRIUM_RESTORED'
        elif convergence['converging']:
            return 'SEEKING_EQUILIBRIUM'
        else:
            return 'IMBALANCED'
    
    def _empty_reset(self) -> Dict[str, Any]:
        """Return empty reset when data is insufficient"""
        return {
            'phase': 'RESET_DETECTION',
            'reset_detected': False,
            'reset_type': 'NONE',
            'unbalanced_pattern': {'pattern': 'INSUFFICIENT_DATA'},
            'convergence': {'converging': False},
            'momentum_exhaustion': {'exhausted': False},
            'equilibrium_status': 'UNKNOWN',
            'error': 'Insufficient data for reset detection'
        }