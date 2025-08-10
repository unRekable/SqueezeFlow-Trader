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
import os
import sys

# Import optimized statistics functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.statistics import (
    efficient_convergence_detection,
    vectorized_momentum_analysis,
    adaptive_significance_threshold,
    set_1s_mode
)


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
            reset_timeframes: Timeframes to analyze (1s-aware)
        """
        # 1s mode awareness
        self.enable_1s_mode = os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true'
        
        if reset_timeframes is None:
            # Timeframes are analysis windows, NOT data resolution!
            # 1s is the data granularity, not a timeframe to analyze
            self.reset_timeframes = ["5m", "15m", "30m"]  # Analysis windows
        else:
            self.reset_timeframes = reset_timeframes
        
        # Configure statistics processor for 1s mode
        set_1s_mode(self.enable_1s_mode)
        
        # Performance optimization: pre-calculate parameters
        self.density_factor = 60 if self.enable_1s_mode else 1
        self.unbalanced_lookback = 1200 if self.enable_1s_mode else 20     # 20 minutes equivalent
        self.convergence_window = 600 if self.enable_1s_mode else 10       # 10 minutes equivalent
        self.momentum_periods = [300, 600, 900] if self.enable_1s_mode else [5, 10, 15]  # Multiple timeframes
        
        # Log 1s mode status
        if self.enable_1s_mode:
            print(f"Phase 3 Reset: 1s mode enabled, using timeframes: {self.reset_timeframes}")
        
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
        """Identify price trends driven primarily by one CVD type (optimized for 1s density)"""
        
        if spot_cvd.empty or futures_cvd.empty:
            return {'pattern': 'INSUFFICIENT_DATA', 'dominant_side': 'NONE'}
        
        # Use pre-calculated lookback with safety checks
        lookback = min(self.unbalanced_lookback, len(spot_cvd) - 1, len(futures_cvd) - 1)
        if lookback < 1:
            return {'pattern': 'INSUFFICIENT_DATA', 'dominant_side': 'NONE'}
        
        # Vectorized momentum calculations using numpy
        if len(spot_cvd) > lookback and len(futures_cvd) > lookback:
            # Get recent slices
            spot_values = spot_cvd.iloc[-lookback:].values
            futures_values = futures_cvd.iloc[-lookback:].values
            
            # Vectorized percentage change calculations
            spot_changes = np.diff(spot_values) / (spot_values[:-1] + 1e-8)
            futures_changes = np.diff(futures_values) / (futures_values[:-1] + 1e-8)
            
            # Remove any inf/nan values
            spot_changes = spot_changes[np.isfinite(spot_changes)]
            futures_changes = futures_changes[np.isfinite(futures_changes)]
            
            spot_momentum = np.mean(spot_changes) if len(spot_changes) > 0 else 0
            futures_momentum = np.mean(futures_changes) if len(futures_changes) > 0 else 0
        else:
            spot_momentum = 0
            futures_momentum = 0
        
        # Vectorized dominance calculation
        total_momentum = abs(spot_momentum) + abs(futures_momentum)
        if total_momentum > 0:
            spot_dominance = abs(spot_momentum) / total_momentum
            futures_dominance = abs(futures_momentum) / total_momentum
        else:
            spot_dominance = 0.5
            futures_dominance = 0.5
        
        # FIXED: More reasonable dominance threshold
        dominance_threshold = 0.6  # Fixed threshold instead of adaptive
        
        # Identify dominant pattern
        if spot_dominance > dominance_threshold:
            pattern = 'SPOT_DOMINATED'
            dominant_side = 'SPOT'
        elif futures_dominance > dominance_threshold:
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
            'is_unbalanced': pattern != 'BALANCED',
            'momentum_strength': total_momentum,
            'threshold_used': dominance_threshold
        }
    
    def _detect_convergence(self, spot_cvd: pd.Series, futures_cvd: pd.Series, 
                          cvd_divergence: pd.Series) -> Dict[str, Any]:
        """Detect if SPOT and PERP CVD are converging (optimized version)"""
        
        if cvd_divergence.empty:
            return {'converging': False, 'rate': 0, 'strength': 0}
        
        # Use optimized convergence detection
        convergence_analysis = efficient_convergence_detection(
            cvd_divergence, window_size=self.convergence_window
        )
        
        return {
            'converging': convergence_analysis['converging'],
            'rate': convergence_analysis['rate'],
            'strength': convergence_analysis['strength'],
            'initial_divergence': abs(cvd_divergence.iloc[0]) if len(cvd_divergence) > 0 else 0,
            'current_divergence': abs(cvd_divergence.iloc[-1]) if len(cvd_divergence) > 0 else 0,
            'trend': convergence_analysis['trend']
        }
    
    def _detect_momentum_exhaustion(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Detect if price momentum is exhausting (optimized vectorized version)"""
        
        if ohlcv.empty:
            return {'exhausted': False, 'deceleration': False}
        
        # Use vectorized momentum analysis with multiple periods
        momentum_analysis = vectorized_momentum_analysis(
            ohlcv, periods=self.momentum_periods
        )
        
        return {
            'exhausted': momentum_analysis['exhausted'],
            'deceleration': momentum_analysis['deceleration'],
            'stagnant': momentum_analysis.get('stagnation', False),
            'momentum_score': momentum_analysis['momentum_score'],
            'trend': momentum_analysis['trend']
        }
    
    def _detect_reset_type_a(self, convergence: Dict[str, Any], 
                           momentum_exhaustion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect Reset Type A - Convergence Exhaustion Pattern
        
        Criteria:
        - CVD convergence detected 
        - Price momentum exhausting
        """
        
        # FIXED: More lenient convergence thresholds
        detected = (
            (convergence['converging'] and convergence['strength'] > 0.05) or  # Reduced from 0.2
            (momentum_exhaustion['exhausted'] and convergence['strength'] > 0.02) or  # Alternative path
            (momentum_exhaustion.get('deceleration', False) and convergence.get('rate', 0) < 0)  # Deceleration counts
        )
        
        return {
            'detected': detected,
            'convergence_strength': convergence['strength'],
            'has_momentum_exhaustion': momentum_exhaustion['exhausted'],
            'pattern': 'CONVERGENCE_EXHAUSTION' if detected else 'NONE'
        }
    
    def _detect_reset_type_b(self, ohlcv: pd.DataFrame, cvd_divergence: pd.Series) -> Dict[str, Any]:
        """
        Detect Reset Type B - Explosive Confirmation (optimized vectorized version)
        
        Criteria:
        - Large price movement detected
        - Supporting CVD movement
        - Volatility spike
        """
        
        if ohlcv.empty:
            return {'detected': False, 'spike_magnitude': 0}
        
        # Use vectorized momentum analysis for volatility detection
        momentum_analysis = vectorized_momentum_analysis(ohlcv)
        
        # Get basic price metrics
        close_col = 'close' if 'close' in ohlcv.columns else (ohlcv.columns[3] if len(ohlcv.columns) > 3 else ohlcv.columns[0])
        closes = ohlcv[close_col]
        
        # Check for explosive price movements
        price_spike = momentum_analysis['momentum_score'] > 0.03  # 3% move threshold
        
        # Volatility spike detection from momentum analysis
        volatility_spike = not momentum_analysis['exhausted'] and momentum_analysis['trend'] == 'STRONG'
        
        # CVD support check with vectorized operations
        if not cvd_divergence.empty and len(cvd_divergence) >= 5:
            recent_cvd = cvd_divergence.iloc[-5:].values
            cvd_support = abs(recent_cvd[-1]) > np.mean(np.abs(recent_cvd[:-1])) * 1.5
        else:
            cvd_support = False
        
        # FIXED: More lenient detection for Type B
        detection_threshold = 0.02  # 2% move threshold
        detected = (volatility_spike and (price_spike or cvd_support)) or momentum_analysis['momentum_score'] > detection_threshold
        
        return {
            'detected': detected,
            'spike_magnitude': momentum_analysis['momentum_score'] / detection_threshold if detection_threshold > 0 else 0,
            'has_volatility_spike': volatility_spike,
            'has_price_spike': price_spike,
            'has_cvd_support': cvd_support,
            'pattern': 'EXPLOSIVE_CONFIRMATION' if detected else 'NONE',
            'momentum_trend': momentum_analysis['trend']
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