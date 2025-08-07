"""
Phase 5: Exit Management Component

Manages position exits based on flow-following principles.
NO fixed stop losses or profit targets - only dynamic exits.

Based on SqueezeFlow.md lines 229-249
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Optional CVD baseline manager import for live trading
try:
    from strategies.squeezeflow.baseline_manager import CVDBaselineManager
except ImportError:
    CVDBaselineManager = None


class ExitManagement:
    """
    Phase 5: Position Management
    
    Objective: Follow the order flow until conditions change
    
    Management Principles:
    - NO fixed profit targets - follow squeeze until invalidation
    - NO fixed stop losses - exit based on condition changes
    - Dynamic exits only - exit when market structure invalidates
    - Track CVD trends continuously
    - Monitor larger timeframe validity
    """
    
    def __init__(self, cvd_baseline_manager: Optional['CVDBaselineManager'] = None):
        """
        Initialize exit management component
        
        Args:
            cvd_baseline_manager: Optional CVD baseline manager for live trading
        """
        self.cvd_baseline_manager = cvd_baseline_manager
        
    def manage_exits(self, dataset: Dict[str, Any], position: Dict[str, Any], 
                    entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if current position should be exited
        
        Exit Conditions:
        1. Flow Reversal Pattern - SPOT/PERP divergence against position
        2. Range Break - Price breaks below entry range/reset low
        3. CVD Trend Invalidation - CVD no longer supporting position
        4. Market Structure Break - Key levels violated
        
        Args:
            dataset: Current market data
            position: Current position details
            entry_analysis: Analysis from when position was entered
            
        Returns:
            Dict containing exit decision and reasoning
        """
        try:
            # Extract data
            spot_cvd = dataset.get('spot_cvd', pd.Series())
            futures_cvd = dataset.get('futures_cvd', pd.Series())
            cvd_divergence = dataset.get('cvd_divergence', pd.Series())
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            
            if spot_cvd.empty or ohlcv.empty or not position:
                return self._no_exit_signal()
                
            # Check for flow reversal
            flow_reversal = self._check_flow_reversal(
                spot_cvd, futures_cvd, position, entry_analysis
            )
            
            # Check for range break
            range_break = self._check_range_break(
                ohlcv, position, entry_analysis
            )
            
            # Check CVD trend invalidation
            cvd_invalidation = self._check_cvd_invalidation(
                spot_cvd, futures_cvd, position, entry_analysis
            )
            
            # Check market structure
            structure_break = self._check_structure_break(
                ohlcv, position, entry_analysis
            )
            
            # Determine if exit is warranted
            should_exit = (
                flow_reversal['detected'] or 
                range_break['detected'] or
                cvd_invalidation['detected'] or
                structure_break['detected']
            )
            
            # Generate exit reasoning
            exit_reasoning = self._generate_exit_reasoning(
                flow_reversal, range_break, cvd_invalidation, structure_break
            )
            
            return {
                'phase': 'EXIT_MANAGEMENT',
                'should_exit': should_exit,
                'exit_reasoning': exit_reasoning,
                'flow_reversal': flow_reversal,
                'range_break': range_break,
                'cvd_invalidation': cvd_invalidation,
                'structure_break': structure_break,
                'position_health': self._assess_position_health(
                    ohlcv, position, spot_cvd, futures_cvd
                ),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'phase': 'EXIT_MANAGEMENT',
                'error': f'Exit management error: {str(e)}',
                'should_exit': False
            }
    
    def _check_flow_reversal(self, spot_cvd: pd.Series, futures_cvd: pd.Series,
                           position: Dict[str, Any], entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for dangerous flow reversal pattern
        
        Dangerous situation:
        - SPOT CVD declining while PERP CVD elevated (for longs)
        - SPOT CVD rising while PERP CVD declining (for shorts)
        - Real money leaving, leverage will get squeezed
        """
        
        if len(spot_cvd) < 10:
            return {'detected': False, 'severity': 'NONE'}
            
        position_side = position.get('side', '').upper()
        
        # Calculate recent CVD trends
        lookback = min(10, len(spot_cvd) // 2)
        spot_trend = spot_cvd.iloc[-1] - spot_cvd.iloc[-lookback]
        futures_trend = futures_cvd.iloc[-1] - futures_cvd.iloc[-lookback]
        
        # Detect dangerous patterns based on position
        dangerous_for_long = (
            position_side == 'BUY' and
            spot_trend < 0 and  # SPOT selling
            futures_trend > 0   # PERP buying (shorts covering)
        )
        
        dangerous_for_short = (
            position_side == 'SELL' and
            spot_trend > 0 and   # SPOT buying
            futures_trend < 0    # PERP selling (longs closing)
        )
        
        detected = dangerous_for_long or dangerous_for_short
        
        # Assess severity
        if detected:
            divergence_magnitude = abs(spot_trend - futures_trend)
            if divergence_magnitude > abs(spot_cvd.iloc[-20:].std() * 2):
                severity = 'HIGH'
            elif divergence_magnitude > abs(spot_cvd.iloc[-20:].std()):
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
        else:
            severity = 'NONE'
            
        return {
            'detected': detected,
            'severity': severity,
            'spot_trend': spot_trend,
            'futures_trend': futures_trend,
            'pattern': 'DANGEROUS_DIVERGENCE' if detected else 'NORMAL'
        }
    
    def _check_range_break(self, ohlcv: pd.DataFrame, position: Dict[str, Any],
                          entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if price breaks below entry range or reset low
        
        Signals reset wasn't complete, needs bigger move
        """
        
        if len(ohlcv) < 5:
            return {'detected': False, 'break_type': 'NONE'}
            
        # Get price columns
        low_col = 'low' if 'low' in ohlcv.columns else ohlcv.columns[2]
        high_col = 'high' if 'high' in ohlcv.columns else ohlcv.columns[1]
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        
        # Get entry price and position side
        entry_price = position.get('entry_price', 0)
        position_side = position.get('side', '').upper()
        
        # Define entry range (approximate from entry)
        entry_range_size = entry_price * 0.005  # 0.5% range
        
        if position_side == 'BUY':
            # For longs, check if price breaks below entry range
            range_low = entry_price - entry_range_size
            current_price = ohlcv[close_col].iloc[-1]
            recent_low = ohlcv[low_col].iloc[-5:].min()
            
            detected = current_price < range_low or recent_low < range_low * 0.995
            break_type = 'BELOW_ENTRY' if detected else 'NONE'
            
        elif position_side == 'SELL':
            # For shorts, check if price breaks above entry range
            range_high = entry_price + entry_range_size
            current_price = ohlcv[close_col].iloc[-1]
            recent_high = ohlcv[high_col].iloc[-5:].max()
            
            detected = current_price > range_high or recent_high > range_high * 1.005
            break_type = 'ABOVE_ENTRY' if detected else 'NONE'
            
        else:
            detected = False
            break_type = 'NONE'
            
        return {
            'detected': detected,
            'break_type': break_type,
            'entry_price': entry_price,
            'current_price': ohlcv[close_col].iloc[-1] if not ohlcv.empty else 0
        }
    
    def _check_cvd_invalidation(self, spot_cvd: pd.Series, futures_cvd: pd.Series,
                               position: Dict[str, Any], entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if CVD trend no longer supports position
        
        Uses CVD baseline manager if available, otherwise falls back to approximation
        """
        
        if len(spot_cvd) < 5:
            return {'detected': False, 'invalidation_type': 'NONE'}
            
        position_side = position.get('side', '').upper()
        current_spot_cvd = spot_cvd.iloc[-1]
        current_futures_cvd = futures_cvd.iloc[-1]
        
        # Try to get real CVD baseline from baseline manager
        baseline_data = self._get_cvd_baseline(position)
        
        if baseline_data:
            # Fix: Safe access with defaults to prevent KeyError
            spot_change_since_entry = current_spot_cvd - baseline_data.get('spot_cvd', current_spot_cvd)
            futures_change_since_entry = current_futures_cvd - baseline_data.get('futures_cvd', current_futures_cvd)
            
        else:
            # Fallback to approximation (existing logic)
            entry_index = position.get('entry_index', -20)
            
            # Validate entry_index bounds
            if abs(entry_index) >= len(spot_cvd) or abs(entry_index) >= len(futures_cvd):
                entry_index = -min(20, len(spot_cvd) - 1, len(futures_cvd) - 1)
                
            # Ensure we have at least 1 data point
            if len(spot_cvd) < 1 or len(futures_cvd) < 1:
                return {'detected': False, 'invalidation_type': 'INSUFFICIENT_DATA'}
                
            spot_change_since_entry = current_spot_cvd - spot_cvd.iloc[entry_index]
            futures_change_since_entry = current_futures_cvd - futures_cvd.iloc[entry_index]
        
        # Check if CVD has reversed against position
        if position_side == 'BUY':
            # For longs, CVD should be increasing
            cvd_reversed = spot_change_since_entry < 0 and futures_change_since_entry < 0
            invalidation_type = 'CVD_DECLINING' if cvd_reversed else 'NONE'
            
        elif position_side == 'SELL':
            # For shorts, CVD should be decreasing
            cvd_reversed = spot_change_since_entry > 0 and futures_change_since_entry > 0
            invalidation_type = 'CVD_RISING' if cvd_reversed else 'NONE'
            
        else:
            cvd_reversed = False
            invalidation_type = 'NONE'
            
        return {
            'detected': cvd_reversed,
            'invalidation_type': invalidation_type,
            'spot_change': spot_change_since_entry,
            'futures_change': futures_change_since_entry,
            'baseline_used': baseline_data is not None
        }
    
    def _check_structure_break(self, ohlcv: pd.DataFrame, position: Dict[str, Any],
                              entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if key market structure levels are violated
        """
        
        if len(ohlcv) < 20:
            return {'detected': False, 'structure_type': 'INTACT'}
            
        # Get price columns
        high_col = 'high' if 'high' in ohlcv.columns else ohlcv.columns[1]
        low_col = 'low' if 'low' in ohlcv.columns else ohlcv.columns[2]
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        
        position_side = position.get('side', '').upper()
        current_price = ohlcv[close_col].iloc[-1]
        
        # Identify key structure levels
        recent_data = ohlcv.iloc[-20:]
        swing_high = recent_data[high_col].max()
        swing_low = recent_data[low_col].min()
        
        structure_broken = False
        structure_type = 'INTACT'
        
        if position_side == 'BUY':
            # For longs, breaking below recent swing low is bearish
            if current_price < swing_low * 0.998:  # Small buffer
                structure_broken = True
                structure_type = 'SWING_LOW_BROKEN'
                
        elif position_side == 'SELL':
            # For shorts, breaking above recent swing high is bullish
            if current_price > swing_high * 1.002:  # Small buffer
                structure_broken = True
                structure_type = 'SWING_HIGH_BROKEN'
                
        return {
            'detected': structure_broken,
            'structure_type': structure_type,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'current_price': current_price
        }
    
    def _assess_position_health(self, ohlcv: pd.DataFrame, position: Dict[str, Any],
                              spot_cvd: pd.Series, futures_cvd: pd.Series) -> Dict[str, Any]:
        """Assess overall health of current position"""
        
        if ohlcv.empty:
            return {'status': 'UNKNOWN', 'pnl_percent': 0}
            
        # Get current price
        close_col = 'close' if 'close' in ohlcv.columns else ohlcv.columns[3]
        current_price = ohlcv[close_col].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        position_side = position.get('side', '').upper()
        
        # Calculate PnL
        if position_side == 'BUY':
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
        elif position_side == 'SELL':
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
        else:
            pnl_percent = 0
            
        # Determine health status
        if pnl_percent > 2:
            status = 'HEALTHY'
        elif pnl_percent > 0:
            status = 'POSITIVE'
        elif pnl_percent > -1:
            status = 'NEUTRAL'
        else:
            status = 'UNHEALTHY'
            
        return {
            'status': status,
            'pnl_percent': pnl_percent,
            'entry_price': entry_price,
            'current_price': current_price
        }
    
    def _generate_exit_reasoning(self, flow_reversal: Dict[str, Any], 
                               range_break: Dict[str, Any],
                               cvd_invalidation: Dict[str, Any],
                               structure_break: Dict[str, Any]) -> str:
        """Generate human-readable exit reasoning"""
        
        reasons = []
        
        if flow_reversal['detected']:
            reasons.append(f"Flow reversal detected ({flow_reversal['severity']} severity)")
            
        if range_break['detected']:
            reasons.append(f"Range break detected ({range_break['break_type']})")
            
        if cvd_invalidation['detected']:
            reasons.append(f"CVD invalidation ({cvd_invalidation['invalidation_type']})")
            
        if structure_break['detected']:
            reasons.append(f"Market structure break ({structure_break['structure_type']})")
            
        if not reasons:
            return "Position conditions remain valid - continue holding"
            
        return " | ".join(reasons)
    
    def _get_cvd_baseline(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get CVD baseline data for position from baseline manager
        
        Args:
            position: Position dictionary with trade_id
            
        Returns:
            Dict with baseline CVD data or None if not available
        """
        if not self.cvd_baseline_manager:
            return None
            
        trade_id = position.get('trade_id')
        if not trade_id:
            return None
            
        try:
            baseline = self.cvd_baseline_manager.get_baseline(trade_id)
            if baseline:
                return {
                    'spot_cvd': baseline.spot_cvd,
                    'futures_cvd': baseline.futures_cvd,
                    'cvd_divergence': baseline.cvd_divergence,
                    'entry_time': baseline.entry_time,
                    'entry_price': baseline.entry_price
                }
        except Exception as e:
            # Log error but don't fail - fall back to approximation
            pass
            
        return None
    
    def _no_exit_signal(self) -> Dict[str, Any]:
        """Return when no exit signal is generated"""
        return {
            'phase': 'EXIT_MANAGEMENT',
            'should_exit': False,
            'exit_reasoning': 'Insufficient data or no position',
            'flow_reversal': {'detected': False},
            'range_break': {'detected': False},
            'cvd_invalidation': {'detected': False},
            'structure_break': {'detected': False}
        }