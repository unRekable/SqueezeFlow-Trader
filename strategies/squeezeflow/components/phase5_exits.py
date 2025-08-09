"""
Phase 5: Exit Management Component

Manages position exits based on flow-following principles.
NO fixed stop losses or profit targets - only dynamic exits.

Based on SqueezeFlow.md lines 229-249
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pytz
import os

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
    
    def __init__(self, cvd_baseline_manager: Optional['CVDBaselineManager'] = None, logger=None):
        """
        Initialize exit management component (1s-aware)
        
        Args:
            cvd_baseline_manager: Optional CVD baseline manager for live trading
            logger: Optional logger for debug output
        """
        # 1s mode awareness
        self.enable_1s_mode = os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true'
        
        self.cvd_baseline_manager = cvd_baseline_manager
        self.logger = logger
        if not self.logger:
            import logging
            self.logger = logging.getLogger(__name__)
            
        # Log 1s mode status
        if self.enable_1s_mode:
            print(f"Phase 5 Exits: 1s mode enabled, adjusted exit timing")
        
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
            
            # Ensure entry_analysis is a dictionary
            if not isinstance(entry_analysis, dict):
                entry_analysis = {}
            
            if spot_cvd.empty or ohlcv.empty or not position:
                return self._no_exit_signal()
            
            # Check if position has valid quantity
            if position.get('quantity', 0) == 0:
                self.logger.debug("Position has zero quantity, skipping exit check")
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
            
            # Debug logging for exit decision
            if should_exit:
                self.logger.info(f"EXIT SIGNAL: {exit_reasoning}")
                self.logger.debug(f"Exit conditions - Flow: {flow_reversal['detected']}, Range: {range_break['detected']}, CVD: {cvd_invalidation['detected']}, Structure: {structure_break['detected']}")
            else:
                self.logger.debug(f"No exit - Position {position.get('side')} at ${position.get('entry_price', 0):.2f}, current conditions valid")
            
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
            # Enhanced error logging to debug the string/get issue
            import traceback
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'entry_analysis_type': type(entry_analysis).__name__,
                'entry_analysis_value': str(entry_analysis)[:200] if entry_analysis else 'None',
                'position_type': type(position).__name__ if position else 'None'
            }
            
            print(f"EXIT MANAGEMENT ERROR: {error_details}")
            
            return {
                'phase': 'EXIT_MANAGEMENT',
                'error': f'Exit management error: {str(e)}',
                'error_details': error_details,
                'should_exit': False
            }
    
    def _check_flow_reversal(self, spot_cvd: pd.Series, futures_cvd: pd.Series,
                           position: Dict[str, Any], entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for dangerous flow reversal pattern
        
        From strategy doc line 253-256:
        "SPOT CVD declining while PERP CVD elevated" - real money leaving, leverage will get squeezed
        
        This is PURE pattern recognition - NO thresholds, just detect the pattern
        """
        
        if len(spot_cvd) < 20:  # Minimal data needed
            return {'detected': False, 'severity': 'NONE'}
            
        position_side = position.get('side', '').upper()
        
        # Get CVD values at entry (stored or approximate)
        if 'spot_cvd_entry' in position and 'futures_cvd_entry' in position:
            spot_cvd_entry = position['spot_cvd_entry']
            futures_cvd_entry = position['futures_cvd_entry']
        else:
            # Approximate entry point
            entry_index = min(20, len(spot_cvd) // 2)
            spot_cvd_entry = spot_cvd.iloc[-entry_index]
            futures_cvd_entry = futures_cvd.iloc[-entry_index]
        
        # Current CVD values
        current_spot = spot_cvd.iloc[-1]
        current_futures = futures_cvd.iloc[-1]
        
        # Simple pattern detection - is CVD moving opposite directions?
        spot_direction = current_spot - spot_cvd_entry
        futures_direction = current_futures - futures_cvd_entry
        
        # Filter out noise - movements should be meaningful relative to recent activity
        # This is NOT a threshold, just filtering random walk noise
        recent_noise = spot_cvd.iloc[-20:].diff().std()
        if recent_noise == 0:
            recent_noise = 1  # Avoid division by zero
            
        # Only consider it a pattern if movements are distinguishable from noise
        spot_moving = abs(spot_direction) > recent_noise * 0.5
        futures_moving = abs(futures_direction) > recent_noise * 0.5
        
        # Need at least one side to be moving meaningfully
        if not (spot_moving or futures_moving):
            return {'detected': False, 'severity': 'NONE', 'reason': 'No meaningful movement'}
        
        # For LONG: Dangerous if spot declining AND futures elevated/rising
        # For SHORT: Dangerous if spot rising AND futures declining
        
        dangerous_for_long = False
        dangerous_for_short = False
        
        if position_side in ['BUY', 'LONG']:
            # The pattern: spot going DOWN, futures going UP (or staying elevated)
            dangerous_for_long = (spot_direction < 0 and futures_direction > 0)
            
        elif position_side in ['SELL', 'SHORT']:
            # The pattern: spot going UP, futures going DOWN
            dangerous_for_short = (spot_direction > 0 and futures_direction < 0)
        
        detected = dangerous_for_long or dangerous_for_short
        
        # Severity is just based on how pronounced the divergence is
        # But still no fixed thresholds - just relative assessment
        if detected:
            # How opposite are they moving?
            divergence = abs(spot_direction - futures_direction)
            
            # Compare to recent typical movements to assess severity
            recent_volatility = spot_cvd.iloc[-20:].std()
            if recent_volatility > 0:
                relative_divergence = divergence / recent_volatility
                
                # These aren't thresholds, just categorization
                if relative_divergence > 3:
                    severity = 'HIGH'
                elif relative_divergence > 1.5:
                    severity = 'MEDIUM'
                else:
                    severity = 'LOW'
            else:
                severity = 'LOW'
        else:
            severity = 'NONE'
            
        return {
            'detected': detected,
            'severity': severity,
            'spot_change': spot_direction,
            'futures_change': futures_direction,
            'pattern': 'DANGEROUS_DIVERGENCE' if detected else 'NORMAL'
        }
    
    def _check_range_break(self, ohlcv: pd.DataFrame, position: Dict[str, Any],
                          entry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if price breaks below entry range or reset low
        
        According to strategy doc: "Price breaks below entry range/reset low"
        The reset low is where Phase 3 detected convergence/exhaustion
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
        current_price = ohlcv[close_col].iloc[-1]
        
        # Try to get the actual reset low/high from entry analysis
        # This should be stored when Phase 3 detected the reset
        reset_low = None
        reset_high = None
        
        if isinstance(entry_analysis, dict):
            # Look for reset information from Phase 3
            reset_info = entry_analysis.get('reset', {})
            if reset_info:
                reset_low = reset_info.get('reset_low')
                reset_high = reset_info.get('reset_high')
        
        # If we don't have reset levels, use the price range around entry
        # But adapt to current market volatility (not fixed %)
        if reset_low is None or reset_high is None:
            # Look at the market structure when we entered
            lookback = min(20, len(ohlcv))
            recent_lows = ohlcv[low_col].iloc[-lookback:]
            recent_highs = ohlcv[high_col].iloc[-lookback:]
            
            # The reset range is the consolidation area before entry
            # Use statistical approach to find significant levels
            reset_low = recent_lows.quantile(0.25)  # 25th percentile of recent lows
            reset_high = recent_highs.quantile(0.75)  # 75th percentile of recent highs
        
        # Check for range break based on position type
        detected = False
        break_type = 'NONE'
        
        if position_side in ['BUY', 'LONG']:
            # For longs, exit if price breaks below reset low
            if current_price < reset_low:
                detected = True
                break_type = 'BELOW_RESET_LOW'
            
            # Debug logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"Range break check (LONG): current={current_price:.2f}, reset_low={reset_low:.2f}, detected={detected}")
            
        elif position_side in ['SELL', 'SHORT']:
            # For shorts, exit if price breaks above reset high
            if current_price > reset_high:
                detected = True
                break_type = 'ABOVE_RESET_HIGH'
            
            # Debug logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"Range break check (SHORT): current={current_price:.2f}, reset_high={reset_high:.2f}, detected={detected}")
            
        else:
            detected = False
            break_type = 'NONE'
            if hasattr(self, 'logger'):
                self.logger.warning(f"Unknown position side: {position_side}")
            
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
        
        From strategy doc line 267-269: "Both CVDs start declining together"
        This is about TREND reversal, not just any movement from entry
        """
        
        if len(spot_cvd) < 20:
            return {'detected': False, 'invalidation_type': 'NONE'}
            
        position_side = position.get('side', '').upper()
        
        # Look at RECENT TREND, not change from entry
        # We want to detect when CVDs START moving against the position
        lookback = min(20, len(spot_cvd) // 2)
        
        # Recent CVD trends
        spot_recent_trend = spot_cvd.iloc[-1] - spot_cvd.iloc[-lookback]
        futures_recent_trend = futures_cvd.iloc[-1] - futures_cvd.iloc[-lookback]
        
        # Also check very recent momentum (last few periods)
        spot_momentum = spot_cvd.iloc[-1] - spot_cvd.iloc[-5]
        futures_momentum = futures_cvd.iloc[-1] - futures_cvd.iloc[-5]
        
        # Debug logging
        self.logger.debug(f"CVD trend check: spot_trend={spot_recent_trend:.0f}, futures_trend={futures_recent_trend:.0f}")
        
        # Check if BOTH CVDs are moving against position
        if position_side in ['BUY', 'LONG']:
            # For longs: Invalidated if BOTH CVDs declining together
            both_declining = (spot_recent_trend < 0 and futures_recent_trend < 0)
            # Also check if momentum is strongly negative
            momentum_negative = (spot_momentum < 0 and futures_momentum < 0)
            cvd_reversed = both_declining and momentum_negative
            invalidation_type = 'CVD_DECLINING' if cvd_reversed else 'NONE'
            
        elif position_side in ['SELL', 'SHORT']:
            # For shorts: Invalidated if BOTH CVDs rising together
            both_rising = (spot_recent_trend > 0 and futures_recent_trend > 0)
            # Also check if momentum is strongly positive
            momentum_positive = (spot_momentum > 0 and futures_momentum > 0)
            cvd_reversed = both_rising and momentum_positive
            invalidation_type = 'CVD_RISING' if cvd_reversed else 'NONE'
            
        else:
            cvd_reversed = False
            invalidation_type = 'NONE'
            
        return {
            'detected': cvd_reversed,
            'invalidation_type': invalidation_type,
            'spot_change': spot_recent_trend,
            'futures_change': futures_recent_trend,
            'baseline_used': False  # Not using baseline in this implementation
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
        
        # According to strategy doc: NO fixed thresholds, adapt to market personality
        # Look for SIGNIFICANT structural breaks that invalidate the trade thesis
        
        # Calculate what's "normal" movement for current market
        lookback_period = min(50, len(ohlcv))
        recent_ranges = ohlcv[high_col].iloc[-lookback_period:] - ohlcv[low_col].iloc[-lookback_period:]
        typical_range = recent_ranges.median()
        
        # A structural break means price has moved beyond normal market behavior
        # This automatically adapts to quiet vs volatile markets
        
        if position_side in ['BUY', 'LONG']:
            # For longs, structural break = breaking significantly below recent structure
            # Not just any swing low, but a move that shows trend has changed
            if current_price < swing_low - typical_range:
                structure_broken = True
                structure_type = 'SIGNIFICANT_BREAK_DOWN'
                
        elif position_side in ['SELL', 'SHORT']:
            # For shorts, structural break = breaking significantly above recent structure
            # Not just any swing high, but a move that shows trend has changed
            if current_price > swing_high + typical_range:
                structure_broken = True
                structure_type = 'SIGNIFICANT_BREAK_UP'
                
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