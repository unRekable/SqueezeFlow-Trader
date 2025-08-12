"""All 5 phases of SqueezeFlow strategy in one file"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

from veloce.core.config import VeloceConfig, CONFIG
from veloce.core.types import PhaseResult

logger = logging.getLogger(__name__)


class FivePhaseAnalyzer:
    """All 5 phases integrated with central configuration"""
    
    def __init__(self, config: VeloceConfig = CONFIG):
        """Initialize with configuration"""
        self.config = config
        self.phase_config = config.get_indicator_config()['phases']
        
    def phase1_context_and_squeeze(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_indicators: Dict[str, Dict[str, Any]]
    ) -> PhaseResult:
        """
        Phase 1: Context Analysis & Squeeze Detection
        - Check for squeeze across timeframes
        - Analyze volume patterns
        - Determine market context
        """
        logger.debug("Phase 1: Context & Squeeze Analysis")
        
        try:
            phase1_config = self.phase_config['phase1']
            score = 0.0
            reasons = []
            data = {}
            
            # Check squeeze on primary timeframe
            primary_tf = self.config.primary_timeframe
            if primary_tf in mtf_indicators:
                primary_signals = mtf_indicators[primary_tf]
                
                # Squeeze detection
                if 'squeeze' in primary_signals:
                    squeeze_data = primary_signals['squeeze']
                    if squeeze_data.get('active', False):
                        score += 30
                        reasons.append(f"Squeeze active on {primary_tf}")
                        data['squeeze_active'] = True
                        
                        # Check momentum direction
                        if squeeze_data.get('momentum_increasing', False):
                            score += 10
                            reasons.append("Momentum increasing")
                    else:
                        data['squeeze_active'] = False
            
            # Multi-timeframe alignment check
            aligned_timeframes = 0
            squeeze_timeframes = []
            
            for tf, indicators in mtf_indicators.items():
                if 'squeeze' in indicators and indicators['squeeze'].get('active', False):
                    squeeze_timeframes.append(tf)
                    aligned_timeframes += 1
            
            if aligned_timeframes >= self.config.mtf_alignment_required:
                score += 20
                reasons.append(f"Squeeze aligned on {aligned_timeframes} timeframes")
                data['mtf_aligned'] = True
            else:
                data['mtf_aligned'] = False
            
            data['squeeze_timeframes'] = squeeze_timeframes
            
            # Volume analysis
            if primary_tf in mtf_data:
                df = mtf_data[primary_tf]
                if not df.empty and 'volume' in df.columns:
                    # Check for volume expansion
                    recent_volume = df['volume'].tail(10).mean()
                    historical_volume = df['volume'].tail(50).mean()
                    
                    if historical_volume > 0:
                        volume_ratio = recent_volume / historical_volume
                        data['volume_ratio'] = float(volume_ratio)
                        
                        if volume_ratio > phase1_config['volume_threshold']:
                            score += 15
                            reasons.append(f"Volume expansion {volume_ratio:.2f}x")
            
            # Determine if phase passes
            passed = score >= 40  # Need at least 40 points to pass
            
            if not passed:
                reasons.append("Insufficient context signals")
            
            result: PhaseResult = {
                'passed': passed,
                'score': float(score),
                'reason': '; '.join(reasons),
                'data': data
            }
            
            logger.info(f"Phase 1 result: passed={passed}, score={score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Phase 1 error: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'reason': f"Phase 1 error: {str(e)}",
                'data': {}
            }
    
    def phase2_divergence_analysis(
        self,
        cvd_data: pd.DataFrame,
        price_data: pd.DataFrame,
        oi_data: Optional[Dict[str, float]]
    ) -> PhaseResult:
        """
        Phase 2: Divergence Analysis
        - CVD divergence (spot vs perp)
        - Price vs volume divergence
        - OI divergence
        """
        logger.debug("Phase 2: Divergence Analysis")
        
        try:
            phase2_config = self.phase_config['phase2']
            score = 0.0
            reasons = []
            data = {}
            
            # CVD Divergence
            if not cvd_data.empty and not price_data.empty:
                # Check for spot/perp divergence
                if 'cvd_divergence' in cvd_data.columns:
                    recent_divergence = cvd_data['cvd_divergence'].tail(20)
                    divergence_strength = abs(recent_divergence.mean())
                    data['cvd_divergence_strength'] = float(divergence_strength)
                    
                    if divergence_strength > phase2_config['divergence_threshold']:
                        score += 30
                        reasons.append(f"CVD divergence detected: {divergence_strength:.4f}")
                        
                        # Determine divergence direction
                        if recent_divergence.iloc[-1] > 0:
                            data['divergence_type'] = 'bullish'
                            reasons.append("Bullish divergence")
                        else:
                            data['divergence_type'] = 'bearish'
                            reasons.append("Bearish divergence")
                
                # Price vs CVD divergence
                price_change = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1) * 100
                
                if 'spot_cvd_cumulative' in cvd_data.columns:
                    cvd_change = cvd_data['spot_cvd_cumulative'].iloc[-1] - cvd_data['spot_cvd_cumulative'].iloc[-20]
                    
                    # Bullish: price down but CVD up
                    if price_change < -1 and cvd_change > 0:
                        score += 20
                        reasons.append("Bullish price/CVD divergence")
                        data['price_cvd_divergence'] = 'bullish'
                    # Bearish: price up but CVD down
                    elif price_change > 1 and cvd_change < 0:
                        score += 20
                        reasons.append("Bearish price/CVD divergence")
                        data['price_cvd_divergence'] = 'bearish'
            
            # OI Divergence
            if oi_data and self.config.oi_enabled:
                total_oi = oi_data.get('TOTAL', {})
                oi_change = total_oi.get('change', 0)
                data['oi_change'] = float(oi_change)
                
                # Significant OI change
                if abs(oi_change) > self.config.oi_threshold:
                    score += 15
                    if oi_change > 0:
                        reasons.append(f"OI increasing: {oi_change:.1f}%")
                    else:
                        reasons.append(f"OI decreasing: {oi_change:.1f}%")
                    
                    # OI vs price divergence
                    if not price_data.empty:
                        price_direction = 1 if price_data['close'].iloc[-1] > price_data['close'].iloc[-2] else -1
                        oi_direction = 1 if oi_change > 0 else -1
                        
                        if price_direction != oi_direction:
                            score += 10
                            reasons.append("OI/Price divergence")
            
            # Determine if phase passes
            passed = score >= 35  # Need at least 35 points to pass
            
            if not passed:
                reasons.append("Insufficient divergence signals")
            
            result: PhaseResult = {
                'passed': passed,
                'score': float(score),
                'reason': '; '.join(reasons),
                'data': data
            }
            
            logger.info(f"Phase 2 result: passed={passed}, score={score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Phase 2 error: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'reason': f"Phase 2 error: {str(e)}",
                'data': {}
            }
    
    def phase3_reset_detection(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_indicators: Dict[str, Dict[str, Any]],
        market_structure: Dict[str, Any]
    ) -> PhaseResult:
        """
        Phase 3: Reset Detection
        - Check for price reset/pullback
        - Analyze momentum reset
        - Confirm entry opportunity
        """
        logger.debug("Phase 3: Reset Detection")
        
        try:
            phase3_config = self.phase_config['phase3']
            score = 0.0
            reasons = []
            data = {}
            
            primary_tf = self.config.primary_timeframe
            
            # Check for momentum reset
            if primary_tf in mtf_indicators:
                indicators = mtf_indicators[primary_tf]
                
                # RSI reset check
                if 'rsi' in indicators:
                    rsi_data = indicators['rsi']
                    rsi_value = rsi_data.get('value', 50)
                    data['rsi'] = float(rsi_value)
                    
                    # Oversold bounce opportunity
                    if rsi_data.get('oversold', False):
                        score += 25
                        reasons.append(f"RSI oversold: {rsi_value:.1f}")
                        data['reset_type'] = 'oversold'
                    # Overbought pullback opportunity
                    elif rsi_data.get('overbought', False):
                        score += 25
                        reasons.append(f"RSI overbought: {rsi_value:.1f}")
                        data['reset_type'] = 'overbought'
                    # Neutral zone reset
                    elif 40 <= rsi_value <= 60:
                        score += 15
                        reasons.append("RSI neutral reset")
                        data['reset_type'] = 'neutral'
                
                # MACD reset check
                if 'macd' in indicators:
                    macd_data = indicators['macd']
                    if macd_data.get('bullish_cross', False):
                        score += 20
                        reasons.append("MACD bullish cross")
                        data['macd_cross'] = 'bullish'
                    elif macd_data.get('bearish_cross', False):
                        score += 20
                        reasons.append("MACD bearish cross")
                        data['macd_cross'] = 'bearish'
            
            # Price reset from support/resistance
            if market_structure:
                current_price = mtf_data[primary_tf]['close'].iloc[-1] if primary_tf in mtf_data and not mtf_data[primary_tf].empty else 0
                
                if current_price > 0:
                    support = market_structure.get('support', 0)
                    resistance = market_structure.get('resistance', 0)
                    
                    # Calculate distance from levels
                    if support > 0:
                        support_distance = abs(current_price - support) / support
                        if support_distance < phase3_config['reset_threshold'] * 0.01:  # Within threshold
                            score += 20
                            reasons.append(f"Near support: {support:.2f}")
                            data['near_support'] = True
                    
                    if resistance > 0:
                        resistance_distance = abs(current_price - resistance) / resistance
                        if resistance_distance < phase3_config['reset_threshold'] * 0.01:
                            score += 20
                            reasons.append(f"Near resistance: {resistance:.2f}")
                            data['near_resistance'] = True
            
            # Squeeze momentum reset
            if primary_tf in mtf_indicators:
                squeeze_data = mtf_indicators[primary_tf].get('squeeze', {})
                if squeeze_data.get('active', False):
                    momentum = squeeze_data.get('momentum', 0)
                    if abs(momentum) < 0.5:  # Near zero momentum
                        score += 15
                        reasons.append("Squeeze momentum reset")
                        data['momentum_reset'] = True
            
            # Determine if phase passes
            passed = score >= 30  # Need at least 30 points to pass
            
            if not passed:
                reasons.append("No clear reset detected")
            
            result: PhaseResult = {
                'passed': passed,
                'score': float(score),
                'reason': '; '.join(reasons),
                'data': data
            }
            
            logger.info(f"Phase 3 result: passed={passed}, score={score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Phase 3 error: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'reason': f"Phase 3 error: {str(e)}",
                'data': {}
            }
    
    def phase4_scoring_and_confirmation(
        self,
        phase1_result: PhaseResult,
        phase2_result: PhaseResult,
        phase3_result: PhaseResult,
        market_structure: Dict[str, Any]
    ) -> PhaseResult:
        """
        Phase 4: Scoring & Final Confirmation
        - Aggregate scores from all phases
        - Apply weighting
        - Final trade decision
        """
        logger.debug("Phase 4: Scoring & Confirmation")
        
        try:
            phase4_config = self.phase_config['phase4']
            
            # Weight each phase
            weights = {
                'phase1': 0.25,
                'phase2': 0.35,
                'phase3': 0.25,
                'structure': 0.15
            }
            
            # Calculate weighted score
            weighted_score = (
                phase1_result['score'] * weights['phase1'] +
                phase2_result['score'] * weights['phase2'] +
                phase3_result['score'] * weights['phase3']
            )
            
            # Add market structure bonus
            structure_score = 0
            if market_structure:
                trend = market_structure.get('trend', 'neutral')
                if trend != 'neutral':
                    structure_score = 20
                    
                    # Extra points for trend alignment
                    if phase2_result['data'].get('divergence_type') == trend:
                        structure_score += 10
            
            weighted_score += structure_score * weights['structure']
            
            # Compile reasons
            reasons = []
            data = {
                'phase1_score': phase1_result['score'],
                'phase2_score': phase2_result['score'],
                'phase3_score': phase3_result['score'],
                'structure_score': structure_score,
                'weighted_score': float(weighted_score),
                'market_trend': market_structure.get('trend', 'neutral') if market_structure else 'unknown'
            }
            
            # Determine signal direction
            signal_direction = None
            
            # Check divergence type from phase 2
            divergence = phase2_result['data'].get('divergence_type')
            reset_type = phase3_result['data'].get('reset_type')
            
            if divergence == 'bullish' or reset_type == 'oversold':
                signal_direction = 'BUY'
                reasons.append("Bullish setup confirmed")
            elif divergence == 'bearish' or reset_type == 'overbought':
                signal_direction = 'SELL'
                reasons.append("Bearish setup confirmed")
            else:
                # Use market structure as tiebreaker
                if market_structure and market_structure.get('trend') == 'bullish':
                    signal_direction = 'BUY'
                    reasons.append("Bullish trend confirmation")
                elif market_structure and market_structure.get('trend') == 'bearish':
                    signal_direction = 'SELL'
                    reasons.append("Bearish trend confirmation")
            
            data['signal_direction'] = signal_direction
            
            # Final confirmation
            passed = (
                weighted_score >= phase4_config['min_score'] and
                signal_direction is not None and
                phase1_result['passed'] and
                phase2_result['passed'] and
                phase3_result['passed']
            )
            
            if passed:
                reasons.append(f"Score {weighted_score:.1f} exceeds minimum {phase4_config['min_score']}")
            else:
                reasons.append(f"Score {weighted_score:.1f} below minimum {phase4_config['min_score']}")
            
            result: PhaseResult = {
                'passed': passed,
                'score': float(weighted_score),
                'reason': '; '.join(reasons),
                'data': data
            }
            
            logger.info(f"Phase 4 result: passed={passed}, score={weighted_score:.1f}, direction={signal_direction}")
            return result
            
        except Exception as e:
            logger.error(f"Phase 4 error: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'reason': f"Phase 4 error: {str(e)}",
                'data': {}
            }
    
    def phase5_exit_management(
        self,
        entry_price: float,
        current_price: float,
        position_side: str,
        mtf_indicators: Dict[str, Dict[str, Any]],
        time_in_position: int
    ) -> PhaseResult:
        """
        Phase 5: Exit Management
        - Calculate exit levels
        - Monitor exit conditions
        - Determine exit signal
        """
        logger.debug("Phase 5: Exit Management")
        
        try:
            phase5_config = self.phase_config['phase5']
            
            # Calculate P&L
            if position_side == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            data = {
                'entry_price': float(entry_price),
                'current_price': float(current_price),
                'pnl_pct': float(pnl_pct),
                'time_in_position': time_in_position,
                'position_side': position_side
            }
            
            reasons = []
            exit_signal = False
            
            # Stop loss check
            if pnl_pct <= -self.config.stop_loss_pct * 100:
                exit_signal = True
                reasons.append(f"Stop loss hit: {pnl_pct:.2f}%")
                data['exit_reason'] = 'stop_loss'
            
            # Take profit check
            elif pnl_pct >= self.config.stop_loss_pct * 100 * phase5_config['exit_multiplier']:
                exit_signal = True
                reasons.append(f"Take profit hit: {pnl_pct:.2f}%")
                data['exit_reason'] = 'take_profit'
            
            # Check for reversal signals
            primary_tf = self.config.primary_timeframe
            if primary_tf in mtf_indicators:
                indicators = mtf_indicators[primary_tf]
                
                # Squeeze release (momentum reversal)
                if 'squeeze' in indicators:
                    squeeze_data = indicators['squeeze']
                    momentum = squeeze_data.get('momentum', 0)
                    
                    if position_side == 'LONG' and momentum < 0:
                        exit_signal = True
                        reasons.append("Momentum turned negative")
                        data['exit_reason'] = 'momentum_reversal'
                    elif position_side == 'SHORT' and momentum > 0:
                        exit_signal = True
                        reasons.append("Momentum turned positive")
                        data['exit_reason'] = 'momentum_reversal'
                
                # RSI extremes
                if 'rsi' in indicators:
                    rsi_value = indicators['rsi'].get('value', 50)
                    if position_side == 'LONG' and rsi_value > 80:
                        exit_signal = True
                        reasons.append(f"RSI extremely overbought: {rsi_value:.1f}")
                        data['exit_reason'] = 'rsi_extreme'
                    elif position_side == 'SHORT' and rsi_value < 20:
                        exit_signal = True
                        reasons.append(f"RSI extremely oversold: {rsi_value:.1f}")
                        data['exit_reason'] = 'rsi_extreme'
            
            # Time-based exit (optional)
            max_hold_time = 1440  # 24 hours in minutes
            if time_in_position > max_hold_time:
                exit_signal = True
                reasons.append(f"Max hold time exceeded: {time_in_position} minutes")
                data['exit_reason'] = 'time_exit'
            
            # Calculate exit score
            score = 100.0 if exit_signal else 0.0
            
            result: PhaseResult = {
                'passed': exit_signal,
                'score': score,
                'reason': '; '.join(reasons) if reasons else 'No exit conditions met',
                'data': data
            }
            
            if exit_signal:
                logger.info(f"Phase 5 EXIT SIGNAL: {data.get('exit_reason')}, P&L: {pnl_pct:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Phase 5 error: {e}")
            return {
                'passed': False,
                'score': 0.0,
                'reason': f"Phase 5 error: {str(e)}",
                'data': {}
            }
    
    def run_all_phases(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        mtf_indicators: Dict[str, Dict[str, Any]],
        cvd_data: pd.DataFrame,
        oi_data: Optional[Dict[str, float]],
        market_structure: Dict[str, Any]
    ) -> Dict[str, PhaseResult]:
        """Run all phases sequentially (for new positions)"""
        
        results = {}
        
        # Phase 1: Context & Squeeze
        results['phase1'] = self.phase1_context_and_squeeze(mtf_data, mtf_indicators)
        
        # Phase 2: Divergence Analysis
        primary_tf = self.config.primary_timeframe
        price_data = mtf_data.get(primary_tf, pd.DataFrame())
        results['phase2'] = self.phase2_divergence_analysis(cvd_data, price_data, oi_data)
        
        # Phase 3: Reset Detection
        results['phase3'] = self.phase3_reset_detection(mtf_data, mtf_indicators, market_structure)
        
        # Phase 4: Scoring & Confirmation
        results['phase4'] = self.phase4_scoring_and_confirmation(
            results['phase1'],
            results['phase2'],
            results['phase3'],
            market_structure
        )
        
        # Phase 5 is only for exit management, not run here
        
        logger.info(f"All phases complete. Signal: {results['phase4']['passed']}")
        return results