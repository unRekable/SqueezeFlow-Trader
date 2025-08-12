"""Signal generation and management for SqueezeFlow strategy"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from veloce.core import (
    CONFIG,
    Signal,
    SignalAction,
    OrderSide,
    OrderType
)
from veloce.core.exceptions import SignalValidationError

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate and validate trading signals"""
    
    def __init__(self, config=CONFIG):
        """Initialize signal generator"""
        self.config = config
        self.active_signals: List[Signal] = []
        self.signal_history: List[Signal] = []
        
    def create_signal(
        self,
        symbol: str,
        phase_results: Dict[str, Any],
        market_data: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Create a trading signal from phase results
        
        Args:
            symbol: Trading symbol
            phase_results: Results from all phases
            market_data: Current market data
            timestamp: Signal timestamp
            
        Returns:
            Signal object or None
        """
        try:
            # Check if Phase 4 passed (final confirmation)
            phase4 = phase_results.get('phase4', {})
            if not phase4.get('passed', False):
                logger.debug(f"No signal - Phase 4 did not pass: {phase4.get('reason')}")
                return None
            
            # Extract signal direction
            signal_direction = phase4['data'].get('signal_direction')
            if not signal_direction:
                logger.warning("Phase 4 passed but no signal direction")
                return None
            
            # Get current price
            if market_data.empty:
                logger.error("Cannot create signal - no market data")
                return None
            
            current_price = float(market_data['close'].iloc[-1])
            
            # Determine order side
            if signal_direction == 'BUY':
                side = OrderSide.BUY
                position_side = 'LONG'
            elif signal_direction == 'SELL':
                side = OrderSide.SELL
                position_side = 'SHORT'
            else:
                logger.error(f"Invalid signal direction: {signal_direction}")
                return None
            
            # Calculate position size based on risk
            position_size = self._calculate_position_size(
                current_price,
                self.config.position_size,
                self.config.max_position_size
            )
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_levels(
                current_price,
                side,
                self.config.stop_loss_pct,
                phase4['data'].get('weighted_score', 50)
            )
            
            # Build signal metadata
            metadata = {
                'phase1_score': phase_results['phase1']['score'],
                'phase2_score': phase_results['phase2']['score'],
                'phase3_score': phase_results['phase3']['score'],
                'phase4_score': phase_results['phase4']['score'],
                'weighted_score': phase4['data']['weighted_score'],
                'divergence_type': phase_results['phase2']['data'].get('divergence_type'),
                'squeeze_active': phase_results['phase1']['data'].get('squeeze_active'),
                'mtf_aligned': phase_results['phase1']['data'].get('mtf_aligned'),
                'reset_type': phase_results['phase3']['data'].get('reset_type'),
                'market_trend': phase4['data'].get('market_trend'),
                'reasons': self._compile_reasons(phase_results)
            }
            
            # Create signal
            signal = Signal(
                id=f"{symbol}_{timestamp.timestamp():.0f}",
                timestamp=timestamp,
                symbol=symbol,
                action=SignalAction.OPEN_LONG if side == OrderSide.BUY else SignalAction.OPEN_SHORT,
                side=side,
                order_type=OrderType.MARKET,
                price=current_price,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=phase4['data']['weighted_score'] / 100.0,  # Normalize to 0-1
                metadata=metadata
            )
            
            # Validate signal
            self._validate_signal(signal)
            
            # Add to active signals
            self.active_signals.append(signal)
            self.signal_history.append(signal)
            
            logger.info(f"Signal generated: {signal.action} {signal.symbol} @ {signal.price:.2f}, confidence: {signal.confidence:.2%}")
            return signal
            
        except Exception as e:
            logger.error(f"Error creating signal: {e}")
            return None
    
    def create_exit_signal(
        self,
        symbol: str,
        phase5_result: Dict[str, Any],
        position: Any,
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Create an exit signal from Phase 5 results
        
        Args:
            symbol: Trading symbol
            phase5_result: Phase 5 exit management results
            position: Current position
            timestamp: Signal timestamp
            
        Returns:
            Exit signal or None
        """
        try:
            if not phase5_result.get('passed', False):
                return None
            
            # Determine exit action
            if position.side == 'LONG':
                action = SignalAction.CLOSE_LONG
                side = OrderSide.SELL
            else:  # SHORT
                action = SignalAction.CLOSE_SHORT
                side = OrderSide.BUY
            
            # Get exit data
            exit_data = phase5_result['data']
            
            # Create exit signal
            signal = Signal(
                id=f"{symbol}_exit_{timestamp.timestamp():.0f}",
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                side=side,
                order_type=OrderType.MARKET,
                price=exit_data['current_price'],
                quantity=position.quantity,
                stop_loss=None,
                take_profit=None,
                confidence=1.0,  # Exit signals have full confidence
                metadata={
                    'exit_reason': exit_data.get('exit_reason', 'unknown'),
                    'pnl_pct': exit_data.get('pnl_pct', 0),
                    'time_in_position': exit_data.get('time_in_position', 0),
                    'entry_price': exit_data.get('entry_price'),
                    'reason': phase5_result.get('reason', '')
                }
            )
            
            # Remove from active signals
            self.active_signals = [s for s in self.active_signals if s.symbol != symbol]
            self.signal_history.append(signal)
            
            logger.info(f"Exit signal: {action} {symbol} @ {signal.price:.2f}, reason: {exit_data.get('exit_reason')}, P&L: {exit_data.get('pnl_pct', 0):.2f}%")
            return signal
            
        except Exception as e:
            logger.error(f"Error creating exit signal: {e}")
            return None
    
    def _calculate_position_size(
        self,
        price: float,
        base_size: float,
        max_size: float
    ) -> float:
        """Calculate position size based on risk parameters"""
        # Simple fixed position size for now
        # Could be enhanced with Kelly Criterion or other sizing methods
        size = min(base_size, max_size)
        
        # Ensure minimum viable position
        min_position_value = 10  # $10 minimum
        if price * size < min_position_value:
            size = min_position_value / price
        
        return size
    
    def _calculate_levels(
        self,
        entry_price: float,
        side: OrderSide,
        stop_loss_pct: float,
        score: float
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        # Dynamic risk/reward based on signal strength
        # Higher score = tighter stop, larger target
        risk_multiplier = 1.0 - (score / 200)  # 0.5 to 1.0
        reward_multiplier = 1.0 + (score / 100)  # 1.0 to 2.0
        
        adjusted_stop = stop_loss_pct * risk_multiplier
        adjusted_target = stop_loss_pct * reward_multiplier * 2  # 2:1 base R:R
        
        if side == OrderSide.BUY:
            stop_loss = entry_price * (1 - adjusted_stop)
            take_profit = entry_price * (1 + adjusted_target)
        else:  # SELL
            stop_loss = entry_price * (1 + adjusted_stop)
            take_profit = entry_price * (1 - adjusted_target)
        
        return stop_loss, take_profit
    
    def _compile_reasons(self, phase_results: Dict[str, Any]) -> List[str]:
        """Compile reasons from all phases"""
        reasons = []
        
        for phase_name in ['phase1', 'phase2', 'phase3', 'phase4']:
            if phase_name in phase_results:
                phase_reason = phase_results[phase_name].get('reason', '')
                if phase_reason:
                    reasons.append(f"{phase_name}: {phase_reason}")
        
        return reasons
    
    def _validate_signal(self, signal: Signal) -> None:
        """Validate signal parameters"""
        
        # Price validation
        if signal.price <= 0:
            raise SignalValidationError(f"Invalid price: {signal.price}")
        
        # Quantity validation
        if signal.quantity <= 0:
            raise SignalValidationError(f"Invalid quantity: {signal.quantity}")
        
        # Stop loss validation
        if signal.stop_loss:
            if signal.action in [SignalAction.OPEN_LONG]:
                if signal.stop_loss >= signal.price:
                    raise SignalValidationError("Stop loss must be below entry for long")
            elif signal.action in [SignalAction.OPEN_SHORT]:
                if signal.stop_loss <= signal.price:
                    raise SignalValidationError("Stop loss must be above entry for short")
        
        # Take profit validation
        if signal.take_profit:
            if signal.action in [SignalAction.OPEN_LONG]:
                if signal.take_profit <= signal.price:
                    raise SignalValidationError("Take profit must be above entry for long")
            elif signal.action in [SignalAction.OPEN_SHORT]:
                if signal.take_profit >= signal.price:
                    raise SignalValidationError("Take profit must be below entry for short")
        
        # Confidence validation
        if not 0 <= signal.confidence <= 1:
            raise SignalValidationError(f"Invalid confidence: {signal.confidence}")
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Get active signals, optionally filtered by symbol"""
        if symbol:
            return [s for s in self.active_signals if s.symbol == symbol]
        return self.active_signals.copy()
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Signal]:
        """Get signal history"""
        history = self.signal_history
        
        if symbol:
            history = [s for s in history if s.symbol == symbol]
        
        # Return most recent signals
        return history[-limit:] if len(history) > limit else history
    
    def clear_expired_signals(self, max_age_minutes: int = 60) -> None:
        """Remove old signals that were never executed"""
        now = datetime.now()
        
        self.active_signals = [
            s for s in self.active_signals
            if (now - s.timestamp).total_seconds() < max_age_minutes * 60
        ]
    
    def calculate_signal_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics for generated signals"""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'win_rate': 0,
                'avg_confidence': 0,
                'signals_by_action': {}
            }
        
        # Count signals by action
        signals_by_action = {}
        for signal in self.signal_history:
            action = signal.action.value
            signals_by_action[action] = signals_by_action.get(action, 0) + 1
        
        # Calculate win rate from exit signals
        exits = [s for s in self.signal_history if 'CLOSE' in s.action.value]
        wins = [s for s in exits if s.metadata.get('pnl_pct', 0) > 0]
        win_rate = len(wins) / len(exits) if exits else 0
        
        # Average confidence
        avg_confidence = np.mean([s.confidence for s in self.signal_history])
        
        return {
            'total_signals': len(self.signal_history),
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'signals_by_action': signals_by_action,
            'active_signals': len(self.active_signals)
        }