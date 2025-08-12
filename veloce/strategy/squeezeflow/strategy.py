"""Main SqueezeFlow strategy implementation using Veloce architecture"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from veloce.core import (
    CONFIG,
    Strategy,
    Signal,
    Position,
    SignalAction,
    StrategyMode
)
from veloce.data import DATA
from veloce.strategy.squeezeflow.indicators import Indicators
from veloce.strategy.squeezeflow.phases import FivePhaseAnalyzer
from veloce.strategy.squeezeflow.signals import SignalGenerator

logger = logging.getLogger(__name__)


class SqueezeFlowStrategy(Strategy):
    """
    SqueezeFlow Strategy - 5-phase implementation
    
    This is THE strategy implementation that replaces the old fragmented system.
    All phases use central configuration, single data provider, and clean interfaces.
    """
    
    def __init__(self, config=CONFIG, data_provider=DATA):
        """
        Initialize strategy with configuration and data provider
        
        Args:
            config: VeloceConfig instance
            data_provider: VeloceDataProvider instance
        """
        self.config = config
        self.data = data_provider
        self.mode = StrategyMode.PRODUCTION
        
        # Initialize components
        self.indicators = Indicators(config)
        self.phase_analyzer = FivePhaseAnalyzer(config)
        self.signal_generator = SignalGenerator(config)
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info(f"SqueezeFlow Strategy initialized")
    
    def analyze(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_hours: Optional[int] = None
    ) -> Optional[Signal]:
        """
        Main strategy analysis method
        
        Args:
            symbol: Trading symbol to analyze
            timestamp: Current timestamp
            lookback_hours: Hours of history to analyze
            
        Returns:
            Trading signal or None
        """
        try:
            logger.debug(f"Analyzing {symbol} at {timestamp}")
            
            # Check if we have an open position
            if symbol in self.positions:
                return self._manage_position(symbol, timestamp)
            
            # Check minimum time between signals
            if not self._can_generate_signal(symbol, timestamp):
                return None
            
            # Get multi-timeframe data
            lookback = lookback_hours or self.config.default_lookback_hours
            start_time = timestamp - timedelta(hours=lookback)
            
            mtf_data = self.data.get_multi_timeframe(
                symbol=symbol,
                timestamp=timestamp,
                lookback_hours=lookback
            )
            
            if not mtf_data:
                logger.warning(f"No multi-timeframe data available for {symbol}")
                return None
            
            # Calculate indicators for each timeframe
            mtf_indicators = {}
            for timeframe, df in mtf_data.items():
                if not df.empty:
                    # Calculate all indicators
                    df_with_indicators = self.indicators.calculate_all(df)
                    # Extract signals
                    mtf_indicators[timeframe] = self.indicators.get_indicator_signals(df_with_indicators)
                    # Update original dataframe
                    mtf_data[timeframe] = df_with_indicators
            
            # Get CVD data
            cvd_data = self.data.get_cvd(
                symbol=symbol,
                start_time=start_time,
                end_time=timestamp
            )
            
            # Get OI data if enabled
            oi_data = None
            if self.config.oi_enabled:
                oi_data = self.data.get_oi(symbol)
            
            # Calculate market structure
            primary_df = mtf_data.get(self.config.primary_timeframe, pd.DataFrame())
            market_structure = self.indicators.calculate_market_structure(primary_df)
            
            # Run all phases
            phase_results = self.phase_analyzer.run_all_phases(
                mtf_data=mtf_data,
                mtf_indicators=mtf_indicators,
                cvd_data=cvd_data,
                oi_data=oi_data,
                market_structure=market_structure
            )
            
            # Generate signal if phases pass
            signal = self.signal_generator.create_signal(
                symbol=symbol,
                phase_results=phase_results,
                market_data=primary_df,
                timestamp=timestamp
            )
            
            if signal:
                self.last_signal_time[symbol] = timestamp
                logger.info(f"Signal generated for {symbol}: {signal.action}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None
    
    def _manage_position(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Optional[Signal]:
        """
        Manage existing position (Phase 5)
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            
        Returns:
            Exit signal or None
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                return None
            
            # Get current market data
            current_data = self.data.get_ohlcv(
                symbol=symbol,
                timeframe=self.config.primary_timeframe,
                start_time=timestamp - timedelta(minutes=5),
                end_time=timestamp
            )
            
            if current_data.empty:
                logger.warning(f"No current data for position management of {symbol}")
                return None
            
            current_price = float(current_data['close'].iloc[-1])
            
            # Calculate time in position
            time_in_position = int((timestamp - position.entry_time).total_seconds() / 60)
            
            # Get current indicators
            mtf_data = self.data.get_multi_timeframe(
                symbol=symbol,
                timestamp=timestamp,
                lookback_hours=1
            )
            
            mtf_indicators = {}
            for timeframe, df in mtf_data.items():
                if not df.empty:
                    df_with_indicators = self.indicators.calculate_all(df)
                    mtf_indicators[timeframe] = self.indicators.get_indicator_signals(df_with_indicators)
            
            # Run Phase 5 exit management
            phase5_result = self.phase_analyzer.phase5_exit_management(
                entry_price=position.entry_price,
                current_price=current_price,
                position_side=position.side,
                mtf_indicators=mtf_indicators,
                time_in_position=time_in_position
            )
            
            # Generate exit signal if needed
            if phase5_result['passed']:
                signal = self.signal_generator.create_exit_signal(
                    symbol=symbol,
                    phase5_result=phase5_result,
                    position=position,
                    timestamp=timestamp
                )
                
                if signal:
                    # Remove position
                    del self.positions[symbol]
                    logger.info(f"Exit signal for {symbol}: {phase5_result['data'].get('exit_reason')}")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error managing position for {symbol}: {e}")
            return None
    
    def _can_generate_signal(self, symbol: str, timestamp: datetime) -> bool:
        """Check if enough time has passed since last signal"""
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = (timestamp - self.last_signal_time[symbol]).total_seconds()
        min_interval = self.config.min_signal_interval * 60  # Convert to seconds
        
        return time_since_last >= min_interval
    
    def backtest(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        initial_balance: float = 10000
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading symbol
            start_time: Backtest start time
            end_time: Backtest end time
            initial_balance: Starting balance
            
        Returns:
            Backtest results
        """
        self.mode = StrategyMode.BACKTEST
        logger.info(f"Starting backtest for {symbol} from {start_time} to {end_time}")
        
        # Reset state
        self.positions.clear()
        self.last_signal_time.clear()
        self.signal_generator.active_signals.clear()
        self.signal_generator.signal_history.clear()
        
        # Track performance
        balance = initial_balance
        trades = []
        equity_curve = []
        
        # Get data for entire period
        primary_df = self.data.get_ohlcv(
            symbol=symbol,
            timeframe=self.config.primary_timeframe,
            start_time=start_time - timedelta(hours=self.config.default_lookback_hours),
            end_time=end_time
        )
        
        if primary_df.empty:
            logger.error("No data available for backtest period")
            return {
                'error': 'No data available',
                'trades': 0,
                'final_balance': initial_balance
            }
        
        # Iterate through each timestamp
        for i in range(self.config.default_lookback_hours * 60, len(primary_df)):
            timestamp = primary_df.index[i]
            current_price = float(primary_df['close'].iloc[i])
            
            # Analyze for signals
            signal = self.analyze(symbol, timestamp)
            
            if signal:
                # Process signal
                if signal.action in [SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT]:
                    # Open position
                    position = Position(
                        symbol=symbol,
                        side='LONG' if signal.action == SignalAction.OPEN_LONG else 'SHORT',
                        entry_price=signal.price,
                        quantity=signal.quantity,
                        entry_time=timestamp
                    )
                    self.positions[symbol] = position
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'OPEN',
                        'side': position.side,
                        'price': signal.price,
                        'quantity': signal.quantity,
                        'confidence': signal.confidence
                    })
                
                elif signal.action in [SignalAction.CLOSE_LONG, SignalAction.CLOSE_SHORT]:
                    # Close position
                    pnl_pct = signal.metadata.get('pnl_pct', 0)
                    pnl = balance * (pnl_pct / 100)
                    balance += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'CLOSE',
                        'price': signal.price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': signal.metadata.get('exit_reason')
                    })
            
            # Track equity
            equity_curve.append({
                'timestamp': timestamp,
                'balance': balance,
                'price': current_price
            })
        
        # Calculate final metrics
        total_trades = len([t for t in trades if t['action'] == 'CLOSE'])
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
        
        total_pnl = balance - initial_balance
        total_return = (total_pnl / initial_balance) * 100
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        equity_values = [e['balance'] for e in equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0
        
        results = {
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time,
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_drawdown_pct': max_dd,
            'trades': trades,
            'equity_curve': equity_curve,
            'signals_generated': len(self.signal_generator.signal_history)
        }
        
        logger.info(f"Backtest complete: {total_trades} trades, {total_return:.2f}% return, {win_rate:.2%} win rate")
        
        self.mode = StrategyMode.LIVE
        return results
    
    def optimize(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        param_ranges: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        
        Args:
            symbol: Trading symbol
            start_time: Optimization start time
            end_time: Optimization end time
            param_ranges: Parameter ranges to test
            
        Returns:
            Optimization results
        """
        self.mode = StrategyMode.OPTIMIZE
        logger.info(f"Starting optimization for {symbol}")
        
        best_params = {}
        best_score = -float('inf')
        results = []
        
        # Simple grid search (can be enhanced with more sophisticated methods)
        # For now, just return a placeholder
        logger.warning("Optimization not fully implemented yet")
        
        self.mode = StrategyMode.PRODUCTION
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        return {
            'mode': self.mode.value,
            'positions': {
                symbol: {
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'quantity': pos.quantity,
                    'entry_time': pos.entry_time.isoformat()
                }
                for symbol, pos in self.positions.items()
            },
            'active_signals': len(self.signal_generator.active_signals),
            'total_signals': len(self.signal_generator.signal_history),
            'performance': self.signal_generator.calculate_signal_performance()
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.positions.clear()
        self.last_signal_time.clear()
        self.signal_generator.active_signals.clear()
        self.signal_generator.signal_history.clear()
        self.performance_metrics.clear()
        logger.info("Strategy state reset")