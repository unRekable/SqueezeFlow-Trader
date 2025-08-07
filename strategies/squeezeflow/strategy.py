"""
SqueezeFlow Strategy - Main Orchestrator

Main strategy class that orchestrates all 5 phases of the SqueezeFlow trading methodology.
Implements the complete strategy as documented in /docs/strategy/SqueezeFlow.md

Architecture:
- Phase 1: Context Assessment (market intelligence)  
- Phase 2: Divergence Detection (setup identification)
- Phase 3: Reset Detection (entry timing)
- Phase 4: Scoring System (0-10 point decision)
- Phase 5: Exit Management (position management)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import pytz
import logging

from strategies.base import BaseStrategy
from strategies.squeezeflow.config import SqueezeFlowConfig
from strategies.squeezeflow.components.phase1_context import ContextAssessment
from strategies.squeezeflow.components.phase2_divergence import DivergenceDetection
from strategies.squeezeflow.components.phase3_reset import ResetDetection
from strategies.squeezeflow.components.phase4_scoring import ScoringSystem
from strategies.squeezeflow.components.phase5_exits import ExitManagement


class SqueezeFlowStrategy(BaseStrategy):
    """
    SqueezeFlow Strategy - Complete 5-Phase Implementation
    
    The complete SqueezeFlow trading methodology implementing:
    - NO fixed thresholds - Dynamic market adaptation
    - Pattern recognition over quantitative metrics  
    - Multi-timeframe validation across 6 timeframes
    - 10-point scoring system for objective entry criteria
    - Flow-following exits until invalidation
    """
    
    def __init__(self, config: Optional[SqueezeFlowConfig] = None):
        """
        Initialize SqueezeFlow strategy with all phase components
        
        Args:
            config: Strategy configuration (uses defaults if None)
        """
        super().__init__(name="SqueezeFlowStrategy")
        
        # Initialize configuration
        self.config = config or SqueezeFlowConfig()
        
        # Initialize all 5 phase components
        self.phase1 = ContextAssessment(context_timeframes=self.config.context_timeframes)
        self.phase2 = DivergenceDetection(divergence_timeframes=self.config.divergence_timeframes)
        self.phase3 = ResetDetection(reset_timeframes=self.config.reset_timeframes)
        self.phase4 = ScoringSystem(
            scoring_weights=self.config.scoring_weights,
            min_entry_score=self.config.min_entry_score
        )
        self.phase5 = ExitManagement()  # CVD baseline manager will be injected later
        
        # CVD baseline manager for live trading (optional)
        self.cvd_baseline_manager = None
        
        # Initialize logging
        self.logger = logging.getLogger(f"squeezeflow.{self.name}")
        
        # Track strategy state
        self.last_analysis = None
        
    def set_cvd_baseline_manager(self, cvd_baseline_manager):
        """
        Set CVD baseline manager for live trading
        
        Args:
            cvd_baseline_manager: CVDBaselineManager instance for tracking baselines
        """
        self.cvd_baseline_manager = cvd_baseline_manager
        # Pass to Phase 5 exit management
        self.phase5 = ExitManagement(cvd_baseline_manager=cvd_baseline_manager)
        
    def process(self, dataset: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main strategy processing - Execute phases based on position state
        
        Process Flow:
        - If position exists: Only run Phase 5 (exit management)
        - If no position: Run Phases 1-4 (entry evaluation)
        
        Args:
            dataset: Market data with spot_cvd, futures_cvd, ohlcv, etc.
            portfolio_state: Current portfolio with positions, cash, etc.
            
        Returns:
            Dict with orders list and phase analysis results
        """
        try:
            symbol = dataset.get('symbol', 'UNKNOWN')
            self.logger.debug(f"Processing {symbol} - Starting analysis")
            
            # Initialize results structure
            results = {
                'orders': [],
                'phase_results': {},
                'metadata': {
                    'strategy': self.name,
                    'symbol': symbol,
                    'timestamp': datetime.now(tz=pytz.UTC),
                    'config': self.config.__dict__
                }
            }
            
            # Check for existing positions FIRST
            existing_positions = self._get_symbol_positions(portfolio_state, symbol)
            
            if existing_positions:
                # If we have positions, ONLY run Phase 5 (exit management)
                self.logger.debug(f"{symbol}: Position exists - running exit management only")
                
                # We need minimal context for exit decisions
                # Phase 5 uses entry_analysis from when position was opened
                context_result = {'market_bias': 'NEUTRAL', 'squeeze_type': 'UNKNOWN'}
                results['phase_results']['phase1_context'] = {'skipped': True, 'reason': 'Position exists'}
                results['phase_results']['phase2_divergence'] = {'skipped': True, 'reason': 'Position exists'}
                results['phase_results']['phase3_reset'] = {'skipped': True, 'reason': 'Position exists'}
                results['phase_results']['phase4_scoring'] = {'skipped': True, 'reason': 'Position exists'}
                
                # Phase 5: Exit Management for existing positions
                results = self._handle_existing_positions(
                    results, dataset, existing_positions, context_result
                )
            else:
                # No positions - run full entry evaluation (Phases 1-4)
                self.logger.debug(f"{symbol}: No position - running full entry evaluation")
                
                # Phase 1: Context Assessment (Market Intelligence)
                context_result = self.phase1.assess_context(dataset)
                results['phase_results']['phase1_context'] = context_result
                
                if context_result.get('error'):
                    self.logger.warning(f"Phase 1 error: {context_result['error']}")
                    return results
                    
                # Phase 2: Divergence Detection (Setup Identification)  
                divergence_result = self.phase2.detect_divergence(dataset, context_result)
                results['phase_results']['phase2_divergence'] = divergence_result
                
                if divergence_result.get('error'):
                    self.logger.warning(f"Phase 2 error: {divergence_result['error']}")
                    return results
                    
                # Phase 3: Reset Detection (Entry Timing)
                reset_result = self.phase3.detect_reset(dataset, context_result, divergence_result)
                results['phase_results']['phase3_reset'] = reset_result
                
                if reset_result.get('error'):
                    self.logger.warning(f"Phase 3 error: {reset_result['error']}")
                    return results
                
                # Phase 4: Scoring System (New Entry Decision)
                scoring_result = self.phase4.calculate_score(
                    context_result, divergence_result, reset_result, dataset
                )
                results['phase_results']['phase4_scoring'] = scoring_result
                
                if scoring_result.get('error'):
                    self.logger.warning(f"Phase 4 error: {scoring_result['error']}")
                    return results
                
                # Generate orders if score >= minimum threshold
                if scoring_result.get('should_trade', False):
                    orders = self._generate_entry_orders(
                        scoring_result, dataset, portfolio_state
                    )
                    results['orders'].extend(orders)
                    
                    # Enhanced logging for debugging
                    setup_type = divergence_result.get('setup_type', 'UNKNOWN')
                    market_bias = context_result.get('market_bias', 'NEUTRAL')
                    
                    self.logger.info(
                        f"{symbol}: Entry signal - Score: {scoring_result.get('total_score', 0):.1f}, "
                        f"Direction: {scoring_result.get('direction', 'NONE')}, "
                        f"Setup: {setup_type}, Bias: {market_bias}, Orders: {len(orders)}"
                    )
                    
                    # Debug log for NONE direction cases
                    if scoring_result.get('direction') == 'NONE' and scoring_result.get('total_score', 0) >= 4.0:
                        self.logger.warning(
                            f"{symbol}: Direction NONE with good score! "
                            f"Setup: {setup_type}, Pattern: {divergence_result.get('cvd_patterns', {}).get('pattern', 'UNKNOWN')}, "
                            f"Spot dir: {divergence_result.get('cvd_patterns', {}).get('spot_direction', 0)}"
                        )
                else:
                    self.logger.debug(
                        f"{symbol}: No entry - Score: {scoring_result.get('total_score', 0):.1f} "
                        f"(min: {self.config.min_entry_score})"
                    )
            
            # Store analysis for future reference
            self.last_analysis = results
            
            return results
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Strategy processing error: {str(e)}")
            self.logger.error(f"Stack trace: {error_details}")
            
            # Debug info for the str/dict issue
            print(f"DEBUG - Error: {e}")
            print(f"DEBUG - Error type: {type(e)}")
            print(f"DEBUG - Stack trace: {error_details}")
            
            return {
                'orders': [],
                'phase_results': {},
                'error': f'Strategy processing failed: {str(e)}',
                'metadata': {'strategy': self.name, 'symbol': dataset.get('symbol', 'UNKNOWN')}
            }
    
    def _get_symbol_positions(self, portfolio_state: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        """Get existing positions for the symbol"""
        positions = portfolio_state.get('positions', [])
        return [pos for pos in positions if pos.get('symbol') == symbol and pos.get('quantity', 0) != 0]
    
    def _handle_existing_positions(self, results: Dict[str, Any], dataset: Dict[str, Any], 
                                 positions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exit management for existing positions"""
        
        for position in positions:
            # Phase 5: Exit Management
            exit_result = self.phase5.manage_exits(
                dataset, position, self.last_analysis or {}
            )
            results['phase_results'][f'phase5_exit_{position.get("id", "unknown")}'] = exit_result
            
            if exit_result.get('should_exit', False):
                # Generate exit order
                exit_order = self._generate_exit_order(position, dataset, exit_result)
                if exit_order:
                    results['orders'].append(exit_order)
                    
                    self.logger.info(
                        f"{dataset.get('symbol')}: Exit signal - "
                        f"Reason: {exit_result.get('exit_reasoning', 'Unknown')}, "
                        f"Position: {position.get('side', 'UNKNOWN')} {position.get('quantity', 0)}"
                    )
        
        return results
    
    def _generate_entry_orders(self, scoring_result: Dict[str, Any], dataset: Dict[str, Any], 
                             portfolio_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate entry orders based on scoring results"""
        
        orders = []
        
        try:
            symbol = dataset.get('symbol', 'UNKNOWN')
            direction = scoring_result.get('direction', 'NONE')
            total_score = scoring_result.get('total_score', 0)
            
            if direction == 'NONE' or total_score < self.config.min_entry_score:
                return orders
            
            # Calculate position sizing based on score
            position_size_factor = self.config.get_position_size_factor(total_score)
            leverage = self.config.get_leverage(total_score)
            
            # Base position size from portfolio
            total_value = portfolio_state.get('total_value', 0)
            base_risk = total_value * self.config.base_risk_per_trade
            position_size = base_risk * position_size_factor
            
            if position_size <= 0:
                self.logger.warning(f"Invalid position size calculated: {position_size}")
                return orders
            
            # Get current price for entry
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            if ohlcv.empty:
                self.logger.warning("No OHLCV data available for entry price")
                return orders
                
            # Safe price access
            if 'close' in ohlcv.columns:
                current_price = ohlcv['close'].iloc[-1]
            elif len(ohlcv.columns) > 3:
                current_price = ohlcv.iloc[-1, 3]  # Use positional access safely
            else:
                self.logger.warning("Cannot determine current price from OHLCV data")
                return orders
            
            # Convert to base quantity (before leverage)
            base_quantity = position_size / current_price
            
            # Get current CVD values for baseline tracking
            spot_cvd_current = None
            futures_cvd_current = None
            if 'spot_cvd' in dataset and 'futures_cvd' in dataset:
                spot_cvd = dataset['spot_cvd']
                futures_cvd = dataset['futures_cvd']
                if not spot_cvd.empty and not futures_cvd.empty:
                    spot_cvd_current = float(spot_cvd.iloc[-1])
                    futures_cvd_current = float(futures_cvd.iloc[-1])

            # Create order
            order = {
                'symbol': symbol,
                'side': 'BUY' if direction == 'LONG' else 'SELL',
                'quantity': base_quantity,
                'price': current_price,
                'timestamp': datetime.now(tz=pytz.UTC),
                'leverage': leverage,
                'signal_type': scoring_result.get('signal_quality', 'UNKNOWN'),
                'confidence': scoring_result.get('confidence', 0),
                'score': total_score,
                'reasoning': scoring_result.get('reasoning', 'SqueezeFlow entry signal')
            }
            
            # Add CVD data for baseline tracking if available
            if spot_cvd_current is not None and futures_cvd_current is not None:
                order['spot_cvd'] = spot_cvd_current
                order['futures_cvd'] = futures_cvd_current
            
            # NO fixed stop loss or take profit - dynamic exits only
            # This follows the SqueezeFlow methodology of flow-following
            
            orders.append(order)
            
        except Exception as e:
            self.logger.error(f"Error generating entry orders: {str(e)}")
            
        return orders
    
    def _generate_exit_order(self, position: Dict[str, Any], dataset: Dict[str, Any], 
                           exit_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate exit order for existing position"""
        
        try:
            symbol = dataset.get('symbol', 'UNKNOWN')
            position_side = position.get('side', 'UNKNOWN')
            position_quantity = abs(position.get('quantity', 0))
            
            if position_quantity <= 0:
                return None
            
            # Get current price
            ohlcv = dataset.get('ohlcv', pd.DataFrame())
            if ohlcv.empty:
                return None
                
            # Safe price access
            if 'close' in ohlcv.columns:
                current_price = ohlcv['close'].iloc[-1]
            elif len(ohlcv.columns) > 3:
                current_price = ohlcv.iloc[-1, 3]
            else:
                return None
            
            # Generate opposite side order to close position
            exit_side = 'SELL' if position_side == 'BUY' else 'BUY'
            
            return {
                'symbol': symbol,
                'side': exit_side,
                'quantity': position_quantity,
                'price': current_price,
                'timestamp': datetime.now(tz=pytz.UTC),
                'signal_type': 'EXIT',
                'confidence': 1.0,  # Exit signals are definitive
                'reasoning': exit_result.get('exit_reasoning', 'SqueezeFlow exit signal'),
                'position_id': position.get('id'),
                'exit_type': 'DYNAMIC'  # Mark as dynamic exit (not fixed stop/target)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating exit order: {str(e)}")
            return None