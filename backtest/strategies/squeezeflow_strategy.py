#!/usr/bin/env python3
"""
SqueezeFlow Trading Strategy - Complete Manual Trading Methodology Implementation
Implements the comprehensive SqueezeFlow methodology from documentation with:
- Market regime detection (1h/4h timeframes)
- CVD divergence detection (15m/30m)
- Reset detection with deceleration measurement
- Entry signals with absorption candle confirmation
- Flow-following exit logic (no fixed targets)
"""

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from influxdb import InfluxDBClient

# Import base strategy class
try:
    from ..strategy import BaseStrategy, TradingSignal, SignalStrength
    from ..strategy_logger import create_strategy_logger
except ImportError:
    from backtest.strategy import BaseStrategy, TradingSignal, SignalStrength
    from backtest.strategy_logger import create_strategy_logger

logger = logging.getLogger(__name__)


class SqueezeFlowState(Enum):
    """SqueezeFlow state machine states"""
    WATCHING_DIVERGENCE = "watching_divergence"
    DIVERGENCE_DETECTED = "divergence_detected" 
    RESET_DETECTED = "reset_detected"
    ENTRY_READY = "entry_ready"
    IN_POSITION = "in_position"
    EXIT_MONITORING = "exit_monitoring"


class MarketRegime(Enum):
    """Market regime classifications"""
    SHORT_SQUEEZE_ENV = "SHORT_SQUEEZE_ENV"  # Bullish environment
    LONG_SQUEEZE_ENV = "LONG_SQUEEZE_ENV"    # Bearish environment
    NEUTRAL_ENV = "NEUTRAL_ENV"              # No clear bias


class SqueezeFlowStrategy(BaseStrategy):
    """
    Complete SqueezeFlow Strategy Implementation
    
    Implements the full manual trading methodology with:
    - State machine tracking signal progression
    - Multi-timeframe analysis (1h/4h/15m/30m/5m)
    - Dynamic thresholds based on market conditions
    - Flow-following position management
    - No fixed stops or targets - exits based on flow changes
    """
    
    def setup_strategy(self):
        """Setup SqueezeFlow strategy parameters"""
        
        # Initialize comprehensive logging system
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.strategy_logger = create_strategy_logger("SqueezeFlow", session_id)
        self.strategy_logger.info("🚀 SqueezeFlow Strategy initialization started")
        
        # Initialize InfluxDB connection
        self.setup_database()
        
        # Initialize state machine
        self.state = SqueezeFlowState.WATCHING_DIVERGENCE
        self.state_data = {}
        self.position_data = {}
        
        # Initialize CVD baseline tracking for dynamic scaling (Option 1: Baseline Reset)
        self.cvd_baselines = {}  # Per-symbol CVD baselines: {symbol: {'spot': baseline, 'perp': baseline, 'timestamp': when_set}}
        
        self.strategy_logger.info(f"🔄 Initial state: {self.state.value}")
        self.strategy_logger.info("📊 CVD Baseline Reset Method initialized - tracking relative changes from entry point")
        
        # Strategy configuration with dynamic baseline approach
        self.strategy_config = {
            # Market regime thresholds (relative to baseline - Option 1: Baseline Reset Method)
            'regime_relative_threshold_strong': 100_000_000,    # 100M delta from baseline = strong trend
            'regime_relative_threshold_medium': 50_000_000,     # 50M delta from baseline = medium trend  
            'regime_relative_threshold_weak': 20_000_000,       # 20M delta from baseline = weak trend
            'regime_lookback_1h': 20,                        # 20 hours lookback
            'regime_lookback_4h': 20,                        # 20 periods (80 hours)
            
            # Divergence detection parameters (relative to baseline)
            'divergence_relative_threshold': 30_000_000,     # 30M delta from baseline minimum
            'divergence_min_periods': 10,                    # Must persist 10 periods (2.5h on 15m)
            'divergence_max_periods': 40,                    # Max 40 periods (10h on 15m)
            'divergence_price_threshold': 0.02,              # 2% price movement
            
            # Reset detection parameters
            'reset_gap_reduction': 0.7,                      # 70% gap reduction minimum
            'reset_price_movement': 0.003,                   # 0.3% price movement (optimized for 5-minute intervals)
            'reset_lookback': 50,                            # 50 periods lookback
            
            # Entry signal parameters
            'deceleration_reduction': 0.7,                   # 30% less CVD movement on second test
            'volume_surge_multiplier': 1.5,                 # 50% above average for absorption
            'cvd_alignment_periods': 3,                      # 3 periods for directional alignment
            'entry_signal_threshold': 4.2,                  # Minimum composite score (optimized for quality trades)
            
            # Exit parameters (relative to baseline)
            'flow_reversal_relative_threshold': 25_000_000,  # 25M delta divergence for flow reversal
            'range_break_buffer': 0.001,                    # 0.1% buffer for range breaks
            
            # Data requirements
            'min_data_points': 240,                         # Minimum data required
            'timeframes': ['1h', '4h', '15m', '30m', '5m']  # Required timeframes
        }
        
        # Initialize data discovery services
        try:
            from utils.symbol_discovery import symbol_discovery
            from utils.market_discovery import market_discovery
            
            self.symbol_discovery = symbol_discovery
            self.market_discovery = market_discovery
            self.available_symbols = self.symbol_discovery.discover_symbols_from_database()
            
            logger.info(f"✅ SqueezeFlow Strategy initialized with {len(self.available_symbols)} symbols")
            logger.info(f"🎯 Config: {self.strategy_config}")
            
        except ImportError as e:
            logger.error(f"Failed to import discovery services: {e}")
            self.available_symbols = ['BTC', 'ETH']  # Fallback
        
    def setup_database(self):
        """Setup InfluxDB connection for data retrieval"""
        self.influx_client = InfluxDBClient(
            host=os.getenv('INFLUX_HOST', 'localhost'),
            port=int(os.getenv('INFLUX_PORT', 8086)),
            database='significant_trades'
        )
    
    def generate_signal(self, symbol: str, timestamp: datetime,
                       lookback_data: Dict[str, pd.Series]) -> TradingSignal:
        """
        Generate SqueezeFlow trading signals using complete methodology
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            lookback_data: Dict with 'price', 'spot_cvd', 'perp_cvd' series
            
        Returns:
            TradingSignal object
        """
        try:
            self.strategy_logger.debug(f"🎯 Starting signal generation for {symbol} at {timestamp}")
            
            # Check data availability
            data_valid = self._validate_data(lookback_data)
            self.strategy_logger.log_data_validation(
                symbol, "input_data", data_valid, 
                sum(len(series) for series in lookback_data.values()) if lookback_data else 0,
                "Missing required data fields" if not data_valid else ""
            )
            
            if not data_valid:
                self.strategy_logger.warning(f"❌ Invalid data for {symbol}, generating NONE signal")
                return self._create_no_signal(symbol, timestamp, lookback_data)
            
            # Phase 1: Market Regime Detection
            self.strategy_logger.debug(f"🏛️ Phase 1: Detecting market regime for {symbol}")
            market_regime = self._detect_market_regime(symbol, timestamp)
            self.strategy_logger.debug(f"🏛️ Market regime detected: {market_regime.value if market_regime else 'None'}")
            
            # Phase 2: CVD Divergence Detection
            self.strategy_logger.debug(f"📊 Phase 2: Detecting CVD divergence for {symbol}")
            divergence_data = self._detect_cvd_divergence(symbol, timestamp)
            self.strategy_logger.debug(f"📊 Divergence data: {len(divergence_data) if divergence_data else 0} entries")
            
            # Phase 3: Reset Detection
            self.strategy_logger.debug(f"🔄 Phase 3: Detecting CVD reset for {symbol}")
            reset_data = self._detect_cvd_reset(lookback_data, timestamp)
            self.strategy_logger.debug(f"🔄 Reset data: {len(reset_data) if reset_data else 0} entries")
            
            # State machine progression
            old_state = self.state.value
            self._update_state(market_regime, divergence_data, reset_data)
            if old_state != self.state.value:
                self.strategy_logger.log_state_transition(symbol, old_state, self.state.value, "Phase progression")
            
            # Phase 4: Entry Signal Generation
            if self.state in [SqueezeFlowState.RESET_DETECTED, SqueezeFlowState.ENTRY_READY]:
                self.strategy_logger.debug(f"🎯 Phase 4: Generating entry signal for {symbol} (State: {self.state.value})")
                entry_signal = self._generate_entry_signal(
                    symbol, timestamp, lookback_data, market_regime, reset_data
                )
                if entry_signal:
                    self.strategy_logger.info(f"✅ Entry signal generated: {entry_signal.signal_type}")
                    return entry_signal
                else:
                    self.strategy_logger.debug(f"❌ No entry signal generated despite being in {self.state.value}")
            
            # Phase 5: Exit Signal Generation (if in position)
            if self.state == SqueezeFlowState.IN_POSITION:
                self.strategy_logger.debug(f"🚪 Phase 5: Generating exit signal for {symbol}")
                exit_signal = self._generate_exit_signal(
                    symbol, timestamp, lookback_data, market_regime
                )
                if exit_signal:
                    self.strategy_logger.info(f"✅ Exit signal generated: {exit_signal.signal_type}")
                    return exit_signal
                else:
                    self.strategy_logger.debug(f"❌ No exit signal generated")
            
            # Return no signal with current state info
            current_price = lookback_data.get('price', pd.Series([0])).iloc[-1] if lookback_data else 0
            spot_cvd = lookback_data.get('spot_cvd', pd.Series([0])).iloc[-1] if lookback_data else 0
            perp_cvd = lookback_data.get('perp_cvd', pd.Series([0])).iloc[-1] if lookback_data else 0
            
            self.strategy_logger.log_signal_generation(
                symbol=symbol,
                signal_type="NONE",
                confidence=0.0,
                price=current_price,
                state=self.state.value,
                market_regime=market_regime.value if market_regime else 'unknown',
                data_points=sum(len(series) for series in lookback_data.values()) if lookback_data else 0,
                spot_cvd=spot_cvd,
                perp_cvd=perp_cvd,
                divergence=divergence_data.get('divergence_strength', 0),
                error_message=""
            )
            
            return self._create_no_signal(symbol, timestamp, lookback_data, {
                'state': self.state.value,
                'market_regime': market_regime.value if market_regime else 'unknown',
                'divergence_strength': divergence_data.get('divergence_strength', 0),
                'reset_detected': reset_data.get('reset_detected', False)
            })
            
        except Exception as e:
            self.strategy_logger.error(f"Error generating SqueezeFlow signal for {symbol}", e, {
                'timestamp': timestamp,
                'data_keys': list(lookback_data.keys()) if lookback_data else []
            })
            return self._create_no_signal(symbol, timestamp, lookback_data)
    
    def _validate_data(self, lookback_data: Dict[str, pd.Series]) -> bool:
        """Validate that we have sufficient data for analysis"""
        required_length = self.strategy_config['min_data_points']
        
        for key in ['price', 'spot_cvd', 'perp_cvd']:
            if key not in lookback_data or len(lookback_data[key]) < required_length:
                return False
        
        return True
    
    def _set_cvd_baseline(self, symbol: str, spot_cvd: float, perp_cvd: float, timestamp: datetime):
        """Set CVD baseline for relative measurements (Option 1: Baseline Reset Method)"""
        self.cvd_baselines[symbol] = {
            'spot': spot_cvd,
            'perp': perp_cvd, 
            'timestamp': timestamp
        }
        self.strategy_logger.info(f"📊 CVD Baseline set for {symbol}: SPOT={spot_cvd:,.0f}, PERP={perp_cvd:,.0f}")
    
    def _get_cvd_relative_to_baseline(self, symbol: str, current_spot: float, current_perp: float) -> Tuple[float, float]:
        """Get CVD values relative to baseline (Option 1: Baseline Reset Method)"""
        if symbol not in self.cvd_baselines:
            # No baseline yet - set current as baseline
            self._set_cvd_baseline(symbol, current_spot, current_perp, datetime.now())
            return 0.0, 0.0  # First measurement is always zero delta
        
        baseline = self.cvd_baselines[symbol]
        spot_delta = current_spot - baseline['spot']
        perp_delta = current_perp - baseline['perp']
        
        return spot_delta, perp_delta
    
    def _detect_market_regime(self, symbol: str, timestamp: datetime) -> MarketRegime:
        """
        Phase 1: Market Regime Detection using 1h/4h timeframes
        Determines the dominant squeeze environment
        """
        try:
            # Calculate time ranges for regime detection
            end_time = timestamp
            
            # Check multiple timeframes as per documentation (30m, 1h, 4h)
            regime_30m = self._calculate_regime_for_timeframe(symbol, '30m', end_time, 20)
            regime_1h = self._calculate_regime_for_timeframe(symbol, '1h', end_time, 20)
            regime_4h = self._calculate_regime_for_timeframe(symbol, '4h', end_time, 20)
            
            # Combine regimes with priority: 4h > 1h > 30m (longer timeframes have higher weight)
            if regime_4h != MarketRegime.NEUTRAL_ENV:
                return regime_4h
            elif regime_1h != MarketRegime.NEUTRAL_ENV:
                return regime_1h
            elif regime_30m != MarketRegime.NEUTRAL_ENV:
                return regime_30m
            else:
                return MarketRegime.NEUTRAL_ENV
                
        except Exception as e:
            logger.warning(f"Error detecting market regime for {symbol}: {e}")
            return MarketRegime.NEUTRAL_ENV
    
    def _calculate_regime_for_timeframe(self, symbol: str, timeframe: str, 
                                      end_time: datetime, lookback: int) -> MarketRegime:
        """Calculate market regime for specific timeframe using baseline-relative approach"""
        try:
            # Get CVD data for the timeframe
            spot_cvd, perp_cvd = self._get_cvd_data(symbol, timeframe, end_time, lookback)
            
            if spot_cvd is None or perp_cvd is None or len(spot_cvd) < lookback:
                return MarketRegime.NEUTRAL_ENV
            
            # Calculate CVD range for dynamic threshold scaling
            spot_range = abs(spot_cvd.max() - spot_cvd.min())
            perp_range = abs(perp_cvd.max() - perp_cvd.min())
            
            # Calculate trend over lookback period (raw values)
            raw_spot_trend = spot_cvd.iloc[-1] - spot_cvd.iloc[0]
            raw_perp_trend = perp_cvd.iloc[-1] - perp_cvd.iloc[0]
            
            # Dynamic threshold based on recent CVD range (Option 1: Adaptive scaling)
            # Use 10% of recent range as medium threshold - pure dynamic scaling
            medium_threshold = max(spot_range, perp_range) * 0.1
            
            spot_trend = raw_spot_trend
            perp_trend = raw_perp_trend
            
            self.strategy_logger.debug(f"📊 {symbol} {timeframe} CVD Analysis: "
                                     f"SPOT_trend={spot_trend:,.0f}, PERP_trend={perp_trend:,.0f}, "
                                     f"threshold={medium_threshold:,.0f}")
            
            # Short squeeze environment: Spot buying + Perp selling (relative to baseline)
            if (spot_trend > medium_threshold and perp_trend < -medium_threshold/2):
                return MarketRegime.SHORT_SQUEEZE_ENV
            
            # Long squeeze environment: Spot selling + Perp buying (relative to baseline)
            elif (spot_trend < -medium_threshold and perp_trend > medium_threshold/2):
                return MarketRegime.LONG_SQUEEZE_ENV
            
            else:
                return MarketRegime.NEUTRAL_ENV
                
        except Exception as e:
            logger.warning(f"Error calculating regime for {timeframe}: {e}")
            return MarketRegime.NEUTRAL_ENV
    
    def _detect_cvd_divergence(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Phase 2: CVD Divergence Detection using 15m/30m timeframes
        Identifies price-CVD imbalances that create trading opportunities
        """
        try:
            # Get CVD data from proper timeframes as specified in documentation
            # Try 15m first, fallback to 30m if insufficient data
            lookback = 20  # 20 periods for analysis
            
            # Get 15m timeframe data first (preferred)
            spot_cvd_15m, perp_cvd_15m = self._get_cvd_data(symbol, '15m', timestamp, lookback + 5)
            
            if (spot_cvd_15m is not None and perp_cvd_15m is not None and 
                len(spot_cvd_15m) >= lookback and len(perp_cvd_15m) >= lookback):
                
                spot_cvd = spot_cvd_15m
                perp_cvd = perp_cvd_15m
                timeframe_used = '15m'
                
            else:
                # Fallback to 30m timeframe
                spot_cvd_30m, perp_cvd_30m = self._get_cvd_data(symbol, '30m', timestamp, lookback + 5)
                
                if (spot_cvd_30m is not None and perp_cvd_30m is not None and 
                    len(spot_cvd_30m) >= lookback and len(perp_cvd_30m) >= lookback):
                    
                    spot_cvd = spot_cvd_30m
                    perp_cvd = perp_cvd_30m
                    timeframe_used = '30m'
                else:
                    self.strategy_logger.warning(f"❌ Insufficient 15m/30m CVD data for {symbol}")
                    return {'divergence_detected': False, 'divergence_strength': 0}
            
            # Calculate price change from CVD momentum (approximation)
            # Use the sum of CVD changes as price momentum proxy
            price_momentum = (spot_cvd.diff().fillna(0) + perp_cvd.diff().fillna(0)).rolling(5).mean()
            
            self.strategy_logger.debug(f"📊 Phase 2: Using {timeframe_used} timeframe for divergence detection "
                                     f"({len(spot_cvd)} data points)")
            
            if len(spot_cvd) < lookback or len(perp_cvd) < lookback:
                return {'divergence_detected': False, 'divergence_strength': 0}
            
            # Calculate rate of change differential
            spot_slope = spot_cvd.diff(lookback).iloc[-1]
            perp_slope = perp_cvd.diff(lookback).iloc[-1]
            
            # Price change calculation using momentum proxy
            price_momentum_change = price_momentum.iloc[-1] - price_momentum.iloc[-lookback]
            price_change_proxy = price_momentum_change / abs(price_momentum.iloc[-lookback]) if price_momentum.iloc[-lookback] != 0 else 0
            
            # Dynamic threshold calculation for divergence (Fix: Remove fixed thresholds)
            cvd_range = max(
                abs(spot_cvd.max() - spot_cvd.min()),
                abs(perp_cvd.max() - perp_cvd.min())
            )
            
            # Divergence thresholds: 15% of recent CVD range for strong movements
            # Pure dynamic scaling - no hardcoded minimums (trust the market's own scale)
            strong_threshold = cvd_range * 0.15  # Always 15% of range
            weak_threshold = cvd_range * 0.08    # Always 8% of range
            
            # Divergence strength calculation
            divergence_strength = abs(spot_slope - perp_slope)
            
            # Long squeeze pattern detection (dynamic thresholds)
            long_squeeze_pattern = (
                price_change_proxy > self.strategy_config['divergence_price_threshold'] and
                spot_slope > weak_threshold and
                perp_slope < -weak_threshold/2
            )
            
            # Short squeeze pattern detection (dynamic thresholds)  
            short_squeeze_pattern = (
                price_change_proxy < -self.strategy_config['divergence_price_threshold'] and
                spot_slope < -weak_threshold and
                perp_slope > weak_threshold/2
            )
            
            threshold_met = divergence_strength > strong_threshold
            
            # Debug logging for dynamic thresholds
            self.strategy_logger.debug(f"📊 CVD Divergence Analysis: "
                                     f"SPOT_slope={spot_slope:,.0f}, PERP_slope={perp_slope:,.0f}, "
                                     f"strong_threshold={strong_threshold:,.0f}, "
                                     f"weak_threshold={weak_threshold:,.0f}, "
                                     f"divergence_strength={divergence_strength:,.0f}")
            
            return {
                'divergence_detected': long_squeeze_pattern or short_squeeze_pattern,
                'divergence_strength': divergence_strength,
                'long_squeeze': long_squeeze_pattern,
                'short_squeeze': short_squeeze_pattern,
                'threshold_met': threshold_met,
                'spot_slope': spot_slope,
                'perp_slope': perp_slope,
                'price_change': price_change_proxy,
                'timeframe_used': timeframe_used
            }
            
        except Exception as e:
            logger.warning(f"Error detecting CVD divergence: {e}")
            return {'divergence_detected': False, 'divergence_strength': 0}
    
    def _detect_cvd_reset(self, lookback_data: Dict[str, pd.Series], 
                         timestamp: datetime) -> Dict[str, Any]:
        """
        Phase 3: Reset Detection
        Identifies when divergence resolves and market seeks new equilibrium
        """
        try:
            spot_cvd = lookback_data['spot_cvd']
            perp_cvd = lookback_data['perp_cvd']
            price = lookback_data['price']
            
            lookback = self.strategy_config['reset_lookback']
            
            if len(spot_cvd) < lookback or len(perp_cvd) < lookback:
                return {'reset_detected': False, 'gap_reduction': 0}
            
            # Absolute convergence detection - track recent trend direction
            # Calculate CVD gaps over recent periods to detect convergence
            convergence_window = min(10, len(spot_cvd) - 1)  # Use last 10 periods or available data
            
            gaps = []
            for i in range(convergence_window):
                gap = abs(spot_cvd.iloc[-(i+1)] - perp_cvd.iloc[-(i+1)])
                gaps.append(gap)
            
            # Gaps list is now [current_gap, gap_1_period_ago, gap_2_periods_ago, ...]
            current_gap = gaps[0]
            
            # Count how many recent periods show gap reduction (convergence trend)
            convergence_count = 0
            total_periods = len(gaps) - 1
            
            for i in range(total_periods):
                if gaps[i] < gaps[i + 1]:  # Current gap smaller than previous gap
                    convergence_count += 1
            
            # Convergence ratio: what fraction of recent periods show convergence?
            convergence_ratio = convergence_count / total_periods if total_periods > 0 else 0
            
            # Also check absolute gap reduction magnitude
            if len(gaps) >= 3:
                recent_gap_reduction = gaps[-1] - gaps[0]  # Oldest gap - current gap
                gap_reduction_magnitude = recent_gap_reduction / gaps[-1] if gaps[-1] > 0 else 0
            else:
                gap_reduction_magnitude = 0
            
            # Price movement during convergence (reduced threshold for realism)
            price_change = abs(price.pct_change(10).iloc[-1])
            
            # Volatility decline check (relaxed for market realism)
            recent_volatility = price.rolling(10).std().iloc[-1]  # Shorter window
            historical_volatility = price.rolling(30).std().iloc[-1]  # Shorter comparison
            volatility_decline = recent_volatility <= historical_volatility * 1.1  # Allow 10% tolerance
            
            # Absolute convergence conditions (adjusted for real market data)
            # "CVD lines return more in line with each other" - focus on trend direction
            strong_convergence = convergence_ratio >= 0.65  # 65% of recent periods show convergence
            moderate_convergence = convergence_ratio >= 0.35  # 35% of recent periods show convergence (optimized)
            
            # Additional validation: meaningful gap reduction magnitude (relaxed)
            meaningful_reduction = gap_reduction_magnitude > 0.1  # 10% reduction in gap size (more realistic)
            
            # Price movement validation (sharp price movement during reset)
            price_movement = price_change > self.strategy_config['reset_price_movement']
            
            # Convergence detection: either strong trend OR moderate trend with meaningful reduction
            convergence_detected = strong_convergence or (moderate_convergence and meaningful_reduction)
            
            reset_detected = convergence_detected and price_movement and volatility_decline
            
            # Debug logging for absolute convergence with ALL conditions
            self.strategy_logger.debug(f"🔄 CVD Reset Analysis: "
                                     f"convergence_ratio={convergence_ratio:.2f}, "
                                     f"gap_reduction_magnitude={gap_reduction_magnitude:.2f}, "
                                     f"current_gap={current_gap:,.0f}, "
                                     f"strong_convergence={strong_convergence}, "
                                     f"moderate_convergence={moderate_convergence}, "
                                     f"meaningful_reduction={meaningful_reduction}, "
                                     f"price_change={price_change:.4f}, "
                                     f"price_movement_met={price_movement}, "
                                     f"volatility_decline={volatility_decline}, "
                                     f"reset_detected={reset_detected}")
            
            return {
                'reset_detected': reset_detected,
                'gap_reduction': convergence_ratio,  # Now represents convergence ratio, not percentage
                'convergence_level': current_gap,
                'price_change': price_change,
                'volatility_decline': volatility_decline,
                'convergence_detected': convergence_detected,
                'price_movement_met': price_movement,
                'convergence_ratio': convergence_ratio,
                'gap_reduction_magnitude': gap_reduction_magnitude
            }
            
        except Exception as e:
            logger.warning(f"Error detecting CVD reset: {e}")
            return {'reset_detected': False, 'gap_reduction': 0}
    
    def _update_state(self, market_regime: MarketRegime, 
                     divergence_data: Dict[str, Any], 
                     reset_data: Dict[str, Any]):
        """Update state machine based on current conditions"""
        
        current_state = self.state
        
        if current_state == SqueezeFlowState.WATCHING_DIVERGENCE:
            if divergence_data.get('divergence_detected', False):
                self.state = SqueezeFlowState.DIVERGENCE_DETECTED
                
        elif current_state == SqueezeFlowState.DIVERGENCE_DETECTED:
            if reset_data.get('reset_detected', False):
                self.state = SqueezeFlowState.RESET_DETECTED
                
        elif current_state == SqueezeFlowState.RESET_DETECTED:
            # Check if conditions are right for entry (removed NEUTRAL_ENV blocking per documentation)
            # Documentation focuses on squeeze environments but doesn't explicitly block neutral markets
            if reset_data.get('gap_reduction', 0) > 0.4:  # Use convergence_ratio - optimized threshold (40%)
                self.state = SqueezeFlowState.ENTRY_READY
    
    def _generate_entry_signal(self, symbol: str, timestamp: datetime,
                              lookback_data: Dict[str, pd.Series],
                              market_regime: MarketRegime,
                              reset_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Phase 4: Entry Signal Detection
        Generate entry signal when reset momentum exhausts and new directional bias begins
        """
        try:
            # Check reset deceleration
            deceleration_data = self._detect_reset_deceleration(lookback_data)
            
            # Check absorption candle
            absorption_data = self._detect_absorption_candle(lookback_data)
            
            # Check CVD alignment
            alignment_data = self._detect_cvd_alignment(lookback_data)
            
            # Combine all entry conditions (removed NEUTRAL_ENV blocking to enable more trades)
            entry_conditions = {
                'reset_quality': reset_data.get('reset_detected', False),
                'deceleration_confirmed': deceleration_data.get('deceleration_detected', False),
                'absorption_candle': absorption_data.get('absorption_detected', False),
                'cvd_alignment': alignment_data.get('alignment_detected', False)
            }
            
            # Calculate signal strength (removed market_regime_favorable component)
            signal_strength = sum([
                2 if entry_conditions['reset_quality'] else 0,
                1.5 if entry_conditions['deceleration_confirmed'] else 0,
                1 if entry_conditions['absorption_candle'] else 0,
                1 if entry_conditions['cvd_alignment'] else 0
            ])
            
            # Check if entry signal should be generated (less strict: most key conditions instead of ALL)
            min_threshold = self.strategy_config['entry_signal_threshold']
            key_conditions_met = entry_conditions['reset_quality'] and entry_conditions['deceleration_confirmed']
            if signal_strength >= min_threshold and key_conditions_met:
                
                # Determine signal direction based on market regime OR CVD divergence pattern
                if market_regime == MarketRegime.SHORT_SQUEEZE_ENV:
                    signal_type = 'LONG'
                elif market_regime == MarketRegime.LONG_SQUEEZE_ENV:
                    signal_type = 'SHORT'
                else:
                    # NEUTRAL_ENV: Use CVD divergence pattern to determine direction
                    # Look at recent CVD trends to determine bias
                    spot_cvd = lookback_data.get('spot_cvd', pd.Series())
                    perp_cvd = lookback_data.get('perp_cvd', pd.Series())
                    
                    if len(spot_cvd) > 5 and len(perp_cvd) > 5:
                        # Simple trend-based direction: if SPOT rising faster than PERP = LONG bias
                        spot_trend = spot_cvd.iloc[-1] - spot_cvd.iloc[-5]
                        perp_trend = perp_cvd.iloc[-1] - perp_cvd.iloc[-5] 
                        
                        if spot_trend > perp_trend:
                            signal_type = 'LONG'  # Spot buying pressure higher
                        else:
                            signal_type = 'SHORT'  # Perp buying pressure higher
                    else:
                        # Fallback: Use CVD absolute levels
                        signal_type = 'LONG' if spot_cvd.iloc[-1] > perp_cvd.iloc[-1] else 'SHORT'
                
                # Update state to in position
                self.state = SqueezeFlowState.IN_POSITION
                
                # Store entry range for proper range break detection
                price_series = lookback_data['price']
                range_lookback = 10
                entry_range_low = price_series.rolling(range_lookback).min().iloc[-1]
                entry_range_high = price_series.rolling(range_lookback).max().iloc[-1]
                
                self.position_data = {
                    'entry_price': lookback_data['price'].iloc[-1],
                    'entry_time': timestamp,
                    'market_regime': market_regime,
                    'signal_strength': signal_strength,
                    'signal_type': signal_type,
                    'entry_range_low': entry_range_low,
                    'entry_range_high': entry_range_high
                }
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=SignalStrength.STRONG if signal_strength > 7 else SignalStrength.MODERATE,
                    confidence=min(signal_strength / 7.5, 1.0),
                    price=lookback_data['price'].iloc[-1],
                    timestamp=timestamp,
                    metadata={
                        'strategy': 'SqueezeFlow',
                        'market_regime': market_regime.value,
                        'signal_strength': signal_strength,
                        'entry_conditions': entry_conditions,
                        'reset_data': reset_data,
                        'deceleration_data': deceleration_data,
                        'absorption_data': absorption_data,
                        'alignment_data': alignment_data
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating entry signal: {e}")
            return None
    
    def _detect_reset_deceleration(self, lookback_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Detect CVD reset deceleration - key entry timing signal"""
        try:
            spot_cvd = lookback_data['spot_cvd']
            price = lookback_data['price']
            
            # Look for double bottom pattern in price
            recent_lows = price.rolling(20).min()
            current_low = price.iloc[-1]
            
            # Find previous test of similar level (within 0.2%)
            similar_level_mask = abs(recent_lows - current_low) / current_low < 0.002
            previous_tests = recent_lows[similar_level_mask]
            
            if len(previous_tests) >= 2:
                # Compare CVD movement during each test
                first_test_idx = previous_tests.index[0]
                second_test_idx = previous_tests.index[-1]
                
                # Get indices for comparison
                first_start = max(0, list(spot_cvd.index).index(first_test_idx) - 5)
                first_end = min(len(spot_cvd) - 1, list(spot_cvd.index).index(first_test_idx) + 5)
                second_start = max(0, list(spot_cvd.index).index(second_test_idx) - 5)
                second_end = min(len(spot_cvd) - 1, list(spot_cvd.index).index(second_test_idx) + 5)
                
                # CVD change during each test
                first_cvd_change = abs(spot_cvd.iloc[first_end] - spot_cvd.iloc[first_start])
                second_cvd_change = abs(spot_cvd.iloc[second_end] - spot_cvd.iloc[second_start])
                
                # Deceleration detected if second test shows less CVD movement
                deceleration = (second_cvd_change < first_cvd_change * 0.7 if first_cvd_change > 0 else False)
                
                return {
                    'deceleration_detected': deceleration,
                    'first_cvd_change': first_cvd_change,
                    'second_cvd_change': second_cvd_change,
                    'reduction_ratio': second_cvd_change / first_cvd_change if first_cvd_change > 0 else 0
                }
            
            return {'deceleration_detected': False}
            
        except Exception as e:
            logger.warning(f"Error detecting reset deceleration: {e}")
            return {'deceleration_detected': False}
    
    def _detect_absorption_candle(self, lookback_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Detect absorption candles - volume confirmation for entries"""
        try:
            # Since we don't have OHLCV in the standard format, we'll approximate
            # using price and volume-like data from CVD
            price = lookback_data['price']
            spot_cvd = lookback_data['spot_cvd']
            
            if len(price) < 10:
                return {'absorption_detected': False}
            
            # Current period data
            current_price = price.iloc[-1]
            prev_price = price.iloc[-2]
            
            # Volume proxy using CVD change
            current_volume_proxy = abs(spot_cvd.diff().iloc[-1])
            avg_volume_proxy = abs(spot_cvd.diff().rolling(10).mean().iloc[-2])
            
            # Volume surge detection
            volume_surge = current_volume_proxy > avg_volume_proxy * self.strategy_config['volume_surge_multiplier']
            
            # Price action analysis (simplified without OHLC)
            price_moving_up = current_price > prev_price
            
            # Look for rejection of lower levels (price bouncing back)
            recent_low = price.rolling(5).min().iloc[-1]
            bouncing_from_low = current_price > recent_low * 1.001  # 0.1% above recent low
            
            absorption_detected = volume_surge and price_moving_up and bouncing_from_low
            
            return {
                'absorption_detected': absorption_detected,
                'volume_ratio': current_volume_proxy / avg_volume_proxy if avg_volume_proxy > 0 else 0,
                'price_moving_up': price_moving_up,
                'bouncing_from_low': bouncing_from_low
            }
            
        except Exception as e:
            logger.warning(f"Error detecting absorption candle: {e}")
            return {'absorption_detected': False}
    
    def _detect_cvd_alignment(self, lookback_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Detect CVD directional alignment - confirmation of new bias"""
        try:
            spot_cvd = lookback_data['spot_cvd']
            perp_cvd = lookback_data['perp_cvd']
            
            min_periods = self.strategy_config['cvd_alignment_periods']
            
            if len(spot_cvd) < min_periods or len(perp_cvd) < min_periods:
                return {'alignment_detected': False}
            
            # Check last N periods for directional alignment
            spot_trend = spot_cvd.diff(min_periods).iloc[-1] > 0
            perp_trend = perp_cvd.diff(min_periods).iloc[-1] > 0
            
            # Leadership pattern detection
            spot_strength = abs(spot_cvd.diff(min_periods).iloc[-1])
            perp_strength = abs(perp_cvd.diff(min_periods).iloc[-1])
            
            if spot_strength > perp_strength * 1.2:
                leadership = "SPOT_LED"
            elif perp_strength > spot_strength * 1.2:
                leadership = "PERP_LED"
            else:
                leadership = "BOTH_ALIGNED"
            
            # Alignment detected if both trending in same direction
            alignment_detected = spot_trend and perp_trend
            
            return {
                'alignment_detected': alignment_detected,
                'leadership_pattern': leadership,
                'spot_strength': spot_strength,
                'perp_strength': perp_strength,
                'spot_trend_up': spot_trend,
                'perp_trend_up': perp_trend
            }
            
        except Exception as e:
            logger.warning(f"Error detecting CVD alignment: {e}")
            return {'alignment_detected': False}
    
    def _generate_exit_signal(self, symbol: str, timestamp: datetime,
                             lookback_data: Dict[str, pd.Series],
                             market_regime: MarketRegime) -> Optional[TradingSignal]:
        """
        Phase 5: Exit Signal Generation
        Flow-following exit logic - no fixed targets, exit on condition changes
        """
        try:
            if not self.position_data:
                return None
            
            entry_price = self.position_data.get('entry_price', 0)
            current_price = lookback_data['price'].iloc[-1]
            
            # Calculate current profit
            if 'signal_type' in self.position_data:
                if self.position_data['signal_type'] == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price
            else:
                # Determine from regime
                if self.position_data.get('market_regime') == MarketRegime.SHORT_SQUEEZE_ENV:
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price
            
            # Check exit conditions
            
            # 1. Flow reversal detection
            flow_reversal = self._detect_flow_reversal(lookback_data, entry_price)
            
            # 2. Range break detection
            range_break = self._detect_range_break(lookback_data, entry_price)
            
            # 3. Larger timeframe invalidation
            regime_invalidation = self._validate_larger_timeframe(
                market_regime, self.position_data.get('market_regime'), profit_pct
            )
            
            # Determine exit action
            exit_signal = False
            exit_reason = None
            
            if flow_reversal.get('flow_reversal_detected', False):
                exit_signal = True
                exit_reason = 'flow_reversal'
            elif range_break.get('range_break_detected', False):
                exit_signal = True
                exit_reason = 'range_break'
            elif regime_invalidation.get('recommended_action') in ['EXIT_IMMEDIATELY', 'SECURE_PROFITS']:
                exit_signal = True
                exit_reason = 'regime_invalidation'
            
            if exit_signal:
                # Store position type before clearing position data
                position_type = self.position_data.get('signal_type', 'LONG')
                
                # Reset state machine and clear position data
                self.state = SqueezeFlowState.WATCHING_DIVERGENCE
                self.position_data = {}  # Clear position data after exit
                
                # Generate exit signal (opposite of entry)
                exit_signal_type = 'SELL' if position_type == 'LONG' else 'BUY'
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=exit_signal_type,
                    strength=SignalStrength.STRONG,
                    confidence=0.9,
                    price=current_price,
                    timestamp=timestamp,
                    metadata={
                        'strategy': 'SqueezeFlow',
                        'exit_reason': exit_reason,
                        'profit_pct': profit_pct,
                        'flow_reversal': flow_reversal,
                        'range_break': range_break,
                        'regime_invalidation': regime_invalidation,
                        'position_data': self.position_data
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating exit signal: {e}")
            return None
    
    def _detect_flow_reversal(self, lookback_data: Dict[str, pd.Series], 
                             entry_price: float) -> Dict[str, Any]:
        """Detect dangerous flow reversal patterns"""
        try:
            spot_cvd = lookback_data['spot_cvd']
            perp_cvd = lookback_data['perp_cvd']
            price = lookback_data['price']
            
            lookback = 5
            current_price = price.iloc[-1]
            
            # Check for dangerous divergence pattern
            spot_trend = spot_cvd.diff(lookback).iloc[-1] < 0  # SPOT selling
            perp_trend = perp_cvd.diff(lookback).iloc[-1] > 0  # PERP buying
            price_struggling = current_price < entry_price * 1.005  # Not making progress
            
            # Volume divergence strength
            divergence_magnitude = (abs(spot_cvd.diff(lookback).iloc[-1]) + 
                                  abs(perp_cvd.diff(lookback).iloc[-1]))
            
            flow_reversal = (spot_trend and perp_trend and price_struggling and 
                           divergence_magnitude > self.strategy_config['flow_reversal_threshold'])
            
            return {
                'flow_reversal_detected': flow_reversal,
                'spot_selling': spot_trend,
                'perp_buying': perp_trend,
                'price_struggling': price_struggling,
                'divergence_magnitude': divergence_magnitude
            }
            
        except Exception as e:
            logger.warning(f"Error detecting flow reversal: {e}")
            return {'flow_reversal_detected': False}
    
    def _detect_range_break(self, lookback_data: Dict[str, pd.Series], 
                           entry_price: float) -> Dict[str, Any]:
        """Detect range breaks that invalidate the setup - Uses stored entry range"""
        try:
            current_price = lookback_data['price'].iloc[-1]
            
            # Use stored entry range from position data (FIX: No more historical range bug)
            if not self.position_data:
                return {'range_break_detected': False, 'reason': 'No position data'}
            
            entry_range_low = self.position_data.get('entry_range_low', entry_price * 0.98)
            entry_range_high = self.position_data.get('entry_range_high', entry_price * 1.02)
            
            # Range break conditions with buffer
            buffer = self.strategy_config['range_break_buffer']
            break_below = current_price < entry_range_low * (1 - buffer)
            break_above = current_price > entry_range_high * (1 + buffer)
            
            return {
                'range_break_detected': break_below or break_above,
                'break_direction': 'down' if break_below else 'up' if break_above else 'none',
                'entry_range_low': entry_range_low,
                'entry_range_high': entry_range_high,
                'current_price': current_price,
                'reason': f"Current price {current_price:.2f} vs entry range [{entry_range_low:.2f}, {entry_range_high:.2f}]"
            }
            
        except Exception as e:
            logger.warning(f"Error detecting range break: {e}")
            return {'range_break_detected': False, 'reason': f'Error: {e}'}
    
    def _validate_larger_timeframe(self, current_regime: MarketRegime, 
                                  entry_regime: MarketRegime, 
                                  profit_pct: float) -> Dict[str, Any]:
        """Validate if larger timeframe still supports the position - Pure flow-following approach"""
        try:
            # Determine if position direction is supported by current regime
            regime_valid = current_regime == entry_regime
            
            # Flow-following decision making (NO profit-based thresholds)
            # Exit immediately if larger timeframe no longer supports position
            if not regime_valid:
                action = 'EXIT_IMMEDIATELY'  # Regime changed = squeeze conditions invalidated
            else:
                action = 'HOLD'  # Continue following the flow while regime supports position
            
            return {
                'regime_valid': regime_valid,
                'recommended_action': action,
                'profit_level': profit_pct,  # For logging only, not decision making
                'current_regime': current_regime.value if current_regime else 'unknown',
                'entry_regime': entry_regime.value if entry_regime else 'unknown'
            }
            
        except Exception as e:
            logger.warning(f"Error validating larger timeframe: {e}")
            return {'regime_valid': True, 'recommended_action': 'HOLD'}
    
    def _get_cvd_data(self, symbol: str, timeframe: str, end_time: datetime,
                     lookback_periods: int) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Get CVD data from InfluxDB aggregated measurements for specific timeframe
        
        Uses the multi-timeframe data created by InfluxDB Continuous Queries:
        - aggr_5m.trades_5m for 5-minute data
        - aggr_15m.trades_15m for 15-minute data  
        - aggr_30m.trades_30m for 30-minute data
        - aggr_1h.trades_1h for 1-hour data
        - aggr_4h.trades_4h for 4-hour data
        """
        try:
            # Map timeframe to InfluxDB retention policy and measurement
            timeframe_mapping = {
                '5m': ('aggr_5m', 'trades_5m', 5),      # 5 minutes per period
                '15m': ('aggr_15m', 'trades_15m', 15),  # 15 minutes per period
                '30m': ('aggr_30m', 'trades_30m', 30),  # 30 minutes per period
                '1h': ('aggr_1h', 'trades_1h', 60),     # 60 minutes per period
                '4h': ('aggr_4h', 'trades_4h', 240)     # 240 minutes per period
            }
            
            if timeframe not in timeframe_mapping:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return None, None
            
            retention_policy, measurement, minutes_per_period = timeframe_mapping[timeframe]
            
            # Calculate start time based on lookback periods
            start_time = end_time - timedelta(minutes=lookback_periods * minutes_per_period)
            
            # Get markets for this symbol
            markets = self.market_discovery.get_markets_by_type(symbol)
            spot_markets = markets['spot']
            perp_markets = markets['perp']
            
            if not spot_markets or not perp_markets:
                self.strategy_logger.error(f"❌ No markets found for {symbol}: spot={len(spot_markets)}, perp={len(perp_markets)}")
                return None, None
            
            self.strategy_logger.debug(f"🔍 Found markets for {symbol}: {len(spot_markets)} SPOT, {len(perp_markets)} PERP")
            
            # Build market filters
            spot_filter = ' OR '.join([f"market = '{market}'" for market in spot_markets])
            perp_filter = ' OR '.join([f"market = '{market}'" for market in perp_markets])
            
            # Convert timestamps to nanoseconds for InfluxDB
            start_time_ns = int(start_time.timestamp() * 1_000_000_000)
            end_time_ns = int(end_time.timestamp() * 1_000_000_000)
            
            # SPOT CVD Query using aggregated data
            spot_query = f"""
            SELECT time, sum(vbuy) - sum(vsell) AS spot_cvd 
            FROM "{retention_policy}"."{measurement}" 
            WHERE ({spot_filter}) 
            AND time >= {start_time_ns} AND time <= {end_time_ns} 
            GROUP BY time({timeframe})
            ORDER BY time
            """
            
            # PERP CVD Query using aggregated data
            perp_query = f"""
            SELECT time, sum(vbuy) - sum(vsell) AS perp_cvd 
            FROM "{retention_policy}"."{measurement}" 
            WHERE ({perp_filter}) 
            AND time >= {start_time_ns} AND time <= {end_time_ns} 
            GROUP BY time({timeframe})
            ORDER BY time
            """
            
            # Log queries for debugging
            self.strategy_logger.debug(f"🔍 SPOT Query: {spot_query[:200]}...")
            self.strategy_logger.debug(f"🔍 PERP Query: {perp_query[:200]}...")
            
            # Execute queries
            spot_result = self.influx_client.query(spot_query)
            perp_result = self.influx_client.query(perp_query)
            
            # Process results
            spot_points = list(spot_result.get_points())
            perp_points = list(perp_result.get_points())
            
            self.strategy_logger.debug(f"📊 Query results: {len(spot_points)} SPOT points, {len(perp_points)} PERP points")
            
            if not spot_points or not perp_points:
                self.strategy_logger.warning(f"❌ Insufficient CVD data for {symbol} {timeframe}: SPOT={len(spot_points)}, PERP={len(perp_points)}")
                return None, None
            
            # Convert to DataFrame and calculate cumulative CVD
            spot_df = pd.DataFrame(spot_points)
            perp_df = pd.DataFrame(perp_points)
            
            spot_df['time'] = pd.to_datetime(spot_df['time'])
            perp_df['time'] = pd.to_datetime(perp_df['time'])
            
            spot_df.set_index('time', inplace=True)
            perp_df.set_index('time', inplace=True)
            
            # Calculate cumulative CVD (industry standard)
            spot_df['spot_cvd_cumulative'] = spot_df['spot_cvd'].fillna(0).cumsum()
            perp_df['perp_cvd_cumulative'] = perp_df['perp_cvd'].fillna(0).cumsum()
            
            self.strategy_logger.debug(f"✅ CVD calculation complete: SPOT range [{spot_df['spot_cvd_cumulative'].min():.0f}, {spot_df['spot_cvd_cumulative'].max():.0f}], PERP range [{perp_df['perp_cvd_cumulative'].min():.0f}, {perp_df['perp_cvd_cumulative'].max():.0f}]")
            
            return spot_df['spot_cvd_cumulative'], perp_df['perp_cvd_cumulative']
            
        except Exception as e:
            self.strategy_logger.error(f"Error getting CVD data for {symbol} {timeframe}", e, {
                'start_time': start_time,
                'end_time': end_time,
                'lookback_periods': lookback_periods
            })
            return None, None
    
    def _create_no_signal(self, symbol: str, timestamp: datetime,
                         lookback_data: Dict[str, pd.Series],
                         metadata: Dict[str, Any] = None) -> TradingSignal:
        """Create a no-signal response with metadata"""
        current_price = lookback_data.get('price', pd.Series([0])).iloc[-1] if lookback_data else 0
        
        base_metadata = {
            'strategy': 'SqueezeFlow',
            'state': self.state.value,
            'data_available': self._validate_data(lookback_data) if lookback_data else False
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        return TradingSignal(
            symbol=symbol,
            signal_type='NONE',
            strength=SignalStrength.NONE,
            confidence=0.0,
            price=current_price,
            timestamp=timestamp,
            metadata=base_metadata
        )
    
    def should_exit_position(self, position, current_data: Dict[str, pd.Series], 
                           timestamp: datetime) -> Tuple[bool, str, float]:
        """
        SqueezeFlow exit logic based on flow following methodology
        Uses internal _generate_exit_signal logic to determine exit conditions
        """
        try:
            # Use internal exit signal generation
            exit_signal = self._generate_exit_signal(
                position.symbol, 
                timestamp, 
                current_data, 
                MarketRegime.NEUTRAL_ENV  # Use NEUTRAL as default since we have CVD-based direction
            )
            
            if exit_signal and exit_signal.signal_type in ['SELL', 'BUY']:
                # Extract exit reason and confidence from metadata
                exit_reason = exit_signal.metadata.get('exit_reason', 'strategy_exit')
                confidence = exit_signal.confidence
                
                self.strategy_logger.info(f"💰 Exit signal generated: {exit_signal.signal_type} "
                                        f"for {position.symbol} (Reason: {exit_reason}, "
                                        f"Confidence: {confidence:.2f})")
                
                return True, exit_reason, confidence
            
            return False, "holding", 0.0
            
        except Exception as e:
            self.strategy_logger.error(f"Error in should_exit_position: {e}")
            return False, f"error: {str(e)}", 0.0