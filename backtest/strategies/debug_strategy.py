#!/usr/bin/env python3
"""
Debug Strategy - Test signal generation with relaxed thresholds
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..strategy import BaseStrategy, TradingSignal, SignalStrength
except ImportError:
    from core.strategy import BaseStrategy, TradingSignal, SignalStrength

class DebugStrategy(BaseStrategy):
    """Debug strategy with very relaxed thresholds to test signal generation"""
    
    def setup_strategy(self):
        """Setup debug strategy with relaxed parameters"""
        self.cvd_normalization = 100_000
        self.min_confidence = 0.3  # Very low confidence requirement
        self.min_score = 0.05      # Very low score requirement
        
        self.logger.info(f"✅ DebugStrategy initialized with relaxed thresholds")
    
    def generate_signal(self, symbol: str, timestamp: datetime,
                       lookback_data: Dict[str, pd.Series]) -> TradingSignal:
        """Generate signal with detailed debugging"""
        try:
            prices = lookback_data['price']
            spot_cvd = lookback_data['spot_cvd']
            perp_cvd = lookback_data['perp_cvd']
            
            print(f"\\n🔍 DEBUG - {symbol} at {timestamp}")
            print(f"Data lengths: price={len(prices)}, spot_cvd={len(spot_cvd)}, perp_cvd={len(perp_cvd)}")
            
            # Very minimal data requirement
            if len(prices) < 10 or len(spot_cvd) < 10 or len(perp_cvd) < 10:
                print("❌ Insufficient data")
                return TradingSignal(
                    symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                    confidence=0.0, price=prices.iloc[-1] if len(prices) > 0 else 0.0, 
                    timestamp=timestamp
                )
            
            current_price = prices.iloc[-1]
            lookback = min(30, len(prices) - 1)
            
            # Simple price change
            price_start = prices.iloc[-lookback]
            price_change = (current_price - price_start) / price_start
            print(f"Price change over {lookback} periods: {price_change*100:.3f}%")
            
            # Simple CVD changes
            spot_start = spot_cvd.iloc[-lookback]
            spot_current = spot_cvd.iloc[-1]
            spot_change = spot_current - spot_start
            
            perp_start = perp_cvd.iloc[-lookback]
            perp_current = perp_cvd.iloc[-1]
            perp_change = perp_current - perp_start
            
            print(f"Spot CVD change: {spot_change:,.0f}")
            print(f"Perp CVD change: {perp_change:,.0f}")
            
            # Very simple scoring
            spot_norm = spot_change / self.cvd_normalization
            perp_norm = perp_change / self.cvd_normalization
            price_norm = price_change / 0.01  # 1% normalization
            
            # Simple divergence
            divergence = abs(spot_norm - perp_norm)
            
            # Very simple score
            score = price_norm * 0.5 + (perp_norm - spot_norm) * 0.5
            
            print(f"Normalized: spot={spot_norm:.3f}, perp={perp_norm:.3f}, price={price_norm:.3f}")
            print(f"Divergence: {divergence:.3f}, Score: {score:.3f}")
            
            # Very relaxed signal classification
            signal_type = 'NONE'
            confidence = 0.0
            
            if abs(score) > self.min_score:
                if score <= -self.min_score:
                    signal_type = 'LONG'
                    confidence = min(abs(score), 1.0)
                elif score >= self.min_score:
                    signal_type = 'SHORT'
                    confidence = min(abs(score), 1.0)
                
                # Add divergence bonus
                confidence += divergence * 0.5
                confidence = min(confidence, 1.0)
            
            print(f"Signal: {signal_type}, Confidence: {confidence:.3f}")
            
            if confidence >= self.min_confidence:
                strength = SignalStrength.STRONG if confidence > 0.7 else (
                    SignalStrength.MODERATE if confidence > 0.5 else SignalStrength.WEAK)
                print(f"✅ Signal generated: {signal_type} with {strength}")
            else:
                signal_type = 'NONE'
                strength = SignalStrength.NONE
                confidence = 0.0
                print(f"❌ Confidence too low: {confidence:.3f} < {self.min_confidence}")
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price=current_price,
                timestamp=timestamp,
                metadata={
                    'score': score,
                    'divergence': divergence,
                    'price_change_pct': price_change * 100,
                    'spot_change': spot_change,
                    'perp_change': perp_change,
                    'strategy': 'DebugStrategy'
                }
            )
            
        except Exception as e:
            print(f"❌ Error in signal generation: {e}")
            import traceback
            traceback.print_exc()
            return TradingSignal(
                symbol=symbol, signal_type='NONE', strength=SignalStrength.NONE,
                confidence=0.0, price=0.0, timestamp=timestamp
            )
    
    def get_strategy_name(self) -> str:
        return "DebugStrategy"