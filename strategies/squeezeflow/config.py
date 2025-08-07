"""
SqueezeFlow Strategy Configuration

Central configuration for all strategy parameters based on the methodology
documented in /docs/strategy/SqueezeFlow.md
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SqueezeFlowConfig:
    """
    Configuration for SqueezeFlow strategy
    
    IMPORTANT: This strategy uses NO fixed thresholds for CVD divergence.
    Instead, it uses pattern recognition and scoring to identify opportunities.
    """
    
    # Timeframes for multi-timeframe analysis (from SqueezeFlow.md)
    primary_timeframe: str = "5m"  # Primary entry timeframe
    reset_timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "30m"])  # Phase 3
    divergence_timeframes: List[str] = field(default_factory=lambda: ["15m", "30m"])  # Phase 2
    context_timeframes: List[str] = field(default_factory=lambda: ["30m", "1h", "4h"])  # Phase 1
    
    # Phase 4 Scoring Weights (from SqueezeFlow.md lines 145-171)
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "cvd_reset_deceleration": 3.5,     # Critical - CVD momentum exhaustion
        "absorption_candle": 2.5,          # High priority - price action validation
        "failed_breakdown": 2.0,           # Medium priority - pattern strength
        "directional_bias": 2.0            # Supporting - new trend confirmation
    })
    
    # Entry Requirements (from SqueezeFlow.md line 180: minimum 4 points)
    min_entry_score: float = 4.0  # Minimum score for entry signal generation
    
    # Position Sizing by Score (from SqueezeFlow.md lines 180-183)
    position_size_by_score: Dict[str, float] = field(default_factory=lambda: {
        "0-3.9": 0.0,    # No trade - insufficient conditions
        "4-5": 0.5,      # Reduced size - medium confidence
        "6-7": 1.0,      # Normal size - high confidence
        "8+": 1.5        # Larger size - premium setup
    })
    
    # Leverage by Score (for both backtest and live trading)
    leverage_by_score: Dict[str, int] = field(default_factory=lambda: {
        "0-3": 0,      # No trade
        "4-5": 2,      # Low leverage - medium confidence
        "6-7": 3,      # Medium leverage - high confidence
        "8+": 5        # Higher leverage - premium setup
    })
    
    # Risk Management Parameters
    base_risk_per_trade: float = 0.02  # 2% base risk per trade
    max_open_positions: int = 2         # Maximum concurrent positions
    
    # Pattern Recognition Parameters (no fixed thresholds!)
    momentum_lookback: int = 5          # Periods for momentum calculation
    volume_surge_multiplier: float = 2.0  # Significant volume = 2x recent average
    
    # Exit Management (from SqueezeFlow.md Phase 5)
    # NO fixed stop losses or profit targets - dynamic exits only
    use_fixed_stops: bool = False       # Strategy uses dynamic exits
    use_fixed_targets: bool = False     # Strategy follows flow until invalidation
    
    def get_position_size_factor(self, score: float) -> float:
        """Get position size factor based on score (UPDATED for new thresholds)"""
        if score >= 8:
            return self.position_size_by_score["8+"]
        elif score >= 5:
            return self.position_size_by_score["5-7"]
        elif score >= 3:
            return self.position_size_by_score["3-5"]
        elif score >= 1.5:
            return self.position_size_by_score["1.5-3"]
        else:
            return self.position_size_by_score["0-1.4"]
    
    def get_leverage(self, score: float) -> int:
        """Get leverage based on score (UPDATED for new thresholds)"""
        if score >= 8:
            return self.leverage_by_score["8+"]
        elif score >= 6:
            return self.leverage_by_score["6-7"]
        elif score >= 4:
            return self.leverage_by_score["4-5"]
        else:
            return self.leverage_by_score["0-3"]