"""
SqueezeFlow Strategy Configuration

Central configuration for all strategy parameters based on the methodology
documented in /docs/strategy/SqueezeFlow.md

1S MODE IMPLEMENTATION (Phase 1.2):
===================================

Environment Variable: SQUEEZEFLOW_ENABLE_1S_MODE=true

Key Changes for 1s Mode:
- Primary timeframe: 1s (instead of 5m)
- Timeframe sets adjusted: [1s,5m,15m] vs [5m,15m,30m]
- Lookback calculations: Converted to seconds (5m = 300s)
- Statistical adjustments: 0.5x sensitivity for noise reduction
- Momentum lookback: 300s (5min) instead of 5 periods
- Volume thresholds: 3x multiplier for higher data density

Phase Component Updates:
- Phase 1 Context: Shortened timeframes [15m,30m,1h]
- Phase 2 Divergence: Uses [5m,15m] for faster detection
- Phase 3 Reset: Includes 1s for immediate reset signals
- Phase 4 Scoring: Statistical adjustment applied
- Phase 5 Exits: 600s (10min) lookback for exit decisions

Backward Compatibility: All existing minute+ timeframes still work
"""

from dataclasses import dataclass, field
from typing import Dict, List
import os


@dataclass
class SqueezeFlowConfig:
    """
    Configuration for SqueezeFlow strategy
    
    IMPORTANT: This strategy uses NO fixed thresholds for CVD divergence.
    Instead, it uses pattern recognition and scoring to identify opportunities.
    
    1s Mode Support: When enable_1s_mode=True:
    - Primary timeframe can be "1s"
    - All lookback calculations use seconds instead of candle counts
    - CVD analysis optimized for high-frequency data
    - Statistical calculations adjusted for higher data density
    """
    
    # 1s Mode Configuration
    enable_1s_mode: bool = field(default_factory=lambda: os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true')
    
    # Timeframes for multi-timeframe analysis (1s-aware)
    # IMPORTANT: 1s is DATA RESOLUTION, not a timeframe! Timeframes are analysis windows.
    primary_timeframe: str = field(default_factory=lambda: "5m")  # Always use 5m as primary, regardless of data resolution
    reset_timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "30m"])  # Phase 3 - analysis windows
    divergence_timeframes: List[str] = field(default_factory=lambda: ["15m", "30m"])  # Phase 2 - analysis windows
    context_timeframes: List[str] = field(default_factory=lambda: ["30m", "1h", "4h"])  # Phase 1 - analysis windows
    
    # Phase 4 Scoring Weights (from SqueezeFlow.md lines 145-171)
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "cvd_reset_deceleration": 3.5,     # Critical - CVD momentum exhaustion
        "absorption_candle": 2.5,          # High priority - price action validation
        "failed_breakdown": 2.0,           # Medium priority - pattern strength
        "directional_bias": 2.0            # Supporting - new trend confirmation
    })
    
    # Entry Requirements (from SqueezeFlow.md line 180: minimum 4 points)
    # FIXED: Lowered to 4.0 for realistic trading (was incorrectly raised to 7.0)
    min_entry_score: float = 4.0  # Minimum score for entry signal generation (realistic threshold)
    
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
    
    # Pattern Recognition Parameters (1s-aware, no fixed thresholds!)
    momentum_lookback: int = field(default_factory=lambda: 300 if os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE', 'false').lower() == 'true' else 5)  # 5min=300s in 1s mode
    volume_surge_multiplier: float = 2.0  # Significant volume = 2x recent average
    
    # Exit Management (from SqueezeFlow.md Phase 5)
    # NO fixed stop losses or profit targets - dynamic exits only
    use_fixed_stops: bool = False       # Strategy uses dynamic exits
    use_fixed_targets: bool = False     # Strategy follows flow until invalidation
    
    # Phase 2.1 Parallel Processing Configuration
    enable_parallel_processing: bool = field(default_factory=lambda: os.getenv('SQUEEZEFLOW_ENABLE_PARALLEL', 'true').lower() == 'true')
    max_parallel_workers: int = field(default_factory=lambda: min(4, int(os.getenv('SQUEEZEFLOW_MAX_WORKERS', '4'))))
    parallel_batch_size: int = field(default_factory=lambda: int(os.getenv('SQUEEZEFLOW_BATCH_SIZE', '10')))
    
    def get_position_size_factor(self, score: float) -> float:
        """Get position size factor based on score (UPDATED for new thresholds)"""
        if score >= 8:
            return self.position_size_by_score["8+"]
        elif score >= 6:
            return self.position_size_by_score["6-7"]
        elif score >= 4:
            return self.position_size_by_score["4-5"]
        else:
            return self.position_size_by_score["0-3.9"]
    
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
    
    def get_lookback_seconds(self, timeframe_minutes: int) -> int:
        """
        Get lookback in seconds for 1s mode calculations
        
        Args:
            timeframe_minutes: Original timeframe in minutes
            
        Returns:
            Lookback period in seconds (for 1s data) or original minutes (for minute+ data)
        """
        if self.enable_1s_mode:
            return timeframe_minutes * 60  # Convert to seconds for 1s granularity
        else:
            return timeframe_minutes  # Keep original for minute+ timeframes
    
    def get_adjusted_lookback(self, base_periods: int, timeframe: str = "1m") -> int:
        """
        Adjust lookback periods for 1s mode data density
        
        Args:
            base_periods: Base number of periods in minute timeframes
            timeframe: Current timeframe being analyzed
            
        Returns:
            Adjusted lookback periods for data density
        """
        if not self.enable_1s_mode:
            return base_periods
            
        # Convert timeframe to multiplier
        if timeframe == "1s":
            # For 1s data, use much larger lookbacks (60x more data points)
            return base_periods * 60
        elif timeframe == "5m":
            # For 5m analysis of 1s data, use 5x larger lookback
            return base_periods * 5
        elif timeframe == "15m":
            # For 15m analysis, keep similar lookback (data is aggregated)
            return base_periods
        else:
            return base_periods
    
    def get_statistical_adjustment(self) -> float:
        """
        Get statistical adjustment factor for 1s mode
        
        Returns:
            Adjustment factor for statistical calculations in 1s mode
        """
        if self.enable_1s_mode:
            # Reduce sensitivity due to higher noise in 1s data
            return 0.5  # 50% adjustment for noise reduction
        else:
            return 1.0  # No adjustment for minute+ data
    
    def get_momentum_smoothing_factor(self) -> float:
        """
        Get momentum smoothing factor for 1s mode noise reduction
        
        Returns:
            Smoothing factor for momentum calculations
        """
        if self.enable_1s_mode:
            return 0.1  # More smoothing for 1s data (reduce noise)
        else:
            return 0.3  # Less smoothing for minute+ data
    
    def get_volume_threshold_multiplier(self) -> float:
        """
        Get volume threshold multiplier for 1s mode
        Higher thresholds needed due to increased data granularity
        
        Returns:
            Volume threshold multiplier
        """
        if self.enable_1s_mode:
            return 3.0  # 3x higher thresholds for 1s data
        else:
            return 1.0  # Normal thresholds for minute+ data
    
    def get_optimal_worker_count(self, total_windows: int) -> int:
        """
        Get optimal worker count based on workload and system capabilities
        
        Args:
            total_windows: Total number of windows to process
            
        Returns:
            Optimal number of workers for parallel processing
        """
        if not self.enable_parallel_processing:
            return 1
            
        # For small workloads, use fewer workers
        if total_windows < 50:
            return 1
        elif total_windows < 200:
            return min(2, self.max_parallel_workers)
        else:
            return self.max_parallel_workers
    
    def get_batch_size_for_data_mode(self) -> int:
        """
        Get optimal batch size based on data mode (1s vs minute+)
        
        Returns:
            Optimal batch size for current data mode
        """
        if self.enable_1s_mode:
            # Smaller batches for 1s data to manage memory
            return min(self.parallel_batch_size, 5)
        else:
            return self.parallel_batch_size
    
    def should_enable_parallel_processing(self, total_operations: int) -> bool:
        """
        Determine if parallel processing should be enabled based on workload
        
        Args:
            total_operations: Total number of operations to perform
            
        Returns:
            True if parallel processing is beneficial
        """
        return (
            self.enable_parallel_processing and 
            self.max_parallel_workers > 1 and
            total_operations > 100  # Only worth it for substantial workloads
        )