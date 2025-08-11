"""
Simple Indicator Configuration for Backtesting
Controls which indicators to load/process - mainly to disable OI and other problematic data
"""

import os
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    """
    Simple flags to disable problematic indicators during backtesting
    Default: Everything enabled except OI (no data available)
    """
    
    # Core data (always required)
    enable_ohlcv: bool = True  # Can't disable - core requirement
    
    # CVD indicators - all enabled by default
    enable_spot_cvd: bool = True
    enable_futures_cvd: bool = True  
    enable_cvd_divergence: bool = True
    
    # Open Interest - ENABLED (OI data confirmed available in open_interest measurement)
    enable_open_interest: bool = True
    
    # Volume data - all enabled by default
    enable_spot_volume: bool = True
    enable_futures_volume: bool = True
    
    @classmethod
    def from_env(cls) -> 'IndicatorConfig':
        """Create config from environment variables"""
        return cls(
            enable_spot_cvd=os.getenv('BACKTEST_ENABLE_SPOT_CVD', 'true').lower() == 'true',
            enable_futures_cvd=os.getenv('BACKTEST_ENABLE_FUTURES_CVD', 'true').lower() == 'true',
            enable_cvd_divergence=os.getenv('BACKTEST_ENABLE_CVD_DIVERGENCE', 'true').lower() == 'true',
            enable_open_interest=os.getenv('BACKTEST_ENABLE_OI', 'false').lower() == 'true',
            enable_spot_volume=os.getenv('BACKTEST_ENABLE_SPOT_VOLUME', 'true').lower() == 'true',
            enable_futures_volume=os.getenv('BACKTEST_ENABLE_FUTURES_VOLUME', 'true').lower() == 'true',
        )


# Global instance
_global_config = None

def get_indicator_config() -> IndicatorConfig:
    """Get the global indicator configuration"""
    global _global_config
    if _global_config is None:
        _global_config = IndicatorConfig.from_env()
    return _global_config

def set_indicator_config(config: IndicatorConfig):
    """Set the global indicator configuration"""
    global _global_config
    _global_config = config