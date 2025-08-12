"""THE configuration system - single source of truth for EVERYTHING"""
import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class VeloceConfig:
    """Complete system configuration - THE only config class"""
    
    # ========== Data Sources ==========
    influx_host: str = "213.136.75.120"  # Production server
    influx_port: int = 8086
    influx_database: str = "significant_trades"
    influx_username: Optional[str] = None
    influx_password: Optional[str] = None
    influx_timeout: int = 30
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_max_connections: int = 50
    
    # ========== Strategy Configuration ==========
    strategy_enabled: bool = True
    strategy_mode: str = "production"  # production, paper, backtest
    strategy_symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    
    # Squeeze Parameters (from indicator_config.py)
    squeeze_period: int = 20
    squeeze_bb_mult: float = 2.0
    squeeze_kc_mult: float = 1.5
    squeeze_momentum_length: int = 12
    squeeze_momentum_smooth: int = 3
    
    # CVD Parameters
    cvd_enabled: bool = True
    cvd_lookback: int = 100
    cvd_threshold: float = 0.02
    cvd_divergence_min_strength: float = 0.5
    
    # Open Interest Parameters
    oi_enabled: bool = True
    oi_threshold: float = 5.0
    oi_lookback_minutes: int = 30
    oi_exchanges: List[str] = field(default_factory=lambda: ["BINANCE_FUTURES", "BYBIT", "OKX"])
    oi_aggregation: str = "FUTURES_AGG"  # or "TOTAL_AGG"
    
    # Phase Thresholds (from phase files)
    phase1_volume_threshold: float = 1.5
    phase1_squeeze_threshold: float = 0.0
    phase2_divergence_threshold: float = 0.02
    phase3_reset_threshold: float = 0.8
    phase4_min_score: float = 60.0
    phase5_exit_multiplier: float = 2.0
    
    # Multi-Timeframe Settings
    timeframes: List[str] = field(default_factory=lambda: ["1s", "1m", "5m", "15m", "30m", "1h", "4h"])
    primary_timeframe: str = "1s"
    mtf_alignment_required: int = 3  # How many timeframes must align
    default_lookback_hours: int = 4  # Default historical data lookback
    
    # ========== Risk Management ==========
    position_size: float = 0.01     # 1% default position size
    max_position_size: float = 0.1  # 10% of capital
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_mult: float = 2.0   # 2:1 risk/reward
    max_daily_trades: int = 20
    max_concurrent_positions: int = 3
    min_trade_interval: int = 60  # seconds between trades
    min_signal_interval: int = 5  # minutes between signals
    
    # ========== Performance Settings ==========
    enable_1s_mode: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 60  # seconds
    batch_size: int = 1000
    parallel_processing: bool = True
    num_workers: int = 4
    query_timeout: int = 30
    
    # ========== Monitoring ==========
    metrics_enabled: bool = True
    metrics_port: int = 9090
    logging_level: str = "INFO"
    log_format: str = "json"
    log_to_file: bool = True
    log_file_path: str = "veloce.log"
    tracing_enabled: bool = False
    tracing_endpoint: Optional[str] = None
    health_check_interval: int = 60  # seconds
    
    # ========== API Settings ==========
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None
    api_rate_limit: int = 100  # requests per minute
    websocket_enabled: bool = True
    websocket_port: int = 8001
    
    # ========== Backtest Settings ==========
    backtest_slippage: float = 0.001  # 0.1%
    backtest_commission: float = 0.001  # 0.1%
    backtest_initial_capital: float = 10000.0
    backtest_data_start_offset: int = 1000  # candles before start for indicators
    
    # ========== Visualization Settings ==========
    dashboard_theme: str = "dark"
    dashboard_auto_refresh: bool = True
    dashboard_refresh_interval: int = 5  # seconds
    chart_height: int = 400
    chart_candle_limit: int = 500
    export_png_enabled: bool = True
    export_png_width: int = 1920
    export_png_height: int = 1080
    
    # ========== Notification Settings ==========
    notifications_enabled: bool = False
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    
    @classmethod
    def from_env(cls) -> 'VeloceConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        for field_name in config.__dataclass_fields__:
            env_name = f"VELOCE_{field_name.upper()}"
            if env_name in os.environ:
                field_type = config.__dataclass_fields__[field_name].type
                value = os.environ[env_name]
                
                # Type conversion
                if field_type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                elif hasattr(field_type, '__origin__'):
                    if field_type.__origin__ == list:
                        value = json.loads(value) if value.startswith('[') else value.split(',')
                    elif field_type.__origin__ == Optional:
                        value = None if value.lower() == 'none' else value
                
                setattr(config, field_name, value)
        
        config.validate()
        return config
    
    @classmethod
    def from_file(cls, filepath: str) -> 'VeloceConfig':
        """Load configuration from YAML/JSON file"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        config = cls(**data)
        config.validate()
        return config
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        errors = []
        
        # Risk validation
        if self.stop_loss_pct >= self.max_position_size:
            errors.append("Stop loss cannot exceed max position size")
        
        # OI validation
        if self.oi_threshold < 0 or self.oi_threshold > 100:
            errors.append("OI threshold must be between 0-100")
        
        # Timeframe validation
        if self.primary_timeframe not in self.timeframes:
            errors.append(f"Primary timeframe {self.primary_timeframe} not in timeframes list")
        
        # Position validation
        if self.max_concurrent_positions < 1:
            errors.append("Must allow at least 1 concurrent position")
        
        # Score validation
        if self.phase4_min_score < 0 or self.phase4_min_score > 100:
            errors.append("Phase 4 min score must be between 0-100")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export configuration as JSON"""
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Export configuration as YAML"""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def get_indicator_config(self) -> Dict[str, Any]:
        """Get indicator-specific configuration (replaces indicator_config.py)"""
        return {
            'squeeze': {
                'period': self.squeeze_period,
                'bb_mult': self.squeeze_bb_mult,
                'kc_mult': self.squeeze_kc_mult,
                'momentum_length': self.squeeze_momentum_length,
                'momentum_smooth': self.squeeze_momentum_smooth
            },
            'cvd': {
                'enabled': self.cvd_enabled,
                'lookback': self.cvd_lookback,
                'threshold': self.cvd_threshold,
                'divergence_min_strength': self.cvd_divergence_min_strength
            },
            'oi': {
                'enabled': self.oi_enabled,
                'threshold': self.oi_threshold,
                'lookback_minutes': self.oi_lookback_minutes,
                'exchanges': self.oi_exchanges,
                'aggregation': self.oi_aggregation
            },
            'phases': {
                'phase1': {
                    'volume_threshold': self.phase1_volume_threshold,
                    'squeeze_threshold': self.phase1_squeeze_threshold
                },
                'phase2': {
                    'divergence_threshold': self.phase2_divergence_threshold
                },
                'phase3': {
                    'reset_threshold': self.phase3_reset_threshold
                },
                'phase4': {
                    'min_score': self.phase4_min_score
                },
                'phase5': {
                    'exit_multiplier': self.phase5_exit_multiplier
                }
            }
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data provider configuration"""
        return {
            'influx': {
                'host': self.influx_host,
                'port': self.influx_port,
                'database': self.influx_database,
                'username': self.influx_username,
                'password': self.influx_password,
                'timeout': self.influx_timeout
            },
            'redis': {
                'host': self.redis_host,
                'port': self.redis_port,
                'db': self.redis_db,
                'password': self.redis_password,
                'max_connections': self.redis_max_connections
            },
            'cache': {
                'enabled': self.cache_enabled,
                'ttl': self.cache_ttl
            },
            'performance': {
                'batch_size': self.batch_size,
                'parallel': self.parallel_processing,
                'workers': self.num_workers
            }
        }


# Create global singleton - THE configuration
try:
    # Try to load from environment first
    CONFIG = VeloceConfig.from_env()
except:
    # Fall back to defaults
    CONFIG = VeloceConfig()
    CONFIG.validate()

# Export
__all__ = ['VeloceConfig', 'CONFIG']