#!/usr/bin/env python3
"""
Unified Configuration System - Environment Variables Only
All configuration comes from docker-compose.yml environment variables
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class UnifiedConfig:
    """Single source of truth for all service configuration"""
    
    # Strategy Runner Settings
    run_interval: int
    max_symbols: int
    lookback_hours: int
    timeframe: str
    min_data_points: int
    log_level: str
    
    # Redis Settings
    redis_host: str
    redis_port: int
    redis_db: int
    
    # InfluxDB Settings
    influx_host: str
    influx_port: int
    influx_database: str
    influx_username: str
    influx_password: str
    
    # FreqTrade Integration
    freqtrade_api_url: str
    freqtrade_api_username: str
    freqtrade_api_password: str
    freqtrade_api_timeout: int
    freqtrade_enable_integration: bool
    
    @classmethod
    def from_env(cls) -> 'UnifiedConfig':
        """Load all configuration from environment variables only"""
        return cls(
            # Strategy Runner Settings
            run_interval=int(os.getenv('SQUEEZEFLOW_RUN_INTERVAL', '60')),
            max_symbols=int(os.getenv('SQUEEZEFLOW_MAX_SYMBOLS', '5')),
            lookback_hours=int(os.getenv('SQUEEZEFLOW_LOOKBACK_HOURS', '4')),
            timeframe=os.getenv('SQUEEZEFLOW_TIMEFRAME', '5m'),
            min_data_points=int(os.getenv('SQUEEZEFLOW_MIN_DATA_POINTS', '40')),
            log_level=os.getenv('SQUEEZEFLOW_LOG_LEVEL', 'INFO'),
            
            # Redis Settings
            redis_host=os.getenv('REDIS_HOST', 'redis'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            
            # InfluxDB Settings
            influx_host=os.getenv('INFLUX_HOST', 'aggr-influx'),
            influx_port=int(os.getenv('INFLUX_PORT', '8086')),
            influx_database=os.getenv('INFLUX_DATABASE', 'significant_trades'),
            influx_username=os.getenv('INFLUX_USERNAME', ''),
            influx_password=os.getenv('INFLUX_PASSWORD', ''),
            
            # FreqTrade Integration
            freqtrade_api_url=os.getenv('FREQTRADE_API_URL', 'http://freqtrade:8080'),
            freqtrade_api_username=os.getenv('FREQTRADE_API_USERNAME', '0xGang'),
            freqtrade_api_password=os.getenv('FREQTRADE_API_PASSWORD', '0xGang'),
            freqtrade_api_timeout=int(os.getenv('FREQTRADE_API_TIMEOUT', '10')),
            freqtrade_enable_integration=os.getenv('FREQTRADE_ENABLE_INTEGRATION', 'true').lower() == 'true'
        )
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis connection configuration"""
        return {
            'host': self.redis_host,
            'port': self.redis_port,
            'db': self.redis_db,
            'decode_responses': True,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'health_check_interval': 30,
            'max_connections': 20,
            'socket_keepalive': True,
            'socket_keepalive_options': {}
        }
    
    def get_influx_config(self) -> Dict[str, Any]:
        """Get InfluxDB connection configuration"""
        import socket
        
        # Smart host detection for Docker/localhost
        final_host = self.influx_host
        if self.influx_host == 'aggr-influx':
            try:
                socket.gethostbyname('aggr-influx')
                final_host = 'aggr-influx'
            except socket.gaierror:
                final_host = 'localhost'
                print(f"⚠️ DNS resolution failed for 'aggr-influx', using localhost")
        
        return {
            'host': final_host,
            'port': self.influx_port,
            'database': self.influx_database,
            'username': self.influx_username,
            'password': self.influx_password,
            'timeout': 120,
            'retries': 5,
            'ssl': False,
            'verify_ssl': False
        }
    
    def get_freqtrade_config(self) -> Dict[str, Any]:
        """Get FreqTrade API configuration"""
        return {
            'api_url': self.freqtrade_api_url,
            'username': self.freqtrade_api_username,
            'password': self.freqtrade_api_password,
            'timeout': self.freqtrade_api_timeout,
            'enabled': self.freqtrade_enable_integration
        }
    
    def get_freqtrade_pairs(self) -> List[str]:
        """Get trading pairs from FreqTrade config file"""
        try:
            config_path = Path('/app/freqtrade/user_data/config.json')
            if not config_path.exists():
                # Try local path if not in container
                config_path = Path('freqtrade/config/config.json')
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    freqtrade_config = json.load(f)
                
                pair_whitelist = freqtrade_config.get('exchange', {}).get('pair_whitelist', [])
                base_symbols = []
                for pair in pair_whitelist:
                    if '/' in pair:
                        base_symbol = pair.split('/')[0]
                        if base_symbol not in base_symbols:
                            base_symbols.append(base_symbol)
                
                return base_symbols if base_symbols else ['BTC', 'ETH']
        except Exception as e:
            print(f"Warning: Could not load FreqTrade pairs: {e}")
        
        return ['BTC', 'ETH']  # Default pairs


# Singleton instance
_config = None

def get_config() -> UnifiedConfig:
    """Get the singleton configuration instance"""
    global _config
    if _config is None:
        _config = UnifiedConfig.from_env()
    return _config


# Backward compatibility wrapper
class ServiceConfig:
    """Backward compatibility wrapper for old ServiceConfig class"""
    
    def __init__(self):
        self._config = get_config()
    
    @property
    def run_interval_seconds(self):
        return self._config.run_interval
    
    @property
    def max_symbols_per_cycle(self):
        return self._config.max_symbols
    
    @property
    def data_lookback_hours(self):
        return self._config.lookback_hours
    
    @property
    def default_timeframe(self):
        return self._config.timeframe
    
    @property
    def min_data_points(self):
        return self._config.min_data_points
    
    @property
    def log_level(self):
        return self._config.log_level
    
    @property
    def redis_host(self):
        return self._config.redis_host
    
    @property
    def redis_port(self):
        return self._config.redis_port
    
    @property
    def redis_db(self):
        return self._config.redis_db
    
    @property
    def influx_host(self):
        return self._config.influx_host
    
    @property
    def influx_port(self):
        return self._config.influx_port
    
    @property
    def influx_database(self):
        return self._config.influx_database
    
    @property
    def influx_username(self):
        return self._config.influx_username
    
    @property
    def influx_password(self):
        return self._config.influx_password
    
    @property
    def freqtrade_api_url(self):
        return self._config.freqtrade_api_url
    
    @property
    def freqtrade_api_username(self):
        return self._config.freqtrade_api_username
    
    @property
    def freqtrade_api_password(self):
        return self._config.freqtrade_api_password
    
    @property
    def freqtrade_api_timeout(self):
        return self._config.freqtrade_api_timeout
    
    @property
    def enable_freqtrade_integration(self):
        return self._config.freqtrade_enable_integration
    
    # Additional properties for compatibility
    enable_position_sizing = True
    enable_leverage_scaling = True
    enable_signal_validation = True
    enable_signal_deduplication = True
    max_signals_per_minute = 20
    max_signals_per_symbol_per_hour = 10
    signal_cooldown_minutes = 15
    redis_signal_ttl = 600
    redis_key_prefix = 'squeezeflow'
    publish_to_redis = True
    store_in_influxdb = True
    signal_retention_hours = 24
    enable_batch_publishing = True
    max_batch_size = 10
    batch_timeout_seconds = 5
    redis_connection_pool_size = 20
    validation_cleanup_interval_hours = 12
    strategy_name = 'SqueezeFlowStrategy'
    max_concurrent_signals = 5
    enable_cvd_baseline_tracking = True
    cvd_baseline_storage_key = 'cvd_baselines'
    enable_performance_monitoring = True
    health_check_interval = 30
    max_retries = 3
    retry_delay_seconds = 5
    graceful_shutdown_timeout = 30
    enable_parallel_processing = True


class ConfigManager:
    """Backward compatibility wrapper for ConfigManager"""
    
    def __init__(self, config_dir: str = None):
        self.config = ServiceConfig()
    
    def get_config(self) -> ServiceConfig:
        return self.config
    
    def get_redis_config(self) -> Dict[str, Any]:
        return get_config().get_redis_config()
    
    def get_influx_config(self) -> Dict[str, Any]:
        return get_config().get_influx_config()
    
    def get_freqtrade_config(self) -> Dict[str, Any]:
        return get_config().get_freqtrade_config()
    
    def get_freqtrade_pairs(self, freqtrade_config_path: str = None) -> List[str]:
        return get_config().get_freqtrade_pairs()


# Factory functions for backward compatibility
def create_config_manager(config_dir: str = None) -> ConfigManager:
    return ConfigManager(config_dir)

def get_service_config(config_dir: str = None) -> ServiceConfig:
    return ServiceConfig()