#!/usr/bin/env python3
"""
Service Configuration - Configuration management for SqueezeFlow services
Centralized configuration handling for the Strategy Runner Service
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServiceConfig:
    """Configuration for SqueezeFlow services"""
    
    # Service execution settings
    run_interval_seconds: int = 60  # Run strategy every minute
    max_symbols_per_cycle: int = 10  # Process max 10 symbols per cycle
    enable_parallel_processing: bool = True
    
    # Data settings
    data_lookback_hours: int = 4  # Load 4 hours of data for analysis (reduced from 48)
    min_data_points: int = 40  # Minimum data points required (reduced from 500, ~40 for 4h of 5m bars)
    default_timeframe: str = "5m"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_signal_ttl: int = 600  # 10 minutes TTL for signals (to handle FreqTrade timing)
    redis_key_prefix: str = "squeezeflow"
    
    # InfluxDB settings
    influx_host: str = "localhost"
    influx_port: int = 8086
    influx_database: str = "significant_trades"
    influx_username: str = ""
    influx_password: str = ""
    
    # Signal generation settings
    publish_to_redis: bool = True
    store_in_influxdb: bool = True
    signal_retention_hours: int = 24
    
    # Enhanced Redis publishing settings
    enable_batch_publishing: bool = True
    max_batch_size: int = 10
    batch_timeout_seconds: int = 5
    redis_connection_pool_size: int = 20
    
    # Signal validation settings
    enable_signal_validation: bool = True
    enable_signal_deduplication: bool = True
    max_signals_per_minute: int = 20
    max_signals_per_symbol_per_hour: int = 10
    validation_cleanup_interval_hours: int = 12
    
    # Strategy settings
    strategy_name: str = "SqueezeFlowStrategy"
    enable_position_sizing: bool = True
    enable_leverage_scaling: bool = True
    
    # Risk management
    max_concurrent_signals: int = 5
    signal_cooldown_minutes: int = 15  # Cooldown between signals for same symbol
    
    # FreqTrade API settings
    freqtrade_api_url: str = "http://localhost:8080"
    freqtrade_api_username: str = "0xGang"
    freqtrade_api_password: str = "0xGang"
    freqtrade_api_timeout: int = 10
    enable_freqtrade_integration: bool = True
    
    # CVD baseline tracking
    enable_cvd_baseline_tracking: bool = True
    cvd_baseline_storage_key: str = "cvd_baselines"
    
    # Monitoring and logging
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    health_check_interval: int = 30  # seconds
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: int = 5
    graceful_shutdown_timeout: int = 30


class ConfigManager:
    """Manages service configuration from multiple sources"""
    
    def __init__(self, config_dir: str = "services/config"):
        self.config_dir = Path(config_dir)
        self.config = ServiceConfig()
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from multiple sources with priority"""
        
        # 1. Load from config file if exists
        config_file = self.config_dir / "service_config.yaml"
        if config_file.exists():
            self._load_from_yaml(config_file)
        
        # 2. Override with environment variables
        self._load_from_environment()
        
        # 3. Validate configuration
        self._validate_configuration()
    
    def _load_from_yaml(self, config_file: Path):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config:
                for key, value in yaml_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        env_mappings = {
            'SQUEEZEFLOW_RUN_INTERVAL': 'run_interval_seconds',
            'SQUEEZEFLOW_MAX_SYMBOLS': 'max_symbols_per_cycle',
            'SQUEEZEFLOW_LOOKBACK_HOURS': 'data_lookback_hours',
            'SQUEEZEFLOW_MIN_DATA_POINTS': 'min_data_points',
            'SQUEEZEFLOW_TIMEFRAME': 'default_timeframe',
            'REDIS_HOST': 'redis_host',
            'REDIS_PORT': 'redis_port',
            'REDIS_DB': 'redis_db',
            'SQUEEZEFLOW_ENABLE_BATCH': 'enable_batch_publishing',
            'SQUEEZEFLOW_MAX_BATCH_SIZE': 'max_batch_size',
            'SQUEEZEFLOW_BATCH_TIMEOUT': 'batch_timeout_seconds',
            'SQUEEZEFLOW_ENABLE_VALIDATION': 'enable_signal_validation',
            'SQUEEZEFLOW_MAX_SIGNALS_PER_MIN': 'max_signals_per_minute',
            'SQUEEZEFLOW_COOLDOWN_MINUTES': 'signal_cooldown_minutes',
            'INFLUX_HOST': 'influx_host',
            'INFLUX_PORT': 'influx_port',
            'INFLUX_DATABASE': 'influx_database',
            'INFLUX_USERNAME': 'influx_username',
            'INFLUX_PASSWORD': 'influx_password',
            'FREQTRADE_API_URL': 'freqtrade_api_url',
            'FREQTRADE_API_USERNAME': 'freqtrade_api_username',
            'FREQTRADE_API_PASSWORD': 'freqtrade_api_password',
            'FREQTRADE_API_TIMEOUT': 'freqtrade_api_timeout',
            'FREQTRADE_ENABLE_INTEGRATION': 'enable_freqtrade_integration',
            'SQUEEZEFLOW_LOG_LEVEL': 'log_level'
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Type conversion - check bool first since bool is subclass of int
                current_value = getattr(self.config, config_attr)
                if isinstance(current_value, bool):
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    env_value = int(env_value)
                
                setattr(self.config, config_attr, env_value)
    
    def _validate_configuration(self):
        """Validate configuration values"""
        
        # Validate intervals
        if self.config.run_interval_seconds < 10:
            raise ValueError("Run interval must be at least 10 seconds")
        
        if self.config.data_lookback_hours < 1:
            raise ValueError("Data lookback must be at least 1 hour")
        
        # Validate batch publishing settings
        if self.config.enable_batch_publishing:
            if self.config.max_batch_size < 1 or self.config.max_batch_size > 100:
                raise ValueError("Batch size must be between 1 and 100")
            
            if self.config.batch_timeout_seconds < 1 or self.config.batch_timeout_seconds > 300:
                raise ValueError("Batch timeout must be between 1 and 300 seconds")
        
        # Validate rate limiting settings
        if self.config.max_signals_per_minute < 1 or self.config.max_signals_per_minute > 1000:
            raise ValueError("Max signals per minute must be between 1 and 1000")
        
        if self.config.signal_cooldown_minutes < 0 or self.config.signal_cooldown_minutes > 1440:
            raise ValueError("Signal cooldown must be between 0 and 1440 minutes (24 hours)")
        
        # Validate Redis settings
        if not (1 <= self.config.redis_port <= 65535):
            raise ValueError("Redis port must be between 1 and 65535")
        
        # Validate InfluxDB settings  
        if not (1 <= self.config.influx_port <= 65535):
            raise ValueError("InfluxDB port must be between 1 and 65535")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if self.config.default_timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.config.default_timeframe}")
    
    def get_config(self) -> ServiceConfig:
        """Get current configuration"""
        return self.config
    
    def get_freqtrade_pairs(self, freqtrade_config_path: str = "freqtrade/user_data/config.json") -> List[str]:
        """
        Extract trading pairs from FreqTrade configuration
        
        Args:
            freqtrade_config_path: Path to FreqTrade config file
            
        Returns:
            List of base symbols (e.g., ['BTC', 'ETH'])
        """
        try:
            config_path = Path(freqtrade_config_path)
            if not config_path.exists():
                print(f"Warning: FreqTrade config not found at {config_path}")
                return ['BTC', 'ETH']  # Default pairs
            
            with open(config_path, 'r') as f:
                freqtrade_config = json.load(f)
            
            # Extract pair whitelist
            pair_whitelist = freqtrade_config.get('exchange', {}).get('pair_whitelist', [])
            
            # Convert to base symbols (e.g., "BTC/USDT:USDT" -> "BTC")
            base_symbols = []
            for pair in pair_whitelist:
                if '/' in pair:
                    base_symbol = pair.split('/')[0]
                    if base_symbol not in base_symbols:
                        base_symbols.append(base_symbol)
            
            return base_symbols if base_symbols else ['BTC', 'ETH']
            
        except Exception as e:
            print(f"Warning: Could not load FreqTrade config: {e}")
            return ['BTC', 'ETH']  # Default fallback
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis connection configuration"""
        redis_config = {
            'host': self.config.redis_host,
            'port': self.config.redis_port,
            'db': self.config.redis_db,
            'decode_responses': True,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            # retry_on_timeout deprecated - use retry with Retry object instead
            'health_check_interval': 30
        }
        
        # Add connection pool settings if batch publishing is enabled
        if getattr(self.config, 'enable_batch_publishing', True):
            redis_config.update({
                'max_connections': getattr(self.config, 'redis_connection_pool_size', 20),
                'socket_keepalive': True,
                'socket_keepalive_options': {}
            })
        
        return redis_config
    
    def get_influx_config(self) -> Dict[str, Any]:
        """Get InfluxDB connection configuration with smart host detection"""
        import socket
        import os
        
        # Smart host detection for Docker/localhost issues
        configured_host = self.config.influx_host
        final_host = configured_host
        
        # If configured host is aggr-influx, test DNS resolution
        if configured_host == 'aggr-influx':
            try:
                socket.gethostbyname('aggr-influx')
                final_host = 'aggr-influx'  # DNS works, use it
            except socket.gaierror:
                final_host = 'localhost'  # DNS fails, use localhost
                print(f"⚠️  DNS resolution failed for 'aggr-influx', using localhost")
        
        return {
            'host': final_host,
            'port': self.config.influx_port,
            'database': self.config.influx_database,
            'username': self.config.influx_username,
            'password': self.config.influx_password,
            'timeout': 120,  # Extended timeout for connection issues
            'retries': 5,
            'ssl': False,
            'verify_ssl': False
        }
    
    def get_freqtrade_config(self) -> Dict[str, Any]:
        """Get FreqTrade API connection configuration"""
        return {
            'api_url': self.config.freqtrade_api_url,
            'username': self.config.freqtrade_api_username,
            'password': self.config.freqtrade_api_password,
            'timeout': self.config.freqtrade_api_timeout,
            'enabled': self.config.enable_freqtrade_integration
        }
    
    def create_sample_config_file(self):
        """Create a sample configuration file"""
        
        sample_config = {
            'run_interval_seconds': 60,
            'max_symbols_per_cycle': 10,
            'data_lookback_hours': 48,
            'default_timeframe': '5m',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'enable_batch_publishing': True,
            'max_batch_size': 10,
            'batch_timeout_seconds': 5,
            'enable_signal_validation': True,
            'enable_signal_deduplication': True,
            'max_signals_per_minute': 20,
            'max_signals_per_symbol_per_hour': 10,
            'signal_cooldown_minutes': 15,
            'influx_host': 'localhost',
            'influx_port': 8086,
            'influx_database': 'significant_trades',
            'log_level': 'INFO',
            'enable_performance_monitoring': True,
            'max_concurrent_signals': 5
        }
        
        config_file = self.config_dir / "service_config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        print(f"Sample configuration created at: {config_file}")


# Factory function for easy import
def create_config_manager(config_dir: str = "services/config") -> ConfigManager:
    """Factory function to create configuration manager"""
    return ConfigManager(config_dir)


# Convenience function for direct config access
def get_service_config(config_dir: str = "services/config") -> ServiceConfig:
    """Get service configuration directly"""
    manager = ConfigManager(config_dir)
    return manager.get_config()