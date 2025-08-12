"""THE data provider - single access pattern for ALL data"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from influxdb import InfluxDBClient
import redis
import logging
from functools import lru_cache
import hashlib
import json

from veloce.core.config import VeloceConfig, CONFIG
from veloce.core.protocols import DataProvider as DataProviderProtocol

logger = logging.getLogger(__name__)


class VeloceDataProvider(DataProviderProtocol):
    """Unified data access layer - THE way to get data"""
    
    def __init__(self, config: VeloceConfig = CONFIG):
        """Initialize data provider with configuration"""
        self.config = config
        self.influx_client = self._setup_influx()
        self.redis_client = self._setup_redis() if config.redis_host else None
        self.cache = {} if config.cache_enabled else None
        self._connection_pool = {}
        
        logger.info(f"VeloceDataProvider initialized with InfluxDB at {config.influx_host}:{config.influx_port}")
    
    def _setup_influx(self) -> InfluxDBClient:
        """Initialize InfluxDB connection"""
        return InfluxDBClient(
            host=self.config.influx_host,
            port=self.config.influx_port,
            username=self.config.influx_username,
            password=self.config.influx_password,
            database=self.config.influx_database,
            timeout=self.config.influx_timeout
        )
    
    def _setup_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        return redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=True,
            max_connections=self.config.redis_max_connections
        )
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_retention_policy(self, timeframe: str) -> str:
        """Get appropriate retention policy for timeframe"""
        if timeframe == '1s':
            return 'aggr_1s'
        elif timeframe in ['1m', '5m']:
            return 'rp_5m' if self._check_retention_policy_exists('rp_5m') else 'autogen'
        elif timeframe in ['15m', '30m']:
            return 'rp_15m' if self._check_retention_policy_exists('rp_15m') else 'autogen'
        else:
            return 'autogen'
    
    def _check_retention_policy_exists(self, rp_name: str) -> bool:
        """Check if retention policy exists"""
        try:
            result = self.influx_client.query("SHOW RETENTION POLICIES")
            for series in result:
                for point in series:
                    if point['name'] == rp_name:
                        return True
        except:
            pass
        return False
    
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str,
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """Get OHLCV data - THE method for price data"""
        
        # Check cache first
        if self.cache is not None:
            cache_key = self._get_cache_key('ohlcv', symbol, timeframe, start, end)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for OHLCV {symbol} {timeframe}")
                return self.cache[cache_key]
        
        # Determine retention policy
        retention_policy = self._get_retention_policy(timeframe)
        
        # Build query - handles both 1s and other timeframes
        if timeframe == '1s':
            query = f"""
            SELECT 
                first(open) as open,
                max(high) as high,
                min(low) as low,
                last(close) as close,
                sum(volume) as volume,
                sum(buy_volume) as buy_volume,
                sum(sell_volume) as sell_volume
            FROM {retention_policy}.trades_{timeframe}
            WHERE market = 'BINANCE:{symbol.lower()}usdt'
            AND time >= '{start.isoformat()}Z'
            AND time <= '{end.isoformat()}Z'
            GROUP BY time({timeframe})
            ORDER BY time ASC
            """
        else:
            # For non-1s timeframes, aggregate from base data
            query = f"""
            SELECT 
                mean(close) as open,
                max(high) as high,
                min(low) as low,
                last(close) as close,
                sum(volume) as volume,
                sum(buy_volume) as buy_volume,
                sum(sell_volume) as sell_volume
            FROM trades_{timeframe}
            WHERE market = 'BINANCE:{symbol.lower()}usdt'
            AND time >= '{start.isoformat()}Z'
            AND time <= '{end.isoformat()}Z'
            GROUP BY time({timeframe})
            ORDER BY time ASC
            """
        
        try:
            # Execute query
            result = self.influx_client.query(query)
            
            # Convert to DataFrame
            df = pd.DataFrame(result.get_points())
            
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Ensure numeric types
                for col in ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill any NaN values
                df.fillna(method='ffill', inplace=True)
                
                # Cache result
                if self.cache is not None:
                    self.cache[cache_key] = df
                    
                logger.debug(f"Retrieved {len(df)} OHLCV records for {symbol} {timeframe}")
            else:
                logger.warning(f"No OHLCV data found for {symbol} {timeframe} between {start} and {end}")
                df = pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            df = pd.DataFrame()
        
        return df
    
    def get_cvd(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get CVD data - THE method for volume delta"""
        
        # Check cache
        if self.cache is not None:
            cache_key = self._get_cache_key('cvd', symbol, start, end)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for CVD {symbol}")
                return self.cache[cache_key]
        
        # Query for spot and perp volume data
        query = f"""
        SELECT 
            sum(buy_volume) as buy_volume,
            sum(sell_volume) as sell_volume
        FROM trades_1m
        WHERE time >= '{start.isoformat()}Z'
        AND time <= '{end.isoformat()}Z'
        GROUP BY market, time(1m)
        ORDER BY time ASC
        """
        
        try:
            result = self.influx_client.query(query)
            
            # Separate spot and perp data
            spot_data = []
            perp_data = []
            
            for series in result:
                market = series['tags'].get('market', '')
                if f'{symbol.lower()}usdt' in market and 'perp' not in market.lower():
                    # Spot market
                    for point in series['values']:
                        spot_data.append(point)
                elif f'{symbol.lower()}' in market and ('perp' in market.lower() or 'futures' in market.lower()):
                    # Perp/futures market
                    for point in series['values']:
                        perp_data.append(point)
            
            # Create DataFrames
            spot_df = pd.DataFrame(spot_data) if spot_data else pd.DataFrame()
            perp_df = pd.DataFrame(perp_data) if perp_data else pd.DataFrame()
            
            # Process and merge
            df = pd.DataFrame()
            
            if not spot_df.empty:
                spot_df['time'] = pd.to_datetime(spot_df['time'])
                spot_df.set_index('time', inplace=True)
                spot_df['spot_cvd'] = (spot_df['buy_volume'] - spot_df['sell_volume']).fillna(0)
                spot_df['spot_cvd_cumulative'] = spot_df['spot_cvd'].cumsum()
                df = spot_df[['spot_cvd', 'spot_cvd_cumulative']]
            
            if not perp_df.empty:
                perp_df['time'] = pd.to_datetime(perp_df['time'])
                perp_df.set_index('time', inplace=True)
                perp_df['perp_cvd'] = (perp_df['buy_volume'] - perp_df['sell_volume']).fillna(0)
                perp_df['perp_cvd_cumulative'] = perp_df['perp_cvd'].cumsum()
                
                if df.empty:
                    df = perp_df[['perp_cvd', 'perp_cvd_cumulative']]
                else:
                    df = df.join(perp_df[['perp_cvd', 'perp_cvd_cumulative']], how='outer')
            
            # Fill NaN values
            df.fillna(0, inplace=True)
            
            # Calculate divergence
            if 'spot_cvd_cumulative' in df.columns and 'perp_cvd_cumulative' in df.columns:
                df['cvd_divergence'] = df['spot_cvd_cumulative'] - df['perp_cvd_cumulative']
            
            # Cache result
            if self.cache is not None and not df.empty:
                self.cache[cache_key] = df
            
            logger.debug(f"Retrieved {len(df)} CVD records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error retrieving CVD data: {e}")
            df = pd.DataFrame()
        
        return df
    
    def get_oi(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get Open Interest data - THE method for OI"""
        
        if not self.config.oi_enabled:
            return None
        
        # Check cache
        if self.cache is not None:
            cache_key = self._get_cache_key('oi', symbol)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for OI {symbol}")
                return self.cache[cache_key]
        
        # Get latest OI from configured exchanges
        query = f"""
        SELECT last(value) as oi, last(change_pct) as oi_change
        FROM open_interest
        WHERE symbol = '{symbol}'
        AND time >= now() - {self.config.oi_lookback_minutes}m
        GROUP BY exchange
        ORDER BY time DESC
        """
        
        try:
            result = self.influx_client.query(query)
            
            oi_data = {}
            total_oi = 0
            total_change = 0
            exchange_count = 0
            
            for series in result:
                exchange = series['tags'].get('exchange', 'UNKNOWN')
                
                # Filter by configured exchanges
                if exchange not in self.config.oi_exchanges and exchange != self.config.oi_aggregation:
                    continue
                
                points = list(series['values'])
                if points and points[0]:
                    oi_value = points[0].get('oi', 0)
                    oi_change = points[0].get('oi_change', 0)
                    
                    if oi_value:
                        oi_data[exchange] = {
                            'value': float(oi_value),
                            'change': float(oi_change) if oi_change else 0.0
                        }
                        total_oi += float(oi_value)
                        total_change += float(oi_change) if oi_change else 0.0
                        exchange_count += 1
            
            # Add aggregate metrics
            if oi_data:
                # Use configured aggregation or calculate total
                if self.config.oi_aggregation in oi_data:
                    oi_data['TOTAL'] = oi_data[self.config.oi_aggregation]
                elif exchange_count > 0:
                    oi_data['TOTAL'] = {
                        'value': total_oi,
                        'change': total_change / exchange_count  # Average change
                    }
                
                # Cache result
                if self.cache is not None:
                    self.cache[cache_key] = oi_data
                
                logger.debug(f"Retrieved OI data for {symbol}: {len(oi_data)} exchanges")
            else:
                logger.warning(f"No OI data found for {symbol}")
            
            return oi_data if oi_data else None
            
        except Exception as e:
            logger.error(f"Error retrieving OI data: {e}")
            return None
    
    def get_multi_timeframe(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get data for all configured timeframes"""
        
        mtf_data = {}
        
        for timeframe in self.config.timeframes:
            # Calculate appropriate lookback for each timeframe
            lookback = self._calculate_lookback(timeframe)
            start = timestamp - lookback
            
            # Get OHLCV for this timeframe
            df = self.get_ohlcv(symbol, timeframe, start, timestamp)
            
            if not df.empty:
                mtf_data[timeframe] = df
                logger.debug(f"Retrieved {len(df)} records for {symbol} {timeframe}")
        
        logger.info(f"Retrieved multi-timeframe data for {symbol}: {list(mtf_data.keys())}")
        return mtf_data
    
    def _calculate_lookback(self, timeframe: str) -> timedelta:
        """Calculate appropriate lookback period for timeframe"""
        lookbacks = {
            '1s': timedelta(minutes=30),    # 30 minutes for 1s
            '1m': timedelta(hours=4),       # 4 hours for 1m
            '5m': timedelta(hours=12),      # 12 hours for 5m
            '15m': timedelta(days=1),       # 1 day for 15m
            '30m': timedelta(days=2),       # 2 days for 30m
            '1h': timedelta(days=5),        # 5 days for 1h
            '4h': timedelta(days=20),       # 20 days for 4h
            '1d': timedelta(days=100)       # 100 days for daily
        }
        
        # Use config offset for backtest
        if self.config.strategy_mode == 'backtest':
            # Add extra data for indicator warmup
            base_lookback = lookbacks.get(timeframe, timedelta(days=1))
            extra_candles = self.config.backtest_data_start_offset
            
            # Convert candles to time based on timeframe
            if '1s' in timeframe:
                extra_time = timedelta(seconds=extra_candles)
            elif 'm' in timeframe:
                minutes = int(timeframe.replace('m', ''))
                extra_time = timedelta(minutes=minutes * extra_candles)
            elif 'h' in timeframe:
                hours = int(timeframe.replace('h', ''))
                extra_time = timedelta(hours=hours * extra_candles)
            else:
                extra_time = timedelta(days=extra_candles)
            
            return base_lookback + extra_time
        
        return lookbacks.get(timeframe, timedelta(days=1))
    
    def clear_cache(self) -> None:
        """Clear data cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Data cache cleared")
    
    def close(self):
        """Close connections"""
        try:
            if self.influx_client:
                self.influx_client.close()
            if self.redis_client:
                self.redis_client.close()
            logger.info("Data provider connections closed")
        except:
            pass


# Create global singleton - THE data provider
DATA = VeloceDataProvider(CONFIG)

# Export
__all__ = ['VeloceDataProvider', 'DATA']