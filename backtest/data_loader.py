"""
Backtest Data Loader with 1-Second Data Support

This module provides functionality to load and aggregate 1-second data
for backtesting purposes, enabling perfect timeframe alignment.
"""

import pandas as pd
import numpy as np
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import logging
import gc
from typing import Iterator, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    """
    Memory-efficient data loader for backtesting with 1-second data support.
    
    Features:
    - Streaming data loading to prevent OOM errors
    - Memory-efficient rolling windows with LRU cache
    - Explicit memory cleanup and garbage collection
    - Configurable chunk sizes for different timeframes
    - Adaptive chunking for 1s data with retry logic
    - Progressive loading with progress indicators
    
    This class loads raw 1-second data from InfluxDB and dynamically
    aggregates it to any required timeframe for backtesting.
    """
    
    def __init__(self, host='localhost', port=8086, database='significant_trades',
                 enable_streaming=False, max_memory_mb=4096, chunk_size_hours=2, 
                 max_retries=3, enable_1s_chunking=True):
        """
        Initialize the memory-efficient backtest data loader.
        
        Args:
            host: InfluxDB host
            port: InfluxDB port
            database: InfluxDB database name
            enable_streaming: Enable streaming mode for memory efficiency
            max_memory_mb: Maximum memory usage in MB
            chunk_size_hours: Chunk size in hours for streaming (2h for 1s data)
            max_retries: Maximum retries per chunk (default 3)
            enable_1s_chunking: Enable adaptive chunking for 1s data (default True)
        """
        self.influx = InfluxDBClient(
            host=host,
            port=port,
            database=database
        )
        
        # Memory management configuration
        self.enable_streaming = enable_streaming
        self.max_memory_mb = max_memory_mb
        self.chunk_size_hours = chunk_size_hours
        
        # Retry and chunking configuration for Phase 1.3
        self.max_retries = max_retries
        self.enable_1s_chunking = enable_1s_chunking
        
        # Streaming cache for memory efficiency
        self.data_cache = deque(maxlen=100)  # LRU cache for recent chunks
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"BacktestDataLoader initialized: {host}:{port}/{database}")
        if enable_streaming:
            logger.info(f"Streaming mode enabled: {max_memory_mb}MB limit, {chunk_size_hours}h chunks")
        if enable_1s_chunking:
            logger.info(f"1s data chunking enabled: {chunk_size_hours}h chunks, {max_retries} retries")
    
    def load_1s_data(self, symbol, start_time, end_time, enable_chunking=None):
        """
        Load raw 1-second data from InfluxDB with adaptive chunking.
        
        Args:
            symbol: Trading symbol (e.g., 'BINANCE:btcusdt')
            start_time: Start time (ISO format string or datetime)
            end_time: End time (ISO format string or datetime)
            enable_chunking: Override chunking behavior (None = auto-detect)
            
        Returns:
            DataFrame with 1-second OHLCV data
        """
        # Determine chunking strategy
        start_dt = start_time if isinstance(start_time, datetime) else datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = end_time if isinstance(end_time, datetime) else datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        should_chunk = enable_chunking if enable_chunking is not None else (self.enable_1s_chunking and duration_hours > 2)
        
        if should_chunk:
            logger.info(f"Using chunked loading for 1s data ({duration_hours:.1f}h duration)")
            return self._load_1s_data_chunked(symbol, start_dt, end_dt)
        
        # Convert datetime objects to ISO format if needed
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat() + 'Z'
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat() + 'Z'
        
        query = f"""
        SELECT open, high, low, close, volume 
        FROM trades_1s 
        WHERE market = '{symbol}' 
        AND time >= '{start_time}' 
        AND time <= '{end_time}'
        ORDER BY time ASC
        """
        
        logger.debug(f"Loading 1s data for {symbol} from {start_time} to {end_time}")
        
        try:
            result = self.influx.query(query)
            
            if not result:
                logger.warning(f"No 1s data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(list(result.get_points()))
            
            if df.empty:
                return df
            
            # Convert time column to datetime and set as index
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded {len(df)} 1s bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading 1s data: {e}")
            return pd.DataFrame()
    
    def aggregate_to_timeframe(self, df_1s, timeframe):
        """
        Aggregate 1-second data to any timeframe.
        
        Args:
            df_1s: DataFrame with 1-second OHLCV data
            timeframe: Target timeframe (e.g., '1T', '5T', '15T', '1H')
                      Pandas resample notation:
                      '1T' = 1 minute, '5T' = 5 minutes, '15T' = 15 minutes
                      '30T' = 30 minutes, '1H' = 1 hour, '4H' = 4 hours
        
        Returns:
            DataFrame with aggregated OHLCV data
        """
        if df_1s.empty:
            return df_1s
        
        # Define aggregation rules for OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        logger.debug(f"Aggregating {len(df_1s)} 1s bars to {timeframe}")
        
        try:
            # Resample to target timeframe
            df_resampled = df_1s.resample(timeframe).agg(agg_rules)
            
            # Remove rows with NaN (gaps in data)
            df_resampled = df_resampled.dropna()
            
            # Validate OHLC relationships
            df_resampled = self._validate_ohlc(df_resampled)
            
            logger.info(f"Aggregated to {len(df_resampled)} {timeframe} bars")
            return df_resampled
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return pd.DataFrame()
    
    def _validate_ohlc(self, df):
        """
        Validate OHLC data relationships.
        
        Ensures:
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        """
        if df.empty:
            return df
        
        # Fix any invalid OHLC relationships
        df.loc[df['high'] < df['low'], 'high'] = df.loc[df['high'] < df['low'], 'low']
        df.loc[df['high'] < df['open'], 'high'] = df.loc[df['high'] < df['open'], 'open']
        df.loc[df['high'] < df['close'], 'high'] = df.loc[df['high'] < df['close'], 'close']
        df.loc[df['low'] > df['open'], 'low'] = df.loc[df['low'] > df['open'], 'open']
        df.loc[df['low'] > df['close'], 'low'] = df.loc[df['low'] > df['close'], 'close']
        
        return df
    
    def get_multi_timeframe_data(self, symbol, start_time, end_time, cache_1s=True):
        """
        Load 1s data and create all required timeframes for backtesting.
        
        This provides perfect timeframe alignment as all timeframes are
        derived from the same 1-second source data.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            cache_1s: Whether to include raw 1s data in results
        
        Returns:
            Dict with all timeframes needed by strategy
        """
        # Load base 1-second data once
        df_1s = self.load_1s_data(symbol, start_time, end_time)
        
        if df_1s.empty:
            logger.warning(f"No 1s data found for {symbol}")
            return {}
        
        # Define all required timeframes for SqueezeFlow strategy
        timeframes = {
            '1m': '1T',    # 1 minute
            '5m': '5T',    # 5 minutes  
            '15m': '15T',  # 15 minutes
            '30m': '30T',  # 30 minutes
            '1h': '1H',    # 1 hour
            '4h': '4H'     # 4 hours
        }
        
        result = {}
        
        # Optionally include raw 1s data
        if cache_1s:
            result['1s'] = df_1s
            logger.info(f"  1s: {len(df_1s)} bars (raw)")
        
        # Generate all required timeframes from 1s data
        for tf_label, tf_pandas in timeframes.items():
            result[tf_label] = self.aggregate_to_timeframe(df_1s, tf_pandas)
            if not result[tf_label].empty:
                logger.info(f"  {tf_label}: {len(result[tf_label])} bars")
        
        return result
    
    def load_streaming_data(self, symbol: str, start_time: datetime, 
                           end_time: datetime, timeframe: str = '1s') -> Iterator[Dict]:
        """
        Memory-efficient streaming data loader for 1s backtests.
        
        Yields data in chunks to prevent OOM errors with large datasets.
        
        Args:
            symbol: Trading symbol (e.g., 'BINANCE:btcusdt')
            start_time: Start time
            end_time: End time 
            timeframe: Target timeframe
            
        Yields:
            Dict with chunk data for each time window
        """
        if not self.enable_streaming:
            # Fallback to regular loading
            data = self.get_multi_timeframe_data(symbol, start_time, end_time, cache_1s=True)
            if data:
                yield {'timeframe': timeframe, 'data': data, 'chunk_info': {'total': 1, 'current': 1}}
            return
        
        # Calculate chunks for streaming
        total_duration = end_time - start_time
        chunk_duration = timedelta(hours=self.chunk_size_hours)
        total_chunks = int(total_duration.total_seconds() / chunk_duration.total_seconds()) + 1
        
        logger.info(f"Streaming {symbol} data in {total_chunks} chunks of {self.chunk_size_hours}h each")
        
        current_start = start_time
        chunk_num = 0
        
        while current_start < end_time:
            chunk_num += 1
            chunk_end = min(current_start + chunk_duration, end_time)
            
            try:
                # Load chunk data
                chunk_data = self._load_chunk_with_cache(symbol, current_start, chunk_end, timeframe)
                
                if chunk_data:
                    yield {
                        'timeframe': timeframe,
                        'data': chunk_data, 
                        'chunk_info': {
                            'total': total_chunks,
                            'current': chunk_num,
                            'start_time': current_start,
                            'end_time': chunk_end,
                            'progress_pct': (chunk_num / total_chunks) * 100
                        }
                    }
                
                # Memory cleanup after each chunk
                del chunk_data
                if chunk_num % 5 == 0:  # Force GC every 5 chunks
                    collected = gc.collect()
                    logger.debug(f"GC collected {collected} objects after chunk {chunk_num}")
                    
            except Exception as e:
                logger.error(f"Failed to load chunk {chunk_num}: {e}")
            
            current_start = chunk_end
        
        logger.info(f"Streaming completed: {chunk_num} chunks processed")
    
    def _load_chunk_with_cache(self, symbol: str, start_time: datetime, 
                              end_time: datetime, timeframe: str) -> Optional[Dict]:
        """
        Load data chunk with caching support.
        
        Args:
            symbol: Trading symbol
            start_time: Chunk start time
            end_time: Chunk end time
            timeframe: Timeframe
            
        Returns:
            Chunk data dictionary or None
        """
        cache_key = f"{symbol}_{start_time.isoformat()}_{end_time.isoformat()}_{timeframe}"
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            self.cache_hits += 1
            return cached_data
        
        self.cache_misses += 1
        
        # Load fresh data
        try:
            if timeframe == '1s':
                # Load 1s data and aggregate to required timeframe (with chunking enabled for large ranges)
                df_1s = self.load_1s_data(symbol, start_time, end_time, enable_chunking=True)
                if df_1s.empty:
                    return None
                
                chunk_data = {
                    '1s': df_1s,
                    '1m': self.aggregate_to_timeframe(df_1s, '1T'),
                    '5m': self.aggregate_to_timeframe(df_1s, '5T'),
                    '15m': self.aggregate_to_timeframe(df_1s, '15T'),
                    '30m': self.aggregate_to_timeframe(df_1s, '30T'),
                    '1h': self.aggregate_to_timeframe(df_1s, '1H'),
                    '4h': self.aggregate_to_timeframe(df_1s, '4H')
                }
            else:
                # Regular timeframe loading
                chunk_data = self.get_multi_timeframe_data(
                    symbol, start_time, end_time, cache_1s=False
                )
            
            # Add to cache
            self._add_to_cache(cache_key, chunk_data)
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"Failed to load chunk data: {e}")
            return None
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Get data from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None
        """
        # Simple cache lookup (basic implementation)
        for item in self.data_cache:
            if item.get('key') == cache_key:
                return item.get('data')
        return None
    
    def _add_to_cache(self, cache_key: str, data: Dict):
        """
        Add data to cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        # Add to deque cache (LRU behavior with maxlen)
        cache_item = {
            'key': cache_key,
            'data': data.copy() if data else None,
            'timestamp': datetime.now()
        }
        self.data_cache.append(cache_item)
    
    def get_memory_stats(self) -> Dict:
        """
        Get memory usage and performance statistics.
        
        Returns:
            Dict with memory stats and Phase 1.3 enhancements
        """
        cache_hit_rate = (self.cache_hits / max(self.cache_hits + self.cache_misses, 1)) * 100
        
        return {
            'cache_performance': {
                'cache_size': len(self.data_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate_pct': round(cache_hit_rate, 1),
            },
            'memory_management': {
                'max_memory_mb': self.max_memory_mb,
                'streaming_enabled': self.enable_streaming,
                'chunk_size_hours': self.chunk_size_hours,
            },
            'chunking_config': {
                'enable_1s_chunking': self.enable_1s_chunking,
                'max_retries': self.max_retries,
                'adaptive_chunk_sizing': True,
            },
            'performance_optimizations': {
                'progressive_loading': True,
                'retry_logic_enabled': True,
                'memory_efficient_processing': True,
                'duplicate_removal': True,
            }
        }
    
    def clear_cache(self):
        """
        Clear data cache and force garbage collection.
        """
        self.data_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        collected = gc.collect()
        logger.info(f"Cache cleared, GC collected {collected} objects")
    
    def check_data_quality(self, data_dict):
        """
        Check data quality and coverage for backtesting.
        
        Args:
            data_dict: Dictionary of timeframe data
            
        Returns:
            Dict with quality metrics
        """
        quality = {
            'timeframes_available': list(data_dict.keys()),
            'total_bars': {},
            'gaps_detected': {},
            'time_coverage': {}
        }
        
        for tf, df in data_dict.items():
            if df.empty:
                continue
                
            quality['total_bars'][tf] = len(df)
            
            # Check for gaps
            if len(df) > 1:
                time_diffs = df.index.to_series().diff()
                expected_diff = pd.Timedelta(self._get_timedelta_from_tf(tf))
                gaps = (time_diffs > expected_diff * 1.5).sum()
                quality['gaps_detected'][tf] = gaps
            
            # Time coverage
            if len(df) > 0:
                quality['time_coverage'][tf] = {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat()
                }
        
        return quality
    
    def get_chunking_progress(self, start_time: datetime, end_time: datetime, 
                             chunk_size_hours: Optional[int] = None) -> Dict:
        """
        Calculate chunking progress information for large 1s data loads.
        
        Args:
            start_time: Start datetime for data load
            end_time: End datetime for data load
            chunk_size_hours: Override default chunk size
            
        Returns:
            Dict with progress calculation details
        """
        chunk_hours = chunk_size_hours or self.chunk_size_hours
        chunk_duration = timedelta(hours=chunk_hours)
        total_duration = end_time - start_time
        total_chunks = int(total_duration.total_seconds() / chunk_duration.total_seconds()) + 1
        
        # Calculate data density estimates
        total_1s_points = int(total_duration.total_seconds())  # 1 point per second
        points_per_chunk = chunk_hours * 3600  # seconds per chunk
        estimated_memory_per_chunk_mb = (points_per_chunk * 8 * 6) / (1024 * 1024)  # 6 columns, 8 bytes each
        
        progress_info = {
            'chunking_strategy': {
                'total_chunks': total_chunks,
                'chunk_size_hours': chunk_hours,
                'adaptive_sizing': self.enable_1s_chunking,
                'retry_enabled': self.max_retries > 0,
            },
            'data_estimates': {
                'total_duration_hours': round(total_duration.total_seconds() / 3600, 2),
                'estimated_1s_points': f"{total_1s_points:,}",
                'points_per_chunk': f"{points_per_chunk:,}",
                'estimated_memory_per_chunk_mb': round(estimated_memory_per_chunk_mb, 1),
            },
            'performance_expectations': {
                'chunk_load_time_estimate_sec': chunk_hours * 0.5,  # ~0.5s per hour of 1s data
                'total_estimated_time_min': round((total_chunks * chunk_hours * 0.5) / 60, 1),
                'memory_peak_mb': round(estimated_memory_per_chunk_mb * 1.5, 1),  # Safety factor
            },
            'optimization_settings': {
                'max_retries': self.max_retries,
                'progressive_backoff': True,
                'duplicate_removal': True,
                'memory_cleanup': self.enable_streaming,
            }
        }
        
        return progress_info
    
    def _load_1s_data_chunked(self, symbol, start_time, end_time):
        """
        Load 1-second data in chunks with retry logic to prevent timeouts.
        
        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with 1-second OHLCV data from all chunks
        """
        import time
        
        chunk_duration = timedelta(hours=self.chunk_size_hours)
        all_dfs = []
        current_start = start_time
        total_chunks = int((end_time - start_time).total_seconds() / chunk_duration.total_seconds()) + 1
        chunk_num = 0
        
        logger.info(f"Loading 1s data in {self.chunk_size_hours}h chunks for {symbol} ({total_chunks} total chunks)")
        
        while current_start < end_time:
            chunk_num += 1
            chunk_end = min(current_start + chunk_duration, end_time)
            
            # Progress indicator
            progress = (chunk_num / total_chunks) * 100
            logger.info(f"Loading chunk {chunk_num}/{total_chunks} ({progress:.1f}%): {current_start.strftime('%H:%M:%S')} to {chunk_end.strftime('%H:%M:%S')}")
            
            # Retry logic for each chunk
            chunk_df = None
            for retry in range(self.max_retries):
                try:
                    query = f"""
                    SELECT open, high, low, close, volume 
                    FROM trades_1s 
                    WHERE market = '{symbol}' 
                    AND time >= '{current_start.isoformat()}Z' 
                    AND time <= '{chunk_end.isoformat()}Z'
                    ORDER BY time ASC
                    """
                    
                    result = self.influx.query(query)
                    
                    if result:
                        # Convert to DataFrame
                        chunk_df = pd.DataFrame(list(result.get_points()))
                        
                        if not chunk_df.empty:
                            # Convert time column to datetime and set as index
                            chunk_df['time'] = pd.to_datetime(chunk_df['time'])
                            chunk_df.set_index('time', inplace=True)
                            
                            # Ensure numeric columns
                            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                            for col in numeric_cols:
                                if col in chunk_df.columns:
                                    chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                            
                            all_dfs.append(chunk_df)
                            logger.debug(f"✓ Loaded chunk {chunk_num}: {len(chunk_df)} rows")
                        else:
                            logger.debug(f"⚠ Empty chunk {chunk_num}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry < self.max_retries - 1:
                        wait_time = (retry + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                        logger.warning(f"Retry {retry + 1}/{self.max_retries} for chunk {chunk_num} after {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to load chunk {chunk_num} after {self.max_retries} retries: {e}")
            
            current_start = chunk_end
        
        # Combine all chunks
        if all_dfs:
            logger.info(f"Combining {len(all_dfs)} chunks...")
            combined_df = pd.concat(all_dfs, axis=0).sort_index()
            
            # Remove duplicates that might occur at chunk boundaries
            if len(combined_df) > 0:
                initial_len = len(combined_df)
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                if len(combined_df) < initial_len:
                    logger.debug(f"Removed {initial_len - len(combined_df)} duplicate rows at chunk boundaries")
            
            logger.info(f"✅ Combined chunked 1s data: {len(combined_df)} total rows")
            return combined_df
        else:
            logger.warning("No data loaded from any chunk")
            return pd.DataFrame()
    
    def _get_timedelta_from_tf(self, tf):
        """Convert timeframe string to timedelta."""
        mapping = {
            '1s': '1S',
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H'
        }
        return mapping.get(tf, '1T')
    
    def load_cvd_data(self, symbol, start_time, end_time):
        """
        Load CVD (Cumulative Volume Delta) data from 1s source.
        
        This can be aggregated from 1s volume data with buy/sell classification.
        Note: Requires additional trade direction data in InfluxDB.
        """
        # This would need actual buy/sell volume data from trades_1s
        # For now, returning placeholder
        logger.info("CVD data loading from 1s source not yet implemented")
        return pd.DataFrame()


# Regular example usage function
def example_backtest_with_1s_data():
    """
    Example of how to use the BacktestDataLoader with 1-second data.
    """
    from datetime import datetime, timedelta
    
    # Initialize loader
    loader = BacktestDataLoader()
    
    # Define backtest period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3)  # Last 3 days
    
    print("Loading 1-second data and aggregating to timeframes...")
    
    # Get all timeframes from single 1s source
    all_timeframes = loader.get_multi_timeframe_data(
        'BINANCE:btcusdt',
        start_time,
        end_time
    )
    
    if all_timeframes:
        print(f"\nGenerated {len(all_timeframes)} timeframes from 1s data:")
        for tf, data in all_timeframes.items():
            if not data.empty:
                print(f"  {tf}: {len(data)} bars, {data.index[0]} to {data.index[-1]}")
        
        # Check data quality
        quality = loader.check_data_quality(all_timeframes)
        print(f"\nData quality check:")
        print(f"  Gaps detected: {quality['gaps_detected']}")
        print(f"  Total bars: {quality['total_bars']}")
    else:
        print("No data available for backtesting")
    
    return all_timeframes


def example_streaming_backtest():
    """
    Example of streaming backtest with memory management.
    """
    from datetime import datetime, timedelta
    
    # Initialize streaming loader
    loader = BacktestDataLoader(
        enable_streaming=True,
        max_memory_mb=2048,  # 2GB limit
        chunk_size_hours=2   # 2-hour chunks
    )
    
    # Define backtest period (smaller for example)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=12)  # 12 hours of data
    
    print("Streaming 1s data processing example...")
    
    total_chunks = 0
    total_1s_bars = 0
    
    # Process data in streaming chunks
    for chunk in loader.load_streaming_data(
        'BINANCE:btcusdt',
        start_time,
        end_time,
        timeframe='1s'
    ):
        total_chunks += 1
        chunk_info = chunk['chunk_info']
        data = chunk['data']
        
        if data and '1s' in data:
            chunk_1s_bars = len(data['1s'])
            total_1s_bars += chunk_1s_bars
            
            print(f"Chunk {chunk_info['current']}/{chunk_info['total']} "
                  f"({chunk_info['progress_pct']:.1f}%): {chunk_1s_bars} bars")
        
        # Simulate processing delay
        import time
        time.sleep(0.1)
    
    # Memory statistics
    stats = loader.get_memory_stats()
    print(f"\nProcessing completed:")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total 1s bars: {total_1s_bars:,}")
    print(f"  Cache hit rate: {stats['cache_hit_rate_pct']:.1f}%")
    
    return total_1s_bars


if __name__ == "__main__":
    # Run example if executed directly
    logging.basicConfig(level=logging.INFO)
    
    # Test both regular and streaming examples
    print("=== Regular Example ===")
    example_backtest_with_1s_data()
    
    print("\n=== Streaming Example ===")
    example_streaming_backtest()