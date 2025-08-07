#!/usr/bin/env python3
"""
Optimized InfluxDB Client for SqueezeFlow Trader
High-performance database operations with query optimization and connection pooling
"""

import pandas as pd
import numpy as np
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError
import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from functools import lru_cache
import hashlib


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for database queries"""
    query_count: int = 0
    total_query_time: float = 0.0
    avg_query_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_queries: int = 0
    slow_queries: int = 0  # > 1000ms
    rows_returned: int = 0
    data_transfer_mb: float = 0.0


@dataclass
class QueryOptimization:
    """Query optimization settings"""
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_entries: int = 1000
    enable_batch_queries: bool = True
    max_batch_size: int = 10
    enable_query_hints: bool = True
    connection_pool_size: int = 25
    query_timeout_seconds: int = 300  # Extended for large backtests and DNS issues


class OptimizedInfluxClient:
    """High-performance InfluxDB client with query optimization"""
    
    def __init__(self, host='localhost', port=8086, username='', password='', 
                 database='significant_trades', optimization_config: QueryOptimization = None):
        """
        Initialize optimized InfluxDB client
        
        Args:
            host: InfluxDB host
            port: InfluxDB port  
            username: Username
            password: Password
            database: Default database
            optimization_config: Query optimization settings
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        
        self.logger = logging.getLogger(__name__)
        
        # Optimization configuration with extended timeout for DNS/connection issues
        self.config = optimization_config or QueryOptimization(
            connection_pool_size=25,  # Increased for better parallelism
            query_timeout_seconds=300  # Extended timeout for large backtests and DNS resolution
        )
        
        # Connection pool for parallel queries
        self.connection_pool = []
        self._pool_lock = threading.Lock()
        self._initialize_connection_pool()
        
        # Query cache
        self.query_cache = {}
        self.cache_metadata = {}  # Cache timestamps and metadata
        self._cache_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = QueryPerformanceMetrics()
        
        # Thread executor for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.connection_pool_size)
        
        # Pre-compiled query templates - CORRECTED for dual data structure
        # Historical data is in 'trades_1m', live data is in 'aggr_1m.trades_1m' 
        self.query_templates = {
            'historical_ohlcv_data': """
                SELECT 
                    mean(open) AS open,
                    mean(high) AS high,
                    mean(low) AS low,
                    mean(close) AS close,
                    sum(volume) AS volume
                FROM "trades_{timeframe}"
                WHERE ({market_conditions})
                AND time >= '{start_time}'
                AND time <= '{end_time}'
                GROUP BY time({timeframe}), market
                ORDER BY time ASC
            """,
            'ohlcv_data': """
                SELECT 
                    mean(open) AS open,
                    mean(high) AS high,
                    mean(low) AS low,
                    mean(close) AS close,
                    sum(vbuy) + sum(vsell) AS volume
                FROM "aggr_{timeframe}".trades_{timeframe}
                WHERE ({market_conditions})
                AND time >= '{start_time}'
                AND time <= '{end_time}'
                GROUP BY time({timeframe}), market
                ORDER BY time ASC
            """,
            
            'volume_data': """
                SELECT 
                    sum(vbuy) AS total_vbuy,
                    sum(vsell) AS total_vsell,
                    sum(cbuy) AS total_cbuy,
                    sum(csell) AS total_csell,
                    sum(lbuy) AS total_lbuy,
                    sum(lsell) AS total_lsell
                FROM "aggr_{timeframe}".trades_{timeframe}
                WHERE ({market_conditions})
                AND time >= '{start_time}'
                AND time <= '{end_time}'
                GROUP BY time({timeframe})
                FILL(0)
                ORDER BY time ASC
            """,
            
            'aggregated_volume': """
                SELECT 
                    sum(vbuy) AS total_vbuy,
                    sum(vsell) AS total_vsell,
                    sum(cbuy) AS total_cbuy,
                    sum(csell) AS total_csell,
                    sum(lbuy) AS total_lbuy,
                    sum(lsell) AS total_lsell
                FROM "aggr_{timeframe}".trades_{timeframe}
                WHERE ({market_conditions})
                AND time >= '{start_time}'
                AND time <= '{end_time}'
                GROUP BY time({group_by_interval})
                FILL(0)
                ORDER BY time ASC
            """
        }
        
        self.logger.info(f"Optimized InfluxDB client initialized with {self.config.connection_pool_size} connections")
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for parallel queries"""
        for i in range(self.config.connection_pool_size):
            try:
                client = InfluxDBClient(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    timeout=self.config.query_timeout_seconds,
                    retries=5,  # More retries for connection issues
                    ssl=False,
                    verify_ssl=False
                )
                
                # Test connection
                client.ping()
                self.connection_pool.append(client)
                
            except Exception as e:
                self.logger.error(f"Failed to create InfluxDB connection {i}: {e}")
        
        if not self.connection_pool:
            raise RuntimeError("Failed to create any InfluxDB connections")
        
        self.logger.info(f"Created {len(self.connection_pool)} InfluxDB connections")
    
    def _get_connection(self) -> InfluxDBClient:
        """Get connection from pool (thread-safe)"""
        with self._pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                # Create temporary connection if pool is exhausted
                return InfluxDBClient(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    timeout=self.config.query_timeout_seconds,
                    retries=5,
                    ssl=False,
                    verify_ssl=False
                )
    
    def _return_connection(self, client: InfluxDBClient):
        """Return connection to pool (thread-safe)"""
        with self._pool_lock:
            if len(self.connection_pool) < self.config.connection_pool_size:
                self.connection_pool.append(client)
            else:
                # Pool is full, close connection
                try:
                    client.close()
                except:
                    pass
    
    def _get_cache_key(self, query: str, database: str = None) -> str:
        """Generate cache key for query"""
        cache_data = {
            'query': query.strip(),
            'database': database or self.database,
            'timestamp': int(time.time() / self.config.cache_ttl_seconds)  # Bucket by TTL
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get query result from cache"""
        if not self.config.enable_query_cache:
            return None
        
        with self._cache_lock:
            if cache_key in self.query_cache:
                cache_time = self.cache_metadata.get(cache_key, {}).get('timestamp', 0)
                
                # Check if cache entry is still valid
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    self.metrics.cache_hits += 1
                    return self.query_cache[cache_key].copy()  # Return copy to avoid mutations
                else:
                    # Remove expired entry
                    del self.query_cache[cache_key]
                    if cache_key in self.cache_metadata:
                        del self.cache_metadata[cache_key]
            
            self.metrics.cache_misses += 1
            return None
    
    def _store_in_cache(self, cache_key: str, df: pd.DataFrame):
        """Store query result in cache"""
        if not self.config.enable_query_cache or df.empty:
            return
        
        with self._cache_lock:
            # Evict oldest entries if cache is full
            if len(self.query_cache) >= self.config.max_cache_entries:
                # Remove 10% of oldest entries
                sorted_entries = sorted(
                    self.cache_metadata.items(),
                    key=lambda x: x[1].get('timestamp', 0)
                )
                evict_count = max(1, len(sorted_entries) // 10)
                
                for key, _ in sorted_entries[:evict_count]:
                    if key in self.query_cache:
                        del self.query_cache[key]
                    if key in self.cache_metadata:
                        del self.cache_metadata[key]
            
            # Store new entry
            self.query_cache[cache_key] = df.copy()
            self.cache_metadata[cache_key] = {
                'timestamp': time.time(),
                'size_bytes': df.memory_usage(deep=True).sum(),
                'rows': len(df)
            }
    
    async def execute_query_async(self, query: str, database: str = None, 
                                enable_cache: bool = True) -> pd.DataFrame:
        """Execute single query asynchronously with caching"""
        start_time = time.time()
        
        try:
            # Check cache first
            if enable_cache:
                cache_key = self._get_cache_key(query, database)
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute query in thread pool
            df = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._execute_query_sync, query, database
            )
            
            # Store in cache
            if enable_cache and not df.empty:
                self._store_in_cache(cache_key, df)
            
            # Update metrics
            query_time = time.time() - start_time
            self._update_metrics(query_time, len(df), df.memory_usage(deep=True).sum())
            
            return df
            
        except Exception as e:
            self.metrics.failed_queries += 1
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def _execute_query_sync(self, query: str, database: str = None) -> pd.DataFrame:
        """Execute query synchronously"""
        client = None
        try:
            client = self._get_connection()
            
            # Set database if specified
            if database and database != self.database:
                client.switch_database(database)
            
            # Execute query
            result = client.query(query)
            
            # Convert to pandas DataFrame
            if result:
                # Handle multiple series in result
                dfs = []
                for series in result:
                    if series:
                        df = pd.DataFrame(series)
                        if not df.empty and 'time' in df.columns:
                            # InfluxDB stores data in UTC - ensure timezone-aware timestamps
                            df['time'] = pd.to_datetime(df['time'], utc=True)
                            df.set_index('time', inplace=True)
                        dfs.append(df)
                
                if dfs:
                    # Concatenate multiple DataFrames if needed
                    if len(dfs) == 1:
                        return dfs[0]
                    else:
                        # Filter out empty DataFrames to avoid FutureWarning
                        non_empty_dfs = [df for df in dfs if not df.empty]
                        if non_empty_dfs:
                            return pd.concat(non_empty_dfs, axis=0, sort=True)
                        else:
                            return pd.DataFrame()
            
            return pd.DataFrame()
            
        finally:
            if client:
                self._return_connection(client)
    
    async def execute_batch_queries_async(self, queries: List[Tuple[str, str]]) -> List[pd.DataFrame]:
        """Execute multiple queries in parallel"""
        if not self.config.enable_batch_queries:
            # Execute sequentially
            results = []
            for query, database in queries:
                result = await self.execute_query_async(query, database)
                results.append(result)
            return results
        
        # Execute in parallel batches
        results = []
        batch_size = min(self.config.max_batch_size, len(queries))
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_tasks = [
                self.execute_query_async(query, database)
                for query, database in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch query failed: {result}")
                    results.append(pd.DataFrame())  # Empty DataFrame for failed queries
                else:
                    results.append(result)
        
        return results
    
    async def get_ohlcv_data_optimized(self, markets: List[str], start_time: datetime,
                                     end_time: datetime, timeframe: str = '5m') -> pd.DataFrame:
        """Optimized OHLCV data retrieval with intelligent batching"""
        try:
            # Create simpler market pattern for better InfluxDB compatibility
            # Instead of complex regex, use simple OR conditions
            # Process all markets for complete data - essential for accurate CVD calculation
            market_conditions = " OR ".join([f"market = '{market}'" for market in markets])
            if len(markets) > 50:
                self.logger.debug(f"Processing {len(markets)} markets for complete OHLCV coverage")
            
            # Note: We'll modify the query template to use market_conditions instead of regex
            
            # Use pre-compiled query template
            query = self.query_templates['ohlcv_data'].format(
                timeframe=timeframe,
                market_conditions=market_conditions,
                start_time=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time=end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            # Add query optimization hints
            if self.config.enable_query_hints:
                query = self._add_query_hints(query, 'ohlcv', len(markets))
            
            # Execute optimized query
            df = await self.execute_query_async(query, 'significant_trades')
            
            if not df.empty and len(markets) > 1:
                # Aggregate data across markets for multiple markets
                df = self._aggregate_ohlcv_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized OHLCV data retrieval failed: {e}")
            return pd.DataFrame()
    
    async def get_volume_data_optimized(self, markets: List[str], start_time: datetime,
                                      end_time: datetime, timeframe: str = '5m') -> pd.DataFrame:
        """Optimized volume data retrieval with aggregation"""
        try:
            # For large market lists, use batch processing
            if len(markets) > 20:
                return await self._get_volume_data_batched(markets, start_time, end_time, timeframe)
            
            # Create simple market conditions for better InfluxDB compatibility
            # Process all markets, not just first 10 - we need complete data for CVD calculation
            market_conditions = " OR ".join([f"market = '{market}'" for market in markets])
            if len(markets) > 50:
                self.logger.debug(f"Processing {len(markets)} markets for complete CVD coverage")
            
            # Use aggregated query for better performance
            query = self.query_templates['aggregated_volume'].format(
                timeframe=timeframe,
                market_conditions=market_conditions,
                start_time=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time=end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                group_by_interval=timeframe
            )
            
            # Add query hints
            if self.config.enable_query_hints:
                query = self._add_query_hints(query, 'volume', len(markets))
            
            # Execute query
            df = await self.execute_query_async(query, 'significant_trades')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized volume data retrieval failed: {e}")
            return pd.DataFrame()
    
    async def _get_volume_data_batched(self, markets: List[str], start_time: datetime,
                                     end_time: datetime, timeframe: str) -> pd.DataFrame:
        """Get volume data using batched queries for large market lists"""
        batch_size = 10  # Process 10 markets per query
        all_dfs = []
        
        # Create batched queries
        queries = []
        for i in range(0, len(markets), batch_size):
            batch_markets = markets[i:i + batch_size]
            market_conditions = " OR ".join([f"market = '{market}'" for market in batch_markets])
            
            query = self.query_templates['volume_data'].format(
                timeframe=timeframe,
                market_conditions=market_conditions,
                start_time=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time=end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            queries.append((query, 'significant_trades'))
        
        # Execute batch queries
        results = await self.execute_batch_queries_async(queries)
        
        # Combine results
        valid_dfs = [df for df in results if not df.empty]
        if valid_dfs:
            combined_df = pd.concat(valid_dfs, axis=0)
            
            # Group by time and sum volumes
            grouped = combined_df.groupby(combined_df.index).sum()
            return grouped.sort_index()
        
        return pd.DataFrame()
    
    def _add_query_hints(self, query: str, query_type: str, market_count: int) -> str:
        """Add InfluxDB query optimization hints"""
        # Disable query hints for now as they cause parsing errors
        # The InfluxDB version in the container may not support these hints
        return query
    
    def _aggregate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate OHLCV data across multiple markets"""
        if df.empty:
            return df
        
        try:
            # Group by time and calculate OHLCV
            grouped = df.groupby(df.index).agg({
                'open': 'first',    # First market's open
                'high': 'max',      # Maximum high across markets
                'low': 'min',       # Minimum low across markets
                'close': 'last',    # Last market's close
                'volume': 'sum'     # Total volume across markets
            })
            
            return grouped.sort_index()
            
        except Exception as e:
            self.logger.error(f"OHLCV aggregation failed: {e}")
            return df
    
    def _update_metrics(self, query_time: float, rows_returned: int, data_transfer_bytes: float):
        """Update query performance metrics"""
        self.metrics.query_count += 1
        self.metrics.total_query_time += query_time
        self.metrics.avg_query_time = self.metrics.total_query_time / self.metrics.query_count
        self.metrics.rows_returned += rows_returned
        self.metrics.data_transfer_mb += data_transfer_bytes / 1024 / 1024
        
        # Count slow queries (> 1000ms)
        if query_time > 1.0:
            self.metrics.slow_queries += 1
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        total_queries = self.metrics.query_count
        cache_total = self.metrics.cache_hits + self.metrics.cache_misses
        
        return {
            'query_performance': {
                'total_queries': total_queries,
                'avg_query_time_ms': round(self.metrics.avg_query_time * 1000, 2),
                'total_query_time_sec': round(self.metrics.total_query_time, 2),
                'slow_queries_count': self.metrics.slow_queries,
                'slow_query_rate_percent': round((self.metrics.slow_queries / max(1, total_queries)) * 100, 2),
                'failed_queries': self.metrics.failed_queries,
                'failure_rate_percent': round((self.metrics.failed_queries / max(1, total_queries)) * 100, 2)
            },
            'cache_performance': {
                'cache_enabled': self.config.enable_query_cache,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'cache_hit_rate_percent': round((self.metrics.cache_hits / max(1, cache_total)) * 100, 2),
                'cache_size': len(self.query_cache),
                'max_cache_size': self.config.max_cache_entries,
                'cache_memory_mb': self._get_cache_memory_usage()
            },
            'data_transfer': {
                'total_rows_returned': self.metrics.rows_returned,
                'total_data_transfer_mb': round(self.metrics.data_transfer_mb, 2),
                'avg_rows_per_query': round(self.metrics.rows_returned / max(1, total_queries), 0),
                'avg_data_per_query_kb': round((self.metrics.data_transfer_mb * 1024) / max(1, total_queries), 2)
            },
            'connection_pool': {
                'pool_size': self.config.connection_pool_size,
                'available_connections': len(self.connection_pool),
                'pool_utilization_percent': round(((self.config.connection_pool_size - len(self.connection_pool)) / self.config.connection_pool_size) * 100, 2)
            },
            'optimization_settings': {
                'query_cache_enabled': self.config.enable_query_cache,
                'batch_queries_enabled': self.config.enable_batch_queries,
                'query_hints_enabled': self.config.enable_query_hints,
                'cache_ttl_seconds': self.config.cache_ttl_seconds,
                'max_batch_size': self.config.max_batch_size
            }
        }
    
    def _get_cache_memory_usage(self) -> float:
        """Calculate cache memory usage in MB"""
        total_bytes = sum(
            metadata.get('size_bytes', 0) 
            for metadata in self.cache_metadata.values()
        )
        return round(total_bytes / 1024 / 1024, 2)
    
    def clear_cache(self):
        """Clear query cache"""
        with self._cache_lock:
            self.query_cache.clear()
            self.cache_metadata.clear()
        self.logger.info("Query cache cleared")
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize settings for specific workload types"""
        if workload_type == 'high_frequency':
            # Optimize for high-frequency, small queries
            self.config.cache_ttl_seconds = 60  # Shorter cache TTL
            self.config.max_batch_size = 15     # Larger batches
            self.config.enable_query_hints = True
            
        elif workload_type == 'bulk_analysis':
            # Optimize for large, infrequent queries
            self.config.cache_ttl_seconds = 1800  # Longer cache TTL (30 min)
            self.config.max_batch_size = 5        # Smaller batches to avoid timeouts
            self.config.query_timeout_seconds = 60
            
        elif workload_type == 'real_time':
            # Optimize for real-time processing
            self.config.cache_ttl_seconds = 30    # Very short cache TTL
            self.config.max_batch_size = 20       # Large batches for efficiency
            self.config.enable_query_hints = True
        
        self.logger.info(f"Optimized settings for workload: {workload_type}")
    
    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        health = {
            'database_connectivity': False,
            'connection_pool_healthy': False,
            'query_performance_ok': False,
            'cache_performance_ok': False,
            'overall_healthy': False
        }
        
        try:
            # Test basic connectivity
            test_query = "SHOW DATABASES"
            result = await self.execute_query_async(test_query, enable_cache=False)
            health['database_connectivity'] = not result.empty
            
            # Check connection pool
            health['connection_pool_healthy'] = len(self.connection_pool) > 0
            
            # Check query performance
            avg_query_time_ms = self.metrics.avg_query_time * 1000
            health['query_performance_ok'] = avg_query_time_ms < 2000  # < 2 seconds
            
            # Check cache performance
            cache_total = self.metrics.cache_hits + self.metrics.cache_misses
            cache_hit_rate = (self.metrics.cache_hits / max(1, cache_total)) * 100
            health['cache_performance_ok'] = cache_hit_rate > 30  # > 30% cache hit rate
            
            health['overall_healthy'] = all([
                health['database_connectivity'],
                health['connection_pool_healthy'],
                health['query_performance_ok']
            ])
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
        
        return health
    
    # Compatibility methods for data pipeline (synchronous versions)
    def get_ohlcv_data(self, markets: List[str], start_time: datetime,
                      end_time: datetime, timeframe: str = '5m') -> pd.DataFrame:
        """Synchronous compatibility wrapper for get_ohlcv_data_optimized"""
        import asyncio
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    self.get_ohlcv_data_optimized(markets, start_time, end_time, timeframe)
                )
                return future.result()
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(self.get_ohlcv_data_optimized(markets, start_time, end_time, timeframe))
    
    def get_volume_data(self, markets: List[str], start_time: datetime,
                       end_time: datetime, timeframe: str = '5m') -> pd.DataFrame:
        """Synchronous compatibility wrapper for get_volume_data_optimized"""
        import asyncio
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    self.get_volume_data_optimized(markets, start_time, end_time, timeframe)
                )
                return future.result()
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(self.get_volume_data_optimized(markets, start_time, end_time, timeframe))
    
    def close(self):
        """Close all connections and cleanup"""
        try:
            # Close all connections in pool
            for client in self.connection_pool:
                try:
                    client.close()
                except:
                    pass
            
            # Shutdown thread executor
            self.executor.shutdown(wait=True)
            
            # Clear caches
            self.clear_cache()
            
            self.logger.info("InfluxDB client closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing InfluxDB client: {e}")


# Factory function for easy integration
def create_optimized_influx_client(config: Dict) -> OptimizedInfluxClient:
    """Create optimized InfluxDB client from configuration with smart host detection"""
    import os
    
    # Smart host detection - fix for DNS resolution issues
    default_host = 'localhost'
    
    # Check if we're in Docker environment
    if os.path.exists('/.dockerenv'):
        # Inside Docker container - use service name
        default_host = 'aggr-influx'
    elif os.getenv('DOCKER_HOST'):
        # Docker Desktop environment - use localhost
        default_host = 'localhost'
    elif config.get('influx_host') == 'aggr-influx':
        # Force localhost if aggr-influx specified but we're not in Docker
        try:
            import socket
            socket.gethostbyname('aggr-influx')
            default_host = 'aggr-influx'  # DNS resolves, use it
        except socket.gaierror:
            default_host = 'localhost'  # DNS fails, fallback to localhost
            print(f"⚠️  DNS resolution failed for 'aggr-influx', using localhost instead")
    
    optimization_config = QueryOptimization(
        enable_query_cache=config.get('enable_query_cache', True),
        cache_ttl_seconds=config.get('cache_ttl_seconds', 300),
        max_cache_entries=config.get('max_cache_entries', 1000),
        enable_batch_queries=config.get('enable_batch_queries', True),
        max_batch_size=config.get('max_batch_size', 10),
        connection_pool_size=config.get('connection_pool_size', 25),
        query_timeout_seconds=config.get('query_timeout_seconds', 300)  # Extended timeout for backtests
    )
    
    final_host = config.get('influx_host', default_host)
    
    return OptimizedInfluxClient(
        host=final_host,
        port=config.get('influx_port', 8086),
        username=config.get('influx_username', ''),
        password=config.get('influx_password', ''),
        database=config.get('influx_database', 'significant_trades'),
        optimization_config=optimization_config
    )