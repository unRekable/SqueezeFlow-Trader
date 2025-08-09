#!/usr/bin/env python3
"""
Unified Data Pipeline - Central data coordination for SqueezeFlow Trader
Coordinates loaders and processors for clean data delivery to strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .loaders.influx_client import OptimizedInfluxClient
from .loaders.symbol_discovery import SymbolDiscovery
from .loaders.market_discovery import MarketDiscovery
from .processors.exchange_mapper import ExchangeMapper
from .processors.cvd_calculator import CVDCalculator


class DataPipeline:
    """Unified data pipeline for SqueezeFlow Trader"""
    
    def __init__(self):
        # Initialize components
        # Smart host detection for InfluxDB connection issues
        import os
        import socket
        
        # Improved host detection logic
        influx_host = 'localhost'  # Safe default
        
        # Check environment variable first
        if os.getenv('INFLUX_HOST'):
            influx_host = os.getenv('INFLUX_HOST')
        # Check if we're in Docker
        elif os.path.exists('/.dockerenv'):
            influx_host = 'aggr-influx'
        # Test DNS resolution as final check
        elif influx_host == 'localhost':
            try:
                # Try to resolve aggr-influx, use it if available
                socket.gethostbyname('aggr-influx')
                influx_host = 'aggr-influx'
            except socket.gaierror:
                # DNS resolution failed, stick with localhost
                influx_host = 'localhost'
        
        # Create client with extended timeout and increased connection pool
        from .loaders.influx_client import QueryOptimization
        optimization_config = QueryOptimization(
            connection_pool_size=25,  # Increased for better parallelism
            query_timeout_seconds=300,  # Extended timeout for large backtests
            enable_query_cache=True,
            cache_ttl_seconds=300
        )
        
        self.influx_client = OptimizedInfluxClient(
            host=influx_host,
            port=8086,
            username='',
            password='',
            database='significant_trades',
            optimization_config=optimization_config
        )
        # Initialize other components with the same host config
        self.symbol_discovery = SymbolDiscovery(influx_host=influx_host)
        self.market_discovery = MarketDiscovery(influx_host=influx_host)
        self.exchange_mapper = ExchangeMapper()
        self.cvd_calculator = CVDCalculator()
        
        # Log the connection details for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"InfluxDB connection configured: {influx_host}:8086")
        
        # Cache for discovered data
        self._symbol_cache = {}
        self._market_cache = {}
    
    def discover_available_symbols(self, min_data_points: int = 500, 
                                  hours_lookback: int = 24) -> List[str]:
        """
        Discover available symbols with sufficient data
        
        Args:
            min_data_points: Minimum data points required
            hours_lookback: Hours to look back for data validation
            
        Returns:
            List of available symbols
        """
        cache_key = f"{min_data_points}_{hours_lookback}"
        if cache_key in self._symbol_cache:
            return self._symbol_cache[cache_key]
        
        symbols = self.symbol_discovery.discover_symbols_from_database(
            min_data_points=min_data_points,
            hours_lookback=hours_lookback
        )
        
        self._symbol_cache[cache_key] = symbols
        return symbols
    
    def discover_markets_for_symbol(self, symbol: str) -> Dict[str, List[str]]:
        """
        Discover available markets for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dict with 'spot' and 'perp' market lists
        """
        if symbol in self._market_cache:
            return self._market_cache[symbol]
        
        markets = self.market_discovery.get_markets_by_type(symbol)
        self._market_cache[symbol] = markets
        return markets
    
    def load_raw_ohlcv_data(self, symbol: str, start_time: datetime, 
                           end_time: datetime, timeframe: str = '5m') -> pd.DataFrame:
        """
        Load raw OHLCV data for a symbol with query splitting for large date ranges
        
        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            timeframe: Timeframe (5m, 15m, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Get markets for symbol
        markets = self.discover_markets_for_symbol(symbol)
        all_markets = markets.get('spot', []) + markets.get('perp', [])
        
        if not all_markets:
            return pd.DataFrame()
        
        # Check if we need to split the query based on date range
        date_diff = (end_time - start_time).days
        
        # Use adaptive chunk strategies based on timeframe and data density
        if timeframe == '1s':
            # For 1s data: use 2-hour chunks if > 2.4 hours of data
            if date_diff > 0.1:  # 0.1 days = 2.4 hours
                return self._load_ohlcv_data_chunked_1s(all_markets, start_time, end_time, timeframe, chunk_hours=2)
            else:
                # Small range, load directly
                return self.influx_client.get_ohlcv_data(
                    markets=all_markets,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
        elif date_diff > 3:  # 3 days for regular timeframes
            return self._load_ohlcv_data_chunked(all_markets, start_time, end_time, timeframe)
        
        # Load data from InfluxDB for smaller ranges
        return self.influx_client.get_ohlcv_data(
            markets=all_markets,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )
    
    def load_raw_volume_data(self, symbol: str, start_time: datetime, 
                            end_time: datetime, timeframe: str = '5m') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw volume data separated by market type with query splitting for large date ranges
        
        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            timeframe: Timeframe
            
        Returns:
            Tuple of (spot_df, futures_df) with volume data
        """
        # Get classified markets
        markets = self.discover_markets_for_symbol(symbol)
        spot_markets = markets.get('spot', [])
        perp_markets = markets.get('perp', [])
        
        # Check if we need to split the query based on date range
        date_diff = (end_time - start_time).days
        
        # Load spot volume data with adaptive chunking for 1s
        spot_df = pd.DataFrame()
        if spot_markets:
            if timeframe == '1s' and date_diff > 0.1:  # 2.4 hours for 1s data
                spot_df = self._load_volume_data_chunked_1s(spot_markets, start_time, end_time, timeframe, 'spot')
            elif date_diff > 3:
                spot_df = self._load_volume_data_chunked(spot_markets, start_time, end_time, timeframe, 'spot')
            else:
                spot_df = self.influx_client.get_volume_data(
                    markets=spot_markets,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
            
            # Rename columns to match CVD calculator expectations
            if not spot_df.empty:
                column_mapping = {
                    'total_vbuy': 'total_vbuy_spot',
                    'total_vsell': 'total_vsell_spot',
                    'total_cbuy': 'total_cbuy_spot',
                    'total_csell': 'total_csell_spot',
                    'total_lbuy': 'total_lbuy_spot',
                    'total_lsell': 'total_lsell_spot'
                }
                spot_df = spot_df.rename(columns=column_mapping)
        
        # Load futures volume data with adaptive chunking for 1s
        futures_df = pd.DataFrame()
        if perp_markets:
            if timeframe == '1s' and date_diff > 0.1:  # 2.4 hours for 1s data
                futures_df = self._load_volume_data_chunked_1s(perp_markets, start_time, end_time, timeframe, 'futures')
            elif date_diff > 3:
                futures_df = self._load_volume_data_chunked(perp_markets, start_time, end_time, timeframe, 'futures')
            else:
                futures_df = self.influx_client.get_volume_data(
                    markets=perp_markets,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
            
            # Rename columns to match CVD calculator expectations
            if not futures_df.empty:
                column_mapping = {
                    'total_vbuy': 'total_vbuy_futures',
                    'total_vsell': 'total_vsell_futures',
                    'total_cbuy': 'total_cbuy_futures',
                    'total_csell': 'total_csell_futures',
                    'total_lbuy': 'total_lbuy_futures',
                    'total_lsell': 'total_lsell_futures'
                }
                futures_df = futures_df.rename(columns=column_mapping)
        
        return spot_df, futures_df
    
    def calculate_cvd_data(self, spot_df: pd.DataFrame, 
                          futures_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate CVD data from volume dataframes
        
        Args:
            spot_df: Spot volume dataframe
            futures_df: Futures volume dataframe
            
        Returns:
            Dict with CVD series and divergence
        """
        result = {
            'spot_cvd': pd.Series(dtype=float),
            'futures_cvd': pd.Series(dtype=float),
            'cvd_divergence': pd.Series(dtype=float)
        }
        
        # Fill NaN values with 0 before CVD calculation
        if not spot_df.empty:
            spot_df = spot_df.fillna(0)
            result['spot_cvd'] = self.cvd_calculator.calculate_spot_cvd(spot_df)
        
        if not futures_df.empty:
            futures_df = futures_df.fillna(0)
            result['futures_cvd'] = self.cvd_calculator.calculate_futures_cvd(futures_df)
        
        # Calculate divergence
        if not result['spot_cvd'].empty and not result['futures_cvd'].empty:
            result['cvd_divergence'] = self.cvd_calculator.calculate_cvd_divergence(
                result['spot_cvd'], result['futures_cvd']
            )
        
        return result
    
    async def get_complete_dataset_async(self, symbol: str, start_time: datetime, 
                                       end_time: datetime, timeframe: str = '5m',
                                       prefer_1s_data: bool = True, max_lookback_minutes: int = 30) -> Dict:
        """
        Get complete dataset for strategy analysis with 1-second data support
        
        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            timeframe: Target timeframe for aggregation
            prefer_1s_data: Try to use 1-second data first
            max_lookback_minutes: Limit lookback for real-time efficiency
            
        Returns:
            Dict with all required data for strategy
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Get markets for symbol
        markets = self.discover_markets_for_symbol(symbol)
        all_markets = markets.get('spot', []) + markets.get('perp', [])
        
        if not all_markets:
            logger.warning(f"No markets found for symbol {symbol}")
            return self._empty_dataset(symbol, timeframe, start_time, end_time)
        
        # Try 1-second data first if preferred and for real-time processing
        if prefer_1s_data and (end_time - start_time).total_seconds() <= max_lookback_minutes * 60:
            try:
                ohlcv_df, combined_volume_df = await self.influx_client.get_1s_data_with_aggregation(
                    markets=all_markets,
                    start_time=start_time,
                    end_time=end_time,
                    target_timeframe=timeframe,
                    max_lookback_minutes=max_lookback_minutes
                )
                
                if not ohlcv_df.empty and not combined_volume_df.empty:
                    # Split volume data by market type
                    spot_df, futures_df = self._split_volume_by_market_type(
                        combined_volume_df, markets
                    )
                    
                    # Calculate CVD data
                    cvd_data = self.calculate_cvd_data(spot_df, futures_df)
                    
                    logger.info(f"Successfully loaded 1s data for {symbol} ({timeframe}): {len(ohlcv_df)} bars")
                    
                    return {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'start_time': start_time,
                        'end_time': end_time,
                        'ohlcv': ohlcv_df,
                        'spot_volume': spot_df,
                        'futures_volume': futures_df,
                        'spot_cvd': cvd_data['spot_cvd'],
                        'futures_cvd': cvd_data['futures_cvd'],
                        'cvd_divergence': cvd_data['cvd_divergence'],
                        'markets': markets,
                        'data_source': '1s_aggregated',
                        'metadata': {
                            'spot_markets_count': len(markets.get('spot', [])),
                            'futures_markets_count': len(markets.get('perp', [])),
                            'data_points': len(ohlcv_df),
                            'lookback_limited': True
                        }
                    }
                    
            except Exception as e:
                logger.warning(f"1s data loading failed for {symbol}, falling back to regular data: {e}")
        
        # Fallback to regular data loading
        logger.debug(f"Loading regular data for {symbol}")
        return self.get_complete_dataset(symbol, start_time, end_time, timeframe)
    
    def get_complete_dataset(self, symbol: str, start_time: datetime, 
                            end_time: datetime, timeframe: str = '5m') -> Dict:
        """
        Get complete dataset for strategy analysis (synchronous fallback)
        
        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            timeframe: Timeframe
            
        Returns:
            Dict with all required data for strategy
        """
        # Load raw data
        ohlcv_df = self.load_raw_ohlcv_data(symbol, start_time, end_time, timeframe)
        spot_df, futures_df = self.load_raw_volume_data(symbol, start_time, end_time, timeframe)
        
        # Calculate CVD data
        cvd_data = self.calculate_cvd_data(spot_df, futures_df)
        
        # Get market information
        markets = self.discover_markets_for_symbol(symbol)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': end_time,
            'ohlcv': ohlcv_df,
            'spot_volume': spot_df,
            'futures_volume': futures_df,
            'spot_cvd': cvd_data['spot_cvd'],
            'futures_cvd': cvd_data['futures_cvd'],
            'cvd_divergence': cvd_data['cvd_divergence'],
            'markets': markets,
            'data_source': 'regular',
            'metadata': {
                'spot_markets_count': len(markets.get('spot', [])),
                'futures_markets_count': len(markets.get('perp', [])),
                'data_points': len(ohlcv_df),
                'lookback_limited': False
            }
        }
    
    def validate_data_quality(self, dataset: Dict) -> Dict[str, bool]:
        """
        Validate data quality for trading strategy
        
        Args:
            dataset: Complete dataset from get_complete_dataset()
            
        Returns:
            Dict with validation results
        """
        validations = {
            'has_price_data': not dataset['ohlcv'].empty,
            'has_spot_volume': not dataset['spot_volume'].empty,
            'has_futures_volume': not dataset['futures_volume'].empty,
            'has_spot_cvd': not dataset['spot_cvd'].empty,
            'has_futures_cvd': not dataset['futures_cvd'].empty,
            'has_cvd_divergence': not dataset['cvd_divergence'].empty,
            'sufficient_data_points': dataset['metadata']['data_points'] >= 40,  # Adjusted for 4h of 5m data
            'multiple_spot_markets': dataset['metadata']['spot_markets_count'] > 1,
            'multiple_futures_markets': dataset['metadata']['futures_markets_count'] > 1
        }
        
        validations['overall_quality'] = all([
            validations['has_price_data'],
            validations['has_spot_cvd'],
            validations['has_futures_cvd'],
            validations['sufficient_data_points']
        ])
        
        return validations
    
    def _load_ohlcv_data_chunked(self, markets: List[str], start_time: datetime, 
                                end_time: datetime, timeframe: str, chunk_hours: int = 72) -> pd.DataFrame:
        """
        Load OHLCV data in chunks for large date ranges to avoid timeouts
        
        Args:
            markets: List of markets to query
            start_time: Start datetime
            end_time: End datetime  
            timeframe: Timeframe
            
        Returns:
            Combined DataFrame from all chunks
        """
        import logging
        logger = logging.getLogger(__name__)
        
        chunk_days = 3  # Process 3 days at a time
        all_dfs = []
        current_start = start_time
        
        logger.info(f"Loading OHLCV data in {chunk_days}-day chunks for {len(markets)} markets")
        
        while current_start < end_time:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_time)
            
            try:
                chunk_df = self.influx_client.get_ohlcv_data(
                    markets=markets,
                    start_time=current_start,
                    end_time=chunk_end,
                    timeframe=timeframe
                )
                
                if not chunk_df.empty:
                    all_dfs.append(chunk_df)
                    logger.debug(f"Loaded OHLCV chunk: {current_start.date()} to {chunk_end.date()} ({len(chunk_df)} rows)")
                
            except Exception as e:
                logger.error(f"Failed to load OHLCV chunk {current_start.date()} to {chunk_end.date()}: {e}")
            
            current_start = chunk_end
        
        # Combine all chunks
        if all_dfs:
            # Filter out empty DataFrames to avoid FutureWarning
            non_empty_dfs = [df for df in all_dfs if not df.empty]
            if non_empty_dfs:
                combined_df = pd.concat(non_empty_dfs, axis=0).sort_index()
            else:
                return pd.DataFrame()
            logger.info(f"Combined OHLCV data: {len(combined_df)} total rows")
            return combined_df
        
        return pd.DataFrame()
    
    def _load_volume_data_chunked(self, markets: List[str], start_time: datetime, 
                                 end_time: datetime, timeframe: str, market_type: str) -> pd.DataFrame:
        """
        Load volume data in chunks for large date ranges to avoid timeouts
        
        Args:
            markets: List of markets to query
            start_time: Start datetime
            end_time: End datetime
            timeframe: Timeframe
            market_type: 'spot' or 'futures' for logging
            
        Returns:
            Combined DataFrame from all chunks
        """
        import logging
        logger = logging.getLogger(__name__)
        
        chunk_days = 3  # Process 3 days at a time
        all_dfs = []
        current_start = start_time
        
        logger.info(f"Loading {market_type} volume data in {chunk_days}-day chunks for {len(markets)} markets")
        
        while current_start < end_time:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_time)
            
            try:
                chunk_df = self.influx_client.get_volume_data(
                    markets=markets,
                    start_time=current_start,
                    end_time=chunk_end,
                    timeframe=timeframe
                )
                
                if not chunk_df.empty:
                    all_dfs.append(chunk_df)
                    logger.debug(f"Loaded {market_type} volume chunk: {current_start.date()} to {chunk_end.date()} ({len(chunk_df)} rows)")
                
            except Exception as e:
                logger.error(f"Failed to load {market_type} volume chunk {current_start.date()} to {chunk_end.date()}: {e}")
            
            current_start = chunk_end
        
        # Combine all chunks
        if all_dfs:
            # Filter out empty DataFrames to avoid FutureWarning
            non_empty_dfs = [df for df in all_dfs if not df.empty]
            if non_empty_dfs:
                combined_df = pd.concat(non_empty_dfs, axis=0).sort_index()
            else:
                return pd.DataFrame()
            logger.info(f"Combined {market_type} volume data: {len(combined_df)} total rows")
            return combined_df
        
        return pd.DataFrame()
    
    def _load_ohlcv_data_chunked_1s(self, markets: List[str], start_time: datetime, 
                                    end_time: datetime, timeframe: str, chunk_hours: int = 2,
                                    max_retries: int = 3) -> pd.DataFrame:
        """
        Load OHLCV data in optimized chunks for 1-second timeframes with retry logic
        
        Args:
            markets: List of markets to query
            start_time: Start datetime
            end_time: End datetime  
            timeframe: Timeframe (should be '1s')
            chunk_hours: Hours per chunk (default 2 for 1s data)
            max_retries: Maximum retries per chunk
            
        Returns:
            Combined DataFrame from all chunks
        """
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        chunk_duration = timedelta(hours=chunk_hours)
        all_dfs = []
        current_start = start_time
        total_chunks = int((end_time - start_time).total_seconds() / chunk_duration.total_seconds()) + 1
        chunk_num = 0
        
        logger.info(f"Loading 1s OHLCV data in {chunk_hours}h chunks for {len(markets)} markets ({total_chunks} total chunks)")
        
        while current_start < end_time:
            chunk_num += 1
            chunk_end = min(current_start + chunk_duration, end_time)
            
            # Progress indicator
            progress = (chunk_num / total_chunks) * 100
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({progress:.1f}%): {current_start.strftime('%H:%M:%S')} to {chunk_end.strftime('%H:%M:%S')}")
            
            # Retry logic for each chunk
            chunk_df = None
            for retry in range(max_retries):
                try:
                    chunk_df = self.influx_client.get_ohlcv_data(
                        markets=markets,
                        start_time=current_start,
                        end_time=chunk_end,
                        timeframe=timeframe
                    )
                    
                    if not chunk_df.empty:
                        all_dfs.append(chunk_df)
                        logger.debug(f"✓ Loaded 1s OHLCV chunk {chunk_num}: {len(chunk_df)} rows")
                    else:
                        logger.debug(f"⚠ Empty 1s OHLCV chunk {chunk_num}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                        logger.warning(f"Retry {retry + 1}/{max_retries} for chunk {chunk_num} after {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to load 1s OHLCV chunk {chunk_num} after {max_retries} retries: {e}")
            
            current_start = chunk_end
        
        # Combine all chunks with memory optimization
        if all_dfs:
            non_empty_dfs = [df for df in all_dfs if not df.empty]
            if non_empty_dfs:
                logger.info(f"Combining {len(non_empty_dfs)} 1s OHLCV chunks...")
                combined_df = pd.concat(non_empty_dfs, axis=0).sort_index()
                
                # Remove duplicates that might occur at chunk boundaries
                if len(combined_df) > 0:
                    initial_len = len(combined_df)
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    if len(combined_df) < initial_len:
                        logger.debug(f"Removed {initial_len - len(combined_df)} duplicate rows at chunk boundaries")
                
                logger.info(f"✅ Combined 1s OHLCV data: {len(combined_df)} total rows")
                return combined_df
            else:
                logger.warning("All 1s OHLCV chunks were empty")
                return pd.DataFrame()
        
        logger.warning("No 1s OHLCV data loaded from any chunk")
        return pd.DataFrame()
    
    def _load_volume_data_chunked_1s(self, markets: List[str], start_time: datetime, 
                                     end_time: datetime, timeframe: str, market_type: str,
                                     chunk_hours: int = 2, max_retries: int = 3) -> pd.DataFrame:
        """
        Load volume data in optimized chunks for 1-second timeframes with retry logic
        
        Args:
            markets: List of markets to query
            start_time: Start datetime
            end_time: End datetime
            timeframe: Timeframe (should be '1s')
            market_type: 'spot' or 'futures' for logging
            chunk_hours: Hours per chunk (default 2 for 1s data)
            max_retries: Maximum retries per chunk
            
        Returns:
            Combined DataFrame from all chunks
        """
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        chunk_duration = timedelta(hours=chunk_hours)
        all_dfs = []
        current_start = start_time
        total_chunks = int((end_time - start_time).total_seconds() / chunk_duration.total_seconds()) + 1
        chunk_num = 0
        
        logger.info(f"Loading 1s {market_type} volume data in {chunk_hours}h chunks for {len(markets)} markets ({total_chunks} total chunks)")
        
        while current_start < end_time:
            chunk_num += 1
            chunk_end = min(current_start + chunk_duration, end_time)
            
            # Progress indicator
            progress = (chunk_num / total_chunks) * 100
            logger.info(f"Processing {market_type} chunk {chunk_num}/{total_chunks} ({progress:.1f}%): {current_start.strftime('%H:%M:%S')} to {chunk_end.strftime('%H:%M:%S')}")
            
            # Retry logic for each chunk
            chunk_df = None
            for retry in range(max_retries):
                try:
                    chunk_df = self.influx_client.get_volume_data(
                        markets=markets,
                        start_time=current_start,
                        end_time=chunk_end,
                        timeframe=timeframe
                    )
                    
                    if not chunk_df.empty:
                        all_dfs.append(chunk_df)
                        logger.debug(f"✓ Loaded 1s {market_type} volume chunk {chunk_num}: {len(chunk_df)} rows")
                    else:
                        logger.debug(f"⚠ Empty 1s {market_type} volume chunk {chunk_num}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                        logger.warning(f"Retry {retry + 1}/{max_retries} for {market_type} chunk {chunk_num} after {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to load 1s {market_type} volume chunk {chunk_num} after {max_retries} retries: {e}")
            
            current_start = chunk_end
        
        # Combine all chunks with memory optimization
        if all_dfs:
            non_empty_dfs = [df for df in all_dfs if not df.empty]
            if non_empty_dfs:
                logger.info(f"Combining {len(non_empty_dfs)} 1s {market_type} volume chunks...")
                combined_df = pd.concat(non_empty_dfs, axis=0).sort_index()
                
                # Remove duplicates that might occur at chunk boundaries
                if len(combined_df) > 0:
                    initial_len = len(combined_df)
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    if len(combined_df) < initial_len:
                        logger.debug(f"Removed {initial_len - len(combined_df)} duplicate rows at chunk boundaries")
                
                logger.info(f"✅ Combined 1s {market_type} volume data: {len(combined_df)} total rows")
                return combined_df
            else:
                logger.warning(f"All 1s {market_type} volume chunks were empty")
                return pd.DataFrame()
        
        logger.warning(f"No 1s {market_type} volume data loaded from any chunk")
        return pd.DataFrame()
    
    def _split_volume_by_market_type(self, combined_volume_df: pd.DataFrame, 
                                   markets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split combined volume data by market type when using 1s aggregated data
        
        Args:
            combined_volume_df: Combined volume data from all markets
            markets: Market classification dict
            
        Returns:
            Tuple of (spot_df, futures_df)
        """
        # For 1s aggregated data, we get totals across all markets
        # We need to estimate the split based on market counts
        spot_markets = markets.get('spot', [])
        futures_markets = markets.get('perp', [])
        
        total_markets = len(spot_markets) + len(futures_markets)
        
        if total_markets == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Estimate proportional split (this is approximate for CVD calculation)
        spot_ratio = len(spot_markets) / total_markets if total_markets > 0 else 0.5
        futures_ratio = len(futures_markets) / total_markets if total_markets > 0 else 0.5
        
        spot_df = combined_volume_df.copy()
        futures_df = combined_volume_df.copy()
        
        # Rename columns and apply ratios
        if not spot_df.empty:
            column_mapping = {
                'total_vbuy': 'total_vbuy_spot',
                'total_vsell': 'total_vsell_spot', 
                'total_cbuy': 'total_cbuy_spot',
                'total_csell': 'total_csell_spot',
                'total_lbuy': 'total_lbuy_spot',
                'total_lsell': 'total_lsell_spot'
            }
            spot_df = spot_df.rename(columns=column_mapping)
            
            # Apply spot ratio
            for col in spot_df.columns:
                if col.endswith('_spot'):
                    spot_df[col] *= spot_ratio
        
        if not futures_df.empty:
            column_mapping = {
                'total_vbuy': 'total_vbuy_futures',
                'total_vsell': 'total_vsell_futures',
                'total_cbuy': 'total_cbuy_futures', 
                'total_csell': 'total_csell_futures',
                'total_lbuy': 'total_lbuy_futures',
                'total_lsell': 'total_lsell_futures'
            }
            futures_df = futures_df.rename(columns=column_mapping)
            
            # Apply futures ratio
            for col in futures_df.columns:
                if col.endswith('_futures'):
                    futures_df[col] *= futures_ratio
        
        return spot_df, futures_df
    
    def _empty_dataset(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> Dict:
        """Return empty dataset structure"""
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': end_time,
            'ohlcv': pd.DataFrame(),
            'spot_volume': pd.DataFrame(),
            'futures_volume': pd.DataFrame(),
            'spot_cvd': pd.Series(dtype=float),
            'futures_cvd': pd.Series(dtype=float),
            'cvd_divergence': pd.Series(dtype=float),
            'markets': {'spot': [], 'perp': []},
            'data_source': 'empty',
            'metadata': {
                'spot_markets_count': 0,
                'futures_markets_count': 0,
                'data_points': 0
            }
        }
    
    def get_complete_dataset_with_1s_support(self, symbol: str, start_time: datetime,
                                            end_time: datetime, timeframe: str = '5m') -> Dict:
        """
        Synchronous wrapper for 1s data support with fallback
        """
        import asyncio
        
        # Limit lookback for real-time processing  
        max_lookback_minutes = 30
        if (end_time - start_time).total_seconds() > max_lookback_minutes * 60:
            # Use regular method for large time ranges
            return self.get_complete_dataset(symbol, start_time, end_time, timeframe)
        
        try:
            # Try to use async method with 1s data
            return asyncio.run(self.get_complete_dataset_async(
                symbol, start_time, end_time, timeframe, 
                prefer_1s_data=True, max_lookback_minutes=max_lookback_minutes
            ))
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Async 1s data loading failed for {symbol}, using regular data: {e}")
            # Fallback to regular synchronous method
            return self.get_complete_dataset(symbol, start_time, end_time, timeframe)

    def clear_cache(self):
        """Clear discovery caches"""
        self._symbol_cache.clear()
        self._market_cache.clear()


# Factory function for easy import
def create_data_pipeline():
    """Factory function to create data pipeline instance"""
    return DataPipeline()