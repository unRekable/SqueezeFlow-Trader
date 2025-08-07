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
        
        # Create client with extended timeout
        from .loaders.influx_client import QueryOptimization
        optimization_config = QueryOptimization(
            query_timeout_seconds=120,  # Extended timeout for connection issues
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
        Load raw OHLCV data for a symbol
        
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
        
        # Load data from InfluxDB
        return self.influx_client.get_ohlcv_data(
            markets=all_markets,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe
        )
    
    def load_raw_volume_data(self, symbol: str, start_time: datetime, 
                            end_time: datetime, timeframe: str = '5m') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw volume data separated by market type
        
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
        
        # Load spot volume data
        spot_df = pd.DataFrame()
        if spot_markets:
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
        
        # Load futures volume data
        futures_df = pd.DataFrame()
        if perp_markets:
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
    
    def get_complete_dataset(self, symbol: str, start_time: datetime, 
                            end_time: datetime, timeframe: str = '5m') -> Dict:
        """
        Get complete dataset for strategy analysis
        
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
            'metadata': {
                'spot_markets_count': len(markets.get('spot', [])),
                'futures_markets_count': len(markets.get('perp', [])),
                'data_points': len(ohlcv_df)
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
    
    def clear_cache(self):
        """Clear discovery caches"""
        self._symbol_cache.clear()
        self._market_cache.clear()


# Factory function for easy import
def create_data_pipeline():
    """Factory function to create data pipeline instance"""
    return DataPipeline()