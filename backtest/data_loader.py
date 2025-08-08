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

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    """
    Data loader for backtesting with 1-second data aggregation support.
    
    This class loads raw 1-second data from InfluxDB and dynamically
    aggregates it to any required timeframe for backtesting.
    """
    
    def __init__(self, host='localhost', port=8086, database='significant_trades'):
        """
        Initialize the backtest data loader.
        
        Args:
            host: InfluxDB host
            port: InfluxDB port
            database: InfluxDB database name
        """
        self.influx = InfluxDBClient(
            host=host,
            port=port,
            database=database
        )
        logger.info(f"BacktestDataLoader initialized with {host}:{port}/{database}")
    
    def load_1s_data(self, symbol, start_time, end_time):
        """
        Load raw 1-second data from InfluxDB.
        
        Args:
            symbol: Trading symbol (e.g., 'BINANCE:btcusdt')
            start_time: Start time (ISO format string or datetime)
            end_time: End time (ISO format string or datetime)
            
        Returns:
            DataFrame with 1-second OHLCV data
        """
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


# Example usage function
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


if __name__ == "__main__":
    # Run example if executed directly
    logging.basicConfig(level=logging.INFO)
    example_backtest_with_1s_data()