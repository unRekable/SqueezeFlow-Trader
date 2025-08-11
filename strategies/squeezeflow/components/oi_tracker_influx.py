"""
Open Interest Tracker - InfluxDB Integration
Reads OI data from remote InfluxDB server instead of fetching from exchanges
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
import os
from influxdb import InfluxDBClient

logger = logging.getLogger(__name__)


class OITrackerInflux:
    """
    OI tracking using data from remote InfluxDB server
    Reads pre-collected OI data just like CVD data from remote server
    """
    
    def __init__(self, rise_threshold: float = 5.0, influx_client: Optional[InfluxDBClient] = None):
        """
        Initialize OI tracker with InfluxDB connection
        
        Args:
            rise_threshold: Minimum OI increase % for squeeze signal (default 5%)
            influx_client: Optional pre-configured InfluxDB client
        """
        self.rise_threshold = rise_threshold
        
        # Setup InfluxDB connection (use remote server like CVD data)
        if influx_client:
            self.client = influx_client
        else:
            # Use environment variables for remote InfluxDB server
            self.client = InfluxDBClient(
                host=os.getenv('INFLUX_HOST', 'localhost'),
                port=int(os.getenv('INFLUX_PORT', 8086)),
                username=os.getenv('INFLUX_USER', ''),
                password=os.getenv('INFLUX_PASSWORD', ''),
                database=os.getenv('INFLUX_DATABASE', 'significant_trades')
            )
        
        logger.info(f"OI Tracker (InfluxDB) initialized with {rise_threshold}% threshold")
        logger.info(f"Connected to InfluxDB at {self.client._host}:{self.client._port}")
    
    def get_oi_from_influx(self, symbol: str, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Fetch OI data from InfluxDB
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            lookback_hours: Hours of historical data to fetch
            
        Returns:
            DataFrame with OI data
        """
        try:
            # Query for aggregated OI data (using TOTAL_AGG for all exchanges)
            query = f"""
            SELECT 
                mean(open_interest) as oi_usd,
                mean(open_interest) as oi_coin
            FROM open_interest 
            WHERE symbol = '{symbol}' 
                AND exchange = 'TOTAL_AGG'
                AND time > now() - {lookback_hours}h
            GROUP BY time(5m)
            ORDER BY time DESC
            """
            
            result = self.client.query(query)
            
            if result:
                df = pd.DataFrame(list(result.get_points()))
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching OI from InfluxDB: {e}")
            return pd.DataFrame()
    
    def get_oi_by_exchange(self, symbol: str, lookback_hours: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Fetch OI data by individual exchanges
        
        Returns:
            Dict mapping exchange names to DataFrames
        """
        try:
            exchanges_data = {}
            
            # Query for each major exchange
            for exchange in ['BINANCE_FUTURES', 'BYBIT', 'OKX', 'DERIBIT']:
                query = f"""
                SELECT 
                    mean(open_interest_usd) as oi_usd,
                    mean(open_interest_coin) as oi_coin
                FROM open_interest 
                WHERE symbol = '{symbol}' 
                    AND exchange = '{exchange}'
                    AND time > now() - {lookback_hours}h
                GROUP BY time(5m)
                ORDER BY time DESC
                """
                
                result = self.client.query(query)
                
                if result:
                    df = pd.DataFrame(list(result.get_points()))
                    if not df.empty:
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                        exchanges_data[exchange] = df
            
            # Also get futures aggregate (top 3 exchanges)
            query = f"""
            SELECT 
                mean(open_interest) as oi_usd,
                mean(open_interest) as oi_coin
            FROM open_interest 
            WHERE symbol = '{symbol}' 
                AND exchange = 'FUTURES_AGG'
                AND time > now() - {lookback_hours}h
            GROUP BY time(5m)
            ORDER BY time DESC
            """
            
            result = self.client.query(query)
            if result:
                df = pd.DataFrame(list(result.get_points()))
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    exchanges_data['FUTURES_AGG'] = df
            
            return exchanges_data
            
        except Exception as e:
            logger.error(f"Error fetching OI by exchange from InfluxDB: {e}")
            return {}
    
    def calculate_oi_change(self, df: pd.DataFrame, periods: int = 24) -> Dict:
        """
        Calculate OI change over different periods
        
        Args:
            df: DataFrame with OI data
            periods: Number of 5m periods to look back (24 = 2 hours)
            
        Returns:
            Dict with OI metrics
        """
        if df.empty or len(df) < 2:
            return {
                'current_oi': 0,
                'oi_change_pct': 0,
                'oi_change_usd': 0,
                'is_rising': False,
                'exceeds_threshold': False
            }
        
        try:
            current_oi = df['oi_usd'].iloc[0] if not pd.isna(df['oi_usd'].iloc[0]) else 0
            
            # Get OI from specified periods ago
            if len(df) > periods:
                past_oi = df['oi_usd'].iloc[periods]
            else:
                past_oi = df['oi_usd'].iloc[-1]
            
            if past_oi > 0:
                oi_change_pct = ((current_oi - past_oi) / past_oi) * 100
                oi_change_usd = current_oi - past_oi
            else:
                oi_change_pct = 0
                oi_change_usd = 0
            
            return {
                'current_oi': current_oi,
                'oi_change_pct': oi_change_pct,
                'oi_change_usd': oi_change_usd,
                'is_rising': oi_change_pct > 0,
                'exceeds_threshold': oi_change_pct >= self.rise_threshold
            }
            
        except Exception as e:
            logger.error(f"Error calculating OI change: {e}")
            return {
                'current_oi': 0,
                'oi_change_pct': 0,
                'oi_change_usd': 0,
                'is_rising': False,
                'exceeds_threshold': False
            }
    
    def get_oi_change_sync(self, symbol: str) -> Dict:
        """
        Synchronous wrapper for getting OI change from InfluxDB
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dict with OI change metrics
        """
        try:
            # Get OI data from InfluxDB
            df = self.get_oi_from_influx(symbol, lookback_hours=4)
            
            if df.empty:
                logger.warning(f"No OI data found for {symbol}")
                return {
                    'current_oi': 0,
                    'oi_change_pct': 0,
                    'oi_change_usd': 0,
                    'is_rising': False,
                    'exceeds_threshold': False,
                    'data_source': 'influxdb',
                    'data_available': False
                }
            
            # Calculate OI change (24 periods = 2 hours at 5m intervals)
            oi_metrics = self.calculate_oi_change(df, periods=24)
            oi_metrics['data_source'] = 'influxdb'
            oi_metrics['data_available'] = True
            
            logger.info(f"OI for {symbol}: Current=${oi_metrics['current_oi']:,.0f}, "
                       f"Change={oi_metrics['oi_change_pct']:.2f}%, "
                       f"Rising={oi_metrics['is_rising']}")
            
            return oi_metrics
            
        except Exception as e:
            logger.error(f"Error getting OI change for {symbol}: {e}")
            return {
                'current_oi': 0,
                'oi_change_pct': 0,
                'oi_change_usd': 0,
                'is_rising': False,
                'exceeds_threshold': False,
                'data_source': 'influxdb',
                'data_available': False,
                'error': str(e)
            }
    
    def validate_squeeze_with_oi(self, symbol: str, squeeze_detected: bool) -> Tuple[bool, str]:
        """
        Validate squeeze signal with OI confirmation from InfluxDB
        
        Args:
            symbol: Trading symbol
            squeeze_detected: Whether technical squeeze was detected
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not squeeze_detected:
            return False, "No squeeze detected"
        
        # Get OI data from InfluxDB
        oi_data = self.get_oi_change_sync(symbol)
        
        if not oi_data.get('data_available'):
            logger.warning(f"No OI data available for {symbol}, allowing squeeze")
            return True, "Squeeze confirmed (OI data unavailable)"
        
        # Squeeze is only valid if OI is rising
        if not oi_data['is_rising']:
            return False, f"OI not rising ({oi_data['oi_change_pct']:.2f}%)"
        
        # Strong squeeze if OI exceeds threshold
        if oi_data['exceeds_threshold']:
            return True, f"Strong squeeze! OI +{oi_data['oi_change_pct']:.2f}%"
        
        return True, f"Squeeze confirmed, OI +{oi_data['oi_change_pct']:.2f}%"
    
    def get_detailed_oi_analysis(self, symbol: str) -> Dict:
        """
        Get detailed OI analysis including per-exchange breakdown
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with comprehensive OI analysis
        """
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_oi': {},
            'by_exchange': {},
            'trend': {}
        }
        
        # Get total OI
        total_df = self.get_oi_from_influx(symbol, lookback_hours=24)
        if not total_df.empty:
            analysis['total_oi'] = self.calculate_oi_change(total_df, periods=24)
            
            # Calculate trend
            if len(total_df) > 48:  # 4 hours of 5m data
                short_term = self.calculate_oi_change(total_df, periods=12)  # 1 hour
                medium_term = self.calculate_oi_change(total_df, periods=48)  # 4 hours
                
                analysis['trend'] = {
                    '1h_change': short_term['oi_change_pct'],
                    '4h_change': medium_term['oi_change_pct'],
                    'trend_direction': 'bullish' if short_term['oi_change_pct'] > medium_term['oi_change_pct'] else 'bearish'
                }
        
        # Get per-exchange breakdown
        exchange_data = self.get_oi_by_exchange(symbol, lookback_hours=24)
        for exchange, df in exchange_data.items():
            if not df.empty:
                analysis['by_exchange'][exchange] = self.calculate_oi_change(df, periods=24)
        
        return analysis


# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Initialize tracker (will use remote InfluxDB based on env vars)
    tracker = OITrackerInflux(rise_threshold=5.0)
    
    # Test with BTC
    symbol = "BTC"
    
    # Test 1: Get OI change from InfluxDB
    print(f"\n{'='*50}")
    print(f"Testing OI Tracker with InfluxDB for {symbol}")
    print(f"{'='*50}")
    
    result = tracker.get_oi_change_sync(symbol)
    print(f"\nOI Data from InfluxDB:")
    print(json.dumps(result, indent=2))
    
    # Test 2: Validate squeeze
    print(f"\n{'='*50}")
    print("Testing Squeeze Validation with OI")
    print(f"{'='*50}")
    
    is_valid, reason = tracker.validate_squeeze_with_oi(symbol, squeeze_detected=True)
    print(f"Squeeze Valid: {is_valid}")
    print(f"Reason: {reason}")
    
    # Test 3: Get detailed analysis
    print(f"\n{'='*50}")
    print("Detailed OI Analysis")
    print(f"{'='*50}")
    
    analysis = tracker.get_detailed_oi_analysis(symbol)
    print(json.dumps(analysis, indent=2, default=str))