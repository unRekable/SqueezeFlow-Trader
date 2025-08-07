#!/usr/bin/env python3
"""
Market Discovery - Robuste Market-Erkennung aus InfluxDB
Environment-aware market discovery compatible with Docker containers and host systems
"""

import logging
import os
from typing import Dict, List, Tuple, Optional
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError

logger = logging.getLogger(__name__)


class MarketDiscovery:
    """
    Robuste Market-Discovery die tatsächlich verfügbare Markets aus InfluxDB holt
    Compatible with both Docker containers and host systems
    """
    
    def __init__(self, influx_host: str = None):
        self.custom_influx_host = influx_host
        self.setup_database()
        
    def setup_database(self):
        """Setup InfluxDB connection with environment detection"""
        # Use custom host if provided, otherwise detect
        influx_host = self.custom_influx_host or self._get_influx_host()
        
        # Unified port configuration - always use 8086
        # Both inside Docker and on host use port 8086 after migration
        influx_port = 8086
        
        try:
            self.influx_client = InfluxDBClient(
                host=influx_host,
                port=influx_port,
                username='',  # No auth for Docker setup
                password='',
                database='significant_trades',
                timeout=120,  # Extended timeout for DNS issues
                retries=5,
                ssl=False,
                verify_ssl=False
            )
            
            # Test connection
            self.influx_client.ping()
            logger.info(f"✅ Connected to InfluxDB at {influx_host}:{influx_port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to InfluxDB at {influx_host}: {e}")
            raise RuntimeError(f"InfluxDB connection failed: {e}")
    
    def _get_influx_host(self) -> str:
        """Determine InfluxDB host based on environment"""
        # Check environment variables first
        if os.getenv('INFLUX_HOST'):
            return os.getenv('INFLUX_HOST')
        
        # Check if running inside Docker container
        if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
            logger.info("🐳 Running inside Docker container, using service name")
            return 'aggr-influx'  # Docker service name
        
        # Default to localhost for host system
        logger.info("🏠 Running on host system, using localhost")
        return 'localhost'
    
    def _execute_influx_query(self, query: str) -> Optional[List[Dict]]:
        """Execute InfluxDB query using direct client connection"""
        try:
            result = self.influx_client.query(query)
            
            if result:
                # Convert result to list of dictionaries
                query_results = []
                for measurement in result:
                    for point in measurement:
                        query_results.append(dict(point))
                return query_results
            
            return []
            
        except (InfluxDBClientError, InfluxDBServerError) as e:
            logger.error(f"InfluxDB query error: {e}")
            return None
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None
    
    def _get_series_list(self) -> List[str]:
        """Get list of all series from InfluxDB"""
        try:
            query = 'SHOW SERIES FROM "aggr_1m"."trades_1m"'
            result = self.influx_client.query(query)
            
            series_list = []
            if result:
                for series in result:
                    for point in series:
                        if 'key' in point:
                            series_list.append(point['key'])
                        elif len(point) > 0:  # Handle different result formats
                            series_list.append(str(point[0]) if isinstance(point, (list, tuple)) else str(point))
            
            return series_list
            
        except Exception as e:
            logger.error(f"Failed to get series list: {e}")
            return []
        
    def get_available_markets_for_symbol(self, symbol: str) -> List[str]:
        """
        Get all available markets for a trading symbol from InfluxDB
        
        Args:
            symbol: Trading symbol (BTC, ETH, SOL, BTCUSDT, etc.)
            
        Returns:
            List of available market identifiers
        """
        logger.info(f"🔍 Discovering available {symbol} markets from database...")
        
        try:
            # Get all series from InfluxDB
            series_list = self._get_series_list()
            
            available_markets = []
            for series_key in series_list:
                if 'market=' in series_key:
                    # Extract market from series key (format: trades_1m,market=BINANCE:btcusdt)
                    market = series_key.split('market=')[1].split(',')[0]
                    
                    # Smart symbol matching - handle different formats
                    symbol_variants = [
                        symbol.upper(),
                        symbol.lower(),
                        symbol.replace('USDT', '').upper(),  # BTCUSDT -> BTC
                        symbol.replace('USDC', '').upper(),  # BTCUSDC -> BTC  
                        symbol.replace('USD', '').upper(),   # BTCUSD -> BTC
                    ]
                    
                    # Check if any variant matches the market
                    market_upper = market.upper()
                    if any(variant in market_upper for variant in symbol_variants if variant):
                        available_markets.append(market)
            
            # Remove duplicates while preserving order
            available_markets = list(dict.fromkeys(available_markets))
            
            logger.debug(f"📊 Found {len(available_markets)} {symbol} markets in database")
            if available_markets:
                logger.debug(f"📈 Sample markets: {available_markets[:5]}")
                
            return available_markets
            
        except Exception as e:
            logger.error(f"Error discovering markets for {symbol}: {e}")
            return []
            
    def classify_markets(self, markets: List[str]) -> Tuple[List[str], List[str]]:
        """
        Classify markets into SPOT and PERP using exchange mapper
        
        Args:
            markets: List of market identifiers
            
        Returns:
            Tuple of (spot_markets, perp_markets)
        """
        from ..processors.exchange_mapper import exchange_mapper
        
        logger.debug(f"🔍 Classifying {len(markets)} markets...")
        
        spot_markets = []
        perp_markets = []
        
        for market in markets:
            try:
                market_type = exchange_mapper.get_market_type(market)
                if market_type == 'SPOT':
                    spot_markets.append(market)
                elif market_type == 'PERP':
                    perp_markets.append(market)
            except Exception as e:
                logger.warning(f"Could not classify market {market}: {e}")
        
        logger.debug(f"✅ Classified: {len(spot_markets)} SPOT, {len(perp_markets)} PERP markets")
        return spot_markets, perp_markets
        
    def get_markets_by_type(self, symbol: str) -> Dict[str, List[str]]:
        """
        Get available markets for symbol, classified by type
        
        Args:
            symbol: Trading symbol (BTC, ETH, SOL, BTCUSDT, etc.)
            
        Returns:
            Dict with 'spot' and 'perp' market lists
        """
        # Discover available markets
        available_markets = self.get_available_markets_for_symbol(symbol)
        
        if not available_markets:
            logger.warning(f"No markets found for {symbol}")
            return {'spot': [], 'perp': []}
        
        # Classify markets
        spot_markets, perp_markets = self.classify_markets(available_markets)
        
        return {
            'spot': spot_markets,
            'perp': perp_markets
        }


# Global instance for easy import
market_discovery = MarketDiscovery()