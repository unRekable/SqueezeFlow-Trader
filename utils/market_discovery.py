#!/usr/bin/env python3
"""
Market Discovery - Robuste Market-Erkennung aus InfluxDB
Basiert auf dem bewährten CVD-Tool Ansatz für Production-Ready Market Discovery
"""

import logging
import os
from typing import Dict, List, Tuple
from influxdb import InfluxDBClient

logger = logging.getLogger(__name__)


class MarketDiscovery:
    """
    Robuste Market-Discovery die tatsächlich verfügbare Markets aus InfluxDB holt
    Ersetzt fragile Market-Generierung mit echter Datenbank-Abfrage
    """
    
    def __init__(self):
        self.setup_database()
        
    def setup_database(self):
        """Setup InfluxDB connection"""
        self.influx_client = InfluxDBClient(
            host=os.getenv('INFLUX_HOST', 'localhost'),
            port=int(os.getenv('INFLUX_PORT', 8086)),
            database='significant_trades'
        )
        
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
            # Query to find all markets with this symbol
            series_query = 'SHOW SERIES FROM "aggr_1m".trades_1m'
            result = self.influx_client.query(series_query)
            
            available_markets = []
            series_data = result.raw.get('series', [])
            
            for series in series_data:
                if 'values' in series:
                    for value in series['values']:
                        series_key = value[0]
                        if 'market=' in series_key:
                            market = series_key.split('market=')[1]
                            
                            # Smart symbol matching - handle different formats
                            symbol_variants = [
                                symbol.upper(),
                                symbol.lower(),
                                symbol.replace('USDT', '').upper(),  # BTCUSDT -> BTC
                                symbol.replace('USDC', '').upper(),  # BTCUSDC -> BTC  
                                symbol.replace('USD', '').upper(),   # BTCUSD -> BTC
                            ]
                            
                            # Check if any variant matches the market
                            if any(variant in market.upper() for variant in symbol_variants if variant):
                                available_markets.append(market)
            
            # Remove duplicates while preserving order
            available_markets = list(dict.fromkeys(available_markets))
            
            logger.info(f"📊 Found {len(available_markets)} {symbol} markets in database")
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
        from utils.exchange_mapper import exchange_mapper
        
        logger.info(f"🔍 Classifying {len(markets)} markets...")
        
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
        
        logger.info(f"✅ Classified: {len(spot_markets)} SPOT, {len(perp_markets)} PERP markets")
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