#!/usr/bin/env python3
"""
Symbol Discovery - Präzise Erkennung verfügbarer Trading-Symbols aus InfluxDB
Ersetzt hardcoded Symbol-Listen mit datengetriebener Discovery
"""

import logging
import os
import re
from typing import Dict, List, Set, Tuple, Optional
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class SymbolDiscovery:
    """
    Robuste Symbol-Discovery die tatsächlich verfügbare und qualitativ hochwertige
    Trading-Symbols aus InfluxDB erkennt
    """
    
    def __init__(self, influx_host: str = None):
        self.influx_host = influx_host or os.getenv('INFLUX_HOST', 'localhost')
        self.setup_database()
        
    def setup_database(self):
        """Setup InfluxDB connection with extended timeout"""
        self.influx_client = InfluxDBClient(
            host=self.influx_host,
            port=int(os.getenv('INFLUX_PORT', 8086)),
            database='significant_trades',
            timeout=120,  # Extended timeout for DNS issues
            retries=5,
            ssl=False,
            verify_ssl=False
        )
    
    def _execute_influx_query(self, query: str) -> Optional[List]:
        """Execute InfluxDB query using direct client connection"""
        try:
            result = self.influx_client.query(query)
            points = list(result.get_points())
            return points if points else None
        except Exception as e:
            logger.error(f"Query error: {e}")
            return None
        
    def extract_symbol_from_market(self, market: str) -> str:
        """
        Extract base symbol from market identifier using dynamic pattern matching
        
        Examples:
            BINANCE:btcusdt -> BTC
            BINANCE_FUTURES:ethusdt -> ETH  
            COINBASE:BTC-USD -> BTC
            BITFINEX:BTCUSD -> BTC
            OKEX:ARB-USDT-SWAP -> ARB
            KUCOIN:ARBUSDTM -> ARB
        """
        # Remove exchange prefix - handle special formats
        if ':' in market:
            parts = market.split(':', 1)  # Split only on first colon
            if len(parts) > 1:
                market_part = parts[1]
                market_clean = market_part.upper()
            else:
                market_part = market
                market_clean = market.upper()
        else:
            market_part = market
            market_clean = market.upper()
        
        # Define known quote currencies to strip
        quote_currencies = ['USDT', 'USDC', 'FDUSD', 'USD', 'BUSD', 'DAI', 'EUR', 'BTC', 'ETH', 'BNB', 'UST', 'USTF0']
        
        # Handle special formats first
        base_symbol = self._extract_from_special_formats(market_clean, market_part)
        if base_symbol:
            return base_symbol
            
        # Handle dash format first: SYMBOL-QUOTE or SYMBOL-QUOTE-SWAP
        if '-' in market_clean:
            parts = market_clean.split('-')
            if len(parts) >= 2:
                first_part = parts[0]
                if first_part and len(first_part) >= 2:
                    # Check if it's a known quote (avoid false positives)
                    remaining = '-'.join(parts[1:])
                    for quote in sorted(quote_currencies, key=len, reverse=True):
                        if remaining.startswith(quote) or remaining == quote:
                            return first_part
                    # If no quote match, still return first part if it looks like a symbol
                    if first_part.isalpha() and 2 <= len(first_part) <= 6:
                        return first_part
        
        # Handle underscore format: SYMBOL_QUOTE
        if '_' in market_clean:
            parts = market_clean.split('_')
            if len(parts) >= 2:
                first_part = parts[0]
                if first_part and len(first_part) >= 2:
                    # Check if remaining looks like quote currency
                    remaining = '_'.join(parts[1:])
                    for quote in sorted(quote_currencies, key=len, reverse=True):
                        if remaining.startswith(quote) or remaining == quote:
                            return first_part
                    # If no quote match, still return first part if it looks like a symbol
                    if first_part.isalpha() and 2 <= len(first_part) <= 6:
                        return first_part
                        
        # Standard extraction: remove quote currencies from end
        for quote in sorted(quote_currencies, key=len, reverse=True):  # Try longest first
            if market_clean.endswith(quote):
                base = market_clean[:-len(quote)]
                if base and len(base) >= 2:  # Valid base symbol
                    return base
        
        # If no quote currency found, try common patterns
        return self._extract_from_common_patterns(market_clean)
    
    def _extract_from_special_formats(self, market_clean: str, market_original: str = None) -> Optional[str]:
        """Extract from special exchange formats"""
        
        # Kraken: XBT -> BTC conversion
        if market_clean.startswith('XBT'):
            return 'BTC'
        
        # Phemex: starts with lowercase 's' for spot (e.g., sBTCUSDT) - check original case
        if (market_original and market_original.startswith('s') and len(market_original) > 1 and 
            any(c.isupper() for c in market_original[1:])):  # Must have uppercase after 's'
            return self.extract_symbol_from_market(f":{market_original[1:]}")
        
        # KuCoin: ends with 'M' for perpetuals (ARBUSDTM -> ARB)
        if market_clean.endswith('M') and len(market_clean) > 1:
            base_without_m = market_clean[:-1]
            return self.extract_symbol_from_market(f":{base_without_m}")
        
        # SWAP format: SYMBOL-QUOTE-SWAP
        if market_clean.endswith('-SWAP'):
            swap_removed = market_clean[:-5]  # Remove '-SWAP'
            return self.extract_symbol_from_market(f":{swap_removed}")
        
        # Bitfinex futures: SYMBOLUSDT -> SYMBOL
        if 'F0:USTF0' in market_clean:
            base = market_clean.split('F0:USTF0')[0]  # Take part before F0:USTF0
            return base if base else None
            
        return None
    
    def _extract_from_common_patterns(self, market_clean: str) -> Optional[str]:
        """Extract from common patterns when no quote currency match"""
        
        # Handle specific edge cases first
        # GATEIO format: SYMBOL_QUOTE-SPOT  
        if market_clean.endswith('-SPOT'):
            base_without_spot = market_clean[:-5]  # Remove '-SPOT'
            return self.extract_symbol_from_market(f":{base_without_spot}")
        
        # If it's 2-10 characters and looks like a symbol, return it
        if 2 <= len(market_clean) <= 10 and market_clean.isalpha():
            return market_clean
        
        # Try to find the symbol part (before numbers/special chars)
        symbol_match = re.match(r'^([A-Z]{2,10})', market_clean)  # Allow up to 10 chars for longer symbols
        if symbol_match:
            return symbol_match.group(1)
            
        return None
    
    def _normalize_symbols(self, symbol_markets: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Normalize and deduplicate symbols to their canonical base form
        
        Examples:
            BTC, BTCUSDT → BTC
            ETH, ETHUSDT, ETH/, CETH → ETH
        """
        normalized = {}
        
        # Define canonical symbol mappings
        canonical_symbols = {
            # Bitcoin variants
            'BTC': 'BTC', 'BTCUSDT': 'BTC', 'BTCUSDC': 'BTC', 'BTCUSD': 'BTC',
            'XBT': 'BTC', 'XBTUSDT': 'BTC', 'XBTUSD': 'BTC',
            
            # Ethereum variants  
            'ETH': 'ETH', 'ETHUSDT': 'ETH', 'ETHUSDC': 'ETH', 'ETHUSD': 'ETH',
            'ETH/': 'ETH', 'CETH': 'ETH',
            
            # Other common variants
            'TONCOIN': 'TON',  # HITBTC uses TONCOIN
        }
        
        for symbol, markets_list in symbol_markets.items():
            # Get canonical form or use original if not in mapping
            canonical = canonical_symbols.get(symbol, symbol)
            
            # Consolidate markets under canonical symbol
            if canonical not in normalized:
                normalized[canonical] = []
            normalized[canonical].extend(markets_list)
        
        # Remove duplicates from market lists
        for symbol in normalized:
            normalized[symbol] = list(set(normalized[symbol]))
            
        return normalized
        
    def _validate_symbol_match(self, market_clean: str, symbol: str) -> bool:
        """Validate that the symbol match is correct and not a false positive"""
        
        # Check against common quote currencies to ensure it's a proper pair
        quote_currencies = ['USDT', 'USDC', 'USD', 'EUR', 'BTC', 'ETH', 'BUSD', 'FDUSD']
        
        for quote in quote_currencies:
            # Check if market follows pattern: SYMBOL + QUOTE
            expected_pattern = f"{symbol}{quote}"
            if market_clean == expected_pattern:
                return True
                
            # Check dash-separated format: SYMBOL-QUOTE  
            expected_dash = f"{symbol}-{quote}"
            if market_clean == expected_dash:
                return True
                
        # Additional patterns for derivatives
        derivative_patterns = [
            f"{symbol}F0:USTF0",  # Bitfinex futures
            f"{symbol}-PERPETUAL", # Deribit
            f"{symbol}_USDT",      # Underscore format
            f"{symbol}_USD"
        ]
        
        for pattern in derivative_patterns:
            if market_clean == pattern:
                return True
                
        return False
        
    def discover_symbols_from_database(self, min_data_points: int = 1000, 
                                     hours_lookback: int = 24) -> List[str]:
        """
        Discover active symbols by analyzing actual market data in InfluxDB
        
        Args:
            min_data_points: Minimum data points required in lookback period
            hours_lookback: Hours to look back for data quality check
            
        Returns:
            List of symbols with sufficient data quality
        """
        logger.info(f"🔍 Discovering symbols with >{min_data_points} data points in last {hours_lookback}h...")
        
        try:
            # Get all available markets from trades_1s measurement
            series_query = 'SHOW SERIES FROM "aggr_1s"."trades_1s"'
            points = self._execute_influx_query(series_query)
            if points:
                data = {"results": [{"series": [{"values": [[point['key']] for point in points]}]}]}
            else:
                data = None
            
            # Extract markets from series
            markets = []
            if data and 'results' in data and data['results']:
                result = data['results'][0]
                if 'series' in result and result['series']:
                    series = result['series'][0]
                    if 'values' in series:
                        for value in series['values']:
                            series_key = value[0]
                            if 'market=' in series_key:
                                market = series_key.split('market=')[1]
                                markets.append(market)
            
            logger.info(f"📊 Found {len(markets)} total markets in database")
            
            # Extract symbols from markets  
            symbol_markets = {}
            for market in markets:
                symbol = self.extract_symbol_from_market(market)
                if symbol:
                    if symbol not in symbol_markets:
                        symbol_markets[symbol] = []
                    symbol_markets[symbol].append(market)
            
            logger.info(f"🎯 Extracted {len(symbol_markets)} potential symbols: {list(symbol_markets.keys())}")
            
            # Normalize and deduplicate symbols
            normalized_symbol_markets = self._normalize_symbols(symbol_markets)
            logger.info(f"🧹 Normalized to {len(normalized_symbol_markets)} unique symbols: {list(normalized_symbol_markets.keys())}")
            
            # Validate data quality for each normalized symbol
            active_symbols = []
            for symbol, markets_list in normalized_symbol_markets.items():
                data_quality = self._check_data_quality(markets_list, hours_lookback)
                
                if data_quality >= min_data_points:
                    active_symbols.append(symbol)
                    logger.info(f"✅ {symbol}: {data_quality} data points ({len(markets_list)} markets)")
                else:
                    logger.info(f"❌ {symbol}: {data_quality} data points (insufficient)")
            
            logger.info(f"🚀 Active symbols with sufficient data: {active_symbols}")
            return sorted(active_symbols)
            
        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []
            
    def _check_data_quality(self, markets: List[str], hours_lookback: int) -> int:
        """Check data quality for a list of markets"""
        try:
            # Build query for all markets of this symbol
            market_filter = ' OR '.join([f"market = '{market}'" for market in markets])
            
            query = f"""
            SELECT COUNT(close) as data_points
            FROM "aggr_1s"."trades_1s"
            WHERE ({market_filter})
            AND time > now() - {hours_lookback}h
            """
            
            # Use Docker exec method instead of standard client
            points = self._execute_influx_query(query)
            if points and len(points) > 0:
                # Get the data_points value from the first point
                data_points = points[0].get('data_points', 0)
                return int(data_points) if data_points else 0
            return 0
            
        except Exception as e:
            logger.warning(f"Error checking data quality: {e}")
            return 0
            
    def get_symbol_market_breakdown(self, symbol: str) -> Dict[str, List[str]]:
        """Get detailed market breakdown for a specific symbol"""
        from data.loaders.market_discovery import MarketDiscovery
        market_discovery = MarketDiscovery(self.influx_client)
        return market_discovery.get_markets_by_type(symbol)
        
    def validate_symbol_availability(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate if a list of symbols has sufficient data"""
        results = {}
        for symbol in symbols:
            markets = self.get_symbol_market_breakdown(symbol)
            total_markets = len(markets['spot']) + len(markets['perp'])
            results[symbol] = total_markets > 0
        return results
        
    def discover_oi_symbols_for_base(self, base_symbol: str, 
                                   min_data_points: int = 100, 
                                   hours_lookback: int = 24) -> List[str]:
        """
        Discover available Open Interest symbols for a base symbol
        
        Args:
            base_symbol: Base trading symbol (e.g., 'BTC', 'ETH')
            min_data_points: Minimum OI data points required
            hours_lookback: Hours to look back for data quality check
            
        Returns:
            List of OI symbols with sufficient data
        """
        logger.info(f"🔍 Discovering OI symbols for {base_symbol}...")
        
        try:
            # Query OI database for available symbols
            series_query = 'SHOW SERIES FROM "open_interest"'
            result = self.influx_client.query(series_query)
            
            # Extract OI symbols from series
            oi_symbols = []
            series_data = result.raw.get('series', [])
            for series in series_data:
                if 'values' in series:
                    for value in series['values']:
                        series_key = value[0]
                        if 'symbol=' in series_key:
                            oi_symbol = series_key.split('symbol=')[1]
                            
                            # Check if this OI symbol matches our base symbol
                            if self._matches_base_symbol(oi_symbol, base_symbol):
                                oi_symbols.append(oi_symbol)
            
            # Remove duplicates while preserving order
            oi_symbols = list(dict.fromkeys(oi_symbols))
            
            if not oi_symbols:
                logger.info(f"❌ No OI symbols found for {base_symbol}")
                return []
            
            logger.info(f"📊 Found {len(oi_symbols)} potential OI symbols for {base_symbol}: {oi_symbols}")
            
            # Validate data quality for each OI symbol
            validated_symbols = []
            for oi_symbol in oi_symbols:
                data_quality = self._check_oi_data_quality(oi_symbol, hours_lookback)
                
                if data_quality >= min_data_points:
                    validated_symbols.append(oi_symbol)
                    logger.info(f"✅ {oi_symbol}: {data_quality} OI data points")
                else:
                    logger.debug(f"❌ {oi_symbol}: {data_quality} OI data points (insufficient)")
            
            logger.info(f"🚀 Validated OI symbols for {base_symbol}: {validated_symbols}")
            return validated_symbols
            
        except Exception as e:
            logger.error(f"Error discovering OI symbols for {base_symbol}: {e}")
            return []
            
    def _matches_base_symbol(self, oi_symbol: str, base_symbol: str) -> bool:
        """Check if OI symbol matches the base symbol"""
        oi_upper = oi_symbol.upper()
        base_upper = base_symbol.upper()
        
        # Direct match
        if oi_upper == base_upper:
            return True
            
        # Common OI symbol patterns
        patterns = [
            f"{base_upper}USD",      # BTCUSD
            f"{base_upper}USDT",     # BTCUSDT
            f"{base_upper}USDC",     # BTCUSDC
            f"{base_upper}-USD",     # BTC-USD
            f"{base_upper}-USDT",    # BTC-USDT
            f"{base_upper}-USDC",    # BTC-USDC
            f"{base_upper}_USD",     # BTC_USD
            f"{base_upper}_USDT",    # BTC_USDT
            f"{base_upper}_USDC",    # BTC_USDC
            f"{base_upper}-PERP",    # BTC-PERP
            f"{base_upper}_PERP",    # BTC_PERP
            f"{base_upper}PERP",     # BTCPERP
            f"XBT" if base_upper == "BTC" else f"{base_upper}",  # Kraken XBT
        ]
        
        # Check if OI symbol matches any pattern
        for pattern in patterns:
            if oi_upper == pattern or oi_upper.startswith(pattern):
                return True
                
        return False
        
    def _check_oi_data_quality(self, oi_symbol: str, hours_lookback: int) -> int:
        """Check Open Interest data quality for a symbol"""
        try:
            query = f"""
            SELECT COUNT(open_interest) as data_points
            FROM "open_interest"
            WHERE symbol = '{oi_symbol}'
            AND time > now() - {hours_lookback}h
            """
            
            result = self.influx_client.query(query)
            points = list(result.get_points())
            
            if points and points[0]['data_points']:
                return int(points[0]['data_points'])
            return 0
            
        except Exception as e:
            logger.warning(f"Error checking OI data quality for {oi_symbol}: {e}")
            return 0


# Global instance for easy import
symbol_discovery = SymbolDiscovery()