#!/usr/bin/env python3
"""
Exchange Mapper - Intelligent Spot/Perp Classification System
Automatically detects market type (spot/perp) for any exchange pair
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExchangePattern:
    """Pattern definition for exchange market identification"""
    exchange: str
    spot_patterns: List[str]
    perp_patterns: List[str]


class ExchangeMapper:
    """
    Intelligent Exchange Mapping System
    Classifies market pairs as SPOT or PERP based on exchange-specific patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ExchangeMapper')
        self.patterns = self._build_exchange_patterns()
        self.known_mappings = self._load_known_mappings()
        
    def _build_exchange_patterns(self) -> Dict[str, ExchangePattern]:
        """Build exchange-specific patterns for automatic detection"""
        patterns = {
            'BINANCE': ExchangePattern(
                exchange='BINANCE',
                spot_patterns=[r'^BINANCE:[a-z0-9]+$'],
                perp_patterns=[r'^BINANCE_FUTURES:[a-z0-9]+$']
            ),
            'BINANCE_US': ExchangePattern(
                exchange='BINANCE_US',
                spot_patterns=[r'^BINANCE_US:[a-z0-9]+$'],
                perp_patterns=[]
            ),
            'BITFINEX': ExchangePattern(
                exchange='BITFINEX',
                spot_patterns=[r'^BITFINEX:[A-Z0-9]+$'],
                perp_patterns=[r'^BITFINEX:[A-Z0-9]+F0:[A-Z0-9]+F0$']
            ),
            'BITGET': ExchangePattern(
                exchange='BITGET',
                spot_patterns=[r'^BITGET:[A-Z0-9]+$'],
                perp_patterns=[r'^BITGET:[A-Z0-9]+_UMCBL$']
            ),
            'BITMART': ExchangePattern(
                exchange='BITMART',
                spot_patterns=[r'^BITMART:[A-Z0-9_]+$'],
                perp_patterns=[]
            ),
            'BITMEX': ExchangePattern(
                exchange='BITMEX',
                spot_patterns=[],
                perp_patterns=[r'^BITMEX:[A-Z0-9_]+$']  # Only perps
            ),
            'BITSTAMP': ExchangePattern(
                exchange='BITSTAMP',
                spot_patterns=[r'^BITSTAMP:[a-z0-9]+$'],
                perp_patterns=[]
            ),
            'BITUNIX': ExchangePattern(
                exchange='BITUNIX',
                spot_patterns=[r'^BITUNIX:[A-Z0-9]+$'],
                perp_patterns=[]
            ),
            'BYBIT': ExchangePattern(
                exchange='BYBIT',
                spot_patterns=[r'^BYBIT:[A-Z0-9]+-SPOT$'],
                perp_patterns=[r'^BYBIT:[A-Z0-9]+$']  # Without -SPOT
            ),
            'COINBASE': ExchangePattern(
                exchange='COINBASE',
                spot_patterns=[r'^COINBASE:[A-Z0-9-]+$'],
                perp_patterns=[r'^COINBASE:[A-Z0-9-]+-PERP-INTX$']
            ),
            'CRYPTOCOM': ExchangePattern(
                exchange='CRYPTOCOM',
                spot_patterns=[r'^CRYPTOCOM:[A-Z0-9_]+$'],
                perp_patterns=[]
            ),
            'DERIBIT': ExchangePattern(
                exchange='DERIBIT',
                spot_patterns=[],
                perp_patterns=[r'^DERIBIT:[A-Z0-9_]+-PERPETUAL$']  # Only perps
            ),
            'DYDX': ExchangePattern(
                exchange='DYDX',
                spot_patterns=[],
                perp_patterns=[r'^DYDX:[A-Z0-9-]+$']  # Only perps
            ),
            'GATEIO': ExchangePattern(
                exchange='GATEIO',
                spot_patterns=[r'^GATEIO:[A-Z0-9_]+-SPOT$'],
                perp_patterns=[r'^GATEIO:[A-Z0-9_]+$']  # Without -SPOT
            ),
            'HITBTC': ExchangePattern(
                exchange='HITBTC',
                spot_patterns=[r'^HITBTC:[A-Z0-9]+$'],
                perp_patterns=[]
            ),
            'HUOBI': ExchangePattern(
                exchange='HUOBI',
                spot_patterns=[r'^HUOBI:[a-z0-9]+$'],
                perp_patterns=[r'^HUOBI:[A-Z0-9-]+$']
            ),
            'HYPERLIQUID': ExchangePattern(
                exchange='HYPERLIQUID',
                spot_patterns=[],
                perp_patterns=[r'^HYPERLIQUID:[A-Z0-9]+$']  # Only perps
            ),
            'KRAKEN': ExchangePattern(
                exchange='KRAKEN',
                spot_patterns=[r'^KRAKEN:[A-Z0-9/]+$'],
                perp_patterns=[r'^KRAKEN:PI_[A-Z0-9]+$']
            ),
            'KUCOIN': ExchangePattern(
                exchange='KUCOIN',
                spot_patterns=[r'^KUCOIN:[A-Z0-9-]+$'],
                perp_patterns=[r'^KUCOIN:[A-Z0-9]+M$']  # Ends with M
            ),
            'MEXC': ExchangePattern(
                exchange='MEXC',
                spot_patterns=[r'^MEXC:[A-Z0-9_]+T$'],  # Ends with T (USDT, etc.)
                perp_patterns=[r'^MEXC:[A-Z0-9_]+D$']   # Ends with D (USD perpetuals)
            ),
            'OKEX': ExchangePattern(
                exchange='OKEX',
                spot_patterns=[r'^OKEX:[A-Z0-9-]+$'],
                perp_patterns=[r'^OKEX:[A-Z0-9-]+-SWAP$']
            ),
            'PHEMEX': ExchangePattern(
                exchange='PHEMEX',
                spot_patterns=[r'^PHEMEX:s[A-Z0-9]+$'],  # Starts with 's'
                perp_patterns=[r'^PHEMEX:[c]?[A-Z0-9]+$']  # No 's' prefix or 'c' prefix
            ),
            'POLONIEX': ExchangePattern(
                exchange='POLONIEX',
                spot_patterns=[r'^POLONIEX:[A-Z0-9_]+$'],
                perp_patterns=[]
            )
        }
        
        return patterns
        
    def _load_known_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Load the complete aggr-server exchange mappings exactly as provided"""
        return {
            "BINANCE": {
                "spot": [
                    "BINANCE:btcusdt", "BINANCE:ethusdt", "BINANCE:btcusdc", 
                    "BINANCE:ethusdc", "BINANCE:btcfdusd", "BINANCE:ethfdusd"
                ],
                "perp": [
                    "BINANCE_FUTURES:btcusdt", "BINANCE_FUTURES:ethusdt",
                    "BINANCE_FUTURES:btcusdc", "BINANCE_FUTURES:ethusdc"
                ]
            },
            "BINANCE_US": {
                "spot": [
                    "BINANCE_US:btcusdt", "BINANCE_US:ethusdt", "BINANCE_US:btcusdc",
                    "BINANCE_US:ethusdc", "BINANCE_US:btcusd", "BINANCE_US:ethusd"
                ],
                "perp": []
            },
            "BITFINEX": {
                "spot": [
                    "BITFINEX:BTCUSD", "BITFINEX:ETHUSD", "BITFINEX:BTCUST", "BITFINEX:ETHUST"
                ],
                "perp": [
                    "BITFINEX:BTCF0:USTF0", "BITFINEX:ETHF0:USTF0"
                ]
            },
            "BITGET": {
                "spot": [
                    "BITGET:BTCUSDT", "BITGET:ETHUSDT", "BITGET:BTCUSDC", "BITGET:ETHUSDC"
                ],
                "perp": [
                    "BITGET:BTCUSDT_UMCBL", "BITGET:ETHUSDT_UMCBL"
                ]
            },
            "BITMART": {
                "spot": [
                    "BITMART:BTC_USDT", "BITMART:ETH_USDT", "BITMART:BTC_USDC", "BITMART:ETH_USDC",
                    "BITMART:BTC_DAI", "BITMART:ETH_DAI", "BITMART:BTCUSDT", "BITMART:ETHUSDT",
                    "BITMART:BTCUSDC", "BITMART:ETHUSDC"
                ],
                "perp": []
            },
            "BITMEX": {
                "spot": [],
                "perp": [
                    "BITMEX:XBTUSD", "BITMEX:ETHUSD", "BITMEX:XBTUSDT", "BITMEX:ETHUSDT",
                    "BITMEX:XBT_USDT", "BITMEX:ETH_USDT"
                ]
            },
            "BITSTAMP": {
                "spot": [
                    "BITSTAMP:btcusd", "BITSTAMP:ethusd", "BITSTAMP:btcusdt", 
                    "BITSTAMP:ethusdt", "BITSTAMP:btcusdc", "BITSTAMP:ethusdc"
                ],
                "perp": []
            },
            "BITUNIX": {
                "spot": [
                    "BITUNIX:BTCUSDT", "BITUNIX:ETHUSDT"
                ],
                "perp": []
            },
            "BYBIT": {
                "spot": [
                    "BYBIT:BTCUSDT-SPOT", "BYBIT:ETHUSDT-SPOT", "BYBIT:BTCUSDC-SPOT",
                    "BYBIT:ETHUSDC-SPOT", "BYBIT:BTCDAI-SPOT", "BYBIT:ETHDAI-SPOT"
                ],
                "perp": [
                    "BYBIT:BTCUSDT", "BYBIT:ETHUSDT"
                ]
            },
            "COINBASE": {
                "spot": [
                    "COINBASE:BTC-USD", "COINBASE:ETH-USD", "COINBASE:BTC-USDT",
                    "COINBASE:ETH-USDT", "COINBASE:ETH-DAI"
                ],
                "perp": [
                    "COINBASE:BTC-PERP-INTX", "COINBASE:ETH-PERP-INTX"
                ]
            },
            "CRYPTOCOM": {
                "spot": [
                    "CRYPTOCOM:BTC_USDT", "CRYPTOCOM:ETH_USDT", "CRYPTOCOM:BTC_USDC", "CRYPTOCOM:ETH_USDC"
                ],
                "perp": []
            },
            "DERIBIT": {
                "spot": [],
                "perp": [
                    "DERIBIT:BTC-PERPETUAL", "DERIBIT:ETH-PERPETUAL",
                    "DERIBIT:BTC_USDC-PERPETUAL", "DERIBIT:ETH_USDC-PERPETUAL"
                ]
            },
            "DYDX": {
                "spot": [],
                "perp": [
                    "DYDX:BTC-USD", "DYDX:ETH-USD"
                ]
            },
            "GATEIO": {
                "spot": [
                    "GATEIO:BTC_USDT-SPOT", "GATEIO:ETH_USDT-SPOT",
                    "GATEIO:BTC_USDC-SPOT", "GATEIO:ETH_USDC-SPOT"
                ],
                "perp": [
                    "GATEIO:BTC_USDT", "GATEIO:ETH_USDT"
                ]
            },
            "HITBTC": {
                "spot": [
                    "HITBTC:BTCUSD", "HITBTC:ETHUSD", "HITBTC:BTCUSDC",
                    "HITBTC:ETHUSDC", "HITBTC:BTCDAI", "HITBTC:ETHDAI"
                ],
                "perp": []
            },
            "HUOBI": {
                "spot": [
                    "HUOBI:btcusdt", "HUOBI:ethusdt", "HUOBI:btcusdc", "HUOBI:ethusdc"
                ],
                "perp": [
                    "HUOBI:BTC-USDT", "HUOBI:ETH-USDT", "HUOBI:BTC-USD", "HUOBI:ETH-USD"
                ]
            },
            "HYPERLIQUID": {
                "spot": [],
                "perp": [
                    "HYPERLIQUID:BTC", "HYPERLIQUID:ETH"
                ]
            },
            "KRAKEN": {
                "spot": [
                    "KRAKEN:XBT/USD", "KRAKEN:XBT/USDT", "KRAKEN:XBT/USDC", "KRAKEN:XBT/DAI",
                    "KRAKEN:ETH/USD", "KRAKEN:ETH/USDT", "KRAKEN:ETH/USDC", "KRAKEN:ETH/DAI"
                ],
                "perp": [
                    "KRAKEN:PI_XBTUSD", "KRAKEN:PI_ETHUSD"
                ]
            },
            "KUCOIN": {
                "spot": [
                    "KUCOIN:BTC-USDT", "KUCOIN:ETH-USDT", "KUCOIN:BTC-USDC",
                    "KUCOIN:ETH-USDC", "KUCOIN:BTC-DAI", "KUCOIN:ETH-DAI"
                ],
                "perp": [
                    "KUCOIN:XBTUSDTM", "KUCOIN:ETHUSDTM"
                ]
            },
            "MEXC": {
                "spot": [
                    "MEXC:BTC_USDT", "MEXC:ETH_USDT", "MEXC:BTC_USDC", "MEXC:ETH_USDC"
                ],
                "perp": [
                    "MEXC:BTC_USD", "MEXC:ETH_USD"
                ]
            },
            "OKEX": {
                "spot": [
                    "OKEX:BTC-USDT", "OKEX:ETH-USDT", "OKEX:BTC-USDC",
                    "OKEX:ETH-USDC", "OKEX:BTC-USD", "OKEX:ETH-USD"
                ],
                "perp": [
                    "OKEX:BTC-USDT-SWAP", "OKEX:ETH-USDT-SWAP", "OKEX:BTC-USDC-SWAP",
                    "OKEX:ETH-USDC-SWAP", "OKEX:BTC-USD-SWAP", "OKEX:ETH-USD-SWAP"
                ]
            },
            "PHEMEX": {
                "spot": [
                    "PHEMEX:sBTCUSDT", "PHEMEX:sETHUSDT", "PHEMEX:sBTCUSDC", "PHEMEX:sETHUSDC"
                ],
                "perp": [
                    "PHEMEX:BTCUSD", "PHEMEX:cETHUSD"
                ]
            },
            "POLONIEX": {
                "spot": [
                    "POLONIEX:BTC_USDT", "POLONIEX:ETH_USDT", "POLONIEX:BTC_USDC", "POLONIEX:ETH_USDC"
                ],
                "perp": []
            }
        }
        
    def get_market_type(self, market_identifier: str) -> Optional[str]:
        """
        Determine if a market is SPOT or PERP
        
        Args:
            market_identifier: Full market string (e.g., "BINANCE:btcusdt")
            
        Returns:
            "SPOT" or "PERP" or None if unknown
        """
        try:
            # Extract exchange name
            exchange = self.extract_exchange(market_identifier)
            if not exchange:
                return None
                
            # First check known mappings (explicit)
            known = self.known_mappings.get(exchange, {})
            if market_identifier in known.get('spot', []):
                return 'SPOT'
            if market_identifier in known.get('perp', []):
                return 'PERP'
                
            # Then use pattern matching (intelligent detection)
            if exchange in self.patterns:
                pattern = self.patterns[exchange]
                
                # Check perp patterns first (usually more specific)
                for perp_pattern in pattern.perp_patterns:
                    if re.match(perp_pattern, market_identifier):
                        return 'PERP'
                        
                # Check spot patterns
                for spot_pattern in pattern.spot_patterns:
                    if re.match(spot_pattern, market_identifier):
                        return 'SPOT'
                        
            # Fallback: guess based on common patterns
            return self._fallback_detection(market_identifier)
            
        except Exception as e:
            self.logger.error(f"Error determining market type for {market_identifier}: {e}")
            return None
            
    def extract_exchange(self, market_identifier: str) -> Optional[str]:
        """Extract exchange name from market identifier"""
        if ':' not in market_identifier:
            return None
        return market_identifier.split(':')[0]
        
    def _fallback_detection(self, market_identifier: str) -> Optional[str]:
        """Fallback detection using common patterns"""
        identifier_lower = market_identifier.lower()
        
        # Common perp indicators
        perp_indicators = [
            'futures', 'perp', 'perpetual', 'swap', '_umcbl', 
            'pi_', 'f0:', '-perpetual', '-swap', '-perp-intx'
        ]
        
        # Common spot indicators  
        spot_indicators = ['-spot', 's.', 'spot.']
        
        # Check for perp indicators
        for indicator in perp_indicators:
            if indicator in identifier_lower:
                return 'PERP'
                
        # Check for spot indicators
        for indicator in spot_indicators:
            if indicator in identifier_lower:
                return 'SPOT'
                
        # Default to SPOT for most exchanges (conservative approach)
        exchange = self.extract_exchange(market_identifier)
        perp_only_exchanges = ['BITMEX', 'DERIBIT', 'DYDX', 'HYPERLIQUID']
        
        if exchange in perp_only_exchanges:
            return 'PERP'
        else:
            return 'SPOT'  # Default assumption
            
    def get_all_markets_by_type(self, symbol: str = None) -> Dict[str, List[str]]:
        """
        Get all spot and perp markets, optionally filtered by symbol
        
        Args:
            symbol: Symbol to filter by (e.g., 'BTC', 'ETH', 'SOL')
            
        Returns:
            {'spot': [...], 'perp': [...]}
        """
        result = {'spot': [], 'perp': []}
        
        # Get from known mappings (explicit definitions)
        for exchange, markets in self.known_mappings.items():
            for market_type, market_list in markets.items():
                for market in market_list:
                    if symbol:
                        # Smart symbol matching - handle different formats
                        symbol_variants = [
                            symbol.upper(),
                            symbol.lower(), 
                            'XBT' if symbol.upper() == 'BTC' else symbol.upper(),  # Kraken uses XBT
                            'BTC' if symbol.upper() == 'XBT' else symbol.upper()
                        ]
                        
                        # Check if any variant is in the market identifier
                        if any(variant in market.upper() for variant in symbol_variants):
                            result[market_type].append(market)
                    else:
                        result[market_type].append(market)
        
        # Market generation removed - use Market Discovery instead for robust symbol detection
        # Old behavior generated non-existent markets, new approach discovers real data
                        
        return result
    
        
    def build_influx_market_filter(self, symbol: str, market_type: str = None) -> str:
        """
        Build InfluxDB market filter for queries
        
        Args:
            symbol: Base symbol (e.g., 'BTC', 'ETH')
            market_type: 'spot', 'perp', or None for both
            
        Returns:
            InfluxDB WHERE clause string
        """
        markets = self.get_all_markets_by_type(symbol)
        
        if market_type:
            market_list = markets.get(market_type, [])
        else:
            market_list = markets['spot'] + markets['perp']
            
        if not market_list:
            return "1=0"  # No matches
            
        # Build market filter
        market_conditions = [f"market = '{market}'" for market in market_list]
        return f"({' OR '.join(market_conditions)})"
        
    def test_new_symbol(self, base_symbol: str, quote_currency: str, exchange: str) -> str:
        """
        Test how a new symbol would be classified
        
        Example: test_new_symbol('SOL', 'USDC', 'POLONIEX') -> 'SPOT'
        """
        # Build market identifier based on exchange patterns
        market_formats = {
            'POLONIEX': f'POLONIEX:{base_symbol.upper()}_{quote_currency.upper()}',
            'BINANCE': f'BINANCE:{base_symbol.lower()}{quote_currency.lower()}',
            'BYBIT': f'BYBIT:{base_symbol.upper()}{quote_currency.upper()}-SPOT',
            'COINBASE': f'COINBASE:{base_symbol.upper()}-{quote_currency.upper()}',
            'BITMART': f'BITMART:{base_symbol.upper()}_{quote_currency.upper()}',
            'KUCOIN': f'KUCOIN:{base_symbol.upper()}-{quote_currency.upper()}',
        }
        
        market_id = market_formats.get(exchange, f'{exchange}:{base_symbol}_{quote_currency}')
        return self.get_market_type(market_id) or 'UNKNOWN'


# Singleton instance
exchange_mapper = ExchangeMapper()


def get_market_type(market_identifier: str) -> Optional[str]:
    """Convenience function to get market type"""
    return exchange_mapper.get_market_type(market_identifier)


def get_markets_for_symbol(symbol: str, market_type: str = None) -> List[str]:
    """Convenience function to get markets for a symbol"""
    markets = exchange_mapper.get_all_markets_by_type(symbol)
    if market_type:
        return markets.get(market_type, [])
    return markets['spot'] + markets['perp']


def build_cvd_query_filter(symbol: str, market_type: str) -> str:
    """Convenience function to build CVD query filters"""
    return exchange_mapper.build_influx_market_filter(symbol, market_type)


if __name__ == "__main__":
    # Test the system
    mapper = ExchangeMapper()
    
    # Test known markets
    test_markets = [
        "BINANCE:btcusdt",           # Should be SPOT
        "BINANCE_FUTURES:btcusdt",   # Should be PERP
        "BYBIT:BTCUSDT-SPOT",        # Should be SPOT
        "BYBIT:BTCUSDT",             # Should be PERP
        "POLONIEX:SOL_USDC",         # Should be SPOT (new)
        "DERIBIT:BTC-PERPETUAL",     # Should be PERP
    ]
    
    print("🔍 Exchange Mapper Testing:")
    for market in test_markets:
        market_type = mapper.get_market_type(market)
        print(f"  {market:<30} → {market_type}")
        
    # Test symbol filtering
    print(f"\n📊 BTC Markets:")
    btc_markets = mapper.get_all_markets_by_type('BTC')
    print(f"  Spot ({len(btc_markets['spot'])}): {btc_markets['spot'][:3]}...")
    print(f"  Perp ({len(btc_markets['perp'])}): {btc_markets['perp'][:3]}...")
    
    # Test query building
    print(f"\n🔍 Query Filters:")
    spot_filter = mapper.build_influx_market_filter('BTC', 'spot')
    print(f"  BTC Spot Filter: {spot_filter[:100]}...")
    
    # Test new symbol classification
    print(f"\n🆕 New Symbol Tests:")
    test_cases = [
        ('SOL', 'USDC', 'POLONIEX'),
        ('AVAX', 'USDT', 'BINANCE'),
        ('DOT', 'USD', 'COINBASE')
    ]
    
    for base, quote, exchange in test_cases:
        result = mapper.test_new_symbol(base, quote, exchange)
        print(f"  {exchange}:{base}/{quote} → {result}")