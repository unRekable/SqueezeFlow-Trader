#!/usr/bin/env python3
"""
Open Interest Tracker for SqueezeFlow Trader
Collects Open Interest data from multiple exchanges
"""

import asyncio
import aiohttp
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import time
# import psycopg2  # Migrated to InfluxDB
# from psycopg2.extras import execute_batch
import redis

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Handle InfluxDBHandler import with fallback
try:
    from data.storage.influxdb_handler import InfluxDBHandler
except ImportError:
    # Fallback implementation for Docker environment
    from influxdb import InfluxDBClient
    class InfluxDBHandler:
        def __init__(self, host='localhost', port=8086, username='', password='', database='significant_trades'):
            self.client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)
            self.database = database
        
        def test_connection(self):
            """Test InfluxDB connection"""
            try:
                self.client.ping()
                return True
            except Exception as e:
                print(f"InfluxDB connection test failed: {e}")
                return False
        
        def write_trading_data(self, measurement: str, points: List[Dict]):
            """Write trading data points to InfluxDB"""
            try:
                # Convert points to InfluxDB format
                influx_points = []
                for point in points:
                    influx_point = {
                        'measurement': measurement,
                        'tags': point.get('tags', {}),
                        'fields': point.get('fields', {}),
                        'time': point.get('timestamp', datetime.now(timezone.utc))
                    }
                    influx_points.append(influx_point)
                
                return self.client.write_points(influx_points)
            except Exception as e:
                print(f"InfluxDB write error: {e}")
                return False

class OITracker:
    def __init__(self, influx_handler: InfluxDBHandler = None):
        self.setup_logging()
        self.setup_influxdb(influx_handler)
        self.setup_redis()
        self.setup_exchanges()
        self.running = False
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('OITracker')
        
    def setup_influxdb(self, influx_handler: InfluxDBHandler = None):
        """Setup InfluxDB connection"""
        if influx_handler:
            self.influx_handler = influx_handler
        else:
            # Use default InfluxDB configuration
            self.influx_handler = InfluxDBHandler(
                host=os.getenv('INFLUX_HOST', 'host.docker.internal'),
                port=int(os.getenv('INFLUX_PORT', 8086)),
                username=os.getenv('INFLUX_USER', 'squeezeflow'),
                password=os.getenv('INFLUX_PASSWORD', 'password123'),
                database=os.getenv('INFLUX_DATABASE', 'significant_trades')
            )
        
        # Test connection
        if not self.influx_handler.test_connection():
            self.logger.error("Failed to connect to InfluxDB")
            raise ConnectionError("InfluxDB connection failed")
        
        self.logger.info("InfluxDB connection established for OI tracking")
        
    def setup_redis(self):
        """Setup Redis connection"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
    # Database URL parsing no longer needed with InfluxDB
        
    def setup_exchanges(self):
        """Setup exchange configurations with optimal intervals"""
        self.exchanges = {
            'binance': {
                'name': 'BINANCE',
                'url': 'https://fapi.binance.com/fapi/v1/openInterest',
                'interval': 900,  # 15 minutes (native frequency)
                'rate_limit': 2400,  # requests per minute
                'parser': self._parse_binance_oi,
                'symbol_format': self._format_binance_symbol
            },
            'bybit': {
                'name': 'BYBIT', 
                'url': 'https://api.bybit.com/v5/market/open-interest',
                'interval': 60,   # 1 minute (real-time capable)
                'rate_limit': 600,
                'parser': self._parse_bybit_oi,
                'symbol_format': self._format_bybit_symbol
            },
            'okx': {
                'name': 'OKX',
                'url': 'https://www.okx.com/api/v5/public/open-interest',
                'interval': 60,   # 1 minute
                'rate_limit': 600,
                'parser': self._parse_okx_oi,
                'symbol_format': self._format_okx_symbol
            },
            'deribit': {
                'name': 'DERIBIT',
                'url': 'https://www.deribit.com/api/v2/public/get_book_summary_by_currency',
                'interval': 60,   # 1 minute via WebSocket eventually
                'rate_limit': 300,
                'parser': self._parse_deribit_oi,
                'symbol_format': self._format_deribit_symbol
            }
        }
        
        # Track last collection time per exchange
        self.last_collection = {exchange: 0 for exchange in self.exchanges}
        
        # Market discovery no longer needed - using config-based symbols
        self.market_discovery = None
        self.symbol_discovery = None
    
    
    def get_configured_symbols(self) -> List[str]:
        """
        Get symbols from environment variable configuration
        Simple, reliable, no database queries needed
        """
        symbols_env = os.getenv('OI_SYMBOLS', 'BTC,ETH')
        symbols = [s.strip().upper() for s in symbols_env.split(',')]
        self.logger.info(f"📋 Using configured symbols: {symbols}")
        return symbols
    
    def discover_active_futures_symbols(self) -> Dict[str, List[str]]:
        """
        Get configured symbols and validate they have futures markets
        Returns dict mapping base symbols to their futures markets
        """
        futures_symbols = {}
        
        # Get symbols from configuration
        configured_symbols = self.get_configured_symbols()
        
        # Generate futures markets dynamically for each symbol
        for symbol in configured_symbols:
            symbol_futures = self._generate_futures_markets(symbol)
            if symbol_futures:
                futures_symbols[symbol] = symbol_futures
                self.logger.info(f"📈 {symbol}: {len(symbol_futures)} futures markets generated")
            else:
                self.logger.warning(f"⚠️ {symbol}: No futures markets available, skipping")
        
        self.logger.info(f"✅ Found {len(futures_symbols)} symbols with futures markets for OI collection")
        return futures_symbols
    
    def _generate_futures_markets(self, symbol: str) -> List[str]:
        """
        Dynamically generate futures market identifiers for a symbol
        Knows the syntax patterns for each exchange
        """
        futures = []
        
        # PI is just a regular symbol, no special handling needed
        
        # Handle XBT/BTC mapping for Kraken
        kraken_symbol = 'XBT' if symbol == 'BTC' else symbol
        
        # Generate futures market identifiers for each exchange
        exchange_patterns = {
            'BINANCE_FUTURES': f"{symbol.lower()}usdt",           # binance_futures:btcusdt
            'BYBIT': f"{symbol.upper()}USDT",                     # BYBIT:BTCUSDT  
            'OKEX': f"{symbol.upper()}-USDT-SWAP",                # OKEX:BTC-USDT-SWAP
            'DERIBIT': f"{symbol.upper()}-PERPETUAL",             # DERIBIT:BTC-PERPETUAL (BTC/ETH only)
        }
        
        for exchange, market_format in exchange_patterns.items():
            # Deribit only has BTC and ETH perpetuals
            if exchange == 'DERIBIT' and symbol not in ['BTC', 'ETH']:
                continue
                
            market_id = f"{exchange}:{market_format}"
            futures.append(market_id)
        
        return futures
    
    def _format_binance_symbol(self, market: str) -> str:
        """Convert market format to Binance API format"""
        # BINANCE_FUTURES:btcusdt -> BTCUSDT
        return market.split(':')[-1].upper()
    
    def _format_bybit_symbol(self, market: str) -> str:
        """Convert market format to Bybit API format"""
        # BYBIT:btcusdt -> BTCUSDT
        return market.split(':')[-1].upper()
    
    def _format_okx_symbol(self, market: str) -> str:
        """Convert market format to OKX API format"""
        # OKEX:BTC-USDT-SWAP -> BTC-USDT-SWAP
        return market.split(':')[-1]
    
    def _format_deribit_symbol(self, market: str) -> str:
        """Convert market format to Deribit API format"""
        # DERIBIT:BTC-PERPETUAL -> BTC
        symbol_part = market.split(':')[-1]
        return symbol_part.split('-')[0]
    
    def _extract_base_symbol(self, symbol: str) -> str:
        """Extract base symbol from trading pair (BTCUSDT -> BTC)"""
        symbol = symbol.upper()
        
        # Known quote currencies to remove
        quote_currencies = ['USDT', 'USDC', 'FDUSD', 'USD', 'BUSD', 'DAI', 'EUR']
        
        # Try removing each quote currency
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if base:  # Make sure we have a base symbol left
                    return base
        
        # Special cases
        if 'BTC' in symbol:
            return 'BTC'
        elif 'ETH' in symbol:
            return 'ETH'
        elif 'XBT' in symbol:  # Kraken uses XBT
            return 'BTC'
        
        # Fallback: return first 3-4 characters
        return symbol[:3] if len(symbol) >= 3 else symbol
    
    async def should_collect_from_exchange(self, exchange: str) -> bool:
        """Check if enough time has passed since last collection for this exchange"""
        current_time = time.time()
        last_time = self.last_collection[exchange]
        interval = self.exchanges[exchange]['interval']
        
        return (current_time - last_time) >= interval
        
    async def test_influxdb_connection(self) -> bool:
        """Test InfluxDB connection"""
        try:
            return self.influx_handler.test_connection()
        except Exception as e:
            self.logger.error(f"InfluxDB connection test failed: {e}")
            return False
    
    async def fetch_oi_data_for_exchange(self, session: aiohttp.ClientSession, 
                                       exchange: str, symbols_dict: Dict[str, List[str]]) -> List[Dict]:
        """Fetch OI data from a specific exchange for relevant symbols"""
        
        config = self.exchanges[exchange]
        oi_data = []
        
        # Find markets for this exchange
        exchange_markets = []
        for symbol, markets in symbols_dict.items():
            for market in markets:
                if market.upper().startswith(exchange.upper()) or (exchange == 'okx' and 'OKEX:' in market.upper()):
                    exchange_markets.append((symbol, market))
        
        if not exchange_markets:
            self.logger.debug(f"No markets found for {exchange}")
            return []
        
        self.logger.info(f"🔄 Collecting OI from {exchange} for {len(exchange_markets)} markets")
        
        for base_symbol, market in exchange_markets:
            try:
                # Format symbol for this exchange's API
                api_symbol = config['symbol_format'](market)
                
                if exchange == 'binance':
                    data = await self._fetch_binance_oi_single(session, api_symbol)
                elif exchange == 'bybit':
                    data = await self._fetch_bybit_oi_single(session, api_symbol)
                elif exchange == 'okx':
                    data = await self._fetch_okx_oi_single(session, api_symbol)
                elif exchange == 'deribit':
                    data = await self._fetch_deribit_oi_single(session, api_symbol)
                
                if data:
                    # Add base symbol info
                    data['base_symbol'] = base_symbol
                    data['market'] = market
                    oi_data.append(data)
                
                # Rate limiting
                await asyncio.sleep(60 / config['rate_limit'])
                
            except Exception as e:
                self.logger.error(f"Error fetching OI from {exchange} for {market}: {e}")
        
        # Update last collection time
        self.last_collection[exchange] = time.time()
        return oi_data
        
    async def _fetch_binance_oi_single(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        """Fetch OI data from Binance for single symbol"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_binance_oi(symbol, data)
        except Exception as e:
            self.logger.error(f"Binance OI fetch error for {symbol}: {e}")
        return None
    
    async def _fetch_bybit_oi_single(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        """Fetch OI data from Bybit for single symbol"""
        try:
            url = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_bybit_oi(symbol, data)
        except Exception as e:
            self.logger.error(f"Bybit OI fetch error for {symbol}: {e}")
        return None
    
    async def _fetch_okx_oi_single(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        """Fetch OI data from OKX for single symbol"""
        try:
            url = f"https://www.okx.com/api/v5/public/open-interest?instId={symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_okx_oi(symbol, data)
        except Exception as e:
            self.logger.error(f"OKX OI fetch error for {symbol}: {e}")
        return None
    
    async def _fetch_deribit_oi_single(self, session: aiohttp.ClientSession, currency: str) -> Optional[Dict]:
        """Fetch OI data from Deribit for single currency"""
        try:
            url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=future"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_deribit_oi(currency, data)
        except Exception as e:
            self.logger.error(f"Deribit OI fetch error for {currency}: {e}")
        return None
            
    async def fetch_oi_data(self, session: aiohttp.ClientSession, exchange: str, config: Dict) -> Optional[List[Dict]]:
        """Fetch OI data from exchange"""
        try:
            if exchange == 'binance':
                return await self._fetch_binance_oi(session, config)
            elif exchange == 'bybit':
                return await self._fetch_bybit_oi(session, config)
            elif exchange == 'okx':
                return await self._fetch_okx_oi(session, config)
            elif exchange == 'deribit':
                return await self._fetch_deribit_oi(session, config)
            else:
                self.logger.warning(f"Unknown exchange: {exchange}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching OI data from {exchange}: {e}")
            return None
            
    async def _fetch_binance_oi(self, session: aiohttp.ClientSession, config: Dict) -> List[Dict]:
        """Fetch OI data from Binance"""
        oi_data = []
        
        for symbol in config['symbols']:
            try:
                url = f"{config['url']}?symbol={symbol}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        parsed = config['parser'](symbol, data)
                        if parsed:
                            oi_data.append(parsed)
                    else:
                        self.logger.warning(f"Binance OI API error for {symbol}: {response.status}")
                        
                # Rate limiting
                await asyncio.sleep(60 / config['rate_limit'])
                
            except Exception as e:
                self.logger.error(f"Error fetching Binance OI for {symbol}: {e}")
                
        return oi_data
        
    async def _fetch_bybit_oi(self, session: aiohttp.ClientSession, config: Dict) -> List[Dict]:
        """Fetch OI data from Bybit"""
        oi_data = []
        
        for symbol in config['symbols']:
            try:
                # Updated Bybit v5 API format
                url = f"{config['url']}?category=linear&symbol={symbol}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        parsed = config['parser'](symbol, data)
                        if parsed:
                            oi_data.append(parsed)
                    else:
                        self.logger.warning(f"Bybit OI API error for {symbol}: {response.status}")
                        
                await asyncio.sleep(60 / config['rate_limit'])
                
            except Exception as e:
                self.logger.error(f"Error fetching Bybit OI for {symbol}: {e}")
                
        return oi_data
        
    async def _fetch_okx_oi(self, session: aiohttp.ClientSession, config: Dict) -> List[Dict]:
        """Fetch OI data from OKX"""
        oi_data = []
        
        for symbol in config['symbols']:
            try:
                url = f"{config['url']}?instId={symbol}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        parsed = config['parser'](symbol, data)
                        if parsed:
                            oi_data.append(parsed)
                    else:
                        self.logger.warning(f"OKX OI API error for {symbol}: {response.status}")
                        
                await asyncio.sleep(60 / config['rate_limit'])
                
            except Exception as e:
                self.logger.error(f"Error fetching OKX OI for {symbol}: {e}")
                
        return oi_data
        
    async def _fetch_deribit_oi(self, session: aiohttp.ClientSession, config: Dict) -> List[Dict]:
        """Fetch OI data from Deribit"""
        oi_data = []
        
        for currency in config['symbols']:
            try:
                url = f"{config['url']}?currency={currency}&kind=future"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        parsed = config['parser'](currency, data)
                        if parsed:
                            oi_data.extend(parsed)  # Deribit returns multiple instruments
                    else:
                        self.logger.warning(f"Deribit OI API error for {currency}: {response.status}")
                        
                await asyncio.sleep(60 / config['rate_limit'])
                
            except Exception as e:
                self.logger.error(f"Error fetching Deribit OI for {currency}: {e}")
                
        return oi_data
        
    def _parse_binance_oi(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Parse Binance OI response"""
        try:
            return {
                'exchange': 'BINANCE',
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'open_interest': float(data.get('openInterest', 0))
            }
        except Exception as e:
            self.logger.error(f"Error parsing Binance OI data: {e}")
            return None
            
    def _parse_bybit_oi(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Parse Bybit OI response"""
        try:
            # Updated for Bybit v5 API response format
            result = data.get('result', {})
            if isinstance(result, dict):
                list_data = result.get('list', [])
                if len(list_data) > 0:
                    oi_data = list_data[0]
                    return {
                        'exchange': 'BYBIT',
                        'symbol': symbol,
                        'timestamp': datetime.now(timezone.utc),
                        'open_interest': float(oi_data.get('openInterest', 0)),
                        'open_interest_value': float(oi_data.get('openInterestValue', 0)),
                        'funding_rate': None,
                        'metadata': {'raw_data': data}
                    }
        except Exception as e:
            self.logger.error(f"Error parsing Bybit OI data: {e}")
            return None
            
    def _parse_okx_oi(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Parse OKX OI response"""
        try:
            result_data = data.get('data', [])
            if len(result_data) > 0:
                oi_data = result_data[0]
                # Convert OKX symbol format to standard
                standard_symbol = symbol.replace('-USDT-SWAP', 'USDT')
                return {
                    'exchange': 'OKX',
                    'symbol': standard_symbol,
                    'timestamp': datetime.now(timezone.utc),
                    'open_interest': float(oi_data.get('oi', 0)),
                    'open_interest_value': float(oi_data.get('oiUsd', 0)),
                    'funding_rate': None,
                    'metadata': {'raw_data': data}
                }
        except Exception as e:
            self.logger.error(f"Error parsing OKX OI data: {e}")
            return None
            
    def _parse_deribit_oi(self, currency: str, data: Dict) -> List[Dict]:
        """Parse Deribit OI response"""
        oi_data = []
        try:
            result = data.get('result', [])
            for instrument in result:
                if instrument.get('instrument_name', '').endswith('PERPETUAL'):
                    # Extract symbol from instrument name (e.g., BTC-PERPETUAL -> BTCUSD)
                    symbol = f"{currency}USD"
                    oi_data.append({
                        'exchange': 'DERIBIT',
                        'symbol': symbol,
                        'timestamp': datetime.now(timezone.utc),
                        'open_interest': float(instrument.get('open_interest', 0)),
                        'open_interest_value': float(instrument.get('open_interest', 0)) * float(instrument.get('mark_price', 0)),
                        'funding_rate': None,
                        'metadata': {'raw_data': instrument}
                    })
        except Exception as e:
            self.logger.error(f"Error parsing Deribit OI data: {e}")
            
        return oi_data
        
    async def store_oi_data(self, oi_data: List[Dict]) -> bool:
        """Store OI data in InfluxDB with individual, futures aggregate, and total aggregate records"""
        if not oi_data:
            return True
            
        try:
            # Step 1: Aggregate OI by base symbol and exchange (individual records)
            individual_aggregates = {}
            
            for oi in oi_data:
                # Extract base symbol (BTCUSDT -> BTC)
                base_symbol = self._extract_base_symbol(oi['symbol'])
                exchange = oi['exchange']
                
                # Create aggregate key
                key = f"{exchange}:{base_symbol}"
                
                if key not in individual_aggregates:
                    individual_aggregates[key] = {
                        'exchange': exchange,
                        'base_symbol': base_symbol,
                        'original_symbol': oi['symbol'],  # Keep one original for reference
                        'market': oi.get('market', ''),
                        'timestamp': oi['timestamp'],
                        'total_open_interest': 0.0,
                        'pair_count': 0
                    }
                
                # Aggregate values
                individual_aggregates[key]['total_open_interest'] += float(oi.get('open_interest', 0))
                individual_aggregates[key]['pair_count'] += 1
            
            # Step 2: Create cross-exchange aggregates by base symbol
            futures_exchanges = ['BINANCE', 'BYBIT', 'OKX']  # Top 3 futures exchanges
            all_exchanges = ['BINANCE', 'BYBIT', 'OKX', 'DERIBIT']  # All 4 exchanges
            
            symbol_aggregates = {}  # Track aggregates by base symbol
            
            for key, individual in individual_aggregates.items():
                base_symbol = individual['base_symbol']
                exchange = individual['exchange']
                oi_value = individual['total_open_interest']
                
                # Initialize base symbol tracking
                if base_symbol not in symbol_aggregates:
                    symbol_aggregates[base_symbol] = {
                        'futures_total': 0.0,  # Top 3 futures exchanges
                        'total_all': 0.0,      # All 4 exchanges  
                        'futures_exchanges': [],
                        'all_exchanges': [],
                        'timestamp': individual['timestamp']
                    }
                
                # Add to futures aggregate (top 3)
                if exchange in futures_exchanges:
                    symbol_aggregates[base_symbol]['futures_total'] += oi_value
                    symbol_aggregates[base_symbol]['futures_exchanges'].append(f"{exchange}:{oi_value:,.0f}")
                
                # Add to total aggregate (all 4)
                if exchange in all_exchanges:
                    symbol_aggregates[base_symbol]['total_all'] += oi_value
                    symbol_aggregates[base_symbol]['all_exchanges'].append(f"{exchange}:{oi_value:,.0f}")
            
            # Step 3: Prepare all InfluxDB points
            influx_points = []
            
            # Individual exchange records
            for key, aggregate in individual_aggregates.items():
                point = {
                    'tags': {
                        'exchange': aggregate['exchange'],
                        'symbol': aggregate['base_symbol'],  # Store as BTC, ETH, etc.
                        'base_symbol': aggregate['base_symbol'],
                        'original_symbol': aggregate['original_symbol'],
                        'market': aggregate['market'],
                        'aggregate_type': 'individual'
                    },
                    'fields': {
                        'open_interest': aggregate['total_open_interest'],
                        'pair_count': aggregate['pair_count']
                    },
                    'timestamp': aggregate['timestamp']
                }
                influx_points.append(point)
            
            # Futures aggregate records (top 3 exchanges combined)
            for base_symbol, agg_data in symbol_aggregates.items():
                if agg_data['futures_total'] > 0:  # Only create if we have futures data
                    point = {
                        'tags': {
                            'exchange': 'FUTURES_AGG',
                            'symbol': base_symbol,
                            'base_symbol': base_symbol,
                            'original_symbol': 'AGGREGATED',
                            'market': 'FUTURES_COMBINED',
                            'aggregate_type': 'futures_aggregate'
                        },
                        'fields': {
                            'open_interest': agg_data['futures_total'],
                            'exchange_count': len(agg_data['futures_exchanges'])
                        },
                        'timestamp': agg_data['timestamp']
                    }
                    influx_points.append(point)
            
            # Total aggregate records (all 4 exchanges combined)  
            for base_symbol, agg_data in symbol_aggregates.items():
                if agg_data['total_all'] > 0:  # Only create if we have data
                    point = {
                        'tags': {
                            'exchange': 'TOTAL_AGG',
                            'symbol': base_symbol,
                            'base_symbol': base_symbol,
                            'original_symbol': 'AGGREGATED',
                            'market': 'ALL_COMBINED',
                            'aggregate_type': 'total_aggregate'
                        },
                        'fields': {
                            'open_interest': agg_data['total_all'],
                            'exchange_count': len(agg_data['all_exchanges'])
                        },
                        'timestamp': agg_data['timestamp']
                    }
                    influx_points.append(point)
            
            # Write to InfluxDB
            measurement = os.getenv('OI_MEASUREMENT', 'open_interest')
            success = self.influx_handler.write_trading_data(measurement, influx_points)
            
            if success:
                individual_count = len(individual_aggregates)
                futures_agg_count = len([s for s in symbol_aggregates.values() if s['futures_total'] > 0])
                total_agg_count = len([s for s in symbol_aggregates.values() if s['total_all'] > 0])
                
                self.logger.info(f"✅ Stored {len(influx_points)} OI records:")
                self.logger.info(f"   📊 {individual_count} individual exchange records")
                self.logger.info(f"   🔥 {futures_agg_count} futures aggregate records (top 3 exchanges)")
                self.logger.info(f"   📈 {total_agg_count} total aggregate records (all 4 exchanges)")
                
                # Log aggregate details
                for base_symbol, agg_data in symbol_aggregates.items():
                    if agg_data['total_all'] > 0:
                        self.logger.debug(f"   {base_symbol}: Futures={agg_data['futures_total']:,.0f}, Total={agg_data['total_all']:,.0f}")
                
                return True
            else:
                self.logger.error("❌ Failed to store OI data in InfluxDB")
                return False
            
        except Exception as e:
            self.logger.error(f"Error storing OI data: {e}")
            return False
            
    async def cache_oi_data(self, oi_data: List[Dict]):
        """Cache OI data in Redis for real-time access"""
        try:
            for oi in oi_data:
                key = f"oi:{oi['exchange']}:{oi['symbol']}"
                value = json.dumps({
                    'open_interest': oi['open_interest'],
                    'open_interest_value': oi.get('open_interest_value'),
                    'funding_rate': oi.get('funding_rate'),
                    'timestamp': oi['timestamp'].isoformat()
                })
                self.redis_client.setex(key, 300, value)  # 5 minute TTL
                
        except Exception as e:
            self.logger.error(f"Error caching OI data: {e}")
            
    async def collect_cycle(self):
        """Enhanced collection cycle with dynamic symbol discovery"""
        self.logger.info("🔄 Starting OI collection cycle...")
        start_time = time.time()
        
        # Discover active futures symbols from database
        futures_symbols = self.discover_active_futures_symbols()
        
        if not futures_symbols:
            self.logger.warning("No futures symbols discovered, skipping cycle")
            return
        
        self.logger.info(f"📊 Collecting OI for {len(futures_symbols)} symbols across {len(self.exchanges)} exchanges")
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'SqueezeFlow-OI-Tracker/1.0'}
        ) as session:
            
            all_oi_data = []
            
            # Collect from each exchange (respecting intervals)
            for exchange in self.exchanges:
                if await self.should_collect_from_exchange(exchange):
                    oi_data = await self.fetch_oi_data_for_exchange(session, exchange, futures_symbols)
                    if oi_data:
                        all_oi_data.extend(oi_data)
                        self.logger.info(f"📈 Collected {len(oi_data)} OI records from {exchange}")
                else:
                    self.logger.debug(f"⏳ Skipping {exchange} (interval not reached)")
            
            # Store all collected data
            if all_oi_data:
                await self.store_oi_data(all_oi_data)
            
        elapsed = time.time() - start_time
        self.logger.info(f"✅ Collection cycle completed in {elapsed:.2f}s, collected {len(all_oi_data)} total OI records")
        
    async def run(self):
        """Main run loop"""
        self.logger.info("Starting Open Interest Tracker")
        self.running = True
        
        # Initial InfluxDB check  
        if not await self.test_influxdb_connection():
            self.logger.error("Cannot connect to InfluxDB, exiting")
            return
        
        # Initial symbol discovery
        futures_symbols = self.discover_active_futures_symbols()
        self.logger.info(f"📊 Initialized with {len(futures_symbols)} symbols for OI tracking")
        
        try:
            while self.running:
                await self.collect_cycle()
                
                # Wait 60 seconds between cycles (master interval)
                collection_interval = int(os.getenv('OI_COLLECTION_INTERVAL', 60))
                await asyncio.sleep(collection_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.running = False
            self.logger.info("Open Interest Tracker stopped")
            
    def stop(self):
        """Stop the tracker"""
        self.running = False

async def main():
    """Main function"""
    tracker = OITracker()
    await tracker.run()

if __name__ == "__main__":
    asyncio.run(main())