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
from data.storage.influxdb_handler import InfluxDBHandler
import redis

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        """Setup exchange configurations"""
        # CONFIGURABLE: Focus on BTC + ETH only with correct symbol formats per exchange
        self.exchanges = {
            'binance': {
                'name': 'BINANCE',
                'url': 'https://fapi.binance.com/fapi/v1/openInterest',
                'symbols': ['BTCUSDT', 'ETHUSDT'],  # Binance futures format
                'rate_limit': 1200,  # requests per minute
                'parser': self._parse_binance_oi
            },
            'bybit': {
                'name': 'BYBIT',
                'url': 'https://api.bybit.com/v5/market/open-interest',  # Updated API endpoint
                'symbols': ['BTCUSDT', 'ETHUSDT'],  # Bybit linear format
                'rate_limit': 600,
                'parser': self._parse_bybit_oi
            },
            'okx': {
                'name': 'OKX',
                'url': 'https://www.okx.com/api/v5/public/open-interest',
                'symbols': ['BTC-USDT-SWAP', 'ETH-USDT-SWAP'],  # OKX swap format
                'rate_limit': 600,
                'parser': self._parse_okx_oi
            },
            'deribit': {
                'name': 'DERIBIT',
                'url': 'https://www.deribit.com/api/v2/public/get_book_summary_by_currency',
                'symbols': ['BTC', 'ETH'],  # Deribit currency format
                'rate_limit': 300,
                'parser': self._parse_deribit_oi
            }
        }
        
    async def test_influxdb_connection(self) -> bool:
        """Test InfluxDB connection"""
        try:
            return self.influx_handler.test_connection()
        except Exception as e:
            self.logger.error(f"InfluxDB connection test failed: {e}")
            return False
            
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
                'open_interest': float(data.get('openInterest', 0)),
                'open_interest_value': float(data.get('openInterest', 0)) * float(data.get('markPrice', 0)),
                'funding_rate': None,  # Would need separate API call
                'metadata': {'raw_data': data}
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
        """Store OI data in InfluxDB"""
        if not oi_data:
            return True
            
        try:
            # Prepare data for InfluxDB insertion
            influx_points = []
            
            for oi in oi_data:
                point = {
                    'tags': {
                        'exchange': oi['exchange'],
                        'symbol': oi['symbol']
                    },
                    'fields': {
                        'open_interest': float(oi.get('open_interest', 0)),
                        'open_interest_usd': float(oi.get('open_interest_value', 0)) if oi.get('open_interest_value') else 0.0,
                        'funding_rate': float(oi.get('funding_rate', 0)) if oi.get('funding_rate') else 0.0
                    },
                    'timestamp': oi['timestamp']
                }
                influx_points.append(point)
            
            # Write to InfluxDB
            success = self.influx_handler.write_trading_data('open_interest', influx_points)
            
            if success:
                self.logger.info(f"Stored {len(influx_points)} OI records in InfluxDB")
                return True
            else:
                self.logger.error("Failed to store OI data in InfluxDB")
                return False
            
        except Exception as e:
            self.logger.error(f"Error storing OI data in InfluxDB: {e}")
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
        """Single collection cycle for all exchanges"""
        self.logger.info("Starting OI collection cycle")
        start_time = time.time()
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'SqueezeFlow-OI-Tracker/1.0'}
        ) as session:
            
            all_oi_data = []
            
            # Collect from all exchanges
            for exchange, config in self.exchanges.items():
                self.logger.info(f"Collecting OI data from {exchange}")
                oi_data = await self.fetch_oi_data(session, exchange, config)
                if oi_data:
                    all_oi_data.extend(oi_data)
                    
            # Store and cache data
            if all_oi_data:
                await self.store_oi_data(all_oi_data)
                await self.cache_oi_data(all_oi_data)
                
        elapsed = time.time() - start_time
        self.logger.info(f"Collection cycle completed in {elapsed:.2f}s, collected {len(all_oi_data)} OI records")
        
    async def run(self):
        """Main run loop"""
        self.logger.info("Starting Open Interest Tracker")
        self.running = True
        
        # Initial InfluxDB check  
        if not await self.test_influxdb_connection():
            self.logger.error("Cannot connect to InfluxDB, exiting")
            return
        
        try:
            while self.running:
                await self.collect_cycle()
                
                # Wait 1 minute between cycles
                await asyncio.sleep(60)
                
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