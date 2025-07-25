#!/usr/bin/env python3
"""
SqueezeFlow Calculator Service
Real-time multi-exchange CVD analysis and squeeze signal generation
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import redis
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage.influxdb_handler import InfluxDBHandler
from indicators.squeeze_score_calculator import SqueezeScoreCalculator
from utils.exchange_mapper import exchange_mapper


class SqueezeFlowCalculatorService:
    """
    Real-time SqueezeFlow Calculator Service
    Processes multi-exchange data and generates squeeze signals
    """
    
    def __init__(self):
        self.setup_logging()
        self.setup_database()
        self.setup_redis()
        self.setup_calculator()
        self.running = False
        
        # Configuration
        self.calculation_interval = 60  # seconds
        # Dynamic symbol discovery from database - no more hardcoded symbols!
        self.symbols = self.discover_active_symbols()
        # Erweiterte Timeframes für bessere Squeeze-Erkennung
        # Entry: 5,10,15,30min für schnelle Reaktion
        # Primär: 60,120,240min (1h,2h,4h) für echte Squeeze-Signale  
        # Bestätigung: 30,60min für Entry-Timing
        self.lookback_periods = [5, 10, 15, 30, 60, 120, 240]  # Multiple timeframes erweitert
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SqueezeFlowCalculator')
        
    def discover_active_symbols(self) -> List[str]:
        """Discover active symbols from database with data quality validation"""
        try:
            from utils.symbol_discovery import symbol_discovery
            
            # Discover symbols with minimum 500 data points in last 24h
            active_symbols = symbol_discovery.discover_symbols_from_database(
                min_data_points=500, 
                hours_lookback=24
            )
            
            if active_symbols:
                self.logger.info(f"🎯 Discovered active symbols: {active_symbols}")
                return active_symbols
            else:
                # Fallback to default symbols if discovery fails
                fallback_symbols = ['BTC', 'ETH']
                self.logger.warning(f"⚠️ No symbols discovered, using fallback: {fallback_symbols}")
                return fallback_symbols
                
        except Exception as e:
            # Robust fallback if discovery fails
            fallback_symbols = ['BTC', 'ETH']
            self.logger.error(f"❌ Symbol discovery failed: {e}, using fallback: {fallback_symbols}")
            return fallback_symbols
        
    def setup_database(self):
        """Setup InfluxDB connection - connect to the real trading data"""
        from influxdb import InfluxDBClient
        
        # Connect to the actual InfluxDB database - use environment variable for host
        self.influx_client = InfluxDBClient(
            host=os.getenv('INFLUX_HOST', 'aggr-influx'),  # InfluxDB container name
            port=int(os.getenv('INFLUX_PORT', 8086)),
            database='significant_trades'  # This is where the real data is!
        )
        
        # Test connection
        try:
            databases = self.influx_client.get_list_database()
            self.logger.info(f"Connected to InfluxDB, available databases: {[db['name'] for db in databases]}")
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            raise ConnectionError(f"InfluxDB connection failed: {e}")
        
        self.logger.info("InfluxDB connection established")
        
    def setup_redis(self):
        """Setup Redis connection"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
            
    def setup_calculator(self):
        """Setup squeeze score calculator"""
        self.calculator = SqueezeScoreCalculator(
            price_weight=0.3,
            spot_cvd_weight=0.35,
            futures_cvd_weight=0.35,
            smoothing_period=5
        )
        
    async def get_market_data(self, symbol: str, timeframe: str = '1m', 
                             limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get market data from InfluxDB
        """
        try:
            query = f"""
            SELECT mean(price) as price, sum(volume) as volume, 
                   last(cvd_spot) as cvd_spot, last(cvd_perp) as cvd_perp,
                   last(oi_value) as oi_value
            FROM market_data
            WHERE symbol = '{symbol}'
            AND time >= now() - {limit}m
            GROUP BY time({timeframe})
            ORDER BY time DESC
            LIMIT {limit}
            """
            
            result = self.influx_handler.client.query(query)
            points = list(result.get_points())
            
            if not points:
                return None
                
            df = pd.DataFrame(points)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
            
    async def calculate_cvd_aggregates(self, symbol: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Calculate aggregated CVD from all exchanges using Exchange Mapper
        """
        try:
            # Use Market Discovery to get available markets from database
            from utils.market_discovery import market_discovery
            markets_by_type = market_discovery.get_markets_by_type(symbol)
            
            spot_markets = markets_by_type['spot']
            perp_markets = markets_by_type['perp']
            
            if not spot_markets or not perp_markets:
                self.logger.warning(f"No spot ({len(spot_markets)}) or perp ({len(perp_markets)}) markets found for {symbol}")
                return None, None
            
            # Build spot CVD query using real database structure - fix InfluxDB syntax
            spot_market_filter = ' OR '.join([f"market = '{market}'" for market in spot_markets])
            spot_query = f"""
            SELECT sum(vbuy) as total_vbuy_spot, sum(vsell) as total_vsell_spot
            FROM "aggr_1m"."trades_1m"
            WHERE ({spot_market_filter})
            AND time >= now() - 5h
            GROUP BY time(1m)
            ORDER BY time DESC
            LIMIT 300
            """
            
            # Build perp CVD query using real database structure - fix InfluxDB syntax
            perp_market_filter = ' OR '.join([f"market = '{market}'" for market in perp_markets])
            perp_query = f"""
            SELECT sum(vbuy) as total_vbuy_perp, sum(vsell) as total_vsell_perp
            FROM "aggr_1m"."trades_1m"
            WHERE ({perp_market_filter})
            AND time >= now() - 5h
            GROUP BY time(1m)
            ORDER BY time DESC
            LIMIT 300
            """
            
            self.logger.info(f"Calculating CVD for {symbol}: {len(spot_markets)} spot markets, {len(perp_markets)} perp markets")
            
            spot_result = self.influx_client.query(spot_query)
            perp_result = self.influx_client.query(perp_query)
            
            spot_points = list(spot_result.get_points())
            perp_points = list(perp_result.get_points())
            
            if not spot_points or not perp_points:
                self.logger.warning(f"No data points found - spot: {len(spot_points)}, perp: {len(perp_points)}")
                return None, None
                
            spot_df = pd.DataFrame(spot_points)
            perp_df = pd.DataFrame(perp_points)
            
            if not spot_df.empty:
                spot_df['time'] = pd.to_datetime(spot_df['time'])
                # Calculate CVD from vbuy - vsell, handle None values (VERIFIED METHODOLOGY)
                spot_df['total_vbuy_spot'] = spot_df['total_vbuy_spot'].fillna(0)
                spot_df['total_vsell_spot'] = spot_df['total_vsell_spot'].fillna(0)
                # Step 1: Calculate per-minute volume delta (Buy Volume - Sell Volume)
                spot_df['total_cvd_spot'] = spot_df['total_vbuy_spot'] - spot_df['total_vsell_spot']
                spot_df = spot_df.set_index('time').sort_index()
                # Step 2: Calculate CUMULATIVE Volume Delta (running total) - EXACT same logic as debug tool
                spot_df['total_cvd_spot_cumulative'] = spot_df['total_cvd_spot'].cumsum()
                spot_cvd = spot_df['total_cvd_spot_cumulative']  # Use cumulative CVD, not per-minute delta
            else:
                spot_cvd = pd.Series(dtype=float)
                
            if not perp_df.empty:
                perp_df['time'] = pd.to_datetime(perp_df['time'])
                # Calculate CVD from vbuy - vsell, handle None values (VERIFIED METHODOLOGY)
                perp_df['total_vbuy_perp'] = perp_df['total_vbuy_perp'].fillna(0)
                perp_df['total_vsell_perp'] = perp_df['total_vsell_perp'].fillna(0)
                # Step 1: Calculate per-minute volume delta (Buy Volume - Sell Volume)
                perp_df['total_cvd_perp'] = perp_df['total_vbuy_perp'] - perp_df['total_vsell_perp']
                perp_df = perp_df.set_index('time').sort_index()
                # Step 2: Calculate CUMULATIVE Volume Delta (running total) - EXACT same logic as debug tool
                perp_df['total_cvd_perp_cumulative'] = perp_df['total_cvd_perp'].cumsum()
                perp_cvd = perp_df['total_cvd_perp_cumulative']  # Use cumulative CVD, not per-minute delta
            else:
                perp_cvd = pd.Series(dtype=float)
            
            self.logger.info(f"CVD data loaded - spot: {len(spot_cvd)} points, perp: {len(perp_cvd)} points")
            
            return spot_cvd, perp_cvd
            
        except Exception as e:
            self.logger.error(f"Error calculating CVD aggregates for {symbol}: {e}")
            return None, None
            
    async def get_price_data(self, symbol: str) -> Optional[pd.Series]:
        """
        Get price data for squeeze calculation using real database structure
        """
        try:
            # Get all markets for the symbol (both spot and perp for price average)
            from utils.market_discovery import market_discovery
            markets_by_type = market_discovery.get_markets_by_type(symbol)
            market_list = markets_by_type['spot'] + markets_by_type['perp']
            
            if not market_list:
                self.logger.warning(f"No markets found for {symbol}")
                return None
            
            # Build market filter for price query
            market_filter = ' OR '.join([f"market = '{market}'" for market in market_list])
            
            query = f"""
            SELECT mean(close) as price
            FROM "aggr_1m"."trades_1m"
            WHERE ({market_filter})
            AND time >= now() - 5h
            GROUP BY time(1m)
            ORDER BY time DESC
            LIMIT 300
            """
            
            result = self.influx_client.query(query)
            points = list(result.get_points())
            
            if not points:
                self.logger.warning(f"No price data found for {symbol}")
                return None
                
            df = pd.DataFrame(points)
            df['time'] = pd.to_datetime(df['time'])
            price_series = df.set_index('time')['price']
            
            self.logger.info(f"Price data loaded for {symbol}: {len(price_series)} points")
            
            return price_series
            
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
            return None
            
    async def calculate_squeeze_signals(self, symbol: str) -> List[Dict]:
        """
        Calculate squeeze signals for different timeframes
        """
        signals = []
        
        # Get data
        price_data = await self.get_price_data(symbol)
        spot_cvd, perp_cvd = await self.calculate_cvd_aggregates(symbol)
        
        if price_data is None or spot_cvd is None or perp_cvd is None:
            self.logger.warning(f"Insufficient data for {symbol}")
            return signals
            
        # DEBUG: Log the data being passed to calculation
        self.logger.info(f"🔍 DATA CHECK {symbol}: price_data type={type(price_data)}, len={len(price_data) if price_data is not None else 'None'}")
        self.logger.info(f"🔍 DATA CHECK {symbol}: spot_cvd type={type(spot_cvd)}, len={len(spot_cvd) if spot_cvd is not None else 'None'}")
        self.logger.info(f"🔍 DATA CHECK {symbol}: perp_cvd type={type(perp_cvd)}, len={len(perp_cvd) if perp_cvd is not None else 'None'}")
        
        if price_data is not None and len(price_data) > 0:
            self.logger.info(f"🔍 PRICE SAMPLE {symbol}: first={price_data.iloc[0]:.2f}, last={price_data.iloc[-1]:.2f}")
        if spot_cvd is not None and len(spot_cvd) > 0:
            self.logger.info(f"🔍 SPOT CVD SAMPLE {symbol}: first={spot_cvd.iloc[0]:.0f}, last={spot_cvd.iloc[-1]:.0f}")
        if perp_cvd is not None and len(perp_cvd) > 0:
            self.logger.info(f"🔍 PERP CVD SAMPLE {symbol}: first={perp_cvd.iloc[0]:.0f}, last={perp_cvd.iloc[-1]:.0f}")

        # Calculate signals for different lookback periods
        for lookback in self.lookback_periods:
            try:
                self.logger.info(f"🔍 About to calculate squeeze score for {symbol} lookback {lookback}")
                
                result = self.calculator.calculate_squeeze_score(
                    price_data=price_data,
                    spot_cvd_data=spot_cvd,
                    futures_cvd_data=perp_cvd,
                    lookback=lookback
                )
                
                # DEBUG: Log squeeze calculation details
                self.logger.info(f"🔍 SQUEEZE RESULT {symbol}_{lookback}: price_component={result.get('price_component', 0):.4f}, "
                               f"spot_trend={result.get('spot_cvd_trend', 0):.4f}, "
                               f"futures_trend={result.get('futures_cvd_trend', 0):.4f}, "
                               f"cvd_divergence={result.get('cvd_divergence', 0):.4f}, "
                               f"FINAL_SCORE={result.get('squeeze_score', 0):.4f}")
                
                signal = {
                    'symbol': symbol,
                    'lookback': lookback,
                    'timestamp': datetime.now(timezone.utc),
                    **result
                }
                
                signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"❌ Error calculating squeeze signal for {symbol} lookback {lookback}: {e}")
                import traceback
                self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
                
        return signals
        
    async def store_signals(self, signals: List[Dict]):
        """
        Store squeeze signals in InfluxDB (both databases for redundancy)
        """
        if not signals:
            return
            
        try:
            influx_points = []
            
            for signal in signals:
                point = {
                    'measurement': 'squeeze_signals',
                    'tags': {
                        'symbol': signal['symbol'],
                        'lookback': str(signal['lookback']),
                        'signal_type': signal['signal_type']
                    },
                    'time': signal['timestamp'],
                    'fields': {
                        'squeeze_score': float(signal['squeeze_score']),
                        'raw_score': float(signal['raw_score']),
                        'price_component': float(signal['price_component']),
                        'spot_cvd_trend': float(signal['spot_cvd_trend']),
                        'futures_cvd_trend': float(signal['futures_cvd_trend']),
                        'cvd_divergence': float(signal['cvd_divergence']),
                        'signal_strength': float(signal['signal_strength'])
                    }
                }
                influx_points.append(point)
                
            # Store in significant_trades database
            self.influx_client.write_points(influx_points, time_precision='s')
            
            # Also store in squeezeflow_market_data database for dashboard access
            try:
                from influxdb import InfluxDBClient
                squeezeflow_client = InfluxDBClient(
                    host=os.getenv('INFLUX_HOST', 'aggr-influx'),  # Same InfluxDB server, different database
                    port=int(os.getenv('INFLUX_PORT', 8086)),
                    database='squeezeflow_market_data'
                )
                squeezeflow_client.write_points(influx_points, time_precision='s')
            except Exception as db2_error:
                self.logger.warning(f"Failed to store in squeezeflow_market_data: {db2_error}")
                
            self.logger.info(f"Stored {len(influx_points)} squeeze signals in InfluxDB")
            
        except Exception as e:
            self.logger.error(f"Error storing signals: {e}")
            
    async def cache_signals(self, signals: List[Dict]):
        """
        Cache latest signals in Redis for fast access
        """
        try:
            for signal in signals:
                key = f"squeeze_signal:{signal['symbol']}:{signal['lookback']}"
                value = json.dumps({
                    'squeeze_score': signal['squeeze_score'],
                    'signal_type': signal['signal_type'],
                    'signal_strength': signal['signal_strength'],
                    'timestamp': signal['timestamp'].isoformat()
                })
                
                # Cache for 5 minutes
                self.redis_client.setex(key, 300, value)
                
        except Exception as e:
            self.logger.error(f"Error caching signals: {e}")
            
    async def calculation_cycle(self):
        """
        Single calculation cycle for all symbols
        """
        self.logger.info("Starting squeeze calculation cycle")
        start_time = time.time()
        
        all_signals = []
        
        for symbol in self.symbols:
            self.logger.info(f"Calculating squeeze signals for {symbol}")
            signals = await self.calculate_squeeze_signals(symbol)
            all_signals.extend(signals)
            
        # Store and cache signals
        if all_signals:
            await self.store_signals(all_signals)
            await self.cache_signals(all_signals)
            
        elapsed = time.time() - start_time
        self.logger.info(f"Calculation cycle completed in {elapsed:.2f}s, generated {len(all_signals)} signals")
        
    async def run(self):
        """
        Main service loop
        """
        self.logger.info("Starting SqueezeFlow Calculator Service")
        self.running = True
        
        try:
            while self.running:
                await self.calculation_cycle()
                
                # Wait for next calculation interval
                await asyncio.sleep(self.calculation_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.running = False
            self.logger.info("SqueezeFlow Calculator Service stopped")
            
    def stop(self):
        """Stop the service"""
        self.running = False


async def main():
    """Main function"""
    service = SqueezeFlowCalculatorService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())