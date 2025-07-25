"""
Open Interest Indicator for SqueezeFlow Trader
CRITICAL for squeeze detection - OI must rise for trapped shorts signal
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)


class OpenInterestIndicator:
    """
    Open Interest tracking and analysis
    OI MUST RISE for long squeeze signals (indicates trapped shorts)
    """
    
    def __init__(self, 
                 lookback_period: int = 10,
                 rise_threshold: float = 0.05,
                 exchanges: Optional[List[str]] = None):
        """
        Initialize Open Interest Indicator
        
        Args:
            lookback_period: Period for OI change calculation
            rise_threshold: Minimum OI increase for signal (5% default)
            exchanges: List of exchanges to track
        """
        self.lookback_period = lookback_period
        self.rise_threshold = rise_threshold
        self.exchanges = exchanges or ['binance', 'bybit', 'okx', 'kraken', 'bitfinex', 'gateio', 'huobi', 'kucoin']
        self.oi_history = {}
        self.exchange_clients = {}
        
        logger.info(f"OpenInterestIndicator initialized with threshold={rise_threshold}, period={lookback_period}")
    
    async def initialize_exchanges(self, api_credentials: Dict[str, Dict[str, str]]):
        """
        Initialize async exchange clients
        
        Args:
            api_credentials: Dict with exchange credentials
        """
        for exchange_name in self.exchanges:
            try:
                if exchange_name in api_credentials:
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchange_clients[exchange_name] = exchange_class({
                        'apiKey': api_credentials[exchange_name].get('api_key'),
                        'secret': api_credentials[exchange_name].get('api_secret'),
                        'password': api_credentials[exchange_name].get('passphrase'),
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                    logger.info(f"Initialized {exchange_name} client")
            except Exception as e:
                logger.error(f"Error initializing {exchange_name}: {str(e)}")
    
    async def fetch_oi_data(self, exchange: str, symbol: str) -> Optional[float]:
        """
        Fetch current open interest from exchange
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            
        Returns:
            Current open interest value or None
        """
        try:
            if exchange not in self.exchange_clients:
                logger.warning(f"Exchange {exchange} not initialized")
                return None
            
            client = self.exchange_clients[exchange]
            
            # Different exchanges have different methods
            if exchange == 'binance':
                # Binance specific endpoint
                markets = await client.load_markets()
                if symbol in markets:
                    response = await client.fapiPublicGetOpenInterest({'symbol': symbol.replace('/', '')})
                    return float(response['openInterest'])
                    
            elif exchange == 'bybit':
                # Bybit specific endpoint
                response = await client.public_get_v5_market_open_interest({
                    'category': 'linear',
                    'symbol': symbol.replace('/', '')
                })
                if response['result']['list']:
                    return float(response['result']['list'][0]['openInterest'])
                    
            elif exchange == 'okx':
                # OKX specific endpoint
                response = await client.public_get_api_v5_public_open_interest({
                    'instType': 'SWAP',
                    'instId': symbol.replace('/', '-') + '-SWAP'
                })
                if response['data']:
                    return float(response['data'][0]['oi'])
                    
            else:
                # Generic method for other exchanges
                ticker = await client.fetch_ticker(symbol)
                if 'openInterest' in ticker and ticker['openInterest']:
                    return float(ticker['openInterest'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching OI from {exchange} for {symbol}: {str(e)}")
            return None
    
    def calculate_oi_change_rate(self, oi_series: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate percentage change in open interest
        
        Args:
            oi_series: Series of OI values
            period: Optional period override
            
        Returns:
            Series with OI change rates
        """
        try:
            period = period or self.lookback_period
            
            # Calculate percentage change
            oi_change = oi_series.pct_change(period)
            
            # Fill NaN values
            oi_change = oi_change.fillna(0)
            
            logger.debug(f"OI change rate calculated: mean={oi_change.mean():.2%}, current={oi_change.iloc[-1]:.2%}")
            return oi_change
            
        except Exception as e:
            logger.error(f"Error calculating OI change rate: {str(e)}")
            raise
    
    async def aggregate_multi_exchange_oi(self, 
                                        exchange_oi_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Aggregate OI from multiple exchanges
        
        Args:
            exchange_oi_data: Dict with exchange names as keys and OI series as values
            
        Returns:
            DataFrame with aggregated OI data
        """
        try:
            # Create DataFrame from exchange data
            oi_df = pd.DataFrame(exchange_oi_data)
            
            # Calculate total OI
            oi_df['total_oi'] = oi_df.sum(axis=1)
            
            # Calculate average OI
            oi_df['avg_oi'] = oi_df.mean(axis=1)
            
            # Calculate OI distribution
            for exchange in exchange_oi_data.keys():
                if exchange in oi_df.columns:
                    oi_df[f'{exchange}_pct'] = oi_df[exchange] / oi_df['total_oi']
            
            # Add change rates
            oi_df['total_oi_change'] = self.calculate_oi_change_rate(oi_df['total_oi'])
            
            # Count active exchanges
            oi_df['active_exchanges'] = oi_df[list(exchange_oi_data.keys())].notna().sum(axis=1)
            
            logger.info(f"Aggregated OI from {len(exchange_oi_data)} exchanges")
            return oi_df
            
        except Exception as e:
            logger.error(f"Error aggregating multi-exchange OI: {str(e)}")
            raise
    
    def detect_oi_increase(self, 
                         oi_series: pd.Series, 
                         threshold: Optional[float] = None) -> pd.Series:
        """
        Detect when OI increases above threshold
        
        Args:
            oi_series: OI data series
            threshold: Optional threshold override
            
        Returns:
            Boolean series indicating OI increase signals
        """
        try:
            threshold = threshold or self.rise_threshold
            
            # Calculate OI change
            oi_change = self.calculate_oi_change_rate(oi_series)
            
            # Detect increases above threshold
            oi_rising = oi_change > threshold
            
            # Add confirmation: OI must be rising consistently
            oi_momentum = oi_change.rolling(window=3).mean()
            oi_confirmed = (oi_rising) & (oi_momentum > 0)
            
            signal_count = oi_confirmed.sum()
            if signal_count > 0:
                logger.info(f"OI increase detected: {signal_count} signals above {threshold:.1%}")
            
            return oi_confirmed
            
        except Exception as e:
            logger.error(f"Error detecting OI increase: {str(e)}")
            raise
    
    def track_oi_history(self, symbol: str, oi_value: float, timestamp: Optional[datetime] = None):
        """
        Track historical OI data
        
        Args:
            symbol: Trading symbol
            oi_value: Current OI value
            timestamp: Optional timestamp (uses now if None)
        """
        try:
            timestamp = timestamp or datetime.now()
            
            if symbol not in self.oi_history:
                self.oi_history[symbol] = []
            
            # Add new data point
            self.oi_history[symbol].append({
                'timestamp': timestamp,
                'oi': oi_value,
                'change': 0  # Will be calculated later
            })
            
            # Calculate change if we have history
            if len(self.oi_history[symbol]) > 1:
                prev_oi = self.oi_history[symbol][-2]['oi']
                if prev_oi > 0:
                    change = (oi_value - prev_oi) / prev_oi
                    self.oi_history[symbol][-1]['change'] = change
            
            # Limit history size (keep last 1000 points)
            if len(self.oi_history[symbol]) > 1000:
                self.oi_history[symbol] = self.oi_history[symbol][-1000:]
            
            logger.debug(f"Tracked OI for {symbol}: {oi_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error tracking OI history: {str(e)}")
    
    def get_oi_divergence(self, 
                         price_series: pd.Series, 
                         oi_series: pd.Series) -> pd.DataFrame:
        """
        Analyze divergence between price and OI
        
        Args:
            price_series: Price data
            oi_series: OI data
            
        Returns:
            DataFrame with divergence analysis
        """
        try:
            # Ensure series are aligned
            price_series = price_series.reindex(oi_series.index)
            
            # Calculate changes
            price_change = price_series.pct_change(self.lookback_period)
            oi_change = self.calculate_oi_change_rate(oi_series)
            
            # Detect divergences
            results = pd.DataFrame({
                'price_change': price_change,
                'oi_change': oi_change,
                'price_up_oi_up': (price_change > 0) & (oi_change > 0),  # Bullish
                'price_up_oi_down': (price_change > 0) & (oi_change < 0),  # Bearish divergence
                'price_down_oi_up': (price_change < 0) & (oi_change > 0),  # Squeeze setup
                'price_down_oi_down': (price_change < 0) & (oi_change < 0),  # Capitulation
            })
            
            # Add divergence strength
            results['divergence_strength'] = abs(price_change - oi_change)
            
            # Key signal: Price up + OI up = trapped shorts
            results['squeeze_setup'] = (
                (price_change > 0.03) &  # Price rising 3%+
                (oi_change > self.rise_threshold)  # OI rising 5%+
            )
            
            logger.debug(f"OI divergence analysis: {results['squeeze_setup'].sum()} squeeze setups")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing OI divergence: {str(e)}")
            raise
    
    def get_oi_stats(self, oi_series: pd.Series) -> Dict[str, float]:
        """
        Calculate OI statistics
        
        Returns:
            Dict with OI statistics
        """
        try:
            if oi_series.empty:
                return {}
            
            oi_change = self.calculate_oi_change_rate(oi_series)
            
            stats = {
                'current_oi': float(oi_series.iloc[-1]) if len(oi_series) > 0 else 0,
                'oi_change_pct': float(oi_change.iloc[-1] * 100) if len(oi_change) > 0 else 0,
                'oi_mean': float(oi_series.mean()),
                'oi_std': float(oi_series.std()),
                'oi_trend': 'increasing' if oi_change.iloc[-1] > 0 else 'decreasing',
                'above_threshold': bool(oi_change.iloc[-1] > self.rise_threshold) if len(oi_change) > 0 else False,
                'max_oi': float(oi_series.max()),
                'min_oi': float(oi_series.min()),
                'oi_volatility': float(oi_change.std() * 100) if len(oi_change) > 1 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating OI stats: {str(e)}")
            return {}
    
    async def get_real_time_oi(self, symbol: str) -> Dict[str, float]:
        """
        Get real-time OI from all available exchanges
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with exchange OI values
        """
        try:
            tasks = []
            exchange_names = []
            
            # Create tasks for all exchanges
            for exchange in self.exchange_clients.keys():
                tasks.append(self.fetch_oi_data(exchange, symbol))
                exchange_names.append(exchange)
            
            # Fetch all OI data concurrently
            oi_values = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            oi_data = {}
            for exchange, oi_value in zip(exchange_names, oi_values):
                if isinstance(oi_value, Exception):
                    logger.warning(f"Error fetching OI from {exchange}: {str(oi_value)}")
                elif oi_value is not None:
                    oi_data[exchange] = oi_value
                    self.track_oi_history(f"{exchange}:{symbol}", oi_value)
            
            # Calculate total OI
            if oi_data:
                oi_data['total'] = sum(oi_data.values())
                oi_data['average'] = oi_data['total'] / len(oi_data)
            
            logger.info(f"Real-time OI for {symbol}: {len(oi_data)} exchanges, total={oi_data.get('total', 0):.2f}")
            return oi_data
            
        except Exception as e:
            logger.error(f"Error getting real-time OI: {str(e)}")
            return {}
    
    def validate_squeeze_conditions(self, 
                                  price_change: float,
                                  cvd_change: float,
                                  oi_change: float) -> Dict[str, bool]:
        """
        Validate all squeeze conditions including OI
        
        Args:
            price_change: Price change percentage
            cvd_change: CVD change percentage
            oi_change: OI change percentage
            
        Returns:
            Dict with condition validation results
        """
        conditions = {
            'price_rising': price_change > 0.03,  # 3%
            'cvd_falling': cvd_change < -0.2,  # -20%
            'oi_rising': oi_change > self.rise_threshold,  # 5%
            'all_conditions_met': False
        }
        
        conditions['all_conditions_met'] = all([
            conditions['price_rising'],
            conditions['cvd_falling'],
            conditions['oi_rising']
        ])
        
        if conditions['all_conditions_met']:
            logger.info("SQUEEZE SIGNAL: All conditions met - Price UP, CVD DOWN, OI UP")
        
        return conditions