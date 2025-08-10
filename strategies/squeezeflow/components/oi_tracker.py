"""
Open Interest Tracker - Simple, Elegant, Direct
Restores critical OI validation for squeeze detection
"""

import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class OITracker:
    """
    Simple, direct OI tracking without over-abstraction
    Focuses on Binance and Bybit (cover 60%+ of market volume)
    """
    
    def __init__(self, rise_threshold: float = 5.0):
        """
        Initialize OI tracker
        
        Args:
            rise_threshold: Minimum OI increase % for squeeze signal (default 5%)
        """
        self.rise_threshold = rise_threshold
        self.cache = {}  # Simple in-memory cache
        self.last_update = {}
        
        # API endpoints (public, no auth needed)
        self.endpoints = {
            'binance': 'https://fapi.binance.com/fapi/v1/openInterest',
            'bybit': 'https://api.bybit.com/v5/market/open-interest'
        }
        
        logger.info(f"OI Tracker initialized with {rise_threshold}% threshold")
    
    async def fetch_binance_oi(self, symbol: str) -> Optional[float]:
        """Fetch OI from Binance Futures"""
        try:
            async with aiohttp.ClientSession() as session:
                # Binance uses format like BTCUSDT
                ticker = f"{symbol}USDT" if not symbol.endswith('USDT') else symbol
                
                async with session.get(
                    self.endpoints['binance'],
                    params={'symbol': ticker},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        oi_value = float(data.get('openInterest', 0))
                        logger.debug(f"Binance OI for {ticker}: {oi_value}")
                        return oi_value
        except Exception as e:
            logger.warning(f"Binance OI fetch failed for {symbol}: {e}")
        return None
    
    async def fetch_bybit_oi(self, symbol: str) -> Optional[float]:
        """Fetch OI from Bybit"""
        try:
            async with aiohttp.ClientSession() as session:
                # Bybit uses format like BTCUSDT
                ticker = f"{symbol}USDT" if not symbol.endswith('USDT') else symbol
                
                async with session.get(
                    self.endpoints['bybit'],
                    params={
                        'category': 'linear',
                        'symbol': ticker
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('result', {}).get('list'):
                            oi_value = float(data['result']['list'][0].get('openInterest', 0))
                            logger.debug(f"Bybit OI for {ticker}: {oi_value}")
                            return oi_value
        except Exception as e:
            logger.warning(f"Bybit OI fetch failed for {symbol}: {e}")
        return None
    
    async def get_aggregated_oi(self, symbol: str) -> float:
        """
        Get aggregated OI from multiple exchanges
        Simple average if both available, single value if only one
        """
        # Fetch from both exchanges in parallel
        binance_oi, bybit_oi = await asyncio.gather(
            self.fetch_binance_oi(symbol),
            self.fetch_bybit_oi(symbol),
            return_exceptions=True
        )
        
        # Handle results
        oi_values = []
        if isinstance(binance_oi, (int, float)) and binance_oi:
            oi_values.append(binance_oi)
        if isinstance(bybit_oi, (int, float)) and bybit_oi:
            oi_values.append(bybit_oi)
        
        if oi_values:
            # Simple average of available values
            aggregated = sum(oi_values) / len(oi_values)
            logger.debug(f"Aggregated OI for {symbol}: {aggregated} from {len(oi_values)} exchanges")
            return aggregated
        
        return 0.0
    
    def calculate_oi_change(self, symbol: str, current_oi: float) -> Dict:
        """
        Calculate OI change percentage
        
        Returns:
            Dict with value, change_pct, and rising flag
        """
        cache_key = symbol.upper()
        
        # Initialize cache for new symbol
        if cache_key not in self.cache:
            self.cache[cache_key] = current_oi
            self.last_update[cache_key] = datetime.now()
            return {
                'value': current_oi,
                'change_pct': 0,
                'rising': False,
                'status': 'initialized'
            }
        
        # Calculate change
        previous_oi = self.cache[cache_key]
        
        if previous_oi > 0:
            change_pct = ((current_oi - previous_oi) / previous_oi) * 100
        else:
            change_pct = 0
        
        # Update cache
        self.cache[cache_key] = current_oi
        self.last_update[cache_key] = datetime.now()
        
        # Determine if OI is rising significantly
        rising = change_pct >= self.rise_threshold
        
        return {
            'value': current_oi,
            'previous': previous_oi,
            'change_pct': round(change_pct, 2),
            'rising': rising,
            'threshold': self.rise_threshold,
            'status': 'squeeze_confirmed' if rising else 'insufficient_oi_rise'
        }
    
    def get_oi_change_sync(self, symbol: str) -> Dict:
        """
        Synchronous wrapper for getting OI change
        Main entry point for strategy integration
        """
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get current OI
            current_oi = loop.run_until_complete(self.get_aggregated_oi(symbol))
            
            # Calculate change
            result = self.calculate_oi_change(symbol, current_oi)
            
            logger.info(f"OI for {symbol}: {result['value']:.2f}, "
                       f"Change: {result['change_pct']}%, "
                       f"Rising: {result['rising']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting OI for {symbol}: {e}")
            return {
                'value': 0,
                'change_pct': 0,
                'rising': False,
                'status': 'error',
                'error': str(e)
            }
        finally:
            loop.close()
    
    def validate_squeeze_signal(self, symbol: str, divergence_detected: bool) -> Tuple[bool, Dict]:
        """
        Validate squeeze signal with OI confirmation
        This is the KEY function for squeeze detection
        
        Args:
            symbol: Trading symbol
            divergence_detected: Whether price/CVD divergence was detected
            
        Returns:
            Tuple of (is_valid_squeeze, oi_data)
        """
        if not divergence_detected:
            return False, {'status': 'no_divergence'}
        
        # Get OI data
        oi_data = self.get_oi_change_sync(symbol)
        
        # Squeeze is only valid if OI is rising
        is_valid = divergence_detected and oi_data['rising']
        
        if is_valid:
            logger.info(f"✅ SQUEEZE CONFIRMED for {symbol}: "
                       f"Divergence + OI rising {oi_data['change_pct']}%")
        else:
            logger.info(f"❌ Squeeze NOT confirmed for {symbol}: "
                       f"OI change {oi_data['change_pct']}% < {self.rise_threshold}% threshold")
        
        return is_valid, oi_data


# Global instance for easy import
oi_tracker = OITracker()


# Test function
if __name__ == "__main__":
    import sys
    
    # Test with command line argument
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    
    print(f"Testing OI tracker for {symbol}...")
    tracker = OITracker()
    
    # Test 1: Get OI change
    result = tracker.get_oi_change_sync(symbol)
    print(f"OI Data: {json.dumps(result, indent=2)}")
    
    # Test 2: Validate squeeze signal
    is_valid, oi_data = tracker.validate_squeeze_signal(symbol, divergence_detected=True)
    print(f"Squeeze Valid: {is_valid}")
    print(f"OI Status: {oi_data.get('status', 'unknown')}")