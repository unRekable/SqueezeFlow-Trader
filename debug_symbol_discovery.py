#!/usr/bin/env python3
"""
Debug Symbol Discovery - Find out why symbols aren't being discovered
"""

import os
import sys
sys.path.append('/Users/u/PycharmProjects/SqueezeFlow Trader')

from data.loaders.symbol_discovery import SymbolDiscovery
from data.loaders.market_discovery import MarketDiscovery

def debug_symbol_discovery():
    print("🔍 Debugging Symbol Discovery...")
    
    # Initialize discovery
    symbol_discovery = SymbolDiscovery()
    market_discovery = MarketDiscovery()
    
    # Test with lower thresholds
    print("\n1. Testing with 500 data points threshold...")
    symbols_500 = symbol_discovery.discover_symbols_from_database(
        min_data_points=500,
        hours_lookback=24
    )
    print(f"Found with 500 threshold: {symbols_500}")
    
    print("\n2. Testing with 100 data points threshold...")
    symbols_100 = symbol_discovery.discover_symbols_from_database(
        min_data_points=100,
        hours_lookback=24
    )
    print(f"Found with 100 threshold: {symbols_100}")
    
    # Check market classification for each symbol
    print("\n3. Checking futures markets for each symbol...")
    for symbol in symbols_100:
        markets_by_type = market_discovery.get_markets_by_type(symbol)
        spot_count = len(markets_by_type.get('spot', []))
        perp_count = len(markets_by_type.get('perp', []))
        print(f"   {symbol}: {spot_count} SPOT, {perp_count} PERP markets")
        
        if perp_count > 0:
            print(f"      PERP markets: {markets_by_type['perp'][:3]}...")

if __name__ == "__main__":
    debug_symbol_discovery()