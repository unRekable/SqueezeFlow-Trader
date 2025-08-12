#!/usr/bin/env python3
"""Quick test to see what's hanging in the data load"""

import os
import sys
from datetime import datetime
import pytz
import time

os.environ['INFLUX_HOST'] = '213.136.75.120'
os.environ['INFLUX_PORT'] = '8086'

from data.pipeline import DataPipeline

def test_data_load():
    print("Testing data load...")
    
    pipeline = DataPipeline()
    
    # Test with BTC for single day
    symbol = 'BTC'
    start_time = datetime(2025, 8, 11, 0, 0, tzinfo=pytz.UTC)
    end_time = datetime(2025, 8, 11, 23, 59, 59, tzinfo=pytz.UTC)
    
    print(f"Loading data for {symbol} from {start_time} to {end_time}")
    print("Requesting 1s timeframe (the only data we have)...")
    
    start = time.time()
    
    try:
        # This is what the backtest engine does
        dataset = pipeline.get_complete_dataset(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='1s'  # Always use 1s as we only have 1s data
        )
        
        elapsed = time.time() - start
        print(f"✅ Data loaded in {elapsed:.2f} seconds")
        
        if dataset and 'ohlcv' in dataset:
            print(f"  OHLCV shape: {dataset['ohlcv'].shape}")
            print(f"  Data points: {dataset['metadata']['data_points']}")
            print(f"  Markets: {dataset['metadata']['spot_markets_count']} SPOT, {dataset['metadata']['futures_markets_count']} FUTURES")
        else:
            print("❌ No data returned")
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Failed after {elapsed:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_data_load()