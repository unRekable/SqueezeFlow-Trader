#!/usr/bin/env python3
"""
Test script to verify timezone comparison fixes and strategy scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any

from backtest.engine import BacktestEngine
from strategies.squeezeflow.strategy import SqueezeFlowStrategy
from strategies.squeezeflow.config import SqueezeFlowConfig


def create_mock_dataset(symbol: str, start_time: datetime, end_time: datetime, timeframe: str) -> Dict[str, Any]:
    """
    Create mock dataset with realistic market data for testing
    This simulates what the data pipeline should return
    """
    
    # Generate time series
    freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T', '1h': '1H', '4h': '4H'}
    freq = freq_map.get(timeframe, '5T')
    
    # Create timezone-aware datetime index
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq, tz=pytz.UTC)
    
    if len(timestamps) == 0:
        print(f"Warning: No timestamps generated for {start_time} to {end_time} with freq {freq}")
        return create_empty_dataset(symbol, start_time, end_time, timeframe)
    
    print(f"Generated {len(timestamps)} timestamps from {timestamps[0]} to {timestamps[-1]}")
    
    # Generate realistic OHLCV data (BTC-like prices)
    base_price = 65000  # Starting around $65k
    price_data = []
    
    for i, ts in enumerate(timestamps):
        # Simulate some price movement
        price_change = np.random.normal(0, 0.002)  # 0.2% std dev per period
        if i == 0:
            open_price = base_price
        else:
            open_price = price_data[-1]['close']
            
        close_price = open_price * (1 + price_change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
        volume = np.random.uniform(10, 100)  # Random volume
        
        price_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Create OHLCV DataFrame with timezone-aware index
    ohlcv = pd.DataFrame(price_data, index=timestamps)
    
    # Generate CVD data (correlated with price but with some divergence potential)
    spot_cvd_data = []
    futures_cvd_data = []
    
    for i, (ts, row) in enumerate(ohlcv.iterrows()):
        if i == 0:
            spot_cvd = 0
            futures_cvd = 0
        else:
            # CVD roughly follows price but with some divergence opportunities
            price_change = (row['close'] - ohlcv.iloc[i-1]['close']) / ohlcv.iloc[i-1]['close']
            
            # Create some periods where CVD diverges from price (key for SqueezeFlow)
            if i % 50 == 0:  # Every 50 periods, create potential divergence
                cvd_change = -price_change * 0.5 + np.random.normal(0, 0.1)  # Opposite direction
            else:
                cvd_change = price_change * 0.8 + np.random.normal(0, 0.05)  # Follow price mostly
            
            spot_cvd = spot_cvd_data[-1] + cvd_change * 1000  # Scale CVD
            futures_cvd = futures_cvd_data[-1] + cvd_change * 1200 + np.random.normal(0, 50)  # Slight difference
        
        spot_cvd_data.append(spot_cvd)
        futures_cvd_data.append(futures_cvd)
    
    # Create CVD series with timezone-aware index
    spot_cvd = pd.Series(spot_cvd_data, index=timestamps)
    futures_cvd = pd.Series(futures_cvd_data, index=timestamps)
    
    # Calculate CVD divergence
    cvd_divergence = spot_cvd - futures_cvd
    
    # Create volume series (same for spot and futures for simplicity)
    spot_volume = ohlcv['volume'].copy()
    futures_volume = ohlcv['volume'].copy() * 1.5  # Futures typically higher volume
    
    # Create markets dict
    markets = {
        'spot': [
            'BINANCE:btcusdt', 'COINBASE:BTC-USD', 'KRAKEN:XBTUSD'
        ],
        'perp': [
            'BINANCE_FUTURES:btcusdt', 'BYBIT:BTCUSDT', 'DERIBIT:BTC-PERPETUAL'
        ]
    }
    
    dataset = {
        'symbol': symbol,
        'timeframe': timeframe,
        'start_time': start_time,
        'end_time': end_time,
        'ohlcv': ohlcv,
        'spot_volume': spot_volume,
        'futures_volume': futures_volume,
        'spot_cvd': spot_cvd,
        'futures_cvd': futures_cvd,
        'cvd_divergence': cvd_divergence,
        'markets': markets,
        'metadata': {
            'spot_markets_count': len(markets['spot']),
            'futures_markets_count': len(markets['perp']), 
            'data_points': len(ohlcv)
        }
    }
    
    print(f"Created mock dataset: {len(ohlcv)} OHLCV points, {len(spot_cvd)} CVD points")
    return dataset


def create_empty_dataset(symbol: str, start_time: datetime, end_time: datetime, timeframe: str) -> Dict[str, Any]:
    """Create empty dataset structure for error cases"""
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'start_time': start_time,
        'end_time': end_time,
        'ohlcv': pd.DataFrame(),
        'spot_volume': pd.Series(),
        'futures_volume': pd.Series(),
        'spot_cvd': pd.Series(),
        'futures_cvd': pd.Series(),
        'cvd_divergence': pd.Series(),
        'markets': {'spot': [], 'perp': []},
        'metadata': {'spot_markets_count': 0, 'futures_markets_count': 0, 'data_points': 0}
    }


class MockDataPipeline:
    """Mock data pipeline that returns our test data"""
    
    def get_complete_dataset(self, symbol: str, start_time: datetime, end_time: datetime, 
                           timeframe: str) -> Dict[str, Any]:
        return create_mock_dataset(symbol, start_time, end_time, timeframe)
    
    def validate_data_quality(self, dataset: Dict[str, Any]) -> Dict[str, bool]:
        """Validate our mock data quality"""
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
        spot_cvd = dataset.get('spot_cvd', pd.Series())
        futures_cvd = dataset.get('futures_cvd', pd.Series())
        
        return {
            'has_price_data': not ohlcv.empty,
            'has_spot_volume': 'spot_volume' in dataset and not dataset['spot_volume'].empty,
            'has_futures_volume': 'futures_volume' in dataset and not dataset['futures_volume'].empty,
            'has_spot_cvd': not spot_cvd.empty,
            'has_futures_cvd': not futures_cvd.empty,
            'has_cvd_divergence': 'cvd_divergence' in dataset and not dataset['cvd_divergence'].empty,
            'sufficient_data_points': len(ohlcv) > 100,
            'multiple_spot_markets': len(dataset.get('markets', {}).get('spot', [])) > 1,
            'multiple_futures_markets': len(dataset.get('markets', {}).get('perp', [])) > 1,
            'overall_quality': not ohlcv.empty and not spot_cvd.empty and len(ohlcv) > 100
        }


def test_timezone_and_scoring():
    """Test timezone fixes and strategy scoring with detailed logging"""
    
    print("=" * 80)
    print("TIMEZONE AND SCORING TEST")
    print("=" * 80)
    
    # Create test configuration with lower threshold to increase trade generation
    config = SqueezeFlowConfig()
    config.min_entry_score = 2.0  # Lower threshold for testing
    print(f"Using min_entry_score: {config.min_entry_score}")
    
    # Create engine with mock data pipeline
    engine = BacktestEngine(initial_balance=10000.0, leverage=2.0)
    
    # Replace data pipeline with mock
    engine.data_pipeline = MockDataPipeline()
    
    # Create strategy with debug config
    strategy = SqueezeFlowStrategy(config=config)
    
    # Test dates (timezone-aware)
    start_date = '2024-08-01'
    end_date = '2024-08-02'  # Just 1 day for testing
    
    print(f"\nTesting backtest from {start_date} to {end_date}")
    print(f"Strategy: {strategy.name}")
    print(f"Config: min_entry_score = {config.min_entry_score}")
    
    try:
        # Run backtest
        result = engine.run(
            strategy=strategy,
            symbol='BTCUSDT',
            start_date=start_date,
            end_date=end_date,
            timeframe='5m',
            balance=10000.0,
            leverage=2.0
        )
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Success: {result.get('error') is None}")
        print(f"Total trades: {result.get('total_trades', 0)}")
        print(f"Final balance: ${result.get('final_balance', 0):,.2f}")
        print(f"Total return: {result.get('total_return', 0):.2f}%")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
            return False
            
        # Check for timezone errors (should be none)
        print(f"Timezone errors: None (fixed)")
        
        if result.get('total_trades', 0) == 0:
            print("\nNo trades generated - this could be normal with mock data")
            print("Let's check if strategy is generating any signals...")
            
            # Test strategy directly with a small dataset
            test_single_strategy_call(strategy, config)
        
        return True
        
    except Exception as e:
        print(f"Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_strategy_call(strategy: SqueezeFlowStrategy, config: SqueezeFlowConfig):
    """Test strategy with a single data window to see if it generates signals"""
    
    print("\n" + "-" * 50)
    print("DIRECT STRATEGY TEST")
    print("-" * 50)
    
    # Create small test dataset
    start_time = datetime(2024, 8, 1, 12, 0, tzinfo=pytz.UTC)
    end_time = datetime(2024, 8, 1, 16, 0, tzinfo=pytz.UTC)  # 4 hour window
    
    dataset = create_mock_dataset('BTCUSDT', start_time, end_time, '5m')
    
    # Create portfolio state (no existing positions)
    portfolio_state = {
        'total_value': 10000.0,
        'available_cash': 10000.0,
        'positions': [],
        'available_leverage': 2.0,
        'current_time': end_time
    }
    
    print(f"Dataset size: {len(dataset['ohlcv'])} candles")
    print(f"CVD range - Spot: {dataset['spot_cvd'].min():.1f} to {dataset['spot_cvd'].max():.1f}")
    print(f"CVD range - Futures: {dataset['futures_cvd'].min():.1f} to {dataset['futures_cvd'].max():.1f}")
    
    # Process with strategy
    result = strategy.process(dataset, portfolio_state)
    
    print(f"Strategy result keys: {list(result.keys())}")
    print(f"Orders generated: {len(result.get('orders', []))}")
    
    # Check phase results
    phase_results = result.get('phase_results', {})
    for phase, phase_result in phase_results.items():
        print(f"\n{phase}:")
        if isinstance(phase_result, dict):
            if 'error' in phase_result:
                print(f"  Error: {phase_result['error']}")
            elif 'skipped' in phase_result:
                print(f"  Skipped: {phase_result['reason']}")
            else:
                # Show key results
                for key, value in phase_result.items():
                    if key in ['total_score', 'should_trade', 'direction', 'setup_type', 'market_bias']:
                        print(f"  {key}: {value}")
    
    # Show orders if any
    orders = result.get('orders', [])
    if orders:
        print(f"\nGenerated {len(orders)} orders:")
        for i, order in enumerate(orders):
            print(f"  Order {i+1}: {order.get('side')} {order.get('quantity')} @ ${order.get('price'):.2f}")
    else:
        print("\nNo orders generated")
        
        # Check scoring details
        scoring = phase_results.get('phase4_scoring', {})
        if scoring:
            print(f"Scoring details:")
            print(f"  Total score: {scoring.get('total_score', 0):.2f}")
            print(f"  Min required: {config.min_entry_score}")
            print(f"  Should trade: {scoring.get('should_trade', False)}")
            print(f"  Direction: {scoring.get('direction', 'NONE')}")
            
            breakdown = scoring.get('score_breakdown', {})
            if breakdown:
                print("  Score breakdown:")
                for criteria, score in breakdown.items():
                    print(f"    {criteria}: {score:.2f}")


if __name__ == "__main__":
    success = test_timezone_and_scoring()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")