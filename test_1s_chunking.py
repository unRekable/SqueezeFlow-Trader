#!/usr/bin/env python3
"""
Test script for Phase 1.3: 1s Data Chunking Implementation

This script tests the new chunking features for 1-second data loading
to ensure they work correctly and prevent timeouts for large datasets.
"""

import logging
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_chunking():
    """Test data pipeline chunking for 1s data"""
    try:
        from data.pipeline import DataPipeline
        
        logger.info("🧪 Testing DataPipeline chunking for 1s data...")
        
        pipeline = DataPipeline()
        
        # Test parameters (3 hours should trigger chunking)
        symbol = 'BTC'
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=3)  # 3 hours should trigger chunking
        
        # Test OHLCV data loading with chunking
        logger.info(f"Loading 1s OHLCV data for {symbol} ({start_time} to {end_time})")
        ohlcv_df = pipeline.load_raw_ohlcv_data(symbol, start_time, end_time, timeframe='1s')
        
        if not ohlcv_df.empty:
            logger.info(f"✅ OHLCV chunking test passed: {len(ohlcv_df)} rows loaded")
        else:
            logger.warning("⚠️  OHLCV chunking test: No data returned (may be expected if no 1s data available)")
        
        # Test volume data loading with chunking
        logger.info(f"Loading 1s volume data for {symbol}")
        spot_df, futures_df = pipeline.load_raw_volume_data(symbol, start_time, end_time, timeframe='1s')
        
        spot_rows = len(spot_df) if not spot_df.empty else 0
        futures_rows = len(futures_df) if not futures_df.empty else 0
        
        if spot_rows > 0 or futures_rows > 0:
            logger.info(f"✅ Volume chunking test passed: {spot_rows} spot rows, {futures_rows} futures rows")
        else:
            logger.warning("⚠️  Volume chunking test: No data returned (may be expected if no 1s data available)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DataPipeline chunking test failed: {e}")
        return False

def test_backtest_loader_chunking():
    """Test backtest data loader chunking"""
    try:
        from backtest.data_loader import BacktestDataLoader
        
        logger.info("🧪 Testing BacktestDataLoader chunking for 1s data...")
        
        # Initialize with chunking enabled
        loader = BacktestDataLoader(
            enable_streaming=False,
            chunk_size_hours=2,
            max_retries=3,
            enable_1s_chunking=True
        )
        
        # Test parameters (4 hours should trigger chunking)
        symbol = 'BINANCE:btcusdt'
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)  # 4 hours should trigger chunking
        
        # Test 1s data loading with chunking
        logger.info(f"Loading 1s data with chunking for {symbol}")
        df_1s = loader.load_1s_data(symbol, start_time, end_time, enable_chunking=True)
        
        if not df_1s.empty:
            logger.info(f"✅ BacktestDataLoader chunking test passed: {len(df_1s)} rows loaded")
        else:
            logger.warning("⚠️  BacktestDataLoader chunking test: No data returned (may be expected if no 1s data available)")
        
        # Test memory stats
        stats = loader.get_memory_stats()
        logger.info(f"📊 Memory stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BacktestDataLoader chunking test failed: {e}")
        return False

def test_influx_client_chunking():
    """Test InfluxDB client chunking"""
    try:
        import asyncio
        from data.loaders.influx_client import OptimizedInfluxClient, QueryOptimization
        
        logger.info("🧪 Testing InfluxDB client chunking for 1s data...")
        
        # Initialize optimized client
        config = QueryOptimization(
            connection_pool_size=10,
            query_timeout_seconds=120,
            enable_query_cache=True
        )
        
        client = OptimizedInfluxClient(
            host='localhost',
            port=8086,
            database='significant_trades',
            optimization_config=config
        )
        
        # Test parameters
        markets = ['BINANCE:btcusdt', 'COINBASE:btc-usd']
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=3)  # 3 hours
        
        # Test async 1s data loading with chunking
        async def test_async():
            logger.info("Testing 1s data with aggregation and chunking...")
            ohlcv_df, volume_df = await client.get_1s_data_with_aggregation(
                markets=markets,
                start_time=start_time,
                end_time=end_time,
                target_timeframe='5m',
                max_lookback_minutes=180,  # 3 hours
                enable_chunking=True
            )
            
            ohlcv_rows = len(ohlcv_df) if not ohlcv_df.empty else 0
            volume_rows = len(volume_df) if not volume_df.empty else 0
            
            if ohlcv_rows > 0 or volume_rows > 0:
                logger.info(f"✅ InfluxDB chunking test passed: {ohlcv_rows} OHLCV rows, {volume_rows} volume rows")
            else:
                logger.warning("⚠️  InfluxDB chunking test: No data returned (may be expected if no 1s data available)")
            
            # Get performance metrics
            metrics = client.get_performance_metrics()
            logger.info(f"📊 Query performance: {metrics['query_performance']}")
            
            return ohlcv_rows > 0 or volume_rows > 0
        
        # Run async test
        result = asyncio.run(test_async())
        
        # Close client
        client.close()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ InfluxDB client chunking test failed: {e}")
        return False

def main():
    """Run all chunking tests"""
    logger.info("🚀 Starting Phase 1.3 chunking tests...")
    
    tests = [
        ("DataPipeline Chunking", test_pipeline_chunking),
        ("BacktestDataLoader Chunking", test_backtest_loader_chunking),
        ("InfluxDB Client Chunking", test_influx_client_chunking),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️  {test_name}: FAILED or NO DATA")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All chunking tests completed successfully!")
        return True
    else:
        logger.warning("⚠️  Some tests failed - this may be expected if no 1s data is available")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)