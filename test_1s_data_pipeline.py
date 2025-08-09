#!/usr/bin/env python3
"""
Test 1-Second Data Pipeline for SqueezeFlow Trader
Comprehensive test of 1s data collection and loading
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.append('/app')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loaders.influx_client import OptimizedInfluxClient, QueryOptimization


async def test_1s_data_pipeline():
    """Test 1-second data pipeline end-to-end"""
    print("🧪 Testing 1-Second Data Pipeline")
    print("=" * 60)
    
    # Initialize client
    config = QueryOptimization(
        connection_pool_size=5,
        query_timeout_seconds=30,
        enable_query_cache=False  # Disable cache for testing
    )
    
    client = OptimizedInfluxClient(
        host='aggr-influx',
        port=8086,
        database='significant_trades',
        optimization_config=config
    )
    
    try:
        # Test 1: Check 1s data availability
        print("\n📊 Test 1: 1s Data Availability Check")
        availability = await client._check_1s_data_availability([], datetime.now(), datetime.now())
        print(f"   Has recent data: {availability.get('has_recent_data', False)}")
        print(f"   Measurement exists: {availability.get('measurement_exists', False)}")
        
        if availability.get('error'):
            print(f"   Error: {availability['error']}")
        
        # Test 2: Query recent 1s data structure
        print("\n📊 Test 2: 1s Data Structure")
        test_query = """
            SELECT * FROM "trades_1s" 
            WHERE time > now() - 30m 
            LIMIT 5
        """
        result = await client.execute_query_async(test_query, 'significant_trades', enable_cache=False)
        print(f"   Sample rows: {len(result)}")
        if not result.empty:
            print(f"   Columns: {list(result.columns)}")
            print(f"   Latest timestamp: {result.index[-1] if not result.empty else 'None'}")
        
        # Test 3: Test 1s data with aggregation
        print("\n📊 Test 3: 1s Data Aggregation Test")
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)  # Last 5 minutes
        
        markets = ['BTC-USD', 'ETH-USD', 'BNB-USD']
        
        try:
            ohlcv_df, volume_df = await client.get_1s_data_with_aggregation(
                markets=markets,
                start_time=start_time,
                end_time=end_time,
                target_timeframe='1m',  # Aggregate to 1-minute
                max_lookback_minutes=10,
                enable_chunking=False
            )
            
            print(f"   OHLCV bars (1m aggregated): {len(ohlcv_df)}")
            print(f"   Volume bars (1m aggregated): {len(volume_df)}")
            
            if not ohlcv_df.empty:
                print(f"   OHLCV columns: {list(ohlcv_df.columns)}")
                print(f"   OHLCV sample: {ohlcv_df.head(1).to_dict('records')}")
            
            if not volume_df.empty:
                print(f"   Volume columns: {list(volume_df.columns)}")
        
        except Exception as e:
            print(f"   Error in aggregation test: {e}")
        
        # Test 4: CVD calculation from 1s data
        print("\n📊 Test 4: CVD Calculation Test")
        try:
            cvd_query = """
                SELECT 
                    sum(vbuy) - sum(vsell) as cvd_delta,
                    sum(vbuy) as total_buy_volume,
                    sum(vsell) as total_sell_volume,
                    COUNT(*) as data_points
                FROM "trades_1s" 
                WHERE time > now() - 30m
                GROUP BY time(5m)
                ORDER BY time DESC
                LIMIT 6
            """
            
            cvd_result = await client.execute_query_async(cvd_query, 'significant_trades', enable_cache=False)
            print(f"   CVD calculation rows: {len(cvd_result)}")
            
            if not cvd_result.empty:
                print(f"   CVD columns: {list(cvd_result.columns)}")
                if len(cvd_result) > 0:
                    latest = cvd_result.iloc[0]
                    print(f"   Latest CVD delta: {latest.get('cvd_delta', 'N/A')}")
                    print(f"   Data points per 5m: {latest.get('data_points', 'N/A')}")
        
        except Exception as e:
            print(f"   Error in CVD calculation: {e}")
        
        # Test 5: Performance metrics
        print("\n📊 Test 5: Performance Metrics")
        metrics = client.get_performance_metrics()
        print(f"   Total queries: {metrics['query_performance']['total_queries']}")
        print(f"   Avg query time: {metrics['query_performance']['avg_query_time_ms']}ms")
        print(f"   Failed queries: {metrics['query_performance']['failed_queries']}")
        
        # Test 6: Continuous queries check
        print("\n📊 Test 6: Continuous Queries Check")
        cq_query = "SHOW CONTINUOUS QUERIES"
        cq_result = await client.execute_query_async(cq_query, 'significant_trades', enable_cache=False)
        print(f"   Continuous queries found: {len(cq_result)}")
        
        # Test 7: Retention policies check
        print("\n📊 Test 7: Retention Policies Check")
        rp_query = "SHOW RETENTION POLICIES"
        rp_result = await client.execute_query_async(rp_query, 'significant_trades', enable_cache=False)
        print(f"   Retention policies: {len(rp_result)}")
        if not rp_result.empty:
            for _, policy in rp_result.iterrows():
                if 'name' in policy:
                    print(f"     - {policy['name']}: {policy.get('duration', 'N/A')}")
        
        print("\n✅ 1s Data Pipeline Test Completed")
        return True
        
    except Exception as e:
        print(f"\n❌ 1s Data Pipeline Test Failed: {e}")
        return False
    
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(test_1s_data_pipeline())