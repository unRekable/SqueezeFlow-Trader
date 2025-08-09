#!/usr/bin/env python3
"""
Simple 1-Second Data Test for SqueezeFlow Trader  
Quick validation of 1s data implementation
"""

import asyncio
import pandas as pd
import redis
import time
from datetime import datetime, timedelta
import sys
import os

# Direct imports without complex config
from influxdb import InfluxDBClient


async def test_1s_implementation():
    """Simple test of 1s data implementation"""
    print("🧪 Simple 1-Second Data Implementation Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: InfluxDB Connection and 1s Data
    print("\n📊 Test 1: InfluxDB 1s Data Check")
    try:
        client = InfluxDBClient(host='aggr-influx', port=8086, database='significant_trades')
        
        # Check measurements
        measurements = client.query("SHOW MEASUREMENTS")
        measurement_list = [point['name'] for point in measurements.get_points()]
        has_trades_1s = 'trades_1s' in measurement_list
        
        print(f"   Available measurements: {measurement_list}")
        print(f"   trades_1s exists: {has_trades_1s}")
        results['trades_1s_exists'] = has_trades_1s
        
        # Check recent data count
        if has_trades_1s:
            recent_query = "SELECT COUNT(*) FROM trades_1s WHERE time > now() - 1h"
            recent_result = client.query(recent_query)
            count = list(recent_result.get_points())[0]['count'] if recent_result else 0
            print(f"   Recent 1s data points (1h): {count}")
            results['recent_1s_count'] = count
        else:
            results['recent_1s_count'] = 0
        
        # Check field structure
        if has_trades_1s:
            fields_query = "SHOW FIELD KEYS FROM trades_1s"
            fields_result = client.query(fields_query)
            fields = [point['fieldKey'] for point in fields_result.get_points()]
            print(f"   1s data fields: {fields}")
            results['field_structure'] = fields
        
        # Check continuous queries
        cq_query = "SHOW CONTINUOUS QUERIES"
        cq_result = client.query(cq_query)
        cq_list = []
        for series in cq_result:
            for point in series:
                if 'name' in point and point['name']:
                    cq_list.append(point['name'])
        
        print(f"   Continuous queries: {cq_list}")
        results['continuous_queries'] = cq_list
        
        client.close()
        
    except Exception as e:
        print(f"   ❌ InfluxDB test failed: {e}")
        results['influx_error'] = str(e)
    
    # Test 2: Redis Connection
    print("\n📊 Test 2: Redis Signal Storage")
    try:
        redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("   Redis connection: ✅ Working")
        
        # Test signal storage
        import json
        test_signal = {
            'symbol': 'BTC',
            'timestamp': datetime.now().isoformat(),
            'test': True
        }
        
        redis_client.lpush('test_1s_signals', json.dumps(test_signal))
        stored = redis_client.rpop('test_1s_signals')
        print("   Signal storage test: ✅ Working")
        results['redis_working'] = True
        
        redis_client.close()
        
    except Exception as e:
        print(f"   ❌ Redis test failed: {e}")
        results['redis_working'] = False
    
    # Test 3: Environment Variables Check
    print("\n📊 Test 3: Environment Variables")
    env_vars = {
        'SQUEEZEFLOW_ENABLE_1S_MODE': os.getenv('SQUEEZEFLOW_ENABLE_1S_MODE'),
        'SQUEEZEFLOW_DATA_INTERVAL': os.getenv('SQUEEZEFLOW_DATA_INTERVAL'),
        'SQUEEZEFLOW_RUN_INTERVAL': os.getenv('SQUEEZEFLOW_RUN_INTERVAL'),
    }
    
    for var, value in env_vars.items():
        print(f"   {var}: {value}")
    
    results['env_vars'] = env_vars
    
    # Test 4: aggr-server Status
    print("\n📊 Test 4: aggr-server Data Collection")
    # Check if aggr-server is writing fresh data
    try:
        client = InfluxDBClient(host='aggr-influx', port=8086, database='significant_trades')
        
        # Check for very recent data (last 5 minutes)
        very_recent_query = "SELECT * FROM trades_1s WHERE time > now() - 5m ORDER BY time DESC LIMIT 1"
        very_recent = client.query(very_recent_query)
        
        if very_recent:
            latest_point = list(very_recent.get_points())[0]
            latest_time = latest_point['time']
            print(f"   Latest 1s data: {latest_time}")
            results['latest_1s_data'] = latest_time
        else:
            print("   No recent 1s data found")
            results['latest_1s_data'] = None
        
        client.close()
        
    except Exception as e:
        print(f"   ❌ aggr-server check failed: {e}")
        results['aggr_server_error'] = str(e)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    issues = []
    successes = []
    
    if results.get('trades_1s_exists'):
        successes.append("✅ trades_1s measurement exists")
    else:
        issues.append("❌ trades_1s measurement missing")
    
    if results.get('recent_1s_count', 0) > 0:
        successes.append(f"✅ Recent 1s data available ({results['recent_1s_count']} points)")
    else:
        issues.append("❌ No recent 1s data")
    
    if results.get('redis_working'):
        successes.append("✅ Redis signal storage working")
    else:
        issues.append("❌ Redis issues detected")
    
    if len(results.get('continuous_queries', [])) > 0:
        successes.append(f"✅ Continuous queries active ({len(results['continuous_queries'])})")
    else:
        issues.append("❌ No continuous queries found")
    
    # Check 1s mode config
    enable_1s = results.get('env_vars', {}).get('SQUEEZEFLOW_ENABLE_1S_MODE')
    if enable_1s == 'true':
        successes.append("✅ 1s mode enabled")
    else:
        issues.append("❌ 1s mode not enabled or missing")
    
    # Print results
    print("\n✅ WORKING COMPONENTS:")
    for success in successes:
        print(f"  {success}")
    
    print("\n❌ ISSUES IDENTIFIED:")
    for issue in issues:
        print(f"  {issue}")
    
    # Critical next steps
    print("\n🔧 CRITICAL NEXT STEPS:")
    if not results.get('trades_1s_exists'):
        print("  1. Fix aggr-server config to write to trades_1s")
    elif results.get('recent_1s_count', 0) == 0:
        print("  1. Check aggr-server is actively collecting data")
        print("  2. Verify continuous query is populating trades_1s")
    
    if enable_1s != 'true':
        print("  3. Set SQUEEZEFLOW_ENABLE_1S_MODE=true in docker-compose.yml")
    
    overall_health = len(successes) / (len(successes) + len(issues)) * 100
    print(f"\n🎯 Overall 1s Implementation Health: {overall_health:.1f}%")
    
    return overall_health >= 70  # 70% threshold for basic functionality


if __name__ == "__main__":
    success = asyncio.run(test_1s_implementation())
    exit(0 if success else 1)