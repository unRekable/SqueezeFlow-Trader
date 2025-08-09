#!/usr/bin/env python3
"""
Setup Continuous Queries for 1-Second Data
Create continuous queries to populate trades_1s measurement from raw trades
"""

import sys
import os
sys.path.append('/app')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loaders.influx_client import OptimizedInfluxClient, QueryOptimization


def setup_1s_continuous_queries():
    """Setup continuous queries for 1-second data aggregation"""
    print("🔧 Setting up 1-Second Continuous Queries")
    print("=" * 60)
    
    # Initialize client
    config = QueryOptimization(
        connection_pool_size=2,
        query_timeout_seconds=60,
        enable_query_cache=False
    )
    
    client = OptimizedInfluxClient(
        host='aggr-influx',
        port=8086,
        database='significant_trades',
        optimization_config=config
    )
    
    try:
        # Step 1: Check current continuous queries
        print("\n📊 Current Continuous Queries:")
        cq_check = client._execute_query_sync("SHOW CONTINUOUS QUERIES", 'significant_trades')
        print(f"   Found queries: {len(cq_check)}")
        
        # Step 2: Drop existing 1s continuous query if it exists
        try:
            drop_cq = """DROP CONTINUOUS QUERY "cq_trades_1s" ON "significant_trades" """
            client._execute_query_sync(drop_cq, 'significant_trades')
            print("   ✅ Dropped existing cq_trades_1s")
        except:
            print("   ℹ️  No existing cq_trades_1s found")
        
        # Step 3: Create continuous query for 1-second aggregation from raw trades
        create_cq_query = """
        CREATE CONTINUOUS QUERY "cq_trades_1s" ON "significant_trades"
        BEGIN
          SELECT 
            first(open) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close) AS close,
            sum(vbuy) AS vbuy,
            sum(vsell) AS vsell,
            sum(cbuy) AS cbuy,
            sum(csell) AS csell,
            sum(lbuy) AS lbuy,
            sum(lsell) AS lsell
          INTO "rp_1s"."trades_1s"
          FROM "trades"
          GROUP BY time(1s), market
        END
        """
        
        print("\n🔧 Creating 1s continuous query...")
        try:
            client._execute_query_sync(create_cq_query, 'significant_trades')
            print("   ✅ Created continuous query 'cq_trades_1s'")
        except Exception as e:
            print(f"   ❌ Failed to create continuous query: {e}")
        
        # Step 4: Verify continuous query was created
        print("\n📊 Verifying Continuous Queries:")
        cq_verify = client._execute_query_sync("SHOW CONTINUOUS QUERIES", 'significant_trades')
        print(f"   Total queries: {len(cq_verify)}")
        
        # Step 5: Check if raw trades data exists
        print("\n📊 Checking Raw Trades Data:")
        try:
            raw_trades_check = client._execute_query_sync(
                "SELECT COUNT(*) FROM trades WHERE time > now() - 1h", 
                'significant_trades'
            )
            if not raw_trades_check.empty:
                count = raw_trades_check.iloc[0].get('count', 0) if 'count' in raw_trades_check.columns else 0
                print(f"   Raw trades in last hour: {count}")
            else:
                print("   No raw trades data found")
                
            # Check measurement structure
            measurements_check = client._execute_query_sync("SHOW MEASUREMENTS", 'significant_trades')
            measurements = measurements_check['name'].tolist() if not measurements_check.empty else []
            print(f"   Available measurements: {measurements}")
            
        except Exception as e:
            print(f"   Error checking raw trades: {e}")
        
        # Step 6: Trigger manual backfill for recent data
        print("\n🔄 Manual backfill for recent 1s data...")
        try:
            backfill_query = """
            SELECT 
                first(open) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close) AS close,
                sum(vbuy) AS vbuy,
                sum(vsell) AS vsell,
                sum(cbuy) AS cbuy,
                sum(csell) AS csell,
                sum(lbuy) AS lbuy,
                sum(lsell) AS lsell
            INTO "rp_1s"."trades_1s"
            FROM trades
            WHERE time > now() - 2h
            GROUP BY time(1s), market
            """
            
            backfill_result = client._execute_query_sync(backfill_query, 'significant_trades')
            print(f"   ✅ Backfill completed: {len(backfill_result)} points")
            
        except Exception as e:
            print(f"   ❌ Backfill failed: {e}")
        
        print("\n✅ 1s Continuous Queries Setup Completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Setup Failed: {e}")
        return False
    
    finally:
        client.close()


if __name__ == "__main__":
    setup_1s_continuous_queries()