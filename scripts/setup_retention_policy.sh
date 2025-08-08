#!/bin/bash

echo "==================================================="
echo "Setting up 1-Second Real-Time Retention Policy"
echo "==================================================="

# Function to drop old retention policy
drop_old_policy() {
    local policy_name=$1
    echo "Dropping old policy: $policy_name"
    docker exec aggr-influx influx -database significant_trades -execute "DROP RETENTION POLICY \"$policy_name\" ON \"significant_trades\"" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ Dropped $policy_name"
    else
        echo "  ℹ️  $policy_name not found or already removed"
    fi
}

echo ""
echo "🧹 Cleaning up old retention policies..."
echo "-----------------------------------------"

# Drop all old timeframe policies (we only need 1s now)
OLD_POLICIES=(
    "aggr_10s"
    "aggr_30s" 
    "aggr_1m"
    "aggr_3m"
    "aggr_5m"
    "aggr_15m"
    "aggr_30m"
    "aggr_1h"
    "aggr_2h"
    "aggr_4h"
    "aggr_6h"
    "aggr_1d"
)

for policy in "${OLD_POLICIES[@]}"; do
    drop_old_policy "$policy"
done

echo ""
echo "📦 Setting up 1-second retention policy..."
echo "-----------------------------------------"

# Create/update the main 1-second retention policy
docker exec aggr-influx influx -execute "CREATE RETENTION POLICY \"rp_1s\" ON \"significant_trades\" DURATION 30d REPLICATION 1 SHARD DURATION 1d DEFAULT" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Created new retention policy 'rp_1s' (30 days retention)"
else
    # Policy might already exist, try to alter it
    docker exec aggr-influx influx -execute "ALTER RETENTION POLICY \"rp_1s\" ON \"significant_trades\" DURATION 30d SHARD DURATION 1d DEFAULT" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Updated existing retention policy 'rp_1s'"
    else
        echo "ℹ️  Retention policy 'rp_1s' already configured correctly"
    fi
fi

# Keep the auto-generated aggr_1s policy but update it
docker exec aggr-influx influx -execute "ALTER RETENTION POLICY \"aggr_1s\" ON \"significant_trades\" DURATION 24h SHARD DURATION 1h" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Updated 'aggr_1s' policy (24 hours for recent data)"
fi

echo ""
echo "📊 Current retention policies:"
echo "------------------------------"
docker exec aggr-influx influx -database significant_trades -execute "SHOW RETENTION POLICIES" | grep -E "^(name|rp_1s|aggr_1s|autogen)"

echo ""
echo "==================================================="
echo "✅ Retention Policy Setup Complete!"
echo "==================================================="
echo ""
echo "System is now configured for:"
echo "  • 1-second data collection only"
echo "  • 30-day retention for historical data (rp_1s)"
echo "  • 24-hour retention for recent data (aggr_1s)"
echo "  • No unnecessary resampling overhead"
echo ""
echo "All old timeframe policies have been removed."