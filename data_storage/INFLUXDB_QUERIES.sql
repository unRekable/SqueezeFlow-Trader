-- ============================================================================
-- INFLUXDB QUERIES - COPY AND PASTE THESE
-- ============================================================================
-- 
-- DATABASE: significant_trades
-- RETENTION POLICY: aggr_1s (NOT DEFAULT!!!)
-- MEASUREMENT: trades_1s
--
-- TO RUN THESE:
-- docker exec aggr-influx influx -execute "QUERY_HERE" -database significant_trades
--
-- ============================================================================

-- CHECK IF DATA EXISTS (ALWAYS RUN THIS FIRST)
SELECT COUNT(*) 
FROM "aggr_1s"."trades_1s" 
WHERE time > now() - 1h;

-- GET AVAILABLE MARKETS
SHOW TAG VALUES 
FROM "aggr_1s"."trades_1s" 
WITH KEY = "market";

-- CHECK ETH DATA
SELECT COUNT(*) 
FROM "aggr_1s"."trades_1s" 
WHERE market =~ /ETH/ 
AND time > now() - 1h;

-- CHECK BTC DATA  
SELECT COUNT(*) 
FROM "aggr_1s"."trades_1s" 
WHERE market =~ /BTC/ 
AND time > now() - 1h;

-- GET LATEST ETH PRICE
SELECT LAST(close) 
FROM "aggr_1s"."trades_1s" 
WHERE market =~ /ETH/;

-- GET LATEST BTC PRICE
SELECT LAST(close) 
FROM "aggr_1s"."trades_1s" 
WHERE market =~ /BTC/;

-- GET 1 MINUTE OF ETH DATA (TEST QUERY)
SELECT * 
FROM "aggr_1s"."trades_1s" 
WHERE market =~ /ETH/ 
AND time > now() - 1m 
LIMIT 60;

-- GET ETH DATA FOR SPECIFIC TIME RANGE
SELECT 
    mean(open) as open,
    max(high) as high,
    min(low) as low,
    mean(close) as close,
    sum(vbuy) + sum(vsell) as volume
FROM "aggr_1s"."trades_1s"
WHERE market =~ /ETH/
AND time >= '2025-08-09T00:00:00Z'
AND time <= '2025-08-09T23:59:59Z'
GROUP BY time(1s);

-- CHECK DATA TIME RANGE
SELECT 
    FIRST(close) as first_data,
    LAST(close) as last_data
FROM "aggr_1s"."trades_1s"
WHERE market =~ /ETH/;

-- ============================================================================
-- WRONG QUERIES (DON'T USE THESE)
-- ============================================================================

-- ❌ WRONG - Missing retention policy
SELECT * FROM trades_1s;  -- NO DATA!

-- ❌ WRONG - Wrong retention policy  
SELECT * FROM "rp_1s"."trades_1s";  -- NO DATA!

-- ❌ WRONG - Default retention policy
SELECT * FROM "autogen"."trades_1s";  -- NO DATA!

-- ============================================================================
-- REMEMBER: IT'S ALWAYS "aggr_1s"."trades_1s"
-- ============================================================================