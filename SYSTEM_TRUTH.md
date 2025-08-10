# System Truth - What Actually Works

## Critical Understanding (DO NOT FORGET)

### 1. Data Location & Storage
- **Database:** `significant_trades`
- **Retention Policy:** `aggr_1s` (NOT default, NOT rp_1s)
- **Measurement:** `trades_1s`
- **Query:** `SELECT * FROM "aggr_1s"."trades_1s"`
- **Data Available:** 24+ hours of 1-second data (2M+ data points)
- **Aggregation:** FIRST/LAST/MAX/MIN for proper OHLCV candles

### 2. 1-Second Data Purpose
- **1s is for EXECUTION LATENCY, not analysis timeframe**
- We collect every 1 second for fast execution
- We ANALYZE on 1m, 5m, 15m, 30m, 1h, 4h built FROM 1s data
- This gives us 1-2 second signal latency vs 60+ seconds
- Strategy runner lookback: 4 hours (240 minutes) for multi-timeframe analysis

### 3. Strategy Reality
- **5 Phases:** Context → Divergence → Reset → Scoring → Exit
- **CRITICAL:** Phase 2 (Divergence) MUST detect divergence for any trade
- **Phase 3 (Reset) IS the liquidity provision moment** - not separate
- **NO fixed thresholds** - Dynamic adaptation only
- **Score 4.0+ required** to enter trades (but ONLY if divergence exists)
- **Realistic frequency:** ~3 trades per day (not 17+)

### 4. CVD (Cumulative Volume Delta)
- **Formula:** `(buy_volume - sell_volume).cumsum()`
- **TRUE Divergence:** When spot CVD and futures CVD move OPPOSITE directions
  - Long Setup: Spot CVD UP + Futures CVD DOWN
  - Short Setup: Spot CVD DOWN + Futures CVD UP
- **NOT Divergence:** Both moving same direction (even if one moves more)
- **Key:** This is about spot vs futures DISAGREEMENT, not magnitude differences

### 5. Architecture That Works
```
aggr-server (Node.js) 
    ↓ [1s data from 80+ markets]
InfluxDB (aggr_1s.trades_1s)
    ↓
DataPipeline 
    ↓
SqueezeFlowStrategy (5 phases)
    ↓
Orders → FreqTrade
```

### 6. Fixed Issues (2025-08-10)
- **Timezone sync:** All services use UTC via docker-compose.timezone.yml
- **InfluxDB connection:** Uses INFLUX_HOST env var (not localhost)
- **OHLCV aggregation:** Fixed to use FIRST/LAST/MAX/MIN (not mean)
- **Lookback period:** Increased from 30min to 4 hours
- **Data queries:** Always use `aggr_1s` retention policy
- **Parallel processing:** Working with 4 workers
- **Backtest speed:** ~4 seconds for 24 hours of data

## Current Configuration

### Environment Variables (docker-compose.yml)
```yaml
INFLUX_HOST: aggr-influx
INFLUX_PORT: 8086
REDIS_HOST: redis
REDIS_PORT: 6379
SQUEEZEFLOW_RUN_INTERVAL: 60  # Strategy runs every 60s
SQUEEZEFLOW_MAX_SYMBOLS: 3
TZ: UTC  # All services use UTC
```

### Service Status
- **aggr-server:** Collecting 1s data from 80+ markets
- **InfluxDB:** Storing in aggr_1s retention policy (7 days)
- **Strategy Runner:** Processing BTC/ETH with 4h lookback
- **FreqTrade:** Connected and receiving signals
- **Redis:** Caching and pub/sub working

## Open Interest (OI) Data Structure

### OI Collection & Aggregation
- **Exchanges:** 4 major exchanges (BINANCE_FUTURES, BYBIT, OKX, DERIBIT)
- **Base Symbols:** Aggregated by base asset (BTCUSDT+BTCUSDC → BTC)
- **Storage:** 3 types of records per symbol

### OI Query Patterns
```bash
# Individual exchange OI
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='BINANCE'

# Top 3 futures combined (BINANCE + BYBIT + OKX)
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='FUTURES_AGG'

# All exchanges combined (includes DERIBIT options)
SELECT * FROM open_interest WHERE symbol='BTC' AND exchange='TOTAL_AGG'

# Compare all OI types for BTC
SELECT exchange, open_interest FROM open_interest 
WHERE symbol='BTC' AND time > now() - 1h ORDER BY time DESC
```

### OI Data Tags
- **aggregate_type:** 'individual', 'futures_aggregate', 'total_aggregate'
- **exchange:** Exchange name or 'FUTURES_AGG'/'TOTAL_AGG'
- **symbol:** Base symbol (BTC, ETH, etc.)

## Commands That Work

```bash
# Check data availability
docker exec aggr-influx influx -execute "SELECT COUNT(*) FROM \"aggr_1s\".\"trades_1s\" WHERE time > now() - 24h" -database significant_trades

# Check OI data availability
docker exec aggr-influx influx -execute "SELECT * FROM open_interest WHERE time > now() - 1h ORDER BY time DESC LIMIT 10" -database significant_trades

# Run backtest (fast - ~4 seconds)
python3 backtest/engine.py --symbol BTC --start-date 2025-08-09 --end-date 2025-08-09

# Check system health
curl http://localhost:8090/health

# View strategy runner logs
docker logs squeezeflow-strategy-runner --tail 20

# Restart services with timezone fix
docker-compose -f docker-compose.yml -f docker-compose.timezone.yml up -d
```

## Anti-Patterns to Avoid

1. **DON'T forget the retention policy** - It's always `aggr_1s`
2. **DON'T use localhost in containers** - Use service names
3. **DON'T use mean() for OHLCV** - Use FIRST/LAST/MAX/MIN
4. **DON'T use short lookbacks** - Need 4h for multi-timeframe
5. **DON'T treat 1s as analysis timeframe** - It's collection frequency
6. **DON'T oversimplify** - The strategy IS complex for good reasons

## File Locations

- **Strategy:** `strategies/squeezeflow/strategy.py`
- **Backtest:** `backtest/engine.py`
- **Data Pipeline:** `data/pipeline.py`
- **InfluxDB Client:** `data/loaders/influx_client.py`
- **Strategy Runner:** `services/strategy_runner.py`
- **Configuration:** `docker-compose.yml` (all config here)