# Performance Optimization TODO

## Critical Performance Issues Identified (2025-08-11)

### 1. Trade Execution Overhead
**Problem**: Speed drops from 4000 points/sec to 173 points/sec when trading is active (20x slower!)
**Solution**: 
- Reduce logging verbosity during trades (currently 3-4 log lines per trade)
- Consider batch logging instead of real-time

### 2. Excessive Trading Frequency
**Problem**: 272 trades in 24 hours (every 5 minutes average) causing massive overhead
**Temporary Fix Applied**: Increased `min_entry_score` from 4.0 to 6.0 in config.py
**Long-term Solutions**:
- Add time-based trade cooldown periods
- Implement trade clustering detection to avoid rapid-fire trades
- Add volatility-adjusted thresholds

### 3. No Strategy Caching
**Problem**: Full recalculation of all indicators for every 1-second tick
**Solutions Needed**:
- Cache Phase 1 context results (market bias, timeframe data) - valid for 1-5 minutes
- Cache Phase 2 divergence calculations - valid until new candle forms
- Cache multi-timeframe aggregations - only recalc when new candle closes
- Use numpy views instead of copies for sliding windows

### 4. DataFrame Operations
**Problem**: Inefficient pandas operations in hot path
**Solutions**:
- Pre-allocate arrays for known sizes
- Use numpy operations instead of pandas where possible
- Vectorize operations instead of loops

### 5. Memory Management
**Problem**: No explicit memory management, potential memory leaks with large datasets
**Solutions**:
- Implement rolling window data structures with fixed memory
- Clear old data from memory after processing
- Use memory-mapped files for large datasets

## Benchmarks (Current Performance)

### With Trading (272 trades):
- Initial: 4000 points/sec
- During trades: 173-500 points/sec
- Total time: ~140 seconds for 84,600 points

### Expected After Optimization:
- Target: 10,000+ points/sec consistently
- Max degradation during trades: 50% (5000 points/sec)
- Total time: <10 seconds for 84,600 points

## Priority Order:
1. **HIGH**: Fix logging overhead (quick win, big impact)
2. **HIGH**: Implement strategy caching (major speedup)
3. **MEDIUM**: Already done - reduce trade frequency via config
4. **LOW**: Memory optimizations (only needed for very long backtests)

## Testing Command:
```bash
# Run this to test performance after changes:
cd "/Users/u/PycharmProjects/SqueezeFlow Trader"
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py \
  --symbol ETH \
  --start-date 2025-08-10 \
  --end-date 2025-08-10 \
  --timeframe 1s \
  --balance 10000 \
  --leverage 1.0 \
  --strategy SqueezeFlowStrategy
```

## Notes:
- min_entry_score increased to 6.0 reduces trades by ~60-70%
- Dashboard visualization fixed (DataFrame.empty issue resolved)
- All changes documented in git commits