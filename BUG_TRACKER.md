# Bug Tracker - SqueezeFlow Trader

## Active Bugs

### 1. Volume Threshold Bug
**Location**: `strategies/squeezeflow/components/phase2_divergence.py:242`
**Issue**: Hardcoded `min_change_threshold = 1e6` (1 million volume)
**Impact**: Low-volume symbols (TON, AVAX, SOL) cannot generate trades
**Evidence**:
- ETH max CVD change: 106M (works fine)
- TON max CVD change: 202K (blocked by 1M threshold)
- AVAX/SOL: Similar low volumes blocked

**Fix Approach**:
```python
# Instead of hardcoded threshold:
if len(spot_cvd) >= 20:
    recent_volumes = np.abs(np.diff(spot_cvd.iloc[-20:].values))
    min_change_threshold = np.percentile(recent_volumes, 75)
    min_change_threshold = max(min_change_threshold, 1000)  # Minimum 1K
```

### 2. Performance Degradation During Trading
**Location**: Throughout backtest engine
**Issue**: Performance drops from 4000 to 173 points/sec (20x slower) during active trading
**Root Cause**: Excessive logging (3-4 lines per trade)
**Temporary Fix**: Increased min_entry_score from 4.0 to 6.0 to reduce trades
**Proper Fix**: Reduce logging verbosity or batch log writes

### 3. Configuration Not Fully Integrated
**Location**: Multiple phase files
**Issue**: Not all components read from central config
- `phase3_reset.py`: NO config import
- `phase5_exits.py`: NO config import
- `phase2_divergence.py`: Has import but doesn't use for threshold
**Impact**: Changes to config don't propagate to all components

### 4. Data Access Inconsistency
**Location**: Strategy and backtest files
**Issue**: Multiple data access patterns
- `backtest/engine.py` → `data/pipeline.py` → `influx_client.py`
- `strategy.py` → Direct InfluxDB queries (bypassing pipeline)
**Impact**: Potential data inconsistencies between backtest and live

## Fixed Bugs

(None documented yet)

## Notes
This file tracks specific bugs for fixing. General system behavior and what works/doesn't work should be in SYSTEM_TRUTH.md.