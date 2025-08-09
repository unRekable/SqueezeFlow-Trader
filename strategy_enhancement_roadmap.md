# SqueezeFlow Strategy Enhancement Roadmap

## Current Focus: 1-Second Step Backtest Implementation

### Objective
Modify the backtest engine to evaluate the strategy every second when using 1s data, ensuring we test against the full granularity of available data. This document tracks all ongoing and planned strategy enhancements.

## Current State
- **Data Collection**: 1-second intervals ✅
- **Backtest Steps**: 1-minute intervals (60x slower than data) ❌
- **Result**: Up to 60-second delay in signal generation during backtests

## Required Changes

### 1. Backtest Engine (`/backtest/engine.py`)

#### A. Streaming Backtest Method (lines 810-818)
**Current:**
```python
if timeframe == '1s':
    window_hours = 1  # Smaller windows for 1s data
    step_minutes = 1  # Step forward 1 minute for dense data
else:
    window_hours = 2  
    step_minutes = 5
```

**Change to:**
```python
if timeframe == '1s':
    window_hours = 1  # Smaller windows for 1s data
    step_seconds = 1  # Step forward 1 SECOND for true 1s granularity
    step_duration = timedelta(seconds=step_seconds)
else:
    window_hours = 2  
    step_minutes = 5
    step_duration = timedelta(minutes=step_minutes)
```

#### B. Rolling Windows Method (lines 307-312)
**Current:**
```python
window_hours = 4  # Each window contains 4 hours of data
step_minutes = 5  # Step forward 5 minutes each iteration
```

**Change to:**
```python
if self.enable_1s_mode or timeframe == '1s':
    window_hours = 1  # 1-hour window for 1s mode (3600 data points)
    step_seconds = 1  # Step forward 1 SECOND
    step_duration = timedelta(seconds=step_seconds)
else:
    window_hours = 4  # Each window contains 4 hours of data
    step_minutes = 5  # Step forward 5 minutes each iteration
    step_duration = timedelta(minutes=step_minutes)
```

### 2. Performance Considerations

#### A. Memory Management
- Window size reduced from 4h to 1h for 1s mode (3,600 points instead of 14,400)
- Still maintains sufficient context for strategy decisions
- Reduces memory pressure during parallel processing

#### B. Processing Optimization
- Expect ~3,600x more strategy evaluations per day (86,400 vs 24)
- May need to disable parallel processing for 1s mode initially
- Monitor CPU usage and adjust thread pool size if needed

#### C. Logging Adjustments
- Reduce logging verbosity in 1s mode to prevent log explosion
- Only log actual trades, not every evaluation
- Add summary statistics every N iterations

### 3. Configuration Updates

#### A. Add Explicit Control Variable
Add to environment variables:
```bash
SQUEEZEFLOW_BACKTEST_STEP_SECONDS=1  # For 1s mode backtesting
SQUEEZEFLOW_BACKTEST_STEP_MINUTES=5  # For regular mode backtesting
```

#### B. Update Config Loader
Modify `services/config/unified_config.py` to read these new variables.

### 4. Testing Plan

1. **Performance Baseline**
   - Run current 1-minute step backtest, note execution time
   - Record memory usage and CPU utilization

2. **1-Second Step Test**
   - Run same period with 1-second steps
   - Compare execution time (expect ~60x longer)
   - Monitor for memory leaks or CPU bottlenecks

3. **Signal Quality Comparison**
   - Compare number of trades generated
   - Analyze entry/exit timing differences
   - Evaluate P&L impact of faster execution

## Implementation Order

1. **Phase 1**: Update streaming backtest method (simplest change)
2. **Phase 2**: Update rolling windows method
3. **Phase 3**: Add configuration variables
4. **Phase 4**: Optimize logging for 1s mode
5. **Phase 5**: Performance testing and optimization

## Expected Impact

### Positive
- True 1-second signal generation matching production behavior
- More accurate backtest results for high-frequency patterns
- Better understanding of strategy behavior at micro timescales

### Challenges
- 60x longer backtest execution time
- Higher memory usage during processing
- Potential for more noise/false signals (to be addressed in strategy tuning)

## Risk Mitigation
- Keep original 1-minute step as fallback option
- Add progress indicators for long-running backtests
- Implement checkpointing for resumable backtests
- Monitor system resources and add automatic throttling if needed

## Future Enhancements (Post 1s-Step Implementation)

### Strategy Noise Reduction
Once we observe strategy behavior at 1-second granularity, we'll address noise through:

1. **Smart Filtering**
   - Minimum movement thresholds (not fixed, but adaptive to volatility)
   - Volume confirmation requirements
   - Multi-tick confirmation for signals

2. **Context-Aware Execution**
   - Different behavior in high vs low volatility periods
   - Time-of-day adjustments (market open/close, news events)
   - Cross-exchange validation for significant moves

3. **Intelligent Debouncing**
   - Prevent flip-flopping between positions
   - Minimum hold periods based on market conditions
   - Signal persistence requirements

### Performance Optimizations

1. **Caching Strategy**
   - Cache Phase 1 context calculations (changes slowly)
   - Reuse divergence calculations across nearby timestamps
   - Smart invalidation of cached results

2. **Selective Processing**
   - Skip full evaluation if price hasn't moved significantly
   - Fast-path exit checks before full analysis
   - Incremental CVD updates instead of full recalculation

3. **Parallel Execution**
   - Process multiple symbols concurrently
   - Pipeline data loading and strategy evaluation
   - Async I/O for database operations

### Live Trading Preparation

1. **Event-Driven Architecture**
   - React to each new 1s candle immediately
   - WebSocket connections for real-time data
   - Queue-based signal processing

2. **Latency Monitoring**
   - Track time from data arrival to signal generation
   - Identify and optimize bottlenecks
   - Set up alerting for latency spikes

3. **Safety Mechanisms**
   - Circuit breakers for excessive trading
   - Position size limits
   - Daily loss limits

---

**Status**: Documentation updated. Ready to proceed with 1-second step implementation.