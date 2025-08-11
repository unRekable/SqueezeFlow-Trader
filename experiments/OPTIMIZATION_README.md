# SqueezeFlow Optimization Framework v2.0

## 🎯 Purpose
This is a Claude-friendly, self-tracking optimization system that actually works with the current architecture. It automatically finds optimal parameters for the SqueezeFlow trading strategy.

## 🚨 Critical Context (READ FIRST)

### What This Fixes
1. **Hardcoded Volume Threshold Bug**: Line 230 in `phase2_divergence.py` has `min_change_threshold = 1e6` which blocks low-volume symbols (TON, AVAX, SOL) from trading
2. **Disconnected Parameters**: Previous experiments tried to modify env vars that weren't connected to the strategy
3. **Wrong Data Source**: Old optimizer assumed local Docker InfluxDB, but we use remote server (213.136.75.120)
4. **Date Range Issues**: System now checks actual data availability before running backtests

### Current System Architecture
```
REMOTE SERVER (213.136.75.120):
- InfluxDB with 1-second market data
- Data stored in retention policy: aggr_1s
- Markets tagged like: BINANCE:ethusdt

LOCAL DEVELOPMENT:
- Runs backtests using remote data
- Must set INFLUX_HOST=213.136.75.120
- Always use --timeframe 1s
```

## 📁 File Structure

```
experiments/
├── optimization_framework.py       # Core framework with parameter definitions
├── autonomous_optimizer_v2.py      # Self-driving optimizer
├── test_optimization_framework.py  # Verification tests
├── OPTIMIZATION_README.md          # This file
└── optimization_data/              # Results storage (created automatically)
    ├── parameters.json             # Current parameter values
    ├── results.json                # All experiment results
    ├── decisions.json              # Optimization decisions
    ├── optimization_state.json     # Current state
    └── report_*.txt                # Generated reports
```

## 🔧 Key Components

### 1. OptimizableParameter
Defines a parameter that can be optimized:
- **Identity**: name, env_var, location in code
- **Bounds**: min/max values, step size
- **Metadata**: impact level, affected symbols
- **Special**: is_dynamic (calculated per symbol), requires_restart

### 2. ExperimentResult
Comprehensive tracking of each experiment:
- Trading metrics (trades, win rate, return)
- Risk metrics (Sharpe, drawdown, Sortino)
- Market conditions during test
- Performance score (0-100)

### 3. OptimizationFramework
Main framework that:
- Manages parameters and results
- Connects to remote InfluxDB
- Runs backtests with proper configuration
- Saves everything in Claude-readable JSON

### 4. AutonomousOptimizer
Self-driving system that:
- Analyzes symbol characteristics
- Decides what to test next
- Evaluates results and makes decisions
- Generates recommendations

## 🚀 Quick Start

### Test the Framework
```bash
# Verify everything is connected
python3 experiments/test_optimization_framework.py
```

### Run Autonomous Optimization
```bash
# Run 5 experiments automatically
python3 experiments/autonomous_optimizer_v2.py
```

### Run Single Experiment
```python
from optimization_framework import OptimizationFramework

framework = OptimizationFramework()

# Check data availability
has_data, first, last = framework.check_data_availability('ETH')
print(f"ETH data: {first} to {last}")

# Run backtest
result = framework.run_backtest('ETH', 'MIN_ENTRY_SCORE', 3.5)
print(f"Score: {result.performance_score}")
```

## 📊 Parameters Being Optimized

### CVD_VOLUME_THRESHOLD (Critical)
- **Problem**: Hardcoded to 1M, blocks low-volume symbols
- **Solution**: Dynamic calculation based on symbol's actual volume
- **Location**: `phase2_divergence.py:230`
- **Impact**: Critical for TON, AVAX, SOL

### MIN_ENTRY_SCORE
- **Current**: 4.0
- **Range**: 3.0 - 6.0
- **Location**: `config.py:70`
- **Impact**: High - controls trade frequency

### OI_RISE_THRESHOLD
- **Current**: 5.0%
- **Range**: 2.0% - 10.0%
- **Location**: `oi_tracker_influx.py:22`
- **Impact**: Medium - OI confirmation sensitivity

### Other Parameters
- MOMENTUM_LOOKBACK: 60-900 seconds
- VOLUME_SURGE_MULTIPLIER: 1.5-4.0x
- SCORING_WEIGHT_CVD_RESET: 2.0-5.0
- DIVERGENCE_TIMEFRAMES: Various combinations

## 🤖 How the Optimizer Works

### Priority System
1. **Critical Issues**: Fix hardcoded thresholds first
2. **Baselines**: Establish baseline for each symbol
3. **Exploration**: Test undertested parameters
4. **Exploitation**: Refine promising parameters
5. **Validation**: Confirm improvements

### Decision Logic
- **Adopt**: Score improvement >20 points, high confidence
- **Promising**: Score improvement >10 points, needs confirmation
- **Continue Testing**: Not enough samples yet
- **Reject**: Poor performance or worse than baseline

### Symbol-Specific Optimization
The system analyzes each symbol's characteristics:
- **BTC/ETH**: High volume, standard thresholds
- **TON/AVAX/SOL**: Low volume, needs dynamic thresholds
- Automatically adjusts test values based on symbol

## 📈 Reading the Results

### Performance Score (0-100)
Weighted combination of:
- Win rate (15%): 60% win rate = perfect
- Profit (25%): 10% return = perfect
- Sharpe (20%): 2.0 = perfect
- Drawdown (15%): -10% = 0 score
- Trades (10%): 20+ trades = perfect
- Consistency (15%): Profit factor 2.0 = perfect

### Optimization Data Files

**parameters.json**: Current parameter values
```json
{
  "MIN_ENTRY_SCORE": {
    "current_value": 3.5,
    "env_var": "SQUEEZEFLOW_MIN_ENTRY_SCORE",
    "impact": "high"
  }
}
```

**results.json**: All experiment results
```json
[{
  "experiment_id": "ETH_MIN_ENTRY_SCORE_3.5_20250810_143022",
  "performance_score": 72.5,
  "total_trades": 15,
  "win_rate": 66.7
}]
```

**decisions.json**: Optimization decisions
```json
[{
  "action": "adopt",
  "confidence": 0.85,
  "reasoning": ["Significant improvement: +22.5 points over baseline"]
}]
```

## ⚠️ Important Notes

### Always Check Data First
```python
# Before running any backtest
has_data, first, last = framework.check_data_availability('TON')
if not has_data:
    print(f"No data for TON")
```

### ETH Data Starts Aug 10
- ETH data begins 2025-08-10 09:54 UTC
- Don't use dates before this
- Other symbols may have different ranges

### Use Remote InfluxDB
- Always set `INFLUX_HOST=213.136.75.120`
- Never use localhost or Docker InfluxDB
- Data is read-only from local

### Symbol Volume Categories
- **Very High**: BTC, ETH (>10M avg volume)
- **High**: Major alts (>1M avg volume)
- **Medium**: Mid-caps (>100K avg volume)
- **Low**: TON, smaller alts (>10K avg volume)
- **Very Low**: Micro-caps (<10K avg volume)

## 🎯 Next Steps for Future Claude Sessions

1. **Run the test suite** to verify connectivity
2. **Check optimization_data/** for previous results
3. **Run autonomous optimizer** to continue where it left off
4. **Apply recommendations** to docker-compose.yml
5. **Fix the hardcoded threshold** in phase2_divergence.py

## 📝 Example Session

```bash
# 1. Check what's been done
ls experiments/optimization_data/

# 2. Test framework
python3 experiments/test_optimization_framework.py

# 3. Continue optimization
python3 experiments/autonomous_optimizer_v2.py

# 4. Check recommendations
cat experiments/optimization_data/report_*.txt

# 5. Apply best parameters
export SQUEEZEFLOW_MIN_ENTRY_SCORE=3.5
export SQUEEZEFLOW_CVD_VOLUME_THRESHOLD=500000
```

## 🔄 Continuous Improvement

The framework is designed to:
- Learn from every experiment
- Adapt to market conditions
- Make intelligent decisions
- Provide clear explanations
- Save everything for future analysis

Each Claude session can pick up where the last one left off, building on accumulated knowledge to continuously improve the strategy's performance.