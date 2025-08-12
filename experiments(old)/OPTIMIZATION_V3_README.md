# Optimization Framework V3 - Adapted for Current System

## 🎯 What's New in V3

This version is carefully adapted to work with the CURRENT system without breaking anything:

1. **Works with TradingView Unified Dashboard** - The new single HTML with tabs
2. **Integrates Visual Validation** - Claude can SEE the dashboards for self-debugging
3. **Respects Actual Data Flow** - Follows the documented dependency chain
4. **Uses Remote InfluxDB** - Always connects to 213.136.75.120
5. **Adaptive Learning Integration** - Builds on previous optimization sessions

## 🚀 Quick Start

### Test Integration First (Safe)
```bash
# Test that everything works without breaking anything
python3 experiments/test_optimization_integration.py
```

### Run Quick Optimization
```bash
# Optimize with minimal parameters (3 values, 1 symbol)
python3 experiments/run_optimization_v3.py --mode quick
```

### Run Full Optimization
```bash
# Full parameter sweep
python3 experiments/run_optimization_v3.py --mode full
```

### Check Status
```bash
# See what's been learned
python3 experiments/run_optimization_v3.py --mode status
```

### Visual Analysis
```bash
# Capture and analyze dashboard screenshots
python3 experiments/run_optimization_v3.py --mode visual
```

## 📁 Files Created

### Core Framework
- `optimization_framework_v3.py` - Main framework adapted for current system
- `run_optimization_v3.py` - Runner with multiple modes
- `test_optimization_integration.py` - Safe integration tests

### Data Storage
```
experiments/
└── optimization_data_v3/        # V3 specific data
    ├── results.json             # All optimization results
    ├── state.json              # Current state and best parameters
    └── report_*.txt            # Human-readable reports
```

## 🔄 How It Works

### 1. Backtest Execution
- Uses the EXACT command structure the system expects
- Always uses remote InfluxDB (213.136.75.120)
- Always uses 1s timeframe
- Sets parameters via environment variables

### 2. Dashboard Analysis
- Finds the generated dashboard in `/results/backtest_*/`
- Uses visual validator to capture screenshot
- Claude can analyze the screenshot to verify results

### 3. Result Extraction
- Parses backtest console output
- Extracts metrics (trades, win rate, return, etc.)
- Calculates optimization score

### 4. Learning Integration
- Records what worked/didn't work
- Updates adaptive learner
- Builds on previous sessions

## 🎯 Parameters That Actually Work

### Currently Connected
- `MIN_ENTRY_SCORE` - Strategy entry threshold (via env var)

### To Be Connected
- `CVD_VOLUME_THRESHOLD` - Dynamic volume threshold
- `MOMENTUM_LOOKBACK` - Lookback period for momentum
- `POSITION_SIZE_FACTORS` - Position sizing by score

## 📊 Visual Validation Flow

```
Backtest runs → Dashboard created → Screenshot captured → Claude analyzes
     ↓               ↓                    ↓                    ↓
  engine.py    tradingview_unified   visual_validator    Can verify results
```

## ⚠️ Important Notes

### Data Requirements
- ETH data available: 2025-08-10 onwards
- BTC data available: Check with data availability function
- Always use 1s timeframe

### Safety Features
- Minimal changes to existing system
- All new code in separate files
- Doesn't modify core strategy files
- Tests with small parameter sets first

### Visual Validation
- Requires Chrome, webkit2png, or Safari
- Works without screenshots (just no visual verification)
- Screenshots saved in dashboard directories

## 🔍 Debugging

### If Backtest Fails
```python
# Check data availability
python3 -c "
from influxdb import InfluxDBClient
client = InfluxDBClient(host='213.136.75.120', port=8086, database='significant_trades')
result = client.query('SELECT COUNT(*) FROM aggr_1s.trades_1s WHERE time > now() - 24h')
print(result)
"
```

### If No Screenshots
```bash
# Check if Chrome is available
which google-chrome || which chromium

# Or install webkit2png (macOS)
brew install webkit2png
```

### Check Results
```bash
# View optimization results
cat experiments/optimization_data_v3/results.json | python -m json.tool

# View best parameters
cat experiments/optimization_data_v3/state.json | python -m json.tool
```

## 📈 Example Session

```bash
$ python3 experiments/run_optimization_v3.py --mode quick

================================================================================
SQUEEZEFLOW OPTIMIZATION FRAMEWORK V3
Integrated with Visual Validation
================================================================================

📊 Checking data availability...
ETH: 86400 data points in last 24h

🚀 Running quick optimization...
   Testing MIN_ENTRY_SCORE with 3 values

🚀 Running backtest: ETH with MIN_ENTRY_SCORE=2.0
   Trades: 45, Win Rate: 42.2%, Return: 2.34%, Score: 58.3
📸 Screenshot saved: dashboard_screenshot_20250811_235500.png

🚀 Running backtest: ETH with MIN_ENTRY_SCORE=3.0
   Trades: 28, Win Rate: 50.0%, Return: 3.67%, Score: 68.5
📸 Screenshot saved: dashboard_screenshot_20250811_235520.png

🚀 Running backtest: ETH with MIN_ENTRY_SCORE=4.0
   Trades: 12, Win Rate: 58.3%, Return: 2.89%, Score: 65.2
📸 Screenshot saved: dashboard_screenshot_20250811_235540.png

📊 Quick Optimization Results:
ETH:
  MIN_ENTRY_SCORE: Best=3.0, Score=68.5

✨ Done!
```

## 🎯 Next Steps

1. **Run integration test** to verify everything works
2. **Run quick optimization** to test with minimal parameters
3. **Check screenshots** to see visual validation working
4. **Run full optimization** when ready for comprehensive testing
5. **Review learnings** to see what the system discovered

## 🚨 What NOT to Change

- Don't modify `backtest/reporting/visualizer.py` - it's working
- Don't change `tradingview_unified.py` - it's the standard now
- Don't alter the backtest engine's core logic
- Don't change how parameters are passed to strategy

## ✅ What This Framework Does

- **Safely integrates** with existing system
- **Visually validates** results for self-debugging
- **Learns and adapts** across sessions
- **Respects the actual** data flow and structure
- **Works with what exists** rather than fighting it

The framework is now ready to optimize your strategy parameters while Claude can visually verify the results!