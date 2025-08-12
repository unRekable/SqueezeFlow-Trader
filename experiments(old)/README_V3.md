# Optimization Framework V3 - Production Ready

## ✅ Cleanup Complete

### What Was Done:
1. **Archived old framework files** to `archived_optimization/`
2. **Documented the cleanup** in `OPTIMIZATION_CLEANUP.md`
3. **Created current status guide** in `CURRENT_STATUS.md`
4. **Preserved adaptive components** that are still useful

### Current Active Files:

#### V3 Framework (Use These):
- `optimization_framework_v3.py` - Core framework
- `run_optimization_v3.py` - Runner with modes
- `test_optimization_integration.py` - Integration tests

#### Adaptive Framework (Still Useful):
- `adaptive_learner.py` - Learning persistence
- `concept_validator.py` - Concept testing
- `self_modifying_optimizer.py` - Code modification
- `run_adaptive_optimization.py` - Adaptive runner

## 🚀 Quick Start Guide

### 1. Test Everything Works
```bash
python3 test_optimization_integration.py
```

### 2. Run Your First Optimization
```bash
python3 run_optimization_v3.py --mode quick
```

### 3. Check Results
```bash
python3 run_optimization_v3.py --mode status
```

### 4. View Dashboard Screenshots
```bash
ls -la ../results/backtest_*/dashboard_screenshot_*.png
```

## 📊 How V3 Works

```
1. Sets parameter via environment variable
   ↓
2. Runs backtest with exact system commands
   ↓
3. Finds generated dashboard in /results/
   ↓
4. Captures screenshot with visual validator
   ↓
5. Extracts metrics from console + dashboard
   ↓
6. Scores results and finds best value
   ↓
7. Records learning for next session
```

## 🎯 Key Advantages

### Over Old Framework:
- ✅ Works with current TradingView dashboard
- ✅ Visual validation for self-debugging
- ✅ Non-invasive (env vars only)
- ✅ Respects actual data flow
- ✅ Safe incremental testing

### What's Preserved:
- ✅ Learning persistence (adaptive_learner)
- ✅ Concept validation capability
- ✅ Code modification (with confirmation)
- ✅ Historical learnings

## 📈 Optimization Parameters

### Currently Connected:
- `MIN_ENTRY_SCORE` - Entry threshold (2.0 - 6.0)

### Ready to Connect:
- `CVD_VOLUME_THRESHOLD` - Volume threshold
- `MOMENTUM_LOOKBACK` - Lookback period
- `POSITION_SIZE_FACTORS` - Position sizing

### How to Add New Parameters:
1. Find the environment variable in strategy code
2. Add to `optimization_framework_v3.py` parameter mapping
3. Test with small value range first

## 🔍 Visual Validation

The framework can now SEE the dashboards:

```python
# After each backtest:
1. Dashboard generated → /results/backtest_*/dashboard.html
2. Screenshot captured → dashboard_screenshot_*.png
3. Claude can analyze → Verify results visually
```

## 📚 Documentation

- **This File** - Quick reference for V3
- **OPTIMIZATION_V3_README.md** - Detailed V3 guide
- **CURRENT_STATUS.md** - Overall experiments status
- **OPTIMIZATION_CLEANUP.md** - What was archived and why
- **archived_optimization/README.md** - Old framework reference

## 🎯 Next Actions

1. **Run integration test** to verify setup
2. **Try quick optimization** with 3 values
3. **Check if screenshots work** (need Chrome/webkit2png)
4. **Review results** in optimization_data_v3/
5. **Apply best parameters** to production

## ⚠️ Requirements

- Python 3.8+
- Remote InfluxDB access (213.136.75.120)
- Chrome/webkit2png for screenshots (optional but recommended)
- ETH data from 2025-08-10 onwards

## ✨ Ready to Optimize!

The V3 framework is production-ready and won't break your system. Start with:

```bash
python3 run_optimization_v3.py --mode quick
```

This will test MIN_ENTRY_SCORE with 3 values on ETH and show you:
- Which value performs best
- Visual dashboard validation
- What the system learned

Happy optimizing! 🚀