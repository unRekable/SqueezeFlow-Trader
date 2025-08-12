# Experiments Directory - Current Status

## 🎯 Active Optimization Framework: V3

### Quick Start
```bash
# Test integration first (safe)
python3 test_optimization_integration.py

# Run quick optimization
python3 run_optimization_v3.py --mode quick

# Check status
python3 run_optimization_v3.py --mode status
```

## 📁 Directory Structure

```
experiments/
├── 🆕 V3 Framework (CURRENT)
│   ├── optimization_framework_v3.py      # Core V3 framework
│   ├── run_optimization_v3.py           # V3 runner
│   ├── test_optimization_integration.py # Integration tests
│   └── OPTIMIZATION_V3_README.md        # V3 documentation
│
├── 🧠 Adaptive Framework (ACTIVE)
│   ├── adaptive_learner.py              # Learning persistence
│   ├── concept_validator.py             # Concept validation
│   ├── self_modifying_optimizer.py      # Code modification
│   └── run_adaptive_optimization.py     # Adaptive runner
│
├── 📊 Data Storage
│   ├── optimization_data_v3/            # V3 results and state
│   ├── adaptive_learning/               # Learning journal
│   ├── concept_validation/              # Validation results
│   └── code_modifications/              # Code change history
│
├── 📚 Documentation
│   ├── CURRENT_STATUS.md               # This file
│   ├── OPTIMIZATION_CLEANUP.md         # Cleanup documentation
│   ├── OPTIMIZATION_V3_README.md       # V3 guide
│   ├── OPTIMIZATION_FRAMEWORK.md       # Framework overview
│   └── README.md                        # Adaptive framework guide
│
└── 📦 Archived (DEPRECATED)
    └── archived_optimization/           # Old framework files
        ├── optimization_framework.py
        ├── autonomous_optimizer_v2.py
        ├── test_optimization_framework.py
        ├── deep_optimizer.py
        ├── run_optimization.py
        └── README.md                   # Archive documentation
```

## ✅ What's Working Now

### V3 Framework Features:
- **Visual Validation** - Captures dashboard screenshots for analysis
- **Proper Integration** - Works with TradingView unified dashboard
- **Safe Testing** - Tests with minimal parameters first
- **Learning Persistence** - Uses adaptive_learner for continuity
- **Non-Breaking** - All changes via environment variables

### Adaptive Features (Still Active):
- **Concept Validation** - Tests if trading concepts actually work
- **Code Modification** - Can modify strategy code (with confirmation)
- **Learning Journal** - Maintains knowledge across sessions
- **Question Tracking** - Knows what to investigate next

## 🚀 How to Use

### For Parameter Optimization (V3):
```bash
# 1. Quick test (3 values, 1 symbol)
python3 run_optimization_v3.py --mode quick

# 2. Full optimization
python3 run_optimization_v3.py --mode full

# 3. Visual analysis of dashboards
python3 run_optimization_v3.py --mode visual

# 4. Check what was learned
python3 run_optimization_v3.py --mode status
```

### For Concept Validation (Adaptive):
```bash
# Run adaptive optimization (validates concepts + modifies code)
python3 run_adaptive_optimization.py
```

## 📊 Results Location

### V3 Optimization:
- Results: `optimization_data_v3/results.json`
- State: `optimization_data_v3/state.json`
- Reports: `optimization_data_v3/report_*.txt`

### Adaptive Learning:
- Journal: `adaptive_learning/learning_journal.json`
- Questions: `adaptive_learning/questions.json`
- Concepts: `concept_validation/results/`

## 🔄 Migration from Old Framework

If you were using the old framework:

| Old Command | New Command |
|-------------|-------------|
| `python3 run_optimization.py` | `python3 run_optimization_v3.py --mode quick` |
| `python3 autonomous_optimizer_v2.py` | `python3 run_optimization_v3.py --mode full` |
| `python3 deep_optimizer.py` | `python3 run_adaptive_optimization.py` |

## 📈 Next Steps

1. **Run Integration Test** - Verify everything works
2. **Quick Optimization** - Test with MIN_ENTRY_SCORE
3. **Check Screenshots** - Verify visual validation works
4. **Review Results** - See what parameters work best
5. **Apply Learnings** - Update strategy with findings

## ⚠️ Important Notes

- **Always use V3** for parameter optimization
- **Adaptive framework** still useful for concept validation
- **Old files archived** but available for reference
- **Visual validation** requires Chrome/webkit2png
- **Remote InfluxDB** always required (213.136.75.120)

## 🎯 Current Focus

The V3 framework is production-ready and integrates properly with:
- TradingView unified dashboard
- Visual validation system
- Adaptive learning persistence
- Current data flow architecture

Use V3 for all optimization tasks going forward!