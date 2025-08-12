# Archived Optimization Framework Files

## ⚠️ These Files Are Deprecated

This directory contains the old optimization framework files that have been superseded by the V3 framework.

### Archived Files:
1. **optimization_framework.py** - Original parameter optimization framework
2. **autonomous_optimizer_v2.py** - V2 autonomous optimization
3. **test_optimization_framework.py** - Tests for old framework
4. **deep_optimizer.py** - Deep optimization with bug fixing
5. **run_optimization.py** - Old runner script

### Why Archived:
- **Replaced by V3** - The new `optimization_framework_v3.py` handles everything better
- **Dashboard Incompatibility** - Didn't work with new TradingView unified dashboard
- **No Visual Validation** - Couldn't capture/analyze screenshots
- **Breaking Changes** - Modified core files directly instead of using env vars

### If You Need These:
These files are kept for reference but should NOT be used with the current system.
Use the V3 framework instead:

```bash
# DON'T use:
python3 experiments/archived_optimization/run_optimization.py

# DO use:
python3 experiments/run_optimization_v3.py --mode quick
```

### What's Still Active:
The following components from the old system are STILL USED:
- `adaptive_learner.py` - Learning persistence (used by V3)
- `concept_validator.py` - Concept validation
- `self_modifying_optimizer.py` - Code modification capability

### Archived Date: 2024-08-12
### Reason: System upgraded to V3 optimization framework with visual validation