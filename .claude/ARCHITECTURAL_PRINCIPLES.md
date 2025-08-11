# Architectural Principles for SqueezeFlow Trader

## 🔴 CRITICAL: Single Source of Truth Pattern

### The Problem We're Solving
When a configuration or behavior change is made in one place, it must automatically propagate to ALL dependent components without manual updates. We wasted hours fixing OI references across multiple files when it should have been controlled from ONE place.

### The Solution: Centralized Configuration with Automatic Propagation

## 1. Configuration Hierarchy

```
/backtest/indicator_config.py (MASTER CONFIG)
    ↓
/data/pipeline.py (reads config, controls data loading)
    ↓
/strategies/*/components/* (read from pipeline/config)
    ↓
/backtest/reporting/* (read from pipeline/config)
```

## 2. Implementation Rules

### RULE 1: Never Hardcode Assumptions
❌ **BAD:**
```python
# In phase2_divergence.py
if has_divergence and OI_TRACKING_AVAILABLE:
    oi_confirmed, oi_data = oi_tracker.validate_squeeze_signal()
```

✅ **GOOD:**
```python
# In phase2_divergence.py
if has_divergence and self.config.enable_open_interest:
    oi_confirmed, oi_data = oi_tracker.validate_squeeze_signal()
```

### RULE 2: All Components Must Read from Config
Every component that might use an indicator MUST:
1. Import the config
2. Check the config before using the indicator
3. Provide sensible defaults when disabled

### RULE 3: Config Changes Must Be Self-Contained
When changing config, the change should automatically affect:
- Data loading (pipeline)
- Strategy calculations (phases)
- Visualizations (reporting)
- Validations (quality checks)

## 3. Standard Pattern for New Features

When adding a new indicator or feature:

```python
# 1. Add to indicator_config.py
@dataclass
class IndicatorConfig:
    enable_new_feature: bool = True  # or False if experimental
    
# 2. In data/pipeline.py
def load_data(self):
    if self.config.enable_new_feature:
        # Load the data
        
# 3. In strategy components
def calculate(self):
    if self.config.enable_new_feature:
        # Use the feature
    else:
        # Provide default/neutral value
        
# 4. In visualization
def visualize(self):
    if self.config.enable_new_feature:
        # Show the visualization
```

## 4. Testing Configuration Changes

After ANY config change, run this checklist:

```bash
# 1. Test config loads correctly
python3 -c "from backtest.indicator_config import get_indicator_config; print(get_indicator_config())"

# 2. Test pipeline respects config
python3 -c "from data.pipeline import DataPipeline; p = DataPipeline(); print(p.config)"

# 3. Run integration test
python3 test_config_integration.py

# 4. Run minimal backtest to verify
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py --symbol BTC --start-date 2025-08-10 --end-date 2025-08-10 --timeframe 5m --balance 10000 --strategy SqueezeFlowStrategy
```

## 5. Common Integration Points

These files MUST always check config:

### Data Layer
- `/data/pipeline.py` - Controls what data is loaded
- `/data/loaders/*` - Should respect pipeline config
- `/data/processors/*` - Should check config before processing

### Strategy Layer  
- `/strategies/squeezeflow/components/phase2_divergence.py`
- `/strategies/squeezeflow/components/phase4_scoring.py`
- `/strategies/squeezeflow/components/phase5_exits.py`

### Visualization Layer
- `/backtest/reporting/interactive_strategy_visualizer.py`
- `/backtest/reporting/visualizer.py`
- `/backtest/reporting/data_serializer.py`

## 6. Environment Variable Convention

All config environment variables follow this pattern:
```bash
BACKTEST_ENABLE_[FEATURE]  # For on/off switches
BACKTEST_[FEATURE]_[PARAM]  # For feature parameters
```

Examples:
- `BACKTEST_ENABLE_OI=false`
- `BACKTEST_ENABLE_SPOT_CVD=true`
- `BACKTEST_OI_THRESHOLD=5.0` (if OI had parameters)

## 7. Adding New Indicators Checklist

When adding a new indicator:

- [ ] Add flag to `/backtest/indicator_config.py`
- [ ] Update `/data/pipeline.py` to check flag before loading
- [ ] Update all strategy phases that might use it
- [ ] Update visualization to skip if disabled
- [ ] Add to `test_config_integration.py`
- [ ] Document in this file
- [ ] Test with flag enabled AND disabled

## 8. Why This Matters

1. **Saves Hours**: One config change vs. hunting through 10+ files
2. **Prevents Bugs**: Can't forget to update a component
3. **Enables Experimentation**: Easy to toggle features on/off
4. **Improves Performance**: Skip unused calculations
5. **Simplifies Debugging**: Disable problematic components

## 9. Current State (2025-08-11)

### Properly Integrated
- ✅ Open Interest (OI) - Disabled via config
- ✅ Spot CVD - Config controlled
- ✅ Futures CVD - Config controlled  
- ✅ CVD Divergence - Config controlled
- ✅ Volume data - Config controlled

### TODO: Future Improvements
- [ ] Make strategy phases directly read from config (not just pipeline)
- [ ] Add config validation on startup
- [ ] Add config presets (fast, full, debug)
- [ ] Make config hot-reloadable

## 10. Remember This Pattern!

**Every time you change ANYTHING that multiple components use:**
1. Put it in config
2. Make all components read from config
3. Test that changes propagate automatically
4. Document the config option

This is not optional - it's the foundation of maintainable code.