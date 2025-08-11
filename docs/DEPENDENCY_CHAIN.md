# Critical Dependency Chain - Dashboard Generation

## The Full Chain (What Actually Happens)

```
1. engine.py:run()
   ↓
2. engine.py:_create_result() → Creates result dict
   ↓
3. engine.py:327 → visualizer.create_backtest_report(result, dataset, executed_orders)
   ↓
4. visualizer.py:create_backtest_report() → Takes 3 params
   ↓
5. visualizer.py:33 → Adds executed_orders to results dict
   ↓
6. visualizer.py:37 → tv_viz.create_dashboard(results, dataset, output_dir)
   ↓
7. tradingview_unified.py:create_dashboard() → Takes 3 params (NOT executed_orders)
   ↓
8. Returns: STRING path to dashboard.html
   ↓
9. engine.py:332 → result['visualization_path'] = visualization_path
   ↓
10. engine.py:372 → _generate_performance_report(result) 
    ↓
11. ERROR (was here - fixed)
```

## Critical Interfaces That MUST Match

### 1. Visualizer Interface
```python
def create_backtest_report(self, results: Dict, dataset: Dict, executed_orders: List[Dict]) -> str:
```
- **Called by**: engine.py line 327
- **MUST return**: STRING path to dashboard
- **Parameters**: EXACTLY 3 - results, dataset, executed_orders

### 2. Dashboard Implementation Interface
```python
def create_dashboard(self, results: Dict, dataset: Dict, output_dir: str) -> str:
```
- **Called by**: visualizer.py
- **MUST return**: STRING path
- **Parameters**: EXACTLY 3 - results (with executed_orders inside), dataset, output_dir

## Common Breaking Points

### 1. Parameter Mismatch
- Engine passes 3 params to visualizer
- If visualizer signature changes → BREAKS
- If dashboard expects different params → BREAKS

### 2. Return Type Mismatch
- Visualizer MUST return string path
- If returns dict or None → BREAKS engine

### 3. Data Structure Assumptions
- TradingView expects executed_orders IN results dict
- If passed separately → BREAKS

### 4. Performance Report Bug (FIXED)
- Line 1549 had `report = {'summary': 'string'}` 
- Should be `report = {'summary': {}}`
- Caused "str has no attribute get" error

## How to NOT Break This Chain

1. **NEVER change method signatures** without checking ALL callers
2. **ALWAYS verify return types** match what caller expects
3. **TEST the full chain** not just individual components
4. **Document expected interfaces** (this file!)

## Testing the Chain

```python
# Test that should ALWAYS pass
from backtest.engine import BacktestEngine
from backtest.reporting.visualizer import BacktestVisualizer

# Check signatures match
engine = BacktestEngine()
viz = BacktestVisualizer("test")

# These should work without errors
result = {'symbol': 'BTC'}
dataset = {'ohlcv': pd.DataFrame()}
orders = []

path = viz.create_backtest_report(result, dataset, orders)
assert isinstance(path, str), "Must return string path!"
```

## The Lesson

The dashboard was NEVER broken! The error was in a completely unrelated performance report that had a typo ('string' instead of {}). But because the error appeared AFTER the dashboard generation, it looked like the dashboard failed.

**Always check the EXACT error location, not just where you think it might be!**