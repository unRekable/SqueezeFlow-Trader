# Adaptive Self-Modifying Optimization Framework

## 🎯 What This Does

This framework **learns WHY** your strategy succeeds or fails, then **modifies its own code** to fix issues. It's not just parameter tuning - it can rewrite the strategy logic itself.

## 🚀 Quick Start

```bash
# Run the complete adaptive optimization
python3 experiments/run_adaptive_optimization.py
```

This will:
1. Check what needs investigation
2. Validate strategy concepts
3. Identify logic problems in code
4. Modify the actual Python files to fix issues
5. Test if modifications helped
6. Learn from results for next time

## 📁 Framework Components

### Core Files
- `run_adaptive_optimization.py` - Main orchestrator (START HERE)
- `concept_validator.py` - Tests if strategy concepts actually work
- `adaptive_learner.py` - Maintains learning across sessions
- `self_modifying_optimizer.py` - Can modify strategy code itself

### Supporting Files
- `optimization_framework.py` - Parameter optimization (if needed)
- `autonomous_optimizer_v2.py` - Autonomous parameter testing

### Generated Folders
```
experiments/
├── adaptive_learning/        # Learning journal & questions
├── code_modifications/       # Code change history & backups
├── concept_validation/       # Validation results
└── optimization_reports/     # Cycle reports
```

## 🔄 How It Works

### The Learning Loop

```
1. VALIDATE: "Can we detect CVD divergence for TON?"
   → No, F1 score = 0.15
   
2. UNDERSTAND: "Why can't we detect it?"
   → Hardcoded 1M threshold, but TON only has 200K volume
   
3. MODIFY: "Change the code"
   → Replace: min_threshold = 1e6
   → With: min_threshold = self._get_dynamic_threshold()
   
4. TEST: "Did it help?"
   → Run backtest: 0 trades → 12 trades ✅
   
5. LEARN: "What worked?"
   → Dynamic thresholds based on symbol characteristics
   
6. APPLY: "Use this learning elsewhere"
   → Check other hardcoded values
```

## 🎯 Key Features

### 1. Concept Validation
Tests if your strategy's assumptions are true:
- Does CVD divergence predict reversals?
- Does OI rise confirm squeezes?
- Can we actually detect these patterns?

### 2. Code Modification
Can change the actual strategy code:
- Fix hardcoded values
- Add missing conditions
- Correct wrong operators
- Refactor static → dynamic

### 3. Adaptive Learning
Maintains knowledge across sessions:
- What we've learned
- What questions remain
- What to test next
- What principles emerged

### 4. Safety Features
- Backs up all files before modifying
- Tests changes with backtests
- Rolls back if changes don't help
- Only applies high-confidence fixes

## 📊 Example Session

```bash
$ python3 run_adaptive_optimization.py

ADAPTIVE OPTIMIZATION PIPELINE
Learning → Understanding → Modifying → Improving
================================================================================

📊 Checking Data Availability...
✅ BTC: 86,400 data points available
✅ ETH: 43,200 data points available
✅ TON: 28,800 data points available

🔍 Validating Concepts for TON...
Divergence Detection F1: 0.15
OI Confirmation F1: 0.65

🚨 Critical Issue Detected: Hardcoded threshold blocks TON

🔧 Modifying Code to Fix Issues...
✅ Modified phase2_divergence.py:230
   Old: min_change_threshold = 1e6
   New: min_change_threshold = self._get_dynamic_threshold(spot_cvd, futures_cvd)

Testing modification...
✅ Success - 12 trades generated (was 0)

📊 Report saved to: optimization_reports/cycle_report_20250810_230500.txt
```

## 🔍 What Gets Modified

### Current Issues It Can Fix

1. **Hardcoded Thresholds**
   - Problem: `min_threshold = 1e6` blocks low-volume symbols
   - Fix: Dynamic calculation based on actual data

2. **Missing Conditions**
   - Problem: No volume validation in divergence
   - Fix: Add volume significance check

3. **Wrong Operators**
   - Problem: Comparison inverted (> should be <)
   - Fix: Correct the operator

### Future Capabilities
The framework is designed to learn and fix any logical issue:
- Timing problems
- Order of operations
- Missing error handling
- Inefficient calculations

## 📈 Continuous Improvement

Each session builds on the last:

### Session 1
"TON can't trade → hardcoded threshold → fix with dynamic calc"

### Session 2  
"Dynamic calc works but too sensitive in volatile markets → add volatility scaling"

### Session 3
"Volatility scaling works → apply to all symbols → create universal principle"

## 🛡️ Safety & Rollback

Every modification is reversible:
```python
# If a change doesn't help:
1. Backtest shows no improvement
2. Automatically rollback from backup
3. Try different approach
4. Learn what didn't work
```

## 📝 For Future Claude Sessions

When you return:

```bash
# 1. Check current status
python3 experiments/adaptive_learner.py

# 2. See what was learned
cat adaptive_learning/learning_journal.json

# 3. Continue optimization
python3 experiments/run_adaptive_optimization.py

# 4. Review modifications
ls code_modifications/modification_history.json
```

The framework will:
- Show you what was learned
- Tell you what questions remain
- Suggest what to test next
- Continue from exactly where we left off

## ⚠️ Important Notes

### Data Requirements
- Must use remote InfluxDB: `INFLUX_HOST=213.136.75.120`
- Check data availability first
- ETH data starts 2025-08-10

### Code Modifications
- Always backs up before changing
- Only modifies strategy files
- Documents changes in SYSTEM_TRUTH.md
- Tests with real backtests

### Learning Persistence
- All learning saved in JSON
- Questions tracked and prioritized
- Principles emerge from patterns
- Can resume anytime

## 🎯 The Goal

**Not**: Find parameters that worked in the past
**But**: Understand WHY things work and fix what doesn't

The framework discovers **invariant properties** (TON is low-volume) rather than **temporary settings** (score=3.5 worked on Tuesday).

## 🚀 Ready to Start?

```bash
python3 experiments/run_adaptive_optimization.py
```

Watch as the system:
1. Learns what's broken
2. Fixes it in the code
3. Tests the fix
4. Remembers what worked
5. Applies learnings broadly

This is true adaptive optimization - the strategy improves itself based on understanding, not curve-fitting.