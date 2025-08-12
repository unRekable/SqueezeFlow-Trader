# Experiments V2 - Practical Optimization System

This is the **clean, working optimization system** that actually connects to your trading infrastructure.

## 📁 What's Here

### Core System
- `insight_engine.py` - The main optimization engine that ACTUALLY WORKS
- `run_insight_engine.py` - Simple command-line runner
- `adaptive_learner.py` - Persistent learning across sessions

### Data Storage
- `insights/` - Stored analysis results from backtests
- `adaptive_learning/` - Persistent knowledge base

## 🚀 Quick Start

```bash
# 1. Check system health
python3 run_insight_engine.py --mode analyze

# 2. Run a test backtest
python3 run_insight_engine.py --mode test

# 3. Find optimal parameters
python3 run_insight_engine.py --mode optimize

# 4. Generate comprehensive report
python3 run_insight_engine.py --mode report
```

## 🎯 Key Features

### What It DOES:
- ✅ Runs REAL backtests (not mock data)
- ✅ Analyzes ACTUAL results
- ✅ Finds REAL patterns in trades
- ✅ Makes SAFE parameter changes (env vars only)
- ✅ Learns PERSISTENTLY across sessions
- ✅ Provides ACTIONABLE recommendations

### What It DOESN'T:
- ❌ No dangerous code modifications
- ❌ No fake AI or "Bayesian optimization"
- ❌ No 758 false positive "bugs"
- ❌ No complex abstractions
- ❌ No theoretical concepts

## 📊 How It Works

1. **Runs Real Backtests:**
   ```python
   subprocess.run(['python3', 'backtest/engine.py', ...])
   ```
   Actually executes your backtest engine with real data.

2. **Analyzes Results:**
   ```python
   metrics = engine._parse_output(backtest_output)
   ```
   Extracts metrics from actual backtest output.

3. **Finds Patterns:**
   ```python
   winning_patterns = analyze_successes(trades)
   losing_patterns = analyze_failures(trades)
   ```
   Identifies what works and what doesn't.

4. **Makes Recommendations:**
   ```python
   if total_trades < 5:
       recommend: lower MIN_ENTRY_SCORE
   if win_rate < 40%:
       recommend: raise MIN_ENTRY_SCORE
   ```
   Simple, logical recommendations based on results.

5. **Learns Persistently:**
   ```python
   learner.record_learning(symbol, concept, finding)
   ```
   Saves insights for future sessions.

## 🔧 Configuration

The system uses environment variables for all configuration:

```bash
export INFLUX_HOST=213.136.75.120  # Remote InfluxDB server
export SQUEEZEFLOW_MIN_ENTRY_SCORE=5.0  # Strategy parameter
```

## 📈 Example Output

```
🧪 TEST BACKTEST
==================================================
Running backtest for ETH on 2025-08-10...

📊 Results:
  Total trades: 15
  Win rate: 60.0%
  Total return: 5.20%
  Sharpe ratio: 1.80
  Max drawdown: -3.50%
  Quality score: 72.4/100

💡 Recommendations:
  - Entry threshold optimal for current conditions

🔧 Suggested parameter adjustments:
  MIN_ENTRY_SCORE = 5.0
```

## 🛡️ Safety Features

1. **No Code Modification** - Only uses environment variables
2. **Backup Everything** - All insights saved to JSON
3. **Timeout Protection** - Backtests timeout after 5 minutes
4. **Error Handling** - Graceful failure with logging
5. **Visual Validation** - Can capture dashboard screenshots

## 📚 Philosophy

This system follows the principle of **"Make it work, make it right, make it fast"**:

1. **It works** - Actually runs backtests and analyzes results
2. **It's right** - Uses your real infrastructure, not mock data
3. **It's fast enough** - Runs backtests in ~30 seconds

No over-engineering. No fake complexity. Just practical optimization.

## 🔄 Continuous Learning

The system maintains knowledge across sessions:

- `adaptive_learning/learning_journal.json` - What we've learned
- `adaptive_learning/discovered_principles.json` - Patterns that work
- `adaptive_learning/open_questions.json` - Things to investigate
- `adaptive_learning/next_steps.json` - Recommended actions

Any Claude session can pick up where the last one left off.

## ⚠️ Requirements

- Python 3.8+
- Access to remote InfluxDB (213.136.75.120)
- ETH data from 2025-08-10 onwards
- Environment variable: `INFLUX_HOST=213.136.75.120`

## 🎯 Why This Approach?

**Simple > Complex**
- One file that works is better than 10 files of theory

**Real > Theoretical**
- Actual backtests beat hypothetical optimization

**Safe > Powerful**
- Env vars are safe; code modification is dangerous

**Practical > Perfect**
- A working 80% solution beats a theoretical 100% solution

## 📞 Support

This system was designed to be simple enough that any developer (or Claude session) can understand and modify it. If something doesn't work:

1. Check `INFLUX_HOST` is set
2. Verify data exists for your date range
3. Look at the actual error message (they're descriptive)

---

**Remember:** This is the optimization system that ACTUALLY WORKS, not the one with 758 false positive "bugs" and 6 layers of abstraction.