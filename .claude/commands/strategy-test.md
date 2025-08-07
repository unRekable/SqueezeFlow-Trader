---
description: Test strategy components and validate implementation
argument-hint: "[component] [coverage]"
allowed-tools: ["Bash", "Read", "Write"]
---

# Strategy Test Command

Run comprehensive tests for SqueezeFlow strategy components.

## Usage Examples:
- `/strategy-test` - Run all strategy tests
- `/strategy-test cvd` - Test only CVD engine
- `/strategy-test phase1` - Test specific phase
- `/strategy-test all --coverage` - Run with coverage report

## Test Categories:
1. **CVD Engine**: Calculation accuracy and aggregation
2. **Strategy Phases**: Individual phase logic (1-5)
3. **Scoring System**: 10-point scoring validation
4. **Integration**: Backtest and FreqTrade adapters
5. **Data Pipeline**: Multi-timeframe loading
6. **Pattern Recognition**: Qualitative pattern detection

## Parameters:
- **component**: `all`, `cvd`, `phase1-5`, `scoring`, `integration`, `data`, `patterns`
- **coverage**: Add `--coverage` flag for detailed coverage report

## Output:
- Test results with pass/fail status
- Performance metrics for each component
- Coverage report (if requested)
- Recommendations for failed tests