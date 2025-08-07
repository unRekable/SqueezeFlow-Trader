# Changelog

All notable changes to SqueezeFlow Trader will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed - Rolling Window Backtest Implementation
- **MAJOR FIX**: Implemented rolling window processing in backtest engine to eliminate lookahead bias and fix reset detection
- **Processing Method**: Backtests now process data in 4-hour rolling windows, stepping forward 5 minutes at a time
- **Reset Detection Fix**: Phase 3 reset detection now works correctly as patterns develop over time sequentially
- **Live Trading Parity**: Backtest processing now exactly matches live trading behavior and data visibility
- **No Strategy Changes**: Existing SqueezeFlow strategy works unchanged with new processing method
- **Performance**: Maintains same execution speed while providing realistic results
- **Benefits**:
  - Eliminates lookahead bias completely
  - Fixes convergence exhaustion pattern detection
  - Ensures backtest results are predictive of live performance
  - Matches live trading data flow exactly

### Changed
- **BacktestEngine.run()**: Now uses rolling window processing by default
- **Documentation**: Updated all documentation to reflect rolling window implementation
  - README.md: Added rolling window explanation in backtest section
  - docs/backtest_engine.md: Comprehensive rolling window documentation
  - docs/system_overview.md: Updated architecture diagrams
  - docs/signal_generation_workflow.md: Added backtest vs live comparison

### Technical Details
- **Window Size**: 4 hours (matching live trading context window)
- **Step Size**: 5 minutes forward progression
- **Data Isolation**: Strategy only sees data up to current time (no future visibility)
- **Error Handling**: Graceful handling of insufficient data windows
- **Progress Tracking**: Logs progress every 100 iterations for long backtests
- **Memory Efficiency**: Processes one window at a time to minimize memory usage

This change represents a major improvement in backtest reliability and eliminates the "works in backtest but fails live" problem that affected reset detection.