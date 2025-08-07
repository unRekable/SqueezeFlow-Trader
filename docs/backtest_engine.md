# SqueezeFlow Trader Backtest Engine Documentation

## Overview

The SqueezeFlow Trader backtest engine is a sophisticated orchestration system that coordinates trading strategy execution with comprehensive logging, visualization, and CVD baseline tracking. The engine implements clean separation of concerns, delegating all trading decisions to the strategy while managing execution, portfolio tracking, CVD baselines, and result analysis.

## Architecture

### Core Components

```mermaid
graph TD
    A["/backtest/"]
    A --> B["engine.py<br/>(Main orchestration engine with CVD baseline support)"]
    A --> C["core/"]
    A --> D["reporting/"]
    A --> E["results/"]
    
    C --> C1["portfolio.py<br/>(Portfolio with position tracking - trade_id, signal_id)"]
    C --> C2["fake_redis.py<br/>(In-memory Redis for backtest CVD baselines)"]
    C --> C3["__init__.py<br/>(Package initialization)"]
    
    D --> D1["logger.py<br/>(Multi-channel logging system)"]
    D --> D2["visualizer.py<br/>(Chart coordination and generation)"]
    D --> D3["html_reporter.py<br/>(Interactive HTML report generation)"]
    D --> D4["png_plotter.py<br/>(Static PNG chart generation)"]
    D --> D5["__init__.py<br/>(Package initialization)"]
    
    E --> E1["logs/<br/>(Timestamped log files)"]
    E --> E2["charts/<br/>(Generated visualizations)"]
```

### Integration Points

The backtest engine integrates seamlessly with the production system:

#### Data Pipeline Integration
- **Market Data**: Uses `data/pipeline.py` for complete dataset loading
- **Symbol Discovery**: Dynamic symbol detection via `data/loaders/symbol_discovery.py`
- **Market Classification**: Exchange mapping via `data/processors/exchange_mapper.py`
- **CVD Calculation**: Industry-standard cumulative volume delta

#### Strategy Integration
- **Modular Strategies**: `/strategies/squeezeflow/` with 5-phase methodology
- **CVD Baseline Manager**: `strategies/squeezeflow/baseline_manager.py` for position tracking
- **Configuration**: `strategies/squeezeflow/config.py` for scoring weights

#### Position Tracking Integration
- **Trade IDs**: Automatic generation for backtest positions
- **Signal IDs**: Correlation between signals and trades
- **CVD Baselines**: Stored at position entry for Phase 5 exits

## Enhanced Execution Flow

### 1. Initialization Phase
```python
# Create engine with CVD baseline support
engine = BacktestEngine(initial_balance=10000, leverage=1.0)

# Engine automatically initializes:
# - FakeRedis for CVD baseline storage
# - CVDBaselineManager for position tracking
# - Portfolio with enhanced position fields
# - Multi-channel logging system
```

### 2. Data Loading Phase
```python
# Load complete dataset through pipeline
dataset = data_pipeline.get_complete_dataset(
    symbol="BTCUSDT",           # Trading pair
    start_time=start,           # Start timestamp
    end_time=end,              # End timestamp
    timeframe="5m"             # Primary timeframe
)

# Dataset includes:
# - OHLCV data from multiple exchanges
# - Pre-calculated spot CVD (cumulative)
# - Pre-calculated futures CVD (cumulative)
# - Market metadata (exchanges, coverage)
# - Data quality metrics
```

### 3. Strategy Processing Phase
```python
# Strategy receives dataset and portfolio state
strategy_result = strategy.process(dataset, portfolio_state)

# Strategy can access:
# - CVDBaselineManager via set_cvd_baseline_manager()
# - Leverage settings via set_leverage()
# - Complete market data with CVD
# - Current portfolio positions

# Strategy returns:
# - Orders with CVD values for baseline storage
# - Signal metadata for tracking
# - Debug information for logging
```

### 4. Order Execution with CVD Baseline Storage
```python
# For each order from strategy:
order = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'quantity': 0.001,
    'price': 50000,
    'spot_cvd': 1234567,      # Current spot CVD
    'futures_cvd': 2345678,   # Current futures CVD
    'signal_id': 'sig_123',   # Signal identifier
    'leverage': 3             # Position leverage
}

# Engine executes order:
# 1. Generates trade_id
# 2. Stores CVD baseline in FakeRedis
# 3. Opens position with tracking info
# 4. Links signal_id to trade_id
```

### 5. CVD Baseline Tracking
```python
# CVDBaselineManager stores at entry:
baseline = {
    'signal_id': 'sig_123',
    'trade_id': 1,
    'symbol': 'BTCUSDT',
    'side': 'buy',
    'entry_price': 50000,
    'spot_cvd': 1234567,
    'futures_cvd': 2345678,
    'timestamp': '2024-08-01T12:00:00Z'
}

# Phase 5 exits can retrieve baseline:
baseline = cvd_baseline_manager.get_baseline_by_trade(trade_id=1)
flow_change = current_cvd - baseline['spot_cvd']
```

### 6. Result Generation
```python
# Comprehensive results include:
result = {
    'symbol': 'BTCUSDT',
    'initial_balance': 10000,
    'final_balance': 12500,
    'total_return': 25.0,
    'total_trades': 45,
    'win_rate': 62.2,
    'executed_orders': [...],      # All executed orders
    'portfolio_state': {...},      # Final portfolio state
    'cvd_baselines': {...},        # All CVD baselines
    'visualization_path': '...'    # Generated reports
}
```

## Multi-Channel Logging System

### Logging Structure
```mermaid
graph TD
    A["backtest/results/"]
    A --> B["logs/"]
    A --> C["charts/"]
    
    B --> B1["backtest_20240801_120000.log<br/>(Main execution log)"]
    B --> B2["trades_20240801_120000.csv<br/>(Trade details CSV)"]
    B --> B3["signals_20240801_120000.csv<br/>(Signal analysis CSV)"]
    B --> B4["summary_20240801_120000.json<br/>(Performance summary)"]
    
    C --> C1["report_20240801_120000/"]
    
    C1 --> C2["report.html<br/>(Interactive HTML report)"]
    C1 --> C3["equity_curve.png<br/>(Portfolio performance)"]
    C1 --> C4["cvd_analysis.png<br/>(CVD divergence charts)"]
    C1 --> C5["price_signals.png<br/>(Price with entry/exit)"]
    C1 --> C6["performance.png<br/>(Metrics dashboard)"]
    C1 --> C7["trades.png<br/>(Trade distribution)"]
```

### Logging Channels

#### 1. Main Application Log
- Strategy decisions and reasoning
- CVD calculations and divergence
- Signal generation with scores
- Order execution details
- Performance metrics

#### 2. Trade CSV Log
```csv
timestamp,symbol,side,quantity,price,leverage,pnl,total_value,trade_id,signal_id
2024-08-01 12:00:00,BTCUSDT,BUY,0.001,50000,3,0,10000,1,sig_123
2024-08-01 13:00:00,BTCUSDT,SELL,0.001,51000,3,50,10050,1,sig_123
```

#### 3. Signal CSV Log
```csv
timestamp,symbol,signal_type,score,spot_cvd,futures_cvd,divergence,action
2024-08-01 12:00:00,BTCUSDT,LONG_SQUEEZE,7.5,1234567,2345678,-1111111,ENTRY
2024-08-01 13:00:00,BTCUSDT,FLOW_EXIT,N/A,1345678,2456789,-1111111,EXIT
```

## Visualization System

### Generated Reports

#### 1. HTML Interactive Report
- Complete trade history table
- Sortable and filterable data
- Performance metrics dashboard
- Strategy configuration display
- Session metadata

#### 2. PNG Chart Suite
- **Equity Curve**: Portfolio value over time with drawdown
- **CVD Analysis**: Spot vs Futures CVD with divergence
- **Price Signals**: Entry/exit points on price chart
- **Performance Dashboard**: Key metrics visualization
- **Trade Distribution**: Win/loss analysis

### Visualization Features
- Multi-panel synchronized charts
- Annotated entry/exit points
- CVD divergence highlighting
- Performance metric overlays
- Professional styling with grid

## Performance Optimization

### Memory Management
- Streaming data processing for large datasets
- Efficient pandas operations with copy control
- Garbage collection after large operations
- Limited chart data points for visualization

### Execution Speed
- Vectorized CVD calculations
- Batch order processing
- Optimized portfolio updates
- Parallel visualization generation

### Data Efficiency
- Resampling for visualization only
- Selective column loading
- Index-based operations
- Memory-mapped file options

## Configuration

### Engine Configuration
```python
# Initialize with custom settings
engine = BacktestEngine(
    initial_balance=10000,      # Starting capital
    leverage=3.0                # Maximum leverage
)

# Run with parameters
result = engine.run(
    strategy=SqueezeFlowStrategy(),
    symbol='BTCUSDT',
    start_date='2024-08-01',
    end_date='2024-08-04',
    timeframe='5m',
    balance=10000,              # Override initial balance
    leverage=3.0                # Override leverage
)
```

### Strategy Configuration
```python
# Strategy receives configuration
class SqueezeFlowStrategy:
    def __init__(self):
        self.config = SqueezeFlowConfig()
        self.cvd_baseline_manager = None
        
    def set_cvd_baseline_manager(self, manager):
        self.cvd_baseline_manager = manager
        
    def set_leverage(self, leverage):
        self.leverage = leverage
```

## Usage Examples

### Basic Backtest
```bash
# Run from command line
python backtest/engine.py --symbol BTCUSDT --start-date 2024-08-01 --end-date 2024-08-04

# Or import and use
from backtest.engine import BacktestEngine
from strategies.squeezeflow.strategy import SqueezeFlowStrategy

engine = BacktestEngine()
result = engine.run(
    strategy=SqueezeFlowStrategy(),
    symbol='BTCUSDT',
    start_date='2024-08-01',
    end_date='2024-08-04'
)
```

### Advanced Usage with CVD Baselines
```python
# Strategy with CVD baseline tracking
strategy = SqueezeFlowStrategy()
engine = BacktestEngine(initial_balance=50000, leverage=5)

# Engine automatically sets up CVD baseline manager
result = engine.run(strategy, 'BTCUSDT', '2024-08-01', '2024-08-04')

# Access CVD baselines from results
for trade_id, baseline in result.get('cvd_baselines', {}).items():
    print(f"Trade {trade_id}: Entry CVD = {baseline['spot_cvd']}")
```

### Custom Strategy Integration
```python
from strategies.base import BaseStrategy

class CustomStrategy(BaseStrategy):
    def process(self, dataset, portfolio_state):
        # Access CVD baseline manager if available
        if self.cvd_baseline_manager:
            # Use for exit decisions
            for position in portfolio_state['positions']:
                baseline = self.cvd_baseline_manager.get_baseline_by_trade(
                    position['trade_id']
                )
                # Calculate flow change since entry
                
        # Return orders with CVD values
        return {
            'orders': [{
                'symbol': dataset['symbol'],
                'side': 'BUY',
                'quantity': 0.001,
                'price': current_price,
                'spot_cvd': dataset['spot_cvd'].iloc[-1],
                'futures_cvd': dataset['futures_cvd'].iloc[-1]
            }]
        }
```

## Integration with Live Trading

The backtest engine uses the same strategy code as live trading:

### Shared Components
- **Strategy**: `/strategies/squeezeflow/` (identical code)
- **CVD Calculation**: Same industry-standard methodology
- **Scoring System**: Identical 10-point scoring
- **Risk Management**: Same position sizing logic

### Differences from Live
| Component | Backtest | Live Trading |
|-----------|----------|--------------|
| Data Source | Historical from InfluxDB | Real-time from InfluxDB |
| CVD Storage | FakeRedis (in-memory) | Redis (persistent) |
| Execution | Simulated with fees | FreqTrade API |
| Latency | None | Network latency |
| Slippage | Configurable simulation | Market reality |

## Troubleshooting

### Common Issues

#### 1. No Data Found
```python
# Check data availability
from data.loaders.symbol_discovery import SymbolDiscovery
discovery = SymbolDiscovery()
symbols = discovery.discover_symbols_from_database()
print(f"Available symbols: {symbols}")
```

#### 2. CVD Baseline Not Available
```python
# Ensure strategy has CVD baseline manager
if not strategy.cvd_baseline_manager:
    print("Warning: CVD baseline manager not set")
# Engine sets this automatically
```

#### 3. Memory Issues with Large Datasets
```python
# Use shorter time ranges or resample
result = engine.run(
    strategy=strategy,
    symbol='BTCUSDT',
    start_date='2024-08-01',
    end_date='2024-08-02',  # Shorter range
    timeframe='15m'          # Larger timeframe
)
```

## Performance Metrics

### Backtest Speed
- **Data Loading**: ~2-5 seconds for 1 week of 5m data
- **Strategy Processing**: ~1-3 seconds per 1000 candles
- **Visualization**: ~3-5 seconds for full report
- **Total Time**: ~10-15 seconds for typical backtest

### Resource Usage
- **Memory**: 200-500 MB for 1 week backtest
- **CPU**: Single-core for strategy, multi-core for visualization
- **Disk**: ~5-10 MB per backtest (logs + charts)

## Best Practices

### 1. Strategy Development
- Use logging extensively for debugging
- Store CVD values in orders for baseline tracking
- Include signal_id for signal-to-trade correlation
- Test with different market conditions

### 2. Performance Testing
- Run backtests over different time periods
- Test with varying initial balances
- Experiment with leverage settings
- Validate against live trading results

### 3. Result Analysis
- Review HTML reports for trade details
- Analyze CVD charts for strategy behavior
- Check signal CSV for pattern recognition
- Monitor equity curve for drawdowns

## Related Documentation

- [Signal Generation Workflow](/docs/signal_generation_workflow.md) - Live trading pipeline
- [CVD Baseline Tracking](/docs/cvd_baseline_tracking.md) - Baseline system details
- [SqueezeFlow Strategy](/docs/squeezeflow_strategy.md) - Trading methodology
- [Services Architecture](/docs/services_architecture.md) - System components

---

*Last Updated: August 2025 - Enhanced with CVD baseline tracking and complete integration*