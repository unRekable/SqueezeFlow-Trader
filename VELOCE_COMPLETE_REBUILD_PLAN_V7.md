# 🚀 Veloce Complete Rebuild Plan V7 - CLAUDE-OPTIMIZED IMPLEMENTATION
> Built specifically for how Claude needs to understand and build systems, addressing all architectural issues

## Executive Summary
Veloce V7 focuses on building a system that Claude can fully understand and maintain. This plan starts with deep vulnerability understanding, builds incrementally with continuous validation, and creates an elegant architecture that's self-documenting and debuggable.

## Core Philosophy
- **Claude-First Design**: Built for how Claude understands systems
- **Elegant Simplicity**: Complex enough to work, simple enough to debug
- **Continuous Validation**: Test assumptions at every step
- **No Hidden Magic**: Everything explicit and traceable

---

# 🔍 PHASE 0: DEEP VULNERABILITY UNDERSTANDING (Week 1)
> Before building anything, understand EXACTLY what breaks and why

## Day 1-2: Cascade Chain Mapping
```python
# Map every "change X, break Y" pattern in SqueezeFlow

CASCADE_VULNERABILITIES = {
    "config_cascade": {
        "chain": ["indicator_config.py", "pipeline.py", "phase*.py", "visualizer.py"],
        "symptom": "Change OI threshold in one place, miss others",
        "fix_in_veloce": "Single VeloceConfig source"
    },
    "data_cascade": {
        "chain": ["influx_client", "pipeline", "strategy", "backtest"],
        "symptom": "Data format change breaks downstream",
        "fix_in_veloce": "Typed data contracts"
    },
    "visualization_cascade": {
        "chain": ["backtest_results", "visualizer", "tradingview", "html"],
        "symptom": "Missing field causes empty charts",
        "fix_in_veloce": "Required field validation"
    }
}
```

## Day 3-4: Circular Dependency Detection
```python
# Find all A→B→A patterns

CIRCULAR_DEPENDENCIES = {
    "strategy_pipeline": {
        "cycle": "Strategy needs Pipeline, Pipeline needs Strategy config",
        "current_workaround": "Import at runtime",
        "veloce_solution": "Dependency injection"
    },
    "config_components": {
        "cycle": "Config imports components, components import config",
        "current_workaround": "Global singleton",
        "veloce_solution": "Config passed down, never imported up"
    }
}
```

## Day 5: Active Component Identification
```python
# What ACTUALLY gets used (answer: surprisingly little)

ACTIVE_COMPONENTS = {
    "visualizer": "tradingview_multi_pane.py (via visualizer.py wrapper)",
    "data_source": "Remote InfluxDB at 213.136.75.120",
    "strategy": "SqueezeFlow 5-phase only",
    "pairs": ["BTC", "ETH", "AVAX", "ARB", "TON", "Others on server"],
    "timeframes": ["1s", "1m", "5m", "15m", "30m", "1h", "4h"]
}

DEAD_CODE = {
    "visualizers": [
        "complete_visualizer.py",
        "enhanced_visualizer.py", 
        "simple_visualizer.py",
        "single_chart_visualizer.py",
        # ... 10+ others
    ],
    "experiments": "entire experiments/ folder",
    "unused_strategies": "Everything except SqueezeFlow"
}
```

---

# 🏗️ PHASE 1: CLAUDE-OPTIMIZED ARCHITECTURE (Week 2)
> Design specifically for Claude's comprehension patterns

## Core Design Principles

### 1. **Explicit Data Flow**
```python
# BAD (SqueezeFlow): Hidden data paths
class Strategy:
    def get_data(self):
        # Sometimes from pipeline, sometimes direct query
        # Claude can't track this
        
# GOOD (Veloce): Single explicit path
class VeloceEngine:
    def data_flow(self):
        """
        Data flow is ALWAYS:
        InfluxDB → DataProvider → Engine → Strategy → Signals
        
        Never any other path.
        """
        return self.data_provider.get_data()  # ONLY way
```

### 2. **No Import Magic**
```python
# BAD: Circular imports, runtime imports
from strategies import *  # Claude loses track

# GOOD: Explicit, directional imports
from veloce.core.types import Signal
from veloce.data.provider import DataProvider
# Never import "up" the tree
```

### 3. **Self-Documenting Structure**
```python
@dataclass
class Signal:
    """
    A trading signal. ALL signals look like this.
    No variations, no special cases.
    """
    symbol: str          # Always present
    side: SignalSide     # Always LONG or SHORT
    score: float         # Always 0.0 to 10.0
    timestamp: datetime  # Always UTC
    reason: str          # Always explains why
```

---

# 📊 PHASE 2: DATA LAYER WITH VALIDATION (Week 3)
> Build data access that can't fail silently

## Single Data Provider
```python
class VeloceDataProvider:
    """
    THE ONLY way to get data. No alternatives.
    If you need data, you call this. Period.
    """
    
    def __init__(self, config: VeloceConfig):
        self.influx_host = "213.136.75.120"  # ALWAYS remote
        self.client = self._create_validated_client()
        
    def get_data(self, symbol: str, timeframe: str, 
                 start: datetime, end: datetime) -> pd.DataFrame:
        """
        Returns data or raises explicit exception.
        Never returns empty/invalid data silently.
        """
        
        # Validate inputs
        self._validate_symbol(symbol)  # Must be in known symbols
        self._validate_timeframe(timeframe)  # Must be valid
        self._validate_dates(start, end)  # Must be reasonable
        
        # Query with explicit error handling
        try:
            data = self._query_influx(symbol, timeframe, start, end)
        except Exception as e:
            raise DataProviderError(
                f"Failed to get {symbol} {timeframe} data: {e}\n"
                f"Check: Is InfluxDB running? Is symbol valid?"
            )
        
        # Validate output
        if data.empty:
            raise DataProviderError(
                f"No data for {symbol} from {start} to {end}\n"
                f"This usually means the date range has no data"
            )
            
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required) - set(data.columns)
        if missing:
            raise DataProviderError(
                f"Data missing columns: {missing}\n"
                f"This means the data format changed"
            )
            
        return data
```

## Market Discovery with Logging
```python
class MarketDiscovery:
    """
    Finds what markets exist for a symbol.
    Logs everything for Claude to understand.
    """
    
    def discover_markets(self, symbol: str) -> Dict[str, List[str]]:
        """
        Returns markets grouped by type.
        Logs the discovery process.
        """
        
        logger.info(f"🔍 Discovering markets for {symbol}")
        
        # Query all markets
        markets = self._query_all_markets(symbol)
        logger.info(f"  Found {len(markets)} total markets")
        
        # Categorize
        categorized = {
            'spot': [],
            'futures': [],
            'perp': []
        }
        
        for market in markets:
            market_type = self._determine_type(market)
            categorized[market_type].append(market)
            logger.debug(f"  {market} → {market_type}")
            
        logger.info(f"  Categorized: {len(categorized['spot'])} spot, "
                   f"{len(categorized['futures'])} futures, "
                   f"{len(categorized['perp'])} perp")
        
        return categorized
```

---

# 🧠 PHASE 3: STRATEGY WITH CLEAR PHASES (Week 4)
> Implement 5-phase strategy with explicit state transitions

## Strategy State Machine
```python
class SqueezeFlowStrategy:
    """
    5-phase strategy with EXPLICIT state tracking.
    Claude can always see what phase we're in and why.
    """
    
    def analyze(self, data: pd.DataFrame, timestamp: datetime) -> Optional[Signal]:
        """
        Process through phases with logging at each step.
        """
        
        # Phase 1: Context Assessment
        logger.info("📊 Phase 1: Context Assessment")
        context = self.phase1_context(data)
        logger.info(f"  Squeeze: {context['squeeze_detected']}")
        logger.info(f"  Environment: {context['market_regime']}")
        
        if not context['squeeze_detected']:
            logger.info("  ❌ No squeeze, stopping")
            return None
            
        # Phase 2: Divergence Detection
        logger.info("🔍 Phase 2: Divergence Detection")
        divergence = self.phase2_divergence(data, context)
        logger.info(f"  Type: {divergence['type']}")
        logger.info(f"  Strength: {divergence['strength']:.2f}")
        
        if divergence['type'] == 'NONE':
            logger.info("  ❌ No divergence, stopping")
            return None
            
        # Phase 3: Reset Detection
        logger.info("🎯 Phase 3: Reset Detection")
        reset = self.phase3_reset(data, divergence)
        logger.info(f"  Reset detected: {reset['detected']}")
        logger.info(f"  Type: {reset['type']}")
        
        if not reset['detected']:
            logger.info("  ❌ No reset, stopping")
            return None
            
        # Phase 4: Scoring
        logger.info("💯 Phase 4: Scoring")
        score = self.phase4_score(context, divergence, reset)
        logger.info(f"  Total score: {score['total']:.2f}")
        logger.info(f"  Components: {score['breakdown']}")
        
        if score['total'] < 4.0:
            logger.info(f"  ❌ Score {score['total']:.2f} < 4.0, stopping")
            return None
            
        # Phase 5: Signal Generation
        logger.info("✅ Phase 5: Signal Generation")
        signal = self.phase5_signal(divergence, score, timestamp)
        logger.info(f"  Signal: {signal.side} @ score {signal.score:.2f}")
        
        return signal
```

---

# 🧪 PHASE 4: INCREMENTAL TESTING (Week 5)
> Build and test one component at a time

## Test-First Development
```python
# Test BEFORE implementing each component

def test_data_provider():
    """Test data provider in isolation"""
    
    config = VeloceConfig()
    provider = VeloceDataProvider(config)
    
    # Test 1: Valid data request
    data = provider.get_data("BTC", "1m", 
                            datetime(2025, 1, 10), 
                            datetime(2025, 1, 10, 1, 0))
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close'])
    print("✅ Valid data request works")
    
    # Test 2: Invalid symbol
    try:
        provider.get_data("INVALID", "1m", datetime.now(), datetime.now())
        assert False, "Should have raised error"
    except DataProviderError as e:
        assert "symbol" in str(e).lower()
        print("✅ Invalid symbol raises clear error")
    
    # Test 3: No data range
    try:
        provider.get_data("BTC", "1m", 
                         datetime(2020, 1, 1),  # Too old
                         datetime(2020, 1, 2))
    except DataProviderError as e:
        assert "no data" in str(e).lower()
        print("✅ No data range raises clear error")
```

## Progressive Integration
```python
# Build up complexity gradually

# Step 1: Data provider alone
test_data_provider()

# Step 2: Data + Strategy
test_strategy_with_data()

# Step 3: Data + Strategy + Portfolio
test_portfolio_management()

# Step 4: Full backtest
test_complete_backtest()

# Each step MUST pass before moving on
```

---

# 🎯 PHASE 5: BACKTEST ENGINE (Week 6)
> Simple, understandable backtest that processes every tick

## Clear Execution Model
```python
class BacktestEngine:
    """
    Processes every tick sequentially.
    No magic, no shortcuts.
    """
    
    def run(self, symbol: str, start: datetime, end: datetime):
        """
        Simple loop through time.
        """
        
        logger.info(f"Starting backtest: {symbol} from {start} to {end}")
        
        # Load data once
        data = self.data_provider.get_data(symbol, "1s", start, end)
        logger.info(f"Loaded {len(data)} ticks")
        
        # Process each tick
        for timestamp, tick in data.iterrows():
            
            # Update portfolio with current prices
            self.portfolio.update_prices(tick)
            
            # Check exits on open positions
            for position in self.portfolio.open_positions:
                if self.should_exit(position, tick):
                    self.portfolio.close_position(position, tick)
                    logger.info(f"Closed {position.id} at {tick['close']}")
            
            # Look for new signals
            signal = self.strategy.analyze(
                data[:timestamp],  # All data up to now
                timestamp
            )
            
            if signal:
                position = self.portfolio.open_position(signal, tick)
                logger.info(f"Opened {position.id} at {tick['close']}")
            
            # Track equity
            self.equity_curve.append(self.portfolio.total_value)
        
        return self.generate_report()
```

---

# 📈 PHASE 6: VISUALIZATION (Week 7)
> One dashboard that actually works

## Single Dashboard Implementation
```python
class VeloceDashboard:
    """
    ONE dashboard. No alternatives.
    Uses TradingView Lightweight Charts.
    """
    
    def create_dashboard(self, backtest_results: BacktestResults) -> str:
        """
        Creates dashboard.html in the results directory.
        """
        
        # Validate inputs
        self._validate_results(backtest_results)
        
        # Create chart data
        chart_data = {
            'ohlcv': self._format_ohlcv(backtest_results.ohlcv),
            'trades': self._format_trades(backtest_results.trades),
            'indicators': self._format_indicators(backtest_results.indicators)
        }
        
        # Generate HTML
        html = self._render_template(chart_data)
        
        # Save and return path
        path = backtest_results.output_dir / "dashboard.html"
        path.write_text(html)
        
        logger.info(f"Dashboard created: {path}")
        return str(path)
```

---

# 🛡️ PHASE 7: SAFETY SYSTEMS (Week 8)
> Simple safety that actually saves the day

## Crash Recovery
```python
class SimpleCrashRecovery:
    """
    If system crashes, close all positions.
    Simple, effective, no complexity.
    """
    
    def __init__(self, freqtrade_client):
        self.freqtrade = freqtrade_client
        self.heartbeat_file = Path("/tmp/veloce_heartbeat")
        
    def start(self):
        """Write heartbeat every second"""
        while True:
            self.heartbeat_file.write_text(str(time.time()))
            time.sleep(1)
            
    def check_and_recover(self):
        """Check if heartbeat is stale"""
        if not self.heartbeat_file.exists():
            return
            
        last_heartbeat = float(self.heartbeat_file.read_text())
        if time.time() - last_heartbeat > 5:
            logger.critical("CRASH DETECTED - Closing all positions")
            self.close_all_positions()
            
    def close_all_positions(self):
        """Emergency close"""
        positions = self.freqtrade.get_positions()
        for pos in positions:
            self.freqtrade.close(pos['id'])
            logger.info(f"Emergency closed: {pos['id']}")
```

---

# 🔍 VALIDATION THROUGHOUT

## After Each Phase
```python
def validate_phase(phase_num: int) -> bool:
    """
    Validate each phase meets criteria before moving on.
    """
    
    validations = {
        1: validate_no_circular_dependencies,
        2: validate_single_data_source,
        3: validate_strategy_phases,
        4: validate_test_coverage,
        5: validate_backtest_accuracy,
        6: validate_dashboard_renders,
        7: validate_safety_systems
    }
    
    validator = validations[phase_num]
    result = validator()
    
    if not result.passed:
        logger.error(f"Phase {phase_num} validation failed: {result.reason}")
        logger.error(f"Fix required: {result.fix_suggestion}")
        return False
        
    logger.info(f"✅ Phase {phase_num} validation passed")
    return True
```

---

# 🎯 SUCCESS CRITERIA

## What Makes Veloce "Right"

1. **Claude Can Understand It**
   - Clear data flow
   - No hidden magic
   - Explicit error messages
   - Comprehensive logging

2. **No Cascade Failures**
   - Change one thing, only one thing changes
   - No configuration cascade
   - No circular dependencies

3. **Elegant Simplicity**
   - Each component does ONE thing
   - No duplicate implementations
   - Clear interfaces

4. **Fast Debugging**
   - Errors point to exact problem
   - Logs show complete flow
   - State is always visible

---

# 📋 WHAT WE'RE NOT BUILDING

Based on dead code analysis:
- ❌ 13 unused visualizers
- ❌ Experimental optimization code
- ❌ Alternative strategies
- ❌ Local InfluxDB support
- ❌ Complex authentication
- ❌ Multi-user support

Just what works, nothing more.

---

# 🔑 KEY DIFFERENCES FROM V6

1. **Starts with vulnerability understanding** (Week 1)
2. **Claude-optimized architecture** (explicit, no magic)
3. **Continuous validation** (after each phase)
4. **Dead code identified** (don't build what's not used)
5. **Clear logging throughout** (Claude can trace everything)
6. **Simple safety systems** (elegant, not complex)
7. **No timeline pressure** (takes what it takes)

---

# 💡 FINAL NOTES

This plan is designed specifically for how Claude understands systems:
- Explicit over implicit
- Logged operations over silent magic
- Clear error messages over cryptic failures
- Single paths over multiple options
- Validation over assumptions

The result will be a system that:
- You can trust
- Claude can maintain
- Actually works
- Is genuinely simpler

**Ready to build Veloce the RIGHT way.**