# 🚀 Veloce Complete Rebuild Plan V8 - DEEP IMPLEMENTATION BLUEPRINT
> Full architectural understanding with concrete implementation details for Claude-optimized system

## Executive Summary
Veloce V8 is a complete rebuild that starts with deep dependency analysis of SqueezeFlow's failures, builds an elegant system Claude can maintain, and uses only remote production data for serious trading work. Every implementation detail is specified, every test uses real data, and every component has explicit error handling.

## Core Philosophy
- **Production-First**: Always use remote server data (213.136.75.120)
- **No Mocks Ever**: Real data, real queries, real results
- **Claude-Optimized**: Built for AI comprehension and maintenance
- **Explicit Everything**: No hidden magic, no implicit behavior
- **Continuous Validation**: Test with real data at every step

---

# 🔍 PHASE 0: DEEP DEPENDENCY ANALYSIS (Week 1)
> Complete understanding of SqueezeFlow's architecture and failure modes

## Day 1-2: Complete Dependency Mapping Script
```python
# dependency_analyzer.py - Run this to understand EVERYTHING
import ast
import json
from pathlib import Path
from typing import Dict, List, Set

class DependencyAnalyzer:
    """
    Deep analysis of SqueezeFlow to understand all connections.
    This will be our bible for what NOT to do in Veloce.
    """
    
    def __init__(self):
        self.project_root = Path("/Users/u/PycharmProjects/SqueezeFlow Trader")
        self.dependencies = {}
        self.circular_refs = []
        self.cascade_chains = []
        self.dead_code = []
        
    def analyze_complete_system(self):
        """Main analysis entry point"""
        
        print("🔍 Phase 1: Mapping all imports...")
        self.map_all_imports()
        
        print("🔄 Phase 2: Finding circular dependencies...")
        self.find_circular_dependencies()
        
        print("⛓️ Phase 3: Identifying cascade chains...")
        self.identify_cascade_chains()
        
        print("💀 Phase 4: Finding dead code...")
        self.find_dead_code()
        
        print("🧠 Phase 5: Analyzing strategy phases...")
        self.analyze_strategy_phases()
        
        print("📊 Phase 6: Tracing data flow paths...")
        self.trace_data_flows()
        
        print("🎨 Phase 7: Mapping visualizer chaos...")
        self.map_visualizer_implementations()
        
        # Save complete analysis
        self.save_analysis_report()
        
    def map_all_imports(self):
        """Build complete import graph"""
        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            with open(py_file) as f:
                try:
                    tree = ast.parse(f.read())
                    imports = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                imports.append(name.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                                
                    self.dependencies[str(py_file.relative_to(self.project_root))] = imports
                except:
                    pass  # Skip unparseable files
                    
    def find_circular_dependencies(self):
        """Detect A→B→A patterns"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node, path=[]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    self.circular_refs.append(cycle)
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for node in self.dependencies:
            if node not in visited:
                has_cycle(node)
                
    def identify_cascade_chains(self):
        """Find components that break when others change"""
        
        # Critical cascade points in SqueezeFlow
        cascade_patterns = {
            "config_cascade": {
                "trigger": "indicator_config.py",
                "affected": [],
                "pattern": "Change config → break strategy phases"
            },
            "data_cascade": {
                "trigger": "influx_client.py", 
                "affected": [],
                "pattern": "Change data format → break everything downstream"
            },
            "visualizer_cascade": {
                "trigger": "visualizer.py",
                "affected": [],
                "pattern": "Change viz → multiple dashboards break"
            }
        }
        
        # Find all files affected by each trigger
        for cascade_type, info in cascade_patterns.items():
            trigger = info["trigger"]
            
            # BFS to find all dependent files
            queue = [trigger]
            visited = set()
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Find all files that import current
                for file, imports in self.dependencies.items():
                    if current.replace(".py", "") in str(imports):
                        info["affected"].append(file)
                        queue.append(file)
                        
            self.cascade_chains.append(info)
            
    def find_dead_code(self):
        """Identify unused code by checking execution paths"""
        
        # Start from entry points
        entry_points = [
            "backtest/engine.py",
            "services/strategy_runner.py",
            "docker-compose.yml"
        ]
        
        # Find all reachable code
        reachable = set()
        queue = entry_points.copy()
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            # Add all imports from this file
            for imported in self.dependencies.get(current, []):
                queue.append(imported)
                
        # Everything not reachable is dead
        all_files = set(self.dependencies.keys())
        self.dead_code = list(all_files - reachable)
        
    def analyze_strategy_phases(self):
        """Deep dive into 5-phase strategy dependencies"""
        
        strategy_phases = {
            "phase1": {
                "file": "strategies/squeezeflow/components/phase1_context.py",
                "depends_on": ["indicator_config.py", "influx_client.py"],
                "produces": ["squeeze_detected", "market_regime"],
                "breaks_if": ["BB calculation changes", "KC calculation changes"]
            },
            "phase2": {
                "file": "strategies/squeezeflow/components/phase2_divergence.py",
                "depends_on": ["phase1 output", "CVD data", "price data"],
                "produces": ["divergence_type", "divergence_strength"],
                "breaks_if": ["CVD format changes", "phase1 output changes"]
            },
            "phase3": {
                "file": "strategies/squeezeflow/components/phase3_reset.py",
                "depends_on": ["phase2 output", "raw CVD values"],
                "produces": ["reset_detected", "reset_type"],
                "breaks_if": ["CVD thresholds not configured", "phase2 changes"]
            },
            "phase4": {
                "file": "strategies/squeezeflow/components/phase4_scoring.py",
                "depends_on": ["phase1", "phase2", "phase3", "OI data"],
                "produces": ["total_score", "score_breakdown"],
                "breaks_if": ["Any prior phase changes", "OI unavailable"]
            },
            "phase5": {
                "file": "strategies/squeezeflow/components/phase5_signal.py",
                "depends_on": ["phase4 score", "risk management"],
                "produces": ["trading_signal"],
                "breaks_if": ["Score format changes", "position sizing changes"]
            }
        }
        
        return strategy_phases
        
    def trace_data_flows(self):
        """Map how data flows through the system"""
        
        data_flows = {
            "price_data_flow": [
                "aggr-server (websocket) → ",
                "InfluxDB (storage) → ",
                "influx_client.py (query) → ",
                "DataPipeline (transform) → ",
                "Strategy (analyze) → ",
                "Signal"
            ],
            "cvd_data_flow": [
                "aggr-server (calculate) → ",
                "InfluxDB (store as buy_volume/sell_volume) → ",
                "DataPipeline (compute CVD) → ",
                "Phase2/Phase3 (analyze divergence)"
            ],
            "oi_data_flow": [
                "OI Tracker (API calls) → ",
                "InfluxDB (store) → ",
                "oi_tracker_influx.py (query) → ",
                "Phase4 (scoring validation)"
            ]
        }
        
        return data_flows
        
    def map_visualizer_implementations(self):
        """Count and categorize all visualizer variants"""
        
        visualizers = []
        viz_dir = self.project_root / "backtest/reporting"
        
        for viz_file in viz_dir.glob("*visualizer*.py"):
            with open(viz_file) as f:
                content = f.read()
                
            info = {
                "file": viz_file.name,
                "uses_tradingview": "tradingview" in content.lower(),
                "creates_html": ".html" in content,
                "imported_by": [],
                "actually_used": False
            }
            
            # Check what imports this
            for file, imports in self.dependencies.items():
                if viz_file.stem in str(imports):
                    info["imported_by"].append(file)
                    
            # Check if used in engine.py
            if "engine.py" in str(info["imported_by"]):
                info["actually_used"] = True
                
            visualizers.append(info)
            
        return visualizers
        
    def save_analysis_report(self):
        """Save complete analysis to JSON"""
        
        report = {
            "timestamp": str(datetime.now()),
            "total_files": len(self.dependencies),
            "circular_dependencies": self.circular_refs,
            "cascade_chains": self.cascade_chains,
            "dead_code_files": len(self.dead_code),
            "dead_code_list": self.dead_code[:20],  # Top 20
            "strategy_phases": self.analyze_strategy_phases(),
            "data_flows": self.trace_data_flows(),
            "visualizers": self.map_visualizer_implementations()
        }
        
        with open("squeezeflow_dependency_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"📊 Analysis saved to squeezeflow_dependency_analysis.json")
        print(f"   - Found {len(self.circular_refs)} circular dependencies")
        print(f"   - Found {len(self.cascade_chains)} cascade chains")
        print(f"   - Found {len(self.dead_code)} dead files")
        
        return report

# Run the analysis
if __name__ == "__main__":
    analyzer = DependencyAnalyzer()
    analyzer.analyze_complete_system()
```

## Day 3-4: Lessons Extraction
```python
# Extract concrete lessons from the analysis
LESSONS_NOT_TO_REPEAT = {
    "cascade_vulnerability": {
        "problem": "Change indicator_config.py → 5 phase files break",
        "veloce_solution": "Phases never import config, config passed down"
    },
    "circular_imports": {
        "problem": "Strategy imports Pipeline, Pipeline imports Strategy config",
        "veloce_solution": "Strict directional imports: Core → Data → Strategy → Engine"
    },
    "multiple_visualizers": {
        "problem": "14 different visualizer files, unclear which is used",
        "veloce_solution": "ONE visualizer, no alternatives"
    },
    "data_path_confusion": {
        "problem": "Strategy uses DataPipeline AND direct InfluxDB queries",
        "veloce_solution": "ONE data provider, no direct queries allowed"
    },
    "implicit_dependencies": {
        "problem": "Phase3 expects CVD but doesn't validate it exists",
        "veloce_solution": "Every input explicitly validated with clear errors"
    },
    "config_duplication": {
        "problem": "Thresholds hardcoded in multiple places",
        "veloce_solution": "Config injected at startup, never imported"
    }
}
```

## Day 5: Validation Criteria Definition
```python
def validate_no_cascades():
    """Ensure changing one file doesn't break others"""
    # Test: Change config value
    # Assert: Only config file modified
    # Assert: System still runs
    pass

def validate_no_circular_deps():
    """Ensure no A→B→A import patterns"""
    # Parse all imports
    # Build dependency graph
    # Assert: Graph is acyclic
    pass

def validate_single_implementation():
    """Ensure no duplicate implementations"""
    # For each feature (visualizer, data source, etc)
    # Assert: Exactly one implementation exists
    pass
```

---

# 🏗️ PHASE 1: VELOCE ARCHITECTURE (Week 2)
> Build architecture that Claude can understand and maintain

## Core Architectural Rules

### 1. **Strict Import Hierarchy**
```python
"""
Import flow is ALWAYS downward:
    veloce/core/ (types, interfaces)
           ↓
    veloce/data/ (data provider)
           ↓  
    veloce/strategy/ (trading logic)
           ↓
    veloce/portfolio/ (position management)
           ↓
    veloce/engine/ (orchestration)
           ↓
    veloce/safety/ (monitoring)

NEVER import upward. NEVER circular. ALWAYS explicit.
"""
```

### 2. **Dependency Injection Pattern**
```python
# veloce/core/container.py
class VeloceContainer:
    """
    All dependencies created here, injected down.
    No component creates its own dependencies.
    """
    
    def __init__(self, config_path: str):
        # Load config once
        self.config = VeloceConfig.from_file(config_path)
        
        # Create all components with config
        self.data_provider = VeloceDataProvider(
            host="213.136.75.120",  # ALWAYS remote
            port=8086,
            database="significant_trades"
        )
        
        # Strategy gets config but never imports it
        self.strategy = SqueezeFlowStrategy(
            config=self.config.strategy_config
        )
        
        # Portfolio gets injected dependencies
        self.portfolio = Portfolio(
            config=self.config.portfolio_config,
            data_provider=self.data_provider
        )
        
        # Engine orchestrates with all injected
        self.engine = BacktestEngine(
            data_provider=self.data_provider,
            strategy=self.strategy,
            portfolio=self.portfolio,
            config=self.config.engine_config
        )
        
    def get_engine(self) -> BacktestEngine:
        """Returns fully configured engine ready to run"""
        return self.engine
```

### 3. **Explicit Type Contracts**
```python
# veloce/core/types.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
import pandas as pd

@dataclass(frozen=True)
class MarketData:
    """
    Immutable market data container.
    This is the ONLY way data moves through system.
    """
    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame  # Always has: open, high, low, close, volume
    cvd: Optional[pd.Series] = None  # May not exist
    oi: Optional[pd.Series] = None   # May not exist
    
    def __post_init__(self):
        """Validate on creation"""
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(self.ohlcv.columns):
            missing = required_cols - set(self.ohlcv.columns)
            raise ValueError(f"OHLCV missing columns: {missing}")

@dataclass(frozen=True)
class Signal:
    """
    Immutable trading signal.
    ALL signals MUST look exactly like this.
    """
    timestamp: datetime
    symbol: str
    side: Literal["LONG", "SHORT"]
    score: float  # 0.0 to 10.0
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: float
    take_profit: float
    
    def __post_init__(self):
        """Validate on creation"""
        if not 0 <= self.score <= 10:
            raise ValueError(f"Score {self.score} not in [0,10]")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence {self.confidence} not in [0,1]")
```

---

# 📊 PHASE 2: DATA LAYER (Week 3)
> Single source of truth for all data access

## The ONE Data Provider
```python
# veloce/data/provider.py
import logging
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from influxdb import InfluxDBClient

logger = logging.getLogger(__name__)

class VeloceDataProvider:
    """
    THE ONLY way to get data in Veloce.
    No alternatives. No exceptions. No direct queries.
    """
    
    def __init__(self, host: str = "213.136.75.120", port: int = 8086, 
                 database: str = "significant_trades"):
        """
        Always connects to remote production server.
        No local data. No mocks. Real data only.
        """
        self.host = host
        self.port = port
        self.database = database
        
        # Create client with validation
        try:
            self.client = InfluxDBClient(host=host, port=port, database=database)
            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to InfluxDB at {host}:{port}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to InfluxDB at {host}:{port}\n"
                f"Error: {e}\n"
                f"Check: Is server reachable? Is InfluxDB running?"
            )
            
        # Cache available symbols/markets for performance
        self._refresh_available_markets()
        
    def _refresh_available_markets(self):
        """Cache what markets exist to avoid repeated queries"""
        try:
            query = "SHOW SERIES FROM trades_1s"
            result = self.client.query(query)
            
            self.available_markets = set()
            for series in result.get_points():
                # Extract market from series key
                if 'market' in series:
                    self.available_markets.add(series['market'])
                    
            logger.info(f"📊 Found {len(self.available_markets)} available markets")
            
        except Exception as e:
            logger.error(f"Failed to refresh markets: {e}")
            self.available_markets = set()
            
    def get_ohlcv(self, symbol: str, timeframe: str, 
                  start: datetime, end: datetime) -> pd.DataFrame:
        """
        Get OHLCV data with explicit validation.
        Returns DataFrame or raises clear exception.
        """
        
        # Input validation
        if timeframe not in ['1s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']:
            raise ValueError(
                f"Invalid timeframe: {timeframe}\n"
                f"Valid: 1s, 1m, 5m, 15m, 30m, 1h, 4h, 1d"
            )
            
        # Build query based on timeframe
        if timeframe == '1s':
            measurement = "aggr_1s.trades_1s"
        else:
            measurement = f"trades_{timeframe}"
            
        market = f"BINANCE:{symbol.lower()}usdt"
        
        query = f"""
        SELECT mean(close) as close, max(high) as high, 
               min(low) as low, first(open) as open, 
               sum(volume) as volume
        FROM {measurement}
        WHERE market = '{market}' 
        AND time >= '{start.isoformat()}Z' 
        AND time <= '{end.isoformat()}Z'
        GROUP BY time({timeframe})
        """
        
        logger.debug(f"Querying: {symbol} {timeframe} from {start} to {end}")
        
        try:
            result = self.client.query(query)
            df = pd.DataFrame(result.get_points())
            
            if df.empty:
                raise DataNotFoundError(
                    f"No data for {symbol} from {start} to {end}\n"
                    f"This usually means:\n"
                    f"1. Date range has no data (check available dates)\n"
                    f"2. Symbol doesn't exist\n"
                    f"3. Timeframe not available"
                )
                
            # Convert time column to datetime index
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Validate required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = set(required) - set(df.columns)
            if missing:
                raise DataFormatError(
                    f"OHLCV missing columns: {missing}\n"
                    f"Got columns: {df.columns.tolist()}"
                )
                
            logger.info(f"✅ Loaded {len(df)} {timeframe} candles for {symbol}")
            return df[required]
            
        except Exception as e:
            if isinstance(e, (DataNotFoundError, DataFormatError)):
                raise
            raise DataProviderError(
                f"Failed to get OHLCV for {symbol}: {e}\n"
                f"Query: {query[:200]}..."
            )
            
    def get_cvd(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        """
        Get Cumulative Volume Delta with validation.
        """
        
        market = f"BINANCE:{symbol.lower()}usdt"
        
        query = f"""
        SELECT buy_volume, sell_volume
        FROM trades_1m
        WHERE market = '{market}'
        AND time >= '{start.isoformat()}Z'
        AND time <= '{end.isoformat()}Z'
        """
        
        try:
            result = self.client.query(query)
            df = pd.DataFrame(result.get_points())
            
            if df.empty:
                logger.warning(f"No CVD data for {symbol}, returning empty series")
                return pd.Series(dtype=float)
                
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Calculate CVD
            df['cvd'] = (df['buy_volume'] - df['sell_volume']).cumsum()
            
            logger.info(f"✅ Calculated CVD for {symbol}: {len(df)} points")
            return df['cvd']
            
        except Exception as e:
            logger.error(f"CVD calculation failed for {symbol}: {e}")
            return pd.Series(dtype=float)
            
    def get_oi(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        """
        Get Open Interest data with validation.
        """
        
        query = f"""
        SELECT value
        FROM open_interest
        WHERE symbol = '{symbol}'
        AND exchange = 'TOTAL_AGG'
        AND time >= '{start.isoformat()}Z'
        AND time <= '{end.isoformat()}Z'
        """
        
        try:
            result = self.client.query(query)
            df = pd.DataFrame(result.get_points())
            
            if df.empty:
                logger.warning(f"No OI data for {symbol}, returning empty series")
                return pd.Series(dtype=float)
                
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            logger.info(f"✅ Loaded OI for {symbol}: {len(df)} points")
            return df['value']
            
        except Exception as e:
            logger.error(f"OI query failed for {symbol}: {e}")
            return pd.Series(dtype=float)
            
    def get_complete_dataset(self, symbol: str, timeframe: str,
                           start: datetime, end: datetime) -> MarketData:
        """
        Get complete dataset for analysis.
        ALWAYS returns valid MarketData object or raises exception.
        """
        
        # Get OHLCV (required)
        ohlcv = self.get_ohlcv(symbol, timeframe, start, end)
        
        # Get CVD (optional but attempted)
        cvd = self.get_cvd(symbol, start, end)
        
        # Get OI (optional but attempted)
        oi = self.get_oi(symbol, start, end)
        
        # Create validated MarketData object
        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            ohlcv=ohlcv,
            cvd=cvd if not cvd.empty else None,
            oi=oi if not oi.empty else None
        )

# Custom exceptions for clear error handling
class DataProviderError(Exception):
    """Base exception for data provider issues"""
    pass

class DataNotFoundError(DataProviderError):
    """No data exists for the requested parameters"""
    pass

class DataFormatError(DataProviderError):
    """Data exists but format is unexpected"""
    pass
```

---

# 🧠 PHASE 3: STRATEGY IMPLEMENTATION (Week 4)
> 5-phase strategy with explicit state and validation

## Clear Phase Implementation
```python
# veloce/strategy/squeezeflow.py
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PhaseConfig:
    """Configuration passed to strategy, never imported"""
    squeeze_length: int = 20
    squeeze_mult_bb: float = 2.0
    squeeze_mult_kc: float = 1.5
    divergence_lookback: int = 100
    reset_threshold: float = 0.01
    min_score: float = 4.0
    position_size: float = 0.1

class SqueezeFlowStrategy:
    """
    5-phase strategy with explicit state tracking.
    Each phase validates inputs and logs progress.
    """
    
    def __init__(self, config: PhaseConfig):
        """Config injected, never imported"""
        self.config = config
        self.phase_states = {}
        
    def analyze(self, data: MarketData, timestamp: datetime) -> Optional[Signal]:
        """
        Process through 5 phases with validation at each step.
        Clear logging shows exactly what's happening.
        """
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🎯 Analyzing {data.symbol} at {timestamp}")
        logger.info(f"{'='*50}")
        
        # Validate we have minimum data
        if len(data.ohlcv) < self.config.divergence_lookback:
            logger.warning(f"Insufficient data: {len(data.ohlcv)} < {self.config.divergence_lookback}")
            return None
            
        # Phase 1: Context Assessment
        phase1_result = self._phase1_context(data)
        if not phase1_result['continue']:
            return None
            
        # Phase 2: Divergence Detection  
        phase2_result = self._phase2_divergence(data, phase1_result)
        if not phase2_result['continue']:
            return None
            
        # Phase 3: Reset Detection
        phase3_result = self._phase3_reset(data, phase2_result)
        if not phase3_result['continue']:
            return None
            
        # Phase 4: Scoring
        phase4_result = self._phase4_scoring(
            phase1_result, phase2_result, phase3_result, data
        )
        if not phase4_result['continue']:
            return None
            
        # Phase 5: Signal Generation
        signal = self._phase5_signal(phase4_result, data, timestamp)
        
        return signal
        
    def _phase1_context(self, data: MarketData) -> Dict[str, Any]:
        """
        Phase 1: Detect squeeze and market regime.
        """
        logger.info("📊 PHASE 1: Context Assessment")
        
        result = {
            'continue': False,
            'squeeze_detected': False,
            'squeeze_type': None,
            'market_regime': None
        }
        
        # Calculate Bollinger Bands
        close = data.ohlcv['close']
        bb_mean = close.rolling(self.config.squeeze_length).mean()
        bb_std = close.rolling(self.config.squeeze_length).std()
        bb_upper = bb_mean + (bb_std * self.config.squeeze_mult_bb)
        bb_lower = bb_mean - (bb_std * self.config.squeeze_mult_bb)
        
        # Calculate Keltner Channels
        high = data.ohlcv['high']
        low = data.ohlcv['low']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(self.config.squeeze_length).mean()
        kc_mean = close.rolling(self.config.squeeze_length).mean()
        kc_upper = kc_mean + (atr * self.config.squeeze_mult_kc)
        kc_lower = kc_mean - (atr * self.config.squeeze_mult_kc)
        
        # Detect squeeze
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        current_squeeze = squeeze.iloc[-1]
        
        result['squeeze_detected'] = current_squeeze
        
        if current_squeeze:
            # Determine squeeze type
            squeeze_sum = squeeze.tail(20).sum()
            if squeeze_sum > 15:
                result['squeeze_type'] = 'STRONG'
            elif squeeze_sum > 10:
                result['squeeze_type'] = 'MEDIUM'
            else:
                result['squeeze_type'] = 'WEAK'
                
            # Determine market regime
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) > 200 else sma50
            
            if close.iloc[-1] > sma50 > sma200:
                result['market_regime'] = 'BULLISH'
            elif close.iloc[-1] < sma50 < sma200:
                result['market_regime'] = 'BEARISH'
            else:
                result['market_regime'] = 'NEUTRAL'
                
            result['continue'] = True
            logger.info(f"  ✅ Squeeze: {result['squeeze_type']}")
            logger.info(f"  ✅ Regime: {result['market_regime']}")
        else:
            logger.info("  ❌ No squeeze detected")
            
        return result
        
    def _phase2_divergence(self, data: MarketData, phase1: Dict) -> Dict[str, Any]:
        """
        Phase 2: Detect CVD divergence.
        """
        logger.info("🔍 PHASE 2: Divergence Detection")
        
        result = {
            'continue': False,
            'divergence_type': None,
            'divergence_strength': 0.0
        }
        
        if data.cvd is None or data.cvd.empty:
            logger.warning("  ⚠️ No CVD data available")
            return result
            
        # Get recent price and CVD
        lookback = self.config.divergence_lookback
        recent_price = data.ohlcv['close'].tail(lookback)
        recent_cvd = data.cvd.tail(lookback)
        
        if len(recent_cvd) < lookback:
            logger.warning(f"  ⚠️ Insufficient CVD data: {len(recent_cvd)} < {lookback}")
            return result
            
        # Calculate trends
        price_change = (recent_price.iloc[-1] - recent_price.iloc[0]) / recent_price.iloc[0]
        cvd_change = recent_cvd.iloc[-1] - recent_cvd.iloc[0]
        
        # Detect divergence
        if price_change > 0.01 and cvd_change < 0:
            result['divergence_type'] = 'BEARISH'
            result['divergence_strength'] = abs(cvd_change) / 1000000  # Normalize
        elif price_change < -0.01 and cvd_change > 0:
            result['divergence_type'] = 'BULLISH'
            result['divergence_strength'] = abs(cvd_change) / 1000000
        else:
            logger.info("  ❌ No divergence detected")
            return result
            
        result['continue'] = True
        logger.info(f"  ✅ Divergence: {result['divergence_type']}")
        logger.info(f"  ✅ Strength: {result['divergence_strength']:.2f}")
        
        return result
        
    def _phase3_reset(self, data: MarketData, phase2: Dict) -> Dict[str, Any]:
        """
        Phase 3: Detect CVD reset.
        """
        logger.info("🎯 PHASE 3: Reset Detection")
        
        result = {
            'continue': False,
            'reset_detected': False,
            'reset_type': None
        }
        
        if data.cvd is None or data.cvd.empty:
            logger.warning("  ⚠️ No CVD data for reset detection")
            return result
            
        # Check for CVD reset based on divergence type
        recent_cvd = data.cvd.tail(20)
        cvd_mean = recent_cvd.mean()
        cvd_current = recent_cvd.iloc[-1]
        
        if phase2['divergence_type'] == 'BULLISH':
            # Look for CVD starting to rise
            if cvd_current > cvd_mean * (1 + self.config.reset_threshold):
                result['reset_detected'] = True
                result['reset_type'] = 'BULLISH_RESET'
        elif phase2['divergence_type'] == 'BEARISH':
            # Look for CVD starting to fall
            if cvd_current < cvd_mean * (1 - self.config.reset_threshold):
                result['reset_detected'] = True
                result['reset_type'] = 'BEARISH_RESET'
                
        if result['reset_detected']:
            result['continue'] = True
            logger.info(f"  ✅ Reset: {result['reset_type']}")
        else:
            logger.info("  ❌ No reset detected")
            
        return result
        
    def _phase4_scoring(self, phase1: Dict, phase2: Dict, 
                       phase3: Dict, data: MarketData) -> Dict[str, Any]:
        """
        Phase 4: Calculate signal score.
        """
        logger.info("💯 PHASE 4: Signal Scoring")
        
        result = {
            'continue': False,
            'total_score': 0.0,
            'breakdown': {}
        }
        
        # Base scores
        squeeze_scores = {'STRONG': 3.0, 'MEDIUM': 2.0, 'WEAK': 1.0}
        result['breakdown']['squeeze'] = squeeze_scores.get(phase1['squeeze_type'], 0)
        
        # Divergence score
        result['breakdown']['divergence'] = min(phase2['divergence_strength'] * 2, 3.0)
        
        # Reset score
        result['breakdown']['reset'] = 2.0 if phase3['reset_detected'] else 0.0
        
        # OI score (if available)
        if data.oi is not None and not data.oi.empty:
            oi_change = (data.oi.iloc[-1] - data.oi.iloc[-20]) / data.oi.iloc[-20]
            if abs(oi_change) > 0.05:  # 5% OI change
                result['breakdown']['oi'] = 2.0
            else:
                result['breakdown']['oi'] = 0.0
        else:
            result['breakdown']['oi'] = 0.0
            
        # Calculate total
        result['total_score'] = sum(result['breakdown'].values())
        
        logger.info(f"  📊 Breakdown: {result['breakdown']}")
        logger.info(f"  📊 Total Score: {result['total_score']:.2f}")
        
        if result['total_score'] >= self.config.min_score:
            result['continue'] = True
            logger.info(f"  ✅ Score {result['total_score']:.2f} >= {self.config.min_score}")
        else:
            logger.info(f"  ❌ Score {result['total_score']:.2f} < {self.config.min_score}")
            
        return result
        
    def _phase5_signal(self, phase4: Dict, data: MarketData, 
                      timestamp: datetime) -> Signal:
        """
        Phase 5: Generate trading signal.
        """
        logger.info("✅ PHASE 5: Signal Generation")
        
        # Determine side based on overall analysis
        if 'divergence' in phase4['breakdown']:
            # Simplified - real logic would be more complex
            side = "LONG" if phase4['breakdown']['divergence'] > 0 else "SHORT"
        else:
            side = "LONG"  # Default
            
        # Calculate stop loss and take profit
        current_price = data.ohlcv['close'].iloc[-1]
        atr = (data.ohlcv['high'] - data.ohlcv['low']).tail(20).mean()
        
        if side == "LONG":
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 4)
        else:
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 4)
            
        signal = Signal(
            timestamp=timestamp,
            symbol=data.symbol,
            side=side,
            score=phase4['total_score'],
            confidence=min(phase4['total_score'] / 10.0, 1.0),
            reason=f"5-phase analysis: {phase4['breakdown']}",
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        logger.info(f"  🎯 Signal: {side} @ score {signal.score:.2f}")
        logger.info(f"  🎯 SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        
        return signal
```

---

# 🧪 PHASE 4: TESTING WITH REAL DATA (Week 5)
> Every test uses real production data from 213.136.75.120

## Test Infrastructure
```python
# veloce/tests/test_with_real_data.py
import pytest
from datetime import datetime
from veloce.core.container import VeloceContainer
from veloce.core.types import MarketData, Signal

class TestWithRealData:
    """
    All tests use REAL data from production server.
    No mocks. No fakes. Real validation.
    """
    
    @pytest.fixture
    def container(self):
        """Create container with real connections"""
        return VeloceContainer("config/test_config.json")
        
    def test_data_provider_real_connection(self, container):
        """Test we can connect to real server"""
        provider = container.data_provider
        
        # Use known good date range
        start = datetime(2025, 8, 10, 12, 0, 0)
        end = datetime(2025, 8, 10, 12, 5, 0)
        
        # Get real data
        data = provider.get_ohlcv("BTC", "1m", start, end)
        
        assert not data.empty
        assert len(data) == 5  # 5 minutes of 1m data
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
    def test_strategy_with_real_data(self, container):
        """Test strategy processes real market data"""
        
        # Get real data
        start = datetime(2025, 8, 10, 10, 0, 0)
        end = datetime(2025, 8, 10, 14, 0, 0)
        
        market_data = container.data_provider.get_complete_dataset(
            "BTC", "1m", start, end
        )
        
        # Run strategy
        signal = container.strategy.analyze(market_data, end)
        
        # Signal might be None (no opportunity) or valid Signal
        if signal:
            assert isinstance(signal, Signal)
            assert signal.symbol == "BTC"
            assert signal.side in ["LONG", "SHORT"]
            assert 0 <= signal.score <= 10
            
    def test_backtest_one_day_real_data(self, container):
        """Test full backtest with one day of real data"""
        
        engine = container.get_engine()
        
        # Run backtest on real data
        results = engine.run(
            symbol="BTC",
            start=datetime(2025, 8, 10, 0, 0, 0),
            end=datetime(2025, 8, 10, 23, 59, 59)
        )
        
        assert results is not None
        assert results.total_trades >= 0
        assert results.equity_curve is not None
        
    def test_error_handling_bad_dates(self, container):
        """Test clear errors for invalid date ranges"""
        
        provider = container.data_provider
        
        # Request data from before it exists
        with pytest.raises(DataNotFoundError) as exc:
            provider.get_ohlcv(
                "BTC", "1m",
                datetime(2020, 1, 1),  # Too old
                datetime(2020, 1, 2)
            )
            
        assert "no data" in str(exc.value).lower()
        assert "date range" in str(exc.value).lower()
```

## Progressive Testing Strategy
```python
# test_runner.py
"""
Run tests in order, stop on first failure.
Each test builds on previous.
"""

def run_progressive_tests():
    test_sequence = [
        "test_data_provider_real_connection",
        "test_cvd_calculation",
        "test_oi_retrieval", 
        "test_strategy_phase1",
        "test_strategy_phase2",
        "test_strategy_phase3",
        "test_strategy_phase4",
        "test_strategy_phase5",
        "test_full_strategy_flow",
        "test_portfolio_management",
        "test_backtest_one_hour",
        "test_backtest_one_day",
        "test_visualization",
        "test_safety_systems"
    ]
    
    for test in test_sequence:
        print(f"Running: {test}")
        result = pytest.main(["-xvs", f"tests/test_with_real_data.py::{test}"])
        
        if result != 0:
            print(f"❌ Test {test} failed. Fix before continuing.")
            return False
            
        print(f"✅ Test {test} passed")
        
    print("✅ All tests passed!")
    return True
```

---

# 🎯 PHASE 5: BACKTEST ENGINE (Week 6)
> Simple sequential processing of every tick

## Clear Backtest Implementation
```python
# veloce/engine/backtest.py
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Processes every tick sequentially.
    No shortcuts. No approximations. Real simulation.
    """
    
    def __init__(self, data_provider, strategy, portfolio, config):
        """All dependencies injected"""
        self.data_provider = data_provider
        self.strategy = strategy
        self.portfolio = portfolio
        self.config = config
        
    def run(self, symbol: str, start: datetime, end: datetime, 
            timeframe: str = "1s") -> BacktestResults:
        """
        Main backtest loop - process every single tick.
        """
        
        logger.info(f"{'='*60}")
        logger.info(f"Starting Backtest: {symbol}")
        logger.info(f"Period: {start} to {end}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"{'='*60}")
        
        # Load all data upfront
        try:
            market_data = self.data_provider.get_complete_dataset(
                symbol, timeframe, start, end
            )
        except DataNotFoundError as e:
            logger.error(f"Cannot run backtest: {e}")
            return BacktestResults(error=str(e))
            
        logger.info(f"Loaded {len(market_data.ohlcv)} {timeframe} candles")
        
        # Initialize tracking
        self.equity_curve = []
        self.trades = []
        self.signals = []
        
        # Process each candle
        for idx, (timestamp, candle) in enumerate(market_data.ohlcv.iterrows()):
            
            # Update portfolio with current prices
            self.portfolio.update_market_price(symbol, candle['close'])
            
            # Check exits on existing positions
            self._check_exits(timestamp, candle)
            
            # Get all data up to current point (no future data)
            historical_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                ohlcv=market_data.ohlcv.iloc[:idx+1],
                cvd=market_data.cvd.iloc[:idx+1] if market_data.cvd is not None else None,
                oi=market_data.oi.iloc[:idx+1] if market_data.oi is not None else None
            )
            
            # Check for new signals
            signal = self.strategy.analyze(historical_data, timestamp)
            
            if signal:
                self.signals.append(signal)
                
                # Check if we can open position
                if self.portfolio.can_open_position(signal):
                    position = self.portfolio.open_position(signal, candle)
                    self.trades.append(position)
                    logger.info(f"📈 Opened {signal.side} @ {candle['close']:.2f}")
                    
            # Track equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.portfolio.total_equity,
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.positions_value
            })
            
            # Log progress every 1000 candles
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(market_data.ohlcv)} candles...")
                
        # Generate final report
        return self._generate_report(market_data)
        
    def _check_exits(self, timestamp: datetime, candle: pd.Series):
        """Check if any positions should be closed"""
        
        for position in self.portfolio.get_open_positions():
            
            # Check stop loss
            if position.side == "LONG" and candle['low'] <= position.stop_loss:
                self.portfolio.close_position(position, position.stop_loss, timestamp)
                logger.info(f"🛑 Stop loss hit: {position.symbol} @ {position.stop_loss:.2f}")
                
            elif position.side == "SHORT" and candle['high'] >= position.stop_loss:
                self.portfolio.close_position(position, position.stop_loss, timestamp)
                logger.info(f"🛑 Stop loss hit: {position.symbol} @ {position.stop_loss:.2f}")
                
            # Check take profit
            elif position.side == "LONG" and candle['high'] >= position.take_profit:
                self.portfolio.close_position(position, position.take_profit, timestamp)
                logger.info(f"✅ Take profit hit: {position.symbol} @ {position.take_profit:.2f}")
                
            elif position.side == "SHORT" and candle['low'] <= position.take_profit:
                self.portfolio.close_position(position, position.take_profit, timestamp)
                logger.info(f"✅ Take profit hit: {position.symbol} @ {position.take_profit:.2f}")
                
    def _generate_report(self, market_data: MarketData) -> BacktestResults:
        """Generate comprehensive backtest report"""
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        results = BacktestResults(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            start_date=market_data.ohlcv.index[0],
            end_date=market_data.ohlcv.index[-1],
            initial_capital=self.config.initial_capital,
            final_capital=self.portfolio.total_equity,
            total_return=(self.portfolio.total_equity - self.config.initial_capital) / self.config.initial_capital,
            total_trades=len(self.trades),
            winning_trades=len([t for t in self.trades if t.pnl > 0]),
            losing_trades=len([t for t in self.trades if t.pnl < 0]),
            equity_curve=equity_df,
            trades=pd.DataFrame([t.to_dict() for t in self.trades]),
            signals=pd.DataFrame([s.to_dict() for s in self.signals]),
            market_data=market_data
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"{'='*60}\n")
        
        return results
```

---

# 📈 PHASE 6: SINGLE VISUALIZATION (Week 7)
> ONE dashboard that works perfectly

## The Only Visualizer
```python
# veloce/visualization/dashboard.py
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

class VeloceDashboard:
    """
    THE ONLY visualization in Veloce.
    Uses TradingView Lightweight Charts.
    No alternatives. No options. Just works.
    """
    
    def __init__(self):
        self.template_path = Path(__file__).parent / "dashboard_template.html"
        
    def create(self, results: BacktestResults) -> str:
        """
        Create dashboard.html from results.
        Returns path to HTML file.
        """
        
        # Validate inputs
        if not results.market_data:
            raise ValueError("Results missing market_data")
        if results.trades is None:
            results.trades = pd.DataFrame()  # Empty is OK
            
        # Prepare data for JavaScript
        chart_data = {
            'ohlcv': self._format_ohlcv(results.market_data.ohlcv),
            'equity': self._format_equity(results.equity_curve),
            'trades': self._format_trades(results.trades),
            'signals': self._format_signals(results.signals)
        }
        
        # Add indicators if available
        if results.market_data.cvd is not None:
            chart_data['cvd'] = self._format_series(results.market_data.cvd)
        if results.market_data.oi is not None:
            chart_data['oi'] = self._format_series(results.market_data.oi)
            
        # Generate HTML
        html = self._render_template(chart_data, results)
        
        # Save to file
        output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_path = output_dir / "dashboard.html"
        dashboard_path.write_text(html)
        
        print(f"✅ Dashboard created: {dashboard_path}")
        return str(dashboard_path)
        
    def _format_ohlcv(self, df: pd.DataFrame) -> list:
        """Format OHLCV for TradingView"""
        formatted = []
        for idx, row in df.iterrows():
            formatted.append({
                'time': int(idx.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        return formatted
        
    def _format_trades(self, trades_df: pd.DataFrame) -> list:
        """Format trades for chart markers"""
        if trades_df.empty:
            return []
            
        formatted = []
        for _, trade in trades_df.iterrows():
            formatted.append({
                'time': int(trade['entry_time'].timestamp()),
                'position': 'belowBar' if trade['side'] == 'SHORT' else 'aboveBar',
                'color': 'red' if trade['side'] == 'SHORT' else 'green',
                'shape': 'arrowDown' if trade['side'] == 'SHORT' else 'arrowUp',
                'text': f"{trade['side']} @ {trade['entry_price']:.2f}"
            })
        return formatted
        
    def _render_template(self, chart_data: dict, results: BacktestResults) -> str:
        """Render HTML template with data"""
        
        template = self.template_path.read_text()
        
        # Replace placeholders
        html = template.replace('{{CHART_DATA}}', json.dumps(chart_data))
        html = html.replace('{{SYMBOL}}', results.symbol)
        html = html.replace('{{TIMEFRAME}}', results.timeframe)
        html = html.replace('{{TOTAL_RETURN}}', f"{results.total_return:.2%}")
        html = html.replace('{{TOTAL_TRADES}}', str(results.total_trades))
        html = html.replace('{{WIN_RATE}}', f"{results.win_rate:.2%}")
        
        return html
```

---

# 🛡️ PHASE 7: SAFETY SYSTEMS (Week 8)
> Simple, effective safety that actually works

## State Persistence
```python
# veloce/safety/state_manager.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class StateManager:
    """
    Simple JSON-based state persistence.
    Saves every minute, keeps last 10 states.
    """
    
    def __init__(self, state_dir: str = "veloce_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        
    def save_state(self, state: Dict[str, Any]):
        """Save current state to timestamped file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = self.state_dir / f"state_{timestamp}.json"
        
        state_data = {
            'timestamp': timestamp,
            'state': state
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
            
        # Clean old states (keep last 10)
        states = sorted(self.state_dir.glob("state_*.json"))
        if len(states) > 10:
            for old_state in states[:-10]:
                old_state.unlink()
                
    def load_latest_state(self) -> Optional[Dict]:
        """Load most recent state if exists"""
        
        states = sorted(self.state_dir.glob("state_*.json"))
        if not states:
            return None
            
        with open(states[-1]) as f:
            return json.load(f)
            
    def get_state_age(self) -> Optional[float]:
        """Get age of latest state in seconds"""
        
        latest = self.load_latest_state()
        if not latest:
            return None
            
        state_time = datetime.strptime(
            latest['timestamp'], 
            "%Y%m%d_%H%M%S"
        )
        age = (datetime.now() - state_time).total_seconds()
        return age
```

## Crash Recovery
```python
# veloce/safety/monitor.py
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class SafetyMonitor:
    """
    External monitor for crash recovery.
    Runs as separate process.
    """
    
    def __init__(self, freqtrade_url: str = "http://localhost:8080"):
        self.freqtrade_url = freqtrade_url
        self.heartbeat_file = Path("/tmp/veloce_heartbeat")
        self.max_heartbeat_age = 10  # seconds
        
    def start_monitoring(self):
        """Main monitoring loop"""
        
        logger.info("🛡️ Safety Monitor Started")
        
        while True:
            try:
                # Check heartbeat
                if self.heartbeat_file.exists():
                    heartbeat_age = time.time() - self.heartbeat_file.stat().st_mtime
                    
                    if heartbeat_age > self.max_heartbeat_age:
                        logger.critical(f"💀 HEARTBEAT DEAD ({heartbeat_age:.1f}s old)")
                        self.emergency_shutdown()
                else:
                    logger.warning("No heartbeat file found")
                    
                # Check FreqTrade connectivity
                try:
                    response = requests.get(f"{self.freqtrade_url}/api/v1/ping", timeout=2)
                    if response.status_code != 200:
                        logger.error(f"FreqTrade unhealthy: {response.status_code}")
                except:
                    logger.error("Cannot reach FreqTrade")
                    
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            time.sleep(1)
            
    def emergency_shutdown(self):
        """Close all positions immediately"""
        
        logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Get all open trades
            response = requests.get(f"{self.freqtrade_url}/api/v1/status")
            if response.status_code == 200:
                trades = response.json()
                
                for trade in trades:
                    # Force exit each trade
                    exit_response = requests.post(
                        f"{self.freqtrade_url}/api/v1/forceexit",
                        json={"tradeid": trade['trade_id']}
                    )
                    
                    if exit_response.status_code == 200:
                        logger.info(f"✅ Emergency closed trade {trade['trade_id']}")
                    else:
                        logger.error(f"❌ Failed to close trade {trade['trade_id']}")
                        
        except Exception as e:
            logger.critical(f"EMERGENCY SHUTDOWN FAILED: {e}")
            # Last resort - could send alert, stop container, etc.
```

---

# 🔍 VALIDATION FRAMEWORK

## Phase Validation Functions
```python
# veloce/validation/validators.py

def validate_phase_0_analysis() -> ValidationResult:
    """Ensure dependency analysis is complete"""
    
    analysis_file = Path("squeezeflow_dependency_analysis.json")
    if not analysis_file.exists():
        return ValidationResult(
            passed=False,
            reason="Dependency analysis not found",
            fix="Run dependency_analyzer.py first"
        )
        
    with open(analysis_file) as f:
        analysis = json.load(f)
        
    # Check we found the key issues
    if not analysis.get('circular_dependencies'):
        return ValidationResult(
            passed=False,
            reason="No circular dependencies identified",
            fix="Check if analyzer is working correctly"
        )
        
    if not analysis.get('cascade_chains'):
        return ValidationResult(
            passed=False,
            reason="No cascade chains identified", 
            fix="Verify cascade detection logic"
        )
        
    return ValidationResult(passed=True)
    
def validate_phase_1_architecture() -> ValidationResult:
    """Ensure no circular imports in Veloce"""
    
    # Check import hierarchy
    veloce_root = Path("veloce")
    
    # Core should import nothing from veloce
    core_imports = check_imports(veloce_root / "core")
    if any("veloce.data" in imp or "veloce.strategy" in imp for imp in core_imports):
        return ValidationResult(
            passed=False,
            reason="Core importing from other modules",
            fix="Core should only define types, not import from other veloce modules"
        )
        
    # Data should only import from core
    data_imports = check_imports(veloce_root / "data")
    if any("veloce.strategy" in imp or "veloce.engine" in imp for imp in data_imports):
        return ValidationResult(
            passed=False,
            reason="Data layer importing from strategy or engine",
            fix="Data should only import from core"
        )
        
    return ValidationResult(passed=True)
    
def validate_phase_2_data_layer() -> ValidationResult:
    """Ensure data layer works with real server"""
    
    try:
        from veloce.data.provider import VeloceDataProvider
        
        # Test connection to real server
        provider = VeloceDataProvider()
        
        # Test known good data
        data = provider.get_ohlcv(
            "BTC", "1m",
            datetime(2025, 8, 10, 12, 0),
            datetime(2025, 8, 10, 12, 1)
        )
        
        if data.empty:
            return ValidationResult(
                passed=False,
                reason="Could not retrieve data from server",
                fix="Check server connectivity and date range"
            )
            
    except Exception as e:
        return ValidationResult(
            passed=False,
            reason=f"Data provider failed: {e}",
            fix="Fix data provider implementation"
        )
        
    return ValidationResult(passed=True)
    
def validate_complete_system() -> bool:
    """Run all validations in sequence"""
    
    phases = [
        ("Phase 0: Dependency Analysis", validate_phase_0_analysis),
        ("Phase 1: Architecture", validate_phase_1_architecture),
        ("Phase 2: Data Layer", validate_phase_2_data_layer),
        ("Phase 3: Strategy", validate_phase_3_strategy),
        ("Phase 4: Testing", validate_phase_4_tests),
        ("Phase 5: Backtest", validate_phase_5_backtest),
        ("Phase 6: Visualization", validate_phase_6_dashboard),
        ("Phase 7: Safety", validate_phase_7_safety)
    ]
    
    all_passed = True
    
    for phase_name, validator in phases:
        print(f"\nValidating {phase_name}...")
        result = validator()
        
        if result.passed:
            print(f"✅ {phase_name} PASSED")
        else:
            print(f"❌ {phase_name} FAILED")
            print(f"   Reason: {result.reason}")
            print(f"   Fix: {result.fix}")
            all_passed = False
            
            # Stop on first failure
            if not all_passed:
                print("\n⛔ Fix this issue before proceeding")
                break
                
    return all_passed
```

---

# 💡 KEY IMPROVEMENTS IN V8

## 1. **Concrete Implementation Scripts**
- Full dependency analyzer script that actually runs
- Specific test implementations with real dates
- Complete code examples, not just concepts

## 2. **Production Data Focus**
- ALWAYS use 213.136.75.120 server
- NEVER mock data
- Real date ranges that exist (2025-08-10)

## 3. **Clear Dependency Flow**
- Strict directional imports (Core → Data → Strategy → Engine)
- Dependency injection pattern explicitly shown
- No circular references possible

## 4. **Single Implementations**
- ONE data provider
- ONE visualizer
- ONE dashboard
- No alternatives or confusion

## 5. **Explicit Validation**
- Every phase has concrete validation function
- Clear pass/fail criteria
- Specific fix instructions

## 6. **Simple State Management**
- JSON files for state (simple, debuggable)
- External monitor process for safety
- No complex distributed state

## 7. **Deep Understanding First**
- Phase 0 thoroughly analyzes SqueezeFlow
- Learn all failure modes before building
- Document lessons to avoid repeating

---

# 🚀 READY TO BUILD

This V8 plan provides:
- **Concrete scripts** Claude can run
- **Real data** from production server
- **Clear architecture** with no ambiguity
- **Explicit validation** at each step
- **Simple solutions** that Claude can maintain

Every question has been answered with specific implementation details. No mocks, no local data, just production-ready code using real market data.

**The path is clear. Let's build Veloce the RIGHT way.**