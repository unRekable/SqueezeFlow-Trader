# SqueezeFlow Clean Rebuild Plan - Generate v2 From Scratch

## Executive Summary

This plan enables Claude to generate a **completely new, clean implementation** in `/v2/` that solves all identified vulnerabilities while preserving functionality. Unlike refactoring, this creates a fresh codebase with proper architecture from day one. The v2 system will be self-contained and can become a standalone repository.

**Core Philosophy:** Extract the logic, discard the spaghetti, rebuild clean.

---

## Why This Approach Works

### Advantages of Clean Rebuild
1. **No legacy constraints** - Start with proper architecture
2. **Claude-optimized** - Built for AI maintenance from day one
3. **Self-contained** - Can run independently of old system
4. **Clean git history** - New repo without years of cruft
5. **Parallel testing** - Compare old vs new side-by-side

### Key Difference from Migration
- **Migration:** Move existing code piece by piece
- **Clean Rebuild:** Extract logic, implement fresh with proper patterns
- **Result:** Same functionality, zero technical debt

---

## Phase 0: Intelligence Extraction (Day 1)
**Duration:** 6 hours
**Objective:** Extract ALL business logic and requirements from existing system

### Morning: Create Knowledge Extractor
```python
# /knowledge_extractor.py
import ast
import json
from typing import Dict, List, Any

class KnowledgeExtractor:
    """Extract all business logic from existing codebase"""
    
    def extract_strategy_logic(self) -> Dict[str, Any]:
        """Extract core strategy rules"""
        logic = {
            "phases": [],
            "indicators": {},
            "thresholds": {},
            "signals": {},
            "risk_rules": {}
        }
        
        # Extract from strategy files
        # Document WHAT it does, not HOW
        return logic
    
    def extract_data_requirements(self) -> Dict[str, List[str]]:
        """Extract what data is actually needed"""
        return {
            "ohlcv_fields": ["open", "high", "low", "close", "volume"],
            "cvd_fields": ["spot_cvd", "perp_cvd"],
            "oi_fields": ["oi_change", "oi_value"],
            "timeframes": ["1s", "1m", "5m", "15m", "30m", "1h", "4h"]
        }
    
    def extract_calculations(self) -> Dict[str, str]:
        """Extract all mathematical formulas"""
        return {
            "squeeze_momentum": "formula here",
            "cvd_divergence": "formula here",
            "position_sizing": "formula here"
        }
    
    def generate_spec(self) -> Dict[str, Any]:
        """Generate complete system specification"""
        return {
            "strategy_logic": self.extract_strategy_logic(),
            "data_requirements": self.extract_data_requirements(),
            "calculations": self.extract_calculations(),
            "api_contracts": self.extract_api_contracts(),
            "config_values": self.extract_config_values()
        }

# Run extraction
extractor = KnowledgeExtractor()
spec = extractor.generate_spec()

# Save specification
with open('system_specification.json', 'w') as f:
    json.dump(spec, f, indent=2)
```

### Afternoon: Document Business Rules
```markdown
# /v2_requirements.md

## Core Business Logic (Extracted from Old System)

### Strategy Rules
1. **Phase 1 (Scan):** Look for squeeze momentum changes
2. **Phase 2 (Filter):** Check multi-timeframe alignment
3. **Phase 3 (Analyze):** Validate with CVD divergence
4. **Phase 4 (Confirm):** Check market structure
5. **Phase 5 (Execute):** Generate signal with position sizing

### Data Requirements
- 1-second OHLCV data from InfluxDB
- CVD (spot and perp) calculations
- Open Interest data from exchanges
- Multi-timeframe analysis (1s to 4h)

### Critical Formulas
- Squeeze Momentum = BB outside KC calculation
- CVD Divergence = Price vs Volume relationship
- Position Size = Risk-based calculation

### Configuration Values
- OI Threshold: 5.0%
- Max Position: 10% of capital
- Stop Loss: 2% from entry
- Take Profit: Dynamic based on volatility
```

### Deliverables
- [ ] `system_specification.json` with all extracted logic
- [ ] `v2_requirements.md` with business rules
- [ ] `formulas.py` with all calculations
- [ ] No code copied, only logic extracted

---

## Phase 1: Clean Architecture Design (Day 2)
**Duration:** 4 hours
**Objective:** Design proper architecture that eliminates all vulnerabilities

### Morning: Create v2 Structure
```bash
# Create clean structure
mkdir -p v2/{core,data,strategy,execution,analysis,api,tests,docs}

# Create architectural contracts
cat > v2/architecture.md << 'EOF'
# v2 Architecture

## Core Principles
1. **Single Source of Truth** - One config, one data path, one dashboard
2. **Dependency Injection** - No component creates its dependencies
3. **Protocol-Based** - All interactions through defined interfaces
4. **Testable** - Every component works in isolation
5. **Observable** - Built-in metrics and logging

## Module Structure
- `/core` - Configuration and shared types
- `/data` - Single data access layer
- `/strategy` - Pure strategy logic
- `/execution` - Order and position management
- `/analysis` - Backtesting and visualization
- `/api` - REST/WebSocket interfaces
- `/tests` - Comprehensive test suite

## Data Flow
Input → Validation → Processing → Strategy → Signals → Execution → Monitoring
         ↑                                                            ↓
         └──────────────── Feedback Loop ────────────────────────────┘
EOF
```

### Afternoon: Define Interfaces
```python
# /v2/core/protocols.py
from typing import Protocol, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class Config(Protocol):
    """Configuration interface"""
    def get(self, key: str, default: Any = None) -> Any: ...
    def validate(self) -> bool: ...

class DataProvider(Protocol):
    """Data access interface"""
    def get_ohlcv(self, symbol: str, timeframe: str, 
                  start: datetime, end: datetime) -> pd.DataFrame: ...
    def get_cvd(self, symbol: str, start: datetime, 
                end: datetime) -> pd.DataFrame: ...
    def get_oi(self, symbol: str) -> Optional[Dict[str, float]]: ...

class Strategy(Protocol):
    """Strategy interface"""
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]: ...
    def generate_signal(self, analysis: Dict[str, Any]) -> Optional[Dict]: ...

class Dashboard(Protocol):
    """Visualization interface"""
    def generate(self, results: Dict[str, Any]) -> str: ...
    def validate(self) -> bool: ...
```

### Deliverables
- [ ] `/v2/` directory structure created
- [ ] `architecture.md` documenting design
- [ ] `protocols.py` with all interfaces
- [ ] Zero coupling between modules

---

## Phase 2: Core Implementation (Day 3)
**Duration:** 6 hours
**Objective:** Implement core modules with single source of truth

### Morning: Configuration System
```python
# /v2/core/config.py
"""THE configuration - single source of truth"""
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class SystemConfig:
    """All configuration in one place"""
    
    # Data Source
    influx_host: str = "213.136.75.120"
    influx_port: int = 8086
    influx_database: str = "significant_trades"
    
    # Strategy Parameters
    oi_enabled: bool = True
    oi_threshold: float = 5.0
    squeeze_period: int = 20
    squeeze_mult: float = 2.0
    kc_mult: float = 1.5
    
    # Risk Management
    max_position_pct: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_mult: float = 2.0
    
    # Performance
    enable_1s_mode: bool = True
    max_lookback_hours: int = 24
    cache_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load from environment variables"""
        return cls(
            influx_host=os.getenv('INFLUX_HOST', cls.influx_host),
            influx_port=int(os.getenv('INFLUX_PORT', cls.influx_port)),
            oi_enabled=os.getenv('OI_ENABLED', 'true').lower() == 'true',
            oi_threshold=float(os.getenv('OI_THRESHOLD', cls.oi_threshold))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary"""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        if self.stop_loss_pct >= self.max_position_pct:
            raise ValueError("Stop loss cannot exceed max position")
        if self.oi_threshold < 0 or self.oi_threshold > 100:
            raise ValueError("OI threshold must be between 0-100")
        return True

# Global singleton - THE configuration
CONFIG = SystemConfig.from_env()
CONFIG.validate()
```

### Afternoon: Data Access Layer
```python
# /v2/data/provider.py
"""THE data provider - single access pattern"""
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
from influxdb import InfluxDBClient
from v2.core.config import CONFIG

class DataProvider:
    """Single source for all data access"""
    
    def __init__(self, config: SystemConfig = CONFIG):
        self.config = config
        self.client = InfluxDBClient(
            host=config.influx_host,
            port=config.influx_port,
            database=config.influx_database
        )
        self._cache = {}
    
    def get_ohlcv(self, symbol: str, timeframe: str,
                  start: datetime, end: datetime) -> pd.DataFrame:
        """Get OHLCV data - THE method"""
        cache_key = f"{symbol}_{timeframe}_{start}_{end}"
        
        if cache_key in self._cache and self.config.cache_enabled:
            return self._cache[cache_key]
        
        query = f"""
        SELECT open, high, low, close, volume 
        FROM trades_{timeframe}
        WHERE market = 'BINANCE:{symbol.lower()}usdt'
        AND time >= '{start.isoformat()}'
        AND time <= '{end.isoformat()}'
        """
        
        result = pd.DataFrame(self.client.query(query).get_points())
        if not result.empty:
            result.set_index('time', inplace=True)
            self._cache[cache_key] = result
        
        return result
    
    def get_cvd(self, symbol: str, start: datetime, 
                end: datetime) -> pd.DataFrame:
        """Get CVD data - THE method"""
        # Implementation
        pass
    
    def get_oi(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get OI data - THE method"""
        if not self.config.oi_enabled:
            return None
        # Implementation
        pass

# Global singleton - THE data provider
DATA = DataProvider(CONFIG)
```

### Deliverables
- [ ] Single config file with ALL settings
- [ ] Single data provider with ALL access
- [ ] No hardcoded values anywhere
- [ ] Cache layer built-in

---

## Phase 3: Strategy Implementation (Day 4)
**Duration:** 6 hours
**Objective:** Implement clean strategy with extracted logic

### Morning: Strategy Core
```python
# /v2/strategy/squeezeflow.py
"""Clean SqueezeFlow implementation"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from v2.core.config import CONFIG
from v2.data.provider import DATA

class SqueezeFlowStrategy:
    """THE strategy - clean implementation"""
    
    def __init__(self, config: SystemConfig = CONFIG, 
                 data: DataProvider = DATA):
        self.config = config
        self.data = data
        self.state = {}
    
    def analyze(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Analyze market at given timestamp"""
        # Get multi-timeframe data
        end_time = timestamp
        start_time = timestamp - timedelta(hours=self.config.max_lookback_hours)
        
        analysis = {}
        for timeframe in ['1m', '5m', '15m', '30m', '1h']:
            df = self.data.get_ohlcv(symbol, timeframe, start_time, end_time)
            analysis[timeframe] = self._analyze_timeframe(df)
        
        # Get CVD and OI
        analysis['cvd'] = self._analyze_cvd(symbol, start_time, end_time)
        analysis['oi'] = self._analyze_oi(symbol)
        
        # Run 5-phase analysis
        analysis['phases'] = self._run_phases(analysis)
        
        return analysis
    
    def _run_phases(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run 5-phase strategy logic"""
        phases = {}
        
        # Phase 1: Scan for squeeze
        phases['scan'] = self._phase1_scan(analysis)
        
        # Phase 2: Filter with multi-timeframe
        if phases['scan']['passed']:
            phases['filter'] = self._phase2_filter(analysis)
        
        # Phase 3: Analyze with CVD
        if phases.get('filter', {}).get('passed'):
            phases['analyze'] = self._phase3_analyze(analysis)
        
        # Phase 4: Confirm market structure
        if phases.get('analyze', {}).get('passed'):
            phases['confirm'] = self._phase4_confirm(analysis)
        
        # Phase 5: Execute signal
        if phases.get('confirm', {}).get('passed'):
            phases['execute'] = self._phase5_execute(analysis)
        
        return phases
    
    def generate_signal(self, symbol: str, 
                       timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Generate trading signal"""
        analysis = self.analyze(symbol, timestamp)
        
        if analysis['phases'].get('execute', {}).get('signal'):
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'action': analysis['phases']['execute']['action'],
                'confidence': analysis['phases']['execute']['confidence'],
                'position_size': self._calculate_position_size(analysis),
                'stop_loss': self._calculate_stop_loss(analysis),
                'take_profit': self._calculate_take_profit(analysis),
                'analysis': analysis
            }
        
        return None
```

### Afternoon: Indicator Calculations
```python
# /v2/strategy/indicators.py
"""All indicator calculations in one place"""
import pandas as pd
import numpy as np
from v2.core.config import CONFIG

class Indicators:
    """Pure indicator calculations"""
    
    @staticmethod
    def squeeze_momentum(df: pd.DataFrame, 
                         period: int = CONFIG.squeeze_period,
                         mult: float = CONFIG.squeeze_mult) -> pd.DataFrame:
        """Calculate squeeze momentum indicator"""
        # Bollinger Bands
        basis = df['close'].rolling(period).mean()
        dev = df['close'].rolling(period).std()
        upper_bb = basis + mult * dev
        lower_bb = basis - mult * dev
        
        # Keltner Channels
        tr = pd.DataFrame()
        tr['h-l'] = df['high'] - df['low']
        tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
        tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
        true_range = tr.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        upper_kc = basis + CONFIG.kc_mult * atr
        lower_kc = basis - CONFIG.kc_mult * atr
        
        # Squeeze detection
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        
        # Momentum
        highest = df['high'].rolling(period).max()
        lowest = df['low'].rolling(period).min()
        mean_hl = (highest + lowest) / 2
        mean_close = df['close'].rolling(period).mean()
        momentum = df['close'] - ((mean_hl + mean_close) / 2)
        
        result = pd.DataFrame({
            'squeeze_on': squeeze_on,
            'squeeze_off': squeeze_off,
            'momentum': momentum
        })
        
        return result
    
    @staticmethod
    def cvd_divergence(price: pd.Series, cvd: pd.Series) -> Dict[str, Any]:
        """Calculate CVD divergence"""
        # Implementation
        pass
```

### Deliverables
- [ ] Clean strategy implementation
- [ ] All indicators in one file
- [ ] Pure functions (no side effects)
- [ ] Testable components

---

## Phase 4: Backtesting Engine (Day 5)
**Duration:** 6 hours
**Objective:** Create clean backtesting engine

### Morning: Engine Core
```python
# /v2/analysis/backtest.py
"""Clean backtesting engine"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
from v2.core.config import CONFIG
from v2.data.provider import DATA
from v2.strategy.squeezeflow import SqueezeFlowStrategy

class BacktestEngine:
    """THE backtesting engine"""
    
    def __init__(self, config: SystemConfig = CONFIG,
                 data: DataProvider = DATA):
        self.config = config
        self.data = data
        self.strategy = SqueezeFlowStrategy(config, data)
        self.results = []
        self.metrics = {}
    
    def run(self, symbol: str, start: datetime, 
            end: datetime, timeframe: str = '1s') -> Dict[str, Any]:
        """Run backtest"""
        # Get data
        df = self.data.get_ohlcv(symbol, timeframe, start, end)
        
        # Initialize portfolio
        portfolio = {
            'cash': 10000,
            'position': 0,
            'trades': [],
            'equity_curve': []
        }
        
        # Process each candle
        for timestamp, row in df.iterrows():
            # Generate signal
            signal = self.strategy.generate_signal(symbol, timestamp)
            
            # Execute trades
            if signal:
                portfolio = self._execute_trade(portfolio, signal, row)
            
            # Update portfolio
            portfolio = self._update_portfolio(portfolio, row)
            
            # Record metrics
            self.results.append({
                'timestamp': timestamp,
                'price': row['close'],
                'position': portfolio['position'],
                'equity': portfolio['cash'] + portfolio['position'] * row['close']
            })
        
        # Calculate final metrics
        self.metrics = self._calculate_metrics(portfolio, self.results)
        
        return {
            'symbol': symbol,
            'start': start,
            'end': end,
            'timeframe': timeframe,
            'metrics': self.metrics,
            'trades': portfolio['trades'],
            'equity_curve': pd.DataFrame(self.results)
        }
    
    def _calculate_metrics(self, portfolio: Dict, 
                          results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics"""
        equity_curve = pd.DataFrame(results)['equity']
        
        return {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100,
            'sharpe_ratio': self._calculate_sharpe(equity_curve),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': self._calculate_win_rate(portfolio['trades']),
            'profit_factor': self._calculate_profit_factor(portfolio['trades']),
            'total_trades': len(portfolio['trades'])
        }
```

### Afternoon: Report Generation
```python
# /v2/analysis/reporter.py
"""Backtest report generation"""
from typing import Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestReporter:
    """Generate backtest reports and dashboards"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.metrics = results['metrics']
        self.trades = results['trades']
        self.equity = results['equity_curve']
    
    def generate_dashboard(self) -> str:
        """Generate HTML dashboard - THE dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Trade Distribution',
                'Drawdown', 'Returns Distribution',
                'Trade Analysis', 'Performance Metrics'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )
        
        # Add all charts
        self._add_equity_curve(fig, row=1, col=1)
        self._add_trade_distribution(fig, row=1, col=2)
        self._add_drawdown(fig, row=2, col=1)
        self._add_returns_dist(fig, row=2, col=2)
        self._add_trade_analysis(fig, row=3, col=1)
        self._add_metrics_table(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results - {self.results['symbol']}",
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Generate HTML
        html = fig.to_html(include_plotlyjs='cdn')
        
        # Save to file
        output_path = f"v2/output/backtest_{self.results['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def generate_summary(self) -> str:
        """Generate text summary"""
        return f"""
        Backtest Summary
        ================
        Symbol: {self.results['symbol']}
        Period: {self.results['start']} to {self.results['end']}
        
        Performance Metrics:
        - Total Return: {self.metrics['total_return']:.2f}%
        - Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
        - Max Drawdown: {self.metrics['max_drawdown']:.2f}%
        - Win Rate: {self.metrics['win_rate']:.2f}%
        - Total Trades: {self.metrics['total_trades']}
        """
```

### Deliverables
- [ ] Clean backtest engine
- [ ] Single dashboard generator
- [ ] Comprehensive metrics
- [ ] Self-contained reports

---

## Phase 5: API and Services (Day 6)
**Duration:** 4 hours
**Objective:** Create clean API layer

### Morning: REST API
```python
# /v2/api/server.py
"""REST API for v2 system"""
from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import Optional
from v2.core.config import CONFIG
from v2.data.provider import DATA
from v2.strategy.squeezeflow import SqueezeFlowStrategy
from v2.analysis.backtest import BacktestEngine

app = FastAPI(title="SqueezeFlow v2 API")

# Initialize components
strategy = SqueezeFlowStrategy(CONFIG, DATA)
backtest = BacktestEngine(CONFIG, DATA)

@app.get("/health")
def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "config_valid": CONFIG.validate()
    }

@app.get("/config")
def get_config():
    """Get current configuration"""
    return CONFIG.to_dict()

@app.post("/signal/{symbol}")
def generate_signal(symbol: str, timestamp: Optional[datetime] = None):
    """Generate trading signal"""
    if timestamp is None:
        timestamp = datetime.now()
    
    signal = strategy.generate_signal(symbol, timestamp)
    
    if signal:
        return signal
    else:
        return {"message": "No signal generated"}

@app.post("/backtest")
def run_backtest(symbol: str, start: datetime, end: datetime, 
                  timeframe: str = "1s"):
    """Run backtest"""
    try:
        results = backtest.run(symbol, start, end, timeframe)
        return {
            "metrics": results['metrics'],
            "trades": len(results['trades']),
            "report_url": f"/reports/{results['report_id']}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Afternoon: Docker Setup
```dockerfile
# /v2/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy v2 code
COPY . .

# Environment variables
ENV INFLUX_HOST=213.136.75.120
ENV INFLUX_PORT=8086
ENV OI_ENABLED=true
ENV ENABLE_1S_MODE=true

# Run API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# /v2/docker-compose.yml
version: '3.8'

services:
  squeezeflow-v2:
    build: .
    container_name: squeezeflow-v2
    ports:
      - "8000:8000"
    environment:
      - INFLUX_HOST=213.136.75.120
      - INFLUX_PORT=8086
      - OI_ENABLED=true
      - ENABLE_1S_MODE=true
    volumes:
      - ./output:/app/output
    restart: unless-stopped
```

### Deliverables
- [ ] REST API with all endpoints
- [ ] Docker containerization
- [ ] Standalone deployment
- [ ] API documentation

---

## Phase 6: Testing Suite (Day 7)
**Duration:** 6 hours
**Objective:** Comprehensive testing

### Morning: Unit Tests
```python
# /v2/tests/test_config.py
"""Test configuration system"""
import pytest
from v2.core.config import SystemConfig

def test_config_singleton():
    """Test config is singleton"""
    from v2.core.config import CONFIG
    config2 = SystemConfig.from_env()
    # Should be same instance
    assert CONFIG.oi_threshold == config2.oi_threshold

def test_config_validation():
    """Test config validation"""
    config = SystemConfig(stop_loss_pct=0.5, max_position_pct=0.1)
    with pytest.raises(ValueError):
        config.validate()

# /v2/tests/test_data.py
"""Test data provider"""
from v2.data.provider import DataProvider
from v2.core.config import SystemConfig

def test_data_singleton():
    """Test data provider is singleton"""
    from v2.data.provider import DATA
    data2 = DataProvider(SystemConfig())
    # Should use same config
    assert DATA.config.influx_host == data2.config.influx_host

def test_data_caching():
    """Test data caching works"""
    # Implementation
    pass

# /v2/tests/test_strategy.py
"""Test strategy logic"""
from v2.strategy.squeezeflow import SqueezeFlowStrategy

def test_strategy_phases():
    """Test 5-phase logic"""
    # Implementation
    pass
```

### Afternoon: Integration Tests
```python
# /v2/tests/test_integration.py
"""End-to-end integration tests"""
import pytest
from datetime import datetime, timedelta
from v2.core.config import SystemConfig
from v2.data.provider import DataProvider
from v2.strategy.squeezeflow import SqueezeFlowStrategy
from v2.analysis.backtest import BacktestEngine

def test_full_backtest():
    """Test complete backtest flow"""
    config = SystemConfig(influx_host="213.136.75.120")
    data = DataProvider(config)
    engine = BacktestEngine(config, data)
    
    # Run backtest
    results = engine.run(
        symbol="BTC",
        start=datetime(2025, 8, 10),
        end=datetime(2025, 8, 10, 1, 0),
        timeframe="1s"
    )
    
    # Verify results
    assert results['metrics']['total_trades'] >= 0
    assert 'equity_curve' in results
    assert len(results['trades']) == results['metrics']['total_trades']

def test_api_endpoints():
    """Test all API endpoints"""
    from fastapi.testclient import TestClient
    from v2.api.server import app
    
    client = TestClient(app)
    
    # Test health
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    
    # Test config
    response = client.get("/config")
    assert response.status_code == 200
    assert 'oi_threshold' in response.json()
```

### Deliverables
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] API tests
- [ ] Test coverage > 80%

---

## Phase 7: Validation and Comparison (Day 8)
**Duration:** 6 hours
**Objective:** Validate v2 matches v1 functionality

### Morning: Comparison Tests
```python
# /comparison_test.py
"""Compare v1 and v2 outputs"""
import sys
import pandas as pd
from datetime import datetime

# Import v1 system
sys.path.append('/Users/u/PycharmProjects/SqueezeFlow Trader')
from backtest.engine import BacktestEngine as V1Engine

# Import v2 system
sys.path.append('/Users/u/PycharmProjects/SqueezeFlow Trader/v2')
from analysis.backtest import BacktestEngine as V2Engine

def compare_backtests():
    """Run same backtest on both systems"""
    
    # Test parameters
    symbol = "BTC"
    start = datetime(2025, 8, 10, 12, 0)
    end = datetime(2025, 8, 10, 13, 0)
    
    # Run v1
    print("Running v1 backtest...")
    v1_results = run_v1_backtest(symbol, start, end)
    
    # Run v2
    print("Running v2 backtest...")
    v2_results = run_v2_backtest(symbol, start, end)
    
    # Compare metrics
    print("\nMetrics Comparison:")
    print(f"V1 Return: {v1_results['return']:.2f}%")
    print(f"V2 Return: {v2_results['metrics']['total_return']:.2f}%")
    
    print(f"V1 Trades: {v1_results['trades']}")
    print(f"V2 Trades: {v2_results['metrics']['total_trades']}")
    
    # Check if results are similar (allowing 5% variance)
    return_diff = abs(v1_results['return'] - v2_results['metrics']['total_return'])
    if return_diff < 5:
        print("✅ Results match within tolerance")
    else:
        print(f"⚠️ Results differ by {return_diff:.2f}%")
```

### Afternoon: Performance Benchmarks
```python
# /v2/tests/benchmark.py
"""Performance benchmarks"""
import time
import memory_profiler
from v2.analysis.backtest import BacktestEngine

@memory_profiler.profile
def benchmark_backtest():
    """Benchmark backtest performance"""
    engine = BacktestEngine()
    
    start_time = time.time()
    results = engine.run(
        symbol="BTC",
        start=datetime(2025, 8, 10),
        end=datetime(2025, 8, 10),
        timeframe="1s"
    )
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Candles processed: {len(results['equity_curve'])}")
    print(f"Candles/second: {len(results['equity_curve']) / (end_time - start_time):.0f}")

def compare_performance():
    """Compare v1 vs v2 performance"""
    # Implementation
    pass
```

### Deliverables
- [ ] Feature parity confirmed
- [ ] Performance benchmarks
- [ ] Comparison report
- [ ] Migration guide

---

## Phase 8: Documentation and Deployment (Day 9)
**Duration:** 4 hours
**Objective:** Complete documentation and deployment ready

### Morning: Documentation
```markdown
# /v2/README.md

# SqueezeFlow v2 - Clean Architecture

## Overview
Complete rewrite of SqueezeFlow with clean architecture, solving all v1 issues:
- ✅ Single source of truth for configuration
- ✅ One data access pattern
- ✅ One dashboard implementation
- ✅ No circular dependencies
- ✅ Fully testable components

## Quick Start
```bash
cd v2
pip install -r requirements.txt
python -m pytest tests/  # Run tests
python api/server.py     # Start API
```

## Architecture
- `/core` - Configuration and types (single source of truth)
- `/data` - Data access layer (one pattern)
- `/strategy` - Strategy implementation (clean logic)
- `/analysis` - Backtesting and reporting (one dashboard)
- `/api` - REST API
- `/tests` - Comprehensive test suite

## Key Improvements from v1
1. **Configuration**: One file, one source
2. **Data Access**: Single provider pattern
3. **Visualization**: One dashboard generator
4. **Dependencies**: Clean, no circular imports
5. **Testing**: 80%+ coverage, isolated tests

## Deployment
```bash
docker-compose up -d
```

## API Endpoints
- `GET /health` - System health
- `GET /config` - Current configuration
- `POST /signal/{symbol}` - Generate signal
- `POST /backtest` - Run backtest
```

### Afternoon: Final Package
```bash
# Create standalone repository
cd /Users/u/PycharmProjects/SqueezeFlow Trader/v2

# Initialize git
git init
git add .
git commit -m "Initial commit: SqueezeFlow v2 clean architecture"

# Create release package
tar -czf squeezeflow_v2.tar.gz --exclude=__pycache__ --exclude=.git *

# Generate migration script
cat > migrate_to_v2.sh << 'EOF'
#!/bin/bash
echo "SqueezeFlow v2 Migration"
echo "========================"

# Test v2
cd v2
python -m pytest tests/

# Compare with v1
python ../comparison_test.py

# If tests pass, ready to migrate
echo "v2 is ready for production"
EOF
```

### Deliverables
- [ ] Complete documentation
- [ ] Standalone git repository
- [ ] Docker deployment ready
- [ ] Migration guide

---

## Success Metrics

### Clean Architecture Achieved
- ✅ **1** configuration file (was 5+)
- ✅ **1** data provider (was 3+)
- ✅ **1** dashboard (was 14+)
- ✅ **0** circular dependencies (was many)
- ✅ **0** hardcoded values (was 50+)

### Performance Maintained
- ✅ Same or better execution speed
- ✅ Same or better memory usage
- ✅ Same trading performance
- ✅ 1-second mode fully supported

### Maintainability Improved
- ✅ Claude can understand entire codebase
- ✅ Changes require editing ONE file
- ✅ Components testable in isolation
- ✅ Clear error messages
- ✅ 80%+ test coverage

---

## Timeline Summary

| Day | Focus | Outcome |
|-----|-------|---------|
| 1 | Extract Logic | Complete specification |
| 2 | Design Architecture | Clean structure |
| 3 | Core Implementation | Config + Data |
| 4 | Strategy | Clean strategy logic |
| 5 | Backtesting | Engine + Dashboard |
| 6 | API + Docker | Deployable service |
| 7 | Testing | Full test suite |
| 8 | Validation | Feature parity confirmed |
| 9 | Documentation | Production ready |

**Total: 9 working days**

---

## Key Advantages of This Approach

### 1. **Clean Slate**
- No legacy code constraints
- Proper architecture from day one
- All vulnerabilities eliminated by design

### 2. **Self-Contained**
- Lives in `/v2/` folder
- Can be extracted as new repository
- No interference with v1 system

### 3. **Parallel Testing**
- Run v1 and v2 side-by-side
- Compare outputs directly
- Validate before switching

### 4. **Claude-Optimized**
- Clear structure Claude can navigate
- Single sources of truth
- Explicit interfaces

### 5. **Migration Path**
- Keep v1 running during development
- Test thoroughly before switching
- Can always rollback if needed

---

## Conclusion

This clean rebuild approach creates a **brand new v2 system** that:
1. **Solves all v1 problems** by design
2. **Maintains all functionality** through extracted logic
3. **Lives independently** in `/v2/` folder
4. **Can become a new repository** easily
5. **Is maintainable by Claude** from day one

The key insight: We're not moving code, we're extracting business logic and implementing it cleanly. This eliminates technical debt while preserving all the valuable trading logic developed in v1.

**Ready to build v2 when you are.**