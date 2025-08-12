# SqueezeFlow Architecture Refactoring Plan - Claude Optimized

## Executive Summary

This plan takes a **surgical refactoring approach** to eliminate the top vulnerabilities causing 80% of debugging pain, while keeping the system operational throughout. Instead of a full rewrite, we'll systematically fix critical issues in-place, leveraging Claude's strengths and working within its limitations.

**Core Philosophy:** Fix the pain points, don't rebuild the universe.

---

## Critical Success Factors

### What Makes This Plan Different
1. **Incremental** - System stays operational throughout
2. **Claude-Aware** - Designed for Claude's context limits and tool patterns
3. **Measurable** - Each fix has clear success criteria
4. **Reversible** - Can rollback any change if needed
5. **Memory-Persistent** - Uses CLAUDE.md to retain knowledge across sessions

---

## Phase 0: Memory Setup & Intelligence Gathering (Day 1)
**Duration:** 4-6 hours
**Objective:** Set up Claude's memory system for maximum effectiveness

### Actions
```bash
# 1. Create vulnerability scanner scripts
cat > scan_vulnerabilities.sh << 'EOF'
#!/bin/bash
echo "=== Configuration Duplicates ==="
grep -r "oi_threshold\|OI_THRESHOLD" --include="*.py" | wc -l

echo "=== Visualizer Count ==="
find . -name "*visual*.py" -o -name "*dashboard*.py" | wc -l

echo "=== Direct InfluxDB Queries ==="
grep -r "InfluxDBClient\|client.query" --include="*.py" | wc -l

echo "=== Circular Imports ==="
python3 -c "import ast; import os
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            # Check for potential circular imports
            pass"
EOF
chmod +x scan_vulnerabilities.sh
```

### Memory Documentation
Create comprehensive CLAUDE.md additions:
```markdown
## Vulnerability Patterns Found
- Configuration cascade affects: [list files]
- Active visualizers: [identify which ones are used]
- Data access patterns: [map all paths]

## Grep Patterns for Common Issues
- Find config usage: `grep -r "CONFIG\|config" --include="*.py"`
- Find visualizers: `find . -name "*visual*.py" -type f`
- Find data access: `grep -r "DataPipeline\|InfluxDBClient" --include="*.py"`

## Validation Commands
- Test config changes: `python3 -c "from backtest.indicator_config import *; print(globals())"`
- Test data pipeline: `python3 -c "from data.pipeline import DataPipeline; dp = DataPipeline()"`
```

### Deliverables
- [ ] `scan_vulnerabilities.sh` script created
- [ ] CLAUDE.md updated with patterns
- [ ] Baseline metrics documented
- [ ] Quick validation scripts ready

---

## Phase 1: Configuration Unification (Days 2-3)
**Duration:** 2 days
**Objective:** Create single source of truth for ALL configuration

### Day 1: Analysis and Design

#### Morning: Map Configuration Chaos
```bash
# Find all configuration points
grep -r "threshold\|THRESHOLD" --include="*.py" > config_instances.txt
grep -r "= [0-9]\|= '[^']*'" --include="*.py" > hardcoded_values.txt
find . -name "*config*.py" -o -name "*settings*.py" > config_files.txt

# Identify which components bypass config
grep -L "indicator_config" strategies/squeezeflow/components/*.py
```

#### Afternoon: Design Unified Config
```python
# /backtest/system_config.py - THE configuration
from typing import Dict, Any
import os
import json

class SystemConfig:
    """Single source of truth for ALL configuration"""
    
    # Core Settings
    INFLUX_HOST = os.getenv('INFLUX_HOST', '213.136.75.120')
    INFLUX_PORT = int(os.getenv('INFLUX_PORT', 8086))
    
    # Strategy Settings
    OI_THRESHOLD = float(os.getenv('OI_THRESHOLD', 5.0))
    OI_ENABLED = os.getenv('OI_ENABLED', 'true').lower() == 'true'
    
    # Indicator Settings
    SQUEEZE_MOMENTUM_PERIOD = 20
    SQUEEZE_MOMENTUM_MULT = 2.0
    SQUEEZE_KC_MULT = 1.5
    
    # Performance Settings
    ENABLE_1S_MODE = os.getenv('ENABLE_1S_MODE', 'true').lower() == 'true'
    MAX_LOOKBACK_HOURS = 24
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export all config as dict"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration consistency"""
        # Add validation logic
        return True

# Global singleton
CONFIG = SystemConfig()
```

### Day 2: Implementation and Migration

#### Morning: Migration Script
```python
# migrate_to_unified_config.py
import os
import re

files_to_update = [
    'strategies/squeezeflow/strategy.py',
    'strategies/squeezeflow/components/phase3_analysis.py',
    'strategies/squeezeflow/components/phase5_execution.py',
    'data/pipeline.py',
    'backtest/engine.py',
    'backtest/reporting/visualizer.py'
]

replacements = {
    r'oi_threshold\s*=\s*[0-9.]+': 'oi_threshold = CONFIG.OI_THRESHOLD',
    r'from .*indicator_config import': 'from backtest.system_config import CONFIG\n# from backtest.indicator_config import',
    r'OI_THRESHOLD\s*=\s*[0-9.]+': 'OI_THRESHOLD = CONFIG.OI_THRESHOLD'
}

for file_path in files_to_update:
    # Update each file
    pass
```

#### Afternoon: Testing and Validation
```bash
# Test that all components use unified config
python3 -c "
from backtest.system_config import CONFIG
print(f'OI Threshold: {CONFIG.OI_THRESHOLD}')

# Test each component loads properly
from strategies.squeezeflow.strategy import SqueezeFlowStrategy
from data.pipeline import DataPipeline
print('✅ All components load with unified config')
"
```

### Success Criteria
- [ ] ONE configuration file exists
- [ ] ALL components import from it
- [ ] Zero hardcoded values remain
- [ ] Changing one value affects all components
- [ ] Tests pass with new config

---

## Phase 2: Visualizer Consolidation (Days 4-5)
**Duration:** 2 days
**Objective:** ONE dashboard to rule them all

### Day 1: Identify and Analyze

#### Morning: Map the Visualizer Maze
```bash
# Find all visualizers
find . -type f -name "*visual*.py" -o -name "*dashboard*.py" | while read file; do
    echo "=== $file ==="
    grep -l "class.*Visualizer\|def.*generate.*html" "$file"
    grep "import.*plotly\|import.*matplotlib" "$file"
done

# Find what actually gets called
grep -r "Visualizer\|visualizer\|generate_dashboard" --include="*.py" | grep -v "^#"

# Identify the ACTIVE visualizer
grep -r "from.*visualizer import" backtest/engine.py
```

#### Afternoon: Choose the Winner
```python
# Create test to compare visualizers
test_data = load_sample_backtest_data()

visualizers = [
    'backtest.reporting.visualizer',
    'backtest.reporting.complete_visualizer',
    'backtest.reporting.enhanced_visualizer'
]

for viz in visualizers:
    try:
        module = __import__(viz)
        # Test each one
        result = module.generate_dashboard(test_data)
        print(f"{viz}: {'✅ Works' if result else '❌ Fails'}")
    except Exception as e:
        print(f"{viz}: ❌ Error - {e}")
```

### Day 2: Consolidate

#### Morning: Create Unified Dashboard
```python
# /backtest/reporting/dashboard.py - THE dashboard
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UnifiedDashboard:
    """THE dashboard - all others will be deleted"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def generate(self, backtest_results: Dict[str, Any]) -> str:
        """Generate THE dashboard HTML"""
        # Consolidated logic from best visualizer
        pass
    
    def validate_output(self) -> bool:
        """Self-validation of generated dashboard"""
        # Check that all expected charts exist
        pass

# Backward compatibility wrapper
def generate_dashboard(results):
    """Legacy interface for compatibility"""
    dashboard = UnifiedDashboard(CONFIG)
    return dashboard.generate(results)
```

#### Afternoon: Delete the Rest
```bash
# First backup
mkdir -p .deprecated/visualizers
mv backtest/reporting/*visualizer*.py .deprecated/visualizers/

# Update all references
grep -r "visualizer" --include="*.py" | while read line; do
    # Update to use dashboard.py
done

# Test everything still works
python3 backtest/engine.py --test-visualization
```

### Success Criteria
- [ ] ONE dashboard.py file exists
- [ ] All other visualizers moved to .deprecated/
- [ ] All components use the unified dashboard
- [ ] Dashboard validates its own output
- [ ] Visual validation confirms charts work

---

## Phase 3: Data Pipeline Standardization (Days 6-7)
**Duration:** 2 days
**Objective:** ONE way to access data

### Day 1: Map Data Access Patterns

#### Morning: Find All Data Paths
```bash
# Find direct InfluxDB access
grep -r "InfluxDBClient" --include="*.py" > direct_influx_access.txt

# Find DataPipeline usage
grep -r "DataPipeline" --include="*.py" > pipeline_usage.txt

# Find mixed usage (files that use both)
comm -12 <(grep -l "InfluxDBClient" --include="*.py" | sort) \
         <(grep -l "DataPipeline" --include="*.py" | sort)
```

#### Afternoon: Design Unified Access
```python
# /data/unified_data_access.py
from typing import Optional, Dict, Any
import pandas as pd

class UnifiedDataAccess:
    """THE way to access data - no alternatives"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self._client = self._setup_client()
        
    def get_ohlcv(self, symbol: str, timeframe: str, 
                  start: str, end: str) -> pd.DataFrame:
        """THE method to get OHLCV data"""
        # Single implementation
        pass
    
    def get_cvd(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """THE method to get CVD data"""
        # Single implementation
        pass
    
    def get_oi(self, symbol: str) -> Optional[Dict[str, float]]:
        """THE method to get OI data"""
        # Single implementation
        pass

# Global singleton
DATA = UnifiedDataAccess(CONFIG)
```

### Day 2: Migrate All Access

#### Morning: Update Strategy
```python
# Before: Multiple data access patterns
class SqueezeFlowStrategy:
    def __init__(self):
        self.pipeline = DataPipeline()
        self.client = InfluxDBClient()
    
    def get_data(self):
        # Sometimes uses pipeline
        data1 = self.pipeline.get_data()
        # Sometimes direct query
        data2 = self.client.query("SELECT...")

# After: Single access pattern
class SqueezeFlowStrategy:
    def __init__(self):
        self.data = DATA  # THE data access
    
    def get_data(self):
        data = self.data.get_ohlcv(...)  # Always same method
```

#### Afternoon: Test and Validate
```bash
# Ensure no direct InfluxDB access remains
if grep -r "InfluxDBClient" --include="*.py" --exclude="unified_data_access.py"; then
    echo "❌ Direct InfluxDB access still exists"
else
    echo "✅ All data access unified"
fi

# Test data consistency
python3 test_data_consistency.py
```

### Success Criteria
- [ ] ONE data access class exists
- [ ] NO direct InfluxDB queries outside it
- [ ] All components use unified access
- [ ] Data format is consistent everywhere
- [ ] Performance maintained or improved

---

## Phase 4: Dependency Decoupling (Days 8-9)
**Duration:** 2 days
**Objective:** Eliminate circular dependencies and implicit coupling

### Day 1: Map Dependencies

#### Morning: Find Circular Dependencies
```python
# dependency_analyzer.py
import ast
import os
from collections import defaultdict

def find_imports(filepath):
    """Extract imports from a Python file"""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([n.name for n in node.names])
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    return imports

# Build dependency graph
deps = defaultdict(list)
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            deps[filepath] = find_imports(filepath)

# Find circular dependencies
def find_cycles(graph):
    # Implementation to find cycles
    pass

cycles = find_cycles(deps)
print(f"Found {len(cycles)} circular dependencies")
```

#### Afternoon: Design Interfaces
```python
# /interfaces/protocols.py
from typing import Protocol, Dict, Any
import pandas as pd

class DataProvider(Protocol):
    """Interface for data access"""
    def get_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame: ...
    def get_cvd(self, symbol: str) -> pd.DataFrame: ...

class SignalGenerator(Protocol):
    """Interface for signal generation"""
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]: ...

class ConfigProvider(Protocol):
    """Interface for configuration"""
    def get(self, key: str) -> Any: ...
```

### Day 2: Break Circular Dependencies

#### Morning: Dependency Injection
```python
# Before: Creates its own dependencies
class Strategy:
    def __init__(self):
        self.data = DataPipeline()  # Creates dependency
        self.config = Config()       # Creates dependency

# After: Dependencies injected
class Strategy:
    def __init__(self, data: DataProvider, config: ConfigProvider):
        self.data = data      # Injected
        self.config = config  # Injected
```

#### Afternoon: Test Isolation
```python
# Test each component in isolation
def test_strategy_isolation():
    # Mock dependencies
    mock_data = MockDataProvider()
    mock_config = MockConfigProvider()
    
    # Strategy works with mocks
    strategy = Strategy(mock_data, mock_config)
    assert strategy.generate_signal(test_data)
    
    print("✅ Strategy works in isolation")
```

### Success Criteria
- [ ] Zero circular dependencies
- [ ] All components can be tested in isolation
- [ ] Clear interfaces defined
- [ ] Dependency injection used throughout
- [ ] No implicit file dependencies

---

## Phase 5: Testing and Validation (Days 10-11)
**Duration:** 2 days
**Objective:** Ensure all fixes work and nothing broke

### Day 1: Regression Testing

#### Morning: Create Test Suite
```python
# /tests/regression_test.py
import unittest
from datetime import datetime, timedelta

class RegressionTests(unittest.TestCase):
    """Ensure refactoring didn't break anything"""
    
    def test_config_changes_propagate(self):
        """Test that config changes affect all components"""
        # Change config value
        CONFIG.OI_THRESHOLD = 10.0
        
        # Verify all components see change
        from strategies.squeezeflow.strategy import SqueezeFlowStrategy
        strategy = SqueezeFlowStrategy()
        assert strategy.oi_threshold == 10.0
    
    def test_single_visualizer(self):
        """Test only one visualizer exists"""
        import glob
        visualizers = glob.glob("**/visualizer*.py", recursive=True)
        assert len(visualizers) == 1
    
    def test_unified_data_access(self):
        """Test all data comes from one source"""
        # Test implementation
        pass
```

#### Afternoon: Performance Testing
```bash
# Benchmark before and after
time python3 backtest/engine.py --symbol BTC --benchmark

# Memory usage comparison
/usr/bin/time -v python3 backtest/engine.py --symbol BTC

# Generate performance report
python3 generate_performance_report.py
```

### Day 2: Integration Testing

#### Morning: End-to-End Tests
```bash
# Full backtest with all features
INFLUX_HOST=213.136.75.120 python3 backtest/engine.py \
    --symbol BTC \
    --start-date 2025-08-10 \
    --end-date 2025-08-10 \
    --timeframe 1s \
    --strategy SqueezeFlowStrategy \
    --generate-dashboard \
    --validate-output

# Verify dashboard generated correctly
python3 backtest/reporting/visual_validator.py

# Compare with baseline results
python3 compare_with_baseline.py
```

#### Afternoon: Documentation Update
```markdown
# Update documentation
- SYSTEM_TRUTH.md - Document what's fixed
- LESSONS_LEARNED.md - Add refactoring lessons
- README.md - Update with new architecture
- CLAUDE.md - Update with new patterns
```

### Success Criteria
- [ ] All regression tests pass
- [ ] Performance maintained or improved
- [ ] Dashboard generates correctly
- [ ] End-to-end tests pass
- [ ] Documentation updated

---

## Phase 6: Cleanup and Finalization (Day 12)
**Duration:** 1 day
**Objective:** Remove deprecated code and finalize

### Morning: Cleanup
```bash
# Remove deprecated code
rm -rf .deprecated/  # After confirming everything works

# Remove unused imports
autoflake --remove-all-unused-imports --recursive --in-place .

# Format code
black .

# Update requirements
pip freeze > requirements.txt
```

### Afternoon: Final Validation
```bash
# Fresh clone test
git clone . ../squeezeflow_test
cd ../squeezeflow_test
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 backtest/engine.py --test

# Production test
docker-compose down
docker-compose build
docker-compose up -d
./scripts/test_implementation.sh
```

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|------------|
| Week 1 | Config & Visualizers | Single config, one dashboard |
| Week 2 | Data & Dependencies | Unified data access, no circular deps |
| Week 3 | Testing & Cleanup | Full test suite, clean codebase |

**Total Duration:** 12 working days (2.5 weeks)

---

## Risk Mitigation

### Risk: Breaking Production
**Mitigation:** 
- All changes tested in isolation first
- Rollback plan for each phase
- Feature flags for gradual rollout

### Risk: Claude Context Overflow
**Mitigation:**
- Work on one module at a time
- Use grep to find patterns, not full file reads
- Update CLAUDE.md progressively

### Risk: Hidden Dependencies
**Mitigation:**
- Comprehensive dependency scanning
- Integration tests after each phase
- Keep old code in .deprecated/ until confirmed

---

## Success Metrics

### Quantitative
- **-80%** debugging time
- **-90%** configuration cascade points
- **1** visualizer (was 14+)
- **1** data access pattern (was 3+)
- **0** circular dependencies

### Qualitative
- Claude can understand the codebase
- New features don't break existing ones
- Changes require editing ONE file
- Components testable in isolation
- Clear error messages

---

## Claude-Specific Optimizations

### Memory Management
```markdown
# Add to CLAUDE.md after each phase
## Refactoring Progress
- ✅ Phase 1: Config unified
- ✅ Phase 2: Visualizers consolidated
- [ ] Phase 3: Data access standardized

## Key Commands
- Test config: `python3 -c "from backtest.system_config import CONFIG; CONFIG.validate()"`
- Test dashboard: `python3 backtest/reporting/dashboard.py --test`
- Find issues: `./scan_vulnerabilities.sh`
```

### Tool Usage Patterns
```python
# Use parallel tool calls for analysis
# Claude can run these simultaneously:
- grep for patterns
- read specific files
- run test scripts
```

### Context Preservation
```python
# Create summary files after each phase
with open('refactoring_progress.json', 'w') as f:
    json.dump({
        'phase': 'current_phase',
        'completed': ['phase1', 'phase2'],
        'next_steps': ['phase3'],
        'rollback_points': ['git_commit_hash']
    }, f)
```

---

## Conclusion

This surgical refactoring approach:
1. **Fixes the real problems** causing 80% of pain
2. **Works within Claude's limitations** (context, memory)
3. **Keeps the system operational** throughout
4. **Provides measurable progress** at each phase
5. **Can be rolled back** if issues arise

The key difference from the original plan: we're not trying to achieve perfection, we're eliminating specific, documented pain points that waste developer time. This pragmatic approach has a much higher chance of success.

**Ready to execute when you are.**