# SqueezeFlow Trader Test Suite

Comprehensive testing framework for the SqueezeFlow Trader system with emphasis on trading system reliability and data integrity.

## 🧪 Test Architecture

### Test Categories

#### 1. **Unit Tests** (`test_strategy_components.py`)
- Individual component testing
- Strategy phase testing (Phase 1-5)
- Configuration validation
- Mock-based isolated testing

#### 2. **Integration Tests** (`test_backtest_engine.py`)
- End-to-end workflow testing
- Portfolio management integration
- Order execution validation
- System component interaction

#### 3. **Data Pipeline Tests** (`test_data_pipeline.py`)
- Symbol/market discovery testing
- CVD calculation validation
- Data quality assurance
- Pipeline performance testing

#### 4. **Property-Based Tests** (`test_property_based.py`)
- Hypothesis-driven testing
- Mathematical invariant validation
- Edge case discovery
- Robust validation patterns

#### 5. **Performance Tests** (`test_performance.py`)
- Load testing
- Memory profiling
- Throughput benchmarks
- Scalability validation

## 🚀 Quick Start

### Install Dependencies
```bash
# Core testing dependencies
pip install pytest hypothesis psutil

# Optional enhanced features
pip install pytest-cov pytest-xdist pytest-html

# Or install all test dependencies
pip install -r tests/requirements-test.txt
```

### Run Tests

#### Quick Smoke Test
```bash
python tests/run_tests.py --quick
```

#### Run All Tests
```bash
python tests/run_tests.py
```

#### Run Specific Categories
```bash
python tests/run_tests.py --category unit
python tests/run_tests.py --category integration
python tests/run_tests.py --category performance
python tests/run_tests.py --category property_based
```

#### Run with Coverage
```bash
python tests/run_tests.py --coverage --html-report
```

#### Run Specific Tests
```bash
python tests/run_tests.py --test test_strategy_components.py
python tests/run_tests.py --test "test_cvd_calculation"
```

## Prerequisites

### System Requirements
- Python 3.8+
- Redis (localhost:6379 or Docker)
- InfluxDB (localhost:8086 or Docker)
- 2GB+ RAM available
- 5GB+ disk space

### Python Dependencies
```bash
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout
pip install redis influxdb pandas numpy psutil matplotlib fastapi uvicorn
```

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_integration_tests.py

# Run with verbose output and coverage
python tests/run_integration_tests.py --verbose --coverage

# Run specific test suite
python tests/run_integration_tests.py --suite strategy_runner_e2e

# Run in parallel (faster)
python tests/run_integration_tests.py --parallel --skip-slow
```

### Using Docker Test Environment
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up --build

# Run tests in isolated environment
docker-compose -f docker-compose.test.yml run test-runner

# Clean up
docker-compose -f docker-compose.test.yml down -v
```

### Direct Pytest Execution
```bash
# Run specific test file
pytest tests/integration/test_strategy_runner_e2e.py -v

# Run with markers
pytest -m "integration and not slow" -v

# Run with coverage
pytest --cov=services --cov-report=html tests/
```

## Test Suites

### 1. Strategy Runner End-to-End Tests
**File**: `tests/integration/test_strategy_runner_e2e.py`

**Coverage**:
- Complete signal flow: Data → Strategy → Redis → InfluxDB
- Batch signal processing and publishing
- Error scenarios and recovery mechanisms
- Signal validation and deduplication
- Performance benchmarks and resource usage
- Concurrent processing capabilities
- Service lifecycle management

**Key Test Classes**:
- `TestStrategyRunnerE2E`: Core end-to-end functionality
- `TestStrategyRunnerPerformance`: Performance-focused tests
- `TestStrategyRunnerAnalytics`: Analytics and reporting tests

### 2. Monitoring Integration Tests
**File**: `tests/integration/test_monitoring_integration.py`

**Coverage**:
- Health Monitor service integration
- Performance Monitor service integration
- Monitoring service interoperability
- Alert and notification systems
- Dashboard data generation and accuracy

**Key Test Classes**:
- `TestHealthMonitorIntegration`: Health monitoring tests
- `TestPerformanceMonitorIntegration`: Performance monitoring tests
- `TestMonitoringServiceInteroperability`: Inter-service communication
- `TestMonitoringPerformanceBenchmarks`: Monitoring overhead tests

## Test Configuration

### Environment Variables
```bash
# Test mode activation
SQUEEZEFLOW_TEST_MODE=true
SQUEEZEFLOW_LOG_LEVEL=DEBUG

# Service connections
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=15                    # Separate DB for tests

INFLUX_HOST=localhost
INFLUX_PORT=8086
INFLUX_DATABASE=test_squeezeflow

# Docker detection
DOCKER_ENV=true                # Auto-detected in containers
```

### Test Markers
Use pytest markers to categorize and filter tests:

```bash
# Available markers
pytest -m integration         # Integration tests only
pytest -m "not slow"         # Skip slow tests
pytest -m docker             # Docker-specific tests
pytest -m performance        # Performance/benchmark tests
pytest -m monitoring         # Monitoring system tests
```

## Performance Benchmarks

The testing framework includes built-in performance benchmarks:

### Benchmark Thresholds
- **Signal Processing**: < 100ms per signal
- **Redis Publishing**: < 50ms per publish
- **Database Operations**: < 200ms per write
- **Cycle Processing**: < 2 seconds per cycle
- **Memory Usage**: < 500MB maximum
- **Batch Processing**: < 500ms for 10 signals

### Performance Tracking
The test runner automatically tracks:
- Execution time for each test suite
- Peak memory usage during testing
- CPU utilization
- Resource usage patterns

## Test Data and Fixtures

### Key Fixtures (conftest.py)
- `redis_client`: Real Redis connection for integration tests
- `influx_client`: Real InfluxDB connection for integration tests
- `sample_market_data`: Realistic OHLCV data with 500 data points
- `sample_dataset`: Complete dataset with CVD calculations
- `mock_strategy`: Predictable mock strategy for testing
- `performance_monitor`: Performance tracking utilities

### Test Data Generation
```python
# Generate stress test data
stress_signals = generate_stress_test_data(100)  # 100 signals

# Create test signal
test_signal = create_test_signal('BTC', 'LONG', 7.5)

# Wait for conditions
success = wait_for_condition(lambda: condition_check(), timeout=5.0)
```

## Test Reports and Output

### Automated Reports
- **JUnit XML**: `data/test_results/{suite}_results.xml`
- **Coverage HTML**: `data/test_results/{suite}_coverage/`
- **JSON Report**: `data/test_results/integration_test_report.json`
- **Performance Charts**: `data/charts/performance_*.png`

### Report Contents
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "environment": {
    "is_docker": false,
    "memory_gb": 16.0,
    "disk_gb": 500.0
  },
  "performance": {
    "duration_seconds": 120.5,
    "initial_memory_mb": 145.2,
    "peak_memory_mb": 267.8,
    "final_memory_mb": 156.3
  },
  "test_results": {
    "strategy_runner_e2e": true,
    "monitoring_integration": true
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping

# Start Redis with Docker
docker run -d -p 6379:6379 redis:7-alpine

# Use different Redis DB for tests
export REDIS_DB=15
```

#### 2. InfluxDB Connection Failed
```bash
# Check InfluxDB is running
curl -i http://localhost:8086/ping

# Start InfluxDB with Docker
docker run -d -p 8086:8086 influxdb:1.8.10

# Create test database
influx -execute "CREATE DATABASE test_squeezeflow"
```

#### 3. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov

# Verify Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 4. Test Timeouts
```bash
# Increase timeout
python tests/run_integration_tests.py --timeout 600

# Skip slow tests
python tests/run_integration_tests.py --skip-slow
```

#### 5. Memory Issues
```bash
# Monitor memory usage
python tests/run_integration_tests.py --verbose

# Run tests sequentially (less memory)
pytest tests/ --maxfail=1 -x
```

### Debug Mode
```bash
# Enable debug logging
export SQUEEZEFLOW_LOG_LEVEL=DEBUG

# Run single test with debugging
pytest tests/integration/test_strategy_runner_e2e.py::TestStrategyRunnerE2E::test_complete_signal_flow_integration -v -s --pdb
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      influxdb:
        image: influxdb:1.8.10
        ports:
          - 8086:8086
        env:
          INFLUXDB_DB: test_squeezeflow

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run integration tests
      run: python tests/run_integration_tests.py --coverage --parallel
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Docker CI Example
```bash
#!/bin/bash
# ci_test.sh - Continuous Integration Test Script

set -e

echo "Starting SqueezeFlow CI Tests..."

# Build and run test environment
docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Extract test results
docker cp squeezeflow-test-runner:/app/data/test_results ./ci_test_results

# Check results
if [ -f "ci_test_results/integration_test_report.json" ]; then
    echo "✅ Tests completed successfully"
    exit 0
else
    echo "❌ Tests failed"
    exit 1
fi
```

## Performance Optimization

### Test Performance Tips
1. **Use markers** to skip slow tests during development
2. **Enable parallel execution** with `pytest-xdist`
3. **Run specific suites** instead of all tests
4. **Use test database isolation** to prevent conflicts
5. **Clean up resources** after each test

### Resource Management
```python
# Example cleanup in tests
@pytest.fixture
async def cleanup_redis(redis_client):
    yield
    # Cleanup after test
    redis_client.flushdb()

# Memory-conscious testing
def test_with_memory_limit():
    import resource
    # Set memory limit for test
    resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, -1))  # 512MB
```

## Contributing to Tests

### Adding New Tests
1. Create test file in appropriate directory
2. Use descriptive test names: `test_<feature>_<scenario>`
3. Add appropriate markers
4. Include performance assertions
5. Add cleanup fixtures
6. Document complex test scenarios

### Test Guidelines
- **Isolation**: Each test should be independent
- **Deterministic**: Tests should produce consistent results
- **Fast**: Optimize for quick feedback
- **Realistic**: Use realistic test data
- **Comprehensive**: Cover happy path and edge cases

### Example Test Structure
```python
class TestNewFeature:
    """Test new feature functionality"""
    
    @pytest.mark.asyncio
    async def test_feature_happy_path(self, mock_config_manager, redis_client):
        """Test normal operation of new feature"""
        # Arrange
        setup_test_data()
        
        # Act
        result = await execute_feature()
        
        # Assert
        assert result is not None
        assert performance_within_limits()
        
        # Cleanup handled by fixtures
```

---

## Support

For issues with the testing framework:
1. Check this documentation
2. Review test logs in `data/logs/`
3. Check environment with `--skip-env-check` disabled
4. Verify service connections manually
5. Run single tests to isolate issues

For questions or contributions, please refer to the main project documentation.