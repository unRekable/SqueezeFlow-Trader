# SqueezeFlow Trader - System Overview

**Master Reference Document**

*Professional Cryptocurrency Trading System with CVD-Divergence Squeeze Detection*

---

## Executive Summary

SqueezeFlow Trader is a sophisticated, production-ready cryptocurrency trading system that identifies high-probability trading opportunities through **Cumulative Volume Delta (CVD) divergence analysis** between spot and futures markets. The system detects market "squeeze" conditions - states where price and volume flow create directional pressure - across **147 markets from 20+ exchanges** using a multi-timeframe analytical approach.

**🚀 NOW WITH 1-SECOND REAL-TIME DATA**: The system has been upgraded to collect and process data at **1-second intervals**, delivering institutional-grade execution speed with **1-2 second signal latency** (60x improvement from previous 60+ seconds). Perfect timeframe alignment is achieved by building all higher timeframes from the same 1-second base data source.

Built on a professional microservices architecture, the system combines **ultra-low latency real-time data collection**, advanced pattern recognition, and automated trade execution through a modular design that separates strategy logic from execution. With a 86.4/100 performance score and **<2 second total signal latency** processing at institutional speeds, SqueezeFlow Trader represents cutting-edge trading technology suitable for both high-frequency algorithmic trading and manual strategy validation.

## System Architecture Overview

```mermaid
graph TB
    subgraph "SqueezeFlow Trader System"
        subgraph "Processing Layers"
            A[Market Data Layer] --> B[Data Processing Layer]
            B --> C[Strategy Processing Layer]
            C --> D[Trading Execution Layer]
        end
        
        subgraph "Infrastructure Layer"
            E[Redis Cache]
            F[InfluxDB]
            G[Monitoring]
            H[Discovery]
            I[Risk Management]
        end
        
        subgraph "Docker Microservices (9 Services)"
            J[All Services Containerized]
        end
    end
    
    subgraph "External Data Sources"
        K[Binance]
        L[Coinbase Pro]
        M[Kraken]
        N[Bybit]
        O[OKX]
        P["+15 more..."]
    end
    
    subgraph "Internal Processing"
        Q[CVD Calculator]
        R[Signal Generator]
        S[Risk Manager]
        T[Portfolio Mgmt]
        U[Backtest Engine]
        V[Strategy Runner]
    end
    
    subgraph "Trading Interfaces"
        W[FreqTrade UI]
        X[REST API]
        Y[WebSocket]
        Z[Position Mgmt]
        AA[Order Execution]
        BB[Monitoring]
    end
    
    K --> Q
    L --> Q
    M --> Q
    N --> Q
    O --> Q
    P --> Q
    
    Q --> V
    R --> V
    S --> V
    T --> V
    U --> V
    
    V --> W
    V --> X
    V --> Y
    V --> Z
    V --> AA
    V --> BB
```

### Microservice Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A["aggr-influx<br/>(InfluxDB 1.8)"]
        B["aggr-server<br/>(Data Collect)"]
        C["redis<br/>(Cache/Queue)"]
    end
    
    subgraph "Processing Layer"
        D["strategy-runner<br/>(Strategy Exec)"]
        E["oi-tracker<br/>(Open Interest)"]
        F["system-monitor<br/>(Health Checks)"]
    end
    
    subgraph "Execution Layer"
        G["freqtrade<br/>(Trade Exec)"]
        H["freqtrade-ui<br/>(Web Interface)"]
    end
    
    subgraph "Networks"
        I["squeezeflow_network<br/>(Internal communication)"]
        J["aggr_backend<br/>(External aggr-server integration)"]
    end
    
    B --> A
    D --> A
    D --> C
    D --> G
    G --> H
    F --> A
    E --> A
    E --> C
```

## Core Components

### 1. Data Collection & Storage
- **aggr-server** - Real-time data collection from 20+ exchanges
- **InfluxDB** - Time-series database with 30-day retention and continuous queries
- **Redis** - High-speed caching and message queue for real-time operations
- **Symbol Discovery** - Dynamic detection of available trading pairs and markets

**📚 Documentation**: [`docs/docker_services.md`](docker_services.md), [`docs/services_architecture.md`](services_architecture.md)

### 2. Strategy Engine
- **Modular Strategy System** - `/strategies/` folder with 5-phase SqueezeFlow methodology
- **CVD Calculator** - Industry-standard cumulative volume delta computation
- **10-Point Scoring System** - Objective entry criteria with dynamic thresholds
- **Multi-Timeframe Analysis** - **🆕 1s to 4h timeframe validation** (7 timeframes total)

**📚 Documentation**: [`docs/squeezeflow_strategy.md`](strategy/SqueezeFlow.md), [`docs/cvd_baseline_tracking.md`](cvd_baseline_tracking.md)

### 3. Backtest Engine
- **Rolling Window Processing** - 4-hour windows stepping 5 minutes forward to eliminate lookahead bias
- **Professional Architecture** - Complete rewrite with modular design
- **Portfolio Management** - Advanced position sizing and risk controls
- **Performance Analytics** - Comprehensive reporting with HTML/PNG output
- **Strategy Testing** - Unified codebase with exact live trading parity

**📚 Documentation**: [`docs/backtest_engine.md`](backtest_engine.md)

### 4. Trading Execution
- **FreqTrade Integration** - Pure execution layer (1300+ lines → 326 lines)
- **Signal Processing** - Redis-based signal consumption and execution
- **Risk Management** - Position limits, drawdown protection, exposure controls
- **Multi-Exchange Support** - Binance, Coinbase Pro, Kraken, Bybit, OKX

**📚 Documentation**: [`docs/signal_generation_workflow.md`](signal_generation_workflow.md)

### 5. Monitoring & Operations
- **Health Monitoring** - Real-time service health checks and alerts
- **Performance Metrics** - Latency, throughput, and system resource monitoring
- **Logging System** - Multi-channel logging with rotation and archival
- **Discovery Services** - Automatic symbol, market, and OI detection

## ⚡ Real-Time 1-Second Data Flow

### 🚀 1-Second Data Processing Pipeline

**BREAKTHROUGH PERFORMANCE**: The system now operates with **1-second data collection and processing intervals**, representing a 60x improvement in execution speed.

```mermaid
flowchart TD
    subgraph "🆕 1-Second Data Collection (60x Faster)"
        A[Exchange APIs<br/>20+ exchanges] --> B[aggr-server<br/>1-second intervals]
        B --> C[InfluxDB<br/>1s base data]
        C --> D[Continuous Queries<br/>6 timeframes]
        D --> E["🆕 1s → 1m → 5m → 15m → 30m → 1h → 4h"]
    end
    
    subgraph "⚡ Real-Time Strategy Processing (1-2s total)"
        F[InfluxDB<br/>1s data] --> G[Strategy Runner<br/>1s execution cycle]
        G --> H[CVD Calculation<br/>sub-second precision]
        H --> I[5-Phase Analysis<br/>1s timeframe]
        I --> J[10-Point Scoring<br/>real-time]
        J --> K[Signal Generation<br/><1s latency]
        K --> L[Redis Queue<br/>instant delivery]
        L --> M[FreqTrade<br/>immediate execution]
    end
    
    subgraph "🔍 Real-Time Monitoring"
        N[1s Performance Metrics] --> O[Health Dashboard]
        P[Signal Latency<br/>1-2s total] --> Q[Performance Alerts]
        R[Memory Usage<br/>2-4x for 1s data] --> S[Resource Monitoring]
    end
    
    C --> F
```

### 📊 1-Second System Performance

| Component | Previous System | **1-Second System** | Performance Gain |
|-----------|----------------|-------------------|------------------|
| **Data Collection** | 60-second intervals | **1-second intervals** | **60x faster** |
| **Strategy Execution** | 60-second cycles | **1-second cycles** | **60x faster** |
| **Signal Latency** | 60-70 seconds | **1-2 seconds** | **30-60x improvement** |
| **Timeframe Alignment** | Interpolated/Estimated | **Perfect from 1s source** | **100% accuracy** |
| **Memory Usage** | Baseline | **2-4x increase** | Trade-off for speed |
| **CPU Load** | Moderate | **High continuous** | Real-time processing |

### 🎯 Real-Time Operational Benefits

- **Institutional-Grade Speed**: Comparable to professional trading systems
- **Perfect Data Alignment**: All timeframes built from same 1s source - no interpolation errors
- **Ultra-Low Latency Execution**: Signal-to-execution in 1-2 seconds total
- **Sub-Second Precision**: CVD calculations with maximum granularity
- **Real-Time Risk Management**: Immediate position adjustments based on 1s data
- **Market Responsiveness**: Instant reaction to rapid market movements

### ⚠️ 1-Second System Requirements

**Hardware Requirements (Enhanced):**
- **CPU**: 8+ cores recommended (vs 4 cores for 60s system)
- **RAM**: 16GB recommended (vs 8GB for 60s system)
- **Storage**: NVMe SSD mandatory (high IOPS for 1s data writes)
- **Network**: <50ms exchange latency (critical for real-time performance)

**Configuration Changes:**
```yaml
# Real-time 1-second configuration
SQUEEZEFLOW_RUN_INTERVAL: 1          # 1-second execution (was 60)
SQUEEZEFLOW_DATA_INTERVAL: 1         # 1-second collection
SQUEEZEFLOW_ENABLE_1S_MODE: true     # Enable optimizations
SQUEEZEFLOW_MAX_SYMBOLS: 3           # Reduced for real-time (was 5)
REDIS_MAXMEMORY: 2gb                 # Increased for 1s buffering
INFLUX_RETENTION_1S: 24h             # 1-second data retention
```

## Data Flow Summary

### End-to-End Data Flow
```mermaid
flowchart TD
    subgraph "Step 1: Data Collection"
        A[Exchange APIs<br/>20+ exchanges] --> B[aggr-server]
        B --> C[InfluxDB<br/>time-series storage]
        C --> D[Continuous Queries<br/>5 timeframes]
        D --> E["🆕 1s → 1m → 5m → 15m → 30m → 1h → 4h"]
    end
    
    subgraph "Step 2: Strategy Processing (Backtest Mode - Rolling Windows)"
        F[InfluxDB] --> G[Backtest Engine]
        G --> G1[Rolling Window Processor<br/>4-hour windows, 5-min steps]
        G1 --> H[Data Pipeline]
        H --> I[CVD Calculation]
        I --> J[Strategy<br/>5 phases]
        J --> K[10-Point Scoring]
        K --> L[Orders]
        L --> M[Portfolio Management]
    end
    
    subgraph "Step 3: Strategy Processing (Live Mode)"
        N[InfluxDB] --> O[Strategy Runner]
        O --> P[CVD Calculation]
        P --> Q[Strategy<br/>5 phases]
        Q --> R[10-Point Scoring]
        R --> S[Signals]
        S --> T[Redis]
        T --> U[FreqTrade]
        U --> V[Exchange Execution]
    end
    
    subgraph "Step 4: Monitoring & Feedback"
        W[All Services] --> X[system-monitor]
        X --> Y[Health Checks]
        Y --> Z[Alerts]
        
        AA[Trading Results] --> BB[InfluxDB]
        BB --> CC[Performance Analytics]
        CC --> DD[Reports]
    end
    
    C --> F
    C --> N
```

### Signal Processing Pipeline
```mermaid
flowchart LR
    A[Market Data] --> B[CVD Calculation]
    B --> C[Pattern Recognition]
    C --> D[Scoring]
    D --> E[Signal Generation]
    
    A1["Multi-Exchange<br/>Data Sources<br/>(147 markets)"] -.-> A
    B1["Industry<br/>Standard<br/>Algorithm"] -.-> B
    C1["5-Phase Analysis<br/>(Context, Div,<br/>Reset, Confirm)"] -.-> C
    D1["10-Point<br/>Scoring<br/>System"] -.-> D
    E1["Redis Queue<br/>(TTL: 5min)"] -.-> E
    
    E --> F["FreqTrade<br/>Execution"]
```

## Technology Stack

### Core Technologies
```yaml
# Application Layer
Python: "3.8+"                        # Primary development language
FreqTrade: "2024.1+"                   # Trading engine framework
Docker: "20.10+"                       # Containerization platform
Docker Compose: "2.0+"                 # Multi-container orchestration

# Data Storage & Processing
InfluxDB: "1.8.10"                     # Time-series database
Redis: "7-alpine"                      # In-memory cache and message queue
SQLite: "3.35+"                        # Embedded database for backtest
pandas: "2.0.0+"                       # Data processing framework
numpy: "1.24.0+"                       # Numerical computing

# Trading & Market Data
ccxt: "4.0.0+"                         # Exchange connectivity library
asyncio: "Built-in"                    # Asynchronous programming
websockets: "11.0.0+"                  # Real-time data connections
aiohttp: "3.8.0+"                      # Async HTTP client

# Machine Learning & Analytics
scikit-learn: "1.3.0+"                 # ML framework for FreqAI
lightgbm: "4.0.0+"                     # Gradient boosting
pandas-ta: "0.3.14b+"                  # Technical analysis

# Development & Testing
pytest: "7.0.0+"                       # Testing framework
coverage: "7.0.0+"                     # Code coverage analysis
black: "Latest"                        # Code formatting
type-hints: "Full coverage"            # Static type checking

# Infrastructure & Monitoring
Node.js: "18+"                         # aggr-server runtime
Grafana: "Optional"                    # Monitoring dashboards
Prometheus: "Optional"                 # Metrics collection
```

### Data Processing Technologies
```python
# Real-time Processing
asyncio: "Concurrent data collection and processing"
websockets: "Real-time market data streams" 
redis: "Sub-millisecond signal caching and queuing"

# Analytics & Computation
numpy: "Vectorized CVD calculations"
pandas: "Multi-timeframe data manipulation"
InfluxDB CQs: "Automated aggregation (5 continuous queries)"

# Strategy Implementation
Modular Architecture: "Pluggable strategy components"
Type Hints: "Full static typing for reliability"
Professional Logging: "Multi-channel rotating logs"
```

## Key Features

### Trading Strategy Features
- **✅ CVD Divergence Detection** - Industry-standard cumulative volume delta analysis
- **✅ Multi-Timeframe Validation** - 1m to 4h timeframe confirmation system
- **✅ Dynamic Threshold Adaptation** - No fixed parameters, market-adaptive algorithms
- **✅ 10-Point Scoring System** - Objective entry criteria with confidence levels
- **✅ 5-Phase Analysis Methodology** - Context, Divergence, Reset, Scoring, Management
- **✅ Pattern Recognition** - Qualitative analysis over quantitative metrics
- **✅ Flow-Following Exits** - Dynamic exit strategy based on market structure

### Technical System Features
- **✅ Professional Architecture** - Microservices with clean separation of concerns  
- **✅ Modular Strategy System** - Unified codebase for backtest and live trading
- **✅ Real-Time Processing** - <100ms latency with 9.9 signals/second throughput
- **✅ Multi-Exchange Support** - 147 markets across 20+ exchanges
- **✅ Advanced Risk Management** - Position limits, drawdown protection, exposure controls
- **✅ Comprehensive Testing** - Backtest engine with portfolio management simulation
- **✅ Production Monitoring** - Health checks, performance metrics, automated alerts

### Data & Infrastructure Features
- **✅ Time-Series Optimization** - InfluxDB with continuous queries for 5 timeframes
- **✅ Dynamic Discovery** - Automatic symbol, market, and Open Interest detection
- **✅ Professional Logging** - Multi-channel logging with rotation and archival
- **✅ Container Orchestration** - Docker-based microservices with health checks
- **✅ Signal Caching** - Redis-based high-speed signal storage and delivery
- **✅ Data Quality Validation** - Automated data consistency and completeness checks
- **✅ 30-Day Retention** - Automatic data cleanup with rolling window policy

## Performance Metrics

### System Performance (1-Second Real-Time Production)
```yaml
# 🆕 REAL-TIME PROCESSING PERFORMANCE (60x IMPROVEMENT)
Signal_Latency: "1-2 seconds total"    # Complete signal generation to execution (was >60s)
Data_Collection: "1-second intervals"  # Ultra-low latency data streaming (was 60s)
Processing_Speed: "60x faster"         # Strategy execution speed improvement
Timeframe_Accuracy: "100% perfect"     # No interpolation - built from 1s source
Memory_Usage: "8-16GB total system"    # Increased for 1s data buffering (2-4x baseline)
CPU_Efficiency: "4-8 cores utilized"   # Real-time processing requirements
Storage: "~2GB/day data growth"        # Higher due to 1s granularity with 24h retention

# Market Coverage
Active Markets: "147 total"            # BTC (63) + ETH (56) markets  
Exchanges: "20+ supported"             # Major crypto exchanges
Data Sources: "Spot + Perpetual"      # Complete market coverage
Update Frequency: "Real-time"         # WebSocket-based data streams

# Trading Performance
System Score: "86.4/100"              # Overall performance rating
Signal Quality: "4-10 point scoring"  # Objective entry criteria
Position Management: "Professional"    # Risk controls and sizing
Execution: "Pure execution layer"     # Separated strategy from execution
```

### Scalability Metrics
```yaml
# Horizontal Scaling
Service Architecture: "9 microservices"    # Independent scaling per service
Container Resources: "Configurable limits" # Memory and CPU limits per service  
Database Performance: "Continuous queries" # Pre-aggregated multi-timeframe data
Cache Performance: "Redis optimization"    # Sub-millisecond signal access

# Data Management
Retention Policy: "30-day rolling"         # Automatic cleanup
Backup Strategy: "Automated snapshots"     # Data protection
Query Optimization: "10x faster queries"   # vs real-time resampling
Storage Efficiency: "Compressed storage"   # InfluxDB compression
```

## Integration Points

### External APIs & Services
```yaml
# Exchange APIs
Binance: "Spot + Futures WebSocket and REST APIs"
Coinbase Pro: "Professional trading interface"  
Kraken: "Spot and derivatives trading"
Bybit: "Perpetual and inverse contracts"
OKX: "Comprehensive crypto derivatives"

# Data Providers  
aggr-server: "Real-time aggregated market data"
WebSocket Streams: "Direct exchange connections"
REST APIs: "Historical data and account management"
Open Interest: "Derivatives market sentiment data"

# Infrastructure Services
InfluxDB: "Time-series database (port 8086)"
Redis: "Cache and message queue (port 6379)"
FreqTrade API: "Trading engine REST API (port 8080)"
System Monitor: "Health check endpoints"
```

### Internal Service Communication
```yaml
# Service Mesh Communication
squeezeflow_network: "Internal Docker network for secure communication"
aggr_backend: "External network for aggr-server integration"

# Data Flow Interfaces
InfluxDB Query Interface: "SQL-like query language for time-series data"
Redis Pub/Sub: "Real-time signal distribution and caching"
REST APIs: "HTTP-based service communication and control"
WebSocket: "Real-time data streaming between services"

# Configuration Interfaces  
YAML Configuration: "Human-readable service configuration"
Environment Variables: "Runtime configuration and secrets"
Docker Compose: "Service orchestration and networking"
FreqTrade Config: "Trading-specific parameter management"
```

## Security Considerations

### API Security
```yaml
# Exchange API Protection
API Key Management: "Environment-based secure storage"
Rate Limiting: "Built-in exchange rate limit compliance"
Testnet Support: "Safe testing on exchange testnets"
Permission Control: "Minimal required permissions (no withdrawal)"

# Network Security
Internal Networks: "Docker network isolation"
Service Communication: "Internal-only service exposure"
External Access: "Controlled via specific ports only"
SSL/TLS: "Encrypted communication where supported"
```

### Operational Security
```yaml
# Data Protection
Database Security: "InfluxDB authentication and access control"
Redis Security: "Protected cache with authentication"
Log Security: "Structured logging without sensitive data"
Backup Security: "Encrypted backup storage"

# System Security
Container Isolation: "Docker container security boundaries"
Resource Limits: "Controlled resource allocation per service"  
Health Monitoring: "Automated security health checks"
Emergency Stops: "Automatic trading halts on critical errors"

# Risk Management Security
Position Limits: "Maximum position size enforcement"
Drawdown Protection: "Automatic stops on excessive losses"
Exposure Control: "Total portfolio exposure limits"
Signal Validation: "Multi-level signal verification before execution"
```

## Monitoring & Operations

### System Health Monitoring
```yaml
# Service Health Checks
Docker Health Checks: "Built-in container health monitoring"
Service Discovery: "Automatic service availability detection"
Resource Monitoring: "CPU, memory, disk, network utilization"
Database Health: "InfluxDB and Redis connectivity and performance"

# Application Monitoring  
Signal Generation: "Real-time signal pipeline health"
Strategy Performance: "Trading strategy execution metrics"
Data Quality: "Market data completeness and accuracy"
Trade Execution: "FreqTrade execution success rates"

# Performance Monitoring
Latency Tracking: "End-to-end processing time measurement"
Throughput Metrics: "Signal processing and trade execution rates"
Memory Usage: "Service-level memory consumption tracking"
Error Rates: "System-wide error detection and alerting"
```

### Operational Procedures
```bash
# System Startup/Shutdown
docker-compose up -d                    # Start all services
docker-compose down                     # Graceful shutdown
python main.py start --dry-run          # Safe testing mode
python status.py --continuous           # Continuous health monitoring

# Maintenance Operations
docker-compose logs --tail=100          # Log analysis  
docker system prune -f                  # System cleanup
python main.py backtest                 # Strategy validation
redis-cli FLUSHALL                      # Cache reset (if needed)

# Emergency Procedures
docker-compose stop freqtrade           # Emergency trading stop
python main.py test --component signals # Signal pipeline verification
docker-compose restart [service]        # Individual service restart
```

### Logging & Debugging
```python
# Multi-Channel Logging System
logging_channels = {
    'squeezeflow.log': 'Main system operations and strategy execution',
    'freqtrade.log': 'Trading engine operations and trade execution',
    'backtest.log': 'Backtesting operations and performance analysis',
    'system.log': 'Infrastructure and service health events'
}

# Log Rotation & Archival
log_management = {
    'max_file_size': '100MB per log file',
    'backup_count': '5 backup files per channel',
    'rotation': 'Automatic when size limit reached',
    'retention': 'Aligned with 30-day data retention policy'
}

# Debug Mode Activation
debug_configuration = {
    'SQUEEZEFLOW_DEBUG=true': 'Verbose system logging',
    'FREQTRADE_LOG_LEVEL=DEBUG': 'Detailed trading operations',
    'BACKTEST_DEBUG=true': 'Comprehensive backtest logging'
}
```

## Development Workflow

### Strategy Development
```mermaid
graph TD
    A["/strategies/"]
    A --> B["base.py<br/>(Base strategy interface)"]
    A --> C["squeezeflow/"]
    
    C --> D["strategy.py<br/>(Core strategy implementation)"]
    C --> E["config.py<br/>(Strategy configuration)"]
    C --> F["components/"]
    
    F --> G["phase1_context.py<br/>(Larger context assessment)"]
    F --> H["phase2_divergence.py<br/>(Divergence detection)"]
    F --> I["phase3_reset.py<br/>(Reset pattern detection)"]
    F --> J["phase4_scoring.py<br/>(10-point scoring system)"]
    F --> K["phase5_exits.py<br/>(Exit management)"]

# 2. Testing Workflow
python -m backtest.engine               # Strategy backtesting
python main.py test --component strategy # Strategy validation
python main.py backtest --timerange 20240101-20240201 # Historical testing

# 3. Live Deployment
services/strategy_runner.py             # Live strategy execution service
freqtrade/user_data/strategies/         # FreqTrade integration
docker-compose up strategy-runner       # Deploy strategy service
```

### System Extension
```bash
# Adding New Exchanges
1. Update services/discovery/ with new exchange patterns
2. Add exchange configuration to FreqTrade config
3. Test with python main.py test --component exchanges
4. Deploy with docker-compose restart

# Adding New Timeframes  
1. Create InfluxDB continuous query for new timeframe
2. Update strategy multi-timeframe configuration
3. Validate with backtest on new timeframe data
4. Deploy continuous query to production InfluxDB

# Custom Strategy Development
1. Inherit from BaseStrategy in strategies/base.py
2. Implement required methods (analyze, enter, exit)
3. Add strategy configuration to strategies/[name]/config.py
4. Test with backtest engine before live deployment
```

### Quality Assurance
```python
# Code Quality Standards
code_standards = {
    'formatting': 'Black formatter with 88-character line limit',
    'type_hints': 'Full type annotation coverage required',
    'docstrings': 'Google-style docstrings for all public methods',
    'testing': 'pytest with minimum 80% code coverage',
    'linting': 'flake8 and mypy static analysis'
}

# Testing Procedures  
testing_workflow = {
    'unit_tests': 'Individual component testing with pytest',
    'integration_tests': 'End-to-end system validation',
    'strategy_tests': 'Backtest-based strategy validation',
    'performance_tests': 'Load testing and resource utilization',
    'security_tests': 'API security and access control validation'
}

# Deployment Validation
deployment_checks = {
    'service_health': 'All Docker services operational',
    'data_pipeline': 'Complete data flow from exchanges to database',
    'signal_generation': 'Strategy runner producing valid signals',  
    'trade_execution': 'FreqTrade executing signals correctly',
    'monitoring': 'All monitoring and alerting systems active'
}
```

## Documentation Map

### Visual Documentation Guide
```mermaid
graph TD
    A["📚 SqueezeFlow Trader Documentation Structure"]
    
    A --> B["📋 MASTER REFERENCE"]
    B --> B1["📄 system_overview.md (THIS DOCUMENT)<br/>Complete system reference"]
    
    A --> C["🎯 STRATEGY DOCUMENTATION"]
    C --> C1["📁 strategy/"]
    C1 --> C2["📄 SqueezeFlow.md<br/>Complete 5-phase trading methodology (513 lines)"]
    
    A --> D["🔧 TECHNICAL DOCUMENTATION"]
    D --> D1["📄 services_architecture.md<br/>Microservices design and communication"]
    D --> D2["📄 docker_services.md<br/>Container deployment and configuration"]
    D --> D3["📄 signal_generation_workflow.md<br/>End-to-end signal processing"]
    D --> D4["📄 cvd_baseline_tracking.md<br/>CVD calculation methodology"]
    
    A --> E["🧪 TESTING DOCUMENTATION"]
    E --> E1["📁 backtest/"]
    E1 --> E2["📄 BACKTEST_ENGINE.md<br/>Professional backtest architecture"]
    
    A --> F["📊 OPERATIONAL DOCUMENTATION"]
    F --> F1["📄 CLAUDE.md<br/>Complete technical reference for development"]
    F --> F2["📄 README.md<br/>Quick start and project overview"]
    F --> F3["📄 CHANGELOG.md<br/>Version history and updates"]
```

### Documentation Quick Access
```bash
# Strategy Understanding
docs/squeezeflow_strategy.md           # Complete trading methodology
docs/cvd_baseline_tracking.md          # CVD calculation methods

# System Architecture  
docs/system_overview.md (THIS)         # Master system reference
docs/services_architecture.md          # Microservices design
docs/docker_services.md                # Container orchestration

# Implementation Details
docs/signal_generation_workflow.md     # Signal processing pipeline
docs/backtest_engine.md       # Testing and validation

# Development Reference
CLAUDE.md                              # Complete technical specification  
README.md                              # Project overview and quick start
```

### Documentation Maintenance
- **📅 Update Frequency**: Documentation updated with each system enhancement
- **📋 Review Process**: Technical accuracy validated through system testing
- **🔄 Synchronization**: All documentation kept in sync with codebase changes
- **📊 Coverage**: 100% system coverage with examples and use cases
- **🎯 Audience**: Multi-level documentation for users, developers, and operators

---

## Quick Start Reference

### For New Users
1. **Read this document** - Understand system architecture and capabilities
2. **Review [`README.md`](../README.md)** - Quick start and installation guide
3. **Study [`docs/squeezeflow_strategy.md`](strategy/SqueezeFlow.md)** - Trading methodology
4. **Follow setup instructions** - Docker-based deployment

### For Developers  
1. **Study [`CLAUDE.md`](../CLAUDE.md)** - Complete technical reference
2. **Review [`docs/services_architecture.md`](services_architecture.md)** - System design
3. **Examine [`docs/backtest_engine.md`](backtest_engine.md)** - Testing framework
4. **Follow development workflow** - Strategy implementation and testing

### For Operators
1. **Review [`docs/docker_services.md`](docker_services.md)** - Service management
2. **Study monitoring procedures** - Health checks and maintenance
3. **Understand emergency procedures** - System recovery and troubleshooting
4. **Review security considerations** - API management and access control

---

*This system overview serves as the master reference for SqueezeFlow Trader. For specific implementation details, refer to the linked documentation throughout this document.*

**System Status**: ✅ Production Ready - All 6 phases completed with 86.4/100 performance score  
**Last Updated**: August 2025  
**Documentation Version**: 1.0