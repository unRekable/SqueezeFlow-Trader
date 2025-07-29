#!/usr/bin/env python3
"""
SqueezeFlow Trader - System Initialization Script
Handles setup and configuration of the complete trading system
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Try to import yaml, install if missing
try:
    import yaml
except ImportError:
    print("📦 Installing PyYAML...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml


class SqueezeFlowInitializer:
    """Initialize SqueezeFlow Trader system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.required_dirs = [
            "config",
            "data/influxdb",
            "data/redis",
            "data/freqtrade",
            "data/logs",
            "freqtrade/user_data/logs",
            "freqtrade/user_data/strategies", 
            "freqtrade/user_data/data",
            "state",
            "models"
        ]
        
    def run_initialization(self, mode: str = "development", force: bool = False):
        """Run complete system initialization"""
        print("🚀 Initializing SqueezeFlow Trader System")
        print("=" * 50)
        
        try:
            # 1. Check system requirements
            self._check_system_requirements()
            
            # 2. Create directory structure
            self._create_directories()
            
            # 3. Setup configuration files
            self._setup_configuration_files(mode, force)
            
            # 4. Setup environment file
            self._setup_environment_file(mode, force)
            
            # 5. Initialize Python environment
            self._setup_python_environment()
            
            # 6. Setup Docker network
            self._setup_docker_network()
            
            # 7. Initialize databases (if in Docker mode)
            if mode in ["production", "docker"]:
                self._initialize_docker_services()
            
            # 8. Validate setup
            self._validate_setup()
            
            print("\n✅ System initialization completed successfully!")
            self._print_next_steps(mode)
            
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            sys.exit(1)
    
    def _check_system_requirements(self):
        """Check system requirements"""
        print("📋 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise SystemError("Python 3.8 or higher is required")
        print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check Docker (optional for development)
        try:
            subprocess.run(["docker", "--version"], 
                         capture_output=True, check=True)
            print("  ✓ Docker available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠️  Docker not available (OK for development mode)")
        
        # Check docker-compose
        try:
            subprocess.run(["docker-compose", "--version"], 
                         capture_output=True, check=True)
            print("  ✓ Docker Compose available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠️  Docker Compose not available (OK for development mode)")
    
    def _create_directories(self):
        """Create required directory structure"""
        print("📁 Creating directory structure...")
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path}")
    
    def _setup_configuration_files(self, mode: str, force: bool):
        """Setup configuration files"""
        print("⚙️  Setting up configuration files...")
        
        configs = {
            "config.yaml": self._generate_main_config(mode),
            "exchanges.yaml": self._generate_exchange_config(mode),
            "risk_management.yaml": self._generate_risk_config(),
            "execution_config.yaml": self._generate_execution_config(),
            "ml_config.yaml": self._generate_ml_config(),
            "trading_parameters.yaml": self._generate_trading_config(),
            "feature_toggles.yaml": self._generate_feature_config()
        }
        
        for filename, config_data in configs.items():
            config_path = self.config_dir / filename
            
            if config_path.exists() and not force:
                print(f"  ⏭️  {filename} exists (use --force to overwrite)")
                continue
                
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"  ✓ {filename}")
    
    def _generate_main_config(self, mode: str) -> dict:
        """Generate main configuration"""
        return {
            "system": {
                "mode": mode,
                "debug": mode == "development",
                "dry_run": mode != "production"
            },
            "logging": {
                "level": "DEBUG" if mode == "development" else "INFO",
                "file": "data/logs/squeezeflow.log",
                "rotation": {
                    "enabled": True,
                    "size": "100MB",
                    "count": 5
                }
            },
            "influxdb": {
                "host": "localhost" if mode == "development" else "aggr-influx",
                "port": 8086,
                "username": "squeezeflow",
                "password": "password123",
                "database": "significant_trades"
            },
            "redis": {
                "host": "localhost" if mode == "development" else "redis",
                "port": 6379,
                "db": 0
            },
            "notifications": {
                "enabled": False,
                "telegram": {
                    "bot_token": "",
                    "chat_id": ""
                }
            }
        }
    
    def _generate_exchange_config(self, mode: str) -> dict:
        """Generate exchange configuration"""
        return {
            "binance": {
                "enabled": True,
                "api_key": "",
                "api_secret": "",
                "testnet": mode != "production",
                "rate_limit": 1200
            },
            "bybit": {
                "enabled": True,
                "api_key": "",
                "api_secret": "",
                "testnet": mode != "production",
                "rate_limit": 120
            },
            "okx": {
                "enabled": True,
                "api_key": "",
                "api_secret": "",
                "passphrase": "",
                "testnet": mode != "production",
                "rate_limit": 60
            }
        }
    
    def _generate_risk_config(self) -> dict:
        """Generate risk management configuration"""
        return {
            "position_sizing": {
                "max_position_size": 0.02,  # 2% of capital per position
                "max_total_exposure": 0.10,  # 10% total exposure
                "min_position_size": 0.001   # 0.1% minimum
            },
            "stop_loss": {
                "enabled": True,
                "default_pct": 0.02,  # 2% stop loss
                "max_pct": 0.05       # 5% max stop loss
            },
            "take_profit": {
                "enabled": False,  # Using signal-based exits
                "default_pct": 0.04
            },
            "risk_limits": {
                "max_daily_loss": 0.05,     # 5% max daily loss
                "max_drawdown": 0.15,       # 15% max drawdown
                "max_consecutive_losses": 5
            }
        }
    
    def _generate_execution_config(self) -> dict:
        """Generate execution configuration"""
        return {
            "order_types": {
                "entry": "market",
                "exit": "market"
            },
            "slippage": {
                "tolerance": 0.001,  # 0.1% slippage tolerance
                "max_retry": 3
            },
            "timing": {
                "order_timeout": 30,
                "cancel_timeout": 10,
                "retry_delay": 1
            }
        }
    
    def _generate_ml_config(self) -> dict:
        """Generate ML configuration"""
        return {
            "enabled": False,  # Start with rule-based strategy
            "model_type": "lightgbm",
            "features": {
                "technical_indicators": True,
                "volume_profile": True,
                "order_flow": True,
                "sentiment": False
            },
            "training": {
                "lookback_days": 90,
                "retrain_frequency": "weekly",
                "validation_split": 0.2
            }
        }
    
    def _generate_trading_config(self) -> dict:
        """Generate trading parameters"""
        return {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframes": ["1m", "5m", "15m"],
            "squeeze_detection": {
                "cvd_threshold": 1000000,  # $1M CVD difference
                "price_momentum_threshold": 0.005,  # 0.5% price movement
                "volume_threshold": 2.0,   # 2x average volume
                "confirmation_candles": 2
            },
            "filters": {
                "min_volume_24h": 100000000,  # $100M 24h volume
                "volatility_filter": True,
                "trending_only": False
            }
        }
    
    def _generate_feature_config(self) -> dict:
        """Generate feature toggle configuration"""
        return {
            "features": {
                "ml_predictions": False,
                "advanced_risk_management": True,
                "multi_timeframe_analysis": True,
                "telegram_notifications": False,
                "web_dashboard": True,
                "backtesting": True,
                "paper_trading": True
            }
        }
    
    def _setup_environment_file(self, mode: str, force: bool):
        """Setup environment file"""
        print("🔧 Setting up environment file...")
        
        env_file = self.project_root / ".env"
        
        if env_file.exists() and not force:
            print("  ⏭️  .env file exists (use --force to overwrite)")
            return
        
        env_content = f"""# SqueezeFlow Trader Environment Configuration
# Mode: {mode}

# InfluxDB Configuration  
INFLUX_HOST={'localhost' if mode == 'development' else 'aggr-influx'}
INFLUX_PORT=8086
INFLUX_USER=squeezeflow
INFLUX_PASSWORD=password123
INFLUX_DATABASE=significant_trades

# Redis Configuration
REDIS_URL=redis://{'localhost' if mode == 'development' else 'redis'}:6379

# Freqtrade Configuration
FREQTRADE_UI_PASSWORD=squeezeflow123

# System Configuration
NODE_ENV={'development' if mode == 'development' else 'production'}
TZ=UTC
LOG_LEVEL={'DEBUG' if mode == 'development' else 'INFO'}

# Security (Replace with your own values)
SECRET_KEY=your-secret-key-here
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Exchange API Keys (Add your own)
BINANCE_API_KEY=
BINANCE_API_SECRET=
BYBIT_API_KEY=
BYBIT_API_SECRET=
OKX_API_KEY=
OKX_API_SECRET=
OKX_PASSPHRASE=
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("  ✓ .env file created")
    
    def _setup_python_environment(self):
        """Setup Python virtual environment and dependencies"""
        print("🐍 Setting up Python environment...")
        
        # Check if virtual environment exists
        venv_path = self.project_root / ".venv"
        
        if not venv_path.exists():
            print("  📦 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                         check=True)
        
        # Determine pip path
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        # Install requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            print("  📚 Installing dependencies...")
            subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], 
                         check=True)
            print("  ✓ Dependencies installed")
        else:
            print("  ⚠️  requirements.txt not found")
    
    def _setup_docker_network(self):
        """Setup Docker network for services"""
        print("🐳 Setting up Docker network...")
        
        try:
            # Create external network for aggr backend
            subprocess.run([
                "docker", "network", "create", 
                "--driver", "bridge", 
                "aggr_backend"
            ], capture_output=True, check=False)
            print("  ✓ Docker network created")
        except Exception:
            print("  ℹ️  Docker network already exists or Docker unavailable")
    
    def _initialize_docker_services(self):
        """Initialize Docker services"""
        print("🚢 Initializing Docker services...")
        
        try:
            # Start infrastructure services
            subprocess.run([
                "docker-compose", "up", "-d", 
                "aggr-influx", "redis"
            ], check=True, cwd=self.project_root)
            
            print("  ✓ Infrastructure services started")
            
            # Wait a moment for services to be ready
            import time
            time.sleep(10)
            
            # Initialize InfluxDB database
            subprocess.run([
                "docker-compose", "exec", "-T", "aggr-influx",
                "influx", "-execute", 
                "CREATE DATABASE IF NOT EXISTS significant_trades"
            ], check=False, cwd=self.project_root)
            
            print("  ✓ InfluxDB database initialized")
            
            # Setup retention policies and continuous queries
            self._setup_influxdb_policies_and_queries()
            
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Docker services initialization failed: {e}")
        except FileNotFoundError:
            print("  ⚠️  Docker Compose not available")
    
    def _setup_influxdb_policies_and_queries(self):
        """Setup InfluxDB retention policies and continuous queries using industry standards"""
        print("🗄️  Setting up InfluxDB retention policies and continuous queries...")
        
        try:
            import time
            time.sleep(5)  # Allow InfluxDB to fully initialize
            
            # 1. Setup retention policies for efficient storage management
            retention_policies = [
                # 30-day retention for 1-minute data (extended from default 3.5 days)
                "ALTER RETENTION POLICY \"aggr_1m\" ON \"significant_trades\" DURATION 30d",
                # Longer retention for aggregated data
                "ALTER RETENTION POLICY \"aggr_5m\" ON \"significant_trades\" DURATION 90d",
                "ALTER RETENTION POLICY \"aggr_15m\" ON \"significant_trades\" DURATION 180d",
                "ALTER RETENTION POLICY \"aggr_30m\" ON \"significant_trades\" DURATION 365d",
                "ALTER RETENTION POLICY \"aggr_1h\" ON \"significant_trades\" DURATION 730d",
                "ALTER RETENTION POLICY \"aggr_4h\" ON \"significant_trades\" DURATION 1825d",
                "ALTER RETENTION POLICY \"aggr_1d\" ON \"significant_trades\" DURATION 0s"  # Infinite retention for daily data
            ]
            
            print("    📊 Updating retention policies...")
            for policy in retention_policies:
                subprocess.run([
                    "docker-compose", "exec", "-T", "aggr-influx",
                    "influx", "-database", "significant_trades", "-execute", policy
                ], check=False, cwd=self.project_root, capture_output=True)
            
            print("    ✓ Retention policies updated")
            
            # 2. Check existing continuous queries to avoid duplicates
            result = subprocess.run([
                "docker-compose", "exec", "-T", "aggr-influx",
                "influx", "-database", "significant_trades", "-execute", "SHOW CONTINUOUS QUERIES"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            existing_cqs = result.stdout if result.returncode == 0 else ""
            
            # 3. Create missing continuous queries (industry-standard OHLCV aggregation)
            continuous_queries = [
                # 2h aggregation (missing from current setup)
                {
                    "name": "cq_2h",
                    "query": """CREATE CONTINUOUS QUERY "cq_2h" ON "significant_trades"
                    RESAMPLE EVERY 15m FOR 4h
                    BEGIN
                      SELECT first(open) AS open, max(high) AS high, min(low) AS low, last(close) AS close,
                             sum(vbuy) AS vbuy, sum(vsell) AS vsell, sum(cbuy) AS cbuy, sum(csell) AS csell,
                             sum(lbuy) AS lbuy, sum(lsell) AS lsell
                      INTO "aggr_2h"."trades_2h"
                      FROM "aggr_1m"."trades_1m"
                      GROUP BY time(2h), *
                    END"""
                },
                # 6h aggregation
                {
                    "name": "cq_6h", 
                    "query": """CREATE CONTINUOUS QUERY "cq_6h" ON "significant_trades"
                    RESAMPLE EVERY 1h FOR 12h
                    BEGIN
                      SELECT first(open) AS open, max(high) AS high, min(low) AS low, last(close) AS close,
                             sum(vbuy) AS vbuy, sum(vsell) AS vsell, sum(cbuy) AS cbuy, sum(csell) AS csell,
                             sum(lbuy) AS lbuy, sum(lsell) AS lsell
                      INTO "aggr_6h"."trades_6h"
                      FROM "aggr_1m"."trades_1m"
                      GROUP BY time(6h), *
                    END"""
                },
                # 12h aggregation
                {
                    "name": "cq_12h",
                    "query": """CREATE CONTINUOUS QUERY "cq_12h" ON "significant_trades"
                    RESAMPLE EVERY 1h FOR 24h
                    BEGIN
                      SELECT first(open) AS open, max(high) AS high, min(low) AS low, last(close) AS close,
                             sum(vbuy) AS vbuy, sum(vsell) AS vsell, sum(cbuy) AS cbuy, sum(csell) AS csell,
                             sum(lbuy) AS lbuy, sum(lsell) AS lsell
                      INTO "aggr_12h"."trades_12h"
                      FROM "aggr_1m"."trades_1m"
                      GROUP BY time(12h), *
                    END"""
                },
                # 1d (24h) aggregation
                {
                    "name": "cq_1d",
                    "query": """CREATE CONTINUOUS QUERY "cq_1d" ON "significant_trades"
                    RESAMPLE EVERY 2h FOR 48h
                    BEGIN
                      SELECT first(open) AS open, max(high) AS high, min(low) AS low, last(close) AS close,
                             sum(vbuy) AS vbuy, sum(vsell) AS vsell, sum(cbuy) AS cbuy, sum(csell) AS csell,
                             sum(lbuy) AS lbuy, sum(lsell) AS lsell
                      INTO "aggr_1d"."trades_1d"
                      FROM "aggr_1m"."trades_1m"
                      GROUP BY time(1d), *
                    END"""
                }
            ]
            
            print("    🔄 Creating missing continuous queries...")
            queries_created = 0
            for cq in continuous_queries:
                if cq["name"] not in existing_cqs:
                    result = subprocess.run([
                        "docker-compose", "exec", "-T", "aggr-influx",
                        "influx", "-database", "significant_trades", "-execute", cq["query"]
                    ], check=False, cwd=self.project_root, capture_output=True)
                    
                    if result.returncode == 0:
                        queries_created += 1
                        print(f"      ✓ Created {cq['name']}")
                    else:
                        print(f"      ⚠️  Failed to create {cq['name']}: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                else:
                    print(f"      ⏭️  {cq['name']} already exists")
            
            if queries_created > 0:
                print(f"    ✅ Created {queries_created} new continuous queries")
            else:
                print("    ℹ️  All continuous queries already exist")
            
            # 4. Verify setup
            print("    🔍 Verifying setup...")
            subprocess.run([
                "docker-compose", "exec", "-T", "aggr-influx",
                "influx", "-database", "significant_trades", "-execute", "SHOW MEASUREMENTS"
            ], check=False, cwd=self.project_root, capture_output=True)
            
            print("  ✅ InfluxDB setup completed with industry-standard configuration")
            print("     📊 Retention: 1m(30d) → 5m(90d) → 15m(180d) → 1h(2y) → 1d(∞)")
            print("     🔄 Continuous queries: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d")
            
        except Exception as e:
            print(f"    ⚠️  InfluxDB setup warning: {e}")
            print("     💡 You can run this setup manually later if needed")
    
    def _validate_setup(self):
        """Validate the setup"""
        print("✅ Validating setup...")
        
        # Check configuration files
        required_configs = [
            "config.yaml", "exchanges.yaml", "risk_management.yaml"
        ]
        
        for config_file in required_configs:
            config_path = self.config_dir / config_file
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file missing: {config_file}")
        
        print("  ✓ Configuration files present")
        
        # Check environment file
        env_file = self.project_root / ".env"
        if not env_file.exists():
            raise FileNotFoundError(".env file missing")
        
        print("  ✓ Environment file present")
        
        # Check directory structure
        for dir_path in self.required_dirs[:5]:  # Check first 5 critical dirs
            full_path = self.project_root / dir_path
            if not full_path.exists():
                raise FileNotFoundError(f"Required directory missing: {dir_path}")
        
        print("  ✓ Directory structure valid")
    
    def _print_next_steps(self, mode: str):
        """Print next steps for the user"""
        print("\n🎯 Next Steps:")
        print("-" * 30)
        
        if mode == "development":
            print("1. Update API keys in .env file")
            print("2. Start development services:")
            print("   python main.py start --dry-run")
            print("3. Or run specific tests:")
            print("   python main.py test")
        
        elif mode in ["production", "docker"]:
            print("1. Update API keys in .env file")
            print("2. Start all services:")
            print("   ./start.sh")
            print("3. Access dashboards:")
            print("   - Grafana: http://localhost:3002")
            print("   - Freqtrade: http://localhost:8080")
        
        print("\n📚 Documentation:")
        print("- Check config/ directory for all settings")
        print("- View logs in data/logs/")
        print("- Use 'python main.py --help' for commands")


def main():
    """Main initialization function"""
    parser = argparse.ArgumentParser(
        description="Initialize SqueezeFlow Trader System"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["development", "production", "docker"],
        default="development",
        help="Setup mode (default: development)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing configuration files"
    )
    
    args = parser.parse_args()
    
    initializer = SqueezeFlowInitializer()
    initializer.run_initialization(mode=args.mode, force=args.force)


if __name__ == "__main__":
    main()