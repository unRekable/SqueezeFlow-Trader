#!/usr/bin/env python3
"""
SqueezeFlow Trader - Main Entry Point
Docker-ready command interface using only working components
"""

import argparse
import asyncio
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

def is_docker_environment():
    """Check if running in Docker container"""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'

def get_service_hosts():
    """Get service hostnames based on environment"""
    if is_docker_environment():
        return {
            'influx_host': 'aggr-influx',
            'redis_host': 'redis',
            'grafana_host': 'grafana',
            'freqtrade_host': 'freqtrade'
        }
    else:
        return {
            'influx_host': 'localhost',
            'redis_host': 'localhost', 
            'grafana_host': 'localhost',
            'freqtrade_host': 'localhost'
        }

async def start_squeezeflow_calculator(dry_run: bool = True):
    """Start the SqueezeFlow Calculator service"""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"🚀 Starting SqueezeFlow Calculator in {mode} mode")
    
    if is_docker_environment():
        print("📦 Running in Docker container")
    else:
        print("💻 Running in local environment")
    
    try:
        from services.squeezeflow_calculator import SqueezeFlowCalculatorService
        
        # Create and start the calculator service
        calculator_service = SqueezeFlowCalculatorService()
        
        print("⚙️  SqueezeFlow Calculator service initialized")
        print("🔍 Starting CVD-based squeeze detection...")
        print("📊 Monitoring multi-exchange volume flow")
        print("🔄 State machine mode detection enabled")
        print("🎯 Dynamic position sizing active")
        print("⏹️  Press Ctrl+C to stop")
        
        await calculator_service.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SqueezeFlow Calculator...")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting SqueezeFlow Calculator: {e}")
        sys.exit(1)

async def run_backtest(period_key: str = "last_week", balance: float = 10000):
    """Run backtest using the working backtest engine"""
    print(f"📊 Starting SqueezeFlow Backtest")
    print(f"📅 Period: {period_key}")
    print(f"💰 Initial Balance: ${balance:,.2f}")
    
    try:
        # Import and run the backtest function
        from run_backtest import run_quick_backtest
        
        result = await run_quick_backtest(period_key, balance)
        
        if result:
            print("✅ Backtest completed successfully")
            return True
        else:
            print("❌ Backtest failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return False

def check_system_status():
    """Check system status using status.py"""
    print("🔍 Checking SqueezeFlow Trader system status...")
    
    try:
        from status import SystemStatusChecker
        
        checker = SystemStatusChecker()
        results = checker.check_all_services()
        
        if all(results.values()):
            print("✅ All systems operational")
            return True
        else:
            print("⚠️  Some systems have issues")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def validate_system_setup():
    """Validate system setup using validate_setup.py"""
    print("✅ Validating SqueezeFlow Trader setup...")
    
    try:
        from validate_setup import SetupValidator
        
        validator = SetupValidator()
        success = validator.run_validation()
        
        if success:
            print("✅ System validation passed")
            return True
        else:
            print("❌ System validation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def initialize_system(mode: str = "development", force: bool = False):
    """Initialize system using init.py"""
    print(f"🔧 Initializing SqueezeFlow Trader system in {mode} mode...")
    
    try:
        from init import SqueezeFlowInitializer
        
        initializer = SqueezeFlowInitializer()
        initializer.run_initialization(mode=mode, force=force)
        
        print("✅ System initialization completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def run_cvd_analysis(symbol: str, time_range: str, output_file: str = None):
    """Run CVD analysis using utils/cvd_analysis_tool.py"""
    print(f"📊 Running CVD analysis for {symbol.upper()}")
    print(f"📅 Time range: {time_range}")
    
    try:
        # Use subprocess to run the CVD analysis tool
        cmd = [
            sys.executable, 
            str(project_root / "utils" / "cvd_analysis_tool.py"),
            symbol,
            time_range
        ]
        
        if output_file:
            cmd.extend(["--output", output_file])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CVD analysis completed successfully")
            print(result.stdout)
            return True
        else:
            print("❌ CVD analysis failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ CVD analysis error: {e}")
        return False

async def check_state_machine_status():
    """Check and display state machine status"""
    try:
        print("🔄 STATE MACHINE STATUS CHECK")
        print("=" * 50)
        
        # Check if state machine modules are available
        try:
            from perpetual_state_machine_system import PerpetualStateMachine, TradingMode
            from enhanced_mode_system import EnhancedModeSystem
            print("✅ State machine modules: AVAILABLE")
            
            # Test instantiation
            state_machine = PerpetualStateMachine()
            enhanced_system = EnhancedModeSystem()
            print("✅ State machine objects: INITIALIZED")
            
        except ImportError as e:
            print(f"❌ State machine modules: NOT AVAILABLE ({e})")
            return False
        
        # Check Redis connectivity for mode caching
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            print("✅ Redis connection: OK")
            
            # Check for existing mode states
            mode_keys = redis_client.keys("trading_mode:*:current")
            if mode_keys:
                print(f"📊 Active mode states: {len(mode_keys)} symbols")
                for key in mode_keys[:3]:  # Show first 3
                    try:
                        mode_data = redis_client.get(key)
                        if mode_data:
                            import json
                            mode_info = json.loads(mode_data)
                            symbol = key.split(':')[1]
                            print(f"   {symbol}: {mode_info.get('mode', 'UNKNOWN')} "
                                  f"(confidence: {mode_info.get('confidence', 0):.2f})")
                    except:
                        pass
            else:
                print("📊 Active mode states: None found")
                
        except Exception as e:
            print(f"❌ Redis connection: FAILED ({e})")
            return False
        
        # Check InfluxDB for signal storage
        try:
            from influxdb import InfluxDBClient
            influx_client = InfluxDBClient(host='localhost', port=8086, database='significant_trades')
            databases = influx_client.get_list_database()
            print("✅ InfluxDB connection: OK")
            
            # Check for recent signals
            try:
                result = influx_client.query("SELECT COUNT(*) FROM squeeze_signals WHERE time > now() - 1h")
                points = list(result.get_points())
                if points:
                    count = points[0].get('count', 0)
                    print(f"📈 Recent signals (1h): {count}")
                else:
                    print("📈 Recent signals (1h): 0")
            except:
                print("📈 Recent signals: Unable to query")
                
        except Exception as e:
            print(f"❌ InfluxDB connection: FAILED ({e})")
            return False
        
        print("=" * 50)
        print("🎯 STATE MACHINE INTEGRATION: READY")
        return True
        
    except Exception as e:
        print(f"❌ State machine status check failed: {e}")
        return False

def show_docker_info():
    """Show Docker-specific information"""
    hosts = get_service_hosts()
    
    print("🐳 DOCKER ENVIRONMENT DETECTED")
    print("=" * 40)
    print("📊 Service Endpoints:")
    print(f"   InfluxDB: {hosts['influx_host']}:8086")
    print(f"   Redis: {hosts['redis_host']}:6379")
    print(f"   Grafana: {hosts['grafana_host']}:3002")
    print(f"   FreqTrade: {hosts['freqtrade_host']}:8080")
    print()
    print("🔧 Docker Commands:")
    print("   docker-compose logs squeezeflow-calculator")
    print("   docker-compose restart squeezeflow-calculator")
    print("   docker-compose exec squeezeflow-calculator python main.py status")
    print("=" * 40)

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='SqueezeFlow Trader - Cryptocurrency Squeeze Detection Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start --dry-run              # Start calculator in dry-run mode
  %(prog)s start --live                 # Start calculator in live mode  
  %(prog)s backtest last_week           # Run backtest for last week
  %(prog)s backtest last_month 20000    # Run backtest with 20k balance
  %(prog)s status                       # Check system status
  %(prog)s validate                     # Validate setup
  %(prog)s init --mode production       # Initialize for production
  %(prog)s analyze BTCUSDT yesterday    # Analyze BTC CVD for yesterday
  %(prog)s docker-info                  # Show Docker environment info

Time Ranges for backtest/analyze:
  last_week, last_month, yesterday, last_24hours
  Custom: "2025-07-23,2025-07-24"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start SqueezeFlow Calculator')
    start_group = start_parser.add_mutually_exclusive_group(required=True)
    start_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (simulation)'
    )
    start_group.add_argument(
        '--live',
        action='store_true',
        help='Run in live trading mode'
    )
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument(
        'period',
        nargs='?',
        default='last_week',
        help='Time period (default: last_week)'
    )
    backtest_parser.add_argument(
        'balance',
        nargs='?',
        type=float,
        default=10000,
        help='Initial balance (default: 10000)'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    status_parser.add_argument(
        '--state-machine',
        action='store_true',
        help='Check state machine integration status'
    )
    
    # Validate command
    subparsers.add_parser('validate', help='Validate system setup')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize system')
    init_parser.add_argument(
        '--mode',
        choices=['development', 'production', 'docker'],
        default='development',
        help='Setup mode (default: development)'
    )
    init_parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing files'
    )
    
    # CVD Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run CVD analysis')
    analyze_parser.add_argument(
        'symbol',
        help='Trading symbol (e.g., BTCUSDT, ETHUSDT)'
    )
    analyze_parser.add_argument(
        'time_range',
        help='Time range (e.g., yesterday, last_24hours, "2025-07-23,2025-07-24")'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        help='Output file name'
    )
    
    # Docker info command
    subparsers.add_parser('docker-info', help='Show Docker environment information')
    
    return parser

async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show header
    print("🎯 SqueezeFlow Trader - CVD-Based Squeeze Detection")
    print("=" * 55)
    
    if not args.command:
        parser.print_help()
        return
    
    # Show Docker info if in container
    if is_docker_environment() and args.command != 'docker-info':
        print("📦 Running in Docker container")
        print()
    
    try:
        if args.command == 'start':
            dry_run = args.dry_run
            await start_squeezeflow_calculator(dry_run=dry_run)
            
        elif args.command == 'backtest':
            success = await run_backtest(args.period, args.balance)
            if not success:
                sys.exit(1)
            
        elif args.command == 'status':
            if args.state_machine:
                success = await check_state_machine_status()
            else:
                success = check_system_status()
            if not success:
                sys.exit(1)
            
        elif args.command == 'validate':
            success = validate_system_setup()
            if not success:
                sys.exit(1)
            
        elif args.command == 'init':
            success = initialize_system(args.mode, args.force)
            if not success:
                sys.exit(1)
            
        elif args.command == 'analyze':
            success = run_cvd_analysis(args.symbol, args.time_range, args.output)
            if not success:
                sys.exit(1)
            
        elif args.command == 'docker-info':
            show_docker_info()
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        sys.exit(1)
    
    # Run main
    asyncio.run(main())