#!/usr/bin/env python3
"""
Strategy Runner Service Launcher

Usage:
    python run_strategy_runner.py                    # Run with default config
    python run_strategy_runner.py --config custom    # Run with custom config
    python run_strategy_runner.py --health           # Check service health
    python run_strategy_runner.py --test-signals     # Test signal generation
"""

import sys
import argparse
import asyncio
import json
from pathlib import Path

from services.strategy_runner import StrategyRunner
from services.config.service_config import ConfigManager


async def run_service(config_dir: str = None):
    """Run the strategy runner service"""
    
    print("🚀 Starting SqueezeFlow Strategy Runner Service...")
    
    # Initialize configuration
    config_manager = ConfigManager(config_dir) if config_dir else ConfigManager()
    
    # Create and start service
    runner = StrategyRunner(config_manager)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        print("\n⏹️  Received shutdown signal...")
    except Exception as e:
        print(f"❌ Service error: {e}")
        return 1
    finally:
        await runner.stop()
    
    return 0


async def check_health(config_dir: str = None):
    """Check service health status"""
    
    print("🔍 Checking Strategy Runner Service Health...")
    
    # Initialize configuration
    config_manager = ConfigManager(config_dir) if config_dir else ConfigManager()
    
    # Create service (don't start)
    runner = StrategyRunner(config_manager)
    
    try:
        # Test connections
        if not await runner._test_connections():
            print("❌ Connection tests failed")
            return 1
        
        # Get configuration status
        config = config_manager.get_config()
        pairs = config_manager.get_freqtrade_pairs()
        
        print("✅ All connections successful")
        print(f"📊 Configuration: {config.run_interval_seconds}s intervals, {len(pairs)} symbols")
        print(f"🎯 Trading symbols: {', '.join(pairs)}")
        print(f"⚙️  Timeframe: {config.default_timeframe}, Lookback: {config.data_lookback_hours}h")
        
        return 0
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1


async def test_signal_generation(config_dir: str = None):
    """Test signal generation without publishing"""
    
    print("🧪 Testing Signal Generation...")
    
    # Initialize configuration
    config_manager = ConfigManager(config_dir) if config_dir else ConfigManager()
    config = config_manager.get_config()
    
    # Temporarily disable publishing
    config.publish_to_redis = False
    config.store_in_influxdb = False
    
    # Create service
    runner = StrategyRunner(config_manager)
    
    try:
        # Test connections
        if not await runner._test_connections():
            print("❌ Connection tests failed")
            return 1
        
        # Get symbols
        symbols = config_manager.get_freqtrade_pairs()
        print(f"📊 Testing with symbols: {', '.join(symbols)}")
        
        # Run one cycle
        await runner._run_strategy_cycle(symbols[:2])  # Test with first 2 symbols
        
        # Show results
        stats = runner.performance_stats
        print(f"✅ Test completed - Signals: {stats['signals_generated']}, Errors: {stats['errors_encountered']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Signal test failed: {e}")
        return 1


def create_sample_config():
    """Create sample configuration file"""
    
    print("📄 Creating sample configuration...")
    
    config_manager = ConfigManager()
    config_manager.create_sample_config_file()
    
    print("✅ Sample configuration created at: services/config/service_config.yaml")
    print("📝 Edit the file to customize settings, then run the service again.")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="SqueezeFlow Strategy Runner Service")
    parser.add_argument("--config", help="Custom configuration directory")
    parser.add_argument("--health", action="store_true", help="Check service health")
    parser.add_argument("--test-signals", action="store_true", help="Test signal generation")
    parser.add_argument("--create-config", action="store_true", help="Create sample config file")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_sample_config()
        return 0
    
    # Run async commands
    try:
        if args.health:
            return asyncio.run(check_health(args.config))
        elif args.test_signals:
            return asyncio.run(test_signal_generation(args.config))
        else:
            return asyncio.run(run_service(args.config))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())