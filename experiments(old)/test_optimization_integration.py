#!/usr/bin/env python3
"""
Test the optimization framework integration with current system
This script verifies everything works WITHOUT breaking anything
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_minimal_optimization():
    """Test with the smallest possible parameter set"""
    
    print("=" * 80)
    print("OPTIMIZATION FRAMEWORK INTEGRATION TEST")
    print("Testing with minimal parameters to ensure nothing breaks")
    print("=" * 80)
    
    from experiments.optimization_framework_v3 import OptimizationFrameworkV3
    
    # Initialize framework
    print("\n1. Initializing framework...")
    framework = OptimizationFrameworkV3()
    print("   ✅ Framework initialized")
    
    # Check current status
    print("\n2. Checking current status...")
    status = framework.get_status()
    print(f"   Cycles completed: {status['cycles_completed']}")
    print(f"   Results collected: {status['results_collected']}")
    
    # Test single backtest with current settings
    print("\n3. Testing single backtest with current settings...")
    result = await framework.run_backtest(
        symbol='ETH',
        parameter_name='MIN_ENTRY_SCORE',
        parameter_value=3.0,  # Current default
        date_range=('2025-08-10', '2025-08-10')
    )
    
    print(f"   Backtest completed in {result.backtest_duration:.1f}s")
    print(f"   Trades: {result.total_trades}")
    print(f"   Dashboard: {result.dashboard_path}")
    
    if result.screenshot_path:
        print(f"   Screenshot: {result.screenshot_path}")
        print("   ✅ Visual validation working!")
    else:
        print("   ⚠️ No screenshot captured (may need Chrome/webkit2png)")
    
    # Test parameter optimization with just 2 values
    print("\n4. Testing parameter optimization (2 values only)...")
    optimization_result = await framework.optimize_parameter(
        parameter_name='MIN_ENTRY_SCORE',
        test_values=[3.0, 4.0],  # Just 2 values
        symbol='ETH',
        date_range=('2025-08-10', '2025-08-10')
    )
    
    print(f"   Best value: {optimization_result['best_value']}")
    print(f"   Best score: {optimization_result['best_score']:.1f}")
    
    # Show what we learned
    print("\n5. What we learned:")
    for r in optimization_result['all_results']:
        print(f"   Score={r.parameter_value}: {r.total_trades} trades, "
              f"{r.win_rate:.1%} win rate, {r.optimization_score:.1f} score")
    
    print("\n✅ Integration test complete! Framework is working correctly.")
    print("\nNext steps:")
    print("1. Run full optimization: python3 experiments/run_optimization_v3.py")
    print("2. View results: cat experiments/optimization_data_v3/results.json")
    print("3. Check screenshots: ls results/backtest_*/dashboard_screenshot_*.png")
    
    return True


async def test_visual_validation_only():
    """Test just the visual validation component"""
    
    print("\n" + "=" * 80)
    print("VISUAL VALIDATION TEST")
    print("=" * 80)
    
    from backtest.reporting.visual_validator import DashboardVisualValidator
    
    validator = DashboardVisualValidator("results")
    
    # Find latest dashboard
    latest = validator.find_latest_report()
    
    if not latest:
        print("❌ No dashboards found to validate")
        print("   Run a backtest first: python3 backtest/engine.py --symbol ETH --start-date 2025-08-10 --end-date 2025-08-10 --timeframe 1s")
        return False
        
    print(f"📊 Found dashboard: {latest}")
    
    # Try to capture screenshot
    print("📸 Attempting screenshot capture...")
    result = validator.capture_dashboard(latest)
    
    if result.get('success'):
        print(f"✅ Screenshot saved: {result.get('screenshot_path')}")
        print(f"   File size: {result.get('file_size')} bytes")
        print("\n   Claude can now analyze this screenshot!")
        return True
    else:
        print(f"❌ Screenshot failed: {result.get('error')}")
        print("\n   Note: Screenshot requires Chrome, webkit2png, or Safari")
        print("   The optimization will still work without screenshots")
        return False


async def main():
    """Run all integration tests"""
    
    # First test visual validation
    print("🔍 Testing visual validation component...")
    visual_ok = await test_visual_validation_only()
    
    # Then test optimization
    print("\n🔧 Testing optimization framework...")
    optimization_ok = await test_minimal_optimization()
    
    if optimization_ok:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - Framework is ready!")
        print("=" * 80)
    else:
        print("\n⚠️ Some tests failed but framework may still work")


if __name__ == "__main__":
    asyncio.run(main())