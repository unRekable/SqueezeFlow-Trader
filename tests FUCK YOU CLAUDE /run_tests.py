#!/usr/bin/env python3
"""
SqueezeFlow Trader Test Runner
Comprehensive test execution with reporting and coverage
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test categories and their descriptions
TEST_CATEGORIES = {
    'unit': {
        'description': 'Unit tests for individual components',
        'marker': 'unit',
        'files': ['test_strategy_components.py', 'test_data_pipeline.py']
    },
    'integration': {
        'description': 'Integration tests for system workflows',
        'marker': 'integration', 
        'files': ['test_backtest_engine.py']
    },
    'performance': {
        'description': 'Performance and load testing',
        'marker': 'performance',
        'files': ['test_performance.py']
    },
    'property_based': {
        'description': 'Property-based testing with hypothesis',
        'marker': 'property_based',
        'files': ['test_property_based.py']
    },
    'all': {
        'description': 'All test categories',
        'marker': None,
        'files': None
    }
}


class TestRunner:
    """Comprehensive test runner for SqueezeFlow Trader"""
    
    def __init__(self, verbose=False, coverage=False):
        self.verbose = verbose
        self.coverage = coverage
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        
    def run_tests(self, category='all', specific_test=None, parallel=False, 
                  stop_on_fail=False, html_report=False):
        """
        Run tests with specified options
        
        Args:
            category: Test category to run ('unit', 'integration', 'performance', 'property_based', 'all')
            specific_test: Run specific test file or test function
            parallel: Run tests in parallel
            stop_on_fail: Stop on first failure
            html_report: Generate HTML coverage report
        """
        
        print(f"🚀 Starting SqueezeFlow Trader Test Suite")
        print(f"📁 Project Root: {self.project_root}")
        print(f"🧪 Test Directory: {self.test_dir}")
        print(f"📊 Category: {category}")
        print("=" * 60)
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add verbose output
        if self.verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.append('-v')
            
        # Add coverage if requested
        if self.coverage:
            cmd.extend([
                '--cov=strategies',
                '--cov=data', 
                '--cov=backtest',
                '--cov-report=term-missing'
            ])
            
            if html_report:
                cmd.append('--cov-report=html:htmlcov')
                
        # Add parallel execution
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(['-n', 'auto'])
            except ImportError:
                print("⚠️  pytest-xdist not available, running sequentially")
                
        # Stop on first failure
        if stop_on_fail:
            cmd.append('-x')
            
        # Add specific test category or file
        if specific_test:
            if specific_test.endswith('.py') or '::' in specific_test:
                # Specific file or test function
                cmd.append(str(self.test_dir / specific_test))
            else:
                # Assume it's a test function pattern
                cmd.extend(['-k', specific_test])
        elif category != 'all':
            if category in TEST_CATEGORIES:
                marker = TEST_CATEGORIES[category]['marker']
                if marker:
                    cmd.extend(['-m', marker])
                files = TEST_CATEGORIES[category]['files']
                if files:
                    for file in files:
                        cmd.append(str(self.test_dir / file))
            else:
                print(f"❌ Unknown test category: {category}")
                return False
        else:
            # Run all tests in test directory
            cmd.append(str(self.test_dir))
            
        # Add test output options
        cmd.extend([
            '--tb=short',           # Short traceback format
            '--strict-markers',     # Strict marker checking
            '--disable-warnings'    # Disable warnings for cleaner output
        ])
        
        # Execute tests
        print(f"🔧 Running command: {' '.join(cmd)}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=False)
            end_time = time.time()
            duration = end_time - start_time
            
            print("=" * 60)
            print(f"⏱️  Test execution completed in {duration:.2f} seconds")
            
            if result.returncode == 0:
                print("✅ All tests PASSED!")
                if html_report and self.coverage:
                    html_path = self.project_root / 'htmlcov' / 'index.html'
                    print(f"📊 Coverage report: {html_path}")
                return True
            else:
                print(f"❌ Tests FAILED (exit code: {result.returncode})")
                return False
                
        except FileNotFoundError:
            print("❌ pytest not found. Please install with: pip install pytest")
            return False
        except KeyboardInterrupt:
            print("\n⏹️  Test execution interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def run_quick_test(self):
        """Run a quick smoke test to verify basic functionality"""
        print("🏃‍♂️ Running quick smoke test...")
        
        return self.run_tests(
            category='unit',
            specific_test='test_strategy_components.py::TestSqueezeFlowStrategy::test_strategy_initialization',
            stop_on_fail=True
        )
    
    def run_full_suite(self):
        """Run complete test suite with coverage"""
        print("🔬 Running full test suite with coverage...")
        
        return self.run_tests(
            category='all',
            parallel=True,
            html_report=True
        )
    
    def run_performance_suite(self):
        """Run performance tests only"""
        print("⚡ Running performance test suite...")
        
        return self.run_tests(
            category='performance',
            parallel=False  # Performance tests should run sequentially
        )
    
    def list_available_tests(self):
        """List all available test files and categories"""
        print("📋 Available Test Categories:")
        print("=" * 40)
        
        for category, info in TEST_CATEGORIES.items():
            print(f"  {category}: {info['description']}")
            if info['files']:
                for file in info['files']:
                    if (self.test_dir / file).exists():
                        print(f"    ✅ {file}")
                    else:
                        print(f"    ❌ {file} (missing)")
            print()
        
        print("📁 Available Test Files:")
        print("=" * 40)
        
        for test_file in sorted(self.test_dir.glob('test_*.py')):
            print(f"  {test_file.name}")
            
    def check_test_dependencies(self):
        """Check if all required test dependencies are installed"""
        print("🔍 Checking test dependencies...")
        
        required_packages = [
            'pytest',
            'pandas', 
            'numpy',
            'hypothesis',
            'psutil'
        ]
        
        optional_packages = [
            ('pytest-cov', 'Coverage reporting'),
            ('pytest-xdist', 'Parallel test execution'),
            ('pytest-html', 'HTML test reports')
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} (REQUIRED)")
                missing_required.append(package)
        
        for package, description in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✅ {package} ({description})")
            except ImportError:
                print(f"  ⚠️  {package} ({description}) - OPTIONAL")
                missing_optional.append(package)
        
        if missing_required:
            print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
            print("Install with: pip install " + ' '.join(missing_required))
            return False
        
        if missing_optional:
            print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
            print("Install with: pip install " + ' '.join(missing_optional))
        
        print("\n✅ All required dependencies available!")
        return True


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='SqueezeFlow Trader Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                              # Run all tests
  python run_tests.py --category unit              # Run only unit tests
  python run_tests.py --category performance       # Run performance tests
  python run_tests.py --test test_strategy.py      # Run specific test file
  python run_tests.py --test "test_cvd"            # Run tests matching pattern
  python run_tests.py --quick                      # Run quick smoke test
  python run_tests.py --full                       # Run full suite with coverage
  python run_tests.py --list                       # List available tests
  python run_tests.py --check-deps                 # Check dependencies
        """
    )
    
    # Test selection options
    parser.add_argument('--category', '-c', 
                       choices=list(TEST_CATEGORIES.keys()),
                       default='all',
                       help='Test category to run')
    
    parser.add_argument('--test', '-t',
                       help='Specific test file or pattern to run')
    
    # Execution options  
    parser.add_argument('--parallel', '-p',
                       action='store_true',
                       help='Run tests in parallel')
    
    parser.add_argument('--stop-on-fail', '-x',
                       action='store_true', 
                       help='Stop on first failure')
    
    parser.add_argument('--coverage',
                       action='store_true',
                       help='Run with coverage reporting')
    
    parser.add_argument('--html-report',
                       action='store_true',
                       help='Generate HTML coverage report')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    # Convenience options
    parser.add_argument('--quick', '-q',
                       action='store_true',
                       help='Run quick smoke test')
    
    parser.add_argument('--full', '-f',
                       action='store_true', 
                       help='Run full test suite with coverage')
    
    parser.add_argument('--performance',
                       action='store_true',
                       help='Run performance tests only')
    
    # Utility options
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='List available tests')
    
    parser.add_argument('--check-deps',
                       action='store_true',
                       help='Check test dependencies')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose, coverage=args.coverage)
    
    # Handle utility commands
    if args.check_deps:
        return 0 if runner.check_test_dependencies() else 1
    
    if args.list:
        runner.list_available_tests()
        return 0
    
    # Handle convenience commands
    if args.quick:
        success = runner.run_quick_test()
        return 0 if success else 1
    
    if args.full:
        success = runner.run_full_suite()
        return 0 if success else 1
    
    if args.performance:
        success = runner.run_performance_suite()
        return 0 if success else 1
    
    # Run specified tests
    success = runner.run_tests(
        category=args.category,
        specific_test=args.test,
        parallel=args.parallel,
        stop_on_fail=args.stop_on_fail,
        html_report=args.html_report
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())