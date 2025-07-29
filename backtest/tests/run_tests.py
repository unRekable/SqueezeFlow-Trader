#!/usr/bin/env python3
"""
Test Runner for SqueezeFlow Backtest Framework
Runs all unit tests and generates coverage reports
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add backtest directory to path
backtest_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backtest_dir)


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_start_time = None
        self.verbosity = verbosity  # Store verbosity for use in methods
    
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()
        if self.verbosity > 1:
            self.stream.write(f"  🧪 {test._testMethodName} ... ")
            self.stream.flush()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            elapsed = time.time() - self.test_start_time
            self.stream.writeln(f"✅ ({elapsed:.3f}s)")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            elapsed = time.time() - self.test_start_time
            self.stream.writeln(f"❌ ERROR ({elapsed:.3f}s)")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            elapsed = time.time() - self.test_start_time
            self.stream.writeln(f"❌ FAIL ({elapsed:.3f}s)")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            elapsed = time.time() - self.test_start_time
            self.stream.writeln(f"⏭️ SKIP ({elapsed:.3f}s) - {reason}")


class TestRunner:
    """Enhanced test runner for the backtest framework"""
    
    def __init__(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_modules = [
            'test_portfolio',
            'test_fees', 
            'test_strategies'
        ]
    
    def discover_tests(self):
        """Discover all test cases"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for module_name in self.test_modules:
            try:
                # Import the test module
                module = __import__(module_name, fromlist=[''])
                
                # Load tests from the module
                module_suite = loader.loadTestsFromModule(module)
                suite.addTests(module_suite)
                
                print(f"✅ Loaded tests from {module_name}")
                
            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {e}")
            except Exception as e:
                print(f"❌ Error loading {module_name}: {e}")
        
        return suite
    
    def run_tests(self, verbosity=2):
        """Run all tests with colored output"""
        print("🧪 SqueezeFlow Backtest Framework - Unit Tests")
        print("=" * 60)
        
        # Discover tests
        test_suite = self.discover_tests()
        test_count = test_suite.countTestCases()
        
        if test_count == 0:
            print("❌ No tests found!")
            return False
        
        print(f"📊 Found {test_count} test cases\n")
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            resultclass=ColoredTextTestResult,
            stream=sys.stdout
        )
        
        start_time = time.time()
        result = runner.run(test_suite)
        end_time = time.time()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        success = total_tests - failures - errors - skipped
        
        print(f"✅ Successful: {success}/{total_tests}")
        print(f"❌ Failed: {failures}/{total_tests}")
        print(f"💥 Errors: {errors}/{total_tests}")
        print(f"⏭️ Skipped: {skipped}/{total_tests}")
        print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
        
        success_rate = (success / total_tests * 100) if total_tests > 0 else 0
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        # Print detailed failures/errors
        if result.failures:
            print(f"\n❌ FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                # Extract error message without backslashes in f-string
                error_parts = traceback.split('AssertionError: ')[-1].split('\n')
                error_msg = error_parts[0] if error_parts else "Unknown error"
                print(f"  • {test}: {error_msg}")
        
        if result.errors:
            print(f"\n💥 ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                # Extract error message without backslashes in f-string
                if '\n' in traceback:
                    error_msg = traceback.split('\n')[-2]
                else:
                    error_msg = traceback
                print(f"  • {test}: {error_msg}")
        
        # Return True if all tests passed
        return failures == 0 and errors == 0
    
    def run_specific_test(self, test_class, test_method=None, verbosity=2):
        """Run a specific test class or method"""
        loader = unittest.TestLoader()
        
        if test_method:
            suite = loader.loadTestsFromName(f"{test_class}.{test_method}")
            print(f"🧪 Running specific test: {test_class}.{test_method}")
        else:
            suite = loader.loadTestsFromName(test_class)
            print(f"🧪 Running test class: {test_class}")
        
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            resultclass=ColoredTextTestResult
        )
        
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    
    def run_with_coverage(self):
        """Run tests with coverage analysis (if coverage.py is available)"""
        try:
            import coverage
            
            print("📊 Running tests with coverage analysis...")
            
            # Start coverage
            cov = coverage.Coverage(source=[backtest_dir])
            cov.start()
            
            # Run tests
            success = self.run_tests(verbosity=1)
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            print("\\n📊 COVERAGE REPORT:")
            print("=" * 40)
            cov.report(show_missing=True)
            
            # Generate HTML report if possible
            try:
                html_dir = os.path.join(self.test_dir, 'coverage_html')
                cov.html_report(directory=html_dir)
                print(f"\\n📄 HTML coverage report generated: {html_dir}/index.html")
            except Exception as e:
                print(f"⚠️ Could not generate HTML report: {e}")
            
            return success
            
        except ImportError:
            print("⚠️ coverage.py not installed. Running tests without coverage...")
            print("💡 Install with: pip install coverage")
            return self.run_tests()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SqueezeFlow backtest tests")
    parser.add_argument(
        '--coverage', 
        action='store_true', 
        help='Run with coverage analysis'
    )
    parser.add_argument(
        '--test', 
        help='Run specific test class (e.g., test_portfolio.TestPortfolioManager)'
    )
    parser.add_argument(
        '--method', 
        help='Run specific test method (requires --test)'
    )
    parser.add_argument(
        '--verbosity', '-v', 
        type=int, 
        default=2, 
        help='Test verbosity level (0-2)'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.test:
        success = runner.run_specific_test(args.test, args.method, args.verbosity)
    elif args.coverage:
        success = runner.run_with_coverage()
    else:
        success = runner.run_tests(args.verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()