#!/usr/bin/env python3
"""
Integration Test Runner for SqueezeFlow Trader
Comprehensive test execution with performance monitoring and reporting

Features:
- Automated test discovery and execution
- Performance benchmarking during tests
- Test result reporting and analysis
- Docker environment detection
- Parallel test execution
- Test coverage analysis
- Performance regression detection
"""

import os
import sys
import time
import json
import argparse
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    import redis
    from influxdb import InfluxDBClient
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies for integration tests: {e}")
    DEPENDENCIES_AVAILABLE = False


class TestEnvironmentChecker:
    """Check test environment prerequisites"""
    
    def __init__(self):
        self.logger = logging.getLogger('test_environment')
        self.checks = []
        self.environment_info = {}
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        is_compatible = version >= (3, 8)
        
        self.checks.append({
            'name': 'Python Version',
            'status': 'PASS' if is_compatible else 'FAIL',
            'details': f"Python {version.major}.{version.minor}.{version.micro}",
            'required': 'Python >= 3.8'
        })
        
        return is_compatible
    
    def check_dependencies(self) -> bool:
        """Check required dependencies"""
        required_packages = [
            'pytest', 'redis', 'influxdb', 'pandas', 'numpy', 
            'psutil', 'asyncio', 'aiohttp'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        is_complete = len(missing_packages) == 0
        
        self.checks.append({
            'name': 'Python Dependencies',
            'status': 'PASS' if is_complete else 'FAIL',
            'details': f"Missing: {missing_packages}" if missing_packages else "All dependencies available",
            'missing': missing_packages
        })
        
        return is_complete
    
    def check_redis_connection(self) -> bool:
        """Check Redis connection"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=15, socket_timeout=3)
            client.ping()
            client.close()
            
            self.checks.append({
                'name': 'Redis Connection',
                'status': 'PASS',
                'details': 'Redis is accessible on localhost:6379'
            })
            return True
            
        except Exception as e:
            self.checks.append({
                'name': 'Redis Connection',
                'status': 'FAIL',
                'details': f'Redis connection failed: {str(e)}'
            })
            return False
    
    def check_influxdb_connection(self) -> bool:
        """Check InfluxDB connection"""
        try:
            client = InfluxDBClient(host='localhost', port=8086, timeout=3)
            client.ping()
            client.close()
            
            self.checks.append({
                'name': 'InfluxDB Connection',
                'status': 'PASS',
                'details': 'InfluxDB is accessible on localhost:8086'
            })
            return True
            
        except Exception as e:
            self.checks.append({
                'name': 'InfluxDB Connection',
                'status': 'FAIL',
                'details': f'InfluxDB connection failed: {str(e)}'
            })
            return False
    
    def check_docker_environment(self) -> bool:
        """Check if running in Docker environment"""
        is_docker = (
            os.path.exists('/.dockerenv') or
            os.environ.get('DOCKER_ENV') == 'true'
        )
        
        self.environment_info['is_docker'] = is_docker
        self.environment_info['redis_host'] = 'redis' if is_docker else 'localhost'
        self.environment_info['influx_host'] = 'aggr-influx' if is_docker else 'localhost'
        
        self.checks.append({
            'name': 'Docker Environment',
            'status': 'INFO',
            'details': f'Running in {"Docker" if is_docker else "local"} environment'
        })
        
        return True
    
    def check_system_resources(self) -> bool:
        """Check system resources"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        memory_gb = memory.total / 1024**3
        disk_gb = disk.total / 1024**3
        
        sufficient_memory = memory_gb >= 2  # 2GB minimum
        sufficient_disk = disk_gb >= 5     # 5GB minimum
        
        self.checks.append({
            'name': 'System Resources',
            'status': 'PASS' if (sufficient_memory and sufficient_disk) else 'WARN',
            'details': f'Memory: {memory_gb:.1f}GB, Disk: {disk_gb:.1f}GB',
            'sufficient': sufficient_memory and sufficient_disk
        })
        
        self.environment_info['memory_gb'] = memory_gb
        self.environment_info['disk_gb'] = disk_gb
        
        return sufficient_memory and sufficient_disk
    
    def run_all_checks(self) -> bool:
        """Run all environment checks"""
        self.logger.info("Running environment checks...")
        
        checks_passed = []
        checks_passed.append(self.check_python_version())
        checks_passed.append(self.check_dependencies())
        checks_passed.append(self.check_redis_connection())
        checks_passed.append(self.check_influxdb_connection())
        checks_passed.append(self.check_docker_environment())
        checks_passed.append(self.check_system_resources())
        
        return all(checks_passed)
    
    def print_check_results(self):
        """Print environment check results"""
        print("\n" + "="*60)
        print("ENVIRONMENT CHECK RESULTS")
        print("="*60)
        
        for check in self.checks:
            status_symbol = {
                'PASS': '✅',
                'FAIL': '❌',
                'WARN': '⚠️',
                'INFO': 'ℹ️'
            }.get(check['status'], '?')
            
            print(f"{status_symbol} {check['name']}: {check['details']}")
        
        print("="*60)


class PerformanceTracker:
    """Track test performance and resource usage"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.initial_memory = None
        self.peak_memory = None
        self.process = psutil.Process()
        self.performance_data = {}
    
    def start_tracking(self):
        """Start performance tracking"""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024**2  # MB
        self.peak_memory = self.initial_memory
    
    def update_peak_memory(self):
        """Update peak memory usage"""
        current_memory = self.process.memory_info().rss / 1024**2  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def end_tracking(self):
        """End performance tracking"""
        self.end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024**2  # MB
        
        self.performance_data = {
            'duration_seconds': self.end_time - self.start_time,
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - self.initial_memory,
            'cpu_percent': self.process.cpu_percent()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'execution_time': f"{self.performance_data['duration_seconds']:.2f}s",
            'memory_usage': {
                'initial': f"{self.performance_data['initial_memory_mb']:.1f}MB",
                'peak': f"{self.performance_data['peak_memory_mb']:.1f}MB",
                'final': f"{self.performance_data['final_memory_mb']:.1f}MB",
                'increase': f"{self.performance_data['memory_increase_mb']:.1f}MB"
            },
            'cpu_usage': f"{self.performance_data['cpu_percent']:.1f}%"
        }


class IntegrationTestRunner:
    """Main integration test runner"""
    
    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logging()
        self.environment_checker = TestEnvironmentChecker()
        self.performance_tracker = PerformanceTracker()
        self.test_results = {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger('integration_test_runner')
    
    def run_tests(self) -> bool:
        """Run integration tests"""
        
        self.logger.info("Starting SqueezeFlow Trader Integration Tests")
        self.performance_tracker.start_tracking()
        
        # Check environment
        if not self._check_environment():
            return False
        
        # Prepare test environment
        if not self._prepare_test_environment():
            return False
        
        # Run test suites
        success = self._execute_test_suites()
        
        # Generate reports
        self._generate_test_reports()
        
        self.performance_tracker.end_tracking()
        self._print_test_summary()
        
        return success
    
    def _check_environment(self) -> bool:
        """Check test environment"""
        if self.args.skip_env_check:
            self.logger.info("Skipping environment checks")
            return True
        
        environment_ok = self.environment_checker.run_all_checks()
        self.environment_checker.print_check_results()
        
        if not environment_ok:
            self.logger.error("Environment checks failed. Use --skip-env-check to proceed anyway.")
            return False
        
        return True
    
    def _prepare_test_environment(self) -> bool:
        """Prepare test environment"""
        try:
            # Create necessary directories
            test_dirs = [
                'data/logs',
                'data/test_results',
                'data/charts'
            ]
            
            for dir_path in test_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Set environment variables for tests
            os.environ['SQUEEZEFLOW_TEST_MODE'] = 'true'
            os.environ['SQUEEZEFLOW_LOG_LEVEL'] = 'DEBUG' if self.args.verbose else 'INFO'
            
            self.logger.info("Test environment prepared")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare test environment: {e}")
            return False
    
    def _execute_test_suites(self) -> bool:
        """Execute test suites"""
        
        test_suites = self._get_test_suites()
        overall_success = True
        
        for suite_name, test_files in test_suites.items():
            if self.args.suite and suite_name not in self.args.suite:
                continue
            
            self.logger.info(f"Running test suite: {suite_name}")
            
            success = self._run_pytest_suite(suite_name, test_files)
            self.test_results[suite_name] = success
            
            if not success:
                overall_success = False
                if not self.args.continue_on_failure:
                    break
            
            # Update peak memory usage
            self.performance_tracker.update_peak_memory()
        
        return overall_success
    
    def _get_test_suites(self) -> Dict[str, List[str]]:
        """Get available test suites"""
        
        test_dir = Path('tests')
        
        suites = {
            'strategy_runner_e2e': [
                'tests/integration/test_strategy_runner_e2e.py'
            ],
            'monitoring_integration': [
                'tests/integration/test_monitoring_integration.py'
            ],
            'full_integration': [
                'tests/integration/test_strategy_runner_e2e.py',
                'tests/integration/test_monitoring_integration.py'
            ]
        }
        
        # Filter existing files
        filtered_suites = {}
        for suite_name, files in suites.items():
            existing_files = [f for f in files if Path(f).exists()]
            if existing_files:
                filtered_suites[suite_name] = existing_files
        
        return filtered_suites
    
    def _run_pytest_suite(self, suite_name: str, test_files: List[str]) -> bool:
        """Run a pytest test suite"""
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test files
        cmd.extend(test_files)
        
        # Add pytest options
        if self.args.verbose:
            cmd.extend(['-v', '-s'])
        
        if self.args.parallel:
            cmd.extend(['-n', 'auto'])  # pytest-xdist for parallel execution
        
        # Add markers
        if self.args.skip_slow:
            cmd.extend(['-m', 'not slow'])
        
        if self.args.docker_only:
            cmd.extend(['-m', 'docker'])
        
        # Output options
        cmd.extend([
            '--tb=short',
            f'--junit-xml=data/test_results/{suite_name}_results.xml',
            '--capture=no' if self.args.verbose else '--capture=sys'
        ])
        
        # Add coverage if requested
        if self.args.coverage:
            cmd.extend([
                '--cov=services',
                '--cov=strategies',
                '--cov=data',
                f'--cov-report=html:data/test_results/{suite_name}_coverage',
                f'--cov-report=term'
            ])
        
        # Run tests
        self.logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.args.timeout)
            end_time = time.time()
            
            # Store results
            suite_result = {
                'return_code': result.returncode,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Log output
            if result.stdout:
                self.logger.info(f"Test output:\n{result.stdout}")
            
            if result.stderr:
                self.logger.warning(f"Test errors:\n{result.stderr}")
            
            success = result.returncode == 0
            
            if success:
                self.logger.info(f"✅ Test suite '{suite_name}' passed in {suite_result['duration']:.2f}s")
            else:
                self.logger.error(f"❌ Test suite '{suite_name}' failed with return code {result.returncode}")
            
            return success
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Test suite '{suite_name}' timed out after {self.args.timeout}s")
            return False
        
        except Exception as e:
            self.logger.error(f"❌ Error running test suite '{suite_name}': {e}")
            return False
    
    def _generate_test_reports(self):
        """Generate test reports"""
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment_checker.environment_info,
            'performance': self.performance_tracker.performance_data,
            'test_results': self.test_results,
            'configuration': {
                'verbose': self.args.verbose,
                'parallel': self.args.parallel,
                'coverage': self.args.coverage,
                'skip_slow': self.args.skip_slow,
                'timeout': self.args.timeout
            }
        }
        
        # Save JSON report
        report_file = Path('data/test_results/integration_test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Test report saved: {report_file}")
    
    def _print_test_summary(self):
        """Print test execution summary"""
        
        performance = self.performance_tracker.get_performance_summary()
        
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        # Test results
        total_suites = len(self.test_results)
        passed_suites = sum(1 for success in self.test_results.values() if success)
        
        print(f"Test Suites: {passed_suites}/{total_suites} passed")
        
        for suite_name, success in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {suite_name}")
        
        # Performance summary
        print(f"\nPerformance:")
        print(f"  Execution Time: {performance['execution_time']}")
        print(f"  Memory Usage: {performance['memory_usage']['initial']} → {performance['memory_usage']['peak']} (peak)")
        print(f"  CPU Usage: {performance['cpu_usage']}")
        
        # Overall result
        overall_success = all(self.test_results.values())
        result_text = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
        print(f"\nOverall Result: {result_text}")
        
        print("="*60)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Run SqueezeFlow Trader Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_integration_tests.py                    # Run all tests
  python tests/run_integration_tests.py --suite strategy_runner_e2e  # Run specific suite
  python tests/run_integration_tests.py --verbose --coverage         # Verbose with coverage
  python tests/run_integration_tests.py --parallel --skip-slow       # Fast parallel execution
        """
    )
    
    # Test selection
    parser.add_argument('--suite', action='append', help='Run specific test suite(s)')
    parser.add_argument('--skip-slow', action='store_true', help='Skip slow tests')
    parser.add_argument('--docker-only', action='store_true', help='Run only Docker tests')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--continue-on-failure', action='store_true', help='Continue running tests after failure')
    
    # Environment options
    parser.add_argument('--skip-env-check', action='store_true', help='Skip environment checks')
    parser.add_argument('--timeout', type=int, default=300, help='Test timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available for integration tests")
        print("Install with: pip install pytest pytest-asyncio pytest-cov redis influxdb")
        sys.exit(1)
    
    # Run tests
    runner = IntegrationTestRunner(args)
    success = runner.run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()