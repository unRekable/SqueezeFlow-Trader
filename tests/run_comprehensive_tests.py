#!/usr/bin/env python3
"""
Comprehensive Test Runner for SqueezeFlow Trader
Executes all test categories with proper setup and reporting
"""

import sys
import os
import subprocess
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ComprehensiveTestRunner:
    """Comprehensive test execution and reporting"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_categories': {},
            'overall_success': False,
            'execution_time': 0,
            'environment_checks': {}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def run_all_tests(self, categories=None, critical_only=False):
        """Run all test categories or specified ones"""
        start_time = time.time()
        
        # Define test categories
        all_categories = {
            'critical_signal_generation': {
                'description': 'CVD calculation, Phase 2 divergence, signal format',
                'files': [
                    'tests/test_cvd_accuracy.py',
                    'tests/test_phase2_divergence.py', 
                    'tests/test_signal_format.py'
                ],
                'markers': 'critical',
                'timeout': 300,  # 5 minutes
                'priority': 1
            },
            'data_flow_testing': {
                'description': 'aggr-server → InfluxDB → Strategy → Redis flow',
                'files': [
                    'tests/test_data_flow.py',
                    'tests/test_strategy_redis.py',
                    'tests/test_freqtrade_integration.py'
                ],
                'markers': 'integration',
                'timeout': 600,  # 10 minutes
                'priority': 1
            },
            'integration_testing': {
                'description': 'E2E signal flow, concurrent processing, error recovery',
                'files': [
                    'tests/integration/test_e2e_signal_flow.py',
                    'tests/integration/test_concurrent_processing.py',
                    'tests/integration/test_error_recovery.py'
                ],
                'markers': 'integration',
                'timeout': 900,  # 15 minutes
                'priority': 2
            },
            'backtest_engine': {
                'description': 'Backtest data loading, execution, portfolio management',
                'files': [
                    'tests/test_backtest_data_loading.py',
                    'tests/test_backtest_execution.py',
                    'tests/test_portfolio_management.py'
                ],
                'markers': 'backtest',
                'timeout': 300,  # 5 minutes
                'priority': 2
            },
            'performance_testing': {
                'description': 'Load testing, database performance, memory usage',
                'files': [
                    'tests/test_performance_load.py',
                    'tests/test_database_performance.py'
                ],
                'markers': 'performance',
                'timeout': 600,  # 10 minutes
                'priority': 3
            },
            'production_readiness': {
                'description': 'Docker health, resource limits, signal TTL',
                'files': [
                    'tests/test_docker_health.py',
                    'tests/test_resource_monitoring.py',
                    'tests/test_signal_ttl.py'
                ],
                'markers': 'production',
                'timeout': 300,  # 5 minutes
                'priority': 3
            }
        }
        
        # Determine categories to run
        if critical_only:
            categories_to_run = ['critical_signal_generation', 'data_flow_testing']
        elif categories:
            categories_to_run = categories
        else:
            categories_to_run = list(all_categories.keys())
        
        # Sort by priority
        categories_to_run = sorted(categories_to_run, 
                                 key=lambda x: all_categories.get(x, {}).get('priority', 99))
        
        self.logger.info("🚀 Starting Comprehensive SqueezeFlow Trader Testing")
        self.logger.info(f"📊 Running {len(categories_to_run)} test categories")
        print("=" * 80)
        
        # Environment checks first
        if not self._run_environment_checks():
            self.logger.error("❌ Environment checks failed - aborting tests")
            return False
        
        overall_success = True
        
        # Run each category
        for category in categories_to_run:
            if category in all_categories:
                config = all_categories[category]
                
                print(f"\n🧪 Running {category.upper()}")
                print(f"📝 {config['description']}")
                print("-" * 60)
                
                success = self._run_test_category(category, config)
                self.results['test_categories'][category] = {
                    'success': success,
                    'timestamp': datetime.now().isoformat(),
                    'description': config['description']
                }
                
                if not success:
                    overall_success = False
                    print(f"❌ {category} FAILED")
                    
                    # Stop on critical failures
                    if config.get('priority', 3) == 1:
                        self.logger.error(f"🚨 Critical test category failed: {category}")
                        break
                else:
                    print(f"✅ {category} PASSED")
            else:
                self.logger.warning(f"⚠️  Unknown category: {category}")
                
        end_time = time.time()
        
        self.results['execution_time'] = end_time - start_time
        self.results['overall_success'] = overall_success
        
        # Generate report
        self._generate_report()
        
        print("\n" + "=" * 80)
        if overall_success:
            print("✅ ALL TESTS PASSED")
            self.logger.info("All tests completed successfully")
        else:
            print("❌ SOME TESTS FAILED")
            self.logger.error("Some tests failed - check results for details")
            
        print(f"⏱️  Total execution time: {self.results['execution_time']:.2f} seconds")
        
        return overall_success
        
    def _run_environment_checks(self):
        """Run environment prerequisite checks"""
        print("🔍 Running Environment Checks...")
        
        checks = {
            'python_dependencies': self._check_python_deps(),
            'redis_connectivity': self._check_redis(),
            'influxdb_connectivity': self._check_influxdb(), 
            'docker_services': self._check_docker_services(),
            'disk_space': self._check_disk_space(),
            'pytest_available': self._check_pytest()
        }
        
        self.results['environment_checks'] = checks
        
        all_passed = True
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check}: {'PASS' if status else 'FAIL'}")
            if not status:
                all_passed = False
                
        return all_passed
        
    def _check_python_deps(self):
        """Check critical Python dependencies"""
        try:
            import pandas, numpy, pytest
            return True
        except ImportError as e:
            self.logger.error(f"Missing Python dependency: {e}")
            return False
            
    def _check_redis(self):
        """Check Redis connectivity"""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, db=15)
            return client.ping()
        except Exception as e:
            self.logger.warning(f"Redis not available: {e}")
            return False
            
    def _check_influxdb(self):
        """Check InfluxDB connectivity"""
        try:
            from influxdb import InfluxDBClient
            client = InfluxDBClient('localhost', 8086, database='significant_trades')
            client.ping()
            return True
        except Exception as e:
            self.logger.warning(f"InfluxDB not available: {e}")
            return False
            
    def _check_docker_services(self):
        """Check Docker services are running"""
        try:
            result = subprocess.run(['docker-compose', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and 'Up' in result.stdout
        except Exception as e:
            self.logger.warning(f"Docker services check failed: {e}")
            return False
            
    def _check_disk_space(self):
        """Check sufficient disk space"""
        try:
            import shutil
            _, _, free = shutil.disk_usage('.')
            free_gb = free / (1024 ** 3)
            return free_gb > 1.0  # At least 1GB free
        except Exception:
            return True  # Skip if can't check
            
    def _check_pytest(self):
        """Check pytest is available"""
        try:
            result = subprocess.run(['python', '-m', 'pytest', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
            
    def _run_test_category(self, category, config):
        """Run a specific test category"""
        
        # Check if test files exist
        existing_files = []
        missing_files = []
        
        for test_file in config['files']:
            if os.path.exists(test_file):
                existing_files.append(test_file)
            else:
                missing_files.append(test_file)
                
        if missing_files:
            self.logger.warning(f"⚠️  Missing test files for {category}: {missing_files}")
            
        if not existing_files:
            self.logger.error(f"❌ No test files found for {category}")
            return False
        
        # Build pytest command
        cmd = [
            'python', '-m', 'pytest',
            '-v',
            '--tb=short',
            f'--timeout={config["timeout"]}',
            '--maxfail=3',  # Stop after 3 failures
        ]
        
        # Add markers if specified
        if config.get('markers'):
            cmd.extend(['-m', config['markers']])
            
        # Add test files
        cmd.extend(existing_files)
        
        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=config['timeout'])
            
            self.logger.info(f"Return code: {result.returncode}")
            
            # Always show output for better debugging
            if result.stdout:
                print("Test Output:")
                print(result.stdout)
                
            if result.stderr and result.returncode != 0:
                print("Test Errors:")
                print(result.stderr)
                
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Tests timed out after {config['timeout']} seconds")
            return False
        except Exception as e:
            self.logger.error(f"❌ Test execution failed: {e}")
            return False
            
    def _generate_report(self):
        """Generate comprehensive test report"""
        
        # Create results directory
        results_dir = Path('data/test_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report file
        timestamp = int(time.time())
        report_file = results_dir / f"comprehensive_test_report_{timestamp}.json"
        
        # Add summary statistics
        self.results['summary'] = {
            'total_categories': len(self.results['test_categories']),
            'passed_categories': sum(1 for r in self.results['test_categories'].values() if r['success']),
            'failed_categories': sum(1 for r in self.results['test_categories'].values() if not r['success']),
            'environment_checks_passed': sum(1 for r in self.results['environment_checks'].values() if r),
            'environment_checks_failed': sum(1 for r in self.results['environment_checks'].values() if not r)
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            self.logger.info(f"📊 Test report saved to: {report_file}")
            
            # Print summary
            summary = self.results['summary']
            print(f"\n📊 TEST SUMMARY:")
            print(f"   Categories: {summary['passed_categories']}/{summary['total_categories']} passed")
            print(f"   Environment: {summary['environment_checks_passed']}/{summary['environment_checks_passed'] + summary['environment_checks_failed']} checks passed")
            print(f"   Report: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")


def main():
    """Main test execution entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SqueezeFlow Comprehensive Test Runner')
    parser.add_argument('--categories', nargs='+', 
                       help='Test categories to run (default: all)')
    parser.add_argument('--critical-only', action='store_true',
                       help='Run only critical tests')
    parser.add_argument('--list-categories', action='store_true',
                       help='List available test categories')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = ComprehensiveTestRunner()
    
    if args.list_categories:
        print("Available Test Categories:")
        categories = [
            ('critical_signal_generation', 'CVD calculation, Phase 2 divergence, signal format'),
            ('data_flow_testing', 'aggr-server → InfluxDB → Strategy → Redis flow'),
            ('integration_testing', 'E2E signal flow, concurrent processing, error recovery'),
            ('backtest_engine', 'Backtest data loading, execution, portfolio management'),
            ('performance_testing', 'Load testing, database performance, memory usage'),
            ('production_readiness', 'Docker health, resource limits, signal TTL')
        ]
        
        for name, desc in categories:
            print(f"  {name:<30} - {desc}")
        return 0
    
    success = runner.run_all_tests(args.categories, args.critical_only)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()