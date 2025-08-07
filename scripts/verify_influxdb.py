#!/usr/bin/env python3
"""
InfluxDB Verification Script for SqueezeFlow Trader
Checks database setup, connectivity, and data flow
"""

import os
import sys
import time
import logging
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InfluxDBVerifier:
    """Comprehensive InfluxDB verification for SqueezeFlow Trader"""
    
    def __init__(self, host='localhost', port=8086, database='significant_trades'):
        self.host = host
        self.port = port
        self.database = database
        self.client = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('influxdb_verifier')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - INFLUX_VERIFY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def connect(self) -> bool:
        """Connect to InfluxDB"""
        try:
            self.logger.info(f"Connecting to InfluxDB at {self.host}:{self.port}")
            
            self.client = InfluxDBClient(
                host=self.host,
                port=self.port,
                username='',
                password='',
                database=self.database,
                timeout=10
            )
            
            # Test connection
            version_info = self.client.ping()
            self.logger.info(f"✅ Connected to InfluxDB version: {version_info}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            return False
    
    def check_database_structure(self) -> Dict:
        """Check database structure and configuration"""
        results = {
            'databases': [],
            'retention_policies': [],
            'continuous_queries': [],
            'measurements': [],
            'series_count': 0,
            'errors': []
        }
        
        try:
            # Check databases
            databases = self.client.get_list_database()
            results['databases'] = [db['name'] for db in databases]
            
            if self.database not in results['databases']:
                results['errors'].append(f"Database '{self.database}' does not exist")
                return results
            
            # Check retention policies
            try:
                rps = self.client.get_list_retention_policies(self.database)
                results['retention_policies'] = [
                    {
                        'name': rp['name'],
                        'duration': rp['duration'],
                        'default': rp['default'],
                        'replication': rp['replicaN']
                    }
                    for rp in rps
                ]
            except Exception as e:
                results['errors'].append(f"Failed to get retention policies: {e}")
            
            # Check continuous queries
            try:
                cq_result = self.client.get_list_continuous_queries(self.database)
                for db_cqs in cq_result:
                    if db_cqs['name'] == self.database:
                        results['continuous_queries'] = [
                            {
                                'name': cq['name'],
                                'sql': cq.get('query', 'N/A')
                            }
                            for cq in db_cqs.get('cqs', [])
                        ]
                        break
            except Exception as e:
                results['errors'].append(f"Failed to get continuous queries: {e}")
            
            # Check measurements
            try:
                measurements_result = self.client.query(f'SHOW MEASUREMENTS ON "{self.database}"')
                results['measurements'] = [
                    point['name'] for point in measurements_result.get_points()
                ]
            except Exception as e:
                results['errors'].append(f"Failed to get measurements: {e}")
            
            # Check series count
            try:
                series_result = self.client.query(f'SHOW SERIES CARDINALITY ON "{self.database}"')
                series_points = list(series_result.get_points())
                if series_points:
                    results['series_count'] = series_points[0].get('count', 0)
            except Exception as e:
                results['errors'].append(f"Failed to get series count: {e}")
            
        except Exception as e:
            results['errors'].append(f"Database structure check failed: {e}")
        
        return results
    
    def check_data_flow(self) -> Dict:
        """Check if data is flowing into the database"""
        results = {
            'recent_data': {},
            'data_by_timeframe': {},
            'market_coverage': [],
            'cvd_data_available': False,
            'errors': []
        }
        
        try:
            # Check for recent data in each timeframe
            timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']
            
            for tf in timeframes:
                try:
                    # Try different retention policy naming patterns
                    rp_patterns = [f'aggr_{tf}', f'trades_{tf}', 'autogen']
                    measurement_patterns = [f'trades_{tf}', 'trades']
                    
                    data_found = False
                    for rp in rp_patterns:
                        for measurement in measurement_patterns:
                            try:
                                query = f'''
                                    SELECT COUNT(*), MAX(time) as latest_time
                                    FROM "{self.database}"."{rp}"."{measurement}"
                                    WHERE time > now() - 24h
                                '''
                                
                                result = self.client.query(query)
                                points = list(result.get_points())
                                
                                if points and points[0].get('count', 0) > 0:
                                    results['data_by_timeframe'][tf] = {
                                        'count': points[0]['count'],
                                        'latest_time': points[0].get('latest_time'),
                                        'retention_policy': rp,
                                        'measurement': measurement
                                    }
                                    data_found = True
                                    break
                            except:
                                continue
                        
                        if data_found:
                            break
                    
                    if not data_found:
                        results['data_by_timeframe'][tf] = {'count': 0, 'error': 'No data found'}
                
                except Exception as e:
                    results['errors'].append(f"Failed to check {tf} data: {e}")
            
            # Check market coverage
            try:
                query = f'''
                    SELECT DISTINCT(market) 
                    FROM /.*/ 
                    WHERE time > now() - 1h 
                    LIMIT 100
                '''
                
                result = self.client.query(query)
                results['market_coverage'] = [
                    point['market'] for point in result.get_points()
                ]
            except Exception as e:
                results['errors'].append(f"Failed to get market coverage: {e}")
            
            # Check CVD calculation prerequisites
            try:
                # Check if we have vbuy/vsell data
                query = f'''
                    SELECT vbuy, vsell, market 
                    FROM /.*/ 
                    WHERE time > now() - 1h 
                    AND vbuy IS NOT NULL 
                    AND vsell IS NOT NULL 
                    LIMIT 10
                '''
                
                result = self.client.query(query)
                cvd_points = list(result.get_points())
                
                if cvd_points:
                    results['cvd_data_available'] = True
                    results['recent_data']['cvd_sample'] = cvd_points[:3]  # Sample data
                else:
                    results['cvd_data_available'] = False
                    
            except Exception as e:
                results['errors'].append(f"Failed to check CVD data: {e}")
        
        except Exception as e:
            results['errors'].append(f"Data flow check failed: {e}")
        
        return results
    
    def test_queries(self) -> Dict:
        """Test critical queries for strategy operations"""
        results = {
            'query_tests': {},
            'performance_metrics': {},
            'errors': []
        }
        
        # Test queries that the strategy will use
        test_queries = {
            'basic_select': 'SELECT * FROM /.*/ LIMIT 5',
            'time_range_query': 'SELECT * FROM /.*/ WHERE time > now() - 1h LIMIT 10',
            'market_filter': "SELECT * FROM /.*/ WHERE market =~ /.*btc.*/ LIMIT 5",
            'aggregation': 'SELECT COUNT(*) FROM /.*/ WHERE time > now() - 1h GROUP BY market',
            'cvd_calculation': '''
                SELECT market, time, vbuy, vsell, (vbuy - vsell) as volume_delta
                FROM /.*/
                WHERE time > now() - 1h AND vbuy IS NOT NULL AND vsell IS NOT NULL
                LIMIT 10
            '''
        }
        
        for test_name, query in test_queries.items():
            try:
                start_time = time.time()
                result = self.client.query(query)
                query_time = time.time() - start_time
                
                points = list(result.get_points())
                
                results['query_tests'][test_name] = {
                    'success': True,
                    'execution_time_ms': round(query_time * 1000, 2),
                    'points_returned': len(points),
                    'sample_data': points[:2] if points else None
                }
                
            except Exception as e:
                results['query_tests'][test_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def run_comprehensive_check(self) -> Dict:
        """Run comprehensive database verification"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPREHENSIVE INFLUXDB VERIFICATION")
        self.logger.info("=" * 60)
        
        verification_report = {
            'timestamp': datetime.now().isoformat(),
            'connection_status': False,
            'database_structure': {},
            'data_flow': {},
            'query_tests': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        try:
            # Step 1: Test connection
            if not self.connect():
                verification_report['critical_issues'].append("Cannot connect to InfluxDB")
                return verification_report
            
            verification_report['connection_status'] = True
            
            # Step 2: Check database structure
            self.logger.info("Checking database structure...")
            structure_results = self.check_database_structure()
            verification_report['database_structure'] = structure_results
            
            # Analyze structure results
            if not structure_results['databases']:
                verification_report['critical_issues'].append("No databases found")
            elif self.database not in structure_results['databases']:
                verification_report['critical_issues'].append(f"Target database '{self.database}' not found")
            
            if not structure_results['retention_policies']:
                verification_report['critical_issues'].append("No retention policies found")
                verification_report['recommendations'].append("Run setup_influxdb.py to create retention policies")
            else:
                # Check for expected retention policies
                rp_names = [rp['name'] for rp in structure_results['retention_policies']]
                expected_rps = ['aggr_1m', 'aggr_5m', 'aggr_15m', 'aggr_30m', 'aggr_1h', 'aggr_4h']
                missing_rps = [rp for rp in expected_rps if rp not in rp_names]
                
                if missing_rps:
                    verification_report['warnings'].append(f"Missing retention policies: {missing_rps}")
                    verification_report['recommendations'].append("Create missing retention policies for proper timeframe aggregation")
            
            if not structure_results['continuous_queries']:
                verification_report['critical_issues'].append("No continuous queries found")
                verification_report['recommendations'].append("Run setup_influxdb.py to create continuous queries for data aggregation")
            
            # Step 3: Check data flow
            self.logger.info("Checking data flow...")
            data_flow_results = self.check_data_flow()
            verification_report['data_flow'] = data_flow_results
            
            # Analyze data flow
            if not any(tf_data.get('count', 0) > 0 for tf_data in data_flow_results['data_by_timeframe'].values()):
                verification_report['critical_issues'].append("No trading data found in any timeframe")
                verification_report['recommendations'].append("Check aggr-server connection and data collection")
            
            if not data_flow_results['cvd_data_available']:
                verification_report['critical_issues'].append("CVD calculation data (vbuy/vsell) not available")
                verification_report['recommendations'].append("Ensure aggr-server is collecting volume data correctly")
            
            if not data_flow_results['market_coverage']:
                verification_report['warnings'].append("No market data found in recent timeframe")
            
            # Step 4: Test queries
            self.logger.info("Testing critical queries...")
            query_results = self.test_queries()
            verification_report['query_tests'] = query_results
            
            # Analyze query performance
            for test_name, test_result in query_results['query_tests'].items():
                if not test_result.get('success', False):
                    verification_report['warnings'].append(f"Query test '{test_name}' failed: {test_result.get('error', 'Unknown error')}")
                elif test_result.get('execution_time_ms', 0) > 1000:  # > 1 second
                    verification_report['warnings'].append(f"Query '{test_name}' is slow ({test_result['execution_time_ms']}ms)")
            
            # Generate final assessment
            critical_count = len(verification_report['critical_issues'])
            warning_count = len(verification_report['warnings'])
            
            if critical_count == 0 and warning_count == 0:
                status = "✅ EXCELLENT"
                verification_report['overall_status'] = "excellent"
            elif critical_count == 0:
                status = "⚠️  GOOD (with warnings)"
                verification_report['overall_status'] = "good"
            elif critical_count <= 2:
                status = "🔧 NEEDS ATTENTION"
                verification_report['overall_status'] = "needs_attention"
            else:
                status = "❌ CRITICAL ISSUES"
                verification_report['overall_status'] = "critical"
            
            self.logger.info("=" * 60)
            self.logger.info(f"INFLUXDB STATUS: {status}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            verification_report['critical_issues'].append(f"Verification failed: {e}")
            self.logger.error(f"Verification failed: {e}")
        
        finally:
            if self.client:
                self.client.close()
        
        return verification_report
    
    def print_detailed_report(self, report: Dict):
        """Print detailed verification report"""
        print("\n" + "=" * 80)
        print("DETAILED INFLUXDB VERIFICATION REPORT")
        print("=" * 80)
        
        # Connection Status
        print(f"\n🔌 CONNECTION:")
        if report['connection_status']:
            print(f"   ✅ Successfully connected to {self.host}:{self.port}")
        else:
            print(f"   ❌ Failed to connect to {self.host}:{self.port}")
        
        # Database Structure
        structure = report.get('database_structure', {})
        print(f"\n🏗️  DATABASE STRUCTURE:")
        print(f"   📁 Databases: {len(structure.get('databases', []))}")
        print(f"   📋 Retention Policies: {len(structure.get('retention_policies', []))}")
        print(f"   🔄 Continuous Queries: {len(structure.get('continuous_queries', []))}")
        print(f"   📊 Measurements: {len(structure.get('measurements', []))}")
        print(f"   🔢 Series Count: {structure.get('series_count', 0)}")
        
        if structure.get('retention_policies'):
            print("\n   Retention Policies:")
            for rp in structure['retention_policies']:
                default_str = " (DEFAULT)" if rp.get('default') else ""
                print(f"     - {rp['name']}: {rp['duration']}{default_str}")
        
        if structure.get('continuous_queries'):
            print("\n   Continuous Queries:")
            for cq in structure['continuous_queries']:
                print(f"     - {cq['name']}")
        
        # Data Flow
        data_flow = report.get('data_flow', {})
        print(f"\n📊 DATA FLOW:")
        print(f"   🎯 Markets: {len(data_flow.get('market_coverage', []))}")
        print(f"   📈 CVD Data Available: {'✅' if data_flow.get('cvd_data_available') else '❌'}")
        
        if data_flow.get('data_by_timeframe'):
            print("\n   Data by Timeframe:")
            for tf, tf_data in data_flow['data_by_timeframe'].items():
                count = tf_data.get('count', 0)
                status = "✅" if count > 0 else "❌"
                print(f"     {status} {tf}: {count} points")
        
        if data_flow.get('market_coverage'):
            print(f"\n   Sample Markets: {', '.join(data_flow['market_coverage'][:5])}")
            if len(data_flow['market_coverage']) > 5:
                print(f"     ... and {len(data_flow['market_coverage']) - 5} more")
        
        # Query Tests
        query_tests = report.get('query_tests', {}).get('query_tests', {})
        if query_tests:
            print(f"\n🔍 QUERY TESTS:")
            for test_name, test_result in query_tests.items():
                if test_result.get('success'):
                    time_ms = test_result.get('execution_time_ms', 0)
                    points = test_result.get('points_returned', 0)
                    print(f"   ✅ {test_name}: {time_ms}ms, {points} points")
                else:
                    print(f"   ❌ {test_name}: {test_result.get('error', 'Unknown error')}")
        
        # Issues and Recommendations
        if report.get('critical_issues'):
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in report['critical_issues']:
                print(f"   ❌ {issue}")
        
        if report.get('warnings'):
            print(f"\n⚠️  WARNINGS:")
            for warning in report['warnings']:
                print(f"   ⚠️  {warning}")
        
        if report.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   🔧 {rec}")
        
        print("\n" + "=" * 80)


def main():
    """Main verification function"""
    verifier = InfluxDBVerifier(
        host='localhost',
        port=8086,
        database='significant_trades'
    )
    
    # Run comprehensive check
    report = verifier.run_comprehensive_check()
    
    # Print detailed report
    verifier.print_detailed_report(report)
    
    # Save report to file
    report_file = f"data/logs/influxdb_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save report to file: {e}")
    
    # Return appropriate exit code
    if report.get('overall_status') in ['excellent', 'good']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())