#!/usr/bin/env python3
"""
InfluxDB Setup Script for SqueezeFlow Trader
Fixes critical database architecture issues and sets up proper timeframe aggregation
"""

import os
import sys
import time
import logging
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InfluxDBSetup:
    """Complete InfluxDB setup and configuration for SqueezeFlow Trader"""
    
    def __init__(self, host='localhost', port=8086, database='significant_trades'):
        """Initialize InfluxDB setup manager"""
        self.host = host
        self.port = port  # Using unified port 8086 after migration
        self.database = database
        self.client = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Timeframe configuration matching aggr-server structure
        self.timeframes = {
            '1m': {
                'retention': '7d',        # 1-minute data for 7 days
                'duration_ms': 60000,
                'base_measurement': 'trades_1m'
            },
            '5m': {
                'retention': '30d',       # 5-minute data for 30 days
                'duration_ms': 300000,
                'base_measurement': 'trades_5m'
            },
            '15m': {
                'retention': '90d',       # 15-minute data for 90 days
                'duration_ms': 900000,
                'base_measurement': 'trades_15m'
            },
            '30m': {
                'retention': '180d',      # 30-minute data for 180 days
                'duration_ms': 1800000,
                'base_measurement': 'trades_30m'
            },
            '1h': {
                'retention': '8760h',     # 1-hour data for 1 year (365*24h)
                'duration_ms': 3600000,
                'base_measurement': 'trades_1h'
            },
            '4h': {
                'retention': '43800h',    # 4-hour data for 5 years (5*365*24h)
                'duration_ms': 14400000,
                'base_measurement': 'trades_4h'
            }
        }
        
        # Retention policy prefix matching aggr-server
        self.rp_prefix = "aggr_"
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for database operations"""
        logger = logging.getLogger('influxdb_setup')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - INFLUX_SETUP - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def connect(self, max_retries: int = 5) -> bool:
        """Connect to InfluxDB with retry logic"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Connecting to InfluxDB at {self.host}:{self.port} (attempt {attempt + 1})")
                
                self.client = InfluxDBClient(
                    host=self.host,
                    port=self.port,
                    username='',
                    password='',
                    database=self.database,
                    timeout=30
                )
                
                # Test connection
                version_info = self.client.ping()
                self.logger.info(f"Connected to InfluxDB version: {version_info}")
                
                return True
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    self.logger.error("Failed to connect to InfluxDB after all retries")
                    return False
        
        return False
    
    def ensure_database(self) -> bool:
        """Ensure database exists"""
        try:
            databases = [db['name'] for db in self.client.get_list_database()]
            
            if self.database not in databases:
                self.logger.info(f"Creating database: {self.database}")
                self.client.create_database(self.database)
            else:
                self.logger.info(f"Database {self.database} already exists")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ensuring database: {e}")
            return False
    
    def create_retention_policies(self) -> bool:
        """Create retention policies for all timeframes"""
        try:
            self.logger.info("Creating retention policies...")
            
            # Get existing retention policies
            existing_rps = {}
            try:
                rps = self.client.get_list_retention_policies(self.database)
                existing_rps = {rp['name']: rp for rp in rps}
                self.logger.info(f"Found {len(existing_rps)} existing retention policies")
            except Exception as e:
                self.logger.warning(f"Could not get existing retention policies: {e}")
            
            success_count = 0
            
            for timeframe, config in self.timeframes.items():
                rp_name = f"{self.rp_prefix}{timeframe}"
                duration = config['retention']
                
                try:
                    if rp_name in existing_rps:
                        self.logger.info(f"Retention policy {rp_name} already exists")
                        
                        # Check if we need to update it
                        existing_duration = existing_rps[rp_name]['duration']
                        if existing_duration != duration:
                            self.logger.info(f"Updating retention policy {rp_name}: {existing_duration} -> {duration}")
                            self.client.alter_retention_policy(
                                name=rp_name,
                                duration=duration,
                                database=self.database
                            )
                    else:
                        self.logger.info(f"Creating retention policy: {rp_name} (duration: {duration})")
                        self.client.create_retention_policy(
                            name=rp_name,
                            duration=duration,
                            replication=1,
                            database=self.database,
                            default=(timeframe == '1m')  # Make 1m the default
                        )
                    
                    success_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to create/update retention policy {rp_name}: {e}")
            
            self.logger.info(f"Successfully processed {success_count}/{len(self.timeframes)} retention policies")
            return success_count == len(self.timeframes)
            
        except Exception as e:
            self.logger.error(f"Error creating retention policies: {e}")
            return False
    
    def create_continuous_queries(self) -> bool:
        """Create continuous queries for automatic data aggregation"""
        try:
            self.logger.info("Creating continuous queries...")
            
            # Get existing continuous queries
            existing_cqs = set()
            try:
                cq_result = self.client.get_list_continuous_queries(self.database)
                for db_cqs in cq_result:
                    if db_cqs['name'] == self.database:
                        existing_cqs = {cq['name'] for cq in db_cqs.get('cqs', [])}
                        break
                self.logger.info(f"Found {len(existing_cqs)} existing continuous queries")
            except Exception as e:
                self.logger.warning(f"Could not get existing continuous queries: {e}")
            
            # Define continuous queries for aggregation chain
            continuous_queries = [
                {
                    'name': 'cq_5m_aggregation',
                    'source_rp': f'{self.rp_prefix}1m',
                    'target_rp': f'{self.rp_prefix}5m',
                    'source_measurement': 'trades_1m',
                    'target_measurement': 'trades_5m',
                    'interval': '5m'
                },
                {
                    'name': 'cq_15m_aggregation',
                    'source_rp': f'{self.rp_prefix}5m',
                    'target_rp': f'{self.rp_prefix}15m',
                    'source_measurement': 'trades_5m',
                    'target_measurement': 'trades_15m',
                    'interval': '15m'
                },
                {
                    'name': 'cq_30m_aggregation',
                    'source_rp': f'{self.rp_prefix}15m',
                    'target_rp': f'{self.rp_prefix}30m',
                    'source_measurement': 'trades_15m',
                    'target_measurement': 'trades_30m',
                    'interval': '30m'
                },
                {
                    'name': 'cq_1h_aggregation',
                    'source_rp': f'{self.rp_prefix}30m',
                    'target_rp': f'{self.rp_prefix}1h',
                    'source_measurement': 'trades_30m',
                    'target_measurement': 'trades_1h',
                    'interval': '1h'
                },
                {
                    'name': 'cq_4h_aggregation',
                    'source_rp': f'{self.rp_prefix}1h',
                    'target_rp': f'{self.rp_prefix}4h',
                    'source_measurement': 'trades_1h',
                    'target_measurement': 'trades_4h',
                    'interval': '4h'
                }
            ]
            
            success_count = 0
            
            for cq_config in continuous_queries:
                cq_name = cq_config['name']
                
                try:
                    if cq_name in existing_cqs:
                        self.logger.info(f"Continuous query {cq_name} already exists")
                    else:
                        # Create the continuous query
                        cq_sql = self._build_continuous_query_sql(cq_config)
                        
                        self.logger.info(f"Creating continuous query: {cq_name}")
                        self.logger.debug(f"CQ SQL: {cq_sql}")
                        
                        self.client.query(cq_sql)
                        self.logger.info(f"Successfully created continuous query: {cq_name}")
                    
                    success_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to create continuous query {cq_name}: {e}")
            
            self.logger.info(f"Successfully processed {success_count}/{len(continuous_queries)} continuous queries")
            return success_count == len(continuous_queries)
            
        except Exception as e:
            self.logger.error(f"Error creating continuous queries: {e}")
            return False
    
    def _build_continuous_query_sql(self, cq_config: Dict) -> str:
        """Build continuous query SQL statement"""
        
        # Build the aggregation fields (matching aggr-server structure)
        fields = [
            'first(open) AS open',
            'max(high) AS high', 
            'min(low) AS low',
            'last(close) AS close',
            'sum(vbuy) AS vbuy',
            'sum(vsell) AS vsell',
            'sum(cbuy) AS cbuy',
            'sum(csell) AS csell',
            'sum(lbuy) AS lbuy',
            'sum(lsell) AS lsell'
        ]
        
        # Build the SQL statement
        sql = f'''
        CREATE CONTINUOUS QUERY "{cq_config['name']}" ON "{self.database}"
        BEGIN
          SELECT {', '.join(fields)}
          INTO "{self.database}"."{cq_config['target_rp']}"."{cq_config['target_measurement']}"
          FROM "{self.database}"."{cq_config['source_rp']}"."{cq_config['source_measurement']}"
          GROUP BY time({cq_config['interval']}), market
        END
        '''
        
        # Clean up whitespace
        return ' '.join(sql.split())
    
    def verify_setup(self) -> Dict:
        """Verify the database setup is working correctly"""
        verification_result = {
            'database_exists': False,
            'retention_policies': {},
            'continuous_queries': [],
            'measurements': [],
            'sample_data_points': 0,
            'errors': []
        }
        
        try:
            # Check database exists
            databases = [db['name'] for db in self.client.get_list_database()]
            verification_result['database_exists'] = self.database in databases
            
            # Check retention policies
            try:
                rps = self.client.get_list_retention_policies(self.database)
                for rp in rps:
                    verification_result['retention_policies'][rp['name']] = {
                        'duration': rp['duration'],
                        'default': rp['default']
                    }
            except Exception as e:
                verification_result['errors'].append(f"Failed to get retention policies: {e}")
            
            # Check continuous queries
            try:
                cq_result = self.client.get_list_continuous_queries(self.database)
                for db_cqs in cq_result:
                    if db_cqs['name'] == self.database:
                        verification_result['continuous_queries'] = [
                            cq['name'] for cq in db_cqs.get('cqs', [])
                        ]
                        break
            except Exception as e:
                verification_result['errors'].append(f"Failed to get continuous queries: {e}")
            
            # Check measurements
            try:
                measurements_result = self.client.query(f'SHOW MEASUREMENTS ON "{self.database}"')
                verification_result['measurements'] = [
                    point['name'] for point in measurements_result.get_points()
                ]
            except Exception as e:
                verification_result['errors'].append(f"Failed to get measurements: {e}")
            
            # Check sample data
            try:
                sample_result = self.client.query(f'SELECT COUNT(*) FROM /.*/ LIMIT 1')
                points = list(sample_result.get_points())
                if points:
                    verification_result['sample_data_points'] = points[0].get('count', 0)
            except Exception as e:
                verification_result['errors'].append(f"Failed to get sample data: {e}")
            
        except Exception as e:
            verification_result['errors'].append(f"Verification failed: {e}")
        
        return verification_result
    
    def run_full_setup(self) -> bool:
        """Run complete database setup"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING INFLUXDB SETUP FOR SQUEEZEFLOW TRADER")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Connect to InfluxDB
            if not self.connect():
                self.logger.error("Failed to connect to InfluxDB")
                return False
            
            # Step 2: Ensure database exists
            if not self.ensure_database():
                self.logger.error("Failed to ensure database exists")
                return False
            
            # Step 3: Create retention policies
            if not self.create_retention_policies():
                self.logger.error("Failed to create retention policies")
                return False
            
            # Step 4: Create continuous queries
            if not self.create_continuous_queries():
                self.logger.error("Failed to create continuous queries")
                return False
            
            # Step 5: Verify setup
            self.logger.info("Verifying database setup...")
            verification = self.verify_setup()
            
            # Print verification results
            self.logger.info("=" * 60)
            self.logger.info("SETUP VERIFICATION RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"Database exists: {verification['database_exists']}")
            self.logger.info(f"Retention policies: {len(verification['retention_policies'])}")
            self.logger.info(f"Continuous queries: {len(verification['continuous_queries'])}")
            self.logger.info(f"Measurements: {len(verification['measurements'])}")
            
            if verification['errors']:
                self.logger.warning("Verification errors:")
                for error in verification['errors']:
                    self.logger.warning(f"  - {error}")
            
            # Print detailed results
            self.logger.info("\nRetention Policies:")
            for rp_name, rp_info in verification['retention_policies'].items():
                default_str = " (DEFAULT)" if rp_info['default'] else ""
                self.logger.info(f"  - {rp_name}: {rp_info['duration']}{default_str}")
            
            self.logger.info(f"\nContinuous Queries:")
            for cq_name in verification['continuous_queries']:
                self.logger.info(f"  - {cq_name}")
            
            if verification['measurements']:
                self.logger.info(f"\nMeasurements:")
                for measurement in verification['measurements']:
                    self.logger.info(f"  - {measurement}")
            
            self.logger.info("=" * 60)
            self.logger.info("INFLUXDB SETUP COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed with error: {e}")
            return False
        
        finally:
            if self.client:
                self.client.close()
    
    def create_sample_data(self) -> bool:
        """Create sample data for testing (optional)"""
        try:
            self.logger.info("Creating sample trading data...")
            
            # Create sample data points
            current_time = int(time.time() * 1000)  # milliseconds
            sample_points = []
            
            markets = ['BINANCE:btcusdt', 'COINBASE:BTC-USD', 'BYBIT:BTCUSDT']
            
            for i, market in enumerate(markets):
                # Create data for the last 10 minutes
                for minute_offset in range(10):
                    timestamp = current_time - (minute_offset * 60000)  # 1 minute intervals
                    
                    point = {
                        'measurement': 'trades_1m',
                        'tags': {
                            'market': market
                        },
                        'time': timestamp,
                        'fields': {
                            'open': 50000.0 + (i * 100),
                            'high': 50100.0 + (i * 100),
                            'low': 49900.0 + (i * 100),
                            'close': 50050.0 + (i * 100),
                            'vbuy': 1000000.0 + (i * 10000),  # Buy volume
                            'vsell': 900000.0 + (i * 10000),  # Sell volume
                            'cbuy': 100 + (i * 10),           # Buy count
                            'csell': 90 + (i * 10),           # Sell count
                            'lbuy': 0.0,                      # Liquidation buy
                            'lsell': 0.0                      # Liquidation sell
                        }
                    }
                    sample_points.append(point)
            
            # Write sample data to 1m retention policy
            self.client.write_points(
                sample_points,
                time_precision='ms',
                retention_policy=f'{self.rp_prefix}1m'
            )
            
            self.logger.info(f"Created {len(sample_points)} sample data points")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create sample data: {e}")
            return False


def main():
    """Main setup function"""
    # Configuration from docker-compose.yml
    setup = InfluxDBSetup(
        host='localhost',
        port=8086,  # Unified port after migration
        database='significant_trades'
    )
    
    # Run the complete setup
    success = setup.run_full_setup()
    
    if success:
        print("\n✅ InfluxDB setup completed successfully!")
        print("🔧 The database is now ready for SqueezeFlow Trader operations")
        print("📊 Retention policies and continuous queries are configured")
        print("⚡ Data aggregation pipeline is active")
        
        # Optionally create sample data for testing
        response = input("\n❓ Would you like to create sample data for testing? (y/N): ")
        if response.lower() in ['y', 'yes']:
            if setup.connect():
                setup.create_sample_data()
                print("✅ Sample data created!")
        
        return 0
    else:
        print("\n❌ InfluxDB setup failed!")
        print("🔍 Check the logs above for specific error details")
        return 1


if __name__ == "__main__":
    sys.exit(main())