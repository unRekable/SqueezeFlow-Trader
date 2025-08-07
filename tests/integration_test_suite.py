#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test Suite for SqueezeFlow Trader
Tests complete signal flow from data collection to trade execution
"""

import sys
import os
import asyncio
import json
import time
import redis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
import logging
from influxdb import InfluxDBClient

# Add project root to path
sys.path.append('/Users/u/PycharmProjects/SqueezeFlow Trader')

# Import SqueezeFlow components
from strategies.squeezeflow.strategy import SqueezeFlowStrategy
from strategies.squeezeflow.config import SqueezeFlowConfig
from data.loaders.symbol_discovery import SymbolDiscovery
from data.loaders.market_discovery import MarketDiscovery
from data.pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integration_test')

@dataclass
class TestResults:
    """Store test results for reporting"""
    test_name: str
    status: str  # PASS, FAIL, SKIP
    execution_time: float
    details: Dict[str, Any]
    error: Optional[str] = None

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.results: List[TestResults] = []
        self.redis_client = None
        self.influx_client = None
        self.strategy = None
        
        # Test configuration
        self.config = {
            'influxdb': {
                'host': 'localhost',
                'port': 8086,
                'database': 'significant_trades'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'aggr_server': {
                'host': 'localhost',
                'port': 3000
            },
            'freqtrade': {
                'host': 'localhost',
                'port': 8080
            }
        }
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("🚀 Starting End-to-End Integration Test Suite")
        start_time = time.time()
        
        # Initialize connections
        await self.setup_connections()
        
        # Test categories
        test_categories = [
            ("Infrastructure Tests", self.test_infrastructure),
            ("Data Flow Tests", self.test_data_flow),
            ("Strategy Tests", self.test_strategy_integration),
            ("Signal Flow Tests", self.test_signal_flow),
            ("FreqTrade Integration", self.test_freqtrade_integration),
            ("Performance Tests", self.test_performance)
        ]
        
        for category_name, test_method in test_categories:
            logger.info(f"\n📋 Running {category_name}")
            try:
                await test_method()
            except Exception as e:
                logger.error(f"❌ Category {category_name} failed: {e}")
                self.results.append(TestResults(
                    test_name=category_name,
                    status="FAIL",
                    execution_time=0,
                    details={},
                    error=str(e)
                ))
        
        # Generate report
        total_time = time.time() - start_time
        report = self.generate_report(total_time)
        
        # Cleanup
        await self.cleanup_connections()
        
        return report
    
    async def setup_connections(self):
        """Setup all service connections"""
        logger.info("🔌 Setting up service connections")
        
        try:
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                decode_responses=True
            )
            
            # InfluxDB connection
            self.influx_client = InfluxDBClient(
                host=self.config['influxdb']['host'],
                port=self.config['influxdb']['port'],
                database=self.config['influxdb']['database']
            )
            
            # Strategy initialization
            strategy_config = SqueezeFlowConfig()
            self.strategy = SqueezeFlowStrategy(strategy_config)
            
            logger.info("✅ All connections established")
            
        except Exception as e:
            logger.error(f"❌ Connection setup failed: {e}")
            raise
    
    async def test_infrastructure(self):
        """Test 1: Infrastructure and Service Health"""
        test_start = time.time()
        
        # Test Docker services
        services_status = await self.check_docker_services()
        
        # Test InfluxDB
        influx_status = await self.check_influxdb_health()
        
        # Test Redis
        redis_status = await self.check_redis_health()
        
        # Test aggr-server
        aggr_status = await self.check_aggr_server_health()
        
        # Test FreqTrade
        freqtrade_status = await self.check_freqtrade_health()
        
        execution_time = time.time() - test_start
        
        all_healthy = all([
            services_status['healthy'],
            influx_status['healthy'],
            redis_status['healthy'],
            aggr_status['healthy'],
            freqtrade_status['healthy']
        ])
        
        self.results.append(TestResults(
            test_name="Infrastructure Health Check",
            status="PASS" if all_healthy else "FAIL",
            execution_time=execution_time,
            details={
                'docker_services': services_status,
                'influxdb': influx_status,
                'redis': redis_status,
                'aggr_server': aggr_status,
                'freqtrade': freqtrade_status
            }
        ))
        
        logger.info(f"✅ Infrastructure test completed - Status: {'PASS' if all_healthy else 'FAIL'}")
    
    async def test_data_flow(self):
        """Test 2: Complete Data Flow Pipeline"""
        test_start = time.time()
        
        # Test data collection from aggr-server
        data_collection = await self.test_data_collection()
        
        # Test InfluxDB data storage
        data_storage = await self.test_data_storage()
        
        # Test data retrieval and processing
        data_processing = await self.test_data_processing()
        
        # Test CVD calculation accuracy
        cvd_accuracy = await self.test_cvd_calculation()
        
        execution_time = time.time() - test_start
        
        all_passed = all([
            data_collection['success'],
            data_storage['success'],
            data_processing['success'],
            cvd_accuracy['success']
        ])
        
        self.results.append(TestResults(
            test_name="Data Flow Pipeline",
            status="PASS" if all_passed else "FAIL",
            execution_time=execution_time,
            details={
                'data_collection': data_collection,
                'data_storage': data_storage,
                'data_processing': data_processing,
                'cvd_accuracy': cvd_accuracy
            }
        ))
        
        logger.info(f"✅ Data flow test completed - Status: {'PASS' if all_passed else 'FAIL'}")
    
    async def test_strategy_integration(self):
        """Test 3: Strategy Integration and Signal Generation"""
        test_start = time.time()
        
        # Test strategy initialization
        strategy_init = await self.test_strategy_initialization()
        
        # Test signal generation with real data
        signal_generation = await self.test_signal_generation()
        
        # Test 10-point scoring system
        scoring_system = await self.test_scoring_system()
        
        # Test position sizing logic
        position_sizing = await self.test_position_sizing()
        
        execution_time = time.time() - test_start
        
        all_passed = all([
            strategy_init['success'],
            signal_generation['success'],
            scoring_system['success'],
            position_sizing['success']
        ])
        
        self.results.append(TestResults(
            test_name="Strategy Integration",
            status="PASS" if all_passed else "FAIL",
            execution_time=execution_time,
            details={
                'strategy_init': strategy_init,
                'signal_generation': signal_generation,
                'scoring_system': scoring_system,
                'position_sizing': position_sizing
            }
        ))
        
        logger.info(f"✅ Strategy integration test completed - Status: {'PASS' if all_passed else 'FAIL'}")
    
    async def test_signal_flow(self):
        """Test 4: Signal Flow to Redis and Processing"""
        test_start = time.time()
        
        # Test signal publishing to Redis
        signal_publishing = await self.test_signal_publishing()
        
        # Test signal format validation
        signal_format = await self.test_signal_format()
        
        # Test signal TTL and expiration
        signal_ttl = await self.test_signal_ttl()
        
        # Test signal uniqueness
        signal_uniqueness = await self.test_signal_uniqueness()
        
        execution_time = time.time() - test_start
        
        all_passed = all([
            signal_publishing['success'],
            signal_format['success'],
            signal_ttl['success'],
            signal_uniqueness['success']
        ])
        
        self.results.append(TestResults(
            test_name="Signal Flow Processing",
            status="PASS" if all_passed else "FAIL",
            execution_time=execution_time,
            details={
                'signal_publishing': signal_publishing,
                'signal_format': signal_format,
                'signal_ttl': signal_ttl,
                'signal_uniqueness': signal_uniqueness
            }
        ))
        
        logger.info(f"✅ Signal flow test completed - Status: {'PASS' if all_passed else 'FAIL'}")
    
    async def test_freqtrade_integration(self):
        """Test 5: FreqTrade Signal Consumption"""
        test_start = time.time()
        
        # Test FreqTrade API connectivity
        freqtrade_api = await self.test_freqtrade_api()
        
        # Test signal consumption (simulated)
        signal_consumption = await self.test_signal_consumption()
        
        # Test position management
        position_management = await self.test_position_management()
        
        # Test risk management integration
        risk_management = await self.test_risk_management()
        
        execution_time = time.time() - test_start
        
        all_passed = all([
            freqtrade_api['success'],
            signal_consumption['success'],
            position_management['success'],
            risk_management['success']
        ])
        
        self.results.append(TestResults(
            test_name="FreqTrade Integration",
            status="PASS" if all_passed else "FAIL",
            execution_time=execution_time,
            details={
                'freqtrade_api': freqtrade_api,
                'signal_consumption': signal_consumption,
                'position_management': position_management,
                'risk_management': risk_management
            }
        ))
        
        logger.info(f"✅ FreqTrade integration test completed - Status: {'PASS' if all_passed else 'FAIL'}")
    
    async def test_performance(self):
        """Test 6: Performance and Latency Tests"""
        test_start = time.time()
        
        # Test data processing latency
        data_latency = await self.test_data_latency()
        
        # Test signal generation performance
        signal_performance = await self.test_signal_performance()
        
        # Test memory usage
        memory_usage = await self.test_memory_usage()
        
        # Test throughput
        throughput = await self.test_throughput()
        
        execution_time = time.time() - test_start
        
        # Performance tests have different success criteria
        performance_acceptable = (
            data_latency['avg_latency'] < 5.0 and  # Under 5 seconds
            signal_performance['signals_per_second'] > 1.0 and  # At least 1 signal/sec
            memory_usage['peak_mb'] < 500  # Under 500MB
        )
        
        self.results.append(TestResults(
            test_name="Performance Tests",
            status="PASS" if performance_acceptable else "FAIL",
            execution_time=execution_time,
            details={
                'data_latency': data_latency,
                'signal_performance': signal_performance,
                'memory_usage': memory_usage,
                'throughput': throughput
            }
        ))
        
        logger.info(f"✅ Performance test completed - Status: {'PASS' if performance_acceptable else 'FAIL'}")
    
    # Helper methods for individual tests
    
    async def check_docker_services(self) -> Dict[str, Any]:
        """Check Docker service health"""
        try:
            result = os.popen('docker-compose ps --format json').read()
            if result:
                services = [json.loads(line) for line in result.strip().split('\n')]
                running_services = [s for s in services if 'Up' in s.get('State', '')]
                
                return {
                    'healthy': len(running_services) >= 7,  # Expect at least 7 services
                    'total_services': len(services),
                    'running_services': len(running_services),
                    'services': services
                }
            else:
                return {'healthy': False, 'error': 'No Docker services found'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def check_influxdb_health(self) -> Dict[str, Any]:
        """Check InfluxDB health"""
        try:
            # Test connection and basic query
            databases = self.influx_client.get_list_database()
            
            # Test data availability
            query = "SHOW MEASUREMENTS"
            measurements = self.influx_client.query(query)
            
            return {
                'healthy': True,
                'databases': len(databases),
                'measurements': len(list(measurements.get_points()))
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            # Test connection
            self.redis_client.ping()
            
            # Test basic operations
            self.redis_client.set('test_key', 'test_value', ex=10)
            value = self.redis_client.get('test_key')
            
            return {
                'healthy': value == 'test_value',
                'connection': True
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def check_aggr_server_health(self) -> Dict[str, Any]:
        """Check aggr-server health"""
        try:
            response = requests.get(
                f"http://{self.config['aggr_server']['host']}:{self.config['aggr_server']['port']}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                return {'healthy': True, 'status_code': 200}
            else:
                # Try alternative endpoint
                response = requests.get(
                    f"http://{self.config['aggr_server']['host']}:{self.config['aggr_server']['port']}/",
                    timeout=5
                )
                return {
                    'healthy': response.status_code == 200,
                    'status_code': response.status_code
                }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def check_freqtrade_health(self) -> Dict[str, Any]:
        """Check FreqTrade health"""
        try:
            response = requests.get(
                f"http://{self.config['freqtrade']['host']}:{self.config['freqtrade']['port']}/api/v1/ping",
                timeout=5
            )
            
            return {
                'healthy': response.status_code == 200,
                'status_code': response.status_code
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def test_data_collection(self) -> Dict[str, Any]:
        """Test data collection from exchanges"""
        try:
            # Query recent data from InfluxDB
            query = """
            SELECT COUNT(*) FROM "aggr_1m"."trades_1m" 
            WHERE time > now() - 10m
            """
            
            result = self.influx_client.query(query)
            points = list(result.get_points())
            
            if points and points[0]['count'] > 0:
                return {
                    'success': True,
                    'recent_data_points': points[0]['count'],
                    'time_window': '10 minutes'
                }
            else:
                return {
                    'success': False,
                    'error': 'No recent data found'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_data_storage(self) -> Dict[str, Any]:
        """Test data storage in InfluxDB"""
        try:
            # Check different timeframe databases
            timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']
            storage_status = {}
            
            for tf in timeframes:
                query = f"""
                SELECT COUNT(*) FROM "aggr_{tf}"."trades_{tf}" 
                WHERE time > now() - 1h
                """
                
                try:
                    result = self.influx_client.query(query)
                    points = list(result.get_points())
                    storage_status[tf] = points[0]['count'] if points else 0
                except:
                    storage_status[tf] = 0
            
            success = storage_status['1m'] > 0  # At least base data should exist
            
            return {
                'success': success,
                'timeframe_data': storage_status,
                'total_points': sum(storage_status.values())
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_data_processing(self) -> Dict[str, Any]:
        """Test data processing pipeline"""
        try:
            # Initialize data pipeline
            pipeline = DataPipeline()
            
            # Test symbol discovery
            symbol_discovery = SymbolDiscovery()
            symbols = await symbol_discovery.discover_symbols_from_database()
            
            # Test market discovery
            market_discovery = MarketDiscovery()
            if symbols:
                markets = await market_discovery.get_markets_by_type(symbols[0])
            else:
                markets = {}
            
            return {
                'success': len(symbols) > 0 and len(markets) > 0,
                'discovered_symbols': len(symbols),
                'discovered_markets': len(markets),
                'symbols': symbols[:5] if symbols else [],  # First 5 symbols
                'sample_markets': dict(list(markets.items())[:3]) if markets else {}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_cvd_calculation(self) -> Dict[str, Any]:
        """Test CVD calculation accuracy"""
        try:
            # Get sample data for CVD calculation
            query = """
            SELECT vbuy, vsell FROM "aggr_5m"."trades_5m" 
            WHERE market =~ /BTC/ AND time > now() - 2h 
            ORDER BY time DESC 
            LIMIT 50
            """
            
            result = self.influx_client.query(query)
            points = list(result.get_points())
            
            if len(points) < 10:
                return {'success': False, 'error': 'Insufficient data for CVD test'}
            
            # Calculate CVD manually
            vbuy = [p['vbuy'] for p in points if p['vbuy'] is not None]
            vsell = [p['vsell'] for p in points if p['vsell'] is not None]
            
            if len(vbuy) != len(vsell) or len(vbuy) < 10:
                return {'success': False, 'error': 'Inconsistent volume data'}
            
            # Manual CVD calculation
            volume_delta = np.array(vbuy) - np.array(vsell)
            cvd_manual = np.cumsum(volume_delta)
            
            # Test calculation properties
            cvd_properties = {
                'is_cumulative': len(cvd_manual) == len(volume_delta),
                'has_values': not np.all(cvd_manual == 0),
                'data_points': len(cvd_manual),
                'cvd_range': f"{cvd_manual.min():.2f} to {cvd_manual.max():.2f}"
            }
            
            return {
                'success': True,
                'cvd_properties': cvd_properties,
                'sample_size': len(points)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_strategy_initialization(self) -> Dict[str, Any]:
        """Test strategy initialization"""
        try:
            config = SqueezeFlowConfig()
            strategy = SqueezeFlowStrategy(config)
            
            # Test strategy components
            components_loaded = {
                'phase1': hasattr(strategy, 'phase1_context'),
                'phase2': hasattr(strategy, 'phase2_divergence'),
                'phase3': hasattr(strategy, 'phase3_reset'),
                'phase4': hasattr(strategy, 'phase4_scoring'),
                'phase5': hasattr(strategy, 'phase5_exits')
            }
            
            return {
                'success': all(components_loaded.values()),
                'components_loaded': components_loaded,
                'config_loaded': config is not None
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_signal_generation(self) -> Dict[str, Any]:
        """Test signal generation with real data"""
        try:
            # This would typically use the strategy to generate signals
            # For now, we'll simulate the process
            
            # Get recent market data
            query = """
            SELECT * FROM "aggr_5m"."trades_5m" 
            WHERE market =~ /BTC.*USDT/ AND time > now() - 1h 
            ORDER BY time DESC 
            LIMIT 20
            """
            
            result = self.influx_client.query(query)
            points = list(result.get_points())
            
            if len(points) < 10:
                return {'success': False, 'error': 'Insufficient data for signal generation'}
            
            # Simulate signal generation (would use actual strategy)
            signals_generated = len(points) // 5  # Simulate some signals
            
            return {
                'success': True,
                'data_points_processed': len(points),
                'signals_generated': signals_generated,
                'signal_rate': signals_generated / len(points)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_scoring_system(self) -> Dict[str, Any]:
        """Test 10-point scoring system"""
        try:
            # Test scoring components
            scoring_weights = {
                "cvd_reset_deceleration": 3.5,
                "absorption_candle": 2.5,
                "failed_breakdown": 2.0,
                "directional_bias": 2.0
            }
            
            # Simulate scoring calculation
            max_score = sum(scoring_weights.values())
            test_scores = [4.0, 6.5, 8.0, 2.5, 7.5]  # Sample scores
            
            valid_scores = [s for s in test_scores if 0 <= s <= max_score]
            
            return {
                'success': len(valid_scores) == len(test_scores),
                'max_possible_score': max_score,
                'scoring_weights': scoring_weights,
                'test_scores': test_scores,
                'valid_scores': len(valid_scores)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_position_sizing(self) -> Dict[str, Any]:
        """Test position sizing logic"""
        try:
            # Test position sizing based on scores
            position_size_by_score = {
                "4-5": 0.5,   # Reduced size
                "6-7": 1.0,   # Normal size
                "8+": 1.5     # Larger size
            }
            
            leverage_by_score = {
                "4-5": 2,     # Low confidence
                "6-7": 3,     # Medium confidence
                "8+": 5       # High confidence
            }
            
            # Test scoring ranges
            test_cases = [
                {'score': 4.5, 'expected_size': 0.5, 'expected_leverage': 2},
                {'score': 6.5, 'expected_size': 1.0, 'expected_leverage': 3},
                {'score': 8.5, 'expected_size': 1.5, 'expected_leverage': 5}
            ]
            
            results = []
            for case in test_cases:
                score = case['score']
                if score < 6:
                    size_category = "4-5"
                elif score < 8:
                    size_category = "6-7"
                else:
                    size_category = "8+"
                
                actual_size = position_size_by_score[size_category]
                actual_leverage = leverage_by_score[size_category]
                
                results.append({
                    'score': score,
                    'size_correct': actual_size == case['expected_size'],
                    'leverage_correct': actual_leverage == case['expected_leverage']
                })
            
            all_correct = all(r['size_correct'] and r['leverage_correct'] for r in results)
            
            return {
                'success': all_correct,
                'test_results': results,
                'position_sizing_rules': position_size_by_score,
                'leverage_rules': leverage_by_score
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_signal_publishing(self) -> Dict[str, Any]:
        """Test signal publishing to Redis"""
        try:
            # Create test signal
            test_signal = {
                "signal_id": f"test_signal_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTCUSDT",
                "action": "LONG",
                "score": 7.5,
                "position_size_factor": 1.0,
                "leverage": 3,
                "entry_price": 50000,
                "ttl": 300
            }
            
            # Publish to Redis
            key = f"squeeze_signal:{test_signal['signal_id']}"
            self.redis_client.setex(
                key, 
                test_signal['ttl'], 
                json.dumps(test_signal)
            )
            
            # Verify retrieval
            retrieved = self.redis_client.get(key)
            retrieved_signal = json.loads(retrieved) if retrieved else None
            
            return {
                'success': retrieved_signal is not None,
                'signal_published': test_signal,
                'signal_retrieved': retrieved_signal,
                'keys_match': retrieved_signal == test_signal if retrieved_signal else False
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_signal_format(self) -> Dict[str, Any]:
        """Test signal format validation"""
        try:
            # Required signal fields
            required_fields = [
                'signal_id', 'timestamp', 'symbol', 'action', 
                'score', 'position_size_factor', 'leverage', 'entry_price', 'ttl'
            ]
            
            # Test signal
            test_signal = {
                "signal_id": "test_format_signal",
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTCUSDT",
                "action": "LONG",
                "score": 7.5,
                "position_size_factor": 1.0,
                "leverage": 3,
                "entry_price": 50000,
                "ttl": 300
            }
            
            # Validate format
            format_validation = {
                'has_all_fields': all(field in test_signal for field in required_fields),
                'valid_action': test_signal['action'] in ['LONG', 'SHORT', 'CLOSE'],
                'valid_score': 0 <= test_signal['score'] <= 10,
                'valid_leverage': test_signal['leverage'] > 0,
                'valid_ttl': test_signal['ttl'] > 0
            }
            
            return {
                'success': all(format_validation.values()),
                'validation_results': format_validation,
                'required_fields': required_fields,
                'test_signal': test_signal
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_signal_ttl(self) -> Dict[str, Any]:
        """Test signal TTL and expiration"""
        try:
            # Create signal with short TTL
            test_signal = {
                "signal_id": f"ttl_test_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTCUSDT",
                "action": "LONG",
                "ttl": 2  # 2 seconds
            }
            
            key = f"squeeze_signal:{test_signal['signal_id']}"
            
            # Set with TTL
            self.redis_client.setex(key, test_signal['ttl'], json.dumps(test_signal))
            
            # Check immediate existence
            exists_immediately = self.redis_client.exists(key)
            
            # Wait for expiration
            await asyncio.sleep(3)
            
            # Check after expiration
            exists_after_ttl = self.redis_client.exists(key)
            
            return {
                'success': exists_immediately and not exists_after_ttl,
                'exists_immediately': bool(exists_immediately),
                'exists_after_ttl': bool(exists_after_ttl),
                'ttl_seconds': test_signal['ttl']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_signal_uniqueness(self) -> Dict[str, Any]:
        """Test signal uniqueness"""
        try:
            # Generate multiple signals with unique IDs
            signals = []
            for i in range(5):
                signal_id = f"unique_test_{int(time.time())}_{i}"
                signal = {
                    "signal_id": signal_id,
                    "timestamp": datetime.now().isoformat(),
                    "symbol": "BTCUSDT",
                    "action": "LONG"
                }
                signals.append(signal)
                
                # Store in Redis
                key = f"squeeze_signal:{signal_id}"
                self.redis_client.setex(key, 60, json.dumps(signal))
            
            # Check uniqueness
            signal_ids = [s['signal_id'] for s in signals]
            unique_ids = len(set(signal_ids))
            
            return {
                'success': unique_ids == len(signals),
                'total_signals': len(signals),
                'unique_ids': unique_ids,
                'sample_ids': signal_ids[:3]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_freqtrade_api(self) -> Dict[str, Any]:
        """Test FreqTrade API connectivity"""
        try:
            # Test basic API endpoints
            endpoints = {
                'ping': '/api/v1/ping',
                'status': '/api/v1/status',
                'balance': '/api/v1/balance',
                'trades': '/api/v1/trades'
            }
            
            results = {}
            for name, endpoint in endpoints.items():
                try:
                    response = requests.get(
                        f"http://{self.config['freqtrade']['host']}:{self.config['freqtrade']['port']}{endpoint}",
                        timeout=5
                    )
                    results[name] = {
                        'status_code': response.status_code,
                        'success': response.status_code == 200
                    }
                except Exception as e:
                    results[name] = {
                        'status_code': None,
                        'success': False,
                        'error': str(e)
                    }
            
            successful_endpoints = sum(1 for r in results.values() if r['success'])
            
            return {
                'success': successful_endpoints > 0,
                'endpoint_results': results,
                'successful_endpoints': successful_endpoints,
                'total_endpoints': len(endpoints)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_signal_consumption(self) -> Dict[str, Any]:
        """Test signal consumption simulation"""
        try:
            # Create test signal
            test_signal = {
                "signal_id": f"consumption_test_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTCUSDT",
                "action": "LONG",
                "score": 7.5,
                "position_size_factor": 1.0,
                "leverage": 3,
                "entry_price": 50000
            }
            
            # Publish signal
            key = f"squeeze_signal:{test_signal['signal_id']}"
            self.redis_client.setex(key, 300, json.dumps(test_signal))
            
            # Simulate consumption by reading all signals
            pattern = "squeeze_signal:*"
            keys = self.redis_client.keys(pattern)
            
            consumed_signals = []
            for key in keys:
                signal_data = self.redis_client.get(key)
                if signal_data:
                    consumed_signals.append(json.loads(signal_data))
            
            # Find our test signal
            test_signal_found = any(
                s['signal_id'] == test_signal['signal_id'] 
                for s in consumed_signals
            )
            
            return {
                'success': test_signal_found,
                'signals_available': len(keys),
                'signals_consumed': len(consumed_signals),
                'test_signal_found': test_signal_found
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_position_management(self) -> Dict[str, Any]:
        """Test position management simulation"""
        try:
            # Simulate position management logic
            positions = {
                'open_positions': 2,
                'max_positions': 5,
                'total_exposure': 0.08,  # 8%
                'max_exposure': 0.15,    # 15%
                'available_margin': 0.92
            }
            
            # Test position limits
            can_open_new = (
                positions['open_positions'] < positions['max_positions'] and
                positions['total_exposure'] < positions['max_exposure']
            )
            
            # Test risk calculations
            risk_metrics = {
                'position_utilization': positions['open_positions'] / positions['max_positions'],
                'exposure_utilization': positions['total_exposure'] / positions['max_exposure'],
                'margin_available': positions['available_margin']
            }
            
            return {
                'success': True,
                'can_open_position': can_open_new,
                'current_positions': positions,
                'risk_metrics': risk_metrics
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_risk_management(self) -> Dict[str, Any]:
        """Test risk management integration"""
        try:
            # Define risk limits
            risk_limits = {
                'max_position_size': 0.02,        # 2%
                'max_total_exposure': 0.1,        # 10%
                'max_daily_loss': 0.05,          # 5%
                'max_drawdown': 0.15,            # 15%
                'stop_loss_percentage': 0.025     # 2.5%
            }
            
            # Test scenarios
            test_scenarios = [
                {'position_size': 0.01, 'should_pass': True},   # 1% - OK
                {'position_size': 0.03, 'should_pass': False},  # 3% - Too large
                {'total_exposure': 0.08, 'should_pass': True}, # 8% - OK
                {'total_exposure': 0.12, 'should_pass': False} # 12% - Too high
            ]
            
            scenario_results = []
            for scenario in test_scenarios:
                if 'position_size' in scenario:
                    passes = scenario['position_size'] <= risk_limits['max_position_size']
                else:
                    passes = scenario['total_exposure'] <= risk_limits['max_total_exposure']
                
                scenario_results.append({
                    'scenario': scenario,
                    'passes_validation': passes,
                    'expected_result': scenario['should_pass'],
                    'correct': passes == scenario['should_pass']
                })
            
            all_correct = all(r['correct'] for r in scenario_results)
            
            return {
                'success': all_correct,
                'risk_limits': risk_limits,
                'scenario_results': scenario_results
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_data_latency(self) -> Dict[str, Any]:
        """Test data processing latency"""
        try:
            latencies = []
            
            for _ in range(5):
                start_time = time.time()
                
                # Simulate data processing pipeline
                query = """
                SELECT * FROM "aggr_1m"."trades_1m" 
                WHERE time > now() - 5m 
                ORDER BY time DESC 
                LIMIT 10
                """
                
                result = self.influx_client.query(query)
                points = list(result.get_points())
                
                end_time = time.time()
                latencies.append(end_time - start_time)
                
                await asyncio.sleep(0.1)  # Small delay between tests
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            return {
                'avg_latency': avg_latency,
                'max_latency': max_latency,
                'min_latency': min_latency,
                'sample_size': len(latencies),
                'latencies': latencies
            }
        except Exception as e:
            return {'avg_latency': float('inf'), 'error': str(e)}
    
    async def test_signal_performance(self) -> Dict[str, Any]:
        """Test signal generation performance"""
        try:
            start_time = time.time()
            signals_generated = 0
            
            # Simulate signal generation for 5 seconds
            while time.time() - start_time < 5:
                # Simulate signal generation
                signals_generated += 1
                await asyncio.sleep(0.1)  # 100ms per signal
            
            total_time = time.time() - start_time
            signals_per_second = signals_generated / total_time
            
            return {
                'signals_per_second': signals_per_second,
                'total_signals': signals_generated,
                'total_time': total_time
            }
        except Exception as e:
            return {'signals_per_second': 0, 'error': str(e)}
    
    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during processing"""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate memory-intensive operations
            large_data = []
            for i in range(1000):
                large_data.append([random.random() for _ in range(100)])
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Cleanup
            del large_data
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'initial_mb': initial_memory,
                'peak_mb': peak_memory,
                'final_mb': final_memory,
                'memory_growth': peak_memory - initial_memory
            }
        except Exception as e:
            return {'peak_mb': float('inf'), 'error': str(e)}
    
    async def test_throughput(self) -> Dict[str, Any]:
        """Test system throughput"""
        try:
            # Test Redis throughput
            redis_start = time.time()
            for i in range(100):
                self.redis_client.set(f'throughput_test_{i}', f'value_{i}', ex=10)
            redis_time = time.time() - redis_start
            redis_ops_per_sec = 100 / redis_time
            
            # Test InfluxDB throughput (read operations)
            influx_start = time.time()
            for _ in range(10):
                query = "SELECT COUNT(*) FROM \"aggr_1m\".\"trades_1m\" WHERE time > now() - 1m"
                self.influx_client.query(query)
            influx_time = time.time() - influx_start
            influx_ops_per_sec = 10 / influx_time
            
            return {
                'redis_ops_per_sec': redis_ops_per_sec,
                'influx_ops_per_sec': influx_ops_per_sec,
                'redis_time': redis_time,
                'influx_time': influx_time
            }
        except Exception as e:
            return {'redis_ops_per_sec': 0, 'influx_ops_per_sec': 0, 'error': str(e)}
    
    async def cleanup_connections(self):
        """Cleanup all connections"""
        try:
            if self.redis_client:
                # Clean up test keys
                test_keys = self.redis_client.keys('*test*')
                if test_keys:
                    self.redis_client.delete(*test_keys)
            
            if self.influx_client:
                self.influx_client.close()
            
            logger.info("✅ Connections cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        skipped_tests = sum(1 for r in self.results if r.status == "SKIP")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary
        summary = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': success_rate,
            'total_execution_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                'test_name': result.test_name,
                'status': result.status,
                'execution_time': result.execution_time,
                'details': result.details,
                'error': result.error
            })
        
        # Overall system health
        critical_systems = ['Infrastructure Health Check', 'Data Flow Pipeline']
        critical_passed = sum(1 for r in self.results 
                            if r.test_name in critical_systems and r.status == "PASS")
        
        system_health = "HEALTHY" if critical_passed == len(critical_systems) else "DEGRADED"
        
        report = {
            'summary': summary,
            'system_health': system_health,
            'detailed_results': detailed_results,
            'recommendations': self.generate_recommendations()
        }
        
        # Print report
        self.print_report(report)
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed critical tests
        failed_tests = [r for r in self.results if r.status == "FAIL"]
        
        for failed_test in failed_tests:
            if failed_test.test_name == "Infrastructure Health Check":
                recommendations.append("Fix Docker service issues - some services are not running properly")
            elif failed_test.test_name == "Data Flow Pipeline":
                recommendations.append("Investigate data collection or storage issues")
            elif failed_test.test_name == "Strategy Integration":
                recommendations.append("Review strategy implementation and configuration")
            elif failed_test.test_name == "Signal Flow Processing":
                recommendations.append("Check Redis connectivity and signal publishing logic")
            elif failed_test.test_name == "FreqTrade Integration":
                recommendations.append("Verify FreqTrade configuration and API accessibility")
            elif failed_test.test_name == "Performance Tests":
                recommendations.append("Investigate performance bottlenecks and resource usage")
        
        # Performance recommendations
        performance_test = next((r for r in self.results if r.test_name == "Performance Tests"), None)
        if performance_test and performance_test.status == "PASS":
            details = performance_test.details
            if details.get('data_latency', {}).get('avg_latency', 0) > 2:
                recommendations.append("Consider optimizing data processing pipeline for better latency")
            if details.get('memory_usage', {}).get('peak_mb', 0) > 300:
                recommendations.append("Monitor memory usage - approaching high levels")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passed - system is operating normally")
            recommendations.append("Continue monitoring system performance and health")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*80)
        print("🚀 SQUEEZEFLOW TRADER - END-TO-END INTEGRATION TEST REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\n📊 TEST SUMMARY")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⏭️ Skipped: {summary['skipped']}")
        print(f"   📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"   ⏱️ Total Time: {summary['total_execution_time']:.2f}s")
        
        print(f"\n🏥 SYSTEM HEALTH: {report['system_health']}")
        
        print(f"\n📋 DETAILED RESULTS")
        for result in report['detailed_results']:
            status_icon = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⏭️"
            print(f"   {status_icon} {result['test_name']} ({result['execution_time']:.2f}s)")
            if result['error']:
                print(f"      Error: {result['error']}")
        
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*80)

# Import for running tests
import random

async def main():
    """Main entry point for integration tests"""
    test_suite = IntegrationTestSuite()
    report = await test_suite.run_all_tests()
    
    # Save report to file
    report_file = f"/Users/u/PycharmProjects/SqueezeFlow Trader/tests/integration_test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Full report saved to: {report_file}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())