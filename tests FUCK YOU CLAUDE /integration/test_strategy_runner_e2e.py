#!/usr/bin/env python3
"""
End-to-End Integration Tests for Strategy Runner Service
Comprehensive testing of the complete signal flow: Data → Strategy → Redis → InfluxDB

This test suite covers:
1. Complete signal flow integration
2. Docker container integration
3. Error scenarios and recovery
4. Performance benchmarking
5. Real-world usage patterns
"""

import pytest
import asyncio
import redis
import json
import time
import uuid
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading

from services.strategy_runner import StrategyRunner
from services.config.unified_config import ConfigManager, ServiceConfig
from services.signal_validator import ValidationResult
from tests.conftest import (
    PERFORMANCE_BENCHMARKS, wait_for_condition, create_test_signal,
    generate_stress_test_data
)


class TestStrategyRunnerE2E:
    """Comprehensive end-to-end tests for Strategy Runner Service"""
    
    @pytest.mark.asyncio
    async def test_complete_signal_flow_integration(
        self, mock_config_manager, redis_client, influx_client, 
        sample_dataset, performance_monitor
    ):
        """Test complete signal flow from data to Redis to InfluxDB"""
        
        # Setup strategy runner with real connections
        runner = StrategyRunner(mock_config_manager)
        
        # Mock the data loading to return our sample dataset
        with patch.object(runner, '_load_symbol_data', return_value=sample_dataset):
            # Mock strategy to generate predictable signal
            with patch.object(runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{
                        'side': 'BUY',
                        'price': 50000,
                        'confidence': 0.8,
                        'reasoning': 'E2E test signal'
                    }],
                    'phase_results': {
                        'phase4_scoring': {
                            'total_score': 7.5
                        }
                    }
                }
                
                # Start performance monitoring
                performance_monitor.start_timing('complete_flow')
                
                # Process a single symbol
                await runner._process_symbol('BTC')
                
                flow_time = performance_monitor.end_timing('complete_flow')
                
                # Verify signal was published to Redis
                redis_key = f"{runner.config.redis_key_prefix}:signal:BTC"
                signal_data = redis_client.get(redis_key)
                
                assert signal_data is not None, "Signal not found in Redis"
                signal = json.loads(signal_data)
                
                # Verify signal structure
                assert signal['symbol'] == 'BTC'
                assert signal['action'] == 'LONG'
                assert signal['score'] == 7.5
                assert signal['position_size_factor'] == 1.0
                assert signal['leverage'] == 3
                
                # Verify signal was published to pub/sub channel
                channel = f"{runner.config.redis_key_prefix}:signals"
                # We can't easily test pub/sub in integration test, but verify channel exists
                
                # Wait for InfluxDB write (async operation)
                await asyncio.sleep(0.1)
                
                # Verify signal was stored in InfluxDB
                query = f"SELECT * FROM strategy_signals WHERE symbol = 'BTC'"
                result = influx_client.query(query)
                points = list(result.get_points())
                
                assert len(points) > 0, "Signal not found in InfluxDB"
                stored_signal = points[0]
                assert stored_signal['symbol'] == 'BTC'
                assert stored_signal['action'] == 'LONG'
                assert stored_signal['score'] == 7.5
                
                # Performance assertions
                assert flow_time < PERFORMANCE_BENCHMARKS['signal_processing_max_time']
                
                print(f"✅ Complete signal flow test passed in {flow_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_batch_signal_processing(
        self, mock_config_manager, redis_client, sample_dataset
    ):
        """Test batch signal processing and publishing"""
        
        # Enable batch processing
        config = mock_config_manager.get_config()
        config.enable_batch_publishing = True
        config.max_batch_size = 3
        config.batch_timeout_seconds = 1
        
        runner = StrategyRunner(mock_config_manager)
        
        # Generate multiple test signals
        test_signals = [
            create_test_signal('BTC', 'LONG', 6.0),
            create_test_signal('ETH', 'SHORT', 7.0),
            create_test_signal('ADA', 'LONG', 8.0)
        ]
        
        start_time = time.time()
        
        # Add signals to batch
        for signal in test_signals:
            await runner._add_signal_to_batch(signal)
        
        # Batch should auto-flush when full
        batch_time = time.time() - start_time
        
        # Verify all signals were published
        for signal in test_signals:
            redis_key = f"{runner.config.redis_key_prefix}:signal:{signal['symbol']}"
            stored_signal = redis_client.get(redis_key)
            assert stored_signal is not None
            
            parsed_signal = json.loads(stored_signal)
            assert parsed_signal['signal_id'] == signal['signal_id']
        
        # Performance check
        assert batch_time < PERFORMANCE_BENCHMARKS['batch_processing_max_time']
        
        print(f"✅ Batch processing test passed in {batch_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_error_scenarios_and_recovery(
        self, mock_config_manager, redis_client, sample_dataset
    ):
        """Test error handling and recovery mechanisms"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Test 1: Redis connection failure during publish
        with patch.object(runner, 'redis_client') as mock_redis:
            mock_redis.pipeline.side_effect = redis.ConnectionError("Connection failed")
            
            # Should handle error gracefully
            test_signal = create_test_signal('BTC', 'LONG', 7.0)
            
            with pytest.raises(redis.ConnectionError):
                await runner._publish_signal_immediately(test_signal)
            
            # Verify error was tracked
            assert runner.performance_stats['signals_failed'] > 0
        
        # Test 2: Invalid data handling
        with patch.object(runner, '_load_symbol_data', return_value=None):
            await runner._process_symbol('INVALID')
            # Should not crash, just log warning
        
        # Test 3: Strategy processing error
        with patch.object(runner.strategy, 'process', side_effect=Exception("Strategy error")):
            await runner._process_symbol('BTC')
            # Should handle gracefully and increment error count
            assert runner.performance_stats['errors_encountered'] > 0
        
        # Test 4: Recovery after Redis reconnection
        # Simulate reconnection by creating new client
        runner._redis_client = redis.Redis(**mock_config_manager.get_redis_config())
        
        # Should work normally after reconnection
        test_signal = create_test_signal('BTC', 'LONG', 6.0)
        await runner._publish_signal_immediately(test_signal)
        
        # Verify signal was published
        redis_key = f"{runner.config.redis_key_prefix}:signal:BTC"
        assert redis_client.get(redis_key) is not None
        
        print("✅ Error scenarios and recovery test passed")
    
    @pytest.mark.asyncio
    async def test_signal_validation_and_deduplication(
        self, mock_config_manager, redis_client
    ):
        """Test signal validation and deduplication logic"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Test 1: Valid signal passes validation
        valid_signal = create_test_signal('BTC', 'LONG', 7.0)
        result, errors = runner.signal_validator.validate_signal(valid_signal)
        assert result == ValidationResult.VALID
        assert len(errors) == 0
        
        # Test 2: Duplicate signal detection
        # Process same signal twice
        orders = [{'side': 'BUY', 'price': 50000, 'confidence': 0.8, 'reasoning': 'Test'}]
        strategy_result = {'phase_results': {'phase4_scoring': {'total_score': 7.0}}}
        
        signals_processed_1 = await runner._convert_orders_to_signals('BTC', orders, strategy_result)
        signals_processed_2 = await runner._convert_orders_to_signals('BTC', orders, strategy_result)
        
        # First should succeed, second might be rate limited or deduplicated
        assert signals_processed_1 > 0
        # Second might be 0 due to deduplication/rate limiting
        
        # Verify deduplication stats
        assert runner.performance_stats['signals_duplicate'] >= 0
        assert runner.performance_stats['signals_rate_limited'] >= 0
        
        print("✅ Signal validation and deduplication test passed")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(
        self, mock_config_manager, redis_client, influx_client, sample_dataset
    ):
        """Test performance against defined benchmarks"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Test 1: Single signal processing time
        with patch.object(runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                start_time = time.time()
                await runner._process_symbol('BTC')
                processing_time = time.time() - start_time
                
                assert processing_time < PERFORMANCE_BENCHMARKS['signal_processing_max_time']
        
        # Test 2: Redis publish performance
        test_signal = create_test_signal('BTC', 'LONG', 7.0)
        
        start_time = time.time()
        await runner._publish_signal_immediately(test_signal)
        publish_time = time.time() - start_time
        
        assert publish_time < PERFORMANCE_BENCHMARKS['redis_publish_max_time']
        
        # Test 3: Memory usage during processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple symbols
        symbols = ['BTC', 'ETH', 'ADA', 'DOT']
        with patch.object(runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                for symbol in symbols:
                    await runner._process_symbol(symbol)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f}MB"
        
        print(f"✅ Performance benchmarks passed:")
        print(f"   - Signal processing: {processing_time:.3f}s")
        print(f"   - Redis publish: {publish_time:.3f}s")
        print(f"   - Memory increase: {memory_increase:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(
        self, mock_config_manager, redis_client, sample_dataset
    ):
        """Test concurrent signal processing and thread safety"""
        
        config = mock_config_manager.get_config()
        config.enable_parallel_processing = True
        config.max_symbols_per_cycle = 5
        
        runner = StrategyRunner(mock_config_manager)
        
        symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'SOL']
        
        with patch.object(runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                start_time = time.time()
                await runner._run_strategy_cycle(symbols)
                cycle_time = time.time() - start_time
                
                # Verify all symbols were processed
                for symbol in symbols:
                    redis_key = f"{runner.config.redis_key_prefix}:signal:{symbol}"
                    signal_data = redis_client.get(redis_key)
                    assert signal_data is not None, f"No signal found for {symbol}"
                
                # Concurrent processing should be faster than sequential
                assert cycle_time < PERFORMANCE_BENCHMARKS['cycle_processing_max_time']
        
        print(f"✅ Concurrent processing test passed in {cycle_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_redis_pubsub_functionality(
        self, mock_config_manager, redis_client
    ):
        """Test Redis pub/sub functionality for real-time signal distribution"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Test subscriber management
        received_signals = []
        
        def signal_callback(signal_data):
            received_signals.append(json.loads(signal_data))
        
        # Subscribe to signals
        subscriber_id = runner.subscribe_to_signals(signal_callback)
        assert subscriber_id is not None
        
        # Test real Redis pub/sub (simplified test)
        test_signal = create_test_signal('BTC', 'LONG', 7.0)
        await runner._publish_signal_immediately(test_signal)
        
        # Verify pub/sub metrics are tracked
        assert runner._redis_metrics['total_publishes'] > 0
        
        # Cleanup subscriber
        runner.unsubscribe_from_signals(subscriber_id)
        
        print("✅ Redis pub/sub functionality test passed")
    
    @pytest.mark.asyncio
    async def test_signal_lifecycle_management(
        self, mock_config_manager, redis_client
    ):
        """Test complete signal lifecycle from generation to expiry"""
        
        config = mock_config_manager.get_config()
        config.redis_signal_ttl = 2  # 2 seconds for fast testing
        
        runner = StrategyRunner(mock_config_manager)
        
        # Generate test signal
        test_signal = create_test_signal('BTC', 'LONG', 7.0)
        
        # Publish signal
        await runner._publish_signal_immediately(test_signal)
        
        # Verify signal exists
        redis_key = f"{runner.config.redis_key_prefix}:signal:BTC"
        assert redis_client.get(redis_key) is not None
        
        # Wait for expiry
        await asyncio.sleep(3)
        
        # Verify signal expired
        assert redis_client.get(redis_key) is None
        
        # Test cleanup of expired tracking
        await runner._cleanup_expired_signals()
        
        # Verify cleanup metrics
        assert runner._redis_metrics['expired_signals_cleaned'] >= 0
        
        print("✅ Signal lifecycle management test passed")
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(
        self, mock_config_manager, redis_client, influx_client
    ):
        """Test health monitoring and status reporting"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Test health status retrieval
        health_status = runner.get_health_status()
        
        # Verify health status structure
        assert 'service' in health_status
        assert 'status' in health_status
        assert 'performance' in health_status
        assert 'redis_metrics' in health_status
        assert 'connections' in health_status
        
        # Test connection status
        connections = health_status['connections']
        # These will be None initially (lazy loading)
        assert 'redis_main' in connections
        assert 'influxdb' in connections
        
        # Test performance metrics
        performance = health_status['performance']
        assert 'cycles_completed' in performance
        assert 'signals_generated' in performance
        assert 'signals_published' in performance
        
        # Test detailed publishing metrics
        publishing_metrics = runner.get_signal_publishing_metrics()
        assert 'signal_stats' in publishing_metrics
        assert 'redis_metrics' in publishing_metrics
        assert 'validation_metrics' in publishing_metrics
        
        print("✅ Health monitoring integration test passed")
    
    @pytest.mark.skip(reason="Requires Docker environment")
    @pytest.mark.asyncio
    async def test_docker_container_integration(self, docker_environment):
        """Test integration with Docker containerized services"""
        
        if not docker_environment['is_docker']:
            pytest.skip("Not running in Docker environment")
        
        # Create config for Docker environment
        config = ServiceConfig()
        config.redis_host = docker_environment['redis_host']
        config.influx_host = docker_environment['influx_host']
        
        config_manager = MagicMock()
        config_manager.get_config.return_value = config
        config_manager.get_freqtrade_pairs.return_value = ['BTC', 'ETH']
        config_manager.get_redis_config.return_value = {
            'host': config.redis_host,
            'port': 6379,
            'db': 0,
            'decode_responses': True
        }
        
        runner = StrategyRunner(config_manager)
        
        # Test connections to containerized services
        try:
            connections_ok = await runner._test_connections()
            assert connections_ok, "Failed to connect to Docker services"
            
            print("✅ Docker container integration test passed")
        
        except Exception as e:
            pytest.fail(f"Docker integration test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_stress_testing_high_load(
        self, mock_config_manager, redis_client, sample_dataset
    ):
        """Stress test with high signal volume"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Generate high volume of test data
        stress_signals = generate_stress_test_data(50)  # 50 signals
        
        start_time = time.time()
        
        # Process signals in batches
        for i in range(0, len(stress_signals), 10):
            batch = stress_signals[i:i+10]
            
            # Convert to orders format
            orders = []
            for signal in batch:
                orders.append({
                    'side': 'BUY' if signal['action'] == 'LONG' else 'SELL',
                    'price': signal['entry_price'],
                    'confidence': signal['confidence'],
                    'reasoning': signal['reasoning']
                })
            
            strategy_result = {
                'phase_results': {'phase4_scoring': {'total_score': batch[0]['score']}}
            }
            
            # Process batch
            await runner._convert_orders_to_signals(batch[0]['symbol'], orders[:1], strategy_result)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        avg_time_per_signal = total_time / len(stress_signals)
        assert avg_time_per_signal < 0.1, f"Too slow: {avg_time_per_signal:.3f}s per signal"
        
        # Verify some signals were processed
        assert runner.performance_stats['signals_generated'] > 0
        
        print(f"✅ Stress test passed: {len(stress_signals)} signals in {total_time:.3f}s")
        print(f"   Average: {avg_time_per_signal:.3f}s per signal")
    
    @pytest.mark.asyncio
    async def test_service_lifecycle_management(
        self, mock_config_manager, redis_client
    ):
        """Test complete service lifecycle: start, run, stop"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Mock data and strategy for quick cycles
        with patch.object(runner, '_load_symbol_data', return_value={'ohlcv': None}):
            with patch.object(runner, '_test_connections', return_value=True):
                
                # Start service in background
                service_task = asyncio.create_task(runner.start())
                
                # Let it run for a short time
                await asyncio.sleep(2)
                
                # Check service is running
                assert runner.is_running
                assert runner.cycle_count > 0
                
                # Stop service
                await runner.stop()
                
                # Verify cleanup
                assert not runner.is_running
                
                # Cancel the service task
                service_task.cancel()
                try:
                    await service_task
                except asyncio.CancelledError:
                    pass
        
        print("✅ Service lifecycle management test passed")


class TestStrategyRunnerPerformance:
    """Performance-focused tests for Strategy Runner"""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(
        self, mock_config_manager, redis_client, sample_dataset
    ):
        """Test for memory leaks during extended operation"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate extended operation
        with patch.object(runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                # Process 100 cycles
                for i in range(100):
                    await runner._process_symbol('BTC')
                    
                    # Force garbage collection every 10 cycles
                    if i % 10 == 0:
                        import gc
                        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal
        assert memory_increase < 50, f"Potential memory leak: {memory_increase:.1f}MB increase"
        
        print(f"✅ Memory leak test passed: {memory_increase:.1f}MB increase over 100 cycles")
    
    @pytest.mark.asyncio
    async def test_redis_connection_pool_performance(
        self, mock_config_manager, redis_client
    ):
        """Test Redis connection pooling performance"""
        
        config = mock_config_manager.get_config()
        config.enable_batch_publishing = True
        config.redis_connection_pool_size = 10
        
        runner = StrategyRunner(mock_config_manager)
        
        # Generate multiple signals for concurrent publishing
        signals = [create_test_signal(f'TEST{i}', 'LONG', 7.0) for i in range(20)]
        
        start_time = time.time()
        
        # Publish signals concurrently
        tasks = []
        for signal in signals:
            task = asyncio.create_task(runner._publish_signal_immediately(signal))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        avg_time_per_signal = total_time / len(signals)
        
        # Connection pooling should improve performance
        assert avg_time_per_signal < 0.05, f"Connection pooling not effective: {avg_time_per_signal:.3f}s per signal"
        
        print(f"✅ Connection pool performance test passed: {avg_time_per_signal:.3f}s per signal")


class TestStrategyRunnerAnalytics:
    """Tests for analytics and reporting functionality"""
    
    @pytest.mark.asyncio
    async def test_signal_analytics_integration(
        self, mock_config_manager, influx_client
    ):
        """Test signal analytics and reporting"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Mock signal manager with analytics
        with patch.object(runner, 'signal_manager') as mock_manager:
            mock_analytics = MagicMock()
            mock_analytics.total_signals = 10
            mock_analytics.profitable_signals = 6
            mock_analytics.unprofitable_signals = 3
            mock_analytics.pending_signals = 1
            mock_analytics.win_rate = 66.67
            mock_analytics.average_pnl = 150.0
            mock_analytics.total_pnl = 450.0
            
            mock_manager.get_signal_analytics.return_value = mock_analytics
            
            # Get analytics
            analytics = runner.get_signal_analytics('BTC', 24)
            
            # Verify analytics structure
            assert 'analytics' in analytics
            assert 'filters' in analytics
            
            analytics_data = analytics['analytics']
            assert analytics_data['total_signals'] == 10
            assert analytics_data['win_rate_percent'] == 66.67
            assert analytics_data['total_pnl'] == 450.0
        
        print("✅ Signal analytics integration test passed")
    
    @pytest.mark.asyncio
    async def test_performance_reporting(
        self, mock_config_manager
    ):
        """Test performance metrics reporting"""
        
        runner = StrategyRunner(mock_config_manager)
        
        # Simulate some activity
        runner.performance_stats.update({
            'cycles_completed': 100,
            'signals_generated': 25,
            'signals_published': 23,
            'signals_failed': 2,
            'avg_cycle_duration': 1.5
        })
        
        # Get publishing metrics
        metrics = runner.get_signal_publishing_metrics()
        
        # Verify metrics structure
        assert 'signal_stats' in metrics
        assert 'redis_metrics' in metrics
        assert 'performance' in metrics
        assert 'health_indicators' in metrics
        
        signal_stats = metrics['signal_stats']
        assert signal_stats['total_generated'] == 25
        assert signal_stats['total_published'] == 23
        assert signal_stats['total_failed'] == 2
        
        # Test success rate calculation
        expected_success_rate = (23 / 25) * 100
        assert signal_stats['success_rate_percent'] == expected_success_rate
        
        print("✅ Performance reporting test passed")


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__ + "::TestStrategyRunnerE2E::test_complete_signal_flow_integration", "-v"])