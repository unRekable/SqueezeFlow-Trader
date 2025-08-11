#!/usr/bin/env python3
"""
Integration Tests for Monitoring Services
Tests for Health Monitor and Performance Monitor integration with Strategy Runner

This test suite covers:
1. Health monitoring integration
2. Performance monitoring integration
3. Monitoring service interoperability
4. Alert and notification systems
5. Dashboard data generation
"""

import pytest
import asyncio
import time
import json
import redis
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock, AsyncMock

from services.health_monitor import HealthMonitor, HealthStatus, HealthCheck
from services.performance_monitor import PerformanceMonitor, PerformanceAlert, PerformanceTimer
from services.strategy_runner import StrategyRunner
from tests.conftest import wait_for_condition, PERFORMANCE_BENCHMARKS


class TestHealthMonitorIntegration:
    """Integration tests for Health Monitor service"""
    
    @pytest.mark.asyncio
    async def test_health_monitor_startup_and_connections(
        self, mock_config_manager, redis_client, influx_client
    ):
        """Test health monitor startup and connection testing"""
        
        monitor = HealthMonitor(mock_config_manager)
        
        # Test connection checks
        redis_health = await monitor._check_redis_health()
        assert redis_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert redis_health.response_time_ms > 0
        
        influx_health = await monitor._check_influxdb_health()
        assert influx_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        system_health = await monitor._check_system_health()
        assert system_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        print("✅ Health monitor connection tests passed")
    
    @pytest.mark.asyncio
    async def test_health_monitoring_with_strategy_runner(
        self, mock_config_manager, redis_client, sample_dataset
    ):
        """Test health monitoring integration with running Strategy Runner"""
        
        # Start health monitor
        health_monitor = HealthMonitor(mock_config_manager)
        
        # Start strategy runner (mocked for quick test)
        strategy_runner = StrategyRunner(mock_config_manager)
        
        # Mock data loading and strategy processing
        with patch.object(strategy_runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(strategy_runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                # Process a few symbols to generate activity
                await strategy_runner._process_symbol('BTC')
                await strategy_runner._process_symbol('ETH')
                
                # Wait a moment for Redis keys to be set
                await asyncio.sleep(0.1)
                
                # Check strategy runner health
                strategy_health = await health_monitor._check_strategy_runner_health()
                
                # Should detect activity
                assert strategy_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
                assert 'redis_key_exists' in strategy_health.details
        
        print("✅ Health monitoring with Strategy Runner integration passed")
    
    @pytest.mark.asyncio
    async def test_health_status_aggregation(
        self, mock_config_manager, redis_client, influx_client
    ):
        """Test overall health status aggregation"""
        
        monitor = HealthMonitor(mock_config_manager)
        
        # Perform health checks
        await monitor._perform_all_health_checks()
        
        # Wait for checks to complete
        await asyncio.sleep(1)
        
        # Get overall health status
        overall_status = monitor.get_overall_health_status()
        
        assert 'status' in overall_status
        assert 'message' in overall_status
        assert 'service_count' in overall_status
        assert 'status_breakdown' in overall_status
        
        # Should have checked multiple services
        assert overall_status['service_count'] > 0
        
        # Get comprehensive report
        comprehensive_report = monitor.get_comprehensive_health_report()
        
        assert 'overall_status' in comprehensive_report
        assert 'services' in comprehensive_report
        assert 'system_metrics' in comprehensive_report
        
        print("✅ Health status aggregation test passed")
    
    @pytest.mark.asyncio
    async def test_health_alert_system(
        self, mock_config_manager, redis_client
    ):
        """Test health monitoring alert system"""
        
        monitor = HealthMonitor(mock_config_manager)
        
        # Create a mock unhealthy service
        unhealthy_check = HealthCheck(
            name='test_service',
            status=HealthStatus.CRITICAL,
            response_time_ms=5000,
            message='Test critical failure',
            details={'error': 'mock_error'},
            timestamp=datetime.now()
        )
        
        monitor.health_checks['test_service'] = unhealthy_check
        monitor.monitored_services['test_service'] = {'critical': True}
        
        # Check alert conditions
        await monitor._check_alert_conditions()
        
        # Should have triggered alerts
        assert monitor.performance_metrics['alerts_triggered'] > 0
        
        print("✅ Health alert system test passed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("fastapi", minversion=None),
        reason="FastAPI not available"
    )
    @pytest.mark.asyncio
    async def test_health_http_endpoints(
        self, mock_config_manager
    ):
        """Test HTTP health endpoints (requires FastAPI)"""
        
        monitor = HealthMonitor(mock_config_manager)
        
        # Setup HTTP server
        await monitor._setup_http_server()
        
        # Verify FastAPI app was created
        assert monitor.app is not None
        
        # Test endpoint creation
        routes = [route.path for route in monitor.app.routes]
        expected_routes = ['/health', '/health/detailed', '/metrics', '/status']
        
        for route in expected_routes:
            assert route in routes or any(route in r for r in routes)
        
        print("✅ Health HTTP endpoints test passed")


class TestPerformanceMonitorIntegration:
    """Integration tests for Performance Monitor service"""
    
    @pytest.mark.asyncio
    async def test_performance_monitor_startup(
        self, mock_config_manager
    ):
        """Test performance monitor startup and initialization"""
        
        monitor = PerformanceMonitor(mock_config_manager)
        
        # Check initialization
        assert monitor.metrics_collector is not None
        assert len(monitor.performance_alerts) > 0  # Default alerts
        assert not monitor.is_monitoring  # Not started yet
        
        # Test metric recording
        monitor.record_metric('test_metric', 100.0, 'milliseconds')
        
        # Check metric was recorded
        metrics = monitor.metrics_collector.get_metrics('test_metric')
        assert len(metrics) == 1
        assert metrics[0].value == 100.0
        
        print("✅ Performance monitor startup test passed")
    
    @pytest.mark.asyncio
    async def test_performance_timing_integration(
        self, mock_config_manager, sample_dataset
    ):
        """Test performance timing with Strategy Runner operations"""
        
        perf_monitor = PerformanceMonitor(mock_config_manager)
        strategy_runner = StrategyRunner(mock_config_manager)
        
        with patch.object(strategy_runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(strategy_runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                # Use performance timer context manager
                with PerformanceTimer(perf_monitor, 'symbol_processing', {'symbol': 'BTC'}):
                    await strategy_runner._process_symbol('BTC')
                
                # Check timing was recorded
                metrics = perf_monitor.metrics_collector.get_metrics('symbol_processing_time_ms')
                assert len(metrics) > 0
                
                # Verify operation count
                assert perf_monitor.operation_counts['symbol_processing'] == 1
        
        print("✅ Performance timing integration test passed")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(
        self, mock_config_manager
    ):
        """Test automated performance metrics collection"""
        
        monitor = PerformanceMonitor(mock_config_manager)
        
        # Record various metrics
        test_metrics = [
            ('cpu_usage_percent', 45.5, 'percent'),
            ('memory_usage_mb', 512.0, 'megabytes'),
            ('signal_processing_time_ms', 150.0, 'milliseconds'),
            ('redis_publish_time_ms', 25.0, 'milliseconds')
        ]
        
        for metric_name, value, unit in test_metrics:
            monitor.record_metric(metric_name, value, unit)
        
        # Test aggregated statistics
        for metric_name, _, _ in test_metrics:
            stats = monitor.metrics_collector.get_aggregated_stats(metric_name)
            assert 'mean' in stats
            assert 'max' in stats
            assert 'min' in stats
            assert 'count' in stats
            assert stats['count'] == 1
        
        print("✅ Performance metrics collection test passed")
    
    @pytest.mark.asyncio
    async def test_performance_alert_system(
        self, mock_config_manager
    ):
        """Test performance alert system"""
        
        monitor = PerformanceMonitor(mock_config_manager)
        
        # Create a test alert
        test_alert = PerformanceAlert(
            metric_name='test_response_time',
            threshold_value=100.0,
            comparison='gt',
            duration_seconds=1,
            severity='warning',
            message_template='High response time: {value}ms'
        )
        
        monitor.add_performance_alert(test_alert)
        
        # Record metrics that should trigger the alert
        for i in range(5):
            monitor.record_metric('test_response_time', 150.0, 'milliseconds')
            time.sleep(0.3)  # Small delay to ensure different timestamps
        
        # Simulate alert monitoring (normally done in background task)
        current_time = datetime.now()
        recent_metrics = monitor.metrics_collector.get_metrics('test_response_time', 5)
        
        if recent_metrics:
            latest_value = recent_metrics[-1].value
            condition_met = test_alert.check_condition(latest_value)
            assert condition_met  # Should trigger alert
        
        print("✅ Performance alert system test passed")
    
    @pytest.mark.asyncio
    async def test_performance_dashboard_data(
        self, mock_config_manager
    ):
        """Test performance dashboard data generation"""
        
        monitor = PerformanceMonitor(mock_config_manager)
        
        # Record sample metrics
        sample_data = [
            ('cpu_usage_percent', [45.0, 50.0, 55.0]),
            ('memory_usage_percent', [60.0, 65.0, 70.0]),
            ('signal_processing_time_ms', [100.0, 120.0, 90.0])
        ]
        
        for metric_name, values in sample_data:
            for value in values:
                monitor.record_metric(metric_name, value)
        
        # Update dashboard data (normally done in background task)
        dashboard_data = {}
        
        for metric_name, _ in sample_data:
            stats = monitor.metrics_collector.get_aggregated_stats(metric_name)
            if stats:
                dashboard_data[metric_name] = stats
        
        # Verify dashboard data structure
        for metric_name, _ in sample_data:
            assert metric_name in dashboard_data
            assert 'mean' in dashboard_data[metric_name]
            assert 'max' in dashboard_data[metric_name]
            assert 'trend' in dashboard_data[metric_name]
        
        print("✅ Performance dashboard data test passed")
    
    @pytest.mark.asyncio
    async def test_memory_tracking_integration(
        self, mock_config_manager
    ):
        """Test memory tracking functionality"""
        
        monitor = PerformanceMonitor(mock_config_manager)
        
        # Enable memory tracking
        monitor.enable_memory_tracking()
        assert monitor.memory_tracker_enabled
        
        # Simulate memory usage
        large_data = [i for i in range(10000)]  # Create some data
        
        # Record memory metric manually (background task would do this)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024**2
        monitor.record_metric('process_memory_mb', memory_mb, 'megabytes')
        
        # Check metric was recorded
        memory_metrics = monitor.metrics_collector.get_metrics('process_memory_mb')
        assert len(memory_metrics) > 0
        assert memory_metrics[0].value > 0
        
        # Clean up
        del large_data
        
        print("✅ Memory tracking integration test passed")


class TestMonitoringServiceInteroperability:
    """Test interaction between monitoring services"""
    
    @pytest.mark.asyncio
    async def test_combined_monitoring_services(
        self, mock_config_manager, redis_client
    ):
        """Test health and performance monitors working together"""
        
        health_monitor = HealthMonitor(mock_config_manager)
        perf_monitor = PerformanceMonitor(mock_config_manager)
        
        # Both should be able to use Redis
        assert health_monitor.redis_client.ping()
        
        # Performance monitor should be able to publish metrics
        perf_monitor.record_metric('test_integration_metric', 75.0, 'percent')
        
        # Health monitor should be able to check Redis health
        redis_health = await health_monitor._check_redis_health()
        assert redis_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        # Verify both services can coexist
        assert len(perf_monitor.metrics_collector.metrics) > 0
        assert redis_health.response_time_ms > 0
        
        print("✅ Combined monitoring services test passed")
    
    @pytest.mark.asyncio
    async def test_monitoring_with_strategy_runner_full_integration(
        self, mock_config_manager, redis_client, influx_client, sample_dataset
    ):
        """Test full integration: Strategy Runner + Health Monitor + Performance Monitor"""
        
        # Initialize all services
        strategy_runner = StrategyRunner(mock_config_manager)
        health_monitor = HealthMonitor(mock_config_manager)
        perf_monitor = PerformanceMonitor(mock_config_manager)
        
        # Mock strategy runner operations
        with patch.object(strategy_runner, '_load_symbol_data', return_value=sample_dataset):
            with patch.object(strategy_runner.strategy, 'process') as mock_process:
                mock_process.return_value = {
                    'orders': [{'side': 'BUY', 'price': 50000, 'confidence': 0.8}],
                    'phase_results': {'phase4_scoring': {'total_score': 7.0}}
                }
                
                # Monitor performance while processing
                with PerformanceTimer(perf_monitor, 'full_integration_test'):
                    # Process symbols
                    await strategy_runner._process_symbol('BTC')
                    await strategy_runner._process_symbol('ETH')
                
                # Check health status
                await health_monitor._perform_all_health_checks()
        
        # Verify integration results
        
        # 1. Strategy Runner should have published signals
        btc_signal = redis_client.get(f"{strategy_runner.config.redis_key_prefix}:signal:BTC")
        assert btc_signal is not None
        
        # 2. Performance Monitor should have timing data
        timing_metrics = perf_monitor.metrics_collector.get_metrics('full_integration_test_time_ms')
        assert len(timing_metrics) > 0
        
        # 3. Health Monitor should show healthy services
        overall_health = health_monitor.get_overall_health_status()
        assert overall_health['service_count'] > 0
        
        # 4. Get comprehensive status from all services
        integration_status = {
            'strategy_runner': strategy_runner.get_health_status(),
            'health_monitor': health_monitor.get_overall_health_status(),
            'performance_monitor': perf_monitor.get_metrics_summary(5)
        }
        
        # Verify all services are reporting
        assert 'service' in integration_status['strategy_runner']
        assert 'status' in integration_status['health_monitor']
        assert 'metrics' in integration_status['performance_monitor']
        
        print("✅ Full monitoring integration test passed")
        print(f"   - Signals published: {strategy_runner.performance_stats['signals_published']}")
        print(f"   - Health checks: {len(health_monitor.health_checks)}")
        print(f"   - Performance metrics: {len(perf_monitor.metrics_collector.metrics)}")
    
    @pytest.mark.asyncio
    async def test_monitoring_alert_coordination(
        self, mock_config_manager, redis_client
    ):
        """Test coordination between health and performance alerts"""
        
        health_monitor = HealthMonitor(mock_config_manager)
        perf_monitor = PerformanceMonitor(mock_config_manager)
        
        # Set up alert listening
        alert_messages = []
        
        def mock_redis_publish(channel, message):
            if 'alert' in channel:
                alert_messages.append(json.loads(message))
            return 1  # Mock successful publish
        
        # Patch Redis publish for both monitors
        with patch.object(health_monitor.redis_client, 'publish', side_effect=mock_redis_publish):
            with patch.object(perf_monitor.redis_client, 'publish', side_effect=mock_redis_publish):
                
                # Trigger health alert
                critical_health = HealthCheck(
                    name='test_service',
                    status=HealthStatus.CRITICAL,
                    response_time_ms=1000,
                    message='Test critical failure',
                    details={},
                    timestamp=datetime.now()
                )
                
                await health_monitor._trigger_alert('CRITICAL: Test service failure', critical_health)
                
                # Trigger performance alert
                perf_alert = PerformanceAlert(
                    metric_name='test_metric',
                    threshold_value=100.0,
                    comparison='gt',
                    duration_seconds=1,
                    severity='warning',
                    message_template='High value: {value}'
                )
                
                await perf_monitor._trigger_performance_alert(perf_alert, 150.0, 60.0)
        
        # Verify alerts were coordinated
        assert len(alert_messages) == 2
        
        # Check alert message structure
        for alert in alert_messages:
            assert 'timestamp' in alert
            assert 'message' in alert
            assert 'severity' in alert
        
        print("✅ Monitoring alert coordination test passed")


class TestMonitoringPerformanceBenchmarks:
    """Performance benchmarks for monitoring services"""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(
        self, mock_config_manager, redis_client, influx_client
    ):
        """Test health check performance benchmarks"""
        
        monitor = HealthMonitor(mock_config_manager)
        
        # Measure health check performance
        start_time = time.time()
        
        # Perform multiple health check cycles
        for _ in range(10):
            await monitor._perform_all_health_checks()
        
        total_time = time.time() - start_time
        avg_time_per_cycle = total_time / 10
        
        # Performance assertions
        assert avg_time_per_cycle < 2.0, f"Health checks too slow: {avg_time_per_cycle:.3f}s per cycle"
        
        # Check individual service response times
        for service_name, health_check in monitor.health_checks.items():
            assert health_check.response_time_ms < 5000, f"{service_name} response time too high: {health_check.response_time_ms}ms"
        
        print(f"✅ Health check performance: {avg_time_per_cycle:.3f}s per cycle")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_overhead(
        self, mock_config_manager
    ):
        """Test performance monitoring overhead"""
        
        monitor = PerformanceMonitor(mock_config_manager)
        
        # Measure overhead of metric recording
        start_time = time.time()
        
        # Record many metrics
        for i in range(1000):
            monitor.record_metric(f'test_metric_{i % 10}', float(i), 'count')
        
        recording_time = time.time() - start_time
        avg_time_per_metric = recording_time / 1000
        
        # Performance assertions
        assert avg_time_per_metric < 0.001, f"Metric recording too slow: {avg_time_per_metric:.6f}s per metric"
        
        # Memory usage should be reasonable
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024**2
        
        # Should not use excessive memory for metrics
        assert memory_mb < 200, f"High memory usage: {memory_mb:.1f}MB"
        
        print(f"✅ Performance metrics overhead: {avg_time_per_metric:.6f}s per metric")
    
    @pytest.mark.asyncio
    async def test_monitoring_resource_usage(
        self, mock_config_manager
    ):
        """Test resource usage of monitoring services"""
        
        health_monitor = HealthMonitor(mock_config_manager)
        perf_monitor = PerformanceMonitor(mock_config_manager)
        
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**2
        initial_cpu = process.cpu_percent()
        
        # Run monitoring operations
        start_time = time.time()
        
        # Simulate monitoring activity
        for i in range(50):
            await health_monitor._perform_all_health_checks()
            perf_monitor.record_metric('test_metric', float(i), 'count')
            
            if i % 10 == 0:
                # Simulate some work
                await asyncio.sleep(0.01)
        
        # Measure final resource usage
        final_memory = process.memory_info().rss / 1024**2
        final_cpu = process.cpu_percent()
        
        memory_increase = final_memory - initial_memory
        total_time = time.time() - start_time
        
        # Resource usage assertions
        assert memory_increase < 50, f"Memory increase too high: {memory_increase:.1f}MB"
        assert total_time < 10, f"Monitoring operations too slow: {total_time:.3f}s"
        
        print(f"✅ Monitoring resource usage: {memory_increase:.1f}MB increase, {total_time:.3f}s total")


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__ + "::TestMonitoringServiceInteroperability::test_combined_monitoring_services", "-v"])