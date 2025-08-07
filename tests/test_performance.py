"""
Performance Tests for SqueezeFlow Trader
Load testing, memory profiling, and throughput benchmarks
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import gc
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Import components to test
from strategies.squeezeflow.strategy import SqueezeFlowStrategy
from data.pipeline import DataPipeline
from backtest.engine import BacktestEngine
from data.processors.cvd_calculator import CVDCalculator


class TestStrategyPerformance:
    """Performance tests for strategy processing"""
    
    @pytest.mark.performance
    def test_strategy_processing_speed(self, sample_dataset, sample_portfolio_state):
        """Test strategy processing speed under normal load"""
        strategy = SqueezeFlowStrategy()
        
        # Warm up
        strategy.process(sample_dataset, sample_portfolio_state)
        
        # Measure processing time
        times = []
        for _ in range(10):
            start_time = time.time()
            result = strategy.process(sample_dataset, sample_portfolio_state)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Performance benchmarks
        assert avg_time < 0.1, f"Average processing time too slow: {avg_time:.3f}s"
        assert max_time < 0.2, f"Maximum processing time too slow: {max_time:.3f}s"
        
        # Verify functionality wasn't compromised
        assert 'orders' in result
        assert 'phase_results' in result
        
    @pytest.mark.performance
    def test_strategy_memory_usage(self, sample_dataset, sample_portfolio_state):
        """Test strategy memory usage during processing"""
        strategy = SqueezeFlowStrategy()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process multiple times to check for memory leaks
        for i in range(50):
            result = strategy.process(sample_dataset, sample_portfolio_state)
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Memory should not grow excessively (allow 50MB growth)
                assert memory_growth < 50_000_000, f"Excessive memory growth: {memory_growth / 1_000_000:.1f}MB"
        
        # Final memory check
        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory
        
        # Total growth should be reasonable
        assert total_growth < 100_000_000, f"Total memory growth too high: {total_growth / 1_000_000:.1f}MB"
        
    @pytest.mark.performance
    def test_strategy_concurrent_processing(self, sample_dataset, sample_portfolio_state):
        """Test strategy performance under concurrent load"""
        
        def process_strategy():
            strategy = SqueezeFlowStrategy()
            start_time = time.time()
            result = strategy.process(sample_dataset, sample_portfolio_state)
            end_time = time.time()
            return end_time - start_time, len(result.get('orders', []))
        
        # Test concurrent processing
        num_threads = 4
        num_processes_per_thread = 5
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            futures = [executor.submit(process_strategy) for _ in range(num_threads * num_processes_per_thread)]
            results = [future.result() for future in futures]
            end_time = time.time()
        
        total_time = end_time - start_time
        processing_times = [result[0] for result in results]
        
        # Performance assertions
        avg_processing_time = sum(processing_times) / len(processing_times)
        assert avg_processing_time < 0.2, f"Concurrent processing too slow: {avg_processing_time:.3f}s"
        assert total_time < 5.0, f"Total concurrent time too slow: {total_time:.3f}s"
        
        # All processes should have completed successfully
        assert len(results) == num_threads * num_processes_per_thread


class TestDataPipelinePerformance:
    """Performance tests for data pipeline"""
    
    @pytest.mark.performance
    def test_data_loading_speed(self):
        """Test data loading performance"""
        pipeline = DataPipeline()
        
        # Mock data loading to test pipeline overhead
        with patch.object(pipeline.influx_client, 'get_ohlcv_data') as mock_ohlcv:
            with patch.object(pipeline.influx_client, 'get_volume_data') as mock_volume:
                
                # Create large mock datasets
                large_ohlcv = pd.DataFrame({
                    'time': pd.date_range('2024-01-01', periods=10000, freq='1min'),
                    'open': np.random.uniform(45000, 55000, 10000),
                    'high': np.random.uniform(50000, 60000, 10000),
                    'low': np.random.uniform(40000, 50000, 10000),
                    'close': np.random.uniform(45000, 55000, 10000),
                    'volume': np.random.uniform(1000, 5000, 10000)
                })
                
                large_volume = pd.DataFrame({
                    'time': pd.date_range('2024-01-01', periods=10000, freq='1min'),
                    'total_volume': np.random.uniform(1000, 5000, 10000),
                    'total_vbuy': np.random.uniform(500, 3000, 10000),
                    'total_vsell': np.random.uniform(300, 2500, 10000)
                })
                
                mock_ohlcv.return_value = large_ohlcv
                mock_volume.return_value = large_volume
                
                # Test loading speed
                start_time = time.time()
                dataset = pipeline.get_complete_dataset(
                    symbol='BTC',
                    start_time=datetime(2024, 1, 1),
                    end_time=datetime(2024, 1, 2),
                    timeframe='1m'
                )
                end_time = time.time()
                
                loading_time = end_time - start_time
                
                # Should handle large datasets efficiently
                assert loading_time < 2.0, f"Data loading too slow: {loading_time:.3f}s"
                assert len(dataset['ohlcv']) == 10000
                
    @pytest.mark.performance  
    def test_cvd_calculation_performance(self):
        """Test CVD calculation performance with large datasets"""
        calculator = CVDCalculator()
        
        # Create large volume dataset
        large_volume_data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=50000, freq='1min'),
            'total_vbuy': np.random.uniform(1000, 10000, 50000),
            'total_vsell': np.random.uniform(800, 8000, 50000)
        })
        
        # Test spot CVD calculation
        start_time = time.time()
        spot_cvd = calculator.calculate_spot_cvd(large_volume_data)
        end_time = time.time()
        
        spot_time = end_time - start_time
        
        # Test futures CVD calculation
        start_time = time.time()
        futures_cvd = calculator.calculate_futures_cvd(large_volume_data)
        end_time = time.time()
        
        futures_time = end_time - start_time
        
        # Test divergence calculation
        start_time = time.time()
        divergence = calculator.calculate_cvd_divergence(spot_cvd, futures_cvd)
        end_time = time.time()
        
        divergence_time = end_time - start_time
        
        # Performance assertions
        assert spot_time < 1.0, f"Spot CVD calculation too slow: {spot_time:.3f}s"
        assert futures_time < 1.0, f"Futures CVD calculation too slow: {futures_time:.3f}s"
        assert divergence_time < 0.5, f"Divergence calculation too slow: {divergence_time:.3f}s"
        
        # Verify results
        assert len(spot_cvd) == 50000
        assert len(futures_cvd) == 50000
        assert len(divergence) == 50000
        
    @pytest.mark.performance
    def test_data_pipeline_caching_performance(self):
        """Test data pipeline caching performance"""
        pipeline = DataPipeline()
        
        # Mock symbol discovery
        with patch.object(pipeline.symbol_discovery, 'discover_symbols_from_database') as mock_discover:
            mock_discover.return_value = ['BTC', 'ETH', 'ADA'] * 100  # Large list
            
            # First call (no cache)
            start_time = time.time()
            symbols1 = pipeline.discover_available_symbols()
            first_call_time = time.time() - start_time
            
            # Second call (with cache)
            start_time = time.time()
            symbols2 = pipeline.discover_available_symbols()
            cached_call_time = time.time() - start_time
            
            # Third call (with cache)
            start_time = time.time()
            symbols3 = pipeline.discover_available_symbols()
            second_cached_time = time.time() - start_time
            
            # Cache should be much faster
            assert cached_call_time < first_call_time / 10, f"Cache not effective: {cached_call_time:.6f}s vs {first_call_time:.6f}s"
            assert second_cached_time < 0.001, f"Repeated cache access too slow: {second_cached_time:.6f}s"
            
            # Results should be identical
            assert symbols1 == symbols2 == symbols3
            
            # Mock should only be called once
            assert mock_discover.call_count == 1


class TestBacktestEnginePerformance:
    """Performance tests for backtest engine"""
    
    @pytest.mark.performance
    def test_backtest_engine_throughput(self, mock_data_pipeline):
        """Test backtest engine order processing throughput"""
        engine = BacktestEngine(initial_balance=100000.0)
        
        # Create large dataset
        large_ohlcv = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=1000, freq='5min'),
            'open': np.random.uniform(45000, 55000, 1000),
            'high': np.random.uniform(50000, 60000, 1000),
            'low': np.random.uniform(40000, 50000, 1000),
            'close': np.random.uniform(45000, 55000, 1000),
            'volume': np.random.uniform(1000, 5000, 1000)
        })
        
        large_dataset = {
            'symbol': 'BTCUSDT',
            'timeframe': '5m',
            'start_time': datetime(2024, 1, 1),
            'end_time': datetime(2024, 1, 5),
            'ohlcv': large_ohlcv,
            'spot_cvd': pd.Series(np.random.uniform(-10000, 10000, 1000)),
            'futures_cvd': pd.Series(np.random.uniform(-15000, 15000, 1000)),
            'cvd_divergence': pd.Series(np.random.uniform(-5000, 5000, 1000)),
            'metadata': {'data_points': 1000}
        }
        
        # Create strategy that generates many orders
        mock_strategy = MagicMock()
        orders = []
        for i in range(100):  # Generate 100 orders
            orders.append({
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.001,
                'price': 50000 + i * 10,
                'timestamp': datetime.now()
            })
        
        mock_strategy.process.return_value = {
            'orders': orders,
            'phase_results': {}
        }
        
        # Mock data pipeline
        with patch.object(engine, 'data_pipeline', mock_data_pipeline):
            mock_data_pipeline.get_complete_dataset.return_value = large_dataset
            mock_data_pipeline.validate_data_quality.return_value = {'overall_quality': True}
            
            start_time = time.time()
            result = engine.run(
                strategy=mock_strategy,
                symbol='BTCUSDT',
                start_date='2024-01-01',
                end_date='2024-01-05'
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Calculate throughput
            orders_per_second = len(orders) / execution_time if execution_time > 0 else float('inf')
            
            # Performance assertions
            assert execution_time < 5.0, f"Backtest execution too slow: {execution_time:.3f}s"
            assert orders_per_second > 50, f"Order throughput too low: {orders_per_second:.1f} orders/sec"
            assert result['total_trades'] > 0
            
    @pytest.mark.performance
    def test_portfolio_update_performance(self):
        """Test portfolio update performance"""
        from backtest.core.portfolio import Portfolio
        
        portfolio = Portfolio(100000.0)
        
        # Test many position updates
        start_time = time.time()
        
        for i in range(1000):
            success = portfolio.open_long_position(
                symbol=f'TEST{i % 10}USDT',  # Reuse symbols to test position updates
                quantity=0.001,
                price=50000 + i,
                timestamp=datetime.now()
            )
            
            if i % 100 == 0:  # Check performance periodically
                current_time = time.time()
                elapsed = current_time - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else float('inf')
                
                # Should maintain good throughput
                assert rate > 100, f"Portfolio update rate too slow: {rate:.1f} updates/sec at iteration {i}"
        
        end_time = time.time()
        total_time = end_time - start_time
        final_rate = 1000 / total_time
        
        assert final_rate > 200, f"Final portfolio update rate too slow: {final_rate:.1f} updates/sec"


class TestConcurrencyPerformance:
    """Performance tests for concurrent operations"""
    
    @pytest.mark.performance
    def test_concurrent_strategy_execution(self, sample_dataset, sample_portfolio_state):
        """Test concurrent strategy execution performance"""
        
        def run_strategy_batch(batch_size=10):
            """Run multiple strategy executions"""
            strategy = SqueezeFlowStrategy()
            times = []
            
            for _ in range(batch_size):
                start_time = time.time()
                result = strategy.process(sample_dataset, sample_portfolio_state)
                end_time = time.time()
                times.append(end_time - start_time)
            
            return times
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for num_workers in concurrency_levels:
            start_time = time.time()
            
            if num_workers == 1:
                # Sequential execution
                times = run_strategy_batch(20)
            else:
                # Concurrent execution
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    batch_size = 20 // num_workers
                    futures = [executor.submit(run_strategy_batch, batch_size) for _ in range(num_workers)]
                    batch_results = [future.result() for future in futures]
                    times = []
                    for batch in batch_results:
                        times.extend(batch)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            results[num_workers] = {
                'total_time': total_time,
                'avg_individual_time': sum(times) / len(times),
                'throughput': len(times) / total_time
            }
        
        # Analyze concurrency performance
        sequential_throughput = results[1]['throughput']
        
        for workers in [2, 4, 8]:
            if workers in results:
                concurrent_throughput = results[workers]['throughput']
                speedup = concurrent_throughput / sequential_throughput
                
                # Should see some speedup with concurrency (at least 1.2x for 2 workers)
                expected_speedup = min(workers * 0.6, 2.0)  # Don't expect linear scaling
                assert speedup >= expected_speedup, f"Insufficient speedup with {workers} workers: {speedup:.2f}x"
                
    @pytest.mark.performance
    def test_memory_usage_under_load(self, sample_dataset, sample_portfolio_state):
        """Test memory usage under sustained load"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        strategy = SqueezeFlowStrategy()
        memory_measurements = []
        
        # Sustained load test
        for i in range(200):
            result = strategy.process(sample_dataset, sample_portfolio_state)
            
            # Measure memory every 20 iterations
            if i % 20 == 0:
                gc.collect()  # Force garbage collection
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                memory_measurements.append(memory_growth)
                
                # Memory growth should be bounded
                assert memory_growth < 200_000_000, f"Excessive memory growth: {memory_growth / 1_000_000:.1f}MB at iteration {i}"
        
        # Check for memory leaks (growth should stabilize)
        if len(memory_measurements) >= 5:
            recent_growth = memory_measurements[-3:]
            early_growth = memory_measurements[:3]
            
            avg_recent = sum(recent_growth) / len(recent_growth)
            avg_early = sum(early_growth) / len(early_growth)
            
            # Recent memory usage shouldn't be dramatically higher than early usage
            growth_ratio = avg_recent / max(avg_early, 1)
            assert growth_ratio < 3.0, f"Potential memory leak detected: {growth_ratio:.2f}x growth"


class TestScalabilityPerformance:
    """Scalability tests for different data sizes"""
    
    @pytest.mark.performance
    def test_data_size_scalability(self):
        """Test performance scaling with different data sizes"""
        calculator = CVDCalculator()
        
        data_sizes = [100, 500, 1000, 5000, 10000]
        performance_results = {}
        
        for size in data_sizes:
            # Create test data of specific size
            test_data = pd.DataFrame({
                'time': pd.date_range('2024-01-01', periods=size, freq='1min'),
                'total_vbuy': np.random.uniform(1000, 5000, size),
                'total_vsell': np.random.uniform(800, 4000, size)
            })
            
            # Measure CVD calculation time
            start_time = time.time()
            cvd = calculator.calculate_spot_cvd(test_data)
            end_time = time.time()
            
            calculation_time = end_time - start_time
            performance_results[size] = calculation_time
            
            # Performance should scale reasonably (roughly linear)
            throughput = size / calculation_time if calculation_time > 0 else float('inf')
            assert throughput > 1000, f"CVD calculation too slow for {size} points: {throughput:.1f} points/sec"
        
        # Check scaling behavior
        small_size_time = performance_results[100]
        large_size_time = performance_results[10000]
        
        # Time should not increase more than proportionally
        scaling_factor = large_size_time / small_size_time
        expected_factor = 10000 / 100  # Linear scaling
        
        # Allow up to 2x worse than linear scaling
        assert scaling_factor <= expected_factor * 2, f"Poor scaling: {scaling_factor:.2f}x for 100x data size"
        
    @pytest.mark.performance  
    def test_strategy_data_size_scaling(self, sample_portfolio_state):
        """Test strategy performance with different dataset sizes"""
        strategy = SqueezeFlowStrategy()
        
        sizes = [50, 200, 500, 1000]
        times = {}
        
        for size in sizes:
            # Create dataset of specific size
            dataset = {
                'symbol': 'BTCUSDT',
                'timeframe': '5m',
                'ohlcv': pd.DataFrame({
                    'time': pd.date_range('2024-01-01', periods=size, freq='5min'),
                    'open': np.random.uniform(45000, 55000, size),
                    'high': np.random.uniform(50000, 60000, size),
                    'low': np.random.uniform(40000, 50000, size),
                    'close': np.random.uniform(45000, 55000, size),
                    'volume': np.random.uniform(1000, 5000, size)
                }),
                'spot_cvd': pd.Series(np.random.uniform(-10000, 10000, size)),
                'futures_cvd': pd.Series(np.random.uniform(-15000, 15000, size)),
                'cvd_divergence': pd.Series(np.random.uniform(-5000, 5000, size)),
                'metadata': {'data_points': size}
            }
            
            # Measure processing time
            start_time = time.time()
            result = strategy.process(dataset, sample_portfolio_state)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times[size] = processing_time
            
            # Should handle larger datasets efficiently
            assert processing_time < 1.0, f"Strategy too slow for {size} points: {processing_time:.3f}s"
        
        # Check that processing time doesn't grow too quickly
        if 1000 in times and 50 in times:
            time_ratio = times[1000] / times[50]
            size_ratio = 1000 / 50
            
            # Processing time should not grow faster than data size
            assert time_ratio <= size_ratio, f"Poor strategy scaling: {time_ratio:.2f}x time for {size_ratio:.1f}x data"


# Performance benchmark constants
PERFORMANCE_BENCHMARKS = {
    'strategy_processing_max_time': 0.1,      # 100ms per strategy execution
    'cvd_calculation_throughput': 10000,      # 10k points per second
    'order_execution_max_time': 0.001,        # 1ms per order
    'backtest_throughput': 50,                # 50 orders per second
    'memory_growth_limit': 100_000_000,       # 100MB max growth
    'concurrent_speedup_min': 1.2,            # 1.2x speedup with 2 workers
    'data_loading_max_time': 2.0,             # 2 seconds for large datasets
    'cache_speedup_min': 10                   # 10x speedup from caching
}


class TestPerformanceBenchmarks:
    """Test against established performance benchmarks"""
    
    @pytest.mark.performance
    def test_all_benchmarks_summary(self, sample_dataset, sample_portfolio_state):
        """Summary test of all performance benchmarks"""
        
        # Strategy processing benchmark
        strategy = SqueezeFlowStrategy()
        start_time = time.time()
        result = strategy.process(sample_dataset, sample_portfolio_state)
        strategy_time = time.time() - start_time
        
        assert strategy_time <= PERFORMANCE_BENCHMARKS['strategy_processing_max_time']
        
        # CVD calculation benchmark
        calculator = CVDCalculator()
        test_data = pd.DataFrame({
            'total_vbuy': np.random.uniform(1000, 5000, 5000),
            'total_vsell': np.random.uniform(800, 4000, 5000)
        })
        
        start_time = time.time()
        cvd = calculator.calculate_spot_cvd(test_data)
        cvd_time = time.time() - start_time
        
        cvd_throughput = 5000 / cvd_time if cvd_time > 0 else float('inf')
        assert cvd_throughput >= PERFORMANCE_BENCHMARKS['cvd_calculation_throughput']
        
        # Memory usage benchmark
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run multiple strategy executions
        for _ in range(10):
            strategy.process(sample_dataset, sample_portfolio_state)
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        assert memory_growth <= PERFORMANCE_BENCHMARKS['memory_growth_limit']
        
        print(f"Performance Summary:")
        print(f"  Strategy Processing: {strategy_time:.3f}s (limit: {PERFORMANCE_BENCHMARKS['strategy_processing_max_time']:.3f}s)")
        print(f"  CVD Throughput: {cvd_throughput:.0f} points/sec (min: {PERFORMANCE_BENCHMARKS['cvd_calculation_throughput']})")
        print(f"  Memory Growth: {memory_growth / 1_000_000:.1f}MB (limit: {PERFORMANCE_BENCHMARKS['memory_growth_limit'] / 1_000_000:.0f}MB)")
        print(f"  All benchmarks: PASSED")