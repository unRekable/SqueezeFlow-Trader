"""
Test Data Pipeline Components
Comprehensive tests for data loading, processing, and CVD calculation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import data pipeline components
from data.pipeline import DataPipeline, create_data_pipeline
from data.loaders.symbol_discovery import SymbolDiscovery
from data.loaders.market_discovery import MarketDiscovery
from data.processors.cvd_calculator import CVDCalculator
from data.processors.exchange_mapper import ExchangeMapper


class TestDataPipeline:
    """Test the main data pipeline orchestrator"""
    
    def test_data_pipeline_initialization(self):
        """Test data pipeline initializes correctly"""
        pipeline = DataPipeline()
        
        assert pipeline.influx_client is not None
        assert pipeline.symbol_discovery is not None
        assert pipeline.market_discovery is not None
        assert pipeline.exchange_mapper is not None
        assert pipeline.cvd_calculator is not None
        
    def test_create_data_pipeline_factory(self):
        """Test factory function creates pipeline"""
        pipeline = create_data_pipeline()
        
        assert isinstance(pipeline, DataPipeline)
        
    @pytest.mark.unit
    def test_discover_available_symbols_caching(self):
        """Test symbol discovery with caching"""
        pipeline = DataPipeline()
        
        # Mock the symbol discovery
        with patch.object(pipeline.symbol_discovery, 'discover_symbols_from_database') as mock_discover:
            mock_discover.return_value = ['BTC', 'ETH', 'ADA']
            
            # First call
            symbols1 = pipeline.discover_available_symbols(min_data_points=500, hours_lookback=24)
            # Second call - should use cache
            symbols2 = pipeline.discover_available_symbols(min_data_points=500, hours_lookback=24)
            
            assert symbols1 == symbols2
            assert symbols1 == ['BTC', 'ETH', 'ADA']
            # Should only call the underlying method once due to caching
            assert mock_discover.call_count == 1
            
    @pytest.mark.unit
    def test_discover_markets_for_symbol_caching(self):
        """Test market discovery with caching"""
        pipeline = DataPipeline()
        
        with patch.object(pipeline.market_discovery, 'get_markets_by_type') as mock_markets:
            mock_markets.return_value = {
                'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD'],
                'perp': ['BINANCE_FUTURES:btcusdt']
            }
            
            # First call
            markets1 = pipeline.discover_markets_for_symbol('BTC')
            # Second call - should use cache
            markets2 = pipeline.discover_markets_for_symbol('BTC')
            
            assert markets1 == markets2
            assert len(markets1['spot']) == 2
            assert len(markets1['perp']) == 1
            # Should only call once due to caching
            assert mock_markets.call_count == 1
            
    @pytest.mark.unit
    def test_load_raw_ohlcv_data(self):
        """Test OHLCV data loading"""
        pipeline = DataPipeline()
        
        # Mock the data loading components
        with patch.object(pipeline, 'discover_markets_for_symbol') as mock_markets:
            mock_markets.return_value = {
                'spot': ['BINANCE:btcusdt'],
                'perp': ['BINANCE_FUTURES:btcusdt']
            }
            
            with patch.object(pipeline.influx_client, 'get_ohlcv_data') as mock_ohlcv:
                # Create sample OHLCV data
                sample_data = pd.DataFrame({
                    'time': pd.date_range('2024-08-01', periods=10, freq='5min'),
                    'open': np.random.uniform(49000, 51000, 10),
                    'high': np.random.uniform(50000, 52000, 10),
                    'low': np.random.uniform(48000, 50000, 10),
                    'close': np.random.uniform(49000, 51000, 10),
                    'volume': np.random.uniform(1000, 5000, 10)
                })
                mock_ohlcv.return_value = sample_data
                
                result = pipeline.load_raw_ohlcv_data(
                    symbol='BTC',
                    start_time=datetime(2024, 8, 1),
                    end_time=datetime(2024, 8, 2),
                    timeframe='5m'
                )
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 10
                assert 'open' in result.columns
                assert 'high' in result.columns
                assert 'low' in result.columns
                assert 'close' in result.columns
                
    @pytest.mark.unit
    def test_load_raw_volume_data(self):
        """Test volume data loading with market separation"""
        pipeline = DataPipeline()
        
        with patch.object(pipeline, 'discover_markets_for_symbol') as mock_markets:
            mock_markets.return_value = {
                'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD'],
                'perp': ['BINANCE_FUTURES:btcusdt']
            }
            
            with patch.object(pipeline.influx_client, 'get_volume_data') as mock_volume:
                # Create sample volume data
                sample_volume = pd.DataFrame({
                    'time': pd.date_range('2024-08-01', periods=10, freq='5min'),
                    'total_volume': np.random.uniform(1000, 5000, 10),
                    'total_vbuy': np.random.uniform(500, 3000, 10),
                    'total_vsell': np.random.uniform(500, 2000, 10)
                })
                mock_volume.return_value = sample_volume
                
                spot_df, futures_df = pipeline.load_raw_volume_data(
                    symbol='BTC',
                    start_time=datetime(2024, 8, 1),
                    end_time=datetime(2024, 8, 2),
                    timeframe='5m'
                )
                
                # Should return two dataframes
                assert isinstance(spot_df, pd.DataFrame)
                assert isinstance(futures_df, pd.DataFrame)
                
                # Mock should be called twice (once for spot, once for futures)
                assert mock_volume.call_count == 2
                
    @pytest.mark.unit
    def test_calculate_cvd_data(self, sample_volume_data, sample_cvd_data):
        """Test CVD calculation from volume data"""
        pipeline = DataPipeline()
        
        # Prepare volume data with proper columns
        spot_df = sample_volume_data.copy()
        futures_df = sample_volume_data.copy()
        
        # Mock CVD calculator
        with patch.object(pipeline.cvd_calculator, 'calculate_spot_cvd') as mock_spot_cvd:
            with patch.object(pipeline.cvd_calculator, 'calculate_futures_cvd') as mock_futures_cvd:
                with patch.object(pipeline.cvd_calculator, 'calculate_cvd_divergence') as mock_divergence:
                    
                    mock_spot_cvd.return_value = sample_cvd_data['spot_cvd']
                    mock_futures_cvd.return_value = sample_cvd_data['futures_cvd']
                    mock_divergence.return_value = sample_cvd_data['cvd_divergence']
                    
                    result = pipeline.calculate_cvd_data(spot_df, futures_df)
                    
                    assert 'spot_cvd' in result
                    assert 'futures_cvd' in result
                    assert 'cvd_divergence' in result
                    
                    assert isinstance(result['spot_cvd'], pd.Series)
                    assert isinstance(result['futures_cvd'], pd.Series)
                    assert isinstance(result['cvd_divergence'], pd.Series)
                    
                    # All methods should be called
                    mock_spot_cvd.assert_called_once()
                    mock_futures_cvd.assert_called_once()
                    mock_divergence.assert_called_once()
                    
    @pytest.mark.integration
    def test_get_complete_dataset(self, sample_ohlcv_data, sample_volume_data, sample_cvd_data):
        """Test complete dataset assembly"""
        pipeline = DataPipeline()
        
        with patch.object(pipeline, 'load_raw_ohlcv_data') as mock_ohlcv:
            with patch.object(pipeline, 'load_raw_volume_data') as mock_volume:
                with patch.object(pipeline, 'calculate_cvd_data') as mock_cvd:
                    with patch.object(pipeline, 'discover_markets_for_symbol') as mock_markets:
                        
                        # Setup mocks
                        mock_ohlcv.return_value = sample_ohlcv_data
                        mock_volume.return_value = (sample_volume_data, sample_volume_data)
                        mock_cvd.return_value = sample_cvd_data
                        mock_markets.return_value = {
                            'spot': ['BINANCE:btcusdt'],
                            'perp': ['BINANCE_FUTURES:btcusdt']
                        }
                        
                        result = pipeline.get_complete_dataset(
                            symbol='BTC',
                            start_time=datetime(2024, 8, 1),
                            end_time=datetime(2024, 8, 2),
                            timeframe='5m'
                        )
                        
                        # Verify complete dataset structure
                        expected_keys = [
                            'symbol', 'timeframe', 'start_time', 'end_time',
                            'ohlcv', 'spot_volume', 'futures_volume',
                            'spot_cvd', 'futures_cvd', 'cvd_divergence',
                            'markets', 'metadata'
                        ]
                        
                        for key in expected_keys:
                            assert key in result, f"Missing key: {key}"
                        
                        assert result['symbol'] == 'BTC'
                        assert result['timeframe'] == '5m'
                        assert result['metadata']['spot_markets_count'] == 1
                        assert result['metadata']['futures_markets_count'] == 1
                        
    @pytest.mark.unit
    def test_validate_data_quality_good_data(self, sample_dataset):
        """Test data quality validation with good data"""
        pipeline = DataPipeline()
        
        result = pipeline.validate_data_quality(sample_dataset)
        
        assert isinstance(result, dict)
        assert 'overall_quality' in result
        
        # Should pass all validation checks for sample data
        expected_checks = [
            'has_price_data', 'has_spot_volume', 'has_futures_volume',
            'has_spot_cvd', 'has_futures_cvd', 'has_cvd_divergence',
            'sufficient_data_points'
        ]
        
        for check in expected_checks:
            assert check in result
            
    @pytest.mark.unit
    def test_validate_data_quality_bad_data(self):
        """Test data quality validation with bad data"""
        pipeline = DataPipeline()
        
        # Create dataset with missing/empty data
        bad_dataset = {
            'ohlcv': pd.DataFrame(),  # Empty
            'spot_volume': pd.DataFrame(),  # Empty
            'futures_volume': pd.DataFrame(),  # Empty
            'spot_cvd': pd.Series(dtype=float),  # Empty
            'futures_cvd': pd.Series(dtype=float),  # Empty
            'cvd_divergence': pd.Series(dtype=float),  # Empty
            'metadata': {
                'data_points': 0,
                'spot_markets_count': 0,
                'futures_markets_count': 0
            }
        }
        
        result = pipeline.validate_data_quality(bad_dataset)
        
        assert result['overall_quality'] == False
        assert result['has_price_data'] == False
        assert result['sufficient_data_points'] == False
        
    def test_clear_cache(self):
        """Test cache clearing functionality"""
        pipeline = DataPipeline()
        
        # Add something to cache
        pipeline._symbol_cache['test'] = ['BTC']
        pipeline._market_cache['BTC'] = {'spot': [], 'perp': []}
        
        # Clear cache
        pipeline.clear_cache()
        
        assert len(pipeline._symbol_cache) == 0
        assert len(pipeline._market_cache) == 0


class TestSymbolDiscovery:
    """Test symbol discovery functionality"""
    
    def test_symbol_discovery_initialization(self):
        """Test SymbolDiscovery initializes correctly"""
        discovery = SymbolDiscovery()
        assert discovery is not None
        
    @pytest.mark.unit
    def test_discover_symbols_from_database_mock(self):
        """Test symbol discovery with mocked database"""
        discovery = SymbolDiscovery()
        
        # Mock the database query
        with patch.object(discovery, '_query_available_symbols') as mock_query:
            mock_query.return_value = ['BTC', 'ETH', 'ADA', 'DOT']
            
            symbols = discovery.discover_symbols_from_database(
                min_data_points=500,
                hours_lookback=24
            )
            
            assert isinstance(symbols, list)
            assert len(symbols) == 4
            assert 'BTC' in symbols
            assert 'ETH' in symbols
            
            # Verify method was called with correct parameters
            mock_query.assert_called_once_with(500, 24)


class TestMarketDiscovery:
    """Test market discovery functionality"""
    
    def test_market_discovery_initialization(self):
        """Test MarketDiscovery initializes correctly"""
        discovery = MarketDiscovery()
        assert discovery is not None
        
    @pytest.mark.unit
    def test_get_markets_by_type_mock(self):
        """Test market discovery with mocked database"""
        discovery = MarketDiscovery()
        
        with patch.object(discovery, '_query_markets_for_symbol') as mock_query:
            mock_query.return_value = {
                'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD', 'KRAKEN:BTCUSD'],
                'perp': ['BINANCE_FUTURES:btcusdt', 'BYBIT:BTCUSDT']
            }
            
            markets = discovery.get_markets_by_type('BTC')
            
            assert isinstance(markets, dict)
            assert 'spot' in markets
            assert 'perp' in markets
            assert len(markets['spot']) == 3
            assert len(markets['perp']) == 2
            
            mock_query.assert_called_once_with('BTC')


class TestCVDCalculator:
    """Test CVD calculation functionality"""
    
    def test_cvd_calculator_initialization(self):
        """Test CVDCalculator initializes correctly"""
        calculator = CVDCalculator()
        assert calculator is not None
        
    @pytest.mark.unit
    def test_calculate_spot_cvd_basic(self, sample_volume_data):
        """Test basic spot CVD calculation"""
        calculator = CVDCalculator()
        
        # Ensure volume data has required columns
        test_data = sample_volume_data.copy()
        if 'total_vbuy' not in test_data.columns:
            test_data['total_vbuy'] = test_data['total_volume'] * 0.6
            test_data['total_vsell'] = test_data['total_volume'] * 0.4
        
        result = calculator.calculate_spot_cvd(test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(test_data)
        
        # CVD should be cumulative
        volume_delta = test_data['total_vbuy'] - test_data['total_vsell']
        expected_cvd = volume_delta.cumsum()
        
        # Should be close to expected (allowing for small floating point differences)
        pd.testing.assert_series_equal(result, expected_cvd, check_names=False)
        
    @pytest.mark.unit
    def test_calculate_futures_cvd_basic(self, sample_volume_data):
        """Test basic futures CVD calculation"""
        calculator = CVDCalculator()
        
        test_data = sample_volume_data.copy()
        if 'total_vbuy' not in test_data.columns:
            test_data['total_vbuy'] = test_data['total_volume'] * 0.6
            test_data['total_vsell'] = test_data['total_volume'] * 0.4
            
        result = calculator.calculate_futures_cvd(test_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(test_data)
        
    @pytest.mark.unit
    def test_calculate_cvd_divergence(self, sample_cvd_data):
        """Test CVD divergence calculation"""
        calculator = CVDCalculator()
        
        spot_cvd = sample_cvd_data['spot_cvd']
        futures_cvd = sample_cvd_data['futures_cvd']
        
        result = calculator.calculate_cvd_divergence(spot_cvd, futures_cvd)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(spot_cvd)
        
        # Divergence should be the difference
        expected_divergence = spot_cvd - futures_cvd
        pd.testing.assert_series_equal(result, expected_divergence, check_names=False)
        
    @pytest.mark.unit
    def test_cvd_calculation_properties(self):
        """Test CVD calculation mathematical properties"""
        calculator = CVDCalculator()
        
        # Create test data with known values
        test_data = pd.DataFrame({
            'total_vbuy': [1000, 1500, 800, 2000, 1200],
            'total_vsell': [800, 1000, 1200, 1500, 1800]
        })
        
        cvd = calculator.calculate_spot_cvd(test_data)
        
        # Test CVD properties
        assert len(cvd) == len(test_data)
        
        # CVD should be monotonic when buy > sell consistently
        # or decrease when sell > buy
        volume_deltas = test_data['total_vbuy'] - test_data['total_vsell']
        expected_cvd = volume_deltas.cumsum()
        
        pd.testing.assert_series_equal(cvd, expected_cvd, check_names=False)
        
    @pytest.mark.unit
    def test_cvd_with_empty_data(self):
        """Test CVD calculation with empty data"""
        calculator = CVDCalculator()
        
        empty_data = pd.DataFrame()
        result = calculator.calculate_spot_cvd(empty_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 0
        
    @pytest.mark.unit 
    def test_cvd_with_missing_columns(self):
        """Test CVD calculation with missing required columns"""
        calculator = CVDCalculator()
        
        # Data missing required columns
        bad_data = pd.DataFrame({
            'volume': [1000, 1500, 800],
            'price': [50000, 50100, 49900]
        })
        
        # Should handle gracefully (return empty series or raise appropriate error)
        try:
            result = calculator.calculate_spot_cvd(bad_data)
            # If it returns something, should be empty or proper error handling
            assert isinstance(result, pd.Series)
        except (KeyError, ValueError):
            # Acceptable to raise error for missing required columns
            pass


class TestExchangeMapper:
    """Test exchange mapping functionality"""
    
    def test_exchange_mapper_initialization(self):
        """Test ExchangeMapper initializes correctly"""
        mapper = ExchangeMapper()
        assert mapper is not None
        
    @pytest.mark.unit
    def test_classify_market_type(self):
        """Test market type classification"""
        mapper = ExchangeMapper()
        
        # Test spot markets
        spot_markets = [
            'BINANCE:btcusdt',
            'COINBASE:BTC-USD',
            'KRAKEN:BTCUSD'
        ]
        
        for market in spot_markets:
            result = mapper.classify_market_type(market)
            assert result == 'spot'
            
        # Test futures/perp markets
        futures_markets = [
            'BINANCE_FUTURES:btcusdt',
            'BYBIT:BTCUSDT',
            'OKEX:BTC-USDT-SWAP'
        ]
        
        for market in futures_markets:
            result = mapper.classify_market_type(market)
            assert result == 'perp'
            
    @pytest.mark.unit
    def test_get_base_symbol(self):
        """Test base symbol extraction"""
        mapper = ExchangeMapper()
        
        test_cases = [
            ('BINANCE:btcusdt', 'BTC'),
            ('COINBASE:BTC-USD', 'BTC'),
            ('BINANCE:ethusdt', 'ETH'),
            ('KRAKEN:ETHUSD', 'ETH')
        ]
        
        for market, expected_base in test_cases:
            result = mapper.get_base_symbol(market)
            assert result == expected_base


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline workflow"""
    
    @pytest.mark.integration
    def test_complete_data_workflow(self, mock_influxdb):
        """Test complete data loading and processing workflow"""
        pipeline = DataPipeline()
        
        # Mock all the database calls
        with patch.object(pipeline.symbol_discovery, 'discover_symbols_from_database') as mock_symbols:
            with patch.object(pipeline.market_discovery, 'get_markets_by_type') as mock_markets:
                with patch.object(pipeline.influx_client, 'get_ohlcv_data') as mock_ohlcv:
                    with patch.object(pipeline.influx_client, 'get_volume_data') as mock_volume:
                        
                        # Setup mocks
                        mock_symbols.return_value = ['BTC', 'ETH']
                        mock_markets.return_value = {
                            'spot': ['BINANCE:btcusdt'],
                            'perp': ['BINANCE_FUTURES:btcusdt']
                        }
                        
                        # Create realistic mock data
                        mock_ohlcv_data = pd.DataFrame({
                            'time': pd.date_range('2024-08-01', periods=20, freq='5min'),
                            'open': np.random.uniform(49000, 51000, 20),
                            'high': np.random.uniform(50000, 52000, 20),
                            'low': np.random.uniform(48000, 50000, 20),
                            'close': np.random.uniform(49000, 51000, 20),
                            'volume': np.random.uniform(1000, 5000, 20)
                        })
                        
                        mock_volume_data = pd.DataFrame({
                            'time': pd.date_range('2024-08-01', periods=20, freq='5min'),
                            'total_volume': np.random.uniform(1000, 5000, 20),
                            'total_vbuy': np.random.uniform(500, 3000, 20),
                            'total_vsell': np.random.uniform(300, 2500, 20)
                        })
                        
                        mock_ohlcv.return_value = mock_ohlcv_data
                        mock_volume.return_value = mock_volume_data
                        
                        # Test complete workflow
                        symbols = pipeline.discover_available_symbols()
                        assert len(symbols) == 2
                        
                        markets = pipeline.discover_markets_for_symbol('BTC')
                        assert 'spot' in markets
                        assert 'perp' in markets
                        
                        dataset = pipeline.get_complete_dataset(
                            symbol='BTC',
                            start_time=datetime(2024, 8, 1),
                            end_time=datetime(2024, 8, 2),
                            timeframe='5m'
                        )
                        
                        # Validate complete dataset
                        quality = pipeline.validate_data_quality(dataset)
                        
                        # Should have all required components
                        assert 'ohlcv' in dataset
                        assert 'spot_cvd' in dataset
                        assert 'futures_cvd' in dataset
                        assert 'cvd_divergence' in dataset
                        
    @pytest.mark.performance
    def test_data_pipeline_performance(self):
        """Test data pipeline performance benchmarks"""
        import time
        
        pipeline = DataPipeline()
        
        # Mock fast responses
        with patch.object(pipeline.symbol_discovery, 'discover_symbols_from_database') as mock_symbols:
            mock_symbols.return_value = ['BTC']
            
            start_time = time.time()
            symbols = pipeline.discover_available_symbols()
            end_time = time.time()
            
            # Symbol discovery should be fast
            processing_time = end_time - start_time
            assert processing_time < 0.1, f"Symbol discovery too slow: {processing_time:.3f}s"
            
        # Test caching performance
        start_time = time.time()
        symbols_cached = pipeline.discover_available_symbols()  # Should use cache
        end_time = time.time()
        
        cached_time = end_time - start_time
        assert cached_time < 0.01, f"Cached lookup too slow: {cached_time:.3f}s"