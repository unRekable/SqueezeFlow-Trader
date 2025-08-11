"""
CVD Calculation Accuracy Tests
Test CVD calculation against industry standards and verify consistency
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.processors.cvd_calculator import CVDCalculator


class TestCVDCalculationAccuracy:
    """Test CVD calculation accuracy and industry standard compliance"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cvd_calculator = CVDCalculator()
    
    def test_cvd_cumulative_calculation_basic(self):
        """Test CVD follows industry standard: cumsum(vbuy - vsell)"""
        # Arrange: Sample volume data
        vbuy = [1000, 1200, 800, 1500, 900]
        vsell = [800, 1000, 1200, 700, 1100]
        
        # Act: Calculate CVD using system
        system_cvd = self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
        
        # Expected: Manual calculation
        volume_delta = [v_b - v_s for v_b, v_s in zip(vbuy, vsell)]
        expected_cvd = []
        running_total = 0
        for delta in volume_delta:
            running_total += delta
            expected_cvd.append(running_total)
        
        # Assert: System matches expected exactly
        assert len(system_cvd) == len(expected_cvd), f"Length mismatch: {len(system_cvd)} vs {len(expected_cvd)}"
        
        for i, (s, e) in enumerate(zip(system_cvd, expected_cvd)):
            assert abs(s - e) < 0.001, f"CVD mismatch at index {i}: {s} vs {e}"
    
    def test_cvd_mathematical_properties(self):
        """Test CVD mathematical properties"""
        # Generate test data
        np.random.seed(42)  # Reproducible
        vbuy = np.random.uniform(1000, 3000, 100)
        vsell = np.random.uniform(1000, 3000, 100)
        
        cvd = self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
        
        # Property 1: CVD should be monotonic cumulative
        volume_delta = vbuy - vsell
        expected_cumsum = np.cumsum(volume_delta)
        
        assert np.allclose(cvd, expected_cumsum), "CVD not properly cumulative"
        
        # Property 2: CVD difference should equal volume delta
        cvd_diff = np.diff(cvd)
        volume_delta_subset = volume_delta[1:]  # Skip first element
        
        assert np.allclose(cvd_diff, volume_delta_subset), "CVD differences don't match volume deltas"
        
        # Property 3: Final CVD should equal sum of all volume deltas
        total_delta = np.sum(volume_delta)
        assert abs(cvd[-1] - total_delta) < 0.001, f"Final CVD {cvd[-1]} != total delta {total_delta}"
    
    def test_cvd_with_real_market_patterns(self):
        """Test CVD calculation with realistic market data patterns"""
        # Create realistic market scenario: accumulation then distribution
        timestamps = pd.date_range(start='2024-08-01', periods=200, freq='5min')
        
        # Phase 1: Accumulation (higher buy volume)
        vbuy_phase1 = np.random.uniform(1200, 2000, 100)
        vsell_phase1 = np.random.uniform(800, 1400, 100)
        
        # Phase 2: Distribution (higher sell volume) 
        vbuy_phase2 = np.random.uniform(800, 1400, 100)
        vsell_phase2 = np.random.uniform(1200, 2000, 100)
        
        # Combine phases
        vbuy = np.concatenate([vbuy_phase1, vbuy_phase2])
        vsell = np.concatenate([vsell_phase1, vsell_phase2])
        
        # Calculate CVD
        cvd = self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
        
        # Validate phases
        cvd_phase1 = cvd[:100]
        cvd_phase2 = cvd[100:]
        
        # Phase 1 should generally trend upward
        phase1_trend = cvd_phase1[-1] - cvd_phase1[0]
        assert phase1_trend > 0, f"Accumulation phase should trend up, got {phase1_trend}"
        
        # Phase 2 should generally trend downward (from phase 1 peak)
        phase2_trend = cvd_phase2[-1] - cvd_phase2[0]
        assert phase2_trend < 0, f"Distribution phase should trend down, got {phase2_trend}"
        
        # No NaN or infinite values
        assert not np.any(pd.isna(cvd)), "CVD contains NaN values"
        assert not np.any(np.isinf(cvd)), "CVD contains infinite values"
    
    def test_cvd_edge_cases(self):
        """Test CVD calculation edge cases"""
        
        # Edge case 1: All zero volumes
        vbuy_zero = [0, 0, 0, 0, 0]
        vsell_zero = [0, 0, 0, 0, 0]
        
        cvd_zero = self.cvd_calculator.calculate_cvd_from_lists(vbuy_zero, vsell_zero)
        expected_zero = [0, 0, 0, 0, 0]
        
        assert cvd_zero == expected_zero, f"Zero volumes should give zero CVD: {cvd_zero}"
        
        # Edge case 2: Single data point
        cvd_single = self.cvd_calculator.calculate_cvd_from_lists([1000], [800])
        assert cvd_single == [200], f"Single point CVD incorrect: {cvd_single}"
        
        # Edge case 3: Large numbers
        large_vbuy = [1e9, 2e9, 1.5e9]
        large_vsell = [0.8e9, 1.8e9, 1.2e9]
        
        cvd_large = self.cvd_calculator.calculate_cvd_from_lists(large_vbuy, large_vsell)
        
        # Manually calculate expected
        deltas = [1e9 - 0.8e9, 2e9 - 1.8e9, 1.5e9 - 1.2e9]  # [0.2e9, 0.2e9, 0.3e9]
        expected_large = [0.2e9, 0.4e9, 0.7e9]
        
        for i, (c, e) in enumerate(zip(cvd_large, expected_large)):
            assert abs(c - e) < 1e6, f"Large number CVD incorrect at {i}: {c} vs {e}"  # 1M tolerance
    
    def test_cvd_dataframe_integration(self):
        """Test CVD calculation with pandas DataFrame (common use case)"""
        
        # Create DataFrame similar to real market data
        df = pd.DataFrame({
            'time': pd.date_range(start='2024-08-01', periods=50, freq='5min'),
            'total_vbuy_spot': np.random.uniform(1000, 3000, 50),
            'total_vsell_spot': np.random.uniform(1000, 3000, 50),
            'total_vbuy_perp': np.random.uniform(800, 2500, 50),
            'total_vsell_perp': np.random.uniform(800, 2500, 50)
        })
        
        # Calculate CVD for both spot and perp
        spot_cvd = self.cvd_calculator.calculate_cvd_from_series(
            df['total_vbuy_spot'], 
            df['total_vsell_spot']
        )
        
        perp_cvd = self.cvd_calculator.calculate_cvd_from_series(
            df['total_vbuy_perp'], 
            df['total_vsell_perp']
        )
        
        # Validate lengths match
        assert len(spot_cvd) == len(df), "Spot CVD length mismatch"
        assert len(perp_cvd) == len(df), "Perp CVD length mismatch"
        
        # Validate CVD divergence calculation
        cvd_divergence = spot_cvd - perp_cvd
        assert len(cvd_divergence) == len(df), "CVD divergence length mismatch"
        
        # No NaN values in any series
        assert not spot_cvd.isna().any(), "Spot CVD contains NaN"
        assert not perp_cvd.isna().any(), "Perp CVD contains NaN"
        assert not cvd_divergence.isna().any(), "CVD divergence contains NaN"
    
    @pytest.mark.parametrize("data_size", [10, 100, 1000, 5000])
    def test_cvd_scalability(self, data_size):
        """Test CVD calculation performance with different data sizes"""
        import time
        
        # Generate test data
        np.random.seed(42)
        vbuy = np.random.uniform(1000, 3000, data_size)
        vsell = np.random.uniform(1000, 3000, data_size)
        
        start_time = time.time()
        
        cvd = self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance requirements based on data size
        max_time_requirements = {
            10: 0.001,      # 1ms for 10 points
            100: 0.01,      # 10ms for 100 points
            1000: 0.1,      # 100ms for 1000 points
            5000: 0.5       # 500ms for 5000 points
        }
        
        max_time = max_time_requirements[data_size]
        
        assert execution_time < max_time, \
            f"CVD calculation too slow for {data_size} points: {execution_time}s > {max_time}s"
        
        # Validate correctness
        assert len(cvd) == data_size, f"CVD length incorrect: {len(cvd)} vs {data_size}"
        
        # Final CVD should equal sum of deltas
        total_delta = np.sum(vbuy - vsell)
        assert abs(cvd[-1] - total_delta) < 0.001, "CVD calculation incorrect at scale"
    
    def test_cvd_consistency_with_strategy_runner(self):
        """Test CVD calculation matches what strategy runner produces"""
        # This test ensures consistency between backtest and live calculation
        
        # Sample data similar to what strategy runner receives
        sample_data = {
            'spot_markets': {
                'BINANCE:btcusdt': {'vbuy': [1500, 1800, 1200], 'vsell': [1200, 1400, 1600]},
                'COINBASE:BTC-USD': {'vbuy': [800, 900, 700], 'vsell': [600, 800, 900]}
            },
            'perp_markets': {
                'BINANCE_FUTURES:btcusdt': {'vbuy': [2000, 2200, 1800], 'vsell': [1800, 2000, 2100]},
                'BYBIT:BTCUSDT': {'vbuy': [1000, 1100, 900], 'vsell': [900, 1000, 1100]}
            }
        }
        
        # Calculate total volumes (as strategy runner does)
        total_vbuy_spot = []
        total_vsell_spot = []
        total_vbuy_perp = []
        total_vsell_perp = []
        
        for i in range(3):  # 3 time periods
            spot_vbuy = sum(market['vbuy'][i] for market in sample_data['spot_markets'].values())
            spot_vsell = sum(market['vsell'][i] for market in sample_data['spot_markets'].values())
            perp_vbuy = sum(market['vbuy'][i] for market in sample_data['perp_markets'].values())
            perp_vsell = sum(market['vsell'][i] for market in sample_data['perp_markets'].values())
            
            total_vbuy_spot.append(spot_vbuy)
            total_vsell_spot.append(spot_vsell)
            total_vbuy_perp.append(perp_vbuy)
            total_vsell_perp.append(perp_vsell)
        
        # Calculate CVD using our calculator
        spot_cvd = self.cvd_calculator.calculate_cvd_from_lists(total_vbuy_spot, total_vsell_spot)
        perp_cvd = self.cvd_calculator.calculate_cvd_from_lists(total_vbuy_perp, total_vsell_perp)
        
        # Expected values (manual calculation)
        # Spot: [2300-1800, 2700-2200, 1900-2500] = [500, 500, -600]
        # Cumulative: [500, 1000, 400]
        expected_spot_cvd = [500, 1000, 400]
        
        # Perp: [3000-2700, 3300-3000, 2700-3200] = [300, 300, -500] 
        # Cumulative: [300, 600, 100]
        expected_perp_cvd = [300, 600, 100]
        
        assert spot_cvd == expected_spot_cvd, f"Spot CVD mismatch: {spot_cvd} vs {expected_spot_cvd}"
        assert perp_cvd == expected_perp_cvd, f"Perp CVD mismatch: {perp_cvd} vs {expected_perp_cvd}"
        
        # CVD divergence
        cvd_divergence = [s - p for s, p in zip(spot_cvd, perp_cvd)]
        expected_divergence = [200, 400, 300]  # [500-300, 1000-600, 400-100]
        
        assert cvd_divergence == expected_divergence, \
            f"CVD divergence mismatch: {cvd_divergence} vs {expected_divergence}"


class TestCVDCalculatorEdgeCases:
    """Test CVD calculator edge cases and error handling"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cvd_calculator = CVDCalculator()
    
    def test_empty_data_handling(self):
        """Test CVD calculation with empty data"""
        
        # Empty lists
        cvd_empty = self.cvd_calculator.calculate_cvd_from_lists([], [])
        assert cvd_empty == [], "Empty data should return empty CVD"
        
    def test_mismatched_length_handling(self):
        """Test CVD calculation with mismatched lengths"""
        
        vbuy = [1000, 1200, 800]
        vsell = [800, 1000]  # Shorter
        
        # Should handle gracefully (truncate to shorter length)
        with pytest.raises(ValueError):
            self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
    
    def test_negative_volume_handling(self):
        """Test CVD calculation with negative volumes"""
        
        # Negative volumes (shouldn't happen in real data but test robustness)
        vbuy = [1000, -500, 800]  # Negative buy volume
        vsell = [800, 1000, 600]
        
        cvd = self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
        
        # Should still calculate correctly
        expected_deltas = [200, -1500, 200]  # [1000-800, -500-1000, 800-600]
        expected_cvd = [200, -1300, -1100]   # Cumulative
        
        assert cvd == expected_cvd, f"Negative volume CVD incorrect: {cvd} vs {expected_cvd}"
    
    def test_floating_point_precision(self):
        """Test CVD calculation with floating point precision issues"""
        
        # Use floating point numbers that might cause precision issues
        vbuy = [1000.1, 1200.7, 800.3]
        vsell = [800.2, 1000.9, 600.1]
        
        cvd = self.cvd_calculator.calculate_cvd_from_lists(vbuy, vsell)
        
        # Expected (with floating point precision)
        expected_deltas = [199.9, 199.8, 200.2]
        expected_cvd = [199.9, 399.7, 599.9]
        
        for i, (c, e) in enumerate(zip(cvd, expected_cvd)):
            assert abs(c - e) < 0.1, f"Floating point CVD incorrect at {i}: {c} vs {e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])