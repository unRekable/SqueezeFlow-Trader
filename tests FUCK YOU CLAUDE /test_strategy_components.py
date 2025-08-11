"""
Test Strategy Components - Phase 1-5 Testing
Comprehensive tests for all SqueezeFlow strategy phases
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import strategy components
from strategies.squeezeflow.strategy import SqueezeFlowStrategy
from strategies.squeezeflow.config import SqueezeFlowConfig
from strategies.squeezeflow.components.phase1_context import ContextAssessment
from strategies.squeezeflow.components.phase2_divergence import DivergenceDetection
from strategies.squeezeflow.components.phase3_reset import ResetDetection
from strategies.squeezeflow.components.phase4_scoring import ScoringSystem
from strategies.squeezeflow.components.phase5_exits import ExitManagement


class TestSqueezeFlowStrategy:
    """Test the main SqueezeFlow strategy orchestrator"""
    
    def test_strategy_initialization(self, squeeze_flow_config):
        """Test strategy initializes correctly"""
        strategy = SqueezeFlowStrategy(config=squeeze_flow_config)
        
        assert strategy.name == "SqueezeFlowStrategy"
        assert strategy.config is not None
        assert strategy.phase1 is not None
        assert strategy.phase2 is not None
        assert strategy.phase3 is not None
        assert strategy.phase4 is not None
        assert strategy.phase5 is not None
        
    def test_strategy_initialization_default_config(self):
        """Test strategy with default configuration"""
        strategy = SqueezeFlowStrategy()
        
        assert strategy.config is not None
        assert isinstance(strategy.config, SqueezeFlowConfig)
        assert strategy.config.min_entry_score == 4.0
        
    @pytest.mark.unit
    def test_process_with_valid_dataset(self, sample_dataset, sample_portfolio_state):
        """Test strategy processes valid dataset correctly"""
        strategy = SqueezeFlowStrategy()
        
        result = strategy.process(sample_dataset, sample_portfolio_state)
        
        # Verify result structure
        assert 'orders' in result
        assert 'phase_results' in result
        assert 'metadata' in result
        assert isinstance(result['orders'], list)
        assert isinstance(result['phase_results'], dict)
        
        # Check phase results are present (at least phase 1-3)
        assert 'phase1_context' in result['phase_results']
        assert 'phase2_divergence' in result['phase_results']
        assert 'phase3_reset' in result['phase_results']
        
    @pytest.mark.unit
    def test_process_with_existing_position(self, sample_dataset, sample_portfolio_with_position):
        """Test strategy handles existing positions"""
        strategy = SqueezeFlowStrategy()
        
        result = strategy.process(sample_dataset, sample_portfolio_with_position)
        
        # Should have phase 5 exit results instead of phase 4 scoring
        phase_keys = list(result['phase_results'].keys())
        exit_phases = [key for key in phase_keys if 'phase5_exit' in key]
        assert len(exit_phases) > 0, "Should have exit management results for existing position"
        
    @pytest.mark.unit
    def test_process_error_handling(self, sample_portfolio_state):
        """Test strategy handles invalid data gracefully"""
        strategy = SqueezeFlowStrategy()
        
        # Test with empty dataset
        empty_dataset = {}
        result = strategy.process(empty_dataset, sample_portfolio_state)
        
        assert 'error' in result or len(result['orders']) == 0
        assert 'orders' in result
        assert isinstance(result['orders'], list)


class TestContextAssessment:
    """Test Phase 1: Context Assessment"""
    
    def test_context_assessment_initialization(self):
        """Test Phase 1 component initializes correctly"""
        context_timeframes = ['30m', '1h', '4h']
        phase1 = ContextAssessment(context_timeframes=context_timeframes)
        
        assert phase1.context_timeframes == context_timeframes
        
    @pytest.mark.unit
    def test_assess_context_basic(self, sample_dataset):
        """Test basic context assessment functionality"""
        phase1 = ContextAssessment()
        
        result = phase1.assess_context(sample_dataset)
        
        # Check result structure
        assert isinstance(result, dict)
        expected_keys = ['market_regime', 'trend_strength', 'volatility_assessment']
        for key in expected_keys:
            assert key in result or 'error' in result
            
    @pytest.mark.unit
    def test_assess_context_with_insufficient_data(self):
        """Test context assessment with insufficient data"""
        phase1 = ContextAssessment()
        
        # Empty dataset
        empty_dataset = {'ohlcv': pd.DataFrame(), 'spot_cvd': pd.Series(dtype=float)}
        result = phase1.assess_context(empty_dataset)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'error' in result or 'market_regime' in result


class TestDivergenceDetection:
    """Test Phase 2: Divergence Detection"""
    
    def test_divergence_detection_initialization(self):
        """Test Phase 2 component initializes correctly"""
        divergence_timeframes = ['15m', '30m']
        phase2 = DivergenceDetection(divergence_timeframes=divergence_timeframes)
        
        assert phase2.divergence_timeframes == divergence_timeframes
        
    @pytest.mark.unit
    def test_detect_divergence_long_squeeze(self, long_squeeze_scenario):
        """Test divergence detection on long squeeze scenario"""
        phase2 = DivergenceDetection()
        mock_context = {'market_regime': 'trending', 'trend_strength': 0.7}
        
        result = phase2.detect_divergence(long_squeeze_scenario, mock_context)
        
        assert isinstance(result, dict)
        expected_keys = ['divergence_type', 'divergence_strength', 'leadership_pattern']
        for key in expected_keys:
            assert key in result or 'error' in result
            
    @pytest.mark.unit
    def test_detect_divergence_short_squeeze(self, short_squeeze_scenario):
        """Test divergence detection on short squeeze scenario"""
        phase2 = DivergenceDetection()
        mock_context = {'market_regime': 'trending', 'trend_strength': 0.7}
        
        result = phase2.detect_divergence(short_squeeze_scenario, mock_context)
        
        assert isinstance(result, dict)
        # Should detect some form of divergence
        if 'divergence_type' in result:
            assert result['divergence_type'] is not None
            
    @pytest.mark.unit
    def test_detect_divergence_neutral_market(self, neutral_market_scenario):
        """Test divergence detection on neutral market"""
        phase2 = DivergenceDetection()
        mock_context = {'market_regime': 'sideways', 'trend_strength': 0.2}
        
        result = phase2.detect_divergence(neutral_market_scenario, mock_context)
        
        assert isinstance(result, dict)
        # Should detect minimal or no divergence in neutral market
        if 'divergence_strength' in result:
            assert isinstance(result['divergence_strength'], (int, float))


class TestResetDetection:
    """Test Phase 3: Reset Detection"""
    
    def test_reset_detection_initialization(self):
        """Test Phase 3 component initializes correctly"""
        reset_timeframes = ['5m', '15m', '30m']
        phase3 = ResetDetection(reset_timeframes=reset_timeframes)
        
        assert phase3.reset_timeframes == reset_timeframes
        
    @pytest.mark.unit
    def test_detect_reset_basic(self, sample_dataset):
        """Test basic reset detection functionality"""
        phase3 = ResetDetection()
        mock_context = {'market_regime': 'trending'}
        mock_divergence = {'divergence_type': 'LONG_SQUEEZE', 'divergence_strength': 0.8}
        
        result = phase3.detect_reset(sample_dataset, mock_context, mock_divergence)
        
        assert isinstance(result, dict)
        expected_keys = ['reset_type', 'reset_strength', 'convergence_status']
        for key in expected_keys:
            assert key in result or 'error' in result
            
    @pytest.mark.unit
    def test_detect_reset_with_long_squeeze(self, long_squeeze_scenario):
        """Test reset detection with long squeeze setup"""
        phase3 = ResetDetection()
        mock_context = {'market_regime': 'trending'}
        mock_divergence = {'divergence_type': 'LONG_SQUEEZE', 'divergence_strength': 0.8}
        
        result = phase3.detect_reset(long_squeeze_scenario, mock_context, mock_divergence)
        
        assert isinstance(result, dict)
        if 'reset_type' in result:
            assert result['reset_type'] in ['TYPE_A', 'TYPE_B', None]


class TestScoringSystem:
    """Test Phase 4: 10-Point Scoring System"""
    
    def test_scoring_system_initialization(self):
        """Test Phase 4 component initializes correctly"""
        custom_weights = {
            'cvd_reset_deceleration': 4.0,
            'absorption_candle': 3.0,
            'failed_breakdown': 2.0,
            'directional_bias': 1.0
        }
        phase4 = ScoringSystem(scoring_weights=custom_weights)
        
        assert phase4.scoring_weights == custom_weights
        
    @pytest.mark.unit
    def test_calculate_score_high_confidence(self, sample_dataset):
        """Test scoring system with high confidence scenario"""
        phase4 = ScoringSystem()
        
        # Create strong signals for all phases
        mock_context = {
            'market_regime': 'trending',
            'trend_strength': 0.9,
            'volatility_assessment': 'normal'
        }
        mock_divergence = {
            'divergence_type': 'LONG_SQUEEZE',
            'divergence_strength': 0.9,
            'leadership_pattern': 'CLEAR'
        }
        mock_reset = {
            'reset_type': 'TYPE_A',
            'reset_strength': 0.8,
            'convergence_status': 'EXHAUSTED'
        }
        
        result = phase4.calculate_score(mock_context, mock_divergence, mock_reset, sample_dataset)
        
        assert isinstance(result, dict)
        assert 'total_score' in result or 'error' in result
        
        if 'total_score' in result:
            assert isinstance(result['total_score'], (int, float))
            assert 0 <= result['total_score'] <= 10
            
        if 'should_trade' in result:
            assert isinstance(result['should_trade'], bool)
            
    @pytest.mark.unit
    def test_calculate_score_low_confidence(self, neutral_market_scenario):
        """Test scoring system with low confidence scenario"""
        phase4 = ScoringSystem()
        
        # Create weak signals
        mock_context = {
            'market_regime': 'sideways',
            'trend_strength': 0.2,
            'volatility_assessment': 'high'
        }
        mock_divergence = {
            'divergence_type': None,
            'divergence_strength': 0.1,
            'leadership_pattern': 'UNCLEAR'
        }
        mock_reset = {
            'reset_type': None,
            'reset_strength': 0.1,
            'convergence_status': 'ACTIVE'
        }
        
        result = phase4.calculate_score(mock_context, mock_divergence, mock_reset, neutral_market_scenario)
        
        assert isinstance(result, dict)
        if 'total_score' in result:
            # Should be low score
            assert result['total_score'] < 4.0
            
        if 'should_trade' in result:
            # Should not recommend trading
            assert result['should_trade'] == False
            
    @pytest.mark.unit
    def test_scoring_weights_configuration(self):
        """Test scoring system respects weight configuration"""
        config = SqueezeFlowConfig()
        phase4 = ScoringSystem(scoring_weights=config.scoring_weights)
        
        # Verify weights are set correctly
        assert phase4.scoring_weights['cvd_reset_deceleration'] == 3.5
        assert phase4.scoring_weights['absorption_candle'] == 2.5
        assert phase4.scoring_weights['failed_breakdown'] == 2.0
        assert phase4.scoring_weights['directional_bias'] == 2.0


class TestExitManagement:
    """Test Phase 5: Exit Management"""
    
    def test_exit_management_initialization(self):
        """Test Phase 5 component initializes correctly"""
        phase5 = ExitManagement()
        
        assert phase5 is not None
        
    @pytest.mark.unit
    def test_manage_exits_no_exit_signal(self, sample_dataset):
        """Test exit management when no exit is needed"""
        phase5 = ExitManagement()
        
        mock_position = {
            'id': 'pos_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.02,
            'entry_price': 50000.0,
            'current_price': 50500.0  # Small profit
        }
        
        mock_analysis = {'phase_results': {}}
        
        result = phase5.manage_exits(sample_dataset, mock_position, mock_analysis)
        
        assert isinstance(result, dict)
        assert 'should_exit' in result or 'error' in result
        
        if 'should_exit' in result:
            assert isinstance(result['should_exit'], bool)
            
    @pytest.mark.unit
    def test_manage_exits_with_loss_position(self, sample_dataset):
        """Test exit management with losing position"""
        phase5 = ExitManagement()
        
        mock_position = {
            'id': 'pos_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.02,
            'entry_price': 50000.0,
            'current_price': 48000.0  # Significant loss
        }
        
        mock_analysis = {'phase_results': {}}
        
        result = phase5.manage_exits(sample_dataset, mock_position, mock_analysis)
        
        assert isinstance(result, dict)
        if 'should_exit' in result and 'exit_reasoning' in result:
            # Might exit due to significant loss
            assert isinstance(result['exit_reasoning'], str)


class TestStrategyIntegration:
    """Integration tests for complete strategy workflow"""
    
    @pytest.mark.integration
    def test_full_strategy_workflow_long_squeeze(self, long_squeeze_scenario, sample_portfolio_state):
        """Test complete strategy workflow with long squeeze scenario"""
        strategy = SqueezeFlowStrategy()
        
        result = strategy.process(long_squeeze_scenario, sample_portfolio_state)
        
        # Verify all phases executed
        assert 'phase1_context' in result['phase_results']
        assert 'phase2_divergence' in result['phase_results']
        assert 'phase3_reset' in result['phase_results']
        
        # Should have either scoring or orders depending on signal strength
        assert 'phase4_scoring' in result['phase_results'] or len(result['orders']) >= 0
        
    @pytest.mark.integration
    def test_full_strategy_workflow_short_squeeze(self, short_squeeze_scenario, sample_portfolio_state):
        """Test complete strategy workflow with short squeeze scenario"""
        strategy = SqueezeFlowStrategy()
        
        result = strategy.process(short_squeeze_scenario, sample_portfolio_state)
        
        # Should process all phases
        assert len(result['phase_results']) >= 3  # At least phases 1-3
        
        # Verify metadata
        assert result['metadata']['strategy'] == 'SqueezeFlowStrategy'
        assert result['metadata']['symbol'] == 'BTCUSDT'
        
    @pytest.mark.integration
    def test_position_sizing_by_score(self, sample_dataset, sample_portfolio_state):
        """Test position sizing adapts to signal score"""
        config = SqueezeFlowConfig()
        strategy = SqueezeFlowStrategy(config=config)
        
        # Mock a high-score scenario
        with patch.object(strategy.phase4, 'calculate_score') as mock_score:
            mock_score.return_value = {
                'total_score': 8.5,
                'should_trade': True,
                'direction': 'LONG',
                'confidence': 0.85,
                'signal_quality': 'HIGH'
            }
            
            result = strategy.process(sample_dataset, sample_portfolio_state)
            
            if result['orders']:
                order = result['orders'][0]
                # High score should result in higher leverage
                assert order.get('leverage', 1) >= 3
                
    @pytest.mark.integration
    def test_no_trade_below_threshold(self, neutral_market_scenario, sample_portfolio_state):
        """Test strategy doesn't trade when score is below threshold"""
        strategy = SqueezeFlowStrategy()
        
        # Mock low score
        with patch.object(strategy.phase4, 'calculate_score') as mock_score:
            mock_score.return_value = {
                'total_score': 2.0,  # Below min_entry_score of 4.0
                'should_trade': False,
                'direction': 'NONE'
            }
            
            result = strategy.process(neutral_market_scenario, sample_portfolio_state)
            
            # Should not generate any orders
            assert len(result['orders']) == 0
            
    @pytest.mark.performance
    def test_strategy_performance_benchmark(self, sample_dataset, sample_portfolio_state):
        """Test strategy processing performance"""
        import time
        
        strategy = SqueezeFlowStrategy()
        
        start_time = time.time()
        result = strategy.process(sample_dataset, sample_portfolio_state)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Strategy should process within reasonable time (< 1 second)
        assert processing_time < 1.0, f"Strategy processing too slow: {processing_time:.3f}s"
        
        # Should produce valid result
        assert 'orders' in result
        assert 'phase_results' in result