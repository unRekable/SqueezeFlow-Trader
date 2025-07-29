#!/usr/bin/env python3
"""
Unit Tests for Fee Calculator
Tests trading fee calculations and cost analysis
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add backtest directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fees import (
    FeeCalculator, TradingCosts, ExchangeFeeStructure,
    calculate_realistic_trading_costs, get_market_fees
)


class TestTradingCosts(unittest.TestCase):
    """Test TradingCosts data class"""
    
    def test_trading_costs_creation(self):
        """Test TradingCosts creation with valid parameters"""
        costs = TradingCosts(
            entry_fee=10.0,
            exit_fee=10.0,
            slippage_cost=2.0,
            funding_cost=1.0,
            total_cost=23.0,
            cost_percentage=0.23
        )
        
        self.assertEqual(costs.entry_fee, 10.0)
        self.assertEqual(costs.exit_fee, 10.0)
        self.assertEqual(costs.slippage_cost, 2.0)
        self.assertEqual(costs.funding_cost, 1.0)
        self.assertEqual(costs.total_cost, 23.0)
        self.assertEqual(costs.cost_percentage, 0.23)
    
    def test_default_trading_costs(self):
        """Test TradingCosts with default parameters"""
        costs = TradingCosts()
        self.assertEqual(costs.entry_fee, 0.0)
        self.assertEqual(costs.exit_fee, 0.0)
        self.assertEqual(costs.slippage_cost, 0.0)
        self.assertEqual(costs.funding_cost, 0.0)
        self.assertEqual(costs.total_cost, 0.0)
        self.assertEqual(costs.cost_percentage, 0.0)


class TestExchangeFeeStructure(unittest.TestCase):
    """Test ExchangeFeeStructure data class"""
    
    def test_exchange_fee_structure_creation(self):
        """Test ExchangeFeeStructure creation"""
        fee_structure = ExchangeFeeStructure(
            name="Binance",
            taker_fee=0.001,
            maker_fee=0.001,
            slippage_bp=1.5,
            funding_rate=0.0001
        )
        
        self.assertEqual(fee_structure.name, "Binance")
        self.assertEqual(fee_structure.taker_fee, 0.001)
        self.assertEqual(fee_structure.maker_fee, 0.001)
        self.assertEqual(fee_structure.slippage_bp, 1.5)
        self.assertEqual(fee_structure.funding_rate, 0.0001)
        self.assertEqual(fee_structure.min_fee, 0.0)  # Default value


class TestFeeCalculator(unittest.TestCase):
    """Test FeeCalculator functionality"""
    
    def setUp(self):
        """Set up test fee calculator"""
        self.calculator = FeeCalculator()
    
    def test_fee_calculator_initialization(self):
        """Test fee calculator initialization"""
        self.assertIsInstance(self.calculator.exchange_fees, dict)
        self.assertIn('BINANCE', self.calculator.exchange_fees)
        self.assertIn('BINANCE_FUTURES', self.calculator.exchange_fees)
        self.assertIn('DEFAULT_SPOT', self.calculator.exchange_fees)
        self.assertIn('DEFAULT_PERP', self.calculator.exchange_fees)
    
    def test_get_market_fee_structure_binance(self):
        """Test getting fee structure for Binance market"""
        market = "BINANCE:btcusdt"
        fee_structure = self.calculator.get_market_fee_structure(market)
        
        self.assertEqual(fee_structure.name, "Binance")
        self.assertEqual(fee_structure.taker_fee, 0.001)
        self.assertEqual(fee_structure.maker_fee, 0.001)
        self.assertEqual(fee_structure.slippage_bp, 1.5)
    
    def test_get_market_fee_structure_binance_futures(self):
        """Test getting fee structure for Binance Futures market"""
        market = "BINANCE_FUTURES:btcusdt"
        fee_structure = self.calculator.get_market_fee_structure(market)
        
        self.assertEqual(fee_structure.name, "Binance Futures")
        self.assertEqual(fee_structure.taker_fee, 0.0004)
        self.assertEqual(fee_structure.maker_fee, 0.0002)
        self.assertEqual(fee_structure.slippage_bp, 2.0)
    
    def test_get_market_fee_structure_unknown_exchange(self):
        """Test getting fee structure for unknown exchange"""
        market = "UNKNOWN:btcusdt"
        fee_structure = self.calculator.get_market_fee_structure(market)
        
        # Should return default spot fees
        self.assertEqual(fee_structure.name, "Default Spot")
        self.assertEqual(fee_structure.taker_fee, 0.001)
        self.assertEqual(fee_structure.maker_fee, 0.001)
    
    @patch('utils.market_discovery.market_discovery')
    @patch('utils.exchange_mapper.exchange_mapper')
    def test_calculate_trading_costs_with_mock_data(self, mock_exchange_mapper, mock_market_discovery):
        """Test trading cost calculation with mocked market data"""
        # Mock market discovery response
        mock_market_discovery.get_markets_by_type.return_value = {
            'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD'],
            'perp': ['BINANCE_FUTURES:btcusdt']
        }
        
        # Mock exchange mapper responses
        mock_exchange_mapper.extract_exchange.side_effect = ['BINANCE', 'COINBASE', 'BINANCE_FUTURES']
        mock_exchange_mapper.get_market_type.side_effect = ['SPOT', 'SPOT', 'PERP']
        
        # Calculate costs
        costs = self.calculator.calculate_trading_costs('BTC', 10000.0, holding_hours=4.0)
        
        self.assertIsInstance(costs, TradingCosts)
        self.assertGreater(costs.total_cost, 0)
        self.assertGreater(costs.cost_percentage, 0)
        self.assertGreaterEqual(costs.entry_fee, 0)
        self.assertGreaterEqual(costs.exit_fee, 0)
        self.assertGreaterEqual(costs.slippage_cost, 0)
    
    @patch('utils.market_discovery.market_discovery')
    def test_calculate_trading_costs_no_markets(self, mock_market_discovery):
        """Test trading cost calculation when no markets found"""
        # Mock no markets found
        mock_market_discovery.get_markets_by_type.return_value = {
            'spot': [],
            'perp': []
        }
        
        # Should use default fees
        costs = self.calculator.calculate_trading_costs('UNKNOWN', 10000.0)
        
        self.assertIsInstance(costs, TradingCosts)
        self.assertGreater(costs.total_cost, 0)
        # Should be using default spot fees (0.1% taker)
        expected_entry_fee = 10000.0 * 0.001
        self.assertEqual(costs.entry_fee, expected_entry_fee)
    
    def test_calculate_trading_costs_error_handling(self):
        """Test trading cost calculation error handling"""
        # Test with None symbol - should handle gracefully
        with patch('utils.market_discovery.market_discovery') as mock_discovery:
            mock_discovery.get_markets_by_type.side_effect = Exception("Test error")
            
            costs = self.calculator.calculate_trading_costs('BTC', 10000.0)
            
            # Should return fallback costs
            self.assertIsInstance(costs, TradingCosts)
            self.assertEqual(costs.entry_fee, 10000.0 * 0.001)  # 0.1% fallback
            self.assertEqual(costs.exit_fee, 10000.0 * 0.001)
            self.assertEqual(costs.slippage_cost, 10000.0 * 0.0002)  # 2bp
    
    @patch('utils.market_discovery.market_discovery')
    @patch('utils.exchange_mapper.exchange_mapper')
    def test_get_symbol_fee_summary(self, mock_exchange_mapper, mock_market_discovery):
        """Test fee summary generation for a symbol"""
        # Mock market discovery response
        mock_market_discovery.get_markets_by_type.return_value = {
            'spot': ['BINANCE:btcusdt', 'COINBASE:BTC-USD'],
            'perp': ['BINANCE_FUTURES:btcusdt']
        }
        
        # Mock exchange mapper responses
        mock_exchange_mapper.extract_exchange.side_effect = ['BINANCE', 'COINBASE', 'BINANCE_FUTURES']
        
        summary = self.calculator.get_symbol_fee_summary('BTC')
        
        self.assertIn('symbol', summary)
        self.assertIn('spot_markets_analyzed', summary)
        self.assertIn('perp_markets_analyzed', summary)
        self.assertIn('spot_fees', summary)
        self.assertIn('perp_fees', summary)
        self.assertIn('averages', summary)
        
        self.assertEqual(summary['symbol'], 'BTC')
        self.assertEqual(summary['spot_markets_analyzed'], 2)
        self.assertEqual(summary['perp_markets_analyzed'], 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    @patch('core.fees.fee_calculator')
    def test_calculate_realistic_trading_costs(self, mock_fee_calculator):
        """Test convenience function for calculating trading costs"""
        # Mock the calculator's method
        mock_costs = TradingCosts(
            entry_fee=10.0, exit_fee=10.0, slippage_cost=2.0,
            funding_cost=1.0, total_cost=23.0, cost_percentage=0.23
        )
        mock_fee_calculator.calculate_trading_costs.return_value = mock_costs
        
        costs = calculate_realistic_trading_costs('BTC', 10000.0, 4.0)
        
        self.assertEqual(costs, mock_costs)
        mock_fee_calculator.calculate_trading_costs.assert_called_once_with('BTC', 10000.0, 4.0)
    
    @patch('core.fees.fee_calculator')
    def test_get_market_fees(self, mock_fee_calculator):
        """Test convenience function for getting market fees"""
        # Mock the calculator's method
        mock_fee_structure = ExchangeFeeStructure(
            name="Binance", taker_fee=0.001, maker_fee=0.001,
            slippage_bp=1.5, funding_rate=0.0001
        )
        mock_fee_calculator.get_market_fee_structure.return_value = mock_fee_structure
        
        fee_structure = get_market_fees("BINANCE:btcusdt")
        
        self.assertEqual(fee_structure, mock_fee_structure)
        mock_fee_calculator.get_market_fee_structure.assert_called_once_with("BINANCE:btcusdt")


class TestFeeCalculatorIntegration(unittest.TestCase):
    """Integration tests for fee calculator"""
    
    def test_all_exchanges_have_valid_fees(self):
        """Test that all configured exchanges have valid fee structures"""
        calculator = FeeCalculator()
        
        for exchange_name, fee_structure in calculator.exchange_fees.items():
            self.assertIsInstance(fee_structure, ExchangeFeeStructure)
            self.assertIsInstance(fee_structure.name, str)
            self.assertGreaterEqual(fee_structure.taker_fee, -0.001)  # Allow for maker rebates
            self.assertGreaterEqual(fee_structure.maker_fee, -0.001)  # Allow for maker rebates
            self.assertGreaterEqual(fee_structure.slippage_bp, 0)
            self.assertGreaterEqual(fee_structure.funding_rate, 0)
            self.assertGreaterEqual(fee_structure.min_fee, 0)
    
    def test_major_exchanges_configured(self):
        """Test that major exchanges are properly configured"""
        calculator = FeeCalculator()
        
        major_exchanges = ['BINANCE', 'BINANCE_FUTURES', 'COINBASE', 'BYBIT', 'OKEX', 'KRAKEN']
        
        for exchange in major_exchanges:
            self.assertIn(exchange, calculator.exchange_fees)
            fee_structure = calculator.exchange_fees[exchange]
            self.assertIsNotNone(fee_structure.name)
            self.assertGreater(len(fee_structure.name), 0)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)