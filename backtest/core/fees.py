#!/usr/bin/env python3
"""
Trading Fees & Costs Calculator - Real Market Fees using Exchange Mapper
Calculates realistic trading costs based on actual exchange fees and market types
"""

import logging
import sys
import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Add project root to path for imports (go up 3 levels: core/ -> backtest/ -> project/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.exchange_mapper import exchange_mapper


@dataclass
class TradingCosts:
    """Complete breakdown of trading costs"""
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    funding_cost: float = 0.0
    total_cost: float = 0.0
    cost_percentage: float = 0.0


@dataclass
class ExchangeFeeStructure:
    """Exchange-specific fee structure"""
    name: str
    taker_fee: float        # Taker fee percentage
    maker_fee: float        # Maker fee percentage  
    slippage_bp: float      # Expected slippage in basis points
    funding_rate: float     # Funding rate for perpetuals (hourly)
    min_fee: float = 0.0    # Minimum fee per trade


class FeeCalculator:
    """
    Real market fee calculation using exchange mapper classification
    Integrates with existing exchange_mapper infrastructure
    """
    
    def __init__(self):
        self.logger = logging.getLogger('FeeCalculator')
        self.exchange_fees = self._setup_exchange_fees()
        
    def _setup_exchange_fees(self) -> Dict[str, ExchangeFeeStructure]:
        """Setup realistic fee structures for major exchanges"""
        return {
            'BINANCE': ExchangeFeeStructure(
                name='Binance',
                taker_fee=0.001,      # 0.10% taker
                maker_fee=0.001,      # 0.10% maker  
                slippage_bp=1.5,      # 1.5 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            'BINANCE_FUTURES': ExchangeFeeStructure(
                name='Binance Futures',
                taker_fee=0.0004,     # 0.04% taker
                maker_fee=0.0002,     # 0.02% maker
                slippage_bp=2.0,      # 2.0 basis points  
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            'COINBASE': ExchangeFeeStructure(
                name='Coinbase Pro',
                taker_fee=0.005,      # 0.50% taker (retail)
                maker_fee=0.005,      # 0.50% maker (retail)
                slippage_bp=3.0,      # 3.0 basis points
                funding_rate=0.0      # No funding for spot
            ),
            'BYBIT': ExchangeFeeStructure(
                name='Bybit',
                taker_fee=0.00055,    # 0.055% taker
                maker_fee=0.0001,     # 0.01% maker
                slippage_bp=2.5,      # 2.5 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            'OKEX': ExchangeFeeStructure(
                name='OKX',
                taker_fee=0.0008,     # 0.08% taker
                maker_fee=0.0006,     # 0.06% maker
                slippage_bp=2.0,      # 2.0 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            'KRAKEN': ExchangeFeeStructure(
                name='Kraken',
                taker_fee=0.0026,     # 0.26% taker
                maker_fee=0.0016,     # 0.16% maker
                slippage_bp=4.0,      # 4.0 basis points
                funding_rate=0.0002   # 0.02% hourly funding
            ),
            'BITFINEX': ExchangeFeeStructure(
                name='Bitfinex',
                taker_fee=0.002,      # 0.20% taker
                maker_fee=0.001,      # 0.10% maker
                slippage_bp=3.0,      # 3.0 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            'BITMEX': ExchangeFeeStructure(
                name='BitMEX',
                taker_fee=0.00075,    # 0.075% taker
                maker_fee=-0.00025,   # -0.025% maker (rebate)
                slippage_bp=5.0,      # 5.0 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            'DERIBIT': ExchangeFeeStructure(
                name='Deribit',
                taker_fee=0.0005,     # 0.05% taker
                maker_fee=0.0,        # 0.00% maker
                slippage_bp=3.0,      # 3.0 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            ),
            # Default fallback fees
            'DEFAULT_SPOT': ExchangeFeeStructure(
                name='Default Spot',
                taker_fee=0.001,      # 0.10% taker
                maker_fee=0.001,      # 0.10% maker
                slippage_bp=2.0,      # 2.0 basis points
                funding_rate=0.0      # No funding for spot
            ),
            'DEFAULT_PERP': ExchangeFeeStructure(
                name='Default Perpetual',
                taker_fee=0.0005,     # 0.05% taker
                maker_fee=0.0002,     # 0.02% maker
                slippage_bp=2.5,      # 2.5 basis points
                funding_rate=0.0001   # 0.01% hourly funding
            )
        }
    
    def get_market_fee_structure(self, market_identifier: str) -> ExchangeFeeStructure:
        """
        Get fee structure for a specific market using exchange mapper
        
        Args:
            market_identifier: Full market string (e.g., "BINANCE:btcusdt")
            
        Returns:
            ExchangeFeeStructure for the market
        """
        try:
            # Extract exchange name
            exchange = exchange_mapper.extract_exchange(market_identifier)
            if not exchange:
                self.logger.warning(f"Could not extract exchange from {market_identifier}")
                return self.exchange_fees['DEFAULT_SPOT']
            
            # Get market type (SPOT or PERP)
            market_type = exchange_mapper.get_market_type(market_identifier)
            
            # Look for exchange-specific fees
            if exchange in self.exchange_fees:
                return self.exchange_fees[exchange]
            
            # Fallback based on market type
            if market_type == 'PERP':
                return self.exchange_fees['DEFAULT_PERP']
            else:
                return self.exchange_fees['DEFAULT_SPOT']
                
        except Exception as e:
            self.logger.error(f"Error getting fee structure for {market_identifier}: {e}")
            return self.exchange_fees['DEFAULT_SPOT']
    
    def calculate_trading_costs(self, symbol: str, trade_value: float, 
                               holding_hours: float = 1.0,
                               use_market_orders: bool = True) -> TradingCosts:
        """
        Calculate complete trading costs for a symbol using real market data
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            trade_value: Total trade value in USD
            holding_hours: Hours position will be held (for funding)
            use_market_orders: Use taker fees if True, maker fees if False
            
        Returns:
            TradingCosts with complete fee breakdown
        """
        try:
            # Get markets for symbol using exchange mapper
            from utils.market_discovery import market_discovery
            markets_by_type = market_discovery.get_markets_by_type(symbol)
            
            # Calculate weighted average fees across all available markets
            total_weight = 0
            weighted_entry_fee = 0
            weighted_exit_fee = 0
            weighted_slippage = 0
            weighted_funding = 0
            
            # Combine spot and perp markets for comprehensive fee calculation
            all_markets = markets_by_type['spot'] + markets_by_type['perp']
            
            if not all_markets:
                self.logger.warning(f"No markets found for {symbol}, using default fees")
                default_fees = self.exchange_fees['DEFAULT_SPOT']
                fee_rate = default_fees.taker_fee if use_market_orders else default_fees.maker_fee
                
                return TradingCosts(
                    entry_fee=trade_value * fee_rate,
                    exit_fee=trade_value * fee_rate,
                    slippage_cost=trade_value * (default_fees.slippage_bp / 10000),
                    funding_cost=0.0,
                    total_cost=trade_value * (fee_rate * 2 + default_fees.slippage_bp / 10000),
                    cost_percentage=(fee_rate * 2 + default_fees.slippage_bp / 10000) * 100
                )
            
            # Weight markets by typical volume (major exchanges get higher weight)
            major_exchanges = ['BINANCE', 'COINBASE', 'BYBIT', 'OKEX', 'KRAKEN']
            
            for market in all_markets[:10]:  # Limit to top 10 markets
                fee_structure = self.get_market_fee_structure(market)
                exchange = exchange_mapper.extract_exchange(market)
                market_type = exchange_mapper.get_market_type(market)
                
                # Assign weights (major exchanges get higher weight)
                weight = 2.0 if exchange in major_exchanges else 1.0
                
                # Use appropriate fee rate
                fee_rate = fee_structure.taker_fee if use_market_orders else fee_structure.maker_fee
                
                # Accumulate weighted fees
                weighted_entry_fee += fee_rate * weight
                weighted_exit_fee += fee_rate * weight
                weighted_slippage += (fee_structure.slippage_bp / 10000) * weight
                
                # Only add funding costs for perpetual contracts
                if market_type == 'PERP':
                    weighted_funding += fee_structure.funding_rate * holding_hours * weight
                
                total_weight += weight
            
            # Calculate average fees
            if total_weight > 0:
                avg_entry_fee = (weighted_entry_fee / total_weight) * trade_value
                avg_exit_fee = (weighted_exit_fee / total_weight) * trade_value
                avg_slippage = (weighted_slippage / total_weight) * trade_value
                avg_funding = (weighted_funding / total_weight) * trade_value
            else:
                # Fallback
                default_fees = self.exchange_fees['DEFAULT_SPOT']
                fee_rate = default_fees.taker_fee if use_market_orders else default_fees.maker_fee
                avg_entry_fee = trade_value * fee_rate
                avg_exit_fee = trade_value * fee_rate
                avg_slippage = trade_value * (default_fees.slippage_bp / 10000)
                avg_funding = 0.0
            
            total_cost = avg_entry_fee + avg_exit_fee + avg_slippage + avg_funding
            cost_percentage = (total_cost / trade_value) * 100 if trade_value > 0 else 0.0
            
            costs = TradingCosts(
                entry_fee=avg_entry_fee,
                exit_fee=avg_exit_fee,
                slippage_cost=avg_slippage,
                funding_cost=avg_funding,
                total_cost=total_cost,
                cost_percentage=cost_percentage
            )
            
            self.logger.debug(f"Trading costs for {symbol} (${trade_value:.2f}): "
                            f"Total: ${total_cost:.2f} ({cost_percentage:.3f}%)")
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Error calculating trading costs for {symbol}: {e}")
            # Return conservative fallback costs
            fallback_fee_rate = 0.001  # 0.1%
            fallback_cost = trade_value * fallback_fee_rate * 2  # Entry + exit
            
            return TradingCosts(
                entry_fee=trade_value * fallback_fee_rate,
                exit_fee=trade_value * fallback_fee_rate,
                slippage_cost=trade_value * 0.0002,  # 2 basis points
                funding_cost=0.0,
                total_cost=fallback_cost,
                cost_percentage=(fallback_cost / trade_value) * 100
            )
    
    def get_symbol_fee_summary(self, symbol: str) -> Dict[str, any]:
        """
        Get comprehensive fee summary for a symbol across all markets
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with fee analysis
        """
        try:
            from utils.market_discovery import market_discovery
            markets_by_type = market_discovery.get_markets_by_type(symbol)
            
            spot_fees = []
            perp_fees = []
            
            # Analyze spot markets
            for market in markets_by_type['spot'][:5]:  # Top 5 spot markets
                fee_structure = self.get_market_fee_structure(market)
                spot_fees.append({
                    'market': market,
                    'exchange': exchange_mapper.extract_exchange(market),
                    'taker_fee': fee_structure.taker_fee * 100,  # Convert to percentage
                    'maker_fee': fee_structure.maker_fee * 100,
                    'slippage_bp': fee_structure.slippage_bp
                })
            
            # Analyze perp markets
            for market in markets_by_type['perp'][:5]:  # Top 5 perp markets
                fee_structure = self.get_market_fee_structure(market)
                perp_fees.append({
                    'market': market,
                    'exchange': exchange_mapper.extract_exchange(market),
                    'taker_fee': fee_structure.taker_fee * 100,
                    'maker_fee': fee_structure.maker_fee * 100,
                    'slippage_bp': fee_structure.slippage_bp,
                    'funding_rate': fee_structure.funding_rate * 100
                })
            
            # Calculate averages
            avg_spot_taker = sum(f['taker_fee'] for f in spot_fees) / len(spot_fees) if spot_fees else 0
            avg_spot_maker = sum(f['maker_fee'] for f in spot_fees) / len(spot_fees) if spot_fees else 0
            avg_perp_taker = sum(f['taker_fee'] for f in perp_fees) / len(perp_fees) if perp_fees else 0
            avg_perp_maker = sum(f['maker_fee'] for f in perp_fees) / len(perp_fees) if perp_fees else 0
            
            return {
                'symbol': symbol,
                'spot_markets_analyzed': len(spot_fees),
                'perp_markets_analyzed': len(perp_fees),
                'spot_fees': spot_fees,
                'perp_fees': perp_fees,
                'averages': {
                    'spot_taker_fee': avg_spot_taker,
                    'spot_maker_fee': avg_spot_maker,
                    'perp_taker_fee': avg_perp_taker,
                    'perp_maker_fee': avg_perp_maker
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting fee summary for {symbol}: {e}")
            return {'error': str(e)}


# Global instance for easy import
fee_calculator = FeeCalculator()


def calculate_realistic_trading_costs(symbol: str, trade_value: float, 
                                    holding_hours: float = 1.0) -> TradingCosts:
    """
    Convenience function to calculate realistic trading costs
    
    Args:
        symbol: Trading symbol (e.g., 'BTC', 'ETH')
        trade_value: Total trade value in USD
        holding_hours: Hours position will be held
        
    Returns:
        TradingCosts object with complete breakdown
    """
    return fee_calculator.calculate_trading_costs(symbol, trade_value, holding_hours)


def get_market_fees(market_identifier: str) -> ExchangeFeeStructure:
    """
    Convenience function to get fee structure for a specific market
    
    Args:
        market_identifier: Full market string (e.g., "BINANCE:btcusdt")
        
    Returns:
        ExchangeFeeStructure for the market
    """
    return fee_calculator.get_market_fee_structure(market_identifier)


if __name__ == "__main__":
    # Test the fee calculation system
    calculator = FeeCalculator()
    
    print("🏦 Trading Fee Calculator Testing")
    print("=" * 50)
    
    # Test individual market fees
    test_markets = [
        "BINANCE:btcusdt",
        "BINANCE_FUTURES:btcusdt", 
        "COINBASE:BTC-USD",
        "BYBIT:BTCUSDT",
        "BITMEX:XBTUSD"
    ]
    
    print("\n📊 Individual Market Fees:")
    for market in test_markets:
        fees = calculator.get_market_fee_structure(market)
        market_type = exchange_mapper.get_market_type(market)
        print(f"  {market:<25} ({market_type}): Taker: {fees.taker_fee*100:.3f}%, "
              f"Maker: {fees.maker_fee*100:.3f}%, Slippage: {fees.slippage_bp:.1f}bp")
    
    # Test trading costs for different symbols
    print("\n💰 Trading Cost Analysis:")
    test_symbols = ['BTC', 'ETH']
    trade_value = 10000  # $10,000 trade
    
    for symbol in test_symbols:
        costs = calculator.calculate_trading_costs(symbol, trade_value, holding_hours=4.0)
        print(f"\n  {symbol} Trading Costs (${trade_value:,} trade, 4h hold):")
        print(f"    Entry Fee:    ${costs.entry_fee:.2f}")
        print(f"    Exit Fee:     ${costs.exit_fee:.2f}")
        print(f"    Slippage:     ${costs.slippage_cost:.2f}")
        print(f"    Funding:      ${costs.funding_cost:.2f}")
        print(f"    Total Cost:   ${costs.total_cost:.2f} ({costs.cost_percentage:.3f}%)")
    
    # Test fee summary
    print("\n📈 Fee Summary for BTC:")
    btc_summary = calculator.get_symbol_fee_summary('BTC')
    if 'averages' in btc_summary:
        avgs = btc_summary['averages']
        print(f"  Spot Markets: {btc_summary['spot_markets_analyzed']} analyzed")
        print(f"    Avg Taker: {avgs['spot_taker_fee']:.3f}%, Avg Maker: {avgs['spot_maker_fee']:.3f}%")
        print(f"  Perp Markets: {btc_summary['perp_markets_analyzed']} analyzed")  
        print(f"    Avg Taker: {avgs['perp_taker_fee']:.3f}%, Avg Maker: {avgs['perp_maker_fee']:.3f}%")