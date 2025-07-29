#!/usr/bin/env python3
"""
CVD Data Analyzer - Examine actual CVD patterns to understand what works
"""

import sys
import os
# Add project root to path for imports (go up 3 levels: analysis/ -> backtest/ -> project/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from data.storage.influxdb_handler import InfluxDBHandler
from utils.market_discovery import MarketDiscovery
from utils.exchange_mapper import ExchangeMapper

class CVDDataAnalyzer:
    """Analyze CVD patterns to find profitable opportunities"""
    
    def __init__(self):
        self.db_handler = InfluxDBHandler(
            host='localhost',
            port=8086,
            username='',
            password='',
            database='significant_trades'
        )
        self.market_discovery = MarketDiscovery()
        self.exchange_mapper = ExchangeMapper()
        
    def load_data(self, symbol: str, start_date: str, end_date: str):
        """Load price and CVD data for analysis"""
        print(f"🔍 Loading {symbol} data from {start_date} to {end_date}...")
        
        # Discover markets
        markets = self.market_discovery.get_markets_by_type(symbol)
        spot_markets = markets.get('spot', [])
        perp_markets = markets.get('perp', [])
        
        print(f"📊 Found {len(spot_markets)} SPOT + {len(perp_markets)} PERP markets")
        
        # Load data
        data = self.db_handler.get_aggregated_ohlcv_data(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date,
            timeframe='1m',
            spot_markets=spot_markets,
            perp_markets=perp_markets
        )
        
        if data is None or data.empty:
            print(f"❌ No data found for {symbol}")
            return None
            
        print(f"✅ Loaded {len(data)} data points")
        return data
    
    def analyze_price_movements(self, data: pd.DataFrame):
        """Find significant price movements and analyze associated CVD patterns"""
        print("\n🔍 Analyzing price movements...")
        
        # Calculate price changes
        data['price_change_1m'] = data['close'].pct_change()
        data['price_change_5m'] = data['close'].pct_change(5)
        data['price_change_15m'] = data['close'].pct_change(15)
        data['price_change_30m'] = data['close'].pct_change(30)
        
        # Calculate CVD changes
        data['spot_cvd_change_30m'] = data['total_cvd_spot_cumulative'].diff(30)
        data['perp_cvd_change_30m'] = data['total_cvd_perp_cumulative'].diff(30)
        data['cvd_divergence_30m'] = data['spot_cvd_change_30m'] - data['perp_cvd_change_30m']
        
        # Find significant moves (>1% in 15 minutes)
        significant_moves = data[abs(data['price_change_15m']) > 0.01].copy()
        
        print(f"📈 Found {len(significant_moves)} significant moves (>1% in 15min)")
        
        if len(significant_moves) == 0:
            print("❌ No significant moves found")
            return None
            
        # Analyze patterns before significant moves
        results = []
        for idx, row in significant_moves.iterrows():
            if idx < 60:  # Need lookback data
                continue
                
            # Get data 30 minutes before the move
            lookback_idx = idx - 30
            pre_move_data = data.iloc[lookback_idx:idx]
            
            if len(pre_move_data) < 30:
                continue
                
            # Calculate pre-move CVD patterns
            spot_cvd_trend = pre_move_data['total_cvd_spot_cumulative'].iloc[-1] - pre_move_data['total_cvd_spot_cumulative'].iloc[0]
            perp_cvd_trend = pre_move_data['total_cvd_perp_cumulative'].iloc[-1] - pre_move_data['total_cvd_perp_cumulative'].iloc[0]
            price_trend = pre_move_data['close'].iloc[-1] - pre_move_data['close'].iloc[0]
            
            # Normalize by price
            spot_cvd_norm = spot_cvd_trend / (row['close'] * 1000)
            perp_cvd_norm = perp_cvd_trend / (row['close'] * 1000)
            price_norm = price_trend / pre_move_data['close'].iloc[0]
            
            results.append({
                'timestamp': row.name,
                'price_move_15m': row['price_change_15m'],
                'price_move_30m': row['price_change_30m'],
                'pre_spot_cvd': spot_cvd_norm,
                'pre_perp_cvd': perp_cvd_norm,
                'pre_price_trend': price_norm,
                'cvd_divergence': spot_cvd_norm - perp_cvd_norm,
                'price': row['close']
            })
        
        results_df = pd.DataFrame(results)
        print(f"✅ Analyzed {len(results_df)} moves with sufficient lookback data")
        
        return results_df
    
    def find_profitable_patterns(self, results_df: pd.DataFrame):
        """Find patterns that predict profitable moves"""
        print("\n🎯 Finding profitable patterns...")
        
        if results_df is None or len(results_df) == 0:
            print("❌ No data to analyze")
            return
            
        # Categorize moves
        strong_up = results_df[results_df['price_move_15m'] > 0.015]  # >1.5% up
        strong_down = results_df[results_df['price_move_15m'] < -0.015]  # >1.5% down
        
        print(f"📈 Strong UP moves: {len(strong_up)}")
        print(f"📉 Strong DOWN moves: {len(strong_down)}")
        
        if len(strong_up) > 0:
            print("\n🔍 Strong UP move patterns:")
            print(f"  Average pre-spot CVD: {strong_up['pre_spot_cvd'].mean():.6f}")
            print(f"  Average pre-perp CVD: {strong_up['pre_perp_cvd'].mean():.6f}")
            print(f"  Average CVD divergence: {strong_up['cvd_divergence'].mean():.6f}")
            print(f"  Average pre-price trend: {strong_up['pre_price_trend'].mean():.4f}")
            
            # Find common patterns
            positive_div = strong_up[strong_up['cvd_divergence'] > 0]
            negative_div = strong_up[strong_up['cvd_divergence'] < 0]
            print(f"  Positive divergence (spot>perp): {len(positive_div)}/{len(strong_up)} ({len(positive_div)/len(strong_up)*100:.1f}%)")
            print(f"  Negative divergence (spot<perp): {len(negative_div)}/{len(strong_up)} ({len(negative_div)/len(strong_up)*100:.1f}%)")
        
        if len(strong_down) > 0:
            print("\n🔍 Strong DOWN move patterns:")
            print(f"  Average pre-spot CVD: {strong_down['pre_spot_cvd'].mean():.6f}")
            print(f"  Average pre-perp CVD: {strong_down['pre_perp_cvd'].mean():.6f}")
            print(f"  Average CVD divergence: {strong_down['cvd_divergence'].mean():.6f}")
            print(f"  Average pre-price trend: {strong_down['pre_price_trend'].mean():.4f}")
            
            # Find common patterns
            positive_div = strong_down[strong_down['cvd_divergence'] > 0]
            negative_div = strong_down[strong_down['cvd_divergence'] < 0]
            print(f"  Positive divergence (spot>perp): {len(positive_div)}/{len(strong_down)} ({len(positive_div)/len(strong_down)*100:.1f}%)")
            print(f"  Negative divergence (spot<perp): {len(negative_div)}/{len(strong_down)} ({len(negative_div)/len(strong_down)*100:.1f}%)")
        
        return results_df
    
    def test_squeeze_thresholds(self, results_df: pd.DataFrame):
        """Test different thresholds to find optimal squeeze detection"""
        print("\n🧪 Testing squeeze detection thresholds...")
        
        if results_df is None or len(results_df) == 0:
            return
            
        # Test different divergence thresholds
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        for threshold in thresholds:
            print(f"\n📊 Threshold: {threshold:.3f}")
            
            # Test LONG signals (positive divergence predicting up moves)
            long_signals = results_df[
                (results_df['cvd_divergence'] > threshold)
            ]
            
            if len(long_signals) > 0:
                profitable_longs = long_signals[long_signals['price_move_15m'] > 0.01]  # >1% up
                long_success_rate = len(profitable_longs) / len(long_signals) * 100
                avg_return = long_signals['price_move_15m'].mean() * 100
                print(f"  LONG signals: {len(long_signals)}, Success: {long_success_rate:.1f}%, Avg return: {avg_return:.2f}%")
            
            # Test SHORT signals (negative divergence predicting down moves)
            short_signals = results_df[
                (results_df['cvd_divergence'] < -threshold)
            ]
            
            if len(short_signals) > 0:
                profitable_shorts = short_signals[short_signals['price_move_15m'] < -0.01]  # >1% down
                short_success_rate = len(profitable_shorts) / len(short_signals) * 100
                avg_return = -short_signals['price_move_15m'].mean() * 100  # Negative for short profits
                print(f"  SHORT signals: {len(short_signals)}, Success: {short_success_rate:.1f}%, Avg return: {avg_return:.2f}%")


def main():
    """Run CVD analysis"""
    analyzer = CVDDataAnalyzer()
    
    # Analyze recent ETH data
    data = analyzer.load_data('ETH', '2025-07-07 00:00:00', '2025-07-28 23:59:59')
    
    if data is not None:
        results = analyzer.analyze_price_movements(data)
        analyzer.find_profitable_patterns(results)
        analyzer.test_squeeze_thresholds(results)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()