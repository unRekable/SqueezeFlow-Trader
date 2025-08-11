#!/usr/bin/env python3
"""Test that HTML dashboard generation works - PROPERLY PLACED TEST"""

import os
from datetime import datetime
from pathlib import Path

def test_visualizer_with_mock_data():
    """Test visualizer can generate HTML with mock data"""
    
    # Create mock results
    results = {
        'symbol': 'BTC',
        'timeframe': '1s',
        'start_time': datetime(2025, 8, 10, 14, 0, 0),
        'end_time': datetime(2025, 8, 10, 15, 0, 0),
        'initial_balance': 10000,
        'final_balance': 10100,
        'executed_orders': [],
        'squeeze_scores': {
            'timestamps': [datetime(2025, 8, 10, 14, 30, 0)],
            'scores': [7.5]
        },
        'metrics': {
            'total_return': 0.01,
            'win_rate': 0.5,
            'total_trades': 1
        }
    }
    
    dataset = {
        'ohlcv': None,  # Would normally have DataFrame
        'spot_cvd': None,
        'futures_cvd': None
    }
    
    executed_orders = [
        {
            'timestamp': datetime(2025, 8, 10, 14, 30, 0),
            'side': 'BUY',
            'price': 118000,
            'quantity': 0.001,
            'fee': 0.1
        }
    ]
    
    print("Testing HTML generation with strategy visualizer...")
    
    try:
        from backtest.reporting.strategy_visualizer import StrategyVisualizer
        
        output_dir = "backtest/results/test_visualizer"
        os.makedirs(output_dir, exist_ok=True)
        
        visualizer = StrategyVisualizer()
        
        # This should create the HTML even with None data
        # because we added error handling
        visualizer.create_dashboard(results, output_dir)
        
        # Check if HTML was created
        files = os.listdir(output_dir)
        html_files = [f for f in files if f.endswith('.html')]
        
        if html_files:
            print(f"✅ TEST PASSED: HTML files created: {html_files}")
            return True
        else:
            print("❌ TEST FAILED: No HTML files created")
            return False
            
    except Exception as e:
        print(f"❌ TEST FAILED with error: {e}")
        return False

if __name__ == "__main__":
    test_visualizer_with_mock_data()