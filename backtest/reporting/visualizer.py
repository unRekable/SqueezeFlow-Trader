"""
Backtest Visualizer - Simple TradingView dashboard
Using TradingView's native capabilities properly
"""

import logging
from pathlib import Path
from typing import Dict, List

# Import TradingView visualizer for proper implementation
from .tradingview_visualizer import TradingViewVisualizer

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Simple, clean dashboard using TradingView properly"""
    
    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use TradingView visualizer for proper implementation
        self.tv = TradingViewVisualizer(output_dir)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create TradingView dashboard"""
        
        # Use the TradingView visualizer
        dashboard_path = self.tv.create_backtest_report(results, dataset, executed_orders)
        
        logger.info(f"Dashboard created: {dashboard_path}")
        
        return dashboard_path