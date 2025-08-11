"""
Backtest Visualizer - Multi-Page Dashboard System
3 pages: Main Trading, Portfolio Analytics, Exchange Analytics
"""

import logging
from pathlib import Path
from typing import Dict, List

# Import Multi-page visualizer for complete dashboard
from .multi_page_visualizer import MultiPageVisualizer

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Multi-page dashboard system with navigation"""
    
    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use Multi-page visualizer
        self.visualizer = MultiPageVisualizer(output_dir)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create strategy dashboard"""
        
        # Use the Strategy visualizer
        dashboard_path = self.visualizer.create_backtest_report(results, dataset, executed_orders)
        
        logger.info(f"Strategy dashboard created: {dashboard_path}")
        
        return dashboard_path