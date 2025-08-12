"""
Backtest Visualizer - TradingView PROPER Multi-Pane Dashboard
Uses ONE chart with multiple price scales for true multi-pane visualization
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import TradingView PROPER implementation - ONE chart, multiple scales
from .tradingview_proper_panes import TradingViewProperPanes

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """TradingView dashboard with PROPER multi-pane using ONE chart"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create TradingView dashboard with ONE chart and multiple price scales"""
        
        # Use PROPER implementation - ONE chart with visual pane separation
        logger.info("Creating TradingView dashboard with ONE chart and proper panes")
        
        # Add executed_orders to results where TradingView expects them
        if isinstance(results, dict) and 'executed_orders' not in results:
            results['executed_orders'] = executed_orders
        
        # Create the dashboard with PROPER implementation
        tv_viz = TradingViewProperPanes()
        dashboard_path = tv_viz.create_dashboard(results, dataset, str(self.output_dir))
        
        logger.info(f"✅ TradingView PROPER multi-pane dashboard created: {dashboard_path}")
        
        return dashboard_path