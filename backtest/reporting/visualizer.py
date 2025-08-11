"""
Backtest Visualizer - TradingView Unified Dashboard
Single HTML file with TradingView charts and tabbed navigation
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import TradingView unified - THE ONLY IMPLEMENTATION WE USE
from .tradingview_unified import TradingViewUnified

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """TradingView unified dashboard - ALWAYS"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create TradingView unified dashboard - ALWAYS THE SAME"""
        
        # ALWAYS use TradingView unified - no conditionals, no fallbacks
        logger.info("Creating TradingView unified dashboard")
        
        # Add executed_orders to results where TradingView expects them
        if isinstance(results, dict) and 'executed_orders' not in results:
            results['executed_orders'] = executed_orders
        
        # Create the dashboard
        tv_viz = TradingViewUnified()
        dashboard_path = tv_viz.create_dashboard(results, dataset, str(self.output_dir))
        
        logger.info(f"✅ TradingView unified dashboard created: {dashboard_path}")
        
        return dashboard_path