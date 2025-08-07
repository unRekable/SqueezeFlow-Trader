#!/usr/bin/env python3
"""
Backtest Visualizer - Coordination for visualization system
Coordinates HTML reporting and PNG chart generation
"""

from datetime import datetime
from typing import Dict, List
from pathlib import Path

from .html_reporter import HTMLReporter
from .png_plotter import PNGPlotter


class BacktestVisualizer:
    """Coordination layer for visualization system"""
    
    def __init__(self, output_dir: str = "backtest/results/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.html_reporter = HTMLReporter()
        self.png_plotter = PNGPlotter()
    
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """
        Create comprehensive backtest report with visualizations
        
        Args:
            results: Backtest results
            dataset: Market dataset
            executed_orders: List of executed orders
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Generate PNG charts
        charts = self.png_plotter.create_all_charts(
            results, dataset, executed_orders, report_dir
        )
        
        # Generate HTML report
        html_report = self.html_reporter.create_html_report(
            results, dataset, executed_orders, charts, report_dir
        )
        
        return str(html_report)
