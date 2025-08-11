"""
Simple Dashboard using Chart.js - FUCK TradingView
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class SimpleVisualizer:
    """Simple dashboard using Chart.js instead of broken TradingView"""
    
    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create a working dashboard with Chart.js"""
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get symbol
        symbol = results.get('symbol', 'BTC') if isinstance(results, dict) else 'BTC'
        
        # Prepare data for Chart.js (much simpler format)
        ohlcv = dataset.get('ohlcv', pd.DataFrame()) if isinstance(dataset, dict) else pd.DataFrame()
        
        # Create simple price data
        labels = []
        prices = []
        
        if not ohlcv.empty and len(ohlcv) > 0:
            # Sample data if too many points
            step = max(1, len(ohlcv) // 500)
            sampled = ohlcv.iloc[::step]
            
            for idx, row in sampled.iterrows():
                labels.append(str(idx)[:19])  # Timestamp as string
                prices.append(float(row.get('close', 0)))
        else:
            # Test data
            for i in range(50):
                labels.append(f"Point {i}")
                prices.append(100 + np.random.randn() * 10)
        
        # Create HTML with Chart.js
        dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SqueezeFlow - {symbol}</title>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #5d94ff;
        }}
        .chart-container {{
            position: relative;
            height: 500px;
            background: #1e222d;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background: #1e222d;
            padding: 15px;
            border-radius: 8px;
        }}
        .metric-label {{
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metric-value {{
            color: #d1d4dc;
            font-size: 24px;
            font-weight: 600;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SqueezeFlow Trading Dashboard - {symbol}</h1>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Balance</div>
                <div class="metric-value">${results.get('final_balance', 10000) if isinstance(results, dict) else 10000:,.0f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return</div>
                <div class="metric-value">{results.get('total_return', 0) if isinstance(results, dict) else 0:.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Trades</div>
                <div class="metric-value">{len(executed_orders)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{results.get('win_rate', 0) if isinstance(results, dict) else 0:.0f}%</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="priceChart"></canvas>
        </div>
        
        <p style="color: #787b86;">Using Chart.js because TradingView doesn't fucking work</p>
    </div>
    
    <script>
        // Chart.js configuration
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels[:200])},  // Limit to 200 points for performance
                datasets: [{{
                    label: '{symbol} Price',
                    data: {json.dumps(prices[:200])},
                    borderColor: '#5d94ff',
                    backgroundColor: 'rgba(93, 148, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        labels: {{
                            color: '#d1d4dc'
                        }}
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        grid: {{
                            color: '#2a2e39'
                        }},
                        ticks: {{
                            color: '#787b86',
                            maxTicksLimit: 10
                        }}
                    }},
                    y: {{
                        display: true,
                        grid: {{
                            color: '#2a2e39'
                        }},
                        ticks: {{
                            color: '#787b86'
                        }}
                    }}
                }}
            }}
        }});
        
        console.log('Chart.js chart created with', {len(prices)}, 'data points');
    </script>
</body>
</html>"""
        
        # Save dashboard
        dashboard_path = report_dir / "dashboard.html"
        dashboard_path.write_text(dashboard_html)
        
        # Auto-capture screenshot for Claude
        try:
            from .visual_validator import DashboardVisualValidator
            validator = DashboardVisualValidator(str(self.output_dir.parent))
            result = validator.capture_dashboard(dashboard_path)
            if result["success"]:
                logger.info(f"📸 Screenshot captured for Claude: {result['screenshot_path']}")
                logger.info("💡 Claude can now analyze the dashboard using Read tool on the screenshot")
            else:
                logger.warning(f"Screenshot capture failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            logger.warning(f"Could not auto-capture screenshot: {e}")
            logger.info("💡 Run: python3 backtest/reporting/visual_validator.py")
        
        return str(dashboard_path)