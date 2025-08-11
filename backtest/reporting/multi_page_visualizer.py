"""
Multi-Page Dashboard Visualizer
Creates 3 navigable pages: Main, Portfolio, Exchange Analytics
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import os

# Import the strategy visualizer and encoder for page 1
from .strategy_visualizer import StrategyVisualizer, DateTimeEncoder

# Import the new TradingView single chart implementation
try:
    from .tradingview_single_chart import TradingViewSingleChart
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiPageVisualizer:
    """Creates complete 3-page dashboard system"""
    
    def __init__(self, output_dir: str = "."):
        # Changed default to current directory
        self.output_dir = Path(output_dir)
        self.strategy_viz = StrategyVisualizer(output_dir)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create all 3 dashboard pages with navigation"""
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Double-check directory exists
        if not report_dir.exists():
            logger.error(f"Failed to create report directory: {report_dir}")
            raise FileNotFoundError(f"Could not create report directory: {report_dir}")
        logger.info(f"Report directory created: {report_dir}")
        
        try:
            # Page 1: Main Trading Dashboard (create directly here)
            main_path = self._create_main_dashboard(report_dir, results, dataset, executed_orders)
            
            # Page 2: Portfolio Analytics
            portfolio_path = self._create_portfolio_page(report_dir, results, dataset, executed_orders)
            
            # Page 3: Exchange Analytics
            exchange_path = self._create_exchange_page(report_dir, results, dataset)
            
            # Create navigation index
            index_path = self._create_index_page(report_dir, results)
            
            # Update all pages with navigation
            self._add_navigation_to_pages(report_dir)
        except Exception as e:
            logger.error(f"Error creating multi-page dashboard: {e}")
            # Ensure at least one HTML file exists
            error_path = report_dir / "error.html"
            error_path.write_text(f"<html><body><h1>Dashboard Error: {e}</h1></body></html>")
            return str(report_dir)
        
        logger.info(f"✅ Multi-page dashboard created: {report_dir}")
        logger.info(f"   Main: {main_path}")
        logger.info(f"   Portfolio: {portfolio_path}")
        logger.info(f"   Exchange: {exchange_path}")
        
        return str(index_path)
    
    def _create_main_dashboard(self, report_dir: Path, results: Dict, 
                              dataset: Dict, executed_orders: List[Dict]) -> str:
        """Create main dashboard using strategy visualizer logic directly"""
        
        # Check if we should use the new TradingView single chart implementation
        use_tradingview = os.environ.get('USE_TRADINGVIEW_PANES', 'false').lower() == 'true'
        
        logger.info(f"USE_TRADINGVIEW_PANES: {os.environ.get('USE_TRADINGVIEW_PANES')}")
        logger.info(f"use_tradingview: {use_tradingview}")
        logger.info(f"TRADINGVIEW_AVAILABLE: {TRADINGVIEW_AVAILABLE}")
        
        if use_tradingview and TRADINGVIEW_AVAILABLE:
            # Use the new TradingView single chart implementation with native panes
            logger.info("Using TradingView single chart implementation with native panes")
            try:
                tv_viz = TradingViewSingleChart()
                
                # Add executed_orders to results if needed
                if 'executed_orders' not in results and executed_orders:
                    results['executed_orders'] = executed_orders
                
                dashboard_path = tv_viz.create_dashboard(results, dataset, str(report_dir))
                logger.info(f"✅ TradingView dashboard created: {dashboard_path}")
                return dashboard_path
            except Exception as e:
                logger.error(f"Failed to create TradingView dashboard: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("Falling back to standard dashboard")
        
        # Original implementation
        # Import what we need from strategy_visualizer
        from .strategy_visualizer import StrategyVisualizer
        import tempfile
        
        # Create temp visualizer to get the HTML
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_viz = StrategyVisualizer(temp_dir)
                temp_path = temp_viz.create_backtest_report(results, dataset, executed_orders)
                
                # Read the generated HTML
                with open(temp_path, 'r') as f:
                    html_content = f.read()
            
            # Write to our directory
            dashboard_path = report_dir / "dashboard.html"
            dashboard_path.write_text(html_content)
            
            return str(dashboard_path)
        except Exception as e:
            logger.error(f"Failed to create main dashboard: {e}")
            # Create a simple error page
            dashboard_path = report_dir / "dashboard.html"
            error_html = f"""<html><body>
            <h1>Dashboard Generation Error</h1>
            <p>Error: {str(e)}</p>
            <p>Please check the logs for more details.</p>
            </body></html>"""
            dashboard_path.write_text(error_html)
            return str(dashboard_path)
    
    def _create_portfolio_page(self, report_dir: Path, results: Dict, 
                              dataset: Dict, executed_orders: List[Dict]) -> str:
        """Create portfolio analytics page"""
        
        symbol = results.get('symbol', 'UNKNOWN') if isinstance(results, dict) else 'UNKNOWN'
        
        # Calculate portfolio metrics
        equity_curve = []
        if executed_orders:
            balance = 10000  # Starting balance
            for order in executed_orders:
                if isinstance(order, dict):
                    # Handle timestamp conversion
                    ts = order.get('timestamp', 0)
                    if hasattr(ts, 'timestamp'):
                        ts = int(ts.timestamp())
                    elif isinstance(ts, str):
                        ts = int(pd.Timestamp(ts).timestamp())
                    else:
                        ts = int(ts) if ts else 0
                    
                    pnl = order.get('pnl', 0)
                    balance += pnl
                    equity_curve.append({
                        'time': ts,
                        'value': balance
                    })
        
        # Get trade statistics
        total_trades = len(executed_orders) if executed_orders else 0
        winning_trades = sum(1 for o in executed_orders if o.get('pnl', 0) > 0) if executed_orders else 0
        losing_trades = sum(1 for o in executed_orders if o.get('pnl', 0) <= 0) if executed_orders else 0
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - Portfolio Analytics</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .nav {{
            background: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            gap: 20px;
        }}
        .nav a {{
            color: #787b86;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .nav a:hover {{ background: #2a2e39; }}
        .nav a.active {{ 
            background: #2962ff;
            color: white;
        }}
        .container {{
            padding: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #2a2e39;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #26a69a;
            margin-bottom: 5px;
        }}
        .stat-value.negative {{ color: #ef5350; }}
        .stat-label {{
            color: #787b86;
            font-size: 14px;
        }}
        #equity-chart {{
            height: 400px;
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .trades-table {{
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2a2e39;
        }}
        th {{ color: #787b86; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="dashboard.html">📊 Main Dashboard</a>
        <a href="portfolio.html" class="active">💼 Portfolio</a>
        <a href="exchange.html">🏛️ Exchange Analytics</a>
    </div>
    
    <div class="container">
        <h1>{symbol} Portfolio Analytics</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_trades}</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{winning_trades}</div>
                <div class="stat-label">Winning Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value negative">{losing_trades}</div>
                <div class="stat-label">Losing Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results.get('win_rate', 0):.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results.get('sharpe_ratio', 0):.2f}</div>
                <div class="stat-label">Sharpe Ratio</div>
            </div>
            <div class="stat-card">
                <div class="stat-value negative">{results.get('max_drawdown', 0):.2f}%</div>
                <div class="stat-label">Max Drawdown</div>
            </div>
        </div>
        
        <div id="equity-chart">
            <h3>Equity Curve</h3>
        </div>
        
        <div class="trades-table">
            <h3>Recent Trades</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Amount</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody id="trades-tbody">
                </tbody>
            </table>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        const equityData = {json.dumps(equity_curve, cls=DateTimeEncoder)};
        const trades = {json.dumps(self._serialize_orders(executed_orders[:10] if executed_orders else []), cls=DateTimeEncoder)};
        
        // Create equity chart
        const chartContainer = document.getElementById('equity-chart');
        const chart = LightweightCharts.createChart(chartContainer, {{
            width: chartContainer.clientWidth - 40,
            height: 360,
            layout: {{
                background: {{ type: 'solid', color: '#1e222d' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2a2e39' }},
                horzLines: {{ color: '#2a2e39' }},
            }}
        }});
        
        if (equityData.length > 0) {{
            const equitySeries = chart.addLineSeries({{
                color: '#26a69a',
                lineWidth: 2,
            }});
            equitySeries.setData(equityData);
        }}
        
        // Populate trades table
        const tbody = document.getElementById('trades-tbody');
        trades.forEach(trade => {{
            const row = tbody.insertRow();
            const time = new Date(trade.timestamp).toLocaleString();
            const pnl = trade.pnl || 0;
            row.innerHTML = `
                <td>${{time}}</td>
                <td>${{trade.side || 'N/A'}}</td>
                <td>${{trade.price ? trade.price.toFixed(2) : 'N/A'}}</td>
                <td>${{trade.amount ? trade.amount.toFixed(6) : 'N/A'}}</td>
                <td class="${{pnl >= 0 ? '' : 'negative'}}">${{pnl.toFixed(2)}}</td>
            `;
        }});
    </script>
</body>
</html>"""
        
        # Ensure directory exists before writing
        report_dir.mkdir(parents=True, exist_ok=True)
        
        portfolio_path = report_dir / "portfolio.html"
        logger.info(f"Writing portfolio to: {portfolio_path}")
        
        try:
            portfolio_path.write_text(html)
            logger.info(f"Successfully wrote portfolio.html to {portfolio_path}")
        except Exception as e:
            logger.error(f"Failed to write portfolio.html: {e}")
            raise
            
        return str(portfolio_path)
    
    def _create_exchange_page(self, report_dir: Path, results: Dict, dataset: Dict) -> str:
        """Create exchange analytics page"""
        
        symbol = results.get('symbol', 'UNKNOWN') if isinstance(results, dict) else 'UNKNOWN'
        
        # Get exchange volume data
        exchange_volumes = {}
        if isinstance(dataset, dict) and 'spot_volume' in dataset:
            vol_df = dataset['spot_volume']
            if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
                for col in vol_df.columns:
                    if 'BINANCE' in col:
                        exchange_volumes['Binance'] = float(vol_df[col].sum())
                    elif 'BYBIT' in col:
                        exchange_volumes['Bybit'] = float(vol_df[col].sum())
                    elif 'OKX' in col:
                        exchange_volumes['OKX'] = float(vol_df[col].sum())
                    elif 'COINBASE' in col:
                        exchange_volumes['Coinbase'] = float(vol_df[col].sum())
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - Exchange Analytics</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .nav {{
            background: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            gap: 20px;
        }}
        .nav a {{
            color: #787b86;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .nav a:hover {{ background: #2a2e39; }}
        .nav a.active {{ 
            background: #2962ff;
            color: white;
        }}
        .container {{
            padding: 20px;
        }}
        .exchange-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .exchange-card {{
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #2a2e39;
        }}
        .exchange-name {{
            font-size: 18px;
            margin-bottom: 10px;
        }}
        .volume-bar {{
            height: 30px;
            background: #2962ff;
            border-radius: 4px;
            margin: 10px 0;
        }}
        #volume-chart {{
            height: 400px;
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="dashboard.html">📊 Main Dashboard</a>
        <a href="portfolio.html">💼 Portfolio</a>
        <a href="exchange.html" class="active">🏛️ Exchange Analytics</a>
    </div>
    
    <div class="container">
        <h1>{symbol} Exchange Analytics</h1>
        
        <div class="exchange-grid">
            {self._create_exchange_cards(exchange_volumes)}
        </div>
        
        <div id="volume-chart">
            <h3>Volume Distribution</h3>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        const volumes = {json.dumps(exchange_volumes, cls=DateTimeEncoder)};
        
        // Create pie chart simulation with bars
        const chartContainer = document.getElementById('volume-chart');
        const chart = LightweightCharts.createChart(chartContainer, {{
            width: chartContainer.clientWidth - 40,
            height: 360,
            layout: {{
                background: {{ type: 'solid', color: '#1e222d' }},
                textColor: '#d1d4dc',
            }}
        }});
        
        // Add volume bars per exchange
        const volumeSeries = chart.addHistogramSeries({{
            color: '#2962ff',
        }});
        
        const volumeData = Object.entries(volumes).map((entry, idx) => ({{
            time: Date.now() / 1000 + idx * 86400,
            value: entry[1],
            color: idx === 0 ? '#ffd700' : idx === 1 ? '#ff9800' : idx === 2 ? '#4caf50' : '#2196f3'
        }}));
        
        volumeSeries.setData(volumeData);
    </script>
</body>
</html>"""
        
        exchange_path = report_dir / "exchange.html"
        exchange_path.write_text(html)
        return str(exchange_path)
    
    def _create_exchange_cards(self, volumes: Dict) -> str:
        """Create exchange volume cards HTML"""
        cards = []
        total_volume = sum(volumes.values()) if volumes else 1
        
        colors = {
            'Binance': '#ffd700',
            'Bybit': '#ff9800',
            'OKX': '#4caf50',
            'Coinbase': '#2196f3'
        }
        
        for exchange, volume in volumes.items():
            pct = (volume / total_volume * 100) if total_volume > 0 else 0
            color = colors.get(exchange, '#2962ff')
            cards.append(f"""
            <div class="exchange-card">
                <div class="exchange-name">{exchange}</div>
                <div class="volume-bar" style="background: {color}; width: {pct}%;"></div>
                <div>Volume: {volume:,.0f}</div>
                <div>{pct:.1f}%</div>
            </div>
            """)
        
        return '\n'.join(cards)
    
    def _create_index_page(self, report_dir: Path, results: Dict) -> str:
        """Create index page that redirects to main dashboard"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=dashboard.html">
</head>
<body>
    <p>Redirecting to dashboard...</p>
</body>
</html>"""
        
        index_path = report_dir / "index.html"
        index_path.write_text(html)
        return str(index_path)
    
    def _serialize_orders(self, orders: List[Dict]) -> List[Dict]:
        """Convert orders to JSON-serializable format"""
        result = []
        for order in orders:
            order_copy = order.copy() if isinstance(order, dict) else {}
            if 'timestamp' in order_copy:
                ts = order_copy['timestamp']
                if hasattr(ts, 'isoformat'):
                    order_copy['timestamp'] = ts.isoformat()
                elif hasattr(ts, 'timestamp'):
                    order_copy['timestamp'] = ts.timestamp()
            result.append(order_copy)
        return result
    
    def _add_navigation_to_pages(self, report_dir: Path):
        """Add navigation bar to all existing HTML files"""
        # This would update the main dashboard.html to include navigation
        # For now, the strategy_visualizer creates its own file
        pass