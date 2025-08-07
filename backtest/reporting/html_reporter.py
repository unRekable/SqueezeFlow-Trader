#!/usr/bin/env python3
"""
HTML Reporter - Professional HTML report generation
Creates comprehensive backtest reports with embedded charts
"""

from datetime import datetime
from typing import Dict, List
from pathlib import Path


class HTMLReporter:
    """Professional HTML report generator for backtest results"""
    
    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'neutral': '#6C757D',
            'background': '#F8F9FA'
        }
    
    def create_html_report(self, results: Dict, dataset: Dict, 
                          executed_orders: List[Dict], charts: Dict, 
                          output_dir: Path) -> Path:
        """
        Create comprehensive HTML report
        
        Args:
            results: Backtest results
            dataset: Market dataset
            executed_orders: List of executed orders
            charts: Chart filenames dictionary
            output_dir: Output directory
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = self._build_html_structure(
            timestamp, results, dataset, executed_orders, charts
        )
        
        # Save HTML report
        report_path = output_dir / 'report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _build_html_structure(self, timestamp: str, results: Dict, dataset: Dict,
                             executed_orders: List[Dict], charts: Dict) -> str:
        """Build complete HTML structure"""
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SqueezeFlow Backtest Report</title>
            {self._get_css_styles()}
        </head>
        <body>
            <div class="container">
                {self._build_header(timestamp, dataset)}
                {self._build_metrics_grid(results)}
                {self._build_charts_section(charts)}
                {self._build_detailed_results(results, dataset)}
                {self._build_trades_table(executed_orders)}
                {self._build_footer()}
            </div>
        </body>
        </html>
        """
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
        return f"""
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: {self.colors['background']};
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 3px solid {self.colors['primary']};
                padding-bottom: 20px;
            }}
            .header h1 {{
                color: {self.colors['primary']};
                margin: 0;
                font-size: 2.5em;
            }}
            .header .subtitle {{
                color: {self.colors['neutral']};
                font-size: 1.1em;
                margin-top: 10px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, {self.colors['primary']}, {self.colors['secondary']});
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s;
            }}
            .metric-card:hover {{
                transform: translateY(-2px);
            }}
            .metric-card h3 {{
                margin: 0;
                font-size: 1.1em;
                opacity: 0.9;
                font-weight: normal;
            }}
            .metric-card .value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .charts-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }}
            .chart-container {{
                text-align: center;
                background-color: {self.colors['background']};
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }}
            .chart-container h3 {{
                color: {self.colors['primary']};
                margin-top: 0;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .details-section {{
                margin: 30px 0;
                padding: 25px;
                background-color: {self.colors['background']};
                border-radius: 8px;
                border-left: 4px solid {self.colors['primary']};
            }}
            .details-section h3 {{
                color: {self.colors['primary']};
                margin-top: 0;
            }}
            .details-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}
            .detail-item {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            .detail-label {{
                font-weight: 600;
                color: {self.colors['neutral']};
            }}
            .detail-value {{
                font-weight: bold;
            }}
            .trades-table {{
                margin-top: 30px;
                overflow-x: auto;
            }}
            .trades-table h3 {{
                color: {self.colors['primary']};
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background-color: white;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: {self.colors['primary']};
                color: white;
                font-weight: 600;
            }}
            tr:hover {{
                background-color: {self.colors['background']};
            }}
            .positive {{ color: {self.colors['success']}; font-weight: bold; }}
            .negative {{ color: {self.colors['danger']}; font-weight: bold; }}
            .neutral {{ color: {self.colors['neutral']}; }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: {self.colors['neutral']};
                font-size: 0.9em;
            }}
            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }}
            .badge-success {{ background-color: {self.colors['success']}; color: white; }}
            .badge-danger {{ background-color: {self.colors['danger']}; color: white; }}
            .badge-primary {{ background-color: {self.colors['primary']}; color: white; }}
        </style>
        """
    
    def _build_header(self, timestamp: str, dataset: Dict) -> str:
        """Build report header"""
        return f"""
        <div class="header">
            <h1>🎯 SqueezeFlow Backtest Report</h1>
            <div class="subtitle">
                <p>Generated on {timestamp}</p>
                <p>
                    Symbol: <strong>{dataset.get('symbol', 'N/A')}</strong> | 
                    Timeframe: <strong>{dataset.get('timeframe', 'N/A')}</strong> |
                    Period: <strong>{dataset.get('start_time', 'N/A')}</strong> to <strong>{dataset.get('end_time', 'N/A')}</strong>
                </p>
            </div>
        </div>
        """
    
    def _build_metrics_grid(self, results: Dict) -> str:
        """Build metrics grid"""
        return f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="value {'positive' if results.get('total_return', 0) > 0 else 'negative'}">
                    {results.get('total_return', 0):.2f}%
                </div>
            </div>
            <div class="metric-card">
                <h3>Final Balance</h3>
                <div class="value">
                    ${results.get('final_balance', 0):,.2f}
                </div>
            </div>
            <div class="metric-card">
                <h3>Total Trades</h3>
                <div class="value">
                    {results.get('total_trades', 0)}
                </div>
            </div>
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="value">
                    {results.get('win_rate', 0):.1f}%
                </div>
            </div>
        </div>
        """
    
    def _build_charts_section(self, charts: Dict) -> str:
        """Build charts section"""
        charts_html = '<div class="charts-grid">'
        
        chart_titles = {
            'equity_curve': '📈 Portfolio Equity Curve',
            'price_signals': '💹 Price Action & Trading Signals',
            'cvd_analysis': '📊 CVD Analysis Dashboard',
            'performance': '🎯 Performance Metrics',
            'trades': '📋 Trade Analysis',
            'squeeze_signals': '⚡ Squeeze Signals'
        }
        
        for chart_name, chart_file in charts.items():
            if chart_file:
                title = chart_titles.get(chart_name, chart_name.replace('_', ' ').title())
                charts_html += f"""
                <div class="chart-container">
                    <h3>{title}</h3>
                    <img src="{chart_file}" alt="{chart_name}" loading="lazy">
                </div>
                """
        
        charts_html += '</div>'
        return charts_html
    
    def _build_detailed_results(self, results: Dict, dataset: Dict) -> str:
        """Build detailed results section"""
        
        initial_balance = results.get('initial_balance', 0)
        final_balance = results.get('final_balance', 0)
        absolute_change = final_balance - initial_balance
        
        quality = results.get('data_quality', {})
        metadata = results.get('metadata', {})
        
        return f"""
        <div class="details-section">
            <h3>📊 Detailed Performance Analysis</h3>
            <div class="details-grid">
                <div>
                    <div class="detail-item">
                        <span class="detail-label">Initial Balance:</span>
                        <span class="detail-value">${initial_balance:,.2f}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Final Balance:</span>
                        <span class="detail-value">${final_balance:,.2f}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Absolute Gain/Loss:</span>
                        <span class="detail-value {'positive' if absolute_change > 0 else 'negative'}">
                            ${absolute_change:,.2f}
                        </span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Total Return:</span>
                        <span class="detail-value {'positive' if results.get('total_return', 0) > 0 else 'negative'}">
                            {results.get('total_return', 0):.2f}%
                        </span>
                    </div>
                </div>
                <div>
                    <div class="detail-item">
                        <span class="detail-label">Winning Trades:</span>
                        <span class="detail-value positive">{results.get('winning_trades', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Losing Trades:</span>
                        <span class="detail-value negative">{results.get('losing_trades', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Win Rate:</span>
                        <span class="detail-value">{results.get('win_rate', 0):.1f}%</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Data Quality:</span>
                        <span class="detail-value">
                            <span class="badge {'badge-success' if quality.get('overall_quality') else 'badge-danger'}">
                                {'GOOD' if quality.get('overall_quality') else 'POOR'}
                            </span>
                        </span>
                    </div>
                </div>
            </div>
            
            <h3 style="margin-top: 25px;">📈 Market Data Summary</h3>
            <div class="details-grid">
                <div>
                    <div class="detail-item">
                        <span class="detail-label">Data Points Analyzed:</span>
                        <span class="detail-value">{metadata.get('data_points', 'N/A')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">SPOT Markets:</span>
                        <span class="detail-value">{metadata.get('spot_markets_count', 'N/A')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">FUTURES Markets:</span>
                        <span class="detail-value">{metadata.get('futures_markets_count', 'N/A')}</span>
                    </div>
                </div>
                <div>
                    <div class="detail-item">
                        <span class="detail-label">Has Price Data:</span>
                        <span class="detail-value">
                            <span class="badge {'badge-success' if quality.get('has_price_data') else 'badge-danger'}">
                                {'YES' if quality.get('has_price_data') else 'NO'}
                            </span>
                        </span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Has CVD Data:</span>
                        <span class="detail-value">
                            <span class="badge {'badge-success' if quality.get('has_cvd_divergence') else 'badge-danger'}">
                                {'YES' if quality.get('has_cvd_divergence') else 'NO'}
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _build_trades_table(self, executed_orders: List[Dict]) -> str:
        """Build trades table"""
        if not executed_orders:
            return """
            <div class="trades-table">
                <h3>📋 Trade History</h3>
                <p class="neutral">No trades were executed during this backtest.</p>
            </div>
            """
        
        trades_html = """
        <div class="trades-table">
            <h3>📋 Trade History</h3>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>PnL</th>
                        <th>Signal Type</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for order in executed_orders:
            pnl = order.get('pnl', 0)
            pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else 'neutral'
            
            side = order.get('side', '').upper()
            side_class = 'positive' if side == 'BUY' else 'negative' if side == 'SELL' else 'neutral'
            
            trades_html += f"""
                <tr>
                    <td>{order.get('timestamp', 'N/A')}</td>
                    <td>{order.get('symbol', 'N/A')}</td>
                    <td class="{side_class}">{side}</td>
                    <td>{order.get('quantity', 0):.4f}</td>
                    <td>${order.get('price', 0):.2f}</td>
                    <td class="{pnl_class}">${pnl:.2f}</td>
                    <td>{order.get('signal_type', 'N/A')}</td>
                    <td>{order.get('confidence', 0):.2f}</td>
                </tr>
            """
        
        trades_html += """
                </tbody>
            </table>
        </div>
        """
        
        return trades_html
    
    def _build_footer(self) -> str:
        """Build report footer"""
        return f"""
        <div class="footer">
            <p>Generated by SqueezeFlow Trader Backtest Engine</p>
            <p>Report created on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
        </div>
        """