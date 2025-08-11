"""
Exchange Analytics Page Implementation
Shows exchange-level statistics and volume breakdown
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ExchangeAnalyticsVisualizer:
    """Creates exchange analytics dashboard"""
    
    def __init__(self):
        self.logger = logger
    
    def create_exchange_analytics(self, results: Dict, dataset: Dict, output_dir: str) -> str:
        """Create exchange analytics page"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract exchange data from markets info
        markets = dataset.get('markets', {}) if dataset else {}
        
        # Prepare exchange statistics
        exchange_stats = self._calculate_exchange_stats(markets, dataset)
        
        # Generate HTML
        html_content = self._generate_html(results.get('symbol', 'BTC'), exchange_stats)
        
        # Save file
        analytics_path = output_path / "exchange_analytics.html"
        analytics_path.write_text(html_content)
        
        return str(analytics_path)
    
    def _calculate_exchange_stats(self, markets: Dict, dataset: Dict) -> Dict:
        """Calculate statistics per exchange"""
        
        stats = {
            'spot_exchanges': {},
            'futures_exchanges': {},
            'total_volume': 0,
            'spot_volume': 0,
            'futures_volume': 0
        }
        
        # Process spot markets
        if 'spot' in markets:
            for market in markets['spot']:
                # Parse exchange from market string (e.g., "BINANCE:btcusdt")
                if ':' in market:
                    exchange = market.split(':')[0]
                    if exchange not in stats['spot_exchanges']:
                        stats['spot_exchanges'][exchange] = {
                            'markets': [],
                            'volume': 0,
                            'trades': 0
                        }
                    stats['spot_exchanges'][exchange]['markets'].append(market)
        
        # Process futures markets
        if 'futures' in markets or 'perp' in markets:
            futures_markets = markets.get('futures', []) + markets.get('perp', [])
            for market in futures_markets:
                if ':' in market:
                    exchange = market.split(':')[0]
                    if exchange not in stats['futures_exchanges']:
                        stats['futures_exchanges'][exchange] = {
                            'markets': [],
                            'volume': 0,
                            'trades': 0
                        }
                    stats['futures_exchanges'][exchange]['markets'].append(market)
        
        # Calculate volumes from dataset if available
        if dataset and 'spot_volume' in dataset:
            spot_vol = dataset['spot_volume']
            if isinstance(spot_vol, pd.DataFrame) and not spot_vol.empty:
                # Sum volumes by exchange
                for col in spot_vol.columns:
                    if ':' in col:
                        exchange = col.split(':')[0]
                        volume = spot_vol[col].sum()
                        if exchange in stats['spot_exchanges']:
                            stats['spot_exchanges'][exchange]['volume'] = float(volume)
                        stats['spot_volume'] += float(volume)
        
        if dataset and 'futures_volume' in dataset:
            futures_vol = dataset['futures_volume']
            if isinstance(futures_vol, pd.DataFrame) and not futures_vol.empty:
                for col in futures_vol.columns:
                    if ':' in col:
                        exchange = col.split(':')[0]
                        volume = futures_vol[col].sum()
                        if exchange in stats['futures_exchanges']:
                            stats['futures_exchanges'][exchange]['volume'] = float(volume)
                        stats['futures_volume'] += float(volume)
        
        stats['total_volume'] = stats['spot_volume'] + stats['futures_volume']
        
        return stats
    
    def _generate_html(self, symbol: str, stats: Dict) -> str:
        """Generate exchange analytics HTML"""
        
        # Prepare data for charts
        spot_exchanges = list(stats['spot_exchanges'].keys())
        spot_volumes = [stats['spot_exchanges'][ex].get('volume', 0) for ex in spot_exchanges]
        
        futures_exchanges = list(stats['futures_exchanges'].keys())
        futures_volumes = [stats['futures_exchanges'][ex].get('volume', 0) for ex in futures_exchanges]
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - Exchange Analytics</title>
    <meta charset="utf-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e222d;
            color: #d1d4dc;
            padding: 20px;
        }}
        
        h1 {{
            font-size: 24px;
            margin-bottom: 20px;
            color: #fff;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #131722;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .stat-title {{
            font-size: 14px;
            color: #787b86;
            margin-bottom: 10px;
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: 500;
            color: #fff;
        }}
        
        .exchange-table {{
            background: #131722;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .exchange-table h2 {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #fff;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #2a2e39;
            color: #787b86;
            font-weight: normal;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #2a2e39;
            color: #d1d4dc;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .chart-container {{
            background: #131722;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
        }}
        
        .exchange-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 5px;
        }}
        
        .binance {{ background: #f0b90b; color: #000; }}
        .bybit {{ background: #ff6225; color: #fff; }}
        .okx {{ background: #00d26a; color: #fff; }}
        .coinbase {{ background: #0052ff; color: #fff; }}
        .kraken {{ background: #5741d9; color: #fff; }}
    </style>
</head>
<body>
    <h1>{symbol} Exchange Analytics</h1>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-title">Total Volume</div>
            <div class="stat-value">${stats['total_volume']:,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Spot Volume</div>
            <div class="stat-value">${stats['spot_volume']:,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Futures Volume</div>
            <div class="stat-value">${stats['futures_volume']:,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Spot Exchanges</div>
            <div class="stat-value">{len(stats['spot_exchanges'])}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Futures Exchanges</div>
            <div class="stat-value">{len(stats['futures_exchanges'])}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Spot/Futures Ratio</div>
            <div class="stat-value">{(stats['spot_volume'] / max(1, stats['futures_volume'])):.2f}</div>
        </div>
    </div>
    
    <div class="exchange-table">
        <h2>Spot Exchanges</h2>
        <table>
            <thead>
                <tr>
                    <th>Exchange</th>
                    <th>Markets</th>
                    <th>Volume</th>
                    <th>Share</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f'''
                <tr>
                    <td><span class="exchange-badge {ex.lower()}">{ex}</span></td>
                    <td>{len(data['markets'])}</td>
                    <td>${data['volume']:,.0f}</td>
                    <td>{(data['volume'] / max(1, stats['spot_volume']) * 100):.1f}%</td>
                </tr>
                ''' for ex, data in stats['spot_exchanges'].items()])}
            </tbody>
        </table>
    </div>
    
    <div class="exchange-table">
        <h2>Futures Exchanges</h2>
        <table>
            <thead>
                <tr>
                    <th>Exchange</th>
                    <th>Markets</th>
                    <th>Volume</th>
                    <th>Share</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f'''
                <tr>
                    <td><span class="exchange-badge {ex.lower()}">{ex}</span></td>
                    <td>{len(data['markets'])}</td>
                    <td>${data['volume']:,.0f}</td>
                    <td>{(data['volume'] / max(1, stats['futures_volume']) * 100):.1f}%</td>
                </tr>
                ''' for ex, data in stats['futures_exchanges'].items()])}
            </tbody>
        </table>
    </div>
    
    <div class="chart-container" id="volume-chart">
        <h2>Volume Distribution</h2>
        <canvas id="volumeCanvas"></canvas>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Create volume distribution chart
        const ctx = document.getElementById('volumeCanvas').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(spot_exchanges + futures_exchanges)},
                datasets: [{{
                    data: {json.dumps(spot_volumes + futures_volumes)},
                    backgroundColor: [
                        '#f0b90b', // Binance
                        '#ff6225', // Bybit
                        '#00d26a', // OKX
                        '#0052ff', // Coinbase
                        '#5741d9', // Kraken
                        '#ff5722', // Others
                        '#4caf50',
                        '#2196f3',
                        '#ff9800',
                        '#9c27b0'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{
                            color: '#d1d4dc'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""