"""
Unified Dashboard - Single HTML file with tabbed navigation
All 3 pages (Main, Portfolio, Exchange) in ONE file
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom encoder for Pandas timestamps"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, 'timestamp'):
            return obj.timestamp()
        return super().default(obj)

class UnifiedDashboard:
    """Single HTML dashboard with all pages in tabs"""
    
    def __init__(self):
        pass
        
    def create_dashboard(self, results: Dict, dataset: Dict, 
                        executed_orders: List[Dict], output_dir: str) -> str:
        """Create single HTML with all 3 dashboard pages as tabs"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get data
        symbol = results.get('symbol', 'UNKNOWN')
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
        
        # Fix corrupted data
        if not ohlcv.empty:
            self._fix_corrupted_data(ohlcv)
        
        # Prepare chart data
        candles, volumes = self._prepare_chart_data(ohlcv)
        spot_cvd = self._get_indicator_data(dataset, 'spot_cvd')
        futures_cvd = self._get_indicator_data(dataset, 'futures_cvd')
        oi_data = self._get_oi_data(dataset)
        signal_data = self._get_signal_data(results, dataset)
        markers = self._get_trade_markers(executed_orders)
        
        # Portfolio data
        equity_curve = self._calculate_equity_curve(executed_orders)
        
        # Exchange data
        exchange_volumes = self._get_exchange_volumes(dataset)
        
        # Get metrics
        metrics = {
            'total_return': results.get('total_return', 0),
            'win_rate': results.get('win_rate', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'total_trades': len(executed_orders) if executed_orders else 0,
            'winning_trades': sum(1 for o in executed_orders if o.get('pnl', 0) > 0) if executed_orders else 0,
            'losing_trades': sum(1 for o in executed_orders if o.get('pnl', 0) <= 0) if executed_orders else 0
        }
        
        # Create unified HTML with tabs
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - Unified Dashboard</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            overflow: hidden;
        }}
        
        /* Tab Navigation */
        .tabs {{
            background: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            gap: 10px;
        }}
        
        .tab {{
            padding: 8px 16px;
            background: #2a2e39;
            border: none;
            color: #d1d4dc;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }}
        
        .tab:hover {{ background: #363a45; }}
        .tab.active {{ 
            background: #2962ff;
            color: white;
        }}
        
        /* Tab Content */
        .tab-content {{
            display: none;
            height: calc(100vh - 50px);
            overflow: auto;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        /* Header with metrics */
        .header {{
            background: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .metrics {{
            display: flex;
            gap: 20px;
        }}
        
        .metric {{
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 16px;
            font-weight: bold;
            color: #26a69a;
        }}
        
        .metric-value.negative {{
            color: #ef5350;
        }}
        
        .metric-label {{
            font-size: 11px;
            color: #787b86;
            margin-top: 2px;
        }}
        
        /* Charts */
        .charts-container {{
            height: calc(100vh - 100px);
            display: flex;
            flex-direction: column;
        }}
        
        .chart-pane {{
            background: #1e222d;
            border-bottom: 1px solid #2a2e39;
            position: relative;
        }}
        
        #main-chart {{ height: 40%; }}
        #spot-cvd-chart {{ height: 15%; }}
        #futures-cvd-chart {{ height: 15%; }}
        #oi-chart {{ height: 15%; }}
        #signal-chart {{ height: 15%; }}
        
        .pane-title {{
            position: absolute;
            top: 5px;
            left: 10px;
            font-size: 11px;
            color: #787b86;
            z-index: 10;
            background: #1e222d;
            padding: 2px 5px;
        }}
        
        /* Portfolio Page */
        .portfolio-container {{
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
            font-size: 24px;
            font-weight: bold;
            color: #26a69a;
            margin-bottom: 5px;
        }}
        
        .stat-value.negative {{ color: #ef5350; }}
        
        #equity-chart {{
            height: 400px;
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
        }}
        
        /* Exchange Page */
        .exchange-container {{
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
    </style>
</head>
<body>
    <!-- Tab Navigation -->
    <div class="tabs">
        <button class="tab active" onclick="showTab('main')">📊 Trading</button>
        <button class="tab" onclick="showTab('portfolio')">💼 Portfolio</button>
        <button class="tab" onclick="showTab('exchange')">🏛️ Exchanges</button>
    </div>
    
    <!-- Main Trading Tab -->
    <div id="main-tab" class="tab-content active">
        <div class="header">
            <h1>{symbol} - Strategy Backtest</h1>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value {'' if metrics['total_return'] >= 0 else 'negative'}">{metrics['total_return']:.2f}%</div>
                    <div class="metric-label">Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['win_rate']:.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                    <div class="metric-label">Sharpe</div>
                </div>
                <div class="metric">
                    <div class="metric-value negative">{metrics['max_drawdown']:.2f}%</div>
                    <div class="metric-label">Max DD</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['total_trades']}</div>
                    <div class="metric-label">Trades</div>
                </div>
            </div>
        </div>
        
        <div class="charts-container">
            <div id="main-chart" class="chart-pane">
                <div class="pane-title">Price & Volume</div>
            </div>
            <div id="spot-cvd-chart" class="chart-pane">
                <div class="pane-title">Spot CVD</div>
            </div>
            <div id="futures-cvd-chart" class="chart-pane">
                <div class="pane-title">Futures CVD</div>
            </div>
            <div id="oi-chart" class="chart-pane">
                <div class="pane-title">Open Interest</div>
            </div>
            <div id="signal-chart" class="chart-pane">
                <div class="pane-title">Strategy Signals</div>
            </div>
        </div>
    </div>
    
    <!-- Portfolio Tab -->
    <div id="portfolio-tab" class="tab-content">
        <div class="portfolio-container">
            <h1>{symbol} Portfolio Analytics</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{metrics['total_trades']}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metrics['winning_trades']}</div>
                    <div class="metric-label">Winning Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value negative">{metrics['losing_trades']}</div>
                    <div class="metric-label">Losing Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{metrics['win_rate']:.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
            
            <div id="equity-chart">
                <h3>Equity Curve</h3>
            </div>
        </div>
    </div>
    
    <!-- Exchange Tab -->
    <div id="exchange-tab" class="tab-content">
        <div class="exchange-container">
            <h1>{symbol} Exchange Analytics</h1>
            
            <div class="exchange-grid">
                {self._create_exchange_cards(exchange_volumes)}
            </div>
            
            <div id="volume-chart" style="height: 400px; background: #1e222d; border-radius: 8px; padding: 20px;">
                <h3>Volume Distribution</h3>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        // Tab switching
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Initialize charts if switching to main tab
            if (tabName === 'main' && !window.mainChartInitialized) {{
                initMainCharts();
                window.mainChartInitialized = true;
            }} else if (tabName === 'portfolio' && !window.portfolioChartInitialized) {{
                initPortfolioChart();
                window.portfolioChartInitialized = true;
            }} else if (tabName === 'exchange' && !window.exchangeChartInitialized) {{
                initExchangeChart();
                window.exchangeChartInitialized = true;
            }}
        }}
        
        // Chart data
        const candleData = {json.dumps(candles, cls=DateTimeEncoder)};
        const volumeData = {json.dumps(volumes, cls=DateTimeEncoder)};
        const spotCvdData = {json.dumps(spot_cvd, cls=DateTimeEncoder)};
        const futuresCvdData = {json.dumps(futures_cvd, cls=DateTimeEncoder)};
        const oiData = {json.dumps(oi_data, cls=DateTimeEncoder)};
        const signalData = {json.dumps(signal_data, cls=DateTimeEncoder)};
        const markers = {json.dumps(markers, cls=DateTimeEncoder)};
        const equityData = {json.dumps(equity_curve, cls=DateTimeEncoder)};
        const exchangeVolumes = {json.dumps(exchange_volumes, cls=DateTimeEncoder)};
        
        function initMainCharts() {{
            const chartOptions = {{
                layout: {{
                    background: {{ type: 'solid', color: '#1e222d' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#2a2e39' }},
                    horzLines: {{ color: '#2a2e39' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: '#2a2e39',
                }},
                timeScale: {{
                    borderColor: '#2a2e39',
                    timeVisible: true,
                    secondsVisible: false,
                }}
            }};
            
            // Main chart
            const mainContainer = document.getElementById('main-chart');
            const mainChart = LightweightCharts.createChart(mainContainer, {{
                ...chartOptions,
                width: mainContainer.clientWidth,
                height: mainContainer.clientHeight
            }});
            
            // Add candlesticks
            const candleSeries = mainChart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            }});
            
            if (candleData.length > 0) {{
                candleSeries.setData(candleData);
                if (markers.length > 0) {{
                    candleSeries.setMarkers(markers);
                }}
            }}
            
            // Add volume
            const volumeSeries = mainChart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '',
                scaleMargins: {{
                    top: 0.7,
                    bottom: 0,
                }},
            }});
            
            if (volumeData.length > 0) {{
                volumeSeries.setData(volumeData);
            }}
            
            // Store charts for syncing
            const charts = [mainChart];
            
            // Create other indicator charts
            const createIndicatorChart = (containerId, data, color, title) => {{
                const container = document.getElementById(containerId);
                const chart = LightweightCharts.createChart(container, {{
                    ...chartOptions,
                    width: container.clientWidth,
                    height: container.clientHeight
                }});
                charts.push(chart);
                
                if (data && data.length > 0) {{
                    const series = chart.addLineSeries({{
                        color: color,
                        lineWidth: 2,
                    }});
                    series.setData(data);
                }}
                
                return chart;
            }};
            
            createIndicatorChart('spot-cvd-chart', spotCvdData, '#2962ff', 'Spot CVD');
            createIndicatorChart('futures-cvd-chart', futuresCvdData, '#ff6b6b', 'Futures CVD');
            createIndicatorChart('oi-chart', oiData, '#ff9800', 'Open Interest');
            createIndicatorChart('signal-chart', signalData, '#4caf50', 'Signals');
            
            // Sync all charts
            charts.forEach((chart, idx) => {{
                chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    charts.forEach((otherChart, otherIdx) => {{
                        if (idx !== otherIdx) {{
                            otherChart.timeScale().setVisibleLogicalRange(range);
                        }}
                    }});
                }});
            }});
            
            mainChart.timeScale().fitContent();
        }}
        
        function initPortfolioChart() {{
            const container = document.getElementById('equity-chart');
            const chart = LightweightCharts.createChart(container, {{
                width: container.clientWidth - 40,
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
        }}
        
        function initExchangeChart() {{
            const container = document.getElementById('volume-chart');
            const chart = LightweightCharts.createChart(container, {{
                width: container.clientWidth - 40,
                height: 360,
                layout: {{
                    background: {{ type: 'solid', color: '#1e222d' }},
                    textColor: '#d1d4dc',
                }}
            }});
            
            const volumeSeries = chart.addHistogramSeries({{
                color: '#2962ff',
            }});
            
            const volumeChartData = Object.entries(exchangeVolumes).map((entry, idx) => ({{
                time: Date.now() / 1000 + idx * 86400,
                value: entry[1],
                color: idx === 0 ? '#ffd700' : idx === 1 ? '#ff9800' : idx === 2 ? '#4caf50' : '#2196f3'
            }}));
            
            if (volumeChartData.length > 0) {{
                volumeSeries.setData(volumeChartData);
            }}
        }}
        
        // Initialize main tab on load
        window.addEventListener('DOMContentLoaded', () => {{
            initMainCharts();
            window.mainChartInitialized = true;
        }});
    </script>
</body>
</html>"""
        
        # Save single HTML file
        dashboard_path = output_dir / "dashboard.html"
        dashboard_path.write_text(html)
        
        logger.info(f"✅ Unified dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def _fix_corrupted_data(self, ohlcv):
        """Fix corrupted OHLC data"""
        if ohlcv.empty:
            return
            
        price_cols = ['open', 'high', 'low', 'close']
        all_prices = []
        for col in price_cols:
            if col in ohlcv.columns:
                all_prices.extend(ohlcv[col].dropna().tolist())
        
        if all_prices:
            price_median = pd.Series(all_prices).median()
            price_std = pd.Series(all_prices).std()
            
            min_reasonable = max(0.01, price_median - 3 * price_std)
            max_reasonable = price_median + 3 * price_std
            
            for col in price_cols:
                if col in ohlcv.columns:
                    mask = (ohlcv[col] < min_reasonable) | (ohlcv[col] > max_reasonable)
                    if mask.any():
                        ohlcv.loc[mask, col] = np.nan
                        ohlcv[col] = ohlcv[col].interpolate(method='linear')
                        ohlcv[col] = ohlcv[col].fillna(method='ffill')
                        ohlcv[col] = ohlcv[col].fillna(method='bfill')
            
            # Fix OHLC relationships
            ohlcv['low'] = ohlcv[['low', 'open', 'close']].min(axis=1)
            ohlcv['high'] = ohlcv[['high', 'open', 'close']].max(axis=1)
    
    def _prepare_chart_data(self, ohlcv):
        """Prepare candlestick and volume data"""
        candles = []
        volumes = []
        
        if not ohlcv.empty:
            step = max(1, len(ohlcv) // 3000)
            sampled = ohlcv.iloc[::step] if step > 1 else ohlcv
            
            for idx, row in sampled.iterrows():
                ts = int(idx.timestamp())
                candles.append({
                    'time': ts,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
                volumes.append({
                    'time': ts,
                    'value': float(row.get('volume', 0)),
                    'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                })
        
        return candles, volumes
    
    def _get_indicator_data(self, dataset, field):
        """Get indicator data from dataset"""
        data = []
        if field in dataset:
            series = dataset[field]
            if isinstance(series, pd.Series) and not series.empty:
                step = max(1, len(series) // 2000)
                for idx, val in series.iloc[::step].items():
                    data.append({
                        'time': int(idx.timestamp()),
                        'value': float(val)
                    })
        return data
    
    def _get_oi_data(self, dataset):
        """Get open interest data"""
        for field in ['open_interest', 'oi', 'total_oi']:
            if field in dataset:
                oi = dataset[field]
                if isinstance(oi, (pd.Series, pd.DataFrame)) and not oi.empty:
                    if isinstance(oi, pd.DataFrame):
                        oi = oi.iloc[:, 0]
                    return self._get_indicator_data({'open_interest': oi}, 'open_interest')
        return []
    
    def _get_signal_data(self, results, dataset):
        """Get strategy signal data"""
        data = []
        
        # Check results for squeeze_scores
        if 'squeeze_scores' in results:
            scores = results['squeeze_scores']
            if isinstance(scores, dict) and 'timestamps' in scores and 'scores' in scores:
                timestamps = scores['timestamps']
                score_values = scores['scores']
                if timestamps and score_values:
                    step = max(1, len(timestamps) // 2000)
                    for i in range(0, len(timestamps), step):
                        ts = timestamps[i]
                        if hasattr(ts, 'timestamp'):
                            unix_time = int(ts.timestamp())
                        else:
                            unix_time = int(ts)
                        data.append({
                            'time': unix_time,
                            'value': float(score_values[i])
                        })
        
        return data
    
    def _get_trade_markers(self, executed_orders):
        """Get trade markers for chart"""
        markers = []
        if executed_orders:
            for order in executed_orders:
                if isinstance(order, dict) and 'timestamp' in order:
                    ts = order['timestamp']
                    if hasattr(ts, 'timestamp'):
                        ts = int(ts.timestamp())
                    elif isinstance(ts, str):
                        ts = int(pd.Timestamp(ts).timestamp())
                    else:
                        ts = int(ts)
                    
                    markers.append({
                        'time': ts,
                        'position': 'belowBar' if order.get('side') == 'buy' else 'aboveBar',
                        'color': '#26a69a' if order.get('side') == 'buy' else '#ef5350',
                        'shape': 'arrowUp' if order.get('side') == 'buy' else 'arrowDown',
                        'text': order.get('side', '').upper()
                    })
        return markers
    
    def _calculate_equity_curve(self, executed_orders):
        """Calculate portfolio equity curve"""
        equity_curve = []
        if executed_orders:
            balance = 10000
            for order in executed_orders:
                if isinstance(order, dict):
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
        return equity_curve
    
    def _get_exchange_volumes(self, dataset):
        """Get exchange volume distribution"""
        volumes = {}
        if 'spot_volume' in dataset:
            vol_df = dataset['spot_volume']
            if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
                for col in vol_df.columns:
                    if 'BINANCE' in col:
                        volumes['Binance'] = float(vol_df[col].sum())
                    elif 'BYBIT' in col:
                        volumes['Bybit'] = float(vol_df[col].sum())
                    elif 'OKX' in col:
                        volumes['OKX'] = float(vol_df[col].sum())
                    elif 'COINBASE' in col:
                        volumes['Coinbase'] = float(vol_df[col].sum())
        return volumes
    
    def _create_exchange_cards(self, volumes):
        """Create exchange volume cards HTML"""
        if not volumes:
            return "<p>No exchange volume data available</p>"
            
        cards = []
        total = sum(volumes.values())
        colors = {
            'Binance': '#ffd700',
            'Bybit': '#ff9800',
            'OKX': '#4caf50',
            'Coinbase': '#2196f3'
        }
        
        for exchange, volume in volumes.items():
            pct = (volume / total * 100) if total > 0 else 0
            color = colors.get(exchange, '#2962ff')
            cards.append(f"""
            <div class="exchange-card">
                <div>{exchange}</div>
                <div style="height: 30px; background: {color}; width: {pct}%; border-radius: 4px; margin: 10px 0;"></div>
                <div>Volume: {volume:,.0f}</div>
                <div>{pct:.1f}%</div>
            </div>
            """)
        
        return '\n'.join(cards)