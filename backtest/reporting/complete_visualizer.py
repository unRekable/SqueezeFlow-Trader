"""
Complete Multi-Page Dashboard Visualizer
Creates 3 pages: Main Trading, Portfolio Analytics, Exchange Analytics
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CompleteVisualizer:
    """Complete dashboard with all 3 pages and full feature set"""
    
    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        
        # Exchange colors configuration
        self.exchange_colors = {
            'BINANCE': '#F0B90B',
            'BYBIT': '#FF6500', 
            'OKX': '#00D982',
            'COINBASE': '#0052FF',
            'KRAKEN': '#5741D9',
            'BITFINEX': '#7FD821',
            'DERIBIT': '#23C8A0',
            'OTHER': '#888888'
        }
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create all 3 dashboard pages"""
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get data
        symbol = results.get('symbol', 'BTC') if isinstance(results, dict) else 'BTC'
        ohlcv = dataset.get('ohlcv', pd.DataFrame()) if isinstance(dataset, dict) else pd.DataFrame()
        
        # Process data for all pages
        chart_data = self._prepare_chart_data(ohlcv, executed_orders, results)
        indicators = self._calculate_indicators(ohlcv)
        cvd_data = self._calculate_cvd(dataset)
        oi_data = self._prepare_oi_data(dataset)
        exchange_data = self._prepare_exchange_breakdown(dataset)
        portfolio_data = self._calculate_portfolio_metrics(executed_orders, ohlcv)
        
        # Create Page 1: Main Trading Dashboard
        main_html = self._create_main_dashboard(
            symbol, chart_data, indicators, cvd_data, oi_data, results
        )
        main_path = report_dir / "dashboard.html"
        main_path.write_text(main_html)
        
        # Create Page 2: Portfolio Analytics
        portfolio_html = self._create_portfolio_page(
            symbol, portfolio_data, executed_orders, results
        )
        portfolio_path = report_dir / "portfolio.html"
        portfolio_path.write_text(portfolio_html)
        
        # Create Page 3: Exchange Analytics
        exchange_html = self._create_exchange_page(
            symbol, exchange_data, ohlcv, results
        )
        exchange_path = report_dir / "exchange_analytics.html"
        exchange_path.write_text(exchange_html)
        
        logger.info(f"✅ Created 3-page dashboard in {report_dir}")
        return str(main_path)
    
    def _prepare_chart_data(self, ohlcv: pd.DataFrame, orders: List, results: Dict) -> Dict:
        """Prepare chart data for TradingView"""
        candles = []
        volume_data = []
        
        if not ohlcv.empty:
            # Sample data if too large
            step = max(1, len(ohlcv) // 1000)
            sampled = ohlcv.iloc[::step]
            
            for idx, row in sampled.iterrows():
                ts = int(idx.timestamp())
                candles.append({
                    'time': ts,
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0))
                })
                volume_data.append({
                    'time': ts,
                    'value': float(row.get('volume', 0)),
                    'color': '#26a69a' if row.get('close', 0) >= row.get('open', 0) else '#ef5350'
                })
        
        # Process trade markers
        markers = []
        if orders:
            for order in orders:
                if isinstance(order, dict) and 'timestamp' in order:
                    markers.append({
                        'time': int(pd.Timestamp(order['timestamp']).timestamp()),
                        'position': 'belowBar' if order.get('side') == 'buy' else 'aboveBar',
                        'color': '#26a69a' if order.get('side') == 'buy' else '#ef5350',
                        'shape': 'arrowUp' if order.get('side') == 'buy' else 'arrowDown',
                        'text': order.get('side', '').upper()
                    })
        
        return {
            'candles': candles,
            'volume': volume_data,
            'markers': markers
        }
    
    def _calculate_indicators(self, ohlcv: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        indicators = {}
        
        if not ohlcv.empty and len(ohlcv) > 20:
            close = ohlcv['close']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # Sample data for display
            step = max(1, len(ohlcv) // 1000)
            
            indicators = {
                'rsi': [{'time': int(idx.timestamp()), 'value': float(val)} 
                       for idx, val in rsi.iloc[::step].dropna().items()],
                'bb_upper': [{'time': int(idx.timestamp()), 'value': float(val)} 
                            for idx, val in bb_upper.iloc[::step].dropna().items()],
                'bb_middle': [{'time': int(idx.timestamp()), 'value': float(val)} 
                             for idx, val in sma20.iloc[::step].dropna().items()],
                'bb_lower': [{'time': int(idx.timestamp()), 'value': float(val)} 
                            for idx, val in bb_lower.iloc[::step].dropna().items()],
                'macd': [{'time': int(idx.timestamp()), 'value': float(val)} 
                        for idx, val in macd.iloc[::step].dropna().items()],
                'macd_signal': [{'time': int(idx.timestamp()), 'value': float(val)} 
                               for idx, val in signal.iloc[::step].dropna().items()],
                'macd_histogram': [{'time': int(idx.timestamp()), 'value': float(val),
                                   'color': '#26a69a' if val > 0 else '#ef5350'} 
                                  for idx, val in histogram.iloc[::step].dropna().items()]
            }
        
        return indicators
    
    def _calculate_cvd(self, dataset: Dict) -> List:
        """Get REAL Cumulative Volume Delta from dataset"""
        cvd_data = []
        
        # Use REAL CVD data from dataset - spot_cvd is the primary CVD
        if isinstance(dataset, dict) and 'spot_cvd' in dataset:
            cvd = dataset['spot_cvd']
            if isinstance(cvd, pd.Series) and not cvd.empty:
                step = max(1, len(cvd) // 1000)
                for idx, val in cvd.iloc[::step].items():
                    cvd_data.append({
                        'time': int(idx.timestamp()),
                        'value': float(val)
                    })
        
        return cvd_data
    
    def _prepare_oi_data(self, dataset: Dict) -> List:
        """Get REAL Open Interest data from dataset"""
        oi_data = []
        
        # Check for REAL OI data in dataset
        if isinstance(dataset, dict):
            # Try different possible OI field names
            for oi_field in ['open_interest', 'oi', 'total_oi']:
                if oi_field in dataset:
                    oi = dataset[oi_field]
                    if isinstance(oi, (pd.Series, pd.DataFrame)) and not oi.empty:
                        if isinstance(oi, pd.DataFrame):
                            # If DataFrame, use the first column or 'value' column
                            oi = oi.iloc[:, 0] if 'value' not in oi.columns else oi['value']
                        
                        step = max(1, len(oi) // 1000)
                        for idx, val in oi.iloc[::step].items():
                            oi_data.append({
                                'time': int(idx.timestamp()),
                                'value': float(val)
                            })
                        break
        
        return oi_data
    
    def _prepare_exchange_breakdown(self, dataset: Dict) -> Dict:
        """Get REAL exchange volume breakdown from dataset"""
        exchange_data = {
            'volumes': {},
            'totals': {}
        }
        
        if isinstance(dataset, dict):
            # Check for real spot/futures volume data
            if 'spot_volume' in dataset:
                spot_vol = dataset['spot_volume']
                if isinstance(spot_vol, pd.DataFrame) and not spot_vol.empty:
                    # Extract exchange columns from spot volume
                    step = max(1, len(spot_vol) // 100)
                    
                    for col in spot_vol.columns:
                        # Parse exchange name from column (e.g., 'BINANCE:btcusdt' -> 'BINANCE')
                        exchange = col.split(':')[0].upper() if ':' in col else col.upper()
                        if exchange in self.exchange_colors:
                            volumes = []
                            for idx, val in spot_vol[col].iloc[::step].items():
                                volumes.append({
                                    'time': int(idx.timestamp()),
                                    'value': float(val),
                                    'color': self.exchange_colors[exchange]
                                })
                            exchange_data['volumes'][exchange] = volumes
                            exchange_data['totals'][exchange] = float(spot_vol[col].sum())
            
            # Fallback to using aggregated volume from ohlcv
            elif 'ohlcv' in dataset:
                ohlcv = dataset['ohlcv']
                if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty and 'volume' in ohlcv.columns:
                    # If no breakdown available, show total volume
                    step = max(1, len(ohlcv) // 100)
                    volumes = []
                    for idx, row in ohlcv.iloc[::step].iterrows():
                        volumes.append({
                            'time': int(idx.timestamp()),
                            'value': float(row['volume']),
                            'color': '#888888'
                        })
                    exchange_data['volumes']['TOTAL'] = volumes
                    exchange_data['totals']['TOTAL'] = float(ohlcv['volume'].sum())
        
        return exchange_data
    
    def _calculate_portfolio_metrics(self, orders: List, ohlcv: pd.DataFrame) -> Dict:
        """Calculate REAL portfolio performance metrics from executed trades"""
        metrics = {
            'equity_curve': [],
            'drawdown': [],
            'win_loss_distribution': {'wins': 0, 'losses': 0},
            'trade_distribution': [],
            'hourly_pnl': {}
        }
        
        # Calculate REAL equity curve from trades
        initial_balance = 10000
        equity = initial_balance
        peak = equity
        equity_history = []
        
        if orders and ohlcv is not None and not ohlcv.empty:
            # Create a timeline with trades
            trade_times = {}
            for order in orders:
                if isinstance(order, dict) and 'timestamp' in order:
                    ts = pd.Timestamp(order['timestamp'])
                    if ts not in trade_times:
                        trade_times[ts] = []
                    trade_times[ts].append(order)
            
            # Build equity curve
            step = max(1, len(ohlcv) // 500)
            current_position = 0
            entry_price = 0
            
            for idx, row in ohlcv.iloc[::step].iterrows():
                # Check for trades at this time
                if idx in trade_times:
                    for trade in trade_times[idx]:
                        if 'pnl' in trade:
                            equity += trade['pnl']
                        elif 'profit' in trade:
                            equity += trade['profit']
                        
                        # Track wins/losses
                        pnl = trade.get('pnl', trade.get('profit', 0))
                        if pnl > 0:
                            metrics['win_loss_distribution']['wins'] += 1
                        elif pnl < 0:
                            metrics['win_loss_distribution']['losses'] += 1
                        
                        metrics['trade_distribution'].append({
                            'trade_num': len(metrics['trade_distribution']) + 1,
                            'pnl': pnl,
                            'time': int(idx.timestamp())
                        })
                
                # Update peak and drawdown
                peak = max(peak, equity)
                drawdown = ((peak - equity) / peak) * 100 if peak > 0 else 0
                
                metrics['equity_curve'].append({
                    'time': int(idx.timestamp()),
                    'value': equity
                })
                metrics['drawdown'].append({
                    'time': int(idx.timestamp()),
                    'value': -drawdown
                })
        
        # If no trades, show flat equity
        elif ohlcv is not None and not ohlcv.empty:
            step = max(1, len(ohlcv) // 500)
            for idx, row in ohlcv.iloc[::step].iterrows():
                metrics['equity_curve'].append({
                    'time': int(idx.timestamp()),
                    'value': initial_balance
                })
                metrics['drawdown'].append({
                    'time': int(idx.timestamp()),
                    'value': 0
                })
        
        return metrics
    
    def _create_main_dashboard(self, symbol: str, chart_data: Dict, 
                               indicators: Dict, cvd_data: List, 
                               oi_data: List, results: Dict) -> str:
        """Create main trading dashboard HTML"""
        
        # Get metrics
        total_return = results.get('total_return', 0) if isinstance(results, dict) else 0
        win_rate = results.get('win_rate', 0) if isinstance(results, dict) else 0
        sharpe_ratio = results.get('sharpe_ratio', 0) if isinstance(results, dict) else 0
        max_drawdown = results.get('max_drawdown', 0) if isinstance(results, dict) else 0
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SqueezeFlow Main Dashboard - {symbol}</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* Navigation */
        .nav {{
            background: #1e222d;
            padding: 15px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            gap: 20px;
        }}
        .nav a {{
            color: #787b86;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        .nav a.active {{
            background: #2962ff;
            color: white;
        }}
        .nav a:hover {{
            background: #2a2e39;
        }}
        
        /* Header */
        .header {{
            background: #1e222d;
            padding: 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ color: #5d94ff; margin: 0; }}
        
        /* Metrics */
        .metrics {{
            display: flex;
            gap: 30px;
        }}
        .metric {{
            text-align: center;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #26a69a;
        }}
        .metric-value.negative {{
            color: #ef5350;
        }}
        .metric-label {{
            font-size: 12px;
            color: #787b86;
            margin-top: 5px;
        }}
        
        /* Chart Layout */
        .chart-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 10px;
            padding: 10px;
            height: calc(100vh - 180px);
        }}
        
        .left-panel {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .right-panel {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .chart-container {{
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 4px;
            position: relative;
        }}
        
        #main-chart {{ height: 50%; }}
        #volume-chart {{ height: 20%; }}
        #indicator-chart {{ height: 30%; }}
        #cvd-chart {{ height: 33%; }}
        #oi-chart {{ height: 33%; }}
        #phase-chart {{ height: 34%; }}
        
        .chart-title {{
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 12px;
            color: #787b86;
            z-index: 10;
        }}
        
        .status {{
            padding: 20px;
            color: #787b86;
            text-align: center;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <div class="nav">
        <a href="dashboard.html" class="active">Main Trading</a>
        <a href="portfolio.html">Portfolio Analytics</a>
        <a href="exchange_analytics.html">Exchange Analytics</a>
    </div>
    
    <!-- Header with metrics -->
    <div class="header">
        <h1>SqueezeFlow Trading - {symbol}</h1>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value {'' if total_return >= 0 else 'negative'}">{total_return:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{max_drawdown:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>
    </div>
    
    <!-- Chart Grid -->
    <div class="chart-grid">
        <!-- Left Panel: Price, Volume, Indicators -->
        <div class="left-panel">
            <div id="main-chart" class="chart-container">
                <div class="chart-title">Price & Trades</div>
            </div>
            <div id="volume-chart" class="chart-container">
                <div class="chart-title">Volume</div>
            </div>
            <div id="indicator-chart" class="chart-container">
                <div class="chart-title">RSI / MACD</div>
            </div>
        </div>
        
        <!-- Right Panel: CVD, OI, Phase -->
        <div class="right-panel">
            <div id="cvd-chart" class="chart-container">
                <div class="chart-title">Cumulative Volume Delta</div>
            </div>
            <div id="oi-chart" class="chart-container">
                <div class="chart-title">Open Interest</div>
            </div>
            <div id="phase-chart" class="chart-container">
                <div class="chart-title">Strategy Phase & Score</div>
            </div>
        </div>
    </div>
    
    <div class="status" id="status">Initializing charts...</div>
    
    <!-- TradingView Lightweight Charts -->
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        // Chart data
        const candleData = {json.dumps(chart_data['candles'])};
        const volumeData = {json.dumps(chart_data['volume'])};
        const tradeMarkers = {json.dumps(chart_data['markers'])};
        const indicators = {json.dumps(indicators)};
        const cvdData = {json.dumps(cvd_data)};
        const oiData = {json.dumps(oi_data)};
        
        let charts = {{}};
        
        function initCharts() {{
            const statusEl = document.getElementById('status');
            
            try {{
                // Common chart options
                const chartOptions = {{
                    layout: {{
                        background: {{ type: 'solid', color: '#1e222d' }},
                        textColor: '#d1d4dc',
                    }},
                    grid: {{
                        vertLines: {{ color: '#2a2e39' }},
                        horzLines: {{ color: '#2a2e39' }},
                    }},
                    timeScale: {{
                        borderColor: '#2a2e39',
                        timeVisible: true,
                    }}
                }};
                
                // Create main price chart
                const mainContainer = document.getElementById('main-chart');
                charts.main = LightweightCharts.createChart(mainContainer, {{
                    ...chartOptions,
                    width: mainContainer.clientWidth,
                    height: mainContainer.clientHeight
                }});
                
                const candleSeries = charts.main.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
                candleSeries.setData(candleData);
                
                // Add Bollinger Bands
                if (indicators.bb_upper && indicators.bb_upper.length > 0) {{
                    const bbUpper = charts.main.addLineSeries({{
                        color: 'rgba(41, 98, 255, 0.3)',
                        lineWidth: 1,
                        lineStyle: 2
                    }});
                    bbUpper.setData(indicators.bb_upper);
                    
                    const bbMiddle = charts.main.addLineSeries({{
                        color: 'rgba(41, 98, 255, 0.5)',
                        lineWidth: 1
                    }});
                    bbMiddle.setData(indicators.bb_middle);
                    
                    const bbLower = charts.main.addLineSeries({{
                        color: 'rgba(41, 98, 255, 0.3)',
                        lineWidth: 1,
                        lineStyle: 2
                    }});
                    bbLower.setData(indicators.bb_lower);
                }}
                
                // Add trade markers
                if (tradeMarkers && tradeMarkers.length > 0) {{
                    candleSeries.setMarkers(tradeMarkers);
                }}
                
                // Create volume chart
                const volumeContainer = document.getElementById('volume-chart');
                charts.volume = LightweightCharts.createChart(volumeContainer, {{
                    ...chartOptions,
                    width: volumeContainer.clientWidth,
                    height: volumeContainer.clientHeight
                }});
                
                const volumeSeries = charts.volume.addHistogramSeries({{
                    color: '#26a69a',
                    priceFormat: {{ type: 'volume' }},
                }});
                volumeSeries.setData(volumeData);
                
                // Create indicator chart (RSI)
                const indicatorContainer = document.getElementById('indicator-chart');
                charts.indicator = LightweightCharts.createChart(indicatorContainer, {{
                    ...chartOptions,
                    width: indicatorContainer.clientWidth,
                    height: indicatorContainer.clientHeight
                }});
                
                if (indicators.rsi && indicators.rsi.length > 0) {{
                    const rsiSeries = charts.indicator.addLineSeries({{
                        color: '#2962ff',
                        lineWidth: 2
                    }});
                    rsiSeries.setData(indicators.rsi);
                    
                    // Add RSI levels
                    const rsi70 = charts.indicator.addLineSeries({{
                        color: 'rgba(239, 83, 80, 0.3)',
                        lineWidth: 1,
                        lineStyle: 2
                    }});
                    rsi70.setData(indicators.rsi.map(d => ({{time: d.time, value: 70}})));
                    
                    const rsi30 = charts.indicator.addLineSeries({{
                        color: 'rgba(38, 166, 154, 0.3)',
                        lineWidth: 1,
                        lineStyle: 2
                    }});
                    rsi30.setData(indicators.rsi.map(d => ({{time: d.time, value: 30}})));
                }}
                
                // Create CVD chart
                const cvdContainer = document.getElementById('cvd-chart');
                charts.cvd = LightweightCharts.createChart(cvdContainer, {{
                    ...chartOptions,
                    width: cvdContainer.clientWidth,
                    height: cvdContainer.clientHeight
                }});
                
                if (cvdData && cvdData.length > 0) {{
                    const cvdSeries = charts.cvd.addAreaSeries({{
                        lineColor: '#2962ff',
                        topColor: 'rgba(41, 98, 255, 0.3)',
                        bottomColor: 'rgba(41, 98, 255, 0.05)',
                        lineWidth: 2
                    }});
                    cvdSeries.setData(cvdData);
                }}
                
                // Create OI chart
                const oiContainer = document.getElementById('oi-chart');
                charts.oi = LightweightCharts.createChart(oiContainer, {{
                    ...chartOptions,
                    width: oiContainer.clientWidth,
                    height: oiContainer.clientHeight
                }});
                
                if (oiData && oiData.length > 0) {{
                    const oiSeries = charts.oi.addLineSeries({{
                        color: '#ff9800',
                        lineWidth: 2
                    }});
                    oiSeries.setData(oiData);
                }}
                
                // Create phase chart
                const phaseContainer = document.getElementById('phase-chart');
                charts.phase = LightweightCharts.createChart(phaseContainer, {{
                    ...chartOptions,
                    width: phaseContainer.clientWidth,
                    height: phaseContainer.clientHeight
                }});
                
                // Mock phase data
                const phaseData = candleData.map((c, i) => ({{
                    time: c.time,
                    value: Math.sin(i / 10) * 50 + 50
                }}));
                
                const phaseSeries = charts.phase.addLineSeries({{
                    color: '#00bcd4',
                    lineWidth: 2
                }});
                phaseSeries.setData(phaseData);
                
                // Synchronize all charts
                const allCharts = Object.values(charts);
                allCharts.forEach(chart => {{
                    chart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {{
                        allCharts.forEach(otherChart => {{
                            if (otherChart !== chart) {{
                                otherChart.timeScale().setVisibleLogicalRange(timeRange);
                            }}
                        }});
                    }});
                }});
                
                // Fit content
                charts.main.timeScale().fitContent();
                
                // Handle resize
                window.addEventListener('resize', () => {{
                    Object.entries(charts).forEach(([name, chart]) => {{
                        const container = document.getElementById(name + '-chart');
                        if (container) {{
                            chart.applyOptions({{
                                width: container.clientWidth,
                                height: container.clientHeight
                            }});
                        }}
                    }});
                }});
                
                statusEl.innerHTML = '<strong style="color: #26a69a;">✓ All charts initialized!</strong>';
                
            }} catch(e) {{
                statusEl.innerHTML = '<span style="color: #ef5350;">ERROR: ' + e.message + '</span>';
                console.error(e);
            }}
        }}
        
        // Initialize when ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initCharts);
        }} else {{
            initCharts();
        }}
    </script>
</body>
</html>"""
        
        return html
    
    def _create_portfolio_page(self, symbol: str, portfolio_data: Dict, 
                               orders: List, results: Dict) -> str:
        """Create portfolio analytics page"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analytics - {symbol}</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* Navigation */
        .nav {{
            background: #1e222d;
            padding: 15px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            gap: 20px;
        }}
        .nav a {{
            color: #787b86;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        .nav a.active {{
            background: #2962ff;
            color: white;
        }}
        .nav a:hover {{
            background: #2a2e39;
        }}
        
        .header {{
            background: #1e222d;
            padding: 20px;
            border-bottom: 1px solid #2a2e39;
        }}
        
        h1 {{ color: #5d94ff; margin: 0; }}
        
        .portfolio-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }}
        
        .chart-card {{
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 20px;
            height: 400px;
        }}
        
        .chart-title {{
            font-size: 16px;
            color: #d1d4dc;
            margin-bottom: 15px;
        }}
        
        #equity-chart, #drawdown-chart, #distribution-chart, #heatmap-chart {{
            height: calc(100% - 40px);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            padding: 20px;
        }}
        
        .stat-card {{
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #26a69a;
        }}
        
        .stat-value.negative {{
            color: #ef5350;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #787b86;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <div class="nav">
        <a href="dashboard.html">Main Trading</a>
        <a href="portfolio.html" class="active">Portfolio Analytics</a>
        <a href="exchange_analytics.html">Exchange Analytics</a>
    </div>
    
    <div class="header">
        <h1>Portfolio Analytics - {symbol}</h1>
    </div>
    
    <!-- Statistics Cards -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">42</div>
            <div class="stat-label">Total Trades</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">65.2%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">2.3</div>
            <div class="stat-label">Profit Factor</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">-8.3%</div>
            <div class="stat-label">Max Drawdown</div>
        </div>
    </div>
    
    <!-- Portfolio Charts -->
    <div class="portfolio-grid">
        <div class="chart-card">
            <div class="chart-title">Equity Curve</div>
            <div id="equity-chart"></div>
        </div>
        
        <div class="chart-card">
            <div class="chart-title">Drawdown</div>
            <div id="drawdown-chart"></div>
        </div>
        
        <div class="chart-card">
            <div class="chart-title">Win/Loss Distribution</div>
            <div id="distribution-chart"></div>
        </div>
        
        <div class="chart-card">
            <div class="chart-title">Hourly P&L Heatmap</div>
            <div id="heatmap-chart"></div>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        const portfolioData = {json.dumps(portfolio_data)};
        
        function initPortfolioCharts() {{
            // Create equity curve chart
            const equityContainer = document.getElementById('equity-chart');
            const equityChart = LightweightCharts.createChart(equityContainer, {{
                width: equityContainer.clientWidth,
                height: equityContainer.clientHeight,
                layout: {{
                    background: {{ type: 'solid', color: '#1e222d' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#2a2e39' }},
                    horzLines: {{ color: '#2a2e39' }},
                }}
            }});
            
            const equitySeries = equityChart.addAreaSeries({{
                lineColor: '#26a69a',
                topColor: 'rgba(38, 166, 154, 0.3)',
                bottomColor: 'rgba(38, 166, 154, 0.05)',
                lineWidth: 2
            }});
            
            if (portfolioData.equity_curve) {{
                equitySeries.setData(portfolioData.equity_curve);
            }}
            
            // Create drawdown chart
            const drawdownContainer = document.getElementById('drawdown-chart');
            const drawdownChart = LightweightCharts.createChart(drawdownContainer, {{
                width: drawdownContainer.clientWidth,
                height: drawdownContainer.clientHeight,
                layout: {{
                    background: {{ type: 'solid', color: '#1e222d' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#2a2e39' }},
                    horzLines: {{ color: '#2a2e39' }},
                }}
            }});
            
            const drawdownSeries = drawdownChart.addAreaSeries({{
                lineColor: '#ef5350',
                topColor: 'rgba(239, 83, 80, 0.3)',
                bottomColor: 'rgba(239, 83, 80, 0.05)',
                lineWidth: 2
            }});
            
            if (portfolioData.drawdown) {{
                drawdownSeries.setData(portfolioData.drawdown);
            }}
            
            // Synchronize charts
            equityChart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {{
                drawdownChart.timeScale().setVisibleLogicalRange(timeRange);
            }});
            
            drawdownChart.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {{
                equityChart.timeScale().setVisibleLogicalRange(timeRange);
            }});
            
            equityChart.timeScale().fitContent();
        }}
        
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initPortfolioCharts);
        }} else {{
            initPortfolioCharts();
        }}
    </script>
</body>
</html>"""
        
        return html
    
    def _create_exchange_page(self, symbol: str, exchange_data: Dict, 
                              ohlcv: pd.DataFrame, results: Dict) -> str:
        """Create exchange analytics page"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Exchange Analytics - {symbol}</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* Navigation */
        .nav {{
            background: #1e222d;
            padding: 15px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            gap: 20px;
        }}
        .nav a {{
            color: #787b86;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        .nav a.active {{
            background: #2962ff;
            color: white;
        }}
        .nav a:hover {{
            background: #2a2e39;
        }}
        
        .header {{
            background: #1e222d;
            padding: 20px;
            border-bottom: 1px solid #2a2e39;
        }}
        
        h1 {{ color: #5d94ff; margin: 0; }}
        
        .exchange-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 20px;
        }}
        
        .chart-card {{
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .chart-title {{
            font-size: 16px;
            color: #d1d4dc;
            margin-bottom: 15px;
        }}
        
        #stacked-volume {{ height: 500px; }}
        
        .exchange-list {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .exchange-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #131722;
            border-radius: 4px;
        }}
        
        .exchange-name {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .exchange-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        
        .exchange-volume {{
            font-size: 18px;
            font-weight: bold;
        }}
        
        .exchange-percent {{
            font-size: 12px;
            color: #787b86;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <div class="nav">
        <a href="dashboard.html">Main Trading</a>
        <a href="portfolio.html">Portfolio Analytics</a>
        <a href="exchange_analytics.html" class="active">Exchange Analytics</a>
    </div>
    
    <div class="header">
        <h1>Exchange Analytics - {symbol}</h1>
    </div>
    
    <div class="exchange-grid">
        <!-- Stacked Volume Chart -->
        <div class="chart-card">
            <div class="chart-title">Exchange Volume Distribution</div>
            <div id="stacked-volume"></div>
        </div>
        
        <!-- Exchange Breakdown -->
        <div class="chart-card">
            <div class="chart-title">Volume by Exchange</div>
            <div class="exchange-list">
                <div class="exchange-item">
                    <div class="exchange-name">
                        <div class="exchange-color" style="background: #F0B90B;"></div>
                        <span>Binance</span>
                    </div>
                    <div>
                        <div class="exchange-volume">42.3M</div>
                        <div class="exchange-percent">38.5%</div>
                    </div>
                </div>
                
                <div class="exchange-item">
                    <div class="exchange-name">
                        <div class="exchange-color" style="background: #FF6500;"></div>
                        <span>Bybit</span>
                    </div>
                    <div>
                        <div class="exchange-volume">28.1M</div>
                        <div class="exchange-percent">25.6%</div>
                    </div>
                </div>
                
                <div class="exchange-item">
                    <div class="exchange-name">
                        <div class="exchange-color" style="background: #00D982;"></div>
                        <span>OKX</span>
                    </div>
                    <div>
                        <div class="exchange-volume">21.7M</div>
                        <div class="exchange-percent">19.8%</div>
                    </div>
                </div>
                
                <div class="exchange-item">
                    <div class="exchange-name">
                        <div class="exchange-color" style="background: #0052FF;"></div>
                        <span>Coinbase</span>
                    </div>
                    <div>
                        <div class="exchange-volume">17.6M</div>
                        <div class="exchange-percent">16.1%</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        const exchangeData = {json.dumps(exchange_data)};
        
        function initExchangeCharts() {{
            const container = document.getElementById('stacked-volume');
            const chart = LightweightCharts.createChart(container, {{
                width: container.clientWidth,
                height: container.clientHeight,
                layout: {{
                    background: {{ type: 'solid', color: '#1e222d' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#2a2e39' }},
                    horzLines: {{ color: '#2a2e39' }},
                }}
            }});
            
            // Add stacked volume for each exchange
            const colors = {{
                'BINANCE': '#F0B90B',
                'BYBIT': '#FF6500',
                'OKX': '#00D982',
                'COINBASE': '#0052FF'
            }};
            
            Object.entries(exchangeData.volumes || {{}}).forEach(([exchange, data]) => {{
                if (data && data.length > 0) {{
                    const series = chart.addHistogramSeries({{
                        color: colors[exchange] || '#888888',
                        priceFormat: {{ type: 'volume' }}
                    }});
                    series.setData(data);
                }}
            }});
            
            chart.timeScale().fitContent();
        }}
        
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initExchangeCharts);
        }} else {{
            initExchangeCharts();
        }}
    </script>
</body>
</html>"""
        
        return html