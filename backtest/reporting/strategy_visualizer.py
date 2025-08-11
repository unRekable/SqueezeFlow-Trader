"""
Strategy Dashboard Visualizer
ONE chart with OUR indicators in proper panes - NO technical analysis bullshit
"""

import pandas as pd
import numpy as np
import json
import logging
from json import JSONEncoder
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class DateTimeEncoder(JSONEncoder):
    """Custom encoder for Pandas timestamps"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, 'timestamp'):
            return obj.timestamp()
        return super().default(obj)

class StrategyVisualizer:
    """Dashboard with OUR strategy indicators only"""
    
    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create dashboard with strategy indicators in separate panes"""
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get REAL data
        symbol = results.get('symbol', 'UNKNOWN') if isinstance(results, dict) else 'UNKNOWN'
        ohlcv = dataset.get('ohlcv', pd.DataFrame()) if isinstance(dataset, dict) else pd.DataFrame()
        
        # Prepare all timeframe data for switching
        all_candles = {}
        all_volumes = {}
        
        # Helper function to resample OHLCV
        def resample_ohlcv(df, rule):
            if df.empty:
                return df
            try:
                return df.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            except:
                return df
        
        # Determine base timeframe from data
        base_timeframe = '1s'
        if not ohlcv.empty:
            total_seconds = (ohlcv.index[-1] - ohlcv.index[0]).total_seconds()
            num_candles = len(ohlcv)
            if num_candles > 0:
                avg_gap = total_seconds / num_candles
                if avg_gap < 2:
                    base_timeframe = '1s'
                elif avg_gap < 120:
                    base_timeframe = '1m'
                else:
                    base_timeframe = '5m'
        
        # Generate data for each timeframe
        timeframes = {
            '1s': None,  # Raw data if 1s
            '1m': '1min',
            '5m': '5min', 
            '15m': '15min',
            '1h': '1h'
        }
        
        for tf_label, tf_rule in timeframes.items():
            if tf_rule is None and base_timeframe == '1s':
                # Use raw 1s data
                tf_data = ohlcv
            elif tf_rule:
                # Resample to this timeframe
                tf_data = resample_ohlcv(ohlcv, tf_rule)
            else:
                # Skip if we don't have 1s data
                tf_data = pd.DataFrame()
            
            # Convert to chart format
            candles = []
            volumes = []
            
            if not tf_data.empty:
                # Limit points for performance
                step = max(1, len(tf_data) // 3000)
                sampled = tf_data.iloc[::step] if step > 1 else tf_data
                
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
            
            all_candles[tf_label] = candles
            all_volumes[tf_label] = volumes
        
        # Default to 1m if available, else use base
        default_timeframe = '1m' if all_candles.get('1m') else base_timeframe
        
        # Spot CVD - OUR indicator from dataset
        spot_cvd_data = []
        if isinstance(dataset, dict) and 'spot_cvd' in dataset:
            cvd = dataset['spot_cvd']
            if isinstance(cvd, pd.Series) and not cvd.empty:
                step = max(1, len(cvd) // 2000)
                for idx, val in cvd.iloc[::step].items():
                    spot_cvd_data.append({
                        'time': int(idx.timestamp()),
                        'value': float(val)
                    })
        
        # Futures/Perp CVD - OUR indicator from dataset
        futures_cvd_data = []
        if isinstance(dataset, dict) and 'futures_cvd' in dataset:
            cvd = dataset['futures_cvd']
            if isinstance(cvd, pd.Series) and not cvd.empty:
                step = max(1, len(cvd) // 2000)
                for idx, val in cvd.iloc[::step].items():
                    futures_cvd_data.append({
                        'time': int(idx.timestamp()),
                        'value': float(val)
                    })
        
        # Open Interest - OUR indicator from dataset
        oi_data = []
        if isinstance(dataset, dict):
            for field in ['open_interest', 'oi', 'total_oi']:
                if field in dataset:
                    oi = dataset[field]
                    if isinstance(oi, (pd.Series, pd.DataFrame)) and not oi.empty:
                        if isinstance(oi, pd.DataFrame):
                            oi = oi.iloc[:, 0]
                        step = max(1, len(oi) // 2000)
                        for idx, val in oi.iloc[::step].items():
                            oi_data.append({
                                'time': int(idx.timestamp()),
                                'value': float(val)
                            })
                        break
        
        # Strategy Signals/Scores - from results or dataset
        signal_data = []
        
        # Check multiple possible fields for strategy scores
        score_fields = ['squeeze_scores', 'strategy_scores', 'signals', 'squeeze_score']
        
        if isinstance(results, dict):
            for field in score_fields:
                if field in results:
                    scores = results[field]
                    if isinstance(scores, pd.Series) and not scores.empty:
                        step = max(1, len(scores) // 2000)
                        for idx, val in scores.iloc[::step].items():
                            signal_data.append({
                                'time': int(idx.timestamp()),
                                'value': float(val)
                            })
                        break
                    elif isinstance(scores, list) and scores:
                        # Handle list format
                        for item in scores[::max(1, len(scores) // 2000)]:
                            if isinstance(item, dict) and 'time' in item and 'value' in item:
                                signal_data.append({
                                    'time': int(item['time']),
                                    'value': float(item['value'])
                                })
                        break
        
        # Also check in dataset
        if not signal_data and isinstance(dataset, dict):
            for field in score_fields:
                if field in dataset:
                    scores = dataset[field]
                    if isinstance(scores, pd.Series) and not scores.empty:
                        step = max(1, len(scores) // 2000)
                        for idx, val in scores.iloc[::step].items():
                            signal_data.append({
                                'time': int(idx.timestamp()),
                                'value': float(val)
                            })
                        break
        
        # If still no data, show a zero line
        if not signal_data and not ohlcv.empty:
            step = max(1, len(ohlcv) // 500)
            for idx in ohlcv.iloc[::step].index:
                signal_data.append({
                    'time': int(idx.timestamp()),
                    'value': 0
                })
        
        # Trade markers
        markers = []
        if executed_orders:
            for order in executed_orders:
                if isinstance(order, dict) and 'timestamp' in order:
                    # Convert timestamp to int
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
        
        # Get metrics
        total_return = results.get('total_return', 0) if isinstance(results, dict) else 0
        win_rate = results.get('win_rate', 0) if isinstance(results, dict) else 0
        sharpe_ratio = results.get('sharpe_ratio', 0) if isinstance(results, dict) else 0
        max_drawdown = results.get('max_drawdown', 0) if isinstance(results, dict) else 0
        total_trades = len(executed_orders) if executed_orders else 0
        
        # Create HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - Strategy Dashboard</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            overflow: hidden;
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
        
        .header {{
            background: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
        }}
        
        .timeframe-selector {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .timeframe-btn {{
            padding: 5px 12px;
            background: #2a2e39;
            border: none;
            color: #d1d4dc;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .timeframe-btn:hover {{ background: #363a45; }}
        .timeframe-btn.active {{ 
            background: #2962ff;
            color: white;
        }}
        
        h1 {{
            color: #5d94ff;
            margin: 0;
            font-size: 20px;
        }}
        
        .metrics {{
            display: flex;
            gap: 25px;
        }}
        
        .metric {{
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
            color: #26a69a;
        }}
        
        .metric-value.negative {{
            color: #ef5350;
        }}
        
        .metric-label {{
            font-size: 11px;
            color: #787b86;
            margin-top: 3px;
        }}
        
        .charts-container {{
            height: calc(100vh - 110px);  /* Nav + Header */
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
        
        .status {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            padding: 8px;
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 4px;
            font-size: 11px;
            color: #787b86;
        }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="dashboard.html" class="active">📊 Main Dashboard</a>
        <a href="portfolio.html">💼 Portfolio</a>
        <a href="exchange.html">🏛️ Exchange Analytics</a>
    </div>
    
    <div class="header">
        <div style="display: flex; align-items: center; gap: 20px;">
            <h1>{symbol} - Strategy Backtest</h1>
            <div class="timeframe-selector">
                <span style="color: #787b86;">Timeframe:</span>
                <button class="timeframe-btn {'active' if default_timeframe == '1s' else ''}" data-tf="1s">1s</button>
                <button class="timeframe-btn {'active' if default_timeframe == '1m' else ''}" data-tf="1m">1m</button>
                <button class="timeframe-btn {'active' if default_timeframe == '5m' else ''}" data-tf="5m">5m</button>
                <button class="timeframe-btn {'active' if default_timeframe == '15m' else ''}" data-tf="15m">15m</button>
                <button class="timeframe-btn {'active' if default_timeframe == '1h' else ''}" data-tf="1h">1h</button>
            </div>
        </div>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value {'' if total_return >= 0 else 'negative'}">{total_return:.2f}%</div>
                <div class="metric-label">Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{max_drawdown:.2f}%</div>
                <div class="metric-label">Max DD</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_trades}</div>
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
            <div class="pane-title">Futures/Perp CVD</div>
        </div>
        <div id="oi-chart" class="chart-pane">
            <div class="pane-title">Open Interest</div>
        </div>
        <div id="signal-chart" class="chart-pane">
            <div class="pane-title">Strategy Signals</div>
        </div>
    </div>
    
    <div class="status" id="status">Loading...</div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        // All timeframe data
        const allCandles = {json.dumps(all_candles, cls=DateTimeEncoder)};
        const allVolumes = {json.dumps(all_volumes, cls=DateTimeEncoder)};
        
        // Indicator data (same for all timeframes)
        const spotCvdData = {json.dumps(spot_cvd_data, cls=DateTimeEncoder)};
        const futuresCvdData = {json.dumps(futures_cvd_data, cls=DateTimeEncoder)};
        const oiData = {json.dumps(oi_data, cls=DateTimeEncoder)};
        const signalData = {json.dumps(signal_data, cls=DateTimeEncoder)};
        const markers = {json.dumps(markers, cls=DateTimeEncoder)};
        
        // Current timeframe
        let currentTimeframe = '{default_timeframe}';
        let candleSeries = null;
        let volumeSeries = null;
        
        function initCharts() {{
            const statusEl = document.getElementById('status');
            
            try {{
                // Chart options
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
                
                // Main chart - Price & Volume
                const mainContainer = document.getElementById('main-chart');
                const mainChart = LightweightCharts.createChart(mainContainer, {{
                    ...chartOptions,
                    width: mainContainer.clientWidth,
                    height: mainContainer.clientHeight
                }});
                
                // Add candlesticks
                candleSeries = mainChart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
                
                // Add volume
                volumeSeries = mainChart.addHistogramSeries({{
                    color: '#26a69a',
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: '',
                    scaleMargins: {{
                        top: 0.7,
                        bottom: 0,
                    }},
                }});
                
                // Set initial data
                const candleData = allCandles[currentTimeframe] || [];
                const volumeData = allVolumes[currentTimeframe] || [];
                
                if (candleData.length > 0) {{
                    candleSeries.setData(candleData);
                    volumeSeries.setData(volumeData);
                    
                    // Add trade markers
                    if (markers && markers.length > 0) {{
                        candleSeries.setMarkers(markers);
                    }}
                }}
                
                // Store all charts for syncing
                const charts = [mainChart];
                
                // Spot CVD chart
                const spotCvdContainer = document.getElementById('spot-cvd-chart');
                const spotCvdChart = LightweightCharts.createChart(spotCvdContainer, {{
                    ...chartOptions,
                    width: spotCvdContainer.clientWidth,
                    height: spotCvdContainer.clientHeight
                }});
                charts.push(spotCvdChart);
                
                if (spotCvdData && spotCvdData.length > 0) {{
                    const spotCvdSeries = spotCvdChart.addLineSeries({{
                        color: '#2962ff',
                        lineWidth: 2,
                    }});
                    spotCvdSeries.setData(spotCvdData);
                }} else {{
                    // Show zero line if no data
                    const zeroSeries = spotCvdChart.addLineSeries({{
                        color: 'rgba(255, 255, 255, 0.1)',
                        lineWidth: 1,
                    }});
                    zeroSeries.setData([{{time: candleData[0]?.time || 0, value: 0}}, 
                                       {{time: candleData[candleData.length-1]?.time || 0, value: 0}}]);
                }}
                
                // Futures CVD chart
                const futuresCvdContainer = document.getElementById('futures-cvd-chart');
                const futuresCvdChart = LightweightCharts.createChart(futuresCvdContainer, {{
                    ...chartOptions,
                    width: futuresCvdContainer.clientWidth,
                    height: futuresCvdContainer.clientHeight
                }});
                charts.push(futuresCvdChart);
                
                if (futuresCvdData && futuresCvdData.length > 0) {{
                    const futuresCvdSeries = futuresCvdChart.addLineSeries({{
                        color: '#ff6b6b',
                        lineWidth: 2,
                    }});
                    futuresCvdSeries.setData(futuresCvdData);
                }} else {{
                    // Show zero line if no data
                    const zeroSeries = futuresCvdChart.addLineSeries({{
                        color: 'rgba(255, 255, 255, 0.1)',
                        lineWidth: 1,
                    }});
                    zeroSeries.setData([{{time: candleData[0]?.time || 0, value: 0}}, 
                                       {{time: candleData[candleData.length-1]?.time || 0, value: 0}}]);
                }}
                
                // OI chart
                const oiContainer = document.getElementById('oi-chart');
                const oiChart = LightweightCharts.createChart(oiContainer, {{
                    ...chartOptions,
                    width: oiContainer.clientWidth,
                    height: oiContainer.clientHeight
                }});
                charts.push(oiChart);
                
                if (oiData && oiData.length > 0) {{
                    const oiSeries = oiChart.addLineSeries({{
                        color: '#ff9800',
                        lineWidth: 2,
                    }});
                    oiSeries.setData(oiData);
                }} else {{
                    // Show zero line if no data
                    const zeroSeries = oiChart.addLineSeries({{
                        color: 'rgba(255, 255, 255, 0.1)',
                        lineWidth: 1,
                    }});
                    zeroSeries.setData([{{time: candleData[0]?.time || 0, value: 0}}, 
                                       {{time: candleData[candleData.length-1]?.time || 0, value: 0}}]);
                }}
                
                // Signal chart for strategy signals
                const signalContainer = document.getElementById('signal-chart');
                const signalChart = LightweightCharts.createChart(signalContainer, {{
                    ...chartOptions,
                    width: signalContainer.clientWidth,
                    height: signalContainer.clientHeight
                }});
                charts.push(signalChart);
                
                if (signalData && signalData.length > 0) {{
                    const signalSeries = signalChart.addLineSeries({{
                        color: '#4caf50',
                        lineWidth: 2,
                    }});
                    signalSeries.setData(signalData);
                    
                    // Add threshold lines if we have actual signals
                    if (signalData.some(d => d.value !== 0)) {{
                        // Buy threshold
                        const buyLine = signalChart.addLineSeries({{
                            color: 'rgba(76, 175, 80, 0.3)',
                            lineWidth: 1,
                            lineStyle: 2,
                        }});
                        buyLine.setData(signalData.map(d => ({{time: d.time, value: 0.7}})));
                        
                        // Sell threshold
                        const sellLine = signalChart.addLineSeries({{
                            color: 'rgba(244, 67, 54, 0.3)',
                            lineWidth: 1,
                            lineStyle: 2,
                        }});
                        sellLine.setData(signalData.map(d => ({{time: d.time, value: -0.7}})));
                    }}
                }}
                
                // Sync all charts together
                charts.forEach((chart, idx) => {{
                    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                        charts.forEach((otherChart, otherIdx) => {{
                            if (idx !== otherIdx) {{
                                otherChart.timeScale().setVisibleLogicalRange(range);
                            }}
                        }});
                    }});
                }});
                
                // Fit content
                mainChart.timeScale().fitContent();
                
                // Handle resize
                window.addEventListener('resize', () => {{
                    mainChart.applyOptions({{
                        width: mainContainer.clientWidth,
                        height: mainContainer.clientHeight
                    }});
                    spotCvdChart.applyOptions({{
                        width: spotCvdContainer.clientWidth,
                        height: spotCvdContainer.clientHeight
                    }});
                    futuresCvdChart.applyOptions({{
                        width: futuresCvdContainer.clientWidth,
                        height: futuresCvdContainer.clientHeight
                    }});
                    oiChart.applyOptions({{
                        width: oiContainer.clientWidth,
                        height: oiContainer.clientHeight
                    }});
                    signalChart.applyOptions({{
                        width: signalContainer.clientWidth,
                        height: signalContainer.clientHeight
                    }});
                }});
                
                statusEl.innerHTML = '✓ Ready';
                statusEl.style.color = '#26a69a';
                
                // Add timeframe switching
                document.querySelectorAll('.timeframe-btn').forEach(btn => {{
                    btn.addEventListener('click', function() {{
                        const newTf = this.dataset.tf;
                        switchTimeframe(newTf);
                    }});
                }});
                
            }} catch(e) {{
                statusEl.innerHTML = 'Error: ' + e.message;
                statusEl.style.color = '#ef5350';
                console.error(e);
            }}
        }}
        
        // Timeframe switching function
        function switchTimeframe(tf) {{
            // Update active button
            document.querySelectorAll('.timeframe-btn').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.dataset.tf === tf) {{
                    btn.classList.add('active');
                }}
            }});
            
            // Update current timeframe
            currentTimeframe = tf;
            
            // Get data for new timeframe
            const newCandles = allCandles[tf] || [];
            const newVolumes = allVolumes[tf] || [];
            
            // Update chart data
            if (candleSeries && newCandles.length > 0) {{
                candleSeries.setData(newCandles);
                volumeSeries.setData(newVolumes);
                
                // Re-add markers
                if (markers && markers.length > 0) {{
                    candleSeries.setMarkers(markers);
                }}
            }}
        }}
        
        // Initialize
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initCharts);
        }} else {{
            initCharts();
        }}
    </script>
</body>
</html>"""
        
        # Save dashboard
        dashboard_path = report_dir / "dashboard.html"
        dashboard_path.write_text(html)
        
        logger.info(f"✅ Strategy dashboard created: {dashboard_path}")
        return str(dashboard_path)