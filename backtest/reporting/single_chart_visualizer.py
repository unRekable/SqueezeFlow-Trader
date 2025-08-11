"""
Single Chart TradingView Visualizer
ONE chart with all indicators - no bullshit
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class SingleChartVisualizer:
    """One fucking chart with everything as indicators"""
    
    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        
    def create_backtest_report(self, results: Dict, dataset: Dict, 
                              executed_orders: List[Dict]) -> str:
        """Create ONE chart with all indicators"""
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get REAL data - no bullshit
        symbol = results.get('symbol', 'UNKNOWN') if isinstance(results, dict) else 'UNKNOWN'
        ohlcv = dataset.get('ohlcv', pd.DataFrame()) if isinstance(dataset, dict) else pd.DataFrame()
        
        # Prepare chart data
        candles = []
        if not ohlcv.empty:
            # Sample if too many candles
            step = max(1, len(ohlcv) // 2000)
            sampled = ohlcv.iloc[::step]
            
            for idx, row in sampled.iterrows():
                candles.append({
                    'time': int(idx.timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
        
        # Volume data
        volume_data = []
        if not ohlcv.empty:
            step = max(1, len(ohlcv) // 2000)
            sampled = ohlcv.iloc[::step]
            for idx, row in sampled.iterrows():
                volume_data.append({
                    'time': int(idx.timestamp()),
                    'value': float(row.get('volume', 0)),
                    'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                })
        
        # CVD data - REAL from dataset
        cvd_data = []
        if isinstance(dataset, dict) and 'spot_cvd' in dataset:
            cvd = dataset['spot_cvd']
            if isinstance(cvd, pd.Series) and not cvd.empty:
                step = max(1, len(cvd) // 2000)
                for idx, val in cvd.iloc[::step].items():
                    cvd_data.append({
                        'time': int(idx.timestamp()),
                        'value': float(val)
                    })
        
        # Trade markers - REAL trades
        markers = []
        if executed_orders:
            for order in executed_orders:
                if isinstance(order, dict) and 'timestamp' in order:
                    markers.append({
                        'time': int(pd.Timestamp(order['timestamp']).timestamp()),
                        'position': 'belowBar' if order.get('side') == 'buy' else 'aboveBar',
                        'color': '#26a69a' if order.get('side') == 'buy' else '#ef5350',
                        'shape': 'arrowUp' if order.get('side') == 'buy' else 'arrowDown',
                        'text': order.get('side', '').upper()
                    })
        
        # Calculate simple indicators
        rsi_data = []
        if not ohlcv.empty and len(ohlcv) > 14:
            close = ohlcv['close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            step = max(1, len(rsi) // 2000)
            for idx, val in rsi.iloc[::step].dropna().items():
                rsi_data.append({
                    'time': int(idx.timestamp()),
                    'value': float(val)
                })
        
        # Get metrics
        total_return = results.get('total_return', 0) if isinstance(results, dict) else 0
        win_rate = results.get('win_rate', 0) if isinstance(results, dict) else 0
        sharpe_ratio = results.get('sharpe_ratio', 0) if isinstance(results, dict) else 0
        max_drawdown = results.get('max_drawdown', 0) if isinstance(results, dict) else 0
        total_trades = len(executed_orders) if executed_orders else 0
        
        # Create HTML with ONE chart
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - Single Chart Dashboard</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        .header {{
            background: #1e222d;
            padding: 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            color: #5d94ff;
            margin: 0;
            font-size: 24px;
        }}
        
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
        
        #chart {{
            width: 100%;
            height: calc(100vh - 100px);
            background: #1e222d;
        }}
        
        .status {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            padding: 10px;
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 4px;
            font-size: 12px;
            color: #787b86;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{symbol} - Backtest Results</h1>
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
            <div class="metric">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>
    </div>
    
    <!-- ONE CHART TO RULE THEM ALL -->
    <div id="chart"></div>
    
    <div class="status" id="status">Loading...</div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        // Data
        const candleData = {json.dumps(candles)};
        const volumeData = {json.dumps(volume_data)};
        const cvdData = {json.dumps(cvd_data)};
        const rsiData = {json.dumps(rsi_data)};
        const markers = {json.dumps(markers)};
        
        function initChart() {{
            const container = document.getElementById('chart');
            const statusEl = document.getElementById('status');
            
            try {{
                // Create chart with multiple panes
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
                }});
                
                // Add candlestick series (main price)
                const candleSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
                candleSeries.setData(candleData);
                
                // Add trade markers
                if (markers && markers.length > 0) {{
                    candleSeries.setMarkers(markers);
                }}
                
                // Add volume as histogram in same pane with secondary scale
                const volumeSeries = chart.addHistogramSeries({{
                    color: '#26a69a',
                    priceFormat: {{
                        type: 'volume',
                    }},
                    priceScaleId: '', // Use default scale
                    scaleMargins: {{
                        top: 0.8, // Volume at bottom 20% of chart
                        bottom: 0,
                    }},
                }});
                volumeSeries.setData(volumeData);
                
                // Add CVD as line overlay if we have data
                if (cvdData && cvdData.length > 0) {{
                    const cvdSeries = chart.addLineSeries({{
                        color: '#2962ff',
                        lineWidth: 2,
                        priceScaleId: 'right',
                        scaleMargins: {{
                            top: 0.1,
                            bottom: 0.4,
                        }},
                    }});
                    cvdSeries.setData(cvdData);
                }}
                
                // Add RSI in separate pane if we have data
                if (rsiData && rsiData.length > 0) {{
                    const rsiSeries = chart.addLineSeries({{
                        color: '#ff9800',
                        lineWidth: 2,
                        priceScaleId: 'rsi',
                        scaleMargins: {{
                            top: 0.8,
                            bottom: 0,
                        }},
                    }});
                    rsiSeries.setData(rsiData);
                    
                    // Add RSI levels
                    const rsi70 = chart.addLineSeries({{
                        color: 'rgba(239, 83, 80, 0.3)',
                        lineWidth: 1,
                        lineStyle: 2,
                        priceScaleId: 'rsi',
                        scaleMargins: {{
                            top: 0.8,
                            bottom: 0,
                        }},
                    }});
                    rsi70.setData(rsiData.map(d => ({{time: d.time, value: 70}})));
                    
                    const rsi30 = chart.addLineSeries({{
                        color: 'rgba(38, 166, 154, 0.3)',
                        lineWidth: 1,
                        lineStyle: 2,
                        priceScaleId: 'rsi',
                        scaleMargins: {{
                            top: 0.8,
                            bottom: 0,
                        }},
                    }});
                    rsi30.setData(rsiData.map(d => ({{time: d.time, value: 30}})));
                }}
                
                // Fit content
                chart.timeScale().fitContent();
                
                // Handle resize
                window.addEventListener('resize', () => {{
                    chart.applyOptions({{
                        width: container.clientWidth,
                        height: container.clientHeight
                    }});
                }});
                
                statusEl.innerHTML = '✓ Chart ready (' + candleData.length + ' candles)';
                statusEl.style.color = '#26a69a';
                
            }} catch(e) {{
                statusEl.innerHTML = 'Error: ' + e.message;
                statusEl.style.color = '#ef5350';
                console.error(e);
            }}
        }}
        
        // Initialize
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initChart);
        }} else {{
            initChart();
        }}
    </script>
</body>
</html>"""
        
        # Save dashboard
        dashboard_path = report_dir / "dashboard.html"
        dashboard_path.write_text(html)
        
        logger.info(f"✅ Single chart dashboard created: {dashboard_path}")
        return str(dashboard_path)