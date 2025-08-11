"""
PROPER TradingView Implementation with Native Panes
Single chart instance with our indicators as proper TradingView indicators in panes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class TradingViewSingleChart:
    """Proper TradingView implementation using native pane system"""
    
    def __init__(self):
        self.logger = logger
        
    def create_dashboard(self, results: Dict, dataset: Dict, output_dir: str) -> str:
        """Create a proper TradingView dashboard with indicators in native panes"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract and prepare data
        symbol = results.get('symbol', 'BTC')
        chart_data = self._prepare_all_data(results, dataset)
        
        # Generate HTML with proper TradingView panes
        html_content = self._generate_proper_tradingview_html(symbol, chart_data, results)
        
        # Save to file
        dashboard_path = output_path / "dashboard.html"
        dashboard_path.write_text(html_content)
        
        return str(dashboard_path)
    
    def _prepare_all_data(self, results: Dict, dataset: Dict) -> Dict:
        """Prepare all data for TradingView format"""
        
        # Ensure results is a dict
        if not isinstance(results, dict):
            self.logger.error(f"Results is not a dict: {type(results)}")
            results = {}
        
        # Ensure dataset is a dict
        if not isinstance(dataset, dict):
            self.logger.error(f"Dataset is not a dict: {type(dataset)}")
            dataset = {}
        
        chart_data = {
            'candlesticks': [],
            'volume': [],
            'spot_cvd': [],
            'futures_cvd': [],
            'oi': [],
            'scores': [],
            'trades': []
        }
        
        # Process OHLCV data
        ohlcv = dataset.get('ohlcv', pd.DataFrame()) if dataset else pd.DataFrame()
        if ohlcv is None:
            ohlcv = pd.DataFrame()
        if not ohlcv.empty:
            for idx, row in ohlcv.iterrows():
                timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                
                # Candlestick data
                chart_data['candlesticks'].append({
                    'time': timestamp,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
                
                # Volume data
                if 'volume' in row:
                    chart_data['volume'].append({
                        'time': timestamp,
                        'value': float(row['volume']),
                        'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                    })
        
        # Process CVD data
        if dataset:
            spot_cvd = dataset.get('spot_cvd', pd.Series())
            if spot_cvd is None:
                spot_cvd = pd.Series()
            if not spot_cvd.empty:
                for idx, value in spot_cvd.items():
                    timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                    chart_data['spot_cvd'].append({
                        'time': timestamp,
                        'value': float(value)
                    })
            
            futures_cvd = dataset.get('futures_cvd', pd.Series())
            if futures_cvd is None:
                futures_cvd = pd.Series()
            if not futures_cvd.empty:
                for idx, value in futures_cvd.items():
                    timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                    chart_data['futures_cvd'].append({
                        'time': timestamp,
                        'value': float(value)
                    })
        
        # Process OI data if available
        if dataset and 'open_interest' in dataset:
            oi = dataset['open_interest']
            if isinstance(oi, pd.Series) and not oi.empty:
                for idx, value in oi.items():
                    timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                    chart_data['oi'].append({
                        'time': timestamp,
                        'value': float(value)
                    })
        
        # Process strategy scores
        if 'squeeze_scores' in results:
            scores_data = results['squeeze_scores']
            if isinstance(scores_data, dict):
                timestamps = scores_data.get('timestamps', [])
                scores = scores_data.get('scores', [])
                for ts, score in zip(timestamps, scores):
                    timestamp = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(ts)
                    chart_data['scores'].append({
                        'time': timestamp,
                        'value': float(score)
                    })
        
        # Process trades
        if 'executed_orders' in results:
            for order in results['executed_orders']:
                if 'timestamp' in order:
                    ts = order['timestamp']
                    timestamp = int(ts.timestamp()) if hasattr(ts, 'timestamp') else int(ts)
                    chart_data['trades'].append({
                        'time': timestamp,
                        'position': 'belowBar' if order.get('side') == 'SELL' else 'aboveBar',
                        'shape': 'arrowDown' if order.get('side') == 'SELL' else 'arrowUp',
                        'color': '#ef5350' if order.get('side') == 'SELL' else '#26a69a',
                        'text': order.get('side', 'TRADE')
                    })
        
        return chart_data
    
    def _generate_proper_tradingview_html(self, symbol: str, chart_data: Dict, results: Dict) -> str:
        """Generate HTML using TradingView's native pane system"""
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - TradingView Dashboard (Proper Implementation)</title>
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
            overflow: hidden;
        }}
        
        #header {{
            height: 40px;
            background: #131722;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            align-items: center;
            padding: 0 20px;
            justify-content: space-between;
        }}
        
        #header h1 {{
            font-size: 16px;
            font-weight: 500;
        }}
        
        #container {{
            height: calc(100vh - 40px);
            position: relative;
        }}
        
        #stats {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(19, 23, 34, 0.9);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 100;
        }}
        
        .stat-row {{
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }}
        
        .stat-label {{
            color: #787b86;
        }}
        
        .stat-value {{
            color: #d1d4dc;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{symbol} Strategy Dashboard</h1>
    </div>
    
    <div id="container"></div>
    
    <div id="stats">
        <div class="stat-row">
            <span class="stat-label">Return:</span>
            <span class="stat-value">{results.get('total_return', 0):.2f}%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Trades:</span>
            <span class="stat-value">{len(results.get('executed_orders', []))}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Win Rate:</span>
            <span class="stat-value">{results.get('metrics', {}).get('win_rate', 0):.1f}%</span>
        </div>
    </div>

    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        // Chart data from Python
        const chartData = {json.dumps(chart_data)};
        
        // Create the SINGLE chart with proper configuration for panes
        const chart = LightweightCharts.createChart(document.getElementById('container'), {{
            width: window.innerWidth,
            height: window.innerHeight - 40,
            layout: {{
                background: {{ color: '#1e222d' }},
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
                visible: true,
            }},
            leftPriceScale: {{
                borderColor: '#2a2e39',
                visible: false,  // We'll enable per-pane as needed
            }},
            timeScale: {{
                borderColor: '#2a2e39',
                timeVisible: true,
                secondsVisible: true,
            }},
        }});
        
        // PANE 0 (default): Candlesticks + Trade Markers
        const candlestickSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
        }});
        
        if (chartData.candlesticks && chartData.candlesticks.length > 0) {{
            candlestickSeries.setData(chartData.candlesticks);
            
            // Add trade markers using the new API
            if (chartData.trades && chartData.trades.length > 0) {{
                try {{
                    candlestickSeries.attachPrimitive(
                        LightweightCharts.createSeriesMarkers(candlestickSeries, chartData.trades)
                    );
                }} catch(e) {{
                    console.log('Markers not supported in this version');
                }}
            }}
        }}
        
        // PANE 1: Volume
        chart.addPane();  // Create second pane (index 1)
        const volumeSeries = chart.addSeries(LightweightCharts.HistogramSeries, {{
            color: '#26a69a',
            priceFormat: {{
                type: 'volume',
            }},
            priceScaleId: 'volume'
        }}, 1);  // Add to pane index 1
        
        if (chartData.volume && chartData.volume.length > 0) {{
            volumeSeries.setData(chartData.volume);
        }}
        
        // Configure volume pane scale
        volumeSeries.priceScale().applyOptions({{
            scaleMargins: {{
                top: 0.7,  // Leave 70% space at top
                bottom: 0,
            }},
        }});
        
        // PANE 2: CVD (Spot and Futures on same pane, different scales)
        chart.addPane();  // Create third pane (index 2)
        const spotCvdSeries = chart.addSeries(LightweightCharts.LineSeries, {{
            color: '#2962ff',
            lineWidth: 2,
            title: 'Spot CVD',
            priceScaleId: 'cvd'
        }}, 2);  // Add to pane index 2
        
        const futuresCvdSeries = chart.addSeries(LightweightCharts.LineSeries, {{
            color: '#ff9800',
            lineWidth: 2,
            title: 'Futures CVD',
            priceScaleId: 'cvd'  // Same scale as spot
        }}, 2);  // Also add to pane index 2
        
        if (chartData.spot_cvd && chartData.spot_cvd.length > 0) {{
            spotCvdSeries.setData(chartData.spot_cvd);
        }}
        
        if (chartData.futures_cvd && chartData.futures_cvd.length > 0) {{
            futuresCvdSeries.setData(chartData.futures_cvd);
        }}
        
        // PANE 3: Open Interest (if available)
        let oiPaneIndex = -1;
        if (chartData.oi && chartData.oi.length > 0) {{
            chart.addPane();  // Create fourth pane (index 3)
            oiPaneIndex = 3;
            const oiSeries = chart.addSeries(LightweightCharts.LineSeries, {{
                color: '#9c27b0',
                lineWidth: 2,
                title: 'Open Interest',
                priceScaleId: 'oi'
            }}, oiPaneIndex);
            oiSeries.setData(chartData.oi);
        }}
        
        // PANE 4 (or 3): Strategy Score
        const scorePaneIndex = oiPaneIndex > 0 ? 4 : 3;
        chart.addPane();  // Create last pane
        const scoreSeries = chart.addSeries(LightweightCharts.LineSeries, {{
            color: '#00bcd4',
            lineWidth: 2,
            title: 'Strategy Score',
            priceScaleId: 'score'
        }}, scorePaneIndex);
        
        if (chartData.scores && chartData.scores.length > 0) {{
            scoreSeries.setData(chartData.scores);
            
            // Add threshold lines
            scoreSeries.createPriceLine({{
                price: 3.0,
                color: '#ff9800',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'Min Entry',
            }});
            
            scoreSeries.createPriceLine({{
                price: 6.0,
                color: '#4caf50',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'Good Entry',
            }});
        }}
        
        // Configure pane heights (proportional)
        // This is a workaround since direct pane height API might not be available
        // We use price scale margins to effectively control visual pane sizes
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            chart.resize(window.innerWidth, window.innerHeight - 40);
        }});
        
        // Fit content on load
        chart.timeScale().fitContent();
        
        // Native timeframe switching would be handled by TradingView's built-in controls
        // For now we're using the raw data timeframe
        
        console.log('Chart initialized with', {{
            candlesticks: chartData.candlesticks.length,
            volume: chartData.volume.length,
            spotCVD: chartData.spot_cvd.length,
            futuresCVD: chartData.futures_cvd.length,
            oi: chartData.oi.length,
            scores: chartData.scores.length,
            trades: chartData.trades.length
        }});
    </script>
</body>
</html>"""