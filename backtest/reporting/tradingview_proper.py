"""
Proper TradingView Lightweight Charts Implementation
Single chart with multiple panes for all indicators
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class TradingViewProperVisualizer:
    """Clean TradingView implementation with proper multi-pane support"""
    
    def __init__(self):
        self.logger = logger
        
    def create_dashboard(self, results: Dict, output_dir: str) -> str:
        """Create a proper TradingView dashboard with all indicators in panes"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        symbol = results.get('symbol', 'BTC')
        ohlcv = results.get('ohlcv', pd.DataFrame())
        spot_cvd = results.get('spot_cvd', pd.Series())
        futures_cvd = results.get('futures_cvd', pd.Series())
        
        # Prepare data for TradingView
        chart_data = self._prepare_chart_data(ohlcv, spot_cvd, futures_cvd, results)
        
        # Generate HTML with proper TradingView setup
        html_content = self._generate_html(symbol, chart_data, results)
        
        # Save to file
        dashboard_path = output_path / "dashboard_proper.html"
        dashboard_path.write_text(html_content)
        
        return str(dashboard_path)
    
    def _prepare_chart_data(self, ohlcv: pd.DataFrame, spot_cvd: pd.Series, 
                           futures_cvd: pd.Series, results: Dict) -> Dict:
        """Prepare all data for the chart"""
        
        chart_data = {
            'candlesticks': [],
            'volume': [],
            'spot_cvd': [],
            'futures_cvd': [],
            'scores': [],
            'trades': []
        }
        
        # Process OHLCV data
        if not ohlcv.empty:
            for idx, row in ohlcv.iterrows():
                timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                
                # Candlestick data with proper color determination
                chart_data['candlesticks'].append({
                    'time': timestamp,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
                
                # Volume as separate series
                if 'volume' in row:
                    chart_data['volume'].append({
                        'time': timestamp,
                        'value': float(row['volume']),
                        'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                    })
        
        # Process CVD data
        if not spot_cvd.empty:
            for idx, value in spot_cvd.items():
                timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                chart_data['spot_cvd'].append({
                    'time': timestamp,
                    'value': float(value)
                })
        
        if not futures_cvd.empty:
            for idx, value in futures_cvd.items():
                timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                chart_data['futures_cvd'].append({
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
    
    def _generate_html(self, symbol: str, chart_data: Dict, results: Dict) -> str:
        """Generate proper TradingView HTML with multiple panes"""
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - TradingView Dashboard</title>
    <meta charset="utf-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #1e222d;
            color: #d1d4dc;
            overflow: hidden;
        }}
        
        #header {{
            height: 50px;
            background: #131722;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            align-items: center;
            padding: 0 20px;
            justify-content: space-between;
        }}
        
        #header h1 {{
            font-size: 18px;
            font-weight: 500;
        }}
        
        #timeframes {{
            display: flex;
            gap: 10px;
        }}
        
        .timeframe-btn {{
            padding: 8px 16px;
            background: #2a2e39;
            border: none;
            color: #d1d4dc;
            cursor: pointer;
            border-radius: 4px;
            transition: background 0.3s;
        }}
        
        .timeframe-btn:hover {{
            background: #363a45;
        }}
        
        .timeframe-btn.active {{
            background: #2962ff;
        }}
        
        #container {{
            height: calc(100vh - 50px);
            display: flex;
            flex-direction: column;
        }}
        
        #main-chart {{
            flex: 3;
            min-height: 0;
        }}
        
        #volume-pane {{
            flex: 1;
            min-height: 0;
            border-top: 1px solid #2a2e39;
        }}
        
        #cvd-pane {{
            flex: 1;
            min-height: 0;
            border-top: 1px solid #2a2e39;
        }}
        
        #score-pane {{
            flex: 1;
            min-height: 0;
            border-top: 1px solid #2a2e39;
        }}
        
        .pane-title {{
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 12px;
            color: #787b86;
            z-index: 10;
            background: #1e222d;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        #stats {{
            position: absolute;
            top: 60px;
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
        <div id="timeframes">
            <button class="timeframe-btn" data-timeframe="1s">1s</button>
            <button class="timeframe-btn active" data-timeframe="1m">1m</button>
            <button class="timeframe-btn" data-timeframe="5m">5m</button>
            <button class="timeframe-btn" data-timeframe="15m">15m</button>
            <button class="timeframe-btn" data-timeframe="1h">1h</button>
            <button class="timeframe-btn" data-timeframe="4h">4h</button>
        </div>
    </div>
    
    <div id="container">
        <div id="main-chart" style="position: relative;">
            <div class="pane-title">Price</div>
        </div>
        <div id="volume-pane" style="position: relative;">
            <div class="pane-title">Volume</div>
        </div>
        <div id="cvd-pane" style="position: relative;">
            <div class="pane-title">CVD (Spot / Futures)</div>
        </div>
        <div id="score-pane" style="position: relative;">
            <div class="pane-title">Strategy Score</div>
        </div>
    </div>
    
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
        // Chart data
        const chartData = {json.dumps(chart_data)};
        
        // Create main chart with proper configuration
        const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {{
            width: document.getElementById('main-chart').clientWidth,
            height: document.getElementById('main-chart').clientHeight,
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
            }},
            timeScale: {{
                borderColor: '#2a2e39',
                timeVisible: true,
                secondsVisible: true,
            }},
        }});
        
        // Add candlestick series
        const candlestickSeries = mainChart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        
        if (chartData.candlesticks && chartData.candlesticks.length > 0) {{
            candlestickSeries.setData(chartData.candlesticks);
        }}
        
        // Add trade markers
        if (chartData.trades && chartData.trades.length > 0) {{
            candlestickSeries.setMarkers(chartData.trades);
        }}
        
        // Create volume chart
        const volumeChart = LightweightCharts.createChart(document.getElementById('volume-pane'), {{
            width: document.getElementById('volume-pane').clientWidth,
            height: document.getElementById('volume-pane').clientHeight,
            layout: {{
                background: {{ color: '#1e222d' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2a2e39' }},
                horzLines: {{ color: '#2a2e39' }},
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
                scaleMargins: {{
                    top: 0.3,
                    bottom: 0.1,
                }},
            }},
            timeScale: {{
                visible: false,
            }},
        }});
        
        const volumeSeries = volumeChart.addHistogramSeries({{
            color: '#26a69a',
            priceFormat: {{
                type: 'volume',
            }},
        }});
        
        if (chartData.volume && chartData.volume.length > 0) {{
            volumeSeries.setData(chartData.volume);
        }}
        
        // Create CVD chart
        const cvdChart = LightweightCharts.createChart(document.getElementById('cvd-pane'), {{
            width: document.getElementById('cvd-pane').clientWidth,
            height: document.getElementById('cvd-pane').clientHeight,
            layout: {{
                background: {{ color: '#1e222d' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2a2e39' }},
                horzLines: {{ color: '#2a2e39' }},
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
            }},
            timeScale: {{
                visible: false,
            }},
        }});
        
        const spotCvdSeries = cvdChart.addLineSeries({{
            color: '#2962ff',
            lineWidth: 2,
            title: 'Spot CVD',
        }});
        
        const futuresCvdSeries = cvdChart.addLineSeries({{
            color: '#ff9800',
            lineWidth: 2,
            title: 'Futures CVD',
        }});
        
        if (chartData.spot_cvd && chartData.spot_cvd.length > 0) {{
            spotCvdSeries.setData(chartData.spot_cvd);
        }}
        
        if (chartData.futures_cvd && chartData.futures_cvd.length > 0) {{
            futuresCvdSeries.setData(chartData.futures_cvd);
        }}
        
        // Create Score chart
        const scoreChart = LightweightCharts.createChart(document.getElementById('score-pane'), {{
            width: document.getElementById('score-pane').clientWidth,
            height: document.getElementById('score-pane').clientHeight,
            layout: {{
                background: {{ color: '#1e222d' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2a2e39' }},
                horzLines: {{ color: '#2a2e39' }},
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
                scaleMargins: {{
                    top: 0.1,
                    bottom: 0.1,
                }},
            }},
            timeScale: {{
                visible: false,
            }},
        }});
        
        const scoreSeries = scoreChart.addLineSeries({{
            color: '#00bcd4',
            lineWidth: 2,
            title: 'Strategy Score',
        }});
        
        // Add horizontal lines for score thresholds
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
        
        if (chartData.scores && chartData.scores.length > 0) {{
            scoreSeries.setData(chartData.scores);
        }}
        
        // Sync all charts time scales
        mainChart.timeScale().subscribeVisibleLogicalRangeChange((timeRange) => {{
            volumeChart.timeScale().setVisibleLogicalRange(timeRange);
            cvdChart.timeScale().setVisibleLogicalRange(timeRange);
            scoreChart.timeScale().setVisibleLogicalRange(timeRange);
        }});
        
        volumeChart.timeScale().subscribeVisibleLogicalRangeChange((timeRange) => {{
            mainChart.timeScale().setVisibleLogicalRange(timeRange);
            cvdChart.timeScale().setVisibleLogicalRange(timeRange);
            scoreChart.timeScale().setVisibleLogicalRange(timeRange);
        }});
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            mainChart.resize(document.getElementById('main-chart').clientWidth, 
                           document.getElementById('main-chart').clientHeight);
            volumeChart.resize(document.getElementById('volume-pane').clientWidth, 
                             document.getElementById('volume-pane').clientHeight);
            cvdChart.resize(document.getElementById('cvd-pane').clientWidth, 
                          document.getElementById('cvd-pane').clientHeight);
            scoreChart.resize(document.getElementById('score-pane').clientWidth, 
                            document.getElementById('score-pane').clientHeight);
        }});
        
        // Timeframe switching
        const timeframeButtons = document.querySelectorAll('.timeframe-btn');
        const originalData = JSON.parse(JSON.stringify(chartData));
        
        function aggregateData(data, timeframe) {{
            // This is a simplified aggregation - in production you'd want proper OHLC aggregation
            console.log('Aggregating to', timeframe);
            // For now, just return original data
            // TODO: Implement proper timeframe aggregation
            return data;
        }}
        
        timeframeButtons.forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                timeframeButtons.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const timeframe = e.target.dataset.timeframe;
                const aggregated = aggregateData(originalData, timeframe);
                
                // Update all series with aggregated data
                if (aggregated.candlesticks) {{
                    candlestickSeries.setData(aggregated.candlesticks);
                }}
                if (aggregated.volume) {{
                    volumeSeries.setData(aggregated.volume);
                }}
                // ... etc
            }});
        }});
        
        // Fit content on load
        mainChart.timeScale().fitContent();
        volumeChart.timeScale().fitContent();
        cvdChart.timeScale().fitContent();
        scoreChart.timeScale().fitContent();
    </script>
</body>
</html>"""
        
        return html_content