"""
TRUE Separate Panes Implementation
Using multiple div containers with separate chart instances for REAL separation
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TrueSeparatePanes:
    """Create TRULY separate panes using multiple chart containers"""
    
    def create_dashboard(self, results: Dict, dataset: Dict, output_dir: str) -> str:
        """Create dashboard with REAL visual separation"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get data
        symbol = results.get('symbol', 'UNKNOWN')
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
        
        if ohlcv.empty:
            logger.warning("⚠️ OHLCV data is empty!")
            return self._create_empty_dashboard(output_dir)
        
        # Prepare data
        chart_data = self._prepare_all_data(results, dataset)
        
        # Generate HTML with TRUE separate panes
        html_content = self._generate_true_panes_html(symbol, chart_data, results)
        
        # Save to file
        dashboard_path = output_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ TRUE separate panes dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def _prepare_all_data(self, results: Dict, dataset: Dict) -> Dict:
        """Prepare all data for charts"""
        
        chart_data = {
            'candlesticks': [],
            'volume': [],
            'spot_cvd': [],
            'futures_cvd': [],
            'scores': [],
            'trades': []
        }
        
        # Process OHLCV data
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
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
                if 'volume' in row and row['volume'] > 0:
                    chart_data['volume'].append({
                        'time': timestamp,
                        'value': float(row['volume']),
                        'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                    })
        
        # Process CVD data
        spot_cvd = dataset.get('spot_cvd', pd.Series())
        if not spot_cvd.empty:
            for idx, value in spot_cvd.items():
                timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx)
                chart_data['spot_cvd'].append({
                    'time': timestamp,
                    'value': float(value)
                })
        
        futures_cvd = dataset.get('futures_cvd', pd.Series())
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
                        'text': order.get('side', 'TRADE')[:1]
                    })
        
        return chart_data
    
    def _generate_true_panes_html(self, symbol: str, chart_data: Dict, results: Dict) -> str:
        """Generate HTML with TRUE separate pane containers"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} - TRUE Separate Panes Dashboard</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
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
        
        .dashboard-container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }}
        
        .header {{
            background: #131722;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
            height: 50px;
        }}
        
        .charts-wrapper {{
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }}
        
        /* TRULY SEPARATE PANE CONTAINERS */
        .pane {{
            position: relative;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            flex-direction: column;
        }}
        
        .pane-label {{
            position: absolute;
            top: 5px;
            left: 10px;
            background: rgba(19, 23, 34, 0.9);
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            color: #848e9c;
            z-index: 10;
        }}
        
        #price-pane {{
            height: 50%;  /* Top 50% for price and trades */
        }}
        
        #volume-pane {{
            height: 20%;  /* 20% for volume */
        }}
        
        #cvd-pane {{
            height: 20%;  /* 20% for CVD */
        }}
        
        #score-pane {{
            height: 10%;  /* 10% for strategy score */
        }}
        
        .stats-panel {{
            position: absolute;
            top: 60px;
            right: 20px;
            background: rgba(19, 23, 34, 0.95);
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #2a2e39;
            min-width: 200px;
            z-index: 100;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        
        .stat-label {{
            color: #848e9c;
        }}
        
        .stat-value {{
            color: #d1d4dc;
            font-weight: 500;
        }}
        
        .stat-value.positive {{
            color: #26a69a;
        }}
        
        .stat-value.negative {{
            color: #ef5350;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>{symbol} - TRUE Separate Panes</h1>
        </div>
        
        <div class="charts-wrapper">
            <!-- SEPARATE PANE 1: Price and Trades -->
            <div id="price-pane" class="pane">
                <span class="pane-label">Price</span>
            </div>
            
            <!-- SEPARATE PANE 2: Volume -->
            <div id="volume-pane" class="pane">
                <span class="pane-label">Volume</span>
            </div>
            
            <!-- SEPARATE PANE 3: CVD -->
            <div id="cvd-pane" class="pane">
                <span class="pane-label">CVD</span>
            </div>
            
            <!-- SEPARATE PANE 4: Strategy Score -->
            <div id="score-pane" class="pane">
                <span class="pane-label">Strategy Score</span>
            </div>
        </div>
        
        <div class="stats-panel">
            <div class="stat-row">
                <span class="stat-label">Total Return:</span>
                <span class="stat-value {('positive' if results.get('total_return', 0) >= 0 else 'negative')}">{results.get('total_return', 0):.2f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Trades:</span>
                <span class="stat-value">{len(results.get('executed_orders', []))}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Win Rate:</span>
                <span class="stat-value">{results.get('metrics', {}).get('win_rate', 0) if results.get('metrics') else 0:.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Final Balance:</span>
                <span class="stat-value">${results.get('final_balance', 0):,.2f}</span>
            </div>
        </div>
    </div>
    
    <script>
        const chartData = {json.dumps(chart_data)};
        
        // Common chart options
        const commonOptions = {{
            layout: {{
                background: {{ type: 'solid', color: 'transparent' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2a2e39' }},
                horzLines: {{ color: '#2a2e39' }},
            }},
            timeScale: {{
                borderColor: '#2a2e39',
                timeVisible: true,
                secondsVisible: false,
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
            }},
        }};
        
        // CREATE SEPARATE CHART FOR EACH PANE
        
        // 1. PRICE CHART
        const priceContainer = document.getElementById('price-pane');
        const priceChart = LightweightCharts.createChart(priceContainer, {{
            ...commonOptions,
            width: priceContainer.clientWidth,
            height: priceContainer.clientHeight,
        }});
        
        const priceSeries = priceChart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        
        if (chartData.candlesticks.length > 0) {{
            priceSeries.setData(chartData.candlesticks);
            if (chartData.trades.length > 0) {{
                try {{
                    priceSeries.setMarkers(chartData.trades);
                }} catch(e) {{
                    console.log('Markers not supported');
                }}
            }}
        }}
        
        // 2. VOLUME CHART
        const volumeContainer = document.getElementById('volume-pane');
        const volumeChart = LightweightCharts.createChart(volumeContainer, {{
            ...commonOptions,
            width: volumeContainer.clientWidth,
            height: volumeContainer.clientHeight,
            timeScale: {{
                ...commonOptions.timeScale,
                visible: false,  // Hide time axis for middle panes
            }},
        }});
        
        const volumeSeries = volumeChart.addHistogramSeries({{
            color: '#26a69a',
            priceFormat: {{ type: 'volume' }},
        }});
        
        if (chartData.volume.length > 0) {{
            volumeSeries.setData(chartData.volume);
        }}
        
        // 3. CVD CHART
        const cvdContainer = document.getElementById('cvd-pane');
        const cvdChart = LightweightCharts.createChart(cvdContainer, {{
            ...commonOptions,
            width: cvdContainer.clientWidth,
            height: cvdContainer.clientHeight,
            timeScale: {{
                ...commonOptions.timeScale,
                visible: false,  // Hide time axis for middle panes
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
        
        if (chartData.spot_cvd.length > 0) {{
            spotCvdSeries.setData(chartData.spot_cvd);
        }}
        if (chartData.futures_cvd.length > 0) {{
            futuresCvdSeries.setData(chartData.futures_cvd);
        }}
        
        // 4. SCORE CHART
        const scoreContainer = document.getElementById('score-pane');
        const scoreChart = LightweightCharts.createChart(scoreContainer, {{
            ...commonOptions,
            width: scoreContainer.clientWidth,
            height: scoreContainer.clientHeight,
        }});
        
        const scoreSeries = scoreChart.addLineSeries({{
            color: '#9c27b0',
            lineWidth: 2,
            title: 'Strategy Score',
        }});
        
        if (chartData.scores.length > 0) {{
            scoreSeries.setData(chartData.scores);
        }}
        
        // SYNCHRONIZE TIME SCALES
        const charts = [priceChart, volumeChart, cvdChart, scoreChart];
        
        // Sync crosshair
        charts.forEach((chart, index) => {{
            chart.subscribeCrosshairMove((param) => {{
                charts.forEach((otherChart, otherIndex) => {{
                    if (index !== otherIndex) {{
                        otherChart.applyOptions({{
                            crosshair: {{
                                horzLine: {{
                                    visible: false,
                                }},
                                vertLine: {{
                                    visible: true,
                                    labelVisible: false,
                                }},
                            }},
                        }});
                        if (param.time) {{
                            otherChart.setCrosshairPosition(0, param.time, null);
                        }}
                    }}
                }});
            }});
        }});
        
        // Sync visible range
        priceChart.timeScale().subscribeVisibleLogicalRangeChange((timeRange) => {{
            volumeChart.timeScale().setVisibleLogicalRange(timeRange);
            cvdChart.timeScale().setVisibleLogicalRange(timeRange);
            scoreChart.timeScale().setVisibleLogicalRange(timeRange);
        }});
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            priceChart.applyOptions({{ width: priceContainer.clientWidth, height: priceContainer.clientHeight }});
            volumeChart.applyOptions({{ width: volumeContainer.clientWidth, height: volumeContainer.clientHeight }});
            cvdChart.applyOptions({{ width: cvdContainer.clientWidth, height: cvdContainer.clientHeight }});
            scoreChart.applyOptions({{ width: scoreContainer.clientWidth, height: scoreContainer.clientHeight }});
        }});
        
        // Fit content
        priceChart.timeScale().fitContent();
        volumeChart.timeScale().fitContent();
        cvdChart.timeScale().fitContent();
        scoreChart.timeScale().fitContent();
        
        console.log('TRUE Separate Panes initialized with:', {{
            candlesticks: chartData.candlesticks.length,
            volume: chartData.volume.length,
            spotCVD: chartData.spot_cvd.length,
            futuresCVD: chartData.futures_cvd.length,
            scores: chartData.scores.length,
            trades: chartData.trades.length,
        }});
    </script>
</body>
</html>"""
    
    def _create_empty_dashboard(self, output_dir: Path) -> str:
        """Create an empty dashboard when no data is available"""
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - No Data</title>
    <style>
        body {
            background: #1e222d;
            color: #d1d4dc;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
    </style>
</head>
<body>
    <h1>No data available</h1>
</body>
</html>"""
        
        dashboard_path = output_dir / "dashboard.html"
        dashboard_path.write_text(html_content)
        return str(dashboard_path)