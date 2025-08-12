"""
TradingView Multi-Pane Dashboard
Proper implementation with separate chart instances for each indicator
Each pane has its own dedicated visual space - no overlapping!
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradingViewMultiPane:
    """Create a proper multi-pane dashboard with synchronized charts"""
    
    def create_dashboard(self, results: Dict, dataset: Dict, output_dir: str) -> str:
        """Create multi-pane dashboard with separate charts for each indicator"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get data
        symbol = results.get('symbol', 'UNKNOWN')
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
        
        if ohlcv.empty:
            logger.warning("⚠️ OHLCV data is empty!")
            return self._create_empty_dashboard(output_dir)
        
        # Prepare data for all timeframes (removed second timeframes)
        timeframes = ['1s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']
        timeframes_data = {}
        
        for tf in timeframes:
            candles, volumes = self._prepare_data_for_timeframe(ohlcv, tf)
            timeframes_data[tf] = {
                'candles': candles,
                'volumes': volumes
            }
        
        # Determine default timeframe based on data size
        default_tf = '5m' if len(timeframes_data.get('1s', {}).get('candles', [])) > 10000 else '1m'
        
        # Get indicator data
        spot_cvd = dataset.get('spot_cvd', pd.Series())
        futures_cvd = dataset.get('futures_cvd', pd.Series())
        
        # Prepare CVD data
        cvd_data = self._prepare_cvd_data(spot_cvd, futures_cvd, ohlcv.index)
        
        # Get strategy scores
        strategy_scores = self._extract_strategy_scores(results, ohlcv.index)
        
        # Get executed orders
        executed_orders = results.get('executed_orders', [])
        trades_data = self._prepare_trades_data(executed_orders)
        
        # Create HTML dashboard
        html_content = self._generate_html(
            symbol=symbol,
            timeframes_data=timeframes_data,
            default_tf=default_tf,
            cvd_data=cvd_data,
            strategy_scores=strategy_scores,
            trades_data=trades_data,
            results=results
        )
        
        # Save to file
        dashboard_path = output_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Multi-pane dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def _prepare_data_for_timeframe(self, ohlcv: pd.DataFrame, timeframe: str) -> tuple:
        """Prepare OHLCV data for specific timeframe"""
        
        if timeframe == '1s':
            # Use raw data for 1s
            df = ohlcv
        else:
            # Aggregate to requested timeframe
            df = self._aggregate_ohlcv(ohlcv, timeframe)
        
        candles = []
        volumes = []
        
        for idx, row in df.iterrows():
            # Skip rows with NaN values
            if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                continue
            
            timestamp = int(idx.timestamp())
            
            candles.append({
                'time': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
            
            volume_val = row.get('volume', 0)
            if not pd.isna(volume_val):
                volumes.append({
                    'time': timestamp,
                    'value': float(volume_val),
                    'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                })
        
        return candles, volumes
    
    def _aggregate_ohlcv(self, ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate 1s data to specified timeframe"""
        
        # Map timeframe to pandas resample rule
        resample_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        rule = resample_map.get(timeframe, '5T')
        
        # Aggregate with proper OHLCV rules
        aggregated = ohlcv.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return aggregated
    
    def _prepare_cvd_data(self, spot_cvd: pd.Series, futures_cvd: pd.Series, time_index) -> Dict:
        """Prepare CVD data aligned with OHLCV timepoints"""
        
        spot_data = []
        futures_data = []
        
        if not spot_cvd.empty and not futures_cvd.empty:
            # Align CVD data with OHLCV time index
            for timestamp in time_index:
                if timestamp in spot_cvd.index:
                    spot_val = float(spot_cvd.loc[timestamp])
                    if np.isfinite(spot_val):
                        spot_data.append({
                            'time': int(timestamp.timestamp()),
                            'value': spot_val
                        })
                
                if timestamp in futures_cvd.index:
                    futures_val = float(futures_cvd.loc[timestamp])
                    if np.isfinite(futures_val):
                        futures_data.append({
                            'time': int(timestamp.timestamp()),
                            'value': futures_val
                        })
        
        return {
            'spot': spot_data,
            'futures': futures_data
        }
    
    def _extract_strategy_scores(self, results: Dict, time_index) -> List[Dict]:
        """Extract strategy scores from results"""
        
        scores = []
        
        # Try to get scores from results
        squeeze_scores = results.get('squeeze_scores', [])
        
        if squeeze_scores:
            for score_data in squeeze_scores:
                if isinstance(score_data, dict) and 'timestamp' in score_data and 'total_score' in score_data:
                    scores.append({
                        'time': int(score_data['timestamp'].timestamp()),
                        'value': float(score_data['total_score'])
                    })
        
        return scores
    
    def _prepare_trades_data(self, executed_orders: List[Dict]) -> Dict:
        """Prepare trade markers data"""
        
        entries = []
        exits = []
        
        for order in executed_orders:
            timestamp = order.get('timestamp')
            if timestamp:
                time_val = int(timestamp.timestamp()) if hasattr(timestamp, 'timestamp') else timestamp
                
                marker = {
                    'time': time_val,
                    'position': 'aboveBar' if order['side'] == 'BUY' else 'belowBar',
                    'color': '#26a69a' if order['side'] == 'BUY' else '#ef5350',
                    'shape': 'arrowUp' if order['side'] == 'BUY' else 'arrowDown',
                    'text': order['side'][:1]  # 'B' or 'S'
                }
                
                if order.get('signal_type') == 'EXIT':
                    exits.append(marker)
                else:
                    entries.append(marker)
        
        return {'entries': entries, 'exits': exits}
    
    def _generate_html(self, symbol: str, timeframes_data: Dict, default_tf: str,
                       cvd_data: Dict, strategy_scores: List, trades_data: Dict,
                       results: Dict) -> str:
        """Generate the HTML dashboard with multiple synchronized charts"""
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SqueezeFlow Multi-Pane Dashboard - {symbol}</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
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
        }}
        
        .header h1 {{
            font-size: 18px;
            font-weight: 500;
            color: #d1d4dc;
        }}
        
        .timeframe-selector {{
            display: flex;
            gap: 5px;
        }}
        
        .tf-button {{
            padding: 6px 12px;
            background: #2a2e39;
            color: #d1d4dc;
            border: 1px solid #363c4e;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        
        .tf-button:hover {{
            background: #363c4e;
        }}
        
        .tf-button.active {{
            background: #2962ff;
            border-color: #2962ff;
            color: white;
        }}
        
        .charts-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .chart-pane {{
            position: relative;
            border-bottom: 1px solid #2a2e39;
        }}
        
        .chart-pane:last-child {{
            border-bottom: none;
        }}
        
        .pane-label {{
            position: absolute;
            top: 5px;
            left: 10px;
            background: rgba(19, 23, 34, 0.9);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            color: #848e9c;
            z-index: 10;
            pointer-events: none;
        }}
        
        #price-chart {{
            height: 55%;
        }}
        
        #volume-chart {{
            height: 15%;
        }}
        
        #cvd-chart {{
            height: 15%;
        }}
        
        #score-chart {{
            height: 15%;
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
            <h1>{symbol} - SqueezeFlow Multi-Pane Dashboard</h1>
            <div class="timeframe-selector">
                {' '.join([f'<button class="tf-button{" active" if tf == default_tf else ""}" data-tf="{tf}">{tf.upper()}</button>' for tf in timeframes_data.keys()])}
            </div>
        </div>
        
        <div class="charts-container">
            <div id="price-chart" class="chart-pane">
                <div class="pane-label">PRICE</div>
            </div>
            <div id="volume-chart" class="chart-pane">
                <div class="pane-label">VOLUME</div>
            </div>
            <div id="cvd-chart" class="chart-pane">
                <div class="pane-label">CVD (Spot vs Futures)</div>
            </div>
            <div id="score-chart" class="chart-pane">
                <div class="pane-label">STRATEGY SCORE</div>
            </div>
        </div>
        
        <div class="stats-panel">
            <div class="stat-row">
                <span class="stat-label">Total Return:</span>
                <span class="stat-value {('positive' if results.get('total_return', 0) >= 0 else 'negative')}">{results.get('total_return', 0):.2f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Trades:</span>
                <span class="stat-value">{results.get('total_trades', 0)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Win Rate:</span>
                <span class="stat-value">{results.get('win_rate', 0):.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Final Balance:</span>
                <span class="stat-value">${results.get('final_balance', 0):,.2f}</span>
            </div>
        </div>
    </div>
    
    <script>
        // Store all data
        const allTimeframesData = {json.dumps(timeframes_data)};
        const cvdData = {json.dumps(cvd_data)};
        const strategyScores = {json.dumps(strategy_scores)};
        const tradesData = {json.dumps(trades_data)};
        
        // Chart instances
        let priceChart, volumeChart, cvdChart, scoreChart;
        let priceSeries, volumeSeries, spotCvdSeries, futuresCvdSeries, scoreSeries;
        let minEntryLine, goodEntryLine;  // Declare threshold lines globally
        let currentTimeframe = '{default_tf}';
        
        // Common chart options
        const commonOptions = {{
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
                vertLine: {{
                    width: 1,
                    color: '#758696',
                    style: LightweightCharts.LineStyle.Dashed,
                }},
                horzLine: {{
                    visible: false,
                    labelVisible: false,
                }},
            }},
        }};
        
        function initCharts() {{
            // Get container dimensions
            const priceContainer = document.getElementById('price-chart');
            const volumeContainer = document.getElementById('volume-chart');
            const cvdContainer = document.getElementById('cvd-chart');
            const scoreContainer = document.getElementById('score-chart');
            
            // Create PRICE chart (main chart with full features)
            priceChart = LightweightCharts.createChart(priceContainer, {{
                ...commonOptions,
                width: priceContainer.clientWidth,
                height: priceContainer.clientHeight,
                rightPriceScale: {{
                    borderColor: '#2a2e39',
                    visible: true,
                }},
                timeScale: {{
                    borderColor: '#2a2e39',
                    visible: true,
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});
            
            // Create VOLUME chart (no time axis)
            volumeChart = LightweightCharts.createChart(volumeContainer, {{
                ...commonOptions,
                width: volumeContainer.clientWidth,
                height: volumeContainer.clientHeight,
                rightPriceScale: {{
                    borderColor: '#2a2e39',
                    visible: true,
                }},
                timeScale: {{
                    visible: false,  // Hide time axis for volume
                }},
            }});
            
            // Create CVD chart (no time axis)
            cvdChart = LightweightCharts.createChart(cvdContainer, {{
                ...commonOptions,
                width: cvdContainer.clientWidth,
                height: cvdContainer.clientHeight,
                rightPriceScale: {{
                    borderColor: '#2a2e39',
                    visible: true,
                }},
                timeScale: {{
                    visible: false,  // Hide time axis for CVD
                }},
            }});
            
            // Create SCORE chart (no time axis)
            scoreChart = LightweightCharts.createChart(scoreContainer, {{
                ...commonOptions,
                width: scoreContainer.clientWidth,
                height: scoreContainer.clientHeight,
                rightPriceScale: {{
                    borderColor: '#2a2e39',
                    visible: true,
                }},
                timeScale: {{
                    visible: false,  // Hide time axis for score
                }},
            }});
            
            // Add series to each chart - USING CORRECT API
            // Check if new API is available (v4+)
            if (typeof priceChart.addSeries === 'function' && typeof LightweightCharts.CandlestickSeries !== 'undefined') {{
                // NEW API (v4+) - This is what actually works!
                priceSeries = priceChart.addSeries(LightweightCharts.CandlestickSeries, {{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderUpColor: '#26a69a',
                    borderDownColor: '#ef5350',
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
            }} else {{
                // OLD API (v3) - Fallback
                priceSeries = priceChart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderUpColor: '#26a69a',
                    borderDownColor: '#ef5350',
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
            }}
            
            // Volume series with API check
            if (typeof volumeChart.addSeries === 'function' && typeof LightweightCharts.HistogramSeries !== 'undefined') {{
                volumeSeries = volumeChart.addSeries(LightweightCharts.HistogramSeries, {{
                    color: '#26a69a',
                    priceFormat: {{
                        type: 'volume',
                    }},
                }});
            }} else {{
                volumeSeries = volumeChart.addHistogramSeries({{
                    color: '#26a69a',
                    priceFormat: {{
                        type: 'volume',
                    }},
                }});
            }}
            
            // CVD series with API check
            if (typeof cvdChart.addSeries === 'function' && typeof LightweightCharts.LineSeries !== 'undefined') {{
                spotCvdSeries = cvdChart.addSeries(LightweightCharts.LineSeries, {{
                    color: '#2962ff',
                    lineWidth: 2,
                    title: 'Spot CVD',
                }});
                
                futuresCvdSeries = cvdChart.addSeries(LightweightCharts.LineSeries, {{
                    color: '#ff9800',
                    lineWidth: 2,
                    title: 'Futures CVD',
                }});
            }} else {{
                spotCvdSeries = cvdChart.addLineSeries({{
                    color: '#2962ff',
                    lineWidth: 2,
                    title: 'Spot CVD',
                }});
                
                futuresCvdSeries = cvdChart.addLineSeries({{
                    color: '#ff9800',
                    lineWidth: 2,
                    title: 'Futures CVD',
                }});
            }}
            
            // Score series with API check
            if (typeof scoreChart.addSeries === 'function' && typeof LightweightCharts.LineSeries !== 'undefined') {{
                scoreSeries = scoreChart.addSeries(LightweightCharts.LineSeries, {{
                    color: '#9c27b0',
                    lineWidth: 2,
                    title: 'Score',
                }});
                
                // Add threshold lines to score chart
                minEntryLine = scoreChart.addSeries(LightweightCharts.LineSeries, {{
                    color: 'rgba(255, 255, 255, 0.2)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    crosshairMarkerVisible: false,
                }});
                
                goodEntryLine = scoreChart.addSeries(LightweightCharts.LineSeries, {{
                    color: 'rgba(76, 175, 80, 0.3)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    crosshairMarkerVisible: false,
                }});
            }} else {{
                scoreSeries = scoreChart.addLineSeries({{
                    color: '#9c27b0',
                    lineWidth: 2,
                    title: 'Score',
                }});
                
                // Add threshold lines to score chart
                minEntryLine = scoreChart.addLineSeries({{
                    color: 'rgba(255, 255, 255, 0.2)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    crosshairMarkerVisible: false,
                }});
                
                goodEntryLine = scoreChart.addLineSeries({{
                    color: 'rgba(76, 175, 80, 0.3)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    crosshairMarkerVisible: false,
                }});
            }}
            
            // Load initial data
            updateCharts(currentTimeframe);
            
            // Synchronize charts
            synchronizeCharts();
            
            // Handle window resize
            window.addEventListener('resize', () => {{
                priceChart.applyOptions({{ width: priceContainer.clientWidth }});
                volumeChart.applyOptions({{ width: volumeContainer.clientWidth }});
                cvdChart.applyOptions({{ width: cvdContainer.clientWidth }});
                scoreChart.applyOptions({{ width: scoreContainer.clientWidth }});
            }});
        }}
        
        function synchronizeCharts() {{
            // Sync visible time range
            function syncTimeRange(sourceChart, targetCharts) {{
                sourceChart.timeScale().subscribeVisibleTimeRangeChange(() => {{
                    const range = sourceChart.timeScale().getVisibleRange();
                    if (range) {{
                        targetCharts.forEach(chart => {{
                            chart.timeScale().setVisibleRange(range);
                        }});
                    }}
                }});
            }}
            
            // Sync from price chart to all others
            syncTimeRange(priceChart, [volumeChart, cvdChart, scoreChart]);
            
            // Sync crosshair
            function syncCrosshair(sourceChart, targetCharts) {{
                sourceChart.subscribeCrosshairMove(param => {{
                    targetCharts.forEach(chart => {{
                        chart.setCrosshairPosition(param.point?.x || 0, param.point?.y || 0, false);
                    }});
                }});
            }}
            
            // Sync crosshair from all charts
            syncCrosshair(priceChart, [volumeChart, cvdChart, scoreChart]);
            syncCrosshair(volumeChart, [priceChart, cvdChart, scoreChart]);
            syncCrosshair(cvdChart, [priceChart, volumeChart, scoreChart]);
            syncCrosshair(scoreChart, [priceChart, volumeChart, cvdChart]);
        }}
        
        function updateCharts(timeframe) {{
            const data = allTimeframesData[timeframe];
            if (!data) return;
            
            // Update price chart
            priceSeries.setData(data.candles);
            
            // Update volume chart
            volumeSeries.setData(data.volumes);
            
            // Update CVD chart
            if (cvdData.spot.length > 0) {{
                spotCvdSeries.setData(cvdData.spot);
            }}
            if (cvdData.futures.length > 0) {{
                futuresCvdSeries.setData(cvdData.futures);
            }}
            
            // Update score chart
            if (strategyScores.length > 0) {{
                scoreSeries.setData(strategyScores);
                
                // Add threshold lines (minEntryLine and goodEntryLine are declared above)
                const times = strategyScores.map(d => d.time);
                if (typeof minEntryLine !== 'undefined' && minEntryLine) {{
                    minEntryLine.setData(times.map(t => ({{ time: t, value: 4.0 }})));
                }}
                if (typeof goodEntryLine !== 'undefined' && goodEntryLine) {{
                    goodEntryLine.setData(times.map(t => ({{ time: t, value: 6.0 }})));
                }}
            }}
            
            // Add trade markers to price chart
            if (tradesData.entries.length > 0) {{
                priceSeries.setMarkers([...tradesData.entries, ...tradesData.exits]);
            }}
            
            // Fit content
            priceChart.timeScale().fitContent();
        }}
        
        // Timeframe button handlers
        document.querySelectorAll('.tf-button').forEach(button => {{
            button.addEventListener('click', (e) => {{
                // Update active state
                document.querySelectorAll('.tf-button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                // Update charts
                currentTimeframe = e.target.dataset.tf;
                updateCharts(currentTimeframe);
            }});
        }});
        
        // Initialize on load
        initCharts();
    </script>
</body>
</html>
"""
    
    def _create_empty_dashboard(self, output_dir: Path) -> str:
        """Create an empty dashboard when no data is available"""
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>SqueezeFlow Dashboard - No Data</title>
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
    <h1>No data available for visualization</h1>
</body>
</html>
"""
        
        dashboard_path = output_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return str(dashboard_path)