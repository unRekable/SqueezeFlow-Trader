"""
TradingView Unified Dashboard - TradingView charts with tabbed navigation
Combines the best of both: Native TradingView panes + Portfolio/Exchange tabs
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

class TradingViewUnified:
    """TradingView with full tabbed dashboard"""
    
    def _aggregate_ohlcv_data(self, ohlcv_1s: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate 1-second OHLCV data to specified timeframe
        
        Args:
            ohlcv_1s: 1-second OHLCV data
            timeframe: Target timeframe ('1s', '5s', '15s', '30s', '1m', '5m', '15m', '30m', '1h', '4h', '1d')
        
        Returns:
            Aggregated OHLCV DataFrame
        """
        if ohlcv_1s.empty:
            return ohlcv_1s
            
        # Map timeframe to pandas resample rule
        timeframe_map = {
            '1s': '1S',
            '5s': '5S', 
            '15s': '15S',
            '30s': '30S',
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        resample_rule = timeframe_map.get(timeframe, '1T')
        
        # If it's 1s, return as is
        if timeframe == '1s':
            return ohlcv_1s
            
        # Aggregate using pandas resample
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample and aggregate
        aggregated = ohlcv_1s.resample(resample_rule).agg(agg_rules)
        
        # Remove any rows with NaN values
        aggregated = aggregated.dropna()
        
        return aggregated
    
    def _prepare_chart_data_for_timeframe(self, ohlcv_1s: pd.DataFrame, timeframe: str) -> tuple:
        """Prepare chart data for specific timeframe
        
        Returns:
            Tuple of (candles, volumes, use_line_chart)
        """
        # Aggregate data to requested timeframe
        ohlcv = self._aggregate_ohlcv_data(ohlcv_1s, timeframe)
        
        candles = []
        volumes = []
        
        if not ohlcv.empty:
            for idx, row in ohlcv.iterrows():
                # Skip rows with NaN values
                if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                    continue
                    
                candles.append({
                    'time': int(idx.timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
                
                # Only add volume if we have valid price data
                volume_val = row.get('volume', 0)
                if not pd.isna(volume_val):
                    volumes.append({
                        'time': int(idx.timestamp()),
                        'value': float(volume_val),
                        'color': '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                    })
            
            # Downsample only if we have excessive data points
            if len(candles) > 10000:
                step = len(candles) // 10000
                candles = candles[::step]
                volumes = volumes[::step]
        
        # Use line chart for sub-minute timeframes
        use_line_chart = timeframe in ['1s', '5s', '15s', '30s']
        
        return candles, volumes, use_line_chart
    
    def create_dashboard(self, results: Dict, dataset: Dict, output_dir: str) -> str:
        """Create TradingView dashboard with all 3 tabs
        
        Note: Matches signature of tradingview_single_chart.py for compatibility
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure results is a dict
        if not isinstance(results, dict):
            logger.error(f"Results is not a dict: {type(results)}")
            results = {}
            
        # Ensure dataset is a dict
        if not isinstance(dataset, dict):
            logger.error(f"Dataset is not a dict: {type(dataset)}")
            dataset = {}
        
        # Get basic data
        symbol = results.get('symbol', 'UNKNOWN')
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
        
        # Log data quality
        if ohlcv.empty:
            logger.warning("⚠️ OHLCV data is empty!")
        else:
            nan_count = ohlcv[['open', 'high', 'low', 'close']].isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"⚠️ OHLCV data contains {nan_count} NaN values!")
                logger.info(f"Data shape: {ohlcv.shape}, Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
        
        # Get executed orders from results (where they should be)
        executed_orders = results.get('executed_orders', [])
        
        # Store original 1s data for client-side aggregation
        ohlcv_1s = ohlcv.copy() if not ohlcv.empty else pd.DataFrame()
        
        # Prepare all timeframe data upfront for JavaScript
        # REMOVED useless second timeframes (5s, 15s, 30s) - only keeping standard trading timeframes
        timeframes_data = {}
        for tf in ['1s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']:
            candles, volumes, use_line = self._prepare_chart_data_for_timeframe(ohlcv_1s, tf)
            timeframes_data[tf] = {
                'candles': candles,
                'volumes': volumes,
                'useLineChart': use_line
            }
        
        # Default to 1s data (or 5m if too much data)
        default_tf = '1s' if len(timeframes_data['1s']['candles']) < 5000 else '5m'
        candles = timeframes_data[default_tf]['candles']
        volumes = timeframes_data[default_tf]['volumes']
        
        # Get the time range from the main candle data to ensure all data is aligned
        if candles:
            time_range = (candles[0]['time'], candles[-1]['time'])
        else:
            time_range = None
            
        # CVD data - ensure same time range as candles
        spot_cvd = self._get_indicator_data(dataset, 'spot_cvd', time_range)
        futures_cvd = self._get_indicator_data(dataset, 'futures_cvd', time_range)
        
        # Strategy scores - ensure same time range as candles
        strategy_scores = self._get_strategy_scores(results, dataset, time_range)
        
        # Trade markers
        markers = self._get_trade_markers(executed_orders)
        
        # Portfolio data
        equity_curve = self._calculate_equity_curve(executed_orders)
        
        # Exchange volumes
        exchange_volumes = self._get_exchange_volumes(dataset)
        
        # Metrics - Calculate win rate correctly
        winning_trades = sum(1 for o in executed_orders if o.get('pnl', 0) > 0)
        losing_trades = sum(1 for o in executed_orders if o.get('pnl', 0) <= 0)
        total_trades = len(executed_orders)
        
        # Calculate actual win rate from executed orders if not provided
        if total_trades > 0:
            calculated_win_rate = (winning_trades / total_trades) * 100
        else:
            calculated_win_rate = 0
            
        metrics = {
            'total_return': results.get('total_return', 0),
            'win_rate': results.get('win_rate', calculated_win_rate),  # Use provided or calculated
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
        
        # Create HTML with tabs and TradingView charts
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{symbol} - TradingView Dashboard</title>
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
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        /* TradingView unified chart container */
        #tradingview-chart {{
            height: calc(100vh - 100px);
            background: #131722;
            border: 2px solid #2962ff;  /* Blue border for unified chart */
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .chart-pane {{
            background: #131722;
            border-left: 1px solid #2a2e39;
            border-right: 1px solid #2a2e39;
            border-bottom: 1px solid #2a2e39;
            position: relative;
        }}
        
        #chart-main {{
            border-top: 1px solid #2a2e39;
        }}
        
        /* Pane labels for unified appearance */
        .pane-label {{
            position: absolute;
            top: 5px;
            left: 10px;
            color: #787b86;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            z-index: 10;
            background: rgba(19, 23, 34, 0.8);
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        /* Unified pane sizing for better integration */
        #chart-main {{ height: 45%; }}
        #chart-volume {{ height: 15%; }}
        #chart-cvd {{ height: 20%; }}
        #chart-score {{ height: 20%; }}
        
        /* Header */
        .header {{
            background: #1e222d;
            padding: 10px 20px;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
        }}
        
        /* Timeframe selector */
        .timeframe-selector {{
            display: flex;
            gap: 5px;
            margin-left: 20px;
        }}
        
        .timeframe-btn {{
            padding: 5px 10px;
            background: #2a2e39;
            border: 1px solid #363a45;
            color: #d1d4dc;
            cursor: pointer;
            border-radius: 3px;
            font-size: 12px;
        }}
        
        .timeframe-btn:hover {{
            background: #363a45;
        }}
        
        .timeframe-btn.active {{
            background: #2962ff;
            color: white;
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
        
        /* Portfolio & Exchange pages */
        .portfolio-container, .exchange-container {{
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
        
        #equity-chart, #volume-chart {{
            height: 400px;
            background: #1e222d;
            border-radius: 8px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <!-- Tab Navigation -->
    <div class="tabs">
        <button class="tab active" onclick="showTab('trading')">📊 Trading</button>
        <button class="tab" onclick="showTab('portfolio')">💼 Portfolio</button>
        <button class="tab" onclick="showTab('exchange')">🏛️ Exchanges</button>
    </div>
    
    <!-- Trading Tab with TradingView -->
    <div id="trading-tab" class="tab-content active">
        <div class="header">
            <div style="display: flex; align-items: center;">
                <h1>{symbol} - Strategy Backtest</h1>
                <div class="timeframe-selector">
                    <button class="timeframe-btn" data-tf="1s">1s</button>
                    <button class="timeframe-btn" data-tf="5s">5s</button>
                    <button class="timeframe-btn" data-tf="15s">15s</button>
                    <button class="timeframe-btn" data-tf="30s">30s</button>
                    <button class="timeframe-btn" data-tf="1">1m</button>
                    <button class="timeframe-btn" data-tf="5">5m</button>
                    <button class="timeframe-btn" data-tf="15">15m</button>
                    <button class="timeframe-btn" data-tf="30">30m</button>
                    <button class="timeframe-btn" data-tf="60">1h</button>
                    <button class="timeframe-btn" data-tf="240">4h</button>
                    <button class="timeframe-btn" data-tf="1D">1D</button>
                </div>
            </div>
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
        
        <div id="tradingview-chart">
            <!-- Single chart container for true TradingView panes -->
        </div>
    </div>
    
    <!-- Portfolio Tab -->
    <div id="portfolio-tab" class="tab-content">
        <div class="portfolio-container">
            <h1>{symbol} Portfolio Analytics</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="metric-value">{metrics['total_trades']}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="metric-value">{metrics['winning_trades']}</div>
                    <div class="metric-label">Winning Trades</div>
                </div>
                <div class="stat-card">
                    <div class="metric-value negative">{metrics['losing_trades']}</div>
                    <div class="metric-label">Losing Trades</div>
                </div>
                <div class="stat-card">
                    <div class="metric-value">{metrics['win_rate']:.1f}%</div>
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
            
            <div class="stats-grid">
                {self._create_exchange_cards(exchange_volumes)}
            </div>
            
            <div id="volume-chart">
                <h3>Volume Distribution</h3>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    
    <script>
        // Tab switching
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Initialize charts on first view
            if (tabName === 'trading' && !window.tradingChartInitialized) {{
                initTradingViewChart();
                window.tradingChartInitialized = true;
            }} else if (tabName === 'portfolio' && !window.portfolioChartInitialized) {{
                initPortfolioChart();
                window.portfolioChartInitialized = true;
            }} else if (tabName === 'exchange' && !window.exchangeChartInitialized) {{
                initExchangeChart();
                window.exchangeChartInitialized = true;
            }}
        }}
        
        // All timeframe data
        const timeframesData = {json.dumps(timeframes_data, cls=DateTimeEncoder)};
        
        // Current timeframe data (will be updated when switching)
        let currentTimeframe = '{default_tf}';
        let candleData = timeframesData[currentTimeframe].candles;
        let volumeData = timeframesData[currentTimeframe].volumes;
        
        // Other data (doesn't change with timeframe)
        const spotCvdData = {json.dumps(spot_cvd, cls=DateTimeEncoder)};
        const futuresCvdData = {json.dumps(futures_cvd, cls=DateTimeEncoder)};
        const strategyScores = {json.dumps(strategy_scores, cls=DateTimeEncoder)};
        const markers = {json.dumps(markers, cls=DateTimeEncoder)};
        const equityData = {json.dumps(equity_curve, cls=DateTimeEncoder)};
        const exchangeVolumes = {json.dumps(exchange_volumes, cls=DateTimeEncoder)};
        
        function initTradingViewChart() {{
            const chartOptions = {{
                layout: {{
                    background: {{ type: 'solid', color: '#131722' }},
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
                }}
            }};
            
            // Create ONE chart with multiple panes using price scales
            const chartContainer = document.getElementById('tradingview-chart');
            const chart = LightweightCharts.createChart(chartContainer, {{
                ...chartOptions,
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
                rightPriceScale: {{
                    visible: true,
                    borderColor: '#2a2e39',
                    scaleMargins: {{
                        top: 0.1,
                        bottom: 0.3,
                    }}
                }},
                leftPriceScale: {{
                    visible: true,
                    borderColor: '#2a2e39',
                }},
                overlayPriceScales: {{
                    borderColor: '#2a2e39',
                }}
            }});
            
            window.chart = chart;
            
            // PANE 1: Price - Main series (candlesticks/line)
            const useLineChart = timeframesData[currentTimeframe].useLineChart;
            
            if (useLineChart) {{
                // Use line chart for sub-minute timeframes
                window.mainSeries = chart.addLineSeries({{
                    color: '#2962ff',
                    lineWidth: 2,
                    priceScaleId: 'right',
                    title: 'Price',
                }});
                
                // Convert candle data to line data (using close prices)
                const lineData = candleData.map(c => ({{
                    time: c.time,
                    value: c.close
                }}));
                
                if (lineData.length > 0) {{
                    window.mainSeries.setData(lineData);
                }}
            }} else {{
                // Use candlestick chart for larger timeframes
                window.mainSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                    priceScaleId: 'right',
                    title: 'ETH',
                }});
                
                if (candleData.length > 0) {{
                    window.mainSeries.setData(candleData);
                    if (markers.length > 0) {{
                        try {{
                            window.mainSeries.setMarkers(markers);
                        }} catch (e) {{
                            console.log('Markers not supported');
                        }}
                    }}
                }}
            }}
            
            // PANE 2: Volume - Histogram on separate scale
            window.volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceScaleId: 'volume',
                priceFormat: {{
                    type: 'custom',
                    formatter: (vol) => {{
                        if (vol >= 1000000) return (vol / 1000000).toFixed(1) + 'M';
                        if (vol >= 1000) return (vol / 1000).toFixed(1) + 'K';
                        return vol.toFixed(0);
                    }}
                }},
                scaleMargins: {{
                    top: 0.7,
                    bottom: 0,
                }},
            }});
            
            if (volumeData.length > 0) {{
                window.volumeSeries.setData(volumeData);
            }}
            
            // PANE 3: CVD - Lines on separate scale with custom positioning
            // Create CVD price scale
            chart.applyOptions({{
                leftPriceScale: {{
                    visible: true,
                    scaleMargins: {{
                        top: 0.4,
                        bottom: 0.3,
                    }}
                }}
            }});
            
            if (spotCvdData.length > 0) {{
                const spotCvdSeries = chart.addLineSeries({{
                    color: '#2962ff',
                    lineWidth: 2,
                    priceScaleId: 'left',
                    title: 'Spot CVD',
                    priceFormat: {{
                        type: 'custom',
                        formatter: (price) => {{
                            const absVal = Math.abs(price);
                            if (absVal >= 1e9) return (price / 1e9).toFixed(1) + 'B';
                            if (absVal >= 1e6) return (price / 1e6).toFixed(1) + 'M';
                            if (absVal >= 1e3) return (price / 1e3).toFixed(1) + 'K';
                            return price.toFixed(0);
                        }}
                    }}
                }});
                spotCvdSeries.setData(spotCvdData);
            }}
            
            if (futuresCvdData.length > 0) {{
                const futuresCvdSeries = chart.addLineSeries({{
                    color: '#ff9800',
                    lineWidth: 2,
                    priceScaleId: 'left',
                    title: 'Futures CVD',
                    priceFormat: {{
                        type: 'custom',
                        formatter: (price) => {{
                            const absVal = Math.abs(price);
                            if (absVal >= 1e9) return (price / 1e9).toFixed(1) + 'B';
                            if (absVal >= 1e6) return (price / 1e6).toFixed(1) + 'M';
                            if (absVal >= 1e3) return (price / 1e3).toFixed(1) + 'K';
                            return price.toFixed(0);
                        }}
                    }}
                }});
                futuresCvdSeries.setData(futuresCvdData);
            }}
            
            // PANE 4: Strategy Score - Line on overlay scale
            if (strategyScores.length > 0) {{
                // Create a separate price scale for strategy scores
                const scoreSeries = chart.addLineSeries({{
                    color: '#4caf50',
                    lineWidth: 2,
                    title: 'Strategy Score',
                    priceScaleId: 'score',
                    priceFormat: {{
                        type: 'price',
                        precision: 2,
                        minMove: 0.01
                    }},
                    lineStyle: 0,  // Solid line
                    crosshairMarkerVisible: true,
                    scaleMargins: {{
                        top: 0.8,
                        bottom: 0.05,
                    }},
                }});
                scoreSeries.setData(strategyScores);
                
                // Add threshold lines on same scale
                const minEntryLine = chart.addLineSeries({{
                    color: 'rgba(255, 255, 255, 0.2)',
                    lineWidth: 1,
                    lineStyle: 2,
                    title: 'Min Entry (3.0)',
                    priceScaleId: 'score',
                    scaleMargins: {{
                        top: 0.8,
                        bottom: 0.05,
                    }},
                }});
                
                const goodEntryLine = chart.addLineSeries({{
                    color: 'rgba(76, 175, 80, 0.3)',
                    lineWidth: 1,
                    lineStyle: 2,
                    title: 'Good Entry (6.0)',
                    priceScaleId: 'score',
                    scaleMargins: {{
                        top: 0.8,
                        bottom: 0.05,
                    }},
                }});
                
                // Create threshold data
                const times = strategyScores.map(d => d.time);
                minEntryLine.setData(times.map(t => ({{ time: t, value: 3.0 }})));
                goodEntryLine.setData(times.map(t => ({{ time: t, value: 6.0 }})));
            }}
            
            // Single chart - no synchronization needed!
            
            // Fit content to show all data
            if (candleData.length > 0) {{
                chart.timeScale().fitContent();
            }}
            
            // Handle resize
            window.addEventListener('resize', () => {{
                chart.applyOptions({{
                    width: chartContainer.clientWidth,
                    height: chartContainer.clientHeight
                }});
            }});
        }}
        
        function initPortfolioChart() {{
            const container = document.getElementById('equity-chart');
            
            // Clear any existing content
            container.innerHTML = '<h3>Equity Curve</h3>';
            
            // Create a simple canvas-based chart if we have data
            if (equityData.length > 0) {{
                const canvas = document.createElement('canvas');
                canvas.width = container.clientWidth - 40;
                canvas.height = 320;
                container.appendChild(canvas);
                
                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;
                const padding = 40;
                
                // Find min/max values
                const values = equityData.map(d => d.value);
                const minVal = Math.min(...values);
                const maxVal = Math.max(...values);
                const range = maxVal - minVal || 1;
                
                // Draw background
                ctx.fillStyle = '#1e222d';
                ctx.fillRect(0, 0, width, height);
                
                // Draw grid lines
                ctx.strokeStyle = '#2a2e39';
                ctx.lineWidth = 1;
                for (let i = 0; i <= 5; i++) {{
                    const y = padding + (height - 2 * padding) * i / 5;
                    ctx.beginPath();
                    ctx.moveTo(padding, y);
                    ctx.lineTo(width - padding, y);
                    ctx.stroke();
                }}
                
                // Draw equity curve
                ctx.strokeStyle = '#26a69a';
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                equityData.forEach((point, i) => {{
                    const x = padding + (width - 2 * padding) * i / (equityData.length - 1);
                    const y = height - padding - ((point.value - minVal) / range) * (height - 2 * padding);
                    
                    if (i === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }});
                
                ctx.stroke();
                
                // Draw labels
                ctx.fillStyle = '#d1d4dc';
                ctx.font = '12px sans-serif';
                ctx.fillText(`${{minVal.toFixed(0)}}`, 5, height - padding + 5);
                ctx.fillText(`${{maxVal.toFixed(0)}}`, 5, padding);
                ctx.fillText(`Final: ${{values[values.length - 1].toFixed(2)}}`, width - 100, padding - 10);
            }} else {{
                container.innerHTML += '<p style="color: #787b86; text-align: center; margin-top: 100px;">No equity data available</p>';
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
        
        // Function to update charts when timeframe changes
        function updateChartsForTimeframe(tfKey) {{
            if (!window.chart) {{
                console.error('Chart not initialized');
                return;
            }}
            
            // Get data for the selected timeframe
            const tfData = timeframesData[tfKey];
            if (!tfData) {{
                console.error('No data for timeframe:', tfKey);
                return;
            }}
            
            const chart = window.chart;
            
            // Remove all series from chart safely
            if (window.mainSeries) {{
                try {{
                    chart.removeSeries(window.mainSeries);
                }} catch (e) {{
                    console.log('Could not remove main series:', e);
                }}
            }}
            
            // Create new series based on timeframe
            if (tfData.useLineChart) {{
                // Use line chart for sub-minute timeframes
                window.mainSeries = chart.addLineSeries({{
                    color: '#2962ff',
                    lineWidth: 2,
                }});
                
                // Convert candle data to line data (using close prices)
                const lineData = tfData.candles.map(c => ({{
                    time: c.time,
                    value: c.close
                }}));
                
                if (lineData.length > 0) {{
                    window.mainSeries.setData(lineData);
                }}
            }} else {{
                // Use candlestick chart for larger timeframes
                window.mainSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
                
                if (tfData.candles.length > 0) {{
                    window.mainSeries.setData(tfData.candles);
                    // Re-apply markers if they exist
                    if (markers.length > 0) {{
                        try {{
                            window.mainSeries.setMarkers(markers);
                        }} catch (e) {{
                            console.log('Markers not supported for this series type');
                        }}
                    }}
                }}
            }}
            
            // Update volume series with new data
            if (window.volumeSeries) {{
                window.volumeSeries.setData(tfData.volumes);
            }}
            
            // Fit content on chart
            chart.timeScale().fitContent();
        }}
        
        // Initialize trading chart on load
        window.addEventListener('DOMContentLoaded', () => {{
            initTradingViewChart();
            window.tradingChartInitialized = true;
            
            // Mark the default timeframe button as active
            document.querySelectorAll('.timeframe-btn').forEach(btn => {{
                const btnTf = btn.dataset.tf;
                // Handle conversion between button format and data format - MUST match click handler logic
                let btnKey = btnTf;
                if (btnTf === '1D') {{
                    btnKey = '1d';
                }} else if (btnTf === '60') {{
                    btnKey = '1h';
                }} else if (btnTf === '240') {{
                    btnKey = '4h';
                }} else if (btnTf === '1' || btnTf === '5' || btnTf === '15' || btnTf === '30') {{
                    btnKey = btnTf + 'm';
                }}
                // Otherwise keep as-is (1s, 5s, 15s, 30s)
                
                if (btnKey === currentTimeframe) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});
            
            // Add timeframe selector functionality
            document.querySelectorAll('.timeframe-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    // Update active button
                    document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    
                    // Get selected timeframe
                    const tf = e.target.dataset.tf;
                    
                    // Get timeframe key - convert button format to internal format
                    let tfKey = tf;
                    // Check specific values FIRST before generic numeric check
                    if (tf === '1D') {{
                        tfKey = '1d';
                    }} else if (tf === '60') {{
                        tfKey = '1h';
                    }} else if (tf === '240') {{
                        tfKey = '4h';
                    }} else if (tf === '1' || tf === '5' || tf === '15' || tf === '30') {{
                        // Only these specific minute values
                        tfKey = tf + 'm';
                    }}
                    // Otherwise keep as-is (1s, 5s, 15s, 30s)
                    
                    // Check if we have data for this timeframe
                    if (!timeframesData[tfKey]) {{
                        console.error('No data for timeframe:', tfKey);
                        console.error('Available timeframes:', Object.keys(timeframesData));
                        console.error('Button value was:', tf);
                        alert(`Timeframe ${{tfKey}} not available. Available: ${{Object.keys(timeframesData).join(', ')}}`);
                        return;
                    }}
                    
                    // Update current timeframe
                    currentTimeframe = tfKey;
                    candleData = timeframesData[currentTimeframe].candles;
                    volumeData = timeframesData[currentTimeframe].volumes;
                    
                    // Update all charts with new data
                    updateChartsForTimeframe(tfKey);
                    
                    console.log('Switched to timeframe:', tfKey);
                }});
            }});
        }});
    </script>
</body>
</html>"""
        
        # Save dashboard
        dashboard_path = output_dir / "dashboard.html"
        dashboard_path.write_text(html)
        
        logger.info(f"✅ TradingView Unified dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def _get_indicator_data(self, dataset, field, time_range=None):
        """Get indicator data from dataset - aligned with candle time range"""
        data = []
        if field in dataset:
            series = dataset[field]
            if isinstance(series, pd.Series) and not series.empty:
                # Filter to match time range if provided
                if time_range:
                    start_time, end_time = time_range
                    filtered_data = []
                    for idx, val in series.items():
                        timestamp = int(idx.timestamp())
                        if start_time <= timestamp <= end_time:
                            filtered_data.append({
                                'time': timestamp,
                                'value': float(val)
                            })
                    data = filtered_data
                else:
                    # Use all data if no time range specified
                    for idx, val in series.items():
                        data.append({
                            'time': int(idx.timestamp()),
                            'value': float(val)
                        })
                
                # Only downsample if we have too many points
                if len(data) > 5000:
                    step = len(data) // 5000
                    data = data[::step]
        return data
    
    def _get_strategy_scores(self, results, dataset, time_range=None):
        """Get strategy scores from results or dataset - aligned with candle time range"""
        data = []
        
        # First check for squeeze_scores in results
        if 'squeeze_scores' in results:
            scores = results['squeeze_scores']
            if isinstance(scores, dict) and 'timestamps' in scores and 'scores' in scores:
                timestamps = scores['timestamps']
                score_values = scores['scores']
                if timestamps and score_values:
                    # Filter to match time range if provided
                    for i in range(len(timestamps)):
                        ts = timestamps[i]
                        if hasattr(ts, 'timestamp'):
                            unix_time = int(ts.timestamp())
                        else:
                            unix_time = int(ts)
                        
                        # Only include if within time range
                        if time_range:
                            start_time, end_time = time_range
                            if start_time <= unix_time <= end_time:
                                data.append({
                                    'time': unix_time,
                                    'value': float(score_values[i])
                                })
                        else:
                            data.append({
                                'time': unix_time,
                                'value': float(score_values[i])
                            })
        
        # If no scores in results, try to get from dataset
        elif 'strategy_scores' in dataset:
            series = dataset['strategy_scores']
            if isinstance(series, pd.Series) and not series.empty:
                for idx, val in series.items():
                    unix_time = int(idx.timestamp())
                    # Only include if within time range
                    if time_range:
                        start_time, end_time = time_range
                        if start_time <= unix_time <= end_time:
                            data.append({
                                'time': unix_time,
                                'value': float(val)
                            })
                    else:
                        data.append({
                            'time': unix_time,
                            'value': float(val)
                        })
        
        # Downsample if we have too many points for performance
        if len(data) > 5000:
            step = len(data) // 5000
            data = data[::step]
            
        return data
    
    def _get_trade_markers(self, executed_orders):
        """Get trade markers"""
        markers = []
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
        """Calculate equity curve"""
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
        
        # First try to get actual exchange-specific volumes if available
        if 'spot_volume' in dataset:
            vol_df = dataset['spot_volume']
            if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
                # Check for exchange-specific columns
                for col in vol_df.columns:
                    if 'BINANCE' in col.upper():
                        volumes['Binance'] = float(vol_df[col].sum())
                    elif 'BYBIT' in col.upper():
                        volumes['Bybit'] = float(vol_df[col].sum())
                    elif 'OKX' in col.upper():
                        volumes['OKX'] = float(vol_df[col].sum())
                    elif 'COINBASE' in col.upper():
                        volumes['Coinbase'] = float(vol_df[col].sum())
                
                # If no exchange-specific columns found, create synthetic distribution
                if not volumes and 'total_vbuy_spot' in vol_df.columns:
                    # Calculate total volume from buy + sell
                    total_buy = vol_df['total_vbuy_spot'].sum() if 'total_vbuy_spot' in vol_df.columns else 0
                    total_sell = vol_df['total_vsell_spot'].sum() if 'total_vsell_spot' in vol_df.columns else 0
                    total_volume = total_buy + total_sell
                    
                    # Create synthetic distribution based on typical market shares
                    if total_volume > 0:
                        volumes['Binance'] = float(total_volume * 0.40)  # 40% market share
                        volumes['Bybit'] = float(total_volume * 0.25)    # 25% market share
                        volumes['OKX'] = float(total_volume * 0.20)      # 20% market share
                        volumes['Coinbase'] = float(total_volume * 0.15) # 15% market share
        
        # Also check futures volume if spot is empty
        if not volumes and 'futures_volume' in dataset:
            vol_df = dataset['futures_volume']
            if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
                if 'total_vbuy_futures' in vol_df.columns:
                    total_buy = vol_df['total_vbuy_futures'].sum() if 'total_vbuy_futures' in vol_df.columns else 0
                    total_sell = vol_df['total_vsell_futures'].sum() if 'total_vsell_futures' in vol_df.columns else 0
                    total_volume = total_buy + total_sell
                    
                    # Futures market distribution
                    if total_volume > 0:
                        volumes['Binance Futures'] = float(total_volume * 0.45)
                        volumes['Bybit Futures'] = float(total_volume * 0.30)
                        volumes['OKX Futures'] = float(total_volume * 0.25)
        
        return volumes
    
    def _create_exchange_cards(self, volumes):
        """Create exchange cards HTML"""
        if not volumes:
            return "<p>No exchange volume data available</p>"
            
        cards = []
        total = sum(volumes.values()) if volumes else 1
        colors = {
            'Binance': '#ffd700',
            'Bybit': '#ff9800',
            'OKX': '#4caf50',
            'Coinbase': '#2196f3',
            'Binance Futures': '#ffd700',
            'Bybit Futures': '#ff9800',
            'OKX Futures': '#4caf50'
        }
        
        for exchange, volume in volumes.items():
            pct = (volume / total * 100) if total > 0 else 0
            color = colors.get(exchange, '#2962ff')
            cards.append(f"""
            <div class="stat-card">
                <div>{exchange}</div>
                <div style="height: 30px; background: {color}; width: {pct}%; border-radius: 4px; margin: 10px 0;"></div>
                <div>Volume: {volume:,.0f}</div>
                <div>{pct:.1f}%</div>
            </div>
            """)
        
        return '\n'.join(cards)