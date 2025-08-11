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
    
    def __init__(self, output_dir: str = "."):
        # Changed default to current directory
        self.output_dir = Path(output_dir)
        self.logger = logger  # Add logger attribute
        
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
        
        # CRITICAL FIX: Data validation and corruption detection
        if not ohlcv.empty:
            logger.info(f"🔍 Validating OHLCV data for {symbol}...")
            
            # Calculate price statistics for outlier detection
            price_cols = ['open', 'high', 'low', 'close']
            all_prices = []
            for col in price_cols:
                if col in ohlcv.columns:
                    all_prices.extend(ohlcv[col].dropna().tolist())
            
            if all_prices:
                price_median = pd.Series(all_prices).median()
                price_std = pd.Series(all_prices).std()
                
                # Define reasonable price bounds (3 standard deviations from median)
                min_reasonable_price = max(0.01, price_median - 3 * price_std)
                max_reasonable_price = price_median + 3 * price_std
                
                # Log corruption detection
                corrupted_lows = ohlcv['low'] < min_reasonable_price
                if corrupted_lows.any():
                    corrupted_count = corrupted_lows.sum()
                    logger.warning(f"🚨 CORRUPTION DETECTED: {corrupted_count} low values below {min_reasonable_price:.2f}")
                    logger.warning(f"   Sample corrupted lows: {ohlcv.loc[corrupted_lows, 'low'].head().tolist()}")
                    
                    # FIX: Replace corrupted values with interpolation
                    for col in price_cols:
                        if col in ohlcv.columns:
                            # Replace outliers with interpolated values
                            mask = (ohlcv[col] < min_reasonable_price) | (ohlcv[col] > max_reasonable_price)
                            if mask.any():
                                logger.info(f"   Fixing {mask.sum()} corrupted {col} values")
                                ohlcv.loc[mask, col] = np.nan
                                # Use linear interpolation, then forward fill, then backward fill
                                ohlcv[col] = ohlcv[col].interpolate(method='linear')
                                ohlcv[col] = ohlcv[col].fillna(method='ffill')
                                ohlcv[col] = ohlcv[col].fillna(method='bfill')
                    
                    logger.info(f"✅ Data corruption fixed for {symbol}")
                
                # Validate OHLC relationships
                invalid_ohlc = (ohlcv['low'] > ohlcv['high']) | (ohlcv['open'] < ohlcv['low']) | (ohlcv['close'] > ohlcv['high'])
                if invalid_ohlc.any():
                    logger.warning(f"🚨 OHLC relationship violations detected: {invalid_ohlc.sum()} candles")
                    # Fix: Ensure low <= open,close <= high
                    ohlcv['low'] = ohlcv[['low', 'open', 'close']].min(axis=1)
                    ohlcv['high'] = ohlcv[['high', 'open', 'close']].max(axis=1)
        
        # Prepare all timeframe data for switching
        all_candles = {}
        all_volumes = {}
        
        # Helper function to resample OHLCV with validation
        def resample_ohlcv(df, rule):
            if df.empty:
                return df
            try:
                # Ensure we have enough data points for meaningful resampling
                if len(df) < 2:
                    return df
                    
                resampled = df.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Validate resampled data
                if not resampled.empty:
                    # Fix any remaining OHLC violations after resampling
                    resampled['low'] = resampled[['low', 'open', 'close']].min(axis=1)
                    resampled['high'] = resampled[['high', 'open', 'close']].max(axis=1)
                
                return resampled
            except Exception as e:
                logger.error(f"Resampling failed for rule {rule}: {e}")
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
        # Only include timeframes that are >= base_timeframe
        all_timeframes = {
            '1s': (None, 1),      # Raw data if 1s, 1 second
            '1m': ('1min', 60),   # 60 seconds
            '5m': ('5min', 300),  # 300 seconds
            '15m': ('15min', 900), # 900 seconds
            '1h': ('1h', 3600)    # 3600 seconds
        }
        
        # Determine which timeframes are available based on base data
        base_seconds = {'1s': 1, '1m': 60, '5m': 300}.get(base_timeframe, 300)
        available_timeframes = {}
        
        for tf_label, (tf_rule, tf_seconds) in all_timeframes.items():
            # Can only aggregate UP from base timeframe
            if tf_seconds >= base_seconds:
                available_timeframes[tf_label] = tf_rule
        
        timeframes = available_timeframes
        
        for tf_label, tf_rule in timeframes.items():
            if tf_rule is None and base_timeframe == '1s':
                # Use raw 1s data
                tf_data = ohlcv
            elif tf_rule:
                # Resample to this timeframe
                tf_data = resample_ohlcv(ohlcv, tf_rule)
            else:
                # For base timeframe without resampling
                tf_data = ohlcv
            
            # Convert to chart format
            candles = []
            volumes = []
            
            if not tf_data.empty:
                # Limit points for performance
                step = max(1, len(tf_data) // 3000)
                sampled = tf_data.iloc[::step] if step > 1 else tf_data
                
                for idx, row in sampled.iterrows():
                    ts = int(idx.timestamp())
                    
                    # Additional validation at conversion time
                    open_val = float(row['open'])
                    high_val = float(row['high']) 
                    low_val = float(row['low'])
                    close_val = float(row['close'])
                    
                    # Final sanity check - ensure OHLC relationships are valid
                    if low_val <= high_val and min(open_val, close_val) >= low_val and max(open_val, close_val) <= high_val:
                        candles.append({
                            'time': ts,
                            'open': open_val,
                            'high': high_val,
                            'low': low_val,
                            'close': close_val
                        })
                        volumes.append({
                            'time': ts,
                            'value': float(row.get('volume', 0)),
                            'color': '#26a69a' if close_val >= open_val else '#ef5350'
                        })
                    else:
                        logger.warning(f"⚠️  Skipping invalid OHLC at {idx}: O={open_val:.2f}, H={high_val:.2f}, L={low_val:.2f}, C={close_val:.2f}")
            
            all_candles[tf_label] = candles
            all_volumes[tf_label] = volumes
        
        # Default to 5m for better visibility, then 1m, then base
        if all_candles.get('5m'):
            default_timeframe = '5m'
        elif all_candles.get('1m'):
            default_timeframe = '1m'
        else:
            default_timeframe = base_timeframe
        
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
        
        # First check for squeeze_scores from the strategy (new format with timestamps and scores)
        if isinstance(results, dict) and 'squeeze_scores' in results:
            squeeze_scores = results['squeeze_scores']
            if isinstance(squeeze_scores, dict) and 'timestamps' in squeeze_scores and 'scores' in squeeze_scores:
                timestamps = squeeze_scores['timestamps']
                scores = squeeze_scores['scores']
                if timestamps and scores and len(timestamps) == len(scores):
                    # Convert to our expected format
                    step = max(1, len(timestamps) // 2000)  # Downsample if > 2000 points
                    for i in range(0, len(timestamps), step):
                        ts = timestamps[i]
                        if hasattr(ts, 'timestamp'):
                            unix_time = int(ts.timestamp())
                        elif isinstance(ts, (int, float)):
                            unix_time = int(ts)
                        else:
                            unix_time = int(pd.Timestamp(ts).timestamp())
                        signal_data.append({
                            'time': unix_time,
                            'value': float(scores[i])
                        })
                    self.logger.info(f"📊 Loaded {len(signal_data)} squeeze scores from strategy")
        
        # Fallback to old format if new format not available
        if not signal_data:
            # Check multiple possible fields for strategy scores
            score_fields = ['strategy_scores', 'signals', 'squeeze_score']
            
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
                {''.join([f'<button class="timeframe-btn {"active" if default_timeframe == tf else ""}" data-tf="{tf}">{tf}</button>' for tf in timeframes.keys()])}
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