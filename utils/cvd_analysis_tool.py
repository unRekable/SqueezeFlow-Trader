#!/usr/bin/env python3
"""
CVD Analysis Tool - Flexible Market Analysis
Analyze any trading pair with customizable time ranges
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import json

# Import exchange mapper from data processors
from data.processors.exchange_mapper import ExchangeMapper
exchange_mapper = ExchangeMapper()

def query_influx(query):
    """Execute InfluxDB query and return results"""
    try:
        cmd = [
            'docker', 'exec', 'aggr-influx', 'influx', 
            '-execute', query, 
            '-database', 'significant_trades',
            '-format', 'json'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return data
        else:
            print(f"❌ Query failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Query error: {e}")
        return None

def process_influx_data(data, value_column):
    """Convert InfluxDB JSON to pandas DataFrame"""
    if not data or 'results' not in data or not data['results']:
        return pd.DataFrame()
    
    result = data['results'][0]
    if 'series' not in result or not result['series']:
        return pd.DataFrame()
    
    series = result['series'][0]
    columns = series['columns']
    values = series['values']
    
    df = pd.DataFrame(values, columns=columns) 
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    return df

def get_available_markets_for_symbol(symbol):
    """Get available markets for a trading symbol from database"""
    print(f"🔍 Getting available {symbol} markets from database...")
    
    # Query to find all markets with this symbol
    series_query = 'SHOW SERIES FROM "aggr_1m".trades_1m'
    series_result = query_influx(series_query)
    
    available_markets = []
    if series_result and 'results' in series_result:
        for result in series_result['results']:
            if 'series' in result and result['series']:
                for series in result['series']:
                    if 'values' in series:
                        for value in series['values']:
                            series_key = value[0]
                            if 'market=' in series_key and symbol.lower() in series_key.lower():
                                market = series_key.split('market=')[1]
                                available_markets.append(market)
    
    print(f"📊 Found {len(available_markets)} {symbol} markets in database")
    return available_markets

def classify_markets(markets, symbol):
    """Classify markets into SPOT and PERP using exchange mapper"""
    print(f"🔍 Classifying {symbol} markets...")
    
    spot_markets = []
    perp_markets = []
    
    for market in markets:
        market_type = exchange_mapper.get_market_type(market)
        if market_type == 'SPOT':
            spot_markets.append(market)
        elif market_type == 'PERP':
            perp_markets.append(market)
    
    print(f"✅ Classified: {len(spot_markets)} SPOT, {len(perp_markets)} PERP markets")
    print(f"📈 SPOT markets (first 5): {spot_markets[:5]}")
    print(f"📉 PERP markets (first 5): {perp_markets[:5]}")
    
    return spot_markets, perp_markets

def parse_time_range(time_range):
    """Parse time range string to start and end times"""
    now = datetime.utcnow()
    
    if time_range == 'last_hour':
        start_time = now - timedelta(hours=1)
        end_time = now
    elif time_range == 'last_6hours':
        start_time = now - timedelta(hours=6)
        end_time = now
    elif time_range == 'last_12hours':
        start_time = now - timedelta(hours=12)
        end_time = now
    elif time_range == 'last_24hours':
        start_time = now - timedelta(hours=24)
        end_time = now
    elif time_range == 'last_week':
        start_time = now - timedelta(days=7)
        end_time = now
    elif time_range == 'yesterday':
        yesterday = now - timedelta(days=1)
        start_time = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
    elif time_range == 'today':
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = now
    elif ',' in time_range:
        # Custom range: "2025-07-23,2025-07-24"
        start_str, end_str = time_range.split(',')
        start_time = datetime.strptime(start_str.strip(), '%Y-%m-%d')
        end_time = datetime.strptime(end_str.strip(), '%Y-%m-%d')
        # Set end time to end of day
        end_time = end_time.replace(hour=23, minute=59, second=59)
    else:
        raise ValueError(f"Unknown time range: {time_range}")
    
    return start_time, end_time

def get_price_reference_market(symbol):
    """Get a reliable reference market for price data"""
    # Prioritize major exchanges for price reference
    if symbol.upper() == 'BTCUSDT':
        return 'BINANCE:btcusdt'
    elif symbol.upper() == 'ETHUSDT':
        return 'BINANCE:ethusdt'
    else:
        # For other symbols, try to find a Binance market first
        markets = get_available_markets_for_symbol(symbol)
        for market in markets:
            if market.startswith('BINANCE:'):
                return market
        # If no Binance market, use first available
        return markets[0] if markets else None

def analyze_cvd(symbol, time_range, output_file=None):
    """Main CVD analysis function"""
    
    print(f"🎯 CVD ANALYSIS TOOL - {symbol.upper()}")
    print("=" * 60)
    
    # Parse time range
    start_time, end_time = parse_time_range(time_range)
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    print(f"📅 Analysis Period: {start_str} to {end_str}")
    print(f"⏱️  Duration: {(end_time - start_time).total_seconds() / 3600:.1f} hours")
    
    # Get available markets
    available_markets = get_available_markets_for_symbol(symbol)
    if not available_markets:
        print(f"❌ No markets found for {symbol}")
        return False
    
    # Classify markets
    spot_markets, perp_markets = classify_markets(available_markets, symbol)
    
    if not spot_markets or not perp_markets:
        print(f"❌ Error: Missing SPOT ({len(spot_markets)}) or PERP ({len(perp_markets)}) markets")
        return False
    
    # Get price reference market
    price_market = get_price_reference_market(symbol)
    if not price_market:
        print(f"❌ No price reference market found for {symbol}")
        return False
    
    print(f"💰 Using price reference: {price_market}")
    
    # Build market filters
    spot_filter = ' OR '.join([f"market = '{market}'" for market in spot_markets])
    perp_filter = ' OR '.join([f"market = '{market}'" for market in perp_markets])
    
    print(f"\n🎯 FETCHING DATA for {symbol.upper()}: {start_str} to {end_str}")
    print("-" * 60)
    
    # Get price data
    print(f"📊 Fetching {symbol} price data...")
    price_query = f'''SELECT time, close FROM "aggr_1m".trades_1m WHERE market = '{price_market}' AND time >= '{start_str}' AND time <= '{end_str}' ORDER BY time'''
    price_data = query_influx(price_query)
    
    # Get SPOT CVD per minute
    print(f"📈 Fetching SPOT CVD data...")
    spot_cvd_query = f'''SELECT time, sum(vbuy) - sum(vsell) AS spot_cvd FROM "aggr_1m".trades_1m WHERE ({spot_filter}) AND time >= '{start_str}' AND time <= '{end_str}' GROUP BY time(1m) ORDER BY time'''
    spot_data = query_influx(spot_cvd_query)
    
    # Get FUTURES CVD per minute
    print(f"📉 Fetching FUTURES CVD data...")
    futures_cvd_query = f'''SELECT time, sum(vbuy) - sum(vsell) AS futures_cvd FROM "aggr_1m".trades_1m WHERE ({perp_filter}) AND time >= '{start_str}' AND time <= '{end_str}' GROUP BY time(1m) ORDER BY time'''
    futures_data = query_influx(futures_cvd_query)
    
    # Process data
    price_df = process_influx_data(price_data, 'close')
    spot_df = process_influx_data(spot_data, 'spot_cvd') 
    futures_df = process_influx_data(futures_data, 'futures_cvd')
    
    print(f"📊 Raw data loaded:")
    print(f"   Price data points: {len(price_df)}")
    print(f"   SPOT CVD data points: {len(spot_df)}")
    print(f"   FUTURES CVD data points: {len(futures_df)}")
    
    if len(price_df) == 0 or len(spot_df) == 0 or len(futures_df) == 0:
        print("❌ No data found - check database and time range")
        return False
    
    # Align all data to same timeframe
    min_len = min(len(price_df), len(spot_df), len(futures_df))
    price_df = price_df.head(min_len).copy()
    spot_df = spot_df.head(min_len).copy()
    futures_df = futures_df.head(min_len).copy()
    
    print(f"📊 Aligned data to {min_len} points")
    
    # Calculate CUMULATIVE Volume Delta (Industry Standard)
    print("\n📊 Calculating CUMULATIVE Volume Delta (Industry Standard)...")
    spot_df['spot_cvd_cumulative'] = spot_df['spot_cvd'].cumsum()
    futures_df['futures_cvd_cumulative'] = futures_df['futures_cvd'].cumsum()
    
    # Calculate divergence
    cvd_divergence = spot_df['spot_cvd_cumulative'] - futures_df['futures_cvd_cumulative']
    
    # Print verification data
    print(f"\n🎯 DATA VERIFICATION:")
    print(f"   💰 {symbol} Price: ${price_df['close'].iloc[0]:.2f} → ${price_df['close'].iloc[-1]:.2f} (${price_df['close'].iloc[-1] - price_df['close'].iloc[0]:.2f} change)")
    print(f"   🔴 SPOT CVD: {spot_df['spot_cvd_cumulative'].iloc[0]/1e6:.1f}M → {spot_df['spot_cvd_cumulative'].iloc[-1]/1e6:.1f}M ({(spot_df['spot_cvd_cumulative'].iloc[-1] - spot_df['spot_cvd_cumulative'].iloc[0])/1e6:.1f}M change)")
    print(f"   🟠 FUTURES CVD: {futures_df['futures_cvd_cumulative'].iloc[0]/1e6:.1f}M → {futures_df['futures_cvd_cumulative'].iloc[-1]/1e6:.1f}M ({(futures_df['futures_cvd_cumulative'].iloc[-1] - futures_df['futures_cvd_cumulative'].iloc[0])/1e6:.1f}M change)")
    print(f"   ⚡ CVD Divergence: {cvd_divergence.iloc[-1]/1e6:.1f}M")
    print(f"   ⏰ Time: {price_df['time'].iloc[0]} → {price_df['time'].iloc[-1]}")
    
    # Create visualization
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'{symbol.upper()} CVD Analysis - {time_range.replace("_", " ").title()}\n'
                 f'{start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")} UTC', 
                 fontsize=16, fontweight='bold')
    
    # Panel 1: Price
    ax1.plot(price_df['time'], price_df['close'], 'blue', linewidth=3, label=f'{symbol.upper()} Price (USD)', alpha=0.9)
    ax1.set_ylabel('Price (USD)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{symbol.upper()} Price: ${price_df["close"].iloc[0]:.2f} → ${price_df["close"].iloc[-1]:.2f} (${price_df["close"].iloc[-1] - price_df["close"].iloc[0]:.2f} change)')
    
    # Panel 2: SPOT CVD
    ax2.plot(spot_df['time'], spot_df['spot_cvd_cumulative'], 'red', linewidth=2, label=f'SPOT CVD Cumulative ({len(spot_markets)} exchanges)', alpha=0.9)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero line')
    ax2.set_ylabel('SPOT CVD (Cumulative)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'SPOT CVD: {spot_df["spot_cvd_cumulative"].iloc[0]/1e6:.1f}M → {spot_df["spot_cvd_cumulative"].iloc[-1]/1e6:.1f}M ({(spot_df["spot_cvd_cumulative"].iloc[-1] - spot_df["spot_cvd_cumulative"].iloc[0])/1e6:.1f}M change)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # Panel 3: FUTURES CVD
    ax3.plot(futures_df['time'], futures_df['futures_cvd_cumulative'], 'orange', linewidth=2, label=f'FUTURES CVD Cumulative ({len(perp_markets)} exchanges)', alpha=0.9)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero line')
    ax3.set_ylabel('FUTURES CVD (Cumulative)', color='orange', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3.legend(loc='upper left', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'FUTURES CVD: {futures_df["futures_cvd_cumulative"].iloc[0]/1e6:.1f}M → {futures_df["futures_cvd_cumulative"].iloc[-1]/1e6:.1f}M ({(futures_df["futures_cvd_cumulative"].iloc[-1] - futures_df["futures_cvd_cumulative"].iloc[0])/1e6:.1f}M change)')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # Panel 4: CVD Divergence
    ax4.plot(spot_df['time'], cvd_divergence, 'purple', linewidth=2, label=f'CVD Divergence (SPOT - FUTURES)', alpha=0.9)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero line')
    ax4.set_ylabel('CVD Divergence', color='purple', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4.legend(loc='upper left', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f'CVD Divergence: {cvd_divergence.iloc[-1]/1e6:.1f}M (Positive = SPOT > FUTURES)')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    ax4.set_xlabel('Time', fontsize=12)
    
    # Format time axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    if not output_file:
        output_file = f'cvd_analysis_{symbol.lower()}_{time_range}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ CVD Analysis plot saved to: {output_file}")
    
    # Final summary
    print(f"\n🎯 ANALYSIS SUMMARY:")
    print(f"=" * 60)
    print(f"📊 Symbol: {symbol.upper()}")
    print(f"📅 Period: {time_range}")
    print(f"📈 SPOT Exchanges: {len(spot_markets)}")
    print(f"📉 PERP Exchanges: {len(perp_markets)}")
    print(f"📊 Data Points: {min_len} minutes")
    print(f"💰 Price Change: ${price_df['close'].iloc[-1] - price_df['close'].iloc[0]:.2f}")
    print(f"🔴 SPOT CVD Change: {(spot_df['spot_cvd_cumulative'].iloc[-1] - spot_df['spot_cvd_cumulative'].iloc[0])/1e6:.1f}M")
    print(f"🟠 FUTURES CVD Change: {(futures_df['futures_cvd_cumulative'].iloc[-1] - futures_df['futures_cvd_cumulative'].iloc[0])/1e6:.1f}M")
    print(f"⚡ Final CVD Divergence: {cvd_divergence.iloc[-1]/1e6:.1f}M")
    
    plt.show()
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='CVD Analysis Tool - Analyze any trading pair with customizable time ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Time Range Options:
  last_hour, last_6hours, last_12hours, last_24hours, last_week
  yesterday, today
  custom: "2025-07-23,2025-07-24" (YYYY-MM-DD,YYYY-MM-DD)

Examples:
  %(prog)s BTCUSDT last_24hours
  %(prog)s ETHUSDT yesterday
  %(prog)s BTCUSDT "2025-07-23,2025-07-24"
  %(prog)s ETHUSDT last_week --output eth_weekly_analysis.png
        """
    )
    
    parser.add_argument(
        'symbol',
        help='Trading symbol (e.g., BTCUSDT, ETHUSDT)'
    )
    
    parser.add_argument(
        'time_range',
        help='Time range for analysis'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file name (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    try:
        success = analyze_cvd(args.symbol, args.time_range, args.output)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()