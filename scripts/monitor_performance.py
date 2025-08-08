#!/usr/bin/env python3
"""
1-Second Data Collection Performance Monitor

This script provides detailed monitoring of the 1-second data collection system,
including performance metrics, data quality, and system health.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import time
import argparse
import json
from typing import Dict, Any


class OneSecondMonitor:
    """Monitor for 1-second data collection performance."""
    
    def __init__(self, host='localhost', port=8086, database='significant_trades'):
        """Initialize the monitor."""
        self.influx = InfluxDBClient(host=host, port=port, database=database)
        self.metrics = {}
    
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check how fresh the latest data is."""
        metrics = {
            '1s_data_available': False,
            'latest_timestamp': None,
            'age_seconds': None,
            'is_stale': False
        }
        
        try:
            # Check for 1-second data
            query = "SELECT last(close) as last_close, last(time) as last_time FROM trades_1s WHERE time > now() - 1m"
            result = self.influx.query(query)
            
            if result:
                points = list(result.get_points())
                if points:
                    metrics['1s_data_available'] = True
                    last_time = points[0].get('last_time')
                    if last_time:
                        last_dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                        metrics['latest_timestamp'] = last_time
                        age = (datetime.now(last_dt.tzinfo) - last_dt).total_seconds()
                        metrics['age_seconds'] = round(age, 1)
                        metrics['is_stale'] = age > 10  # Data older than 10 seconds is stale
            
            # If no 1s data, check for 10s data (fallback)
            if not metrics['1s_data_available']:
                query_10s = "SELECT last(close), last(time) as last_time FROM trades_10s WHERE time > now() - 1m"
                result_10s = self.influx.query(query_10s)
                if result_10s:
                    points_10s = list(result_10s.get_points())
                    if points_10s:
                        metrics['fallback_10s_available'] = True
                        metrics['note'] = "Still using 10s configuration"
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def check_ingestion_rate(self) -> Dict[str, Any]:
        """Check the data ingestion rate."""
        metrics = {
            'bars_per_minute': 0,
            'markets_count': 0,
            'bars_per_market_per_minute': 0,
            'expected_bars_per_market': 60,
            'performance_ratio': 0
        }
        
        try:
            # Count bars in last minute
            query_count = "SELECT COUNT(close) as count FROM trades_1s WHERE time > now() - 1m"
            result = self.influx.query(query_count)
            
            if result:
                points = list(result.get_points())
                if points:
                    metrics['bars_per_minute'] = points[0].get('count', 0)
            
            # Count distinct markets
            query_markets = "SELECT COUNT(DISTINCT(market)) as markets FROM trades_1s WHERE time > now() - 1m"
            result_markets = self.influx.query(query_markets)
            
            if result_markets:
                points = list(result_markets.get_points())
                if points:
                    metrics['markets_count'] = points[0].get('markets', 0)
            
            # Calculate per-market rate
            if metrics['markets_count'] > 0:
                metrics['bars_per_market_per_minute'] = round(
                    metrics['bars_per_minute'] / metrics['markets_count'], 1
                )
                metrics['performance_ratio'] = round(
                    metrics['bars_per_market_per_minute'] / metrics['expected_bars_per_market'], 2
                )
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def check_data_distribution(self, minutes=5) -> Dict[str, Any]:
        """Check data distribution over time."""
        metrics = {
            'time_buckets': [],
            'total_bars': 0,
            'gaps_detected': 0
        }
        
        try:
            # Get data distribution by minute
            query = f"""
            SELECT COUNT(close) as bars 
            FROM trades_1s 
            WHERE time > now() - {minutes}m 
            GROUP BY time(1m)
            """
            result = self.influx.query(query)
            
            if result:
                for series in result:
                    for point in series:
                        bucket = {
                            'time': point['time'],
                            'bars': point.get('bars', 0)
                        }
                        metrics['time_buckets'].append(bucket)
                        metrics['total_bars'] += bucket['bars']
                        
                        # Detect gaps (minutes with < 30 bars might indicate issues)
                        if bucket['bars'] < 30:
                            metrics['gaps_detected'] += 1
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def check_storage_usage(self) -> Dict[str, Any]:
        """Estimate storage usage and growth rate."""
        metrics = {
            'bars_last_24h': 0,
            'estimated_daily_mb': 0,
            'estimated_monthly_gb': 0,
            'bars_last_hour': 0,
            'hourly_growth_mb': 0
        }
        
        try:
            # Count bars in last 24 hours
            query_24h = "SELECT COUNT(close) as count FROM trades_1s WHERE time > now() - 24h"
            result_24h = self.influx.query(query_24h)
            
            if result_24h:
                points = list(result_24h.get_points())
                if points:
                    metrics['bars_last_24h'] = points[0].get('count', 0)
                    # Estimate ~100 bytes per bar
                    metrics['estimated_daily_mb'] = round(metrics['bars_last_24h'] * 100 / 1048576, 2)
                    metrics['estimated_monthly_gb'] = round(metrics['estimated_daily_mb'] * 30 / 1024, 2)
            
            # Count bars in last hour for more recent rate
            query_1h = "SELECT COUNT(close) as count FROM trades_1s WHERE time > now() - 1h"
            result_1h = self.influx.query(query_1h)
            
            if result_1h:
                points = list(result_1h.get_points())
                if points:
                    metrics['bars_last_hour'] = points[0].get('count', 0)
                    metrics['hourly_growth_mb'] = round(metrics['bars_last_hour'] * 100 / 1048576, 2)
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def check_market_coverage(self) -> Dict[str, Any]:
        """Check which markets are being tracked."""
        metrics = {
            'active_markets': [],
            'total_markets': 0,
            'top_volume_markets': []
        }
        
        try:
            # Get list of active markets in last 5 minutes
            query = """
            SELECT COUNT(close) as bars, SUM(volume) as total_volume 
            FROM trades_1s 
            WHERE time > now() - 5m 
            GROUP BY market
            """
            result = self.influx.query(query)
            
            if result:
                markets = []
                for series in result:
                    market_name = series[0][1].get('market', 'unknown')
                    for point in series:
                        markets.append({
                            'market': market_name,
                            'bars': point.get('bars', 0),
                            'volume': point.get('total_volume', 0)
                        })
                
                # Sort by volume
                markets.sort(key=lambda x: x['volume'], reverse=True)
                
                metrics['active_markets'] = [m['market'] for m in markets]
                metrics['total_markets'] = len(markets)
                metrics['top_volume_markets'] = markets[:5]  # Top 5 by volume
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate a comprehensive monitoring report."""
        report = []
        report.append("=" * 60)
        report.append("1-SECOND DATA COLLECTION PERFORMANCE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Data Freshness
        freshness = self.check_data_freshness()
        report.append("\n📊 DATA FRESHNESS")
        report.append("-" * 40)
        
        if freshness.get('1s_data_available'):
            report.append(f"✅ 1-second data is available")
            report.append(f"   Latest data: {freshness.get('age_seconds', 'N/A')} seconds ago")
            if freshness.get('is_stale'):
                report.append(f"   ⚠️  WARNING: Data is stale (>10 seconds)")
        else:
            report.append(f"❌ No 1-second data found")
            if freshness.get('fallback_10s_available'):
                report.append(f"   ℹ️  10-second data is available (old config)")
        
        # Ingestion Rate
        ingestion = self.check_ingestion_rate()
        report.append("\n📈 INGESTION RATE")
        report.append("-" * 40)
        report.append(f"Bars per minute: {ingestion['bars_per_minute']}")
        report.append(f"Active markets: {ingestion['markets_count']}")
        
        if ingestion['markets_count'] > 0:
            report.append(f"Bars per market: {ingestion['bars_per_market_per_minute']}/min")
            report.append(f"Performance: {ingestion['performance_ratio']*100:.1f}% of expected")
            
            if ingestion['performance_ratio'] < 0.8:
                report.append("   ⚠️  Low ingestion rate detected")
        
        # Data Distribution
        distribution = self.check_data_distribution()
        report.append("\n📊 DATA DISTRIBUTION (5 min)")
        report.append("-" * 40)
        report.append(f"Total bars: {distribution['total_bars']}")
        report.append(f"Time gaps detected: {distribution['gaps_detected']}")
        
        # Storage Usage
        storage = self.check_storage_usage()
        report.append("\n💾 STORAGE USAGE")
        report.append("-" * 40)
        report.append(f"Last 24h: {storage['bars_last_24h']:,} bars")
        report.append(f"Daily growth: ~{storage['estimated_daily_mb']} MB")
        report.append(f"Monthly estimate: ~{storage['estimated_monthly_gb']} GB")
        
        # Market Coverage
        markets = self.check_market_coverage()
        report.append("\n🌍 MARKET COVERAGE")
        report.append("-" * 40)
        report.append(f"Active markets: {markets['total_markets']}")
        
        if markets['top_volume_markets']:
            report.append("Top markets by volume:")
            for i, market in enumerate(markets['top_volume_markets'], 1):
                report.append(f"  {i}. {market['market']}: {market['bars']} bars")
        
        # Performance Summary
        report.append("\n⚡ PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        if freshness.get('1s_data_available'):
            report.append("✅ 1-second data collection is ACTIVE")
            report.append("   Expected signal latency: 5-10 seconds")
            report.append("   Previous (10s): 60-70 seconds")
            report.append("   Improvement: ~6-12x faster")
        else:
            report.append("❌ 1-second data collection is NOT ACTIVE")
            report.append("   Action: Restart aggr-server to activate")
            report.append("   Command: docker-compose restart aggr-server")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def continuous_monitor(self, interval=10):
        """Run continuous monitoring with updates every N seconds."""
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        print(f"Updates every {interval} seconds\n")
        
        try:
            while True:
                # Clear screen (works on Unix-like systems)
                os.system('clear' if os.name != 'nt' else 'cls')
                
                # Generate and print report
                report = self.generate_report()
                print(report)
                
                # Wait for next update
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def export_metrics(self, filepath='1s_metrics.json'):
        """Export current metrics to JSON file."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'freshness': self.check_data_freshness(),
            'ingestion': self.check_ingestion_rate(),
            'distribution': self.check_data_distribution(),
            'storage': self.check_storage_usage(),
            'markets': self.check_market_coverage()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Metrics exported to {filepath}")
        return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monitor 1-second data collection performance'
    )
    
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuous monitoring'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Update interval for continuous monitoring (seconds)'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export metrics to JSON file'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='InfluxDB host'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8086,
        help='InfluxDB port'
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = OneSecondMonitor(host=args.host, port=args.port)
    
    if args.continuous:
        # Run continuous monitoring
        monitor.continuous_monitor(interval=args.interval)
    else:
        # Generate single report
        report = monitor.generate_report()
        print(report)
        
        # Export if requested
        if args.export:
            monitor.export_metrics(args.export)


if __name__ == "__main__":
    main()