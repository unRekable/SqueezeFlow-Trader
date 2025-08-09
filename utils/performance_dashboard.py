#!/usr/bin/env python3
"""
Performance Dashboard Generator for SqueezeFlow Trader

Creates HTML dashboards and reports for comprehensive performance monitoring
including real-time metrics, bottleneck analysis, and optimization recommendations.

Features:
- Interactive HTML performance dashboard
- Real-time metrics visualization
- Bottleneck identification with severity levels
- Performance comparison charts (1s vs regular mode)
- Memory and CPU usage trending
- Operation timing histograms
- Export capabilities (JSON, CSV)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # Set style for better looking charts
    plt.style.use('default')
    sns.set_palette("husl")
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from utils.performance_monitor import PerformanceMonitorIntegration


class PerformanceDashboard:
    """
    Performance Dashboard Generator
    
    Creates comprehensive HTML dashboards with interactive charts
    and performance analysis
    """
    
    def __init__(self, performance_monitor: PerformanceMonitorIntegration):
        self.monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
        
    def generate_dashboard(self, output_path: str = None, time_window_minutes: int = 60) -> str:
        """
        Generate comprehensive HTML performance dashboard
        
        Args:
            output_path: Output path for HTML file (auto-generated if None)
            time_window_minutes: Time window for metrics analysis
            
        Returns:
            Path to generated HTML dashboard
        """
        if output_path is None:
            dashboard_dir = Path("data/performance_dashboards")
            dashboard_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(dashboard_dir / f"performance_dashboard_{int(time.time())}.html")
        
        # Collect all performance data
        dashboard_data = self._collect_dashboard_data(time_window_minutes)
        
        # Generate HTML dashboard
        html_content = self._generate_html_dashboard(dashboard_data)
        
        # Save dashboard
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"Performance dashboard generated: {output_path}")
        return output_path
        
    def _collect_dashboard_data(self, time_window_minutes: int) -> Dict[str, Any]:
        """Collect all data needed for dashboard"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'time_window_minutes': time_window_minutes,
            'summary': self.monitor.get_performance_summary(),
            'system_stats': self.monitor.analyzer.get_system_stats(time_window_minutes),
            'window_stats': self.monitor.analyzer.get_window_processing_stats(),
            'cache_stats': self.monitor.analyzer.get_cache_stats(),
            'bottlenecks': self.monitor.analyzer.identify_bottlenecks(),
            'recommendations': self.monitor.analyzer._generate_recommendations(),
            'operation_stats': self._get_operation_statistics(),
            'charts': self._generate_chart_data(),
            '1s_comparison': self.monitor.get_1s_vs_regular_comparison()
        }
        
    def _get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tracked operations"""
        stats = {}
        
        with self.monitor.metrics._lock:
            for op_name in self.monitor.metrics.operation_timings.keys():
                stats[op_name] = self.monitor.analyzer.get_operation_stats(op_name, 60)
                
        return stats
        
    def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate chart data for visualization"""
        charts = {}
        
        # Memory usage chart
        with self.monitor.metrics._lock:
            if self.monitor.metrics.memory_usage_history:
                charts['memory_usage'] = {
                    'timestamps': [ts.isoformat() for ts in self.monitor.metrics.memory_timestamps],
                    'memory_mb': self.monitor.metrics.memory_usage_history,
                    'title': 'Memory Usage Over Time'
                }
                
        # CPU usage chart
        with self.monitor.metrics._lock:
            if self.monitor.metrics.cpu_usage_history:
                charts['cpu_usage'] = {
                    'timestamps': [ts.isoformat() for ts in self.monitor.metrics.memory_timestamps[:len(self.monitor.metrics.cpu_usage_history)]],
                    'cpu_percent': self.monitor.metrics.cpu_usage_history,
                    'title': 'CPU Usage Over Time'
                }
                
        # Window processing times
        with self.monitor.metrics._lock:
            if self.monitor.metrics.window_processing_times:
                charts['window_processing'] = {
                    'timestamps': [ts.isoformat() for ts in self.monitor.metrics.window_timestamps],
                    'processing_times_ms': self.monitor.metrics.window_processing_times,
                    'title': 'Window Processing Times'
                }
                
        # Operation timing histograms
        charts['operation_histograms'] = {}
        with self.monitor.metrics._lock:
            for op_name, timings in list(self.monitor.metrics.operation_timings.items())[:5]:  # Top 5 operations
                if timings:
                    charts['operation_histograms'][op_name] = {
                        'durations': [t.duration_ms for t in timings],
                        'title': f'{op_name} Duration Distribution'
                    }
                    
        return charts
        
    def _generate_html_dashboard(self, data: Dict[str, Any]) -> str:
        """Generate HTML dashboard content"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SqueezeFlow Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
        }}
        .card h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: 600;
            color: #555;
        }}
        .metric-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
        }}
        .bottleneck {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .bottleneck.critical {{
            background: #fff5f5;
            border-left-color: #e53e3e;
            color: #c53030;
        }}
        .bottleneck.high {{
            background: #fefcbf;
            border-left-color: #d69e2e;
            color: #b7791f;
        }}
        .bottleneck.medium {{
            background: #e6fffa;
            border-left-color: #38b2ac;
            color: #285e61;
        }}
        .recommendation {{
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            color: #22543d;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-good {{ background-color: #48bb78; }}
        .status-warning {{ background-color: #ed8936; }}
        .status-critical {{ background-color: #e53e3e; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        .comparison-table th {{
            background-color: #f7fafc;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚀 SqueezeFlow Performance Dashboard</h1>
            <p>Generated: {timestamp} | Window: {time_window} minutes</p>
        </div>
        
        <!-- Summary Cards -->
        <div class="grid">
            <!-- System Performance -->
            <div class="card">
                <h2>🖥️ System Performance</h2>
                {system_metrics}
            </div>
            
            <!-- Window Processing -->
            <div class="card">
                <h2>🔄 Window Processing</h2>
                {window_metrics}
            </div>
            
            <!-- Cache Performance -->
            <div class="card">
                <h2>⚡ Cache Performance</h2>
                {cache_metrics}
            </div>
            
            <!-- Operation Summary -->
            <div class="card">
                <h2>⏱️ Operation Summary</h2>
                {operation_summary}
            </div>
        </div>
        
        <!-- Performance Charts -->
        <div class="chart-container full-width">
            <h2>📊 Performance Charts</h2>
            {performance_charts}
        </div>
        
        <!-- Bottlenecks -->
        <div class="card full-width">
            <h2>⚠️ Performance Bottlenecks</h2>
            {bottlenecks_section}
        </div>
        
        <!-- 1s vs Regular Mode Comparison -->
        {mode_comparison}
        
        <!-- Recommendations -->
        <div class="card full-width">
            <h2>💡 Optimization Recommendations</h2>
            {recommendations_section}
        </div>
        
        <!-- Operation Details -->
        <div class="card full-width">
            <h2>📈 Detailed Operation Statistics</h2>
            {operation_details}
        </div>
        
        <div class="footer">
            <p>SqueezeFlow Trader Performance Dashboard | Real-time monitoring for optimal trading performance</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 5 minutes (300000 ms)
        setTimeout(function(){{
            location.reload();
        }}, 300000);
        
        // Add timestamp to show when dashboard was loaded
        console.log('Dashboard loaded at:', new Date().toISOString());
    </script>
</body>
</html>
        """
        
        # Generate content sections
        system_metrics = self._generate_system_metrics_html(data.get('system_stats', {}))
        window_metrics = self._generate_window_metrics_html(data.get('window_stats', {}))
        cache_metrics = self._generate_cache_metrics_html(data.get('cache_stats', {}))
        operation_summary = self._generate_operation_summary_html(data.get('operation_stats', {}))
        performance_charts = self._generate_charts_html(data.get('charts', {}))
        bottlenecks_section = self._generate_bottlenecks_html(data.get('bottlenecks', []))
        recommendations_section = self._generate_recommendations_html(data.get('recommendations', []))
        operation_details = self._generate_operation_details_html(data.get('operation_stats', {}))
        mode_comparison = self._generate_mode_comparison_html(data.get('1s_comparison', {}))
        
        # Fill in the template
        return html_template.format(
            timestamp=data['timestamp'],
            time_window=data['time_window_minutes'],
            system_metrics=system_metrics,
            window_metrics=window_metrics,
            cache_metrics=cache_metrics,
            operation_summary=operation_summary,
            performance_charts=performance_charts,
            bottlenecks_section=bottlenecks_section,
            recommendations_section=recommendations_section,
            operation_details=operation_details,
            mode_comparison=mode_comparison
        )
        
    def _generate_system_metrics_html(self, stats: Dict[str, Any]) -> str:
        """Generate system metrics HTML"""
        if stats.get('error'):
            return '<p class="metric">No system metrics available</p>'
            
        cpu_stats = stats.get('cpu_usage_percent', {})
        memory_stats = stats.get('memory_usage_mb', {})
        
        # Determine status
        cpu_current = cpu_stats.get('current', 0)
        memory_current = memory_stats.get('current', 0)
        
        cpu_status = 'status-good' if cpu_current < 50 else 'status-warning' if cpu_current < 80 else 'status-critical'
        memory_status = 'status-good' if memory_current < 1000 else 'status-warning' if memory_current < 2000 else 'status-critical'
        
        return f"""
        <div class="metric">
            <span class="metric-label">
                <span class="status-indicator {cpu_status}"></span>CPU Usage
            </span>
            <span class="metric-value">{cpu_current:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">CPU Average</span>
            <span class="metric-value">{cpu_stats.get('mean', 0):.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">
                <span class="status-indicator {memory_status}"></span>Memory Usage
            </span>
            <span class="metric-value">{memory_current:.0f}MB</span>
        </div>
        <div class="metric">
            <span class="metric-label">Memory Average</span>
            <span class="metric-value">{memory_stats.get('mean', 0):.0f}MB</span>
        </div>
        """
        
    def _generate_window_metrics_html(self, stats: Dict[str, Any]) -> str:
        """Generate window processing metrics HTML"""
        if stats.get('error'):
            return '<p class="metric">No window processing data available</p>'
            
        processing_stats = stats.get('processing_time_stats_ms', {})
        
        return f"""
        <div class="metric">
            <span class="metric-label">Windows Processed</span>
            <span class="metric-value">{stats.get('total_windows_processed', 0):,}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Average Time</span>
            <span class="metric-value">{processing_stats.get('mean', 0):.1f}ms</span>
        </div>
        <div class="metric">
            <span class="metric-label">95th Percentile</span>
            <span class="metric-value">{processing_stats.get('p95', 0):.1f}ms</span>
        </div>
        <div class="metric">
            <span class="metric-label">Throughput</span>
            <span class="metric-value">{stats.get('throughput_windows_per_minute', 0):.1f}/min</span>
        </div>
        """
        
    def _generate_cache_metrics_html(self, stats: Dict[str, Any]) -> str:
        """Generate cache performance metrics HTML"""
        if stats.get('error'):
            return '<p class="metric">No cache data available</p>'
            
        hit_rate = stats.get('hit_rate_percent', 0)
        status = 'status-good' if hit_rate > 80 else 'status-warning' if hit_rate > 50 else 'status-critical'
        
        return f"""
        <div class="metric">
            <span class="metric-label">
                <span class="status-indicator {status}"></span>Hit Rate
            </span>
            <span class="metric-value">{hit_rate:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Cache Hits</span>
            <span class="metric-value">{stats.get('cache_hits', 0):,}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Cache Misses</span>
            <span class="metric-value">{stats.get('cache_misses', 0):,}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Requests</span>
            <span class="metric-value">{stats.get('total_requests', 0):,}</span>
        </div>
        """
        
    def _generate_operation_summary_html(self, stats: Dict[str, Any]) -> str:
        """Generate operation summary HTML"""
        if not stats:
            return '<p class="metric">No operation data available</p>'
            
        total_ops = len(stats)
        successful_ops = len([s for s in stats.values() if not s.get('error')])
        
        # Find slowest operation
        slowest_op = None
        slowest_time = 0
        for op_name, op_stats in stats.items():
            if not op_stats.get('error'):
                duration_stats = op_stats.get('duration_stats_ms', {})
                mean_time = duration_stats.get('mean', 0)
                if mean_time > slowest_time:
                    slowest_time = mean_time
                    slowest_op = op_name
                    
        return f"""
        <div class="metric">
            <span class="metric-label">Total Operations</span>
            <span class="metric-value">{total_ops}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Active Operations</span>
            <span class="metric-value">{successful_ops}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Slowest Operation</span>
            <span class="metric-value" style="font-size: 0.9em;">{slowest_op or 'None'}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Slowest Avg Time</span>
            <span class="metric-value">{slowest_time:.1f}ms</span>
        </div>
        """
        
    def _generate_charts_html(self, charts: Dict[str, Any]) -> str:
        """Generate performance charts HTML"""
        if not PLOTLY_AVAILABLE:
            return '<p>Charts require plotly library. Please install with: pip install plotly</p>'
            
        html_parts = []
        
        # Memory usage chart
        if 'memory_usage' in charts:
            memory_data = charts['memory_usage']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=memory_data['timestamps'],
                y=memory_data['memory_mb'],
                mode='lines',
                name='Memory Usage (MB)',
                line=dict(color='#667eea', width=2)
            ))
            fig.update_layout(
                title='Memory Usage Over Time',
                xaxis_title='Time',
                yaxis_title='Memory (MB)',
                height=300
            )
            html_parts.append('<div id="memory-chart"></div>')
            html_parts.append(f'<script>Plotly.newPlot("memory-chart", {fig.to_json()});</script>')
            
        # Window processing times
        if 'window_processing' in charts:
            window_data = charts['window_processing']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=window_data['timestamps'],
                y=window_data['processing_times_ms'],
                mode='lines',
                name='Processing Time (ms)',
                line=dict(color='#48bb78', width=2)
            ))
            fig.update_layout(
                title='Window Processing Times',
                xaxis_title='Time',
                yaxis_title='Processing Time (ms)',
                height=300
            )
            html_parts.append('<div id="window-chart"></div>')
            html_parts.append(f'<script>Plotly.newPlot("window-chart", {fig.to_json()});</script>')
            
        return '\n'.join(html_parts) if html_parts else '<p>No chart data available</p>'
        
    def _generate_bottlenecks_html(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """Generate bottlenecks HTML"""
        if not bottlenecks:
            return '<div class="recommendation">✅ No significant performance bottlenecks detected</div>'
            
        html_parts = []
        for bottleneck in bottlenecks[:5]:  # Top 5
            severity = bottleneck.get('severity', 'medium')
            bottleneck_type = bottleneck.get('type', 'unknown')
            operation_name = bottleneck.get('operation_name', 'System')
            
            if bottleneck_type == 'slow_operation':
                message = f"Slow operation detected: {operation_name} averaging {bottleneck.get('avg_duration_ms', 0):.1f}ms"
            elif bottleneck_type == 'high_error_rate':
                message = f"High error rate: {operation_name} at {bottleneck.get('error_rate_percent', 0):.1f}%"
            elif bottleneck_type == 'high_memory_usage':
                message = f"High memory usage detected: {bottleneck.get('max_memory_mb', 0):.0f}MB peak"
            elif bottleneck_type == 'poor_cache_performance':
                message = f"Poor cache performance: {bottleneck.get('hit_rate_percent', 0):.1f}% hit rate"
            else:
                message = f"Performance issue detected: {bottleneck_type}"
                
            html_parts.append(f'<div class="bottleneck {severity}">{message}</div>')
            
        return '\n'.join(html_parts)
        
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate recommendations HTML"""
        if not recommendations:
            return '<div class="recommendation">✅ Performance is optimal - no specific recommendations</div>'
            
        html_parts = []
        for rec in recommendations[:5]:  # Top 5
            html_parts.append(f'<div class="recommendation">💡 {rec}</div>')
            
        return '\n'.join(html_parts)
        
    def _generate_operation_details_html(self, stats: Dict[str, Any]) -> str:
        """Generate detailed operation statistics HTML"""
        if not stats:
            return '<p>No detailed operation data available</p>'
            
        html_parts = ['<div style="overflow-x: auto;">']
        html_parts.append('<table class="comparison-table">')
        html_parts.append('''
        <tr>
            <th>Operation</th>
            <th>Total Calls</th>
            <th>Success Rate</th>
            <th>Avg Duration</th>
            <th>P95 Duration</th>
            <th>Throughput</th>
        </tr>
        ''')
        
        # Sort operations by average duration (slowest first)
        sorted_ops = sorted(stats.items(), key=lambda x: x[1].get('duration_stats_ms', {}).get('mean', 0), reverse=True)
        
        for op_name, op_stats in sorted_ops[:10]:  # Top 10
            if op_stats.get('error'):
                continue
                
            duration_stats = op_stats.get('duration_stats_ms', {})
            throughput = op_stats.get('throughput', {})
            
            html_parts.append(f'''
            <tr>
                <td>{op_name}</td>
                <td>{op_stats.get('total_calls', 0):,}</td>
                <td>{op_stats.get('success_rate_percent', 0):.1f}%</td>
                <td>{duration_stats.get('mean', 0):.1f}ms</td>
                <td>{duration_stats.get('p95', 0):.1f}ms</td>
                <td>{throughput.get('calls_per_minute', 0):.1f}/min</td>
            </tr>
            ''')
            
        html_parts.append('</table>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
        
    def _generate_mode_comparison_html(self, comparison: Dict[str, Any]) -> str:
        """Generate 1s vs regular mode comparison HTML"""
        if not comparison.get('data_available'):
            return ''
            
        html_parts = ['<div class="card full-width">']
        html_parts.append('<h2>🔄 1s vs Regular Mode Performance Comparison</h2>')
        
        html_parts.append('<table class="comparison-table">')
        html_parts.append('<tr><th>Mode</th><th>Avg Duration</th><th>P95 Duration</th><th>Max Duration</th><th>Throughput</th></tr>')
        
        for mode in ['1s_mode', 'regular_mode']:
            if mode in comparison:
                mode_data = comparison[mode]
                display_mode = '1-Second Mode' if mode == '1s_mode' else 'Regular Mode'
                
                html_parts.append(f'''
                <tr>
                    <td>{display_mode}</td>
                    <td>{mode_data.get('avg_duration_ms', 0):.1f}ms</td>
                    <td>{mode_data.get('p95_duration_ms', 0):.1f}ms</td>
                    <td>{mode_data.get('max_duration_ms', 0):.1f}ms</td>
                    <td>{mode_data.get('throughput_ops_per_min', 0):.1f}/min</td>
                </tr>
                ''')
                
        html_parts.append('</table>')
        
        # Add recommendations
        if comparison.get('recommendations'):
            html_parts.append('<h3>Mode-Specific Recommendations:</h3>')
            for rec in comparison['recommendations']:
                html_parts.append(f'<div class="recommendation">💡 {rec}</div>')
                
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
        
    def generate_simple_report(self, output_path: str = None) -> str:
        """
        Generate simple text-based performance report
        
        Args:
            output_path: Output path for report file
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            report_dir = Path("data/performance_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(report_dir / f"performance_report_{int(time.time())}.txt")
            
        # Get performance data
        report = self.monitor.analyzer.generate_performance_report()
        
        # Generate text report
        report_lines = [
            "=" * 60,
            "SQUEEZEFLOW PERFORMANCE REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "SUMMARY:",
            f"  Operations tracked: {report['summary'].get('total_operations_tracked', 0)}",
            f"  Total measurements: {report['summary'].get('total_measurements', 0)}",
            f"  Monitoring uptime: {report['summary'].get('monitoring_uptime_minutes', 0):.1f} minutes",
            f"  Slow operations: {report['summary'].get('slow_operations_detected', 0)}",
            "",
        ]
        
        # System performance
        system_perf = report.get('system_performance', {})
        if not system_perf.get('error'):
            report_lines.extend([
                "SYSTEM PERFORMANCE:",
                f"  CPU usage: {system_perf['cpu_usage_percent']['current']:.1f}% (avg: {system_perf['cpu_usage_percent']['mean']:.1f}%)",
                f"  Memory usage: {system_perf['memory_usage_mb']['current']:.0f}MB (avg: {system_perf['memory_usage_mb']['mean']:.0f}MB)",
                ""
            ])
            
        # Bottlenecks
        bottlenecks = report.get('bottlenecks', [])
        if bottlenecks:
            report_lines.extend(["PERFORMANCE BOTTLENECKS:"])
            for bottleneck in bottlenecks[:5]:
                report_lines.append(f"  - {bottleneck['type']}: {bottleneck.get('operation_name', 'System')} ({bottleneck.get('severity', 'unknown')})")
            report_lines.append("")
            
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            report_lines.extend(["RECOMMENDATIONS:"])
            for rec in recommendations[:5]:
                report_lines.append(f"  - {rec}")
            report_lines.append("")
            
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        self.logger.info(f"Performance report generated: {output_path}")
        return output_path


if __name__ == "__main__":
    # Demonstration of dashboard generation
    from utils.performance_monitor import PerformanceMonitorIntegration
    import numpy as np
    import time
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Initialize monitor
    monitor = PerformanceMonitorIntegration()
    dashboard = PerformanceDashboard(monitor)
    
    try:
        logger.info("Running dashboard demonstration...")
        
        # Generate some sample performance data
        for i in range(50):
            with monitor.timer("sample_operation", {"iteration": i}):
                time.sleep(0.01 + np.random.random() * 0.05)
                
            # Simulate cache operations
            if np.random.random() < 0.8:
                monitor.record_cache_hit()
            else:
                monitor.record_cache_miss()
                
        # Wait for system metrics
        time.sleep(2)
        
        # Generate dashboard
        dashboard_path = dashboard.generate_dashboard()
        print(f"\nDashboard generated: {dashboard_path}")
        
        # Generate simple report  
        report_path = dashboard.generate_simple_report()
        print(f"Report generated: {report_path}")
        
        print("\nDashboard demonstration completed!")
        
    except Exception as e:
        logger.error(f"Dashboard demonstration error: {e}")
    finally:
        monitor.cleanup()