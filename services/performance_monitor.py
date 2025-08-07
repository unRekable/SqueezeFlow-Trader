#!/usr/bin/env python3
"""
Performance Monitor Service - Advanced performance monitoring for SqueezeFlow services
Comprehensive metrics collection, performance dashboards, and real-time monitoring

Features:
- Real-time performance metrics collection
- Signal latency and throughput monitoring
- Memory and CPU profiling
- Custom performance dashboards
- Performance alerting and notifications
- Historical performance analysis
- Bottleneck detection and optimization suggestions
"""

import asyncio
import time
import json
import psutil
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import tracemalloc
import gc

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Chart generation disabled.")

from services.config.unified_config import ConfigManager


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq'
    duration_seconds: int  # How long condition must persist
    severity: str  # 'info', 'warning', 'critical'
    message_template: str
    enabled: bool = True
    
    def check_condition(self, value: float) -> bool:
        """Check if alert condition is met"""
        if self.comparison == 'gt':
            return value > self.threshold_value
        elif self.comparison == 'lt':
            return value < self.threshold_value
        elif self.comparison == 'eq':
            return abs(value - self.threshold_value) < 0.001
        return False


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_data_points: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.collection_lock = threading.Lock()
        
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric"""
        with self.collection_lock:
            self.metrics[metric.metric_name].append(metric)
            self._update_aggregations(metric.metric_name)
    
    def _update_aggregations(self, metric_name: str):
        """Update aggregated statistics for a metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return
        
        values = [m.value for m in self.metrics[metric_name]]
        
        self.aggregated_metrics[metric_name] = {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'latest': values[-1],
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """Calculate trend direction for recent values"""
        if len(values) < window:
            return 'insufficient_data'
        
        recent_values = values[-window:]
        if len(recent_values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > np.std(recent_values) * 0.1:
            return 'increasing'
        elif slope < -np.std(recent_values) * 0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_metrics(self, metric_name: str, time_window_minutes: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metrics for a specific metric name"""
        with self.collection_lock:
            all_metrics = list(self.metrics.get(metric_name, []))
            
            if time_window_minutes is None:
                return all_metrics
            
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            return [m for m in all_metrics if m.timestamp >= cutoff_time]
    
    def get_aggregated_stats(self, metric_name: str) -> Dict[str, float]:
        """Get aggregated statistics for a metric"""
        return self.aggregated_metrics.get(metric_name, {})


class PerformanceMonitor:
    """Advanced performance monitoring service"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize Performance Monitor
        
        Args:
            config_manager: Configuration manager (creates default if None)
        """
        # Initialize configuration
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self._setup_logging()
        
        # Metrics collection
        self.metrics_collector = MetricsCollector()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.performance_timers: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Memory tracking
        self.memory_tracker_enabled = False
        self.memory_snapshots: List[Dict[str, Any]] = []
        
        # Alert system
        self.performance_alerts: List[PerformanceAlert] = []
        self.alert_states: Dict[str, Dict[str, Any]] = {}
        
        # Thread pool for heavy operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='perf_monitor')
        
        # Dashboard data
        self.dashboard_data: Dict[str, Any] = {}
        self.dashboard_update_interval = 30  # seconds
        
        # Custom metric collectors
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Initialize default alerts
        self._setup_default_alerts()
        
        # Redis connection for metrics publishing
        self._redis_client = None
        
        self.logger.info("Performance Monitor initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('performance_monitor')
        self.logger.info(f"Performance monitoring logging configured at level: {self.config.log_level}")
    
    def _setup_default_alerts(self):
        """Setup default performance alerts"""
        
        default_alerts = [
            PerformanceAlert(
                metric_name='signal_processing_time_ms',
                threshold_value=1000.0,  # 1 second
                comparison='gt',
                duration_seconds=60,
                severity='warning',
                message_template='Signal processing time exceeded {threshold}ms: {value}ms'
            ),
            PerformanceAlert(
                metric_name='redis_publish_time_ms',
                threshold_value=100.0,  # 100ms
                comparison='gt',
                duration_seconds=30,
                severity='warning',
                message_template='Redis publish time high: {value}ms'
            ),
            PerformanceAlert(
                metric_name='memory_usage_mb',
                threshold_value=1000.0,  # 1GB
                comparison='gt',
                duration_seconds=300,  # 5 minutes
                severity='critical',
                message_template='High memory usage: {value}MB'
            ),
            PerformanceAlert(
                metric_name='cpu_usage_percent',
                threshold_value=80.0,
                comparison='gt',
                duration_seconds=120,  # 2 minutes
                severity='warning',
                message_template='High CPU usage: {value}%'
            ),
            PerformanceAlert(
                metric_name='error_rate_percent',
                threshold_value=5.0,
                comparison='gt',
                duration_seconds=60,
                severity='critical',
                message_template='High error rate: {value}%'
            )
        ]
        
        self.performance_alerts.extend(default_alerts)
        self.logger.info(f"Configured {len(default_alerts)} default performance alerts")
    
    @property
    def redis_client(self):
        """Lazy loading Redis client for metrics publishing"""
        if self._redis_client is None:
            try:
                import redis
                redis_config = self.config_manager.get_redis_config()
                self._redis_client = redis.Redis(**redis_config)
            except Exception as e:
                self.logger.warning(f"Could not initialize Redis client: {e}")
        
        return self._redis_client
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.logger.info("Starting Performance Monitor...")
        
        self.is_monitoring = True
        
        # Start system metrics collection
        system_task = asyncio.create_task(self._collect_system_metrics())
        self.monitoring_tasks.append(system_task)
        
        # Start alert monitoring
        alert_task = asyncio.create_task(self._monitor_alerts())
        self.monitoring_tasks.append(alert_task)
        
        # Start dashboard updates
        dashboard_task = asyncio.create_task(self._update_dashboard())
        self.monitoring_tasks.append(dashboard_task)
        
        # Start memory tracking if enabled
        if self.memory_tracker_enabled:
            memory_task = asyncio.create_task(self._track_memory_usage())
            self.monitoring_tasks.append(memory_task)
        
        # Start custom collectors
        for name, collector in self.custom_collectors.items():
            collector_task = asyncio.create_task(self._run_custom_collector(name, collector))
            self.monitoring_tasks.append(collector_task)
        
        self.logger.info(f"Performance Monitor started with {len(self.monitoring_tasks)} monitoring tasks")
        
        try:
            await asyncio.gather(*self.monitoring_tasks)
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.logger.info("Stopping Performance Monitor...")
        
        self.is_monitoring = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Close Redis connection
        if self._redis_client:
            self._redis_client.close()
        
        self.logger.info("Performance Monitor stopped")
    
    def start_timer(self, operation_name: str) -> str:
        """Start a performance timer for an operation"""
        timer_id = f"{operation_name}_{int(time.time() * 1000000)}"  # microsecond precision
        self.performance_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, operation_name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """End a performance timer and record the metric"""
        if timer_id not in self.performance_timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        duration_seconds = time.time() - self.performance_timers[timer_id]
        duration_ms = duration_seconds * 1000
        
        # Remove timer
        del self.performance_timers[timer_id]
        
        # Record metric
        self.record_metric(
            f"{operation_name}_time_ms",
            duration_ms,
            "milliseconds",
            tags or {},
            {'timer_id': timer_id}
        )
        
        # Update operation count
        self.operation_counts[operation_name] += 1
        
        return duration_ms
    
    def record_metric(self, metric_name: str, value: float, unit: str = "", 
                     tags: Optional[Dict[str, str]] = None, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics_collector.add_metric(metric)
        
        # Publish to Redis if available
        if self.redis_client:
            try:
                redis_key = f"{self.config.redis_key_prefix}:metrics:{metric_name}"
                self.redis_client.lpush(redis_key, json.dumps(metric.to_dict(), default=str))
                self.redis_client.ltrim(redis_key, 0, 999)  # Keep last 1000 metrics
                self.redis_client.expire(redis_key, 3600)  # 1 hour expiry
            except Exception as e:
                self.logger.debug(f"Could not publish metric to Redis: {e}")
    
    def record_error(self, operation_name: str, error_message: str, 
                    tags: Optional[Dict[str, str]] = None):
        """Record an error occurrence"""
        
        self.error_counts[operation_name] += 1
        
        # Record error rate metric
        total_operations = self.operation_counts.get(operation_name, 0) + self.error_counts[operation_name]
        error_rate = (self.error_counts[operation_name] / max(1, total_operations)) * 100
        
        self.record_metric(
            f"{operation_name}_error_rate_percent",
            error_rate,
            "percent",
            tags,
            {'error_message': error_message, 'total_operations': total_operations}
        )
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        
        while self.is_monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric('cpu_usage_percent', cpu_percent, 'percent')
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_metric('memory_usage_percent', memory.percent, 'percent')
                self.record_metric('memory_usage_mb', memory.used / 1024**2, 'megabytes')
                self.record_metric('memory_available_mb', memory.available / 1024**2, 'megabytes')
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_metric('disk_usage_percent', (disk.used / disk.total) * 100, 'percent')
                self.record_metric('disk_free_gb', disk.free / 1024**3, 'gigabytes')
                
                # Network metrics (if available)
                try:
                    network = psutil.net_io_counters()
                    self.record_metric('network_bytes_sent', network.bytes_sent, 'bytes')
                    self.record_metric('network_bytes_recv', network.bytes_recv, 'bytes')
                except Exception:
                    pass  # Network metrics not available on all systems
                
                # Process-specific metrics
                process = psutil.Process()
                self.record_metric('process_memory_mb', process.memory_info().rss / 1024**2, 'megabytes')
                self.record_metric('process_cpu_percent', process.cpu_percent(), 'percent')
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_alerts(self):
        """Monitor performance alerts"""
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                for alert in self.performance_alerts:
                    if not alert.enabled:
                        continue
                    
                    # Get recent metrics for this alert
                    recent_metrics = self.metrics_collector.get_metrics(alert.metric_name, 5)  # Last 5 minutes
                    
                    if not recent_metrics:
                        continue
                    
                    # Check if condition is met
                    latest_value = recent_metrics[-1].value
                    condition_met = alert.check_condition(latest_value)
                    
                    alert_key = f"{alert.metric_name}_{alert.threshold_value}"
                    
                    if condition_met:
                        # Check if this is a new alert or continuing condition
                        if alert_key not in self.alert_states:
                            self.alert_states[alert_key] = {
                                'first_triggered': current_time,
                                'last_triggered': current_time,
                                'triggered_count': 1
                            }
                        else:
                            self.alert_states[alert_key]['last_triggered'] = current_time
                            self.alert_states[alert_key]['triggered_count'] += 1
                        
                        # Check if duration threshold is met
                        duration = (current_time - self.alert_states[alert_key]['first_triggered']).total_seconds()
                        
                        if duration >= alert.duration_seconds:
                            await self._trigger_performance_alert(alert, latest_value, duration)
                    
                    else:
                        # Condition not met, clear alert state
                        if alert_key in self.alert_states:
                            del self.alert_states[alert_key]
                
                await asyncio.sleep(30)  # Check alerts every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring alerts: {e}")
                await asyncio.sleep(30)
    
    async def _trigger_performance_alert(self, alert: PerformanceAlert, current_value: float, duration: float):
        """Trigger a performance alert"""
        
        message = alert.message_template.format(
            threshold=alert.threshold_value,
            value=current_value,
            duration=duration
        )
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': alert.metric_name,
            'severity': alert.severity,
            'message': message,
            'current_value': current_value,
            'threshold_value': alert.threshold_value,
            'duration_seconds': duration
        }
        
        # Log alert
        if alert.severity == 'critical':
            self.logger.critical(f"PERFORMANCE ALERT: {message}")
        elif alert.severity == 'warning':
            self.logger.warning(f"PERFORMANCE ALERT: {message}")
        else:
            self.logger.info(f"PERFORMANCE ALERT: {message}")
        
        # Publish to Redis
        if self.redis_client:
            try:
                alert_channel = f"{self.config.redis_key_prefix}:performance_alerts"
                self.redis_client.publish(alert_channel, json.dumps(alert_data))
            except Exception as e:
                self.logger.error(f"Failed to publish performance alert: {e}")
    
    async def _update_dashboard(self):
        """Update dashboard data"""
        
        while self.is_monitoring:
            try:
                dashboard_data = {}
                
                # Collect key metrics for dashboard
                key_metrics = [
                    'cpu_usage_percent',
                    'memory_usage_percent',
                    'signal_processing_time_ms',
                    'redis_publish_time_ms',
                    'error_rate_percent'
                ]
                
                for metric_name in key_metrics:
                    stats = self.metrics_collector.get_aggregated_stats(metric_name)
                    if stats:
                        dashboard_data[metric_name] = stats
                
                # Add operation counts
                dashboard_data['operation_counts'] = dict(self.operation_counts)
                dashboard_data['error_counts'] = dict(self.error_counts)
                
                # Add system info
                dashboard_data['system_info'] = {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                    'disk_total_gb': psutil.disk_usage('/').total / 1024**3,
                    'boot_time': psutil.boot_time()
                }
                
                # Add alert status
                dashboard_data['active_alerts'] = len(self.alert_states)
                dashboard_data['alert_states'] = {
                    k: {
                        'duration_seconds': (datetime.now() - v['first_triggered']).total_seconds(),
                        'triggered_count': v['triggered_count']
                    }
                    for k, v in self.alert_states.items()
                }
                
                dashboard_data['last_updated'] = datetime.now().isoformat()
                
                self.dashboard_data = dashboard_data
                
                # Publish dashboard data to Redis
                if self.redis_client:
                    try:
                        dashboard_key = f"{self.config.redis_key_prefix}:dashboard:performance"
                        self.redis_client.setex(dashboard_key, 3600, json.dumps(dashboard_data, default=str))
                    except Exception as e:
                        self.logger.debug(f"Could not publish dashboard data: {e}")
                
                await asyncio.sleep(self.dashboard_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(self.dashboard_update_interval)
    
    async def _track_memory_usage(self):
        """Track detailed memory usage"""
        
        if not self.memory_tracker_enabled:
            return
        
        # Start tracemalloc
        tracemalloc.start()
        
        while self.is_monitoring:
            try:
                # Get current memory snapshot
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                # Collect memory info
                memory_info = {
                    'timestamp': datetime.now().isoformat(),
                    'total_size_mb': sum(stat.size for stat in top_stats) / 1024**2,
                    'top_allocations': [
                        {
                            'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                            'size_mb': stat.size / 1024**2,
                            'count': stat.count
                        }
                        for stat in top_stats[:10]  # Top 10 allocations
                    ]
                }
                
                self.memory_snapshots.append(memory_info)
                
                # Keep only recent snapshots
                if len(self.memory_snapshots) > 100:
                    self.memory_snapshots = self.memory_snapshots[-100:]
                
                # Record total memory metric
                self.record_metric('python_memory_mb', memory_info['total_size_mb'], 'megabytes')
                
                await asyncio.sleep(60)  # Track every minute
                
            except Exception as e:
                self.logger.error(f"Error tracking memory: {e}")
                await asyncio.sleep(60)
    
    async def _run_custom_collector(self, name: str, collector_func: Callable):
        """Run a custom metrics collector"""
        
        while self.is_monitoring:
            try:
                # Run collector function
                if asyncio.iscoroutinefunction(collector_func):
                    await collector_func(self)
                else:
                    collector_func(self)
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in custom collector {name}: {e}")
                await asyncio.sleep(30)
    
    def enable_memory_tracking(self):
        """Enable detailed memory tracking"""
        self.memory_tracker_enabled = True
        self.logger.info("Memory tracking enabled")
    
    def add_custom_collector(self, name: str, collector_func: Callable):
        """Add a custom metrics collector"""
        self.custom_collectors[name] = collector_func
        self.logger.info(f"Added custom metrics collector: {name}")
    
    def add_performance_alert(self, alert: PerformanceAlert):
        """Add a custom performance alert"""
        self.performance_alerts.append(alert)
        self.logger.info(f"Added performance alert for {alert.metric_name}")
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        
        summary = {
            'time_window_minutes': time_window_minutes,
            'metrics': {},
            'operation_counts': dict(self.operation_counts),
            'error_counts': dict(self.error_counts),
            'active_alerts': len(self.alert_states),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get aggregated stats for all metrics
        for metric_name in self.metrics_collector.metrics.keys():
            stats = self.metrics_collector.get_aggregated_stats(metric_name)
            if stats:
                # Add recent data points
                recent_metrics = self.metrics_collector.get_metrics(metric_name, time_window_minutes)
                
                summary['metrics'][metric_name] = {
                    **stats,
                    'recent_data_points': len(recent_metrics),
                    'time_range': {
                        'start': recent_metrics[0].timestamp.isoformat() if recent_metrics else None,
                        'end': recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
                    }
                }
        
        return summary
    
    def generate_performance_chart(self, metric_name: str, time_window_minutes: int = 60, 
                                 output_path: Optional[str] = None) -> Optional[str]:
        """Generate a performance chart for a metric"""
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available. Cannot generate charts.")
            return None
        
        # Get metrics data
        metrics = self.metrics_collector.get_metrics(metric_name, time_window_minutes)
        
        if not metrics:
            self.logger.warning(f"No data available for metric: {metric_name}")
            return None
        
        # Prepare data
        timestamps = [m.timestamp for m in metrics]
        values = [m.value for m in metrics]
        
        # Create chart
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, linewidth=2, color='blue')
        plt.title(f'Performance Metric: {metric_name}')
        plt.xlabel('Time')
        plt.ylabel(f'Value ({metrics[0].unit})' if metrics[0].unit else 'Value')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, time_window_minutes // 10)))
        plt.xticks(rotation=45)
        
        # Add statistics
        stats = self.metrics_collector.get_aggregated_stats(metric_name)
        if stats:
            plt.axhline(y=stats['mean'], color='red', linestyle='--', alpha=0.7, label=f"Mean: {stats['mean']:.2f}")
            plt.axhline(y=stats['p95'], color='orange', linestyle='--', alpha=0.7, label=f"95th %ile: {stats['p95']:.2f}")
            plt.legend()
        
        plt.tight_layout()
        
        # Save chart
        if output_path is None:
            output_path = f"data/charts/performance_{metric_name}_{int(time.time())}.png"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance chart saved: {output_path}")
        return output_path
    
    def export_metrics(self, output_path: str, time_window_minutes: int = 60, format: str = 'csv'):
        """Export metrics data to file"""
        
        all_metrics = []
        
        for metric_name in self.metrics_collector.metrics.keys():
            metrics = self.metrics_collector.get_metrics(metric_name, time_window_minutes)
            for metric in metrics:
                all_metrics.append(metric.to_dict())
        
        if not all_metrics:
            self.logger.warning("No metrics data to export")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Export based on format
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(all_metrics)} metrics to {output_path}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            'monitoring_status': {
                'is_monitoring': self.is_monitoring,
                'monitoring_tasks': len(self.monitoring_tasks),
                'memory_tracking_enabled': self.memory_tracker_enabled
            },
            'metrics_summary': self.get_metrics_summary(60),  # Last hour
            'dashboard_data': self.dashboard_data,
            'performance_alerts': {
                'total_alerts': len(self.performance_alerts),
                'active_alerts': len(self.alert_states),
                'alert_definitions': [
                    {
                        'metric_name': alert.metric_name,
                        'threshold': alert.threshold_value,
                        'severity': alert.severity,
                        'enabled': alert.enabled
                    }
                    for alert in self.performance_alerts
                ]
            },
            'memory_tracking': {
                'enabled': self.memory_tracker_enabled,
                'snapshots_collected': len(self.memory_snapshots),
                'latest_snapshot': self.memory_snapshots[-1] if self.memory_snapshots else None
            },
            'custom_collectors': list(self.custom_collectors.keys()),
            'timestamp': datetime.now().isoformat()
        }


# Context manager for performance timing
class PerformanceTimer:
    """Context manager for measuring performance"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, 
                 tags: Optional[Dict[str, str]] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.tags = tags or {}
        self.timer_id = None
    
    def __enter__(self):
        self.timer_id = self.monitor.start_timer(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            duration = self.monitor.end_timer(self.timer_id, self.operation_name, self.tags)
            
            # Record error if exception occurred
            if exc_type is not None:
                self.monitor.record_error(self.operation_name, str(exc_val), self.tags)


async def main():
    """Main entry point for running the performance monitor service"""
    
    # Initialize service
    config_manager = ConfigManager()
    monitor = PerformanceMonitor(config_manager)
    
    # Enable memory tracking for demonstration
    monitor.enable_memory_tracking()
    
    try:
        # Start monitoring
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    except Exception as e:
        print(f"Performance Monitor error: {e}")
    finally:
        await monitor.stop_monitoring()


if __name__ == "__main__":
    # Setup logging for standalone execution
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('performance_monitor_main')
    logger.info("Starting SqueezeFlow Performance Monitor")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.fatal(f"Failed to start Performance Monitor: {e}")
        sys.exit(1)