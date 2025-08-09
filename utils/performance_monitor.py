#!/usr/bin/env python3
"""
Performance Monitoring Utilities for SqueezeFlow Trader

This module provides comprehensive performance monitoring capabilities including:
- Processing time tracking per window
- CPU and memory usage monitoring  
- Operation timing with context managers
- Bottleneck identification
- Performance dashboard generation
- 1s vs regular mode comparison

Key features:
- Real-time metrics collection
- Memory usage tracking with psutil
- CPU utilization monitoring  
- Operation timing with context managers
- Performance report generation
- Bottleneck identification
- Cache hit rate monitoring
- Thread-safe operation
"""

import time
import psutil
import threading
import contextlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, NamedTuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import gc
import sys
import tracemalloc
from functools import wraps

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class TimingResult(NamedTuple):
    """Result from timing an operation"""
    operation_name: str
    duration_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics collection"""
    
    # Timing metrics
    operation_timings: Dict[str, List[TimingResult]] = field(default_factory=lambda: defaultdict(list))
    
    # System metrics  
    cpu_usage_history: List[float] = field(default_factory=list)
    memory_usage_history: List[float] = field(default_factory=list)
    memory_timestamps: List[datetime] = field(default_factory=list)
    
    # Window processing metrics
    window_processing_times: List[float] = field(default_factory=list)
    window_timestamps: List[datetime] = field(default_factory=list)
    
    # Data loading metrics
    data_loading_times: List[float] = field(default_factory=list)
    data_loading_sizes: List[int] = field(default_factory=list)
    
    # CVD calculation metrics
    cvd_calculation_times: List[float] = field(default_factory=list)
    cvd_data_points: List[int] = field(default_factory=list)
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rates: List[float] = field(default_factory=list)
    
    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    slow_operations: List[TimingResult] = field(default_factory=list)
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Configuration
    slow_threshold_ms: float = 1000.0  # 1 second default
    max_history_size: int = 10000


class PerformanceTimer:
    """Context manager for measuring operation performance"""
    
    def __init__(self, metrics: PerformanceMetrics, operation_name: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 record_slow: bool = True):
        self.metrics = metrics
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.record_slow = record_slow
        self.start_time = None
        self.success = True
        self.error_message = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return
            
        duration_seconds = time.perf_counter() - self.start_time
        duration_ms = duration_seconds * 1000
        
        # Handle exceptions
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val) if exc_val else str(exc_type)
            
        # Create timing result
        timing_result = TimingResult(
            operation_name=self.operation_name,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            metadata=self.metadata,
            success=self.success,
            error_message=self.error_message
        )
        
        # Record timing with thread safety
        with self.metrics._lock:
            self.metrics.operation_timings[self.operation_name].append(timing_result)
            
            # Limit history size
            if len(self.metrics.operation_timings[self.operation_name]) > self.metrics.max_history_size:
                self.metrics.operation_timings[self.operation_name] = \
                    self.metrics.operation_timings[self.operation_name][-self.metrics.max_history_size:]
            
            # Track slow operations
            if self.record_slow and duration_ms > self.metrics.slow_threshold_ms:
                self.metrics.slow_operations.append(timing_result)
                if len(self.metrics.slow_operations) > 1000:  # Limit slow operations history
                    self.metrics.slow_operations = self.metrics.slow_operations[-1000:]
            
            # Track errors
            if not self.success:
                self.metrics.error_counts[self.operation_name] += 1


class SystemMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self, metrics: PerformanceMetrics, interval_seconds: float = 1.0):
        self.metrics = metrics
        self.interval_seconds = interval_seconds
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                current_time = datetime.now()
                
                with self.metrics._lock:
                    self.metrics.cpu_usage_history.append(cpu_percent)
                    self.metrics.memory_usage_history.append(memory_mb)
                    self.metrics.memory_timestamps.append(current_time)
                    
                    # Limit history
                    max_points = self.metrics.max_history_size
                    if len(self.metrics.cpu_usage_history) > max_points:
                        self.metrics.cpu_usage_history = self.metrics.cpu_usage_history[-max_points:]
                    if len(self.metrics.memory_usage_history) > max_points:
                        self.metrics.memory_usage_history = self.metrics.memory_usage_history[-max_points:]
                    if len(self.metrics.memory_timestamps) > max_points:
                        self.metrics.memory_timestamps = self.metrics.memory_timestamps[-max_points:]
                
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                # Log error but continue monitoring
                logging.warning(f"System monitoring error: {e}")
                time.sleep(self.interval_seconds)


class PerformanceAnalyzer:
    """Analysis and reporting for performance metrics"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
    def get_operation_stats(self, operation_name: str, time_window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive statistics for an operation"""
        
        with self.metrics._lock:
            timings = self.metrics.operation_timings.get(operation_name, [])
            
        if not timings:
            return {"error": f"No timing data for operation: {operation_name}"}
            
        # Filter by time window if specified
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            timings = [t for t in timings if t.timestamp >= cutoff_time]
            
        if not timings:
            return {"error": f"No recent timing data for operation: {operation_name}"}
            
        durations = [t.duration_ms for t in timings]
        successful_timings = [t for t in timings if t.success]
        failed_timings = [t for t in timings if not t.success]
        
        stats = {
            "operation_name": operation_name,
            "total_calls": len(timings),
            "successful_calls": len(successful_timings),
            "failed_calls": len(failed_timings),
            "success_rate_percent": (len(successful_timings) / len(timings)) * 100,
            "duration_stats_ms": {
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "p95": float(np.percentile(durations, 95)),
                "p99": float(np.percentile(durations, 99))
            },
            "throughput": {
                "calls_per_minute": len(timings) / max(1, time_window_minutes or 60),
                "avg_calls_per_second": len(timings) / max(1, (timings[-1].timestamp - timings[0].timestamp).total_seconds()) if len(timings) > 1 else 0
            },
            "time_range": {
                "start": timings[0].timestamp.isoformat() if timings else None,
                "end": timings[-1].timestamp.isoformat() if timings else None
            }
        }
        
        # Add error analysis if there are failures
        if failed_timings:
            error_messages = [t.error_message for t in failed_timings if t.error_message]
            error_types = {}
            for msg in error_messages:
                error_type = msg.split(":")[0] if ":" in msg else msg
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
            stats["error_analysis"] = {
                "most_common_errors": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        return stats
        
    def get_system_stats(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get system resource statistics"""
        
        with self.metrics._lock:
            cpu_history = self.metrics.cpu_usage_history.copy()
            memory_history = self.metrics.memory_usage_history.copy()
            timestamps = self.metrics.memory_timestamps.copy()
            
        if not cpu_history or not memory_history:
            return {"error": "No system monitoring data available"}
            
        # Filter by time window
        if timestamps and time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            filtered_data = [(cpu, mem, ts) for cpu, mem, ts in zip(cpu_history, memory_history, timestamps) 
                           if ts >= cutoff_time]
            
            if filtered_data:
                cpu_history, memory_history, timestamps = zip(*filtered_data)
                cpu_history = list(cpu_history)
                memory_history = list(memory_history)
                
        if not cpu_history:
            return {"error": "No recent system monitoring data available"}
            
        return {
            "cpu_usage_percent": {
                "current": cpu_history[-1] if cpu_history else 0,
                "mean": float(np.mean(cpu_history)),
                "max": float(np.max(cpu_history)),
                "min": float(np.min(cpu_history))
            },
            "memory_usage_mb": {
                "current": memory_history[-1] if memory_history else 0,
                "mean": float(np.mean(memory_history)),
                "max": float(np.max(memory_history)),
                "min": float(np.min(memory_history))
            },
            "data_points": len(cpu_history),
            "time_window_minutes": time_window_minutes,
            "monitoring_duration_minutes": (timestamps[-1] - timestamps[0]).total_seconds() / 60 if len(timestamps) > 1 else 0
        }
        
    def get_window_processing_stats(self) -> Dict[str, Any]:
        """Get window processing performance statistics"""
        
        with self.metrics._lock:
            processing_times = self.metrics.window_processing_times.copy()
            timestamps = self.metrics.window_timestamps.copy()
            
        if not processing_times:
            return {"error": "No window processing data available"}
            
        return {
            "total_windows_processed": len(processing_times),
            "processing_time_stats_ms": {
                "mean": float(np.mean(processing_times)),
                "median": float(np.median(processing_times)),
                "std": float(np.std(processing_times)),
                "min": float(np.min(processing_times)),
                "max": float(np.max(processing_times)),
                "p95": float(np.percentile(processing_times, 95)),
                "p99": float(np.percentile(processing_times, 99))
            },
            "throughput_windows_per_minute": len(processing_times) / max(1, 
                (timestamps[-1] - timestamps[0]).total_seconds() / 60) if len(timestamps) > 1 else 0,
            "total_processing_time_minutes": sum(processing_times) / (1000 * 60)
        }
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        with self.metrics._lock:
            total_requests = self.metrics.cache_hits + self.metrics.cache_misses
            
        if total_requests == 0:
            return {"error": "No cache data available"}
            
        return {
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": (self.metrics.cache_hits / total_requests) * 100,
            "miss_rate_percent": (self.metrics.cache_misses / total_requests) * 100
        }
        
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        # Check for slow operations
        with self.metrics._lock:
            slow_ops = self.metrics.slow_operations.copy()
            
        if slow_ops:
            # Group by operation name
            slow_by_operation = defaultdict(list)
            for op in slow_ops:
                slow_by_operation[op.operation_name].append(op)
                
            for op_name, operations in slow_by_operation.items():
                durations = [op.duration_ms for op in operations]
                bottlenecks.append({
                    "type": "slow_operation",
                    "operation_name": op_name,
                    "slow_call_count": len(operations),
                    "avg_duration_ms": float(np.mean(durations)),
                    "max_duration_ms": float(np.max(durations)),
                    "severity": "high" if np.mean(durations) > 5000 else "medium"
                })
                
        # Check for high error rates
        with self.metrics._lock:
            for op_name, timings in self.metrics.operation_timings.items():
                if len(timings) < 10:  # Need minimum sample size
                    continue
                    
                failed_count = sum(1 for t in timings if not t.success)
                error_rate = (failed_count / len(timings)) * 100
                
                if error_rate > 5:  # More than 5% error rate
                    bottlenecks.append({
                        "type": "high_error_rate", 
                        "operation_name": op_name,
                        "error_rate_percent": error_rate,
                        "total_calls": len(timings),
                        "failed_calls": failed_count,
                        "severity": "critical" if error_rate > 20 else "high"
                    })
                    
        # Check for high memory usage
        system_stats = self.get_system_stats(30)  # Last 30 minutes
        if not system_stats.get("error") and system_stats.get("memory_usage_mb", {}).get("max", 0) > 2000:  # >2GB
            bottlenecks.append({
                "type": "high_memory_usage",
                "max_memory_mb": system_stats["memory_usage_mb"]["max"],
                "current_memory_mb": system_stats["memory_usage_mb"]["current"],
                "severity": "high" if system_stats["memory_usage_mb"]["max"] > 4000 else "medium"
            })
            
        # Check for poor cache performance
        cache_stats = self.get_cache_stats()
        if not cache_stats.get("error") and cache_stats.get("hit_rate_percent", 100) < 50:  # <50% hit rate
            bottlenecks.append({
                "type": "poor_cache_performance",
                "hit_rate_percent": cache_stats["hit_rate_percent"],
                "total_requests": cache_stats["total_requests"],
                "severity": "medium"
            })
            
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        bottlenecks.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return bottlenecks
        
    def generate_performance_report(self, output_format: str = "dict") -> Union[Dict[str, Any], str]:
        """Generate comprehensive performance report"""
        
        report_time = datetime.now()
        
        # Collect all statistics
        operation_stats = {}
        with self.metrics._lock:
            for op_name in self.metrics.operation_timings.keys():
                operation_stats[op_name] = self.get_operation_stats(op_name, 60)  # Last hour
                
        report = {
            "report_timestamp": report_time.isoformat(),
            "summary": {
                "total_operations_tracked": len(operation_stats),
                "total_measurements": sum(len(timings) for timings in self.metrics.operation_timings.values()),
                "monitoring_uptime_minutes": self._get_monitoring_uptime(),
                "slow_operations_detected": len(self.metrics.slow_operations)
            },
            "operation_performance": operation_stats,
            "system_performance": self.get_system_stats(60),
            "window_processing": self.get_window_processing_stats(),
            "cache_performance": self.get_cache_stats(),
            "bottlenecks": self.identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
        
        if output_format == "json":
            return json.dumps(report, indent=2, default=str)
        elif output_format == "dict":
            return report
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    def _get_monitoring_uptime(self) -> float:
        """Get monitoring uptime in minutes"""
        with self.metrics._lock:
            if not self.metrics.memory_timestamps:
                return 0.0
            return (datetime.now() - self.metrics.memory_timestamps[0]).total_seconds() / 60
            
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        
        recommendations = []
        
        # Check bottlenecks for recommendations
        bottlenecks = self.identify_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_operation":
                recommendations.append(
                    f"Optimize {bottleneck['operation_name']} operation - "
                    f"average {bottleneck['avg_duration_ms']:.0f}ms is above threshold"
                )
            elif bottleneck["type"] == "high_error_rate":
                recommendations.append(
                    f"Investigate {bottleneck['operation_name']} errors - "
                    f"{bottleneck['error_rate_percent']:.1f}% failure rate"
                )
            elif bottleneck["type"] == "high_memory_usage":
                recommendations.append(
                    f"Monitor memory usage - peaked at {bottleneck['max_memory_mb']:.0f}MB. "
                    "Consider implementing memory cleanup or reducing data retention"
                )
            elif bottleneck["type"] == "poor_cache_performance":
                recommendations.append(
                    f"Improve cache strategy - only {bottleneck['hit_rate_percent']:.1f}% hit rate"
                )
                
        # General recommendations based on patterns
        system_stats = self.get_system_stats(60)
        if not system_stats.get("error"):
            if system_stats["cpu_usage_percent"]["mean"] > 80:
                recommendations.append("High CPU usage detected - consider parallel processing or optimization")
            if system_stats["memory_usage_mb"]["current"] > 1000:
                recommendations.append("Memory usage is high - monitor for memory leaks")
                
        if not recommendations:
            recommendations.append("Performance is within normal parameters")
            
        return recommendations


class PerformanceMonitorIntegration:
    """Integration utilities for existing systems"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.metrics = PerformanceMetrics()
        self.analyzer = PerformanceAnalyzer(self.metrics)
        self.system_monitor = None
        self.enable_monitoring = enable_monitoring
        self.logger = logging.getLogger(__name__)
        
        if enable_monitoring:
            self.start_system_monitoring()
            
    def start_system_monitoring(self):
        """Start system resource monitoring"""
        if self.system_monitor is None:
            self.system_monitor = SystemMonitor(self.metrics)
            self.system_monitor.start_monitoring()
            self.logger.info("System performance monitoring started")
            
    def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
            self.system_monitor = None
            self.logger.info("System performance monitoring stopped")
            
    def timer(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceTimer:
        """Create a performance timer context manager"""
        return PerformanceTimer(self.metrics, operation_name, metadata)
        
    def record_window_processing_time(self, duration_ms: float):
        """Record window processing time"""
        with self.metrics._lock:
            self.metrics.window_processing_times.append(duration_ms)
            self.metrics.window_timestamps.append(datetime.now())
            
    def record_data_loading(self, duration_ms: float, data_size: int):
        """Record data loading performance"""
        with self.metrics._lock:
            self.metrics.data_loading_times.append(duration_ms)
            self.metrics.data_loading_sizes.append(data_size)
            
    def record_cvd_calculation(self, duration_ms: float, data_points: int):
        """Record CVD calculation performance"""
        with self.metrics._lock:
            self.metrics.cvd_calculation_times.append(duration_ms)
            self.metrics.cvd_data_points.append(data_points)
            
    def record_cache_hit(self):
        """Record cache hit"""
        with self.metrics._lock:
            self.metrics.cache_hits += 1
            
    def record_cache_miss(self):
        """Record cache miss"""
        with self.metrics._lock:
            self.metrics.cache_misses += 1
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get quick performance summary"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system": self.analyzer.get_system_stats(10),  # Last 10 minutes
            "operations": {op: self.analyzer.get_operation_stats(op, 10) 
                         for op in list(self.metrics.operation_timings.keys())[:10]},  # Top 10 operations
            "bottlenecks": self.analyzer.identify_bottlenecks()[:5]  # Top 5 bottlenecks
        }
        
    def generate_performance_chart(self, metric_type: str = "memory", output_path: Optional[str] = None) -> Optional[str]:
        """Generate performance visualization chart"""
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - cannot generate charts")
            return None
            
        plt.figure(figsize=(12, 8))
        
        if metric_type == "memory":
            with self.metrics._lock:
                memory_data = self.metrics.memory_usage_history.copy()
                timestamps = self.metrics.memory_timestamps.copy()
                
            if not memory_data:
                self.logger.warning("No memory data available for charting")
                return None
                
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, memory_data, 'b-', linewidth=2)
            plt.title('Memory Usage Over Time')
            plt.ylabel('Memory (MB)')
            plt.grid(True, alpha=0.3)
            
            # CPU usage
            with self.metrics._lock:
                cpu_data = self.metrics.cpu_usage_history.copy()
                
            plt.subplot(2, 1, 2)
            plt.plot(timestamps[:len(cpu_data)], cpu_data, 'r-', linewidth=2)
            plt.title('CPU Usage Over Time')
            plt.ylabel('CPU %')
            plt.xlabel('Time')
            plt.grid(True, alpha=0.3)
            
        elif metric_type == "operations":
            # Show operation timing distributions
            with self.metrics._lock:
                op_names = list(self.metrics.operation_timings.keys())[:4]  # Top 4 operations
                
            if not op_names:
                self.logger.warning("No operation timing data available")
                return None
                
            for i, op_name in enumerate(op_names):
                plt.subplot(2, 2, i + 1)
                timings = [t.duration_ms for t in self.metrics.operation_timings[op_name]]
                if timings:
                    plt.hist(timings, bins=20, alpha=0.7, edgecolor='black')
                    plt.title(f'{op_name} Duration Distribution')
                    plt.xlabel('Duration (ms)')
                    plt.ylabel('Frequency')
                    
        else:
            self.logger.warning(f"Unknown metric type for charting: {metric_type}")
            return None
            
        plt.tight_layout()
        
        if output_path is None:
            output_path = f"performance_{metric_type}_{int(time.time())}.png"
            
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance chart saved: {output_path}")
        return output_path
        
    def export_metrics(self, output_path: str, format: str = "csv"):
        """Export metrics to file"""
        
        all_timings = []
        
        with self.metrics._lock:
            for op_name, timings in self.metrics.operation_timings.items():
                for timing in timings:
                    all_timings.append({
                        "timestamp": timing.timestamp.isoformat(),
                        "operation": timing.operation_name,
                        "duration_ms": timing.duration_ms,
                        "success": timing.success,
                        "error_message": timing.error_message,
                        "metadata": json.dumps(timing.metadata)
                    })
                    
        if not all_timings:
            self.logger.warning("No timing data to export")
            return
            
        df = pd.DataFrame(all_timings)
        
        if format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            df.to_json(output_path, orient="records", date_format="iso")
        elif format.lower() == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        self.logger.info(f"Exported {len(all_timings)} timing records to {output_path}")
        
    def get_1s_vs_regular_comparison(self) -> Dict[str, Any]:
        """Compare 1s mode vs regular mode performance if data available"""
        
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "data_available": False,
            "recommendations": []
        }
        
        # Look for operations that might indicate different modes
        with self.metrics._lock:
            operations = self.metrics.operation_timings.copy()
            
        # Check for patterns that suggest 1s vs regular mode
        mode_indicators = {}
        for op_name, timings in operations.items():
            if "1s" in op_name.lower() or "second" in op_name.lower():
                mode_indicators["1s_mode"] = mode_indicators.get("1s_mode", []) + timings
            elif "5m" in op_name.lower() or "minute" in op_name.lower():
                mode_indicators["regular_mode"] = mode_indicators.get("regular_mode", []) + timings
                
        if len(mode_indicators) >= 2:
            comparison["data_available"] = True
            
            for mode, timings in mode_indicators.items():
                if timings:
                    durations = [t.duration_ms for t in timings]
                    comparison[mode] = {
                        "count": len(timings),
                        "avg_duration_ms": float(np.mean(durations)),
                        "p95_duration_ms": float(np.percentile(durations, 95)),
                        "max_duration_ms": float(np.max(durations)),
                        "throughput_ops_per_min": len(timings) / max(1, 
                            (timings[-1].timestamp - timings[0].timestamp).total_seconds() / 60) if len(timings) > 1 else 0
                    }
                    
            # Generate recommendations
            if "1s_mode" in comparison and "regular_mode" in comparison:
                if comparison["1s_mode"]["avg_duration_ms"] > comparison["regular_mode"]["avg_duration_ms"] * 2:
                    comparison["recommendations"].append(
                        "1s mode shows significantly higher processing times - consider optimization"
                    )
                if comparison["1s_mode"]["throughput_ops_per_min"] < comparison["regular_mode"]["throughput_ops_per_min"]:
                    comparison["recommendations"].append(
                        "1s mode has lower throughput - ensure adequate system resources"
                    )
        else:
            comparison["recommendations"].append("Insufficient data to compare 1s vs regular mode performance")
            
        return comparison
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop_system_monitoring()


# Decorator for automatic timing
def timed_operation(monitor: PerformanceMonitorIntegration, operation_name: str, 
                   metadata: Optional[Dict[str, Any]] = None):
    """Decorator to automatically time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.timer(operation_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global instance for easy access
_global_monitor: Optional[PerformanceMonitorIntegration] = None

def get_global_monitor() -> PerformanceMonitorIntegration:
    """Get or create global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitorIntegration()
    return _global_monitor


def initialize_global_monitor(enable_monitoring: bool = True) -> PerformanceMonitorIntegration:
    """Initialize global performance monitor"""
    global _global_monitor
    _global_monitor = PerformanceMonitorIntegration(enable_monitoring)
    return _global_monitor


def cleanup_global_monitor():
    """Cleanup global performance monitor"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.cleanup()
        _global_monitor = None


# Convenience functions for common operations
def time_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceTimer:
    """Convenience function to time an operation using global monitor"""
    return get_global_monitor().timer(operation_name, metadata)


if __name__ == "__main__":
    # Demonstration of the performance monitoring system
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Initialize monitor
    monitor = PerformanceMonitorIntegration()
    
    try:
        # Simulate some operations
        logger.info("Running performance monitoring demonstration...")
        
        # Simulate window processing
        for i in range(10):
            with monitor.timer("window_processing", {"window_id": i}):
                time.sleep(0.1 + np.random.random() * 0.2)  # Simulate variable processing time
                
        # Simulate data loading
        for i in range(5):
            with monitor.timer("data_loading"):
                time.sleep(0.05 + np.random.random() * 0.1)
                monitor.record_data_loading((0.05 + np.random.random() * 0.1) * 1000, 1000 + i * 100)
                
        # Simulate CVD calculations
        for i in range(20):
            with monitor.timer("cvd_calculation"):
                time.sleep(0.01 + np.random.random() * 0.02)
                monitor.record_cvd_calculation((0.01 + np.random.random() * 0.02) * 1000, 500 + i * 10)
                
        # Simulate cache operations
        for i in range(100):
            if np.random.random() < 0.7:  # 70% hit rate
                monitor.record_cache_hit()
            else:
                monitor.record_cache_miss()
                
        # Wait for some system metrics
        time.sleep(2)
        
        # Generate reports
        logger.info("Generating performance report...")
        report = monitor.analyzer.generate_performance_report()
        
        print("\n=== PERFORMANCE REPORT ===")
        print(f"Operations tracked: {report['summary']['total_operations_tracked']}")
        print(f"Total measurements: {report['summary']['total_measurements']}")
        
        print("\n=== BOTTLENECKS ===")
        for bottleneck in report['bottlenecks'][:3]:
            print(f"- {bottleneck['type']}: {bottleneck.get('operation_name', 'System')} "
                  f"({bottleneck.get('severity', 'unknown')} severity)")
                  
        print("\n=== RECOMMENDATIONS ===")
        for rec in report['recommendations'][:3]:
            print(f"- {rec}")
            
        # Generate charts if possible
        if MATPLOTLIB_AVAILABLE:
            chart_path = monitor.generate_performance_chart("memory", "demo_performance_chart.png")
            if chart_path:
                print(f"\nPerformance chart saved to: {chart_path}")
                
        # Export metrics
        monitor.export_metrics("demo_performance_metrics.csv")
        print("Metrics exported to: demo_performance_metrics.csv")
        
        print("\nPerformance monitoring demonstration completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
    finally:
        monitor.cleanup()