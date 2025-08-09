#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring Script for SqueezeFlow Trader

Real-time performance monitoring with comprehensive metrics collection,
bottleneck detection, and dashboard generation.

Usage:
    python scripts/performance_monitor_comprehensive.py [--duration MINUTES] [--dashboard] [--export FORMAT]

Features:
- Real-time performance metrics collection
- Memory and CPU monitoring
- Operation timing tracking
- Cache performance analysis
- Bottleneck identification
- HTML dashboard generation
- Metrics export (CSV, JSON, Parquet)
- 1s vs regular mode comparison
"""

import sys
import os
import time
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.performance_monitor import (
        PerformanceMonitorIntegration, 
        initialize_global_monitor,
        cleanup_global_monitor,
        time_operation
    )
    from utils.performance_dashboard import PerformanceDashboard
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Error importing performance monitoring: {e}")
    MONITORING_AVAILABLE = False
    sys.exit(1)


class PerformanceMonitoringService:
    """
    Performance Monitoring Service for SqueezeFlow Trader
    
    Provides comprehensive real-time performance monitoring with
    visualization and reporting capabilities.
    """
    
    def __init__(self, duration_minutes: int = 10):
        self.duration_minutes = duration_minutes
        self.start_time = None
        self.monitor = None
        self.dashboard = None
        self.logger = logging.getLogger(__name__)
        
    async def run_monitoring_session(self, generate_dashboard: bool = True, 
                                   export_format: str = None) -> Dict[str, Any]:
        """
        Run a complete monitoring session
        
        Args:
            generate_dashboard: Whether to generate HTML dashboard
            export_format: Export format ('csv', 'json', 'parquet', None)
            
        Returns:
            Session results with paths to generated files
        """
        session_results = {
            'start_time': datetime.now().isoformat(),
            'duration_minutes': self.duration_minutes,
            'files_generated': [],
            'metrics_collected': 0,
            'bottlenecks_detected': 0,
            'recommendations': []
        }
        
        try:
            # Initialize monitoring
            self.logger.info(f"🚀 Starting {self.duration_minutes}-minute performance monitoring session...")
            self.monitor = PerformanceMonitorIntegration(enable_monitoring=True)
            self.dashboard = PerformanceDashboard(self.monitor)
            
            self.start_time = time.time()
            
            # Run monitoring with simulated load
            await self._run_simulated_workload()
            
            # Collect final metrics
            session_results.update(await self._collect_session_metrics())
            
            # Generate dashboard
            if generate_dashboard:
                dashboard_path = await self._generate_dashboard()
                if dashboard_path:
                    session_results['files_generated'].append({
                        'type': 'dashboard',
                        'path': dashboard_path,
                        'description': 'Interactive HTML performance dashboard'
                    })
            
            # Export metrics
            if export_format:
                export_path = await self._export_metrics(export_format)
                if export_path:
                    session_results['files_generated'].append({
                        'type': 'metrics_export',
                        'path': export_path,
                        'description': f'Metrics data in {export_format.upper()} format'
                    })
            
            # Generate text report
            report_path = await self._generate_text_report()
            if report_path:
                session_results['files_generated'].append({
                    'type': 'text_report',
                    'path': report_path,
                    'description': 'Comprehensive performance text report'
                })
                
            self.logger.info("✅ Monitoring session completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring session failed: {e}")
            session_results['error'] = str(e)
            
        finally:
            if self.monitor:
                self.monitor.cleanup()
                
        session_results['end_time'] = datetime.now().isoformat()
        return session_results
    
    async def _run_simulated_workload(self):
        """
        Run simulated workload to generate performance metrics
        This simulates typical SqueezeFlow operations
        """
        self.logger.info("🔧 Running simulated SqueezeFlow workload...")
        
        end_time = time.time() + (self.duration_minutes * 60)
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            
            # Simulate strategy processing phases
            await self._simulate_strategy_phases(iteration)
            
            # Simulate data loading
            await self._simulate_data_operations(iteration)
            
            # Simulate CVD calculations
            await self._simulate_cvd_calculations(iteration)
            
            # Simulate window processing
            await self._simulate_window_processing(iteration)
            
            # Progress logging
            if iteration % 50 == 0:
                elapsed = time.time() - self.start_time
                remaining = (self.duration_minutes * 60) - elapsed
                self.logger.info(f"📊 Iteration {iteration:,} | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
            
        self.logger.info(f"🎯 Completed {iteration:,} iterations of simulated workload")
    
    async def _simulate_strategy_phases(self, iteration: int):
        """Simulate SqueezeFlow strategy phase processing"""
        
        # Phase 1: Context Assessment
        with self.monitor.timer("phase1_context", {"iteration": iteration, "symbol": "BTCUSDT"}):
            await asyncio.sleep(0.005 + (iteration % 10) * 0.001)  # Variable timing
            
        # Phase 2: Divergence Detection
        with self.monitor.timer("phase2_divergence", {"iteration": iteration, "symbol": "BTCUSDT"}):
            await asyncio.sleep(0.008 + (iteration % 15) * 0.0005)
            
        # Phase 3: Reset Detection
        with self.monitor.timer("phase3_reset", {"iteration": iteration, "symbol": "BTCUSDT"}):
            await asyncio.sleep(0.003 + (iteration % 8) * 0.0002)
            
        # Phase 4: Scoring
        with self.monitor.timer("phase4_scoring", {"iteration": iteration, "symbol": "BTCUSDT"}):
            await asyncio.sleep(0.010 + (iteration % 12) * 0.0008)
            
        # Occasionally simulate Phase 5 (exits)
        if iteration % 20 == 0:
            with self.monitor.timer("phase5_exits", {"iteration": iteration, "position_id": f"pos_{iteration}"}):
                await asyncio.sleep(0.006 + (iteration % 5) * 0.0003)
    
    async def _simulate_data_operations(self, iteration: int):
        """Simulate data loading and processing operations"""
        
        # Data loading
        with self.monitor.timer("data_loading", {"iteration": iteration, "mode": "regular"}):
            data_size = 1000 + (iteration % 500)  # Variable data size
            await asyncio.sleep(0.020 + (data_size / 100000))  # Size-dependent timing
            self.monitor.record_data_loading(20 + (data_size / 50), data_size)
            
        # Occasionally simulate 1s mode data loading
        if iteration % 30 == 0:
            with self.monitor.timer("data_loading_1s", {"iteration": iteration, "mode": "1s"}):
                data_size = 3000 + (iteration % 1000)  # Larger for 1s mode
                await asyncio.sleep(0.050 + (data_size / 60000))  # Slower for 1s mode
                self.monitor.record_data_loading(50 + (data_size / 30), data_size)
    
    async def _simulate_cvd_calculations(self, iteration: int):
        """Simulate CVD calculation operations"""
        
        # Spot CVD calculation
        with self.monitor.timer("spot_cvd_calculation", {"iteration": iteration}):
            data_points = 500 + (iteration % 200)
            await asyncio.sleep(0.012 + (data_points / 50000))
            
        # Futures CVD calculation  
        with self.monitor.timer("futures_cvd_calculation", {"iteration": iteration}):
            data_points = 450 + (iteration % 250)
            await asyncio.sleep(0.015 + (data_points / 45000))
            
        # CVD divergence calculation
        with self.monitor.timer("cvd_divergence_calculation", {"iteration": iteration}):
            total_points = 950 + (iteration % 400)
            await asyncio.sleep(0.008 + (total_points / 70000))
            self.monitor.record_cvd_calculation(8 + (total_points / 8750), total_points)
            
        # Simulate cache operations
        if iteration % 3 == 0:  # 66% hit rate
            self.monitor.record_cache_hit()
        else:
            self.monitor.record_cache_miss()
    
    async def _simulate_window_processing(self, iteration: int):
        """Simulate window processing operations"""
        
        # Regular window processing
        window_time = 15 + (iteration % 20) + ((iteration % 100) / 10)  # Variable timing with trends
        await asyncio.sleep(window_time / 1000)  # Convert ms to seconds
        self.monitor.record_window_processing_time(window_time)
        
        # Occasionally simulate slow window (bottleneck)
        if iteration % 100 == 99:  # Every 100th iteration
            slow_window_time = 1500 + (iteration % 500)  # Slow window
            await asyncio.sleep(slow_window_time / 1000)
            self.monitor.record_window_processing_time(slow_window_time)
            
        # Strategy processing
        with self.monitor.timer("strategy_processing", {
            "iteration": iteration, 
            "mode": "1s" if iteration % 25 == 0 else "regular"
        }):
            strategy_time = 25 + (iteration % 30)
            await asyncio.sleep(strategy_time / 1000)
    
    async def _collect_session_metrics(self) -> Dict[str, Any]:
        """Collect final session metrics"""
        
        # Generate performance report
        report = self.monitor.analyzer.generate_performance_report()
        
        # Count metrics and bottlenecks
        metrics_collected = report.get('summary', {}).get('total_measurements', 0)
        bottlenecks = report.get('bottlenecks', [])
        recommendations = report.get('recommendations', [])
        
        return {
            'metrics_collected': metrics_collected,
            'bottlenecks_detected': len(bottlenecks),
            'recommendations': recommendations[:5],  # Top 5 recommendations
            'system_performance': report.get('system_performance', {}),
            'operation_performance': len(report.get('operation_performance', {}))
        }
    
    async def _generate_dashboard(self) -> str:
        """Generate HTML dashboard"""
        try:
            self.logger.info("📊 Generating HTML performance dashboard...")
            dashboard_path = self.dashboard.generate_dashboard()
            self.logger.info(f"✅ Dashboard generated: {dashboard_path}")
            return dashboard_path
        except Exception as e:
            self.logger.error(f"❌ Dashboard generation failed: {e}")
            return None
    
    async def _export_metrics(self, format: str) -> str:
        """Export metrics in specified format"""
        try:
            self.logger.info(f"💾 Exporting metrics in {format.upper()} format...")
            
            export_dir = Path("data/performance_exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            export_path = str(export_dir / f"performance_metrics_{timestamp}.{format}")
            
            self.monitor.export_metrics(export_path, format)
            self.logger.info(f"✅ Metrics exported: {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"❌ Metrics export failed: {e}")
            return None
    
    async def _generate_text_report(self) -> str:
        """Generate comprehensive text report"""
        try:
            self.logger.info("📄 Generating text performance report...")
            report_path = self.dashboard.generate_simple_report()
            self.logger.info(f"✅ Report generated: {report_path}")
            return report_path
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return None


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'performance_monitoring.log')
        ]
    )


async def main():
    """Main entry point"""
    
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        description='SqueezeFlow Comprehensive Performance Monitoring Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/performance_monitor_comprehensive.py                    # 10-minute monitoring with dashboard
  python scripts/performance_monitor_comprehensive.py --duration 5      # 5-minute monitoring
  python scripts/performance_monitor_comprehensive.py --export csv      # Export metrics to CSV
  python scripts/performance_monitor_comprehensive.py --no-dashboard    # Skip dashboard generation
        """
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=10,
        help='Monitoring duration in minutes (default: 10)'
    )
    
    parser.add_argument(
        '--dashboard', 
        action='store_true',
        default=True,
        help='Generate HTML dashboard (default: True)'
    )
    
    parser.add_argument(
        '--no-dashboard', 
        action='store_true',
        help='Skip dashboard generation'
    )
    
    parser.add_argument(
        '--export',
        choices=['csv', 'json', 'parquet'],
        help='Export metrics in specified format'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/performance_monitoring',
        help='Output directory for generated files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.duration <= 0:
        logger.error("Duration must be positive")
        return 1
        
    generate_dashboard = args.dashboard and not args.no_dashboard
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🚀 SQUEEZEFLOW COMPREHENSIVE PERFORMANCE MONITORING")
    logger.info("=" * 60)
    logger.info(f"Duration: {args.duration} minutes")
    logger.info(f"Dashboard: {'Yes' if generate_dashboard else 'No'}")
    logger.info(f"Export format: {args.export or 'None'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    if not MONITORING_AVAILABLE:
        logger.error("❌ Performance monitoring not available")
        return 1
        
    try:
        # Initialize monitoring service
        service = PerformanceMonitoringService(args.duration)
        
        # Run monitoring session
        session_results = await service.run_monitoring_session(
            generate_dashboard=generate_dashboard,
            export_format=args.export
        )
        
        # Display results
        logger.info("=" * 60)
        logger.info("📊 MONITORING SESSION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Duration: {args.duration} minutes")
        logger.info(f"Metrics collected: {session_results.get('metrics_collected', 0):,}")
        logger.info(f"Bottlenecks detected: {session_results.get('bottlenecks_detected', 0)}")
        logger.info(f"Files generated: {len(session_results.get('files_generated', []))}")
        
        # List generated files
        for file_info in session_results.get('files_generated', []):
            logger.info(f"  📄 {file_info['description']}: {file_info['path']}")
            
        # Show top recommendations
        recommendations = session_results.get('recommendations', [])
        if recommendations:
            logger.info("")
            logger.info("💡 Top Performance Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                logger.info(f"  {i}. {rec}")
                
        logger.info("")
        logger.info("✅ Performance monitoring completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Monitoring interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}")
        logger.exception("Full error details:")
        return 1


if __name__ == "__main__":
    try:
        # Run the monitoring service
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)