#!/usr/bin/env python3
"""
System Monitor Service for SqueezeFlow Trader
Monitors system performance and health metrics
"""

import asyncio
import logging
import os
import sys
import time
import psutil
import docker
from datetime import datetime, timezone
from typing import Dict, List

# Add parent directory to path for Docker container
import sys
import os
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.storage.influxdb_handler import InfluxDBHandler
except ImportError:
    # Fallback for Docker environment
    from influxdb import InfluxDBClient
    class InfluxDBHandler:
        def __init__(self, host='influx', port=8086, username='', password='', database='significant_trades'):
            self.client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)
            self.database = database
        
        def write_points(self, points):
            try:
                return self.client.write_points(points)
            except Exception as e:
                print(f"InfluxDB write error: {e}")
                return False


class SystemMonitorService:
    """
    System monitoring service for SqueezeFlow Trader
    """
    
    def __init__(self):
        self.setup_logging()
        self.setup_database()
        self.setup_docker()
        self.running = False
        
        # Configuration
        self.monitoring_interval = 30  # seconds
        self.services_to_monitor = [
            'squeezeflow-influxdb',
            'squeezeflow-redis',
            'squeezeflow-aggr-server',
            'squeezeflow-oi-tracker',
            'squeezeflow-calculator',
            'squeezeflow-freqtrade'
        ]
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemMonitor')
        
    def setup_database(self):
        """Setup InfluxDB connection"""
        self.influx_handler = InfluxDBHandler(
            host=os.getenv('INFLUX_HOST', 'localhost'),
            port=int(os.getenv('INFLUX_PORT', 8086)),
            username=os.getenv('INFLUX_USER', 'squeezeflow'),
            password=os.getenv('INFLUX_PASSWORD', 'password123'),
            database=os.getenv('INFLUX_DATABASE', 'significant_trades')
        )
        
        # Test connection - don't fail on startup if InfluxDB is not ready
        try:
            if not self.influx_handler.test_connection():
                self.logger.warning("InfluxDB connection not ready, will retry during monitoring")
        except Exception as e:
            self.logger.warning(f"InfluxDB not available during startup: {e}")
            # Continue without failing - will retry during monitoring cycle
        
        self.logger.info("System monitor InfluxDB setup completed")
        
    def setup_docker(self):
        """Setup Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
            
    def get_system_metrics(self) -> Dict:
        """Get system-wide metrics"""
        metrics = {}
        
        try:
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            metrics['cpu_count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_total_gb'] = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            metrics['disk_used_gb'] = disk.used / (1024**3)
            metrics['disk_total_gb'] = disk.total / (1024**3)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics['network_sent_gb'] = net_io.bytes_sent / (1024**3)
            metrics['network_recv_gb'] = net_io.bytes_recv / (1024**3)
            
            # Load average (Unix only)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                metrics['load_avg_1m'] = load_avg[0]
                metrics['load_avg_5m'] = load_avg[1]
                metrics['load_avg_15m'] = load_avg[2]
                
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            
        return metrics
        
    def get_container_metrics(self) -> List[Dict]:
        """Get Docker container metrics"""
        container_metrics = []
        
        if not self.docker_client:
            return container_metrics
            
        try:
            for service_name in self.services_to_monitor:
                try:
                    container = self.docker_client.containers.get(service_name)
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_percent = 0
                    if 'cpu_stats' in stats and 'precpu_stats' in stats:
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                        if system_delta > 0:
                            cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                    
                    # Calculate memory usage
                    memory_usage = 0
                    memory_limit = 0
                    if 'memory_stats' in stats:
                        memory_usage = stats['memory_stats'].get('usage', 0)
                        memory_limit = stats['memory_stats'].get('limit', 0)
                    
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    # Network I/O
                    network_rx = 0
                    network_tx = 0
                    if 'networks' in stats:
                        for interface, data in stats['networks'].items():
                            network_rx += data.get('rx_bytes', 0)
                            network_tx += data.get('tx_bytes', 0)
                    
                    container_metrics.append({
                        'container_name': service_name,
                        'status': container.status,
                        'cpu_percent': cpu_percent,
                        'memory_usage_mb': memory_usage / (1024**2),
                        'memory_percent': memory_percent,
                        'network_rx_mb': network_rx / (1024**2),
                        'network_tx_mb': network_tx / (1024**2),
                        'restart_count': container.attrs['RestartCount']
                    })
                    
                except docker.errors.NotFound:
                    container_metrics.append({
                        'container_name': service_name,
                        'status': 'not_found',
                        'cpu_percent': 0,
                        'memory_usage_mb': 0,
                        'memory_percent': 0,
                        'network_rx_mb': 0,
                        'network_tx_mb': 0,
                        'restart_count': 0
                    })
                    
        except Exception as e:
            self.logger.error(f"Error getting container metrics: {e}")
            
        return container_metrics
        
    async def store_system_metrics(self, metrics: Dict):
        """Store system metrics in InfluxDB"""
        try:
            influx_points = []
            
            for metric_name, value in metrics.items():
                point = {
                    'measurement': 'system_metrics',
                    'tags': {
                        'component': 'system',
                        'metric_name': metric_name
                    },
                    'time': datetime.now(timezone.utc),
                    'fields': {
                        'value': float(value)
                    }
                }
                influx_points.append(point)
                
            self.influx_handler.client.write_points(influx_points, time_precision='s')
            
        except Exception as e:
            self.logger.error(f"Error storing system metrics: {e}")
            
    async def store_container_metrics(self, container_metrics: List[Dict]):
        """Store container metrics in InfluxDB"""
        try:
            influx_points = []
            
            for container in container_metrics:
                container_name = container['container_name']
                
                for metric_name, value in container.items():
                    if metric_name == 'container_name':
                        continue
                        
                    point = {
                        'measurement': 'container_metrics',
                        'tags': {
                            'container_name': container_name,
                            'metric_name': metric_name
                        },
                        'time': datetime.now(timezone.utc),
                        'fields': {
                            'value': float(value) if isinstance(value, (int, float)) else 0
                        }
                    }
                    influx_points.append(point)
                    
            self.influx_handler.client.write_points(influx_points, time_precision='s')
            
        except Exception as e:
            self.logger.error(f"Error storing container metrics: {e}")
            
    async def check_system_health(self) -> Dict:
        """Check overall system health"""
        health_status = {
            'status': 'healthy',
            'issues': []
        }
        
        # Get current metrics
        system_metrics = self.get_system_metrics()
        container_metrics = self.get_container_metrics()
        
        # Check system resource usage
        if system_metrics.get('cpu_percent', 0) > 90:
            health_status['issues'].append('High CPU usage')
            health_status['status'] = 'warning'
            
        if system_metrics.get('memory_percent', 0) > 90:
            health_status['issues'].append('High memory usage')
            health_status['status'] = 'warning'
            
        if system_metrics.get('disk_percent', 0) > 95:
            health_status['issues'].append('High disk usage')
            health_status['status'] = 'critical'
            
        # Check container health
        for container in container_metrics:
            if container['status'] not in ['running', 'healthy']:
                health_status['issues'].append(f"Container {container['container_name']} is {container['status']}")
                health_status['status'] = 'critical'
                
        return health_status
        
    async def monitoring_cycle(self):
        """Single monitoring cycle"""
        self.logger.info("Starting system monitoring cycle")
        start_time = time.time()
        
        try:
            # Get metrics
            system_metrics = self.get_system_metrics()
            container_metrics = self.get_container_metrics()
            
            # Store metrics
            await self.store_system_metrics(system_metrics)
            await self.store_container_metrics(container_metrics)
            
            # Check health
            health_status = await self.check_system_health()
            
            # Log health status
            if health_status['status'] != 'healthy':
                self.logger.warning(f"System health: {health_status['status']} - Issues: {health_status['issues']}")
            else:
                self.logger.info("System health: OK")
                
            # Store health status
            await self.store_system_metrics({
                'health_status': 1 if health_status['status'] == 'healthy' else 0,
                'issue_count': len(health_status['issues'])
            })
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
            
        elapsed = time.time() - start_time
        self.logger.info(f"Monitoring cycle completed in {elapsed:.2f}s")
        
    async def run(self):
        """Main monitoring loop"""
        self.logger.info("Starting System Monitor Service")
        self.running = True
        
        try:
            while self.running:
                await self.monitoring_cycle()
                
                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.running = False
            self.logger.info("System Monitor Service stopped")
            
    def stop(self):
        """Stop the service"""
        self.running = False


async def main():
    """Main function"""
    service = SystemMonitorService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())