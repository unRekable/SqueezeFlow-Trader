#!/usr/bin/env python3
"""
Health Monitor Service - Comprehensive health monitoring for SqueezeFlow services
Provides HTTP health endpoints, dependency monitoring, and Docker-compatible health checks

Features:
- HTTP health endpoints for Docker health checks
- Comprehensive service status monitoring
- Dependency health validation (Redis, InfluxDB)
- Performance metrics collection
- Real-time health reporting
- Alerting for critical issues
"""

import asyncio
import json
import time
import psutil
import redis
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from influxdb import InfluxDBClient
from concurrent.futures import ThreadPoolExecutor

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. HTTP endpoints disabled.")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available. HTTP-based service checks disabled.")

from services.config.unified_config import ConfigManager


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ServiceType(Enum):
    """Service types for monitoring"""
    STRATEGY_RUNNER = "strategy_runner"
    REDIS = "redis"
    INFLUXDB = "influxdb"
    SYSTEM = "system"
    DOCKER = "docker"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_free_gb: float
    load_average: List[float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class HealthMonitor:
    """Comprehensive health monitoring service"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize Health Monitor
        
        Args:
            config_manager: Configuration manager (creates default if None)
        """
        # Initialize configuration
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self._setup_logging()
        
        # Health check registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_history: Dict[str, List[HealthCheck]] = {}
        self.max_history_size = 100
        
        # Monitoring state
        self.is_monitoring = False
        self.last_system_metrics: Optional[SystemMetrics] = None
        self.monitoring_interval = getattr(self.config, 'health_check_interval', 30)
        
        # Service registry
        self.monitored_services: Dict[str, Dict[str, Any]] = {}
        
        # Restart tracking (avoid restart loops)
        self.restart_history: Dict[str, List[datetime]] = {}
        self.max_restarts_per_hour = 3
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 5000.0,
            'error_rate_percent': 10.0
        }
        
        # Connection pools
        self._redis_client: Optional[redis.Redis] = None
        self._influx_client: Optional[InfluxDBClient] = None
        
        # HTTP server
        self.app: Optional[FastAPI] = None
        self.server_task: Optional[asyncio.Task] = None
        
        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='health_monitor')
        
        # Performance tracking
        self.performance_metrics = {
            'total_checks_performed': 0,
            'failed_checks': 0,
            'average_check_time_ms': 0.0,
            'alerts_triggered': 0,
            'last_alert_time': None
        }
        
        self.logger.info("Health Monitor initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('health_monitor')
        self.logger.info(f"Health monitoring logging configured at level: {self.config.log_level}")
    
    @property
    def redis_client(self) -> redis.Redis:
        """Lazy loading Redis client for health checks"""
        if self._redis_client is None:
            redis_config = self.config_manager.get_redis_config()
            redis_config.update({
                'socket_timeout': 2,  # Short timeout for health checks
                'socket_connect_timeout': 2,
                'retry_on_timeout': False  # Fail fast for health checks
            })
            self._redis_client = redis.Redis(**redis_config)
        return self._redis_client
    
    @property
    def influx_client(self) -> InfluxDBClient:
        """Lazy loading InfluxDB client for health checks"""
        if self._influx_client is None:
            influx_config = self.config_manager.get_influx_config()
            self._influx_client = InfluxDBClient(**influx_config)
        return self._influx_client
    
    async def start_monitoring(self):
        """Start health monitoring service"""
        self.logger.info("Starting Health Monitor service...")
        
        # Initialize HTTP server if FastAPI is available
        if FASTAPI_AVAILABLE:
            await self._setup_http_server()
        
        # Start monitoring loop
        self.is_monitoring = True
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Register core services
        await self._register_core_services()
        
        self.logger.info(f"Health Monitor started - checking every {self.monitoring_interval}s")
        
        try:
            await monitoring_task
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop health monitoring service"""
        self.logger.info("Stopping Health Monitor service...")
        
        self.is_monitoring = False
        
        # Stop HTTP server
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Close connections
        if self._redis_client:
            self._redis_client.close()
        if self._influx_client:
            self._influx_client.close()
        
        self.logger.info("Health Monitor service stopped")
    
    async def _setup_http_server(self):
        """Setup FastAPI HTTP server for health endpoints"""
        if not FASTAPI_AVAILABLE:
            return
        
        self.app = FastAPI(
            title="SqueezeFlow Health Monitor",
            description="Health monitoring and status endpoints",
            version="1.0.0"
        )
        
        # Health endpoints
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint"""
            overall_status = self.get_overall_health_status()
            
            if overall_status['status'] in ['healthy', 'degraded']:
                return JSONResponse(content=overall_status, status_code=200)
            else:
                return JSONResponse(content=overall_status, status_code=503)
        
        @self.app.get("/health/detailed")
        async def detailed_health():
            """Detailed health information"""
            return self.get_comprehensive_health_report()
        
        @self.app.get("/health/service/{service_name}")
        async def service_health(service_name: str):
            """Health check for specific service"""
            if service_name not in self.health_checks:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            
            health_check = self.health_checks[service_name]
            return JSONResponse(content=health_check.to_dict())
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-style metrics endpoint"""
            return self._generate_prometheus_metrics()
        
        @self.app.get("/status")
        async def status():
            """System status summary"""
            return {
                'service': 'health_monitor',
                'status': 'running' if self.is_monitoring else 'stopped',
                'uptime_seconds': time.time() - self._start_time if hasattr(self, '_start_time') else 0,
                'monitoring_interval': self.monitoring_interval,
                'monitored_services': len(self.monitored_services),
                'total_checks': self.performance_metrics['total_checks_performed'],
                'timestamp': datetime.now().isoformat()
            }
        
        # Start server
        port = getattr(self.config, 'health_monitor_port', 8080)
        self.server_task = asyncio.create_task(
            self._run_uvicorn_server(port)
        )
        
        self.logger.info(f"Health Monitor HTTP server starting on port {port}")
    
    async def _run_uvicorn_server(self, port: int):
        """Run uvicorn server in async context"""
        try:
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=port,
                log_level="warning",  # Reduce uvicorn logging
                access_log=False
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            self.logger.error(f"HTTP server error: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        self._start_time = time.time()
        
        while self.is_monitoring:
            try:
                loop_start = time.time()
                
                # Perform all health checks
                await self._perform_all_health_checks()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                # Update performance metrics
                loop_time = (time.time() - loop_start) * 1000  # ms
                self._update_performance_metrics(loop_time)
                
                # Log periodic status
                if self.performance_metrics['total_checks_performed'] % 10 == 0:
                    self._log_monitoring_status()
                
                # Sleep until next check
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _register_core_services(self):
        """Register core services for monitoring"""
        
        # Strategy Runner service
        self.monitored_services['strategy_runner'] = {
            'type': ServiceType.STRATEGY_RUNNER,
            'check_function': self._check_strategy_runner_health,
            'critical': True,
            'timeout_seconds': 5,
            'auto_restart': True,
            'restart_command': 'docker-compose restart strategy-runner'
        }
        
        # Redis service
        self.monitored_services['redis'] = {
            'type': ServiceType.REDIS,
            'check_function': self._check_redis_health,
            'critical': True,
            'timeout_seconds': 2,
            'auto_restart': False
        }
        
        # InfluxDB service
        self.monitored_services['influxdb'] = {
            'type': ServiceType.INFLUXDB,
            'check_function': self._check_influxdb_health,
            'critical': True,
            'timeout_seconds': 3,
            'auto_restart': False
        }
        
        # Aggr-server service (data collection)
        self.monitored_services['aggr_server'] = {
            'type': ServiceType.DOCKER,
            'check_function': self._check_aggr_server_health,
            'critical': True,
            'timeout_seconds': 5,
            'auto_restart': True,
            'restart_command': 'docker-compose restart aggr-server',
            'data_gap_threshold_minutes': 5
        }
        
        # System resources
        self.monitored_services['system'] = {
            'type': ServiceType.SYSTEM,
            'check_function': self._check_system_health,
            'critical': True,
            'timeout_seconds': 1,
            'auto_restart': False
        }
        
        self.logger.info(f"Registered {len(self.monitored_services)} services for monitoring")
    
    async def _perform_all_health_checks(self):
        """Perform health checks for all registered services"""
        
        check_tasks = []
        
        for service_name, service_config in self.monitored_services.items():
            task = asyncio.create_task(
                self._perform_service_health_check(service_name, service_config)
            )
            check_tasks.append(task)
        
        # Wait for all checks to complete
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        for service_name, result in zip(self.monitored_services.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed for {service_name}: {result}")
                self._record_failed_health_check(service_name, str(result))
    
    async def _perform_service_health_check(self, service_name: str, service_config: Dict[str, Any]):
        """Perform health check for a specific service"""
        
        start_time = time.time()
        
        try:
            # Run check function with timeout
            check_function = service_config['check_function']
            timeout = service_config.get('timeout_seconds', 5)
            
            health_check = await asyncio.wait_for(
                check_function(),
                timeout=timeout
            )
            
            # Record successful check
            self._record_health_check(service_name, health_check)
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            health_check = HealthCheck(
                name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check timeout after {timeout}s",
                details={'error': 'timeout', 'timeout_seconds': timeout},
                timestamp=datetime.now()
            )
            self._record_health_check(service_name, health_check)
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health_check = HealthCheck(
                name=service_name,
                status=HealthStatus.CRITICAL,
                response_time_ms=response_time,
                message=f"Health check failed: {str(e)}",
                details={'error': str(e), 'error_type': type(e).__name__},
                timestamp=datetime.now()
            )
            self._record_health_check(service_name, health_check)
    
    async def _check_strategy_runner_health(self) -> HealthCheck:
        """Check Strategy Runner service health"""
        start_time = time.time()
        
        try:
            # Try to import and check if service is responsive
            # This is a simplified check - in production you'd check actual service
            from services.strategy_runner import StrategyRunner
            
            # Check if Redis key exists (strategy runner should be publishing)
            redis_key = f"{self.config.redis_key_prefix}:stats:signals_published"
            exists = self.redis_client.exists(redis_key)
            
            response_time = (time.time() - start_time) * 1000
            
            if exists:
                status = HealthStatus.HEALTHY
                message = "Strategy Runner service is active"
                details = {'redis_key_exists': True}
            else:
                status = HealthStatus.DEGRADED
                message = "Strategy Runner service may not be publishing signals"
                details = {'redis_key_exists': False}
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.CRITICAL
            message = f"Cannot check Strategy Runner: {str(e)}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name='strategy_runner',
            status=status,
            response_time_ms=response_time,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    async def _check_redis_health(self) -> HealthCheck:
        """Check Redis service health"""
        start_time = time.time()
        
        try:
            # Test basic operations
            ping_result = self.redis_client.ping()
            info = self.redis_client.info()
            
            response_time = (time.time() - start_time) * 1000
            
            memory_usage_mb = info.get('used_memory', 0) / 1024 / 1024
            connected_clients = info.get('connected_clients', 0)
            
            # Determine status based on metrics
            if response_time > 1000:  # 1 second
                status = HealthStatus.DEGRADED
                message = f"Redis responding slowly ({response_time:.0f}ms)"
            elif memory_usage_mb > 1000:  # 1GB
                status = HealthStatus.DEGRADED
                message = f"Redis memory usage high ({memory_usage_mb:.0f}MB)"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis is healthy"
            
            details = {
                'ping_result': ping_result,
                'memory_usage_mb': round(memory_usage_mb, 2),
                'connected_clients': connected_clients,
                'version': info.get('redis_version', 'unknown')
            }
            
        except redis.ConnectionError as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.CRITICAL
            message = f"Redis connection failed: {str(e)}"
            details = {'error': str(e), 'error_type': 'connection_error'}
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            message = f"Redis check failed: {str(e)}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name='redis',
            status=status,
            response_time_ms=response_time,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    async def _check_influxdb_health(self) -> HealthCheck:
        """Check InfluxDB service health with data flow monitoring"""
        start_time = time.time()
        
        try:
            # Test connection and basic query
            ping_result = self.influx_client.ping()
            databases = self.influx_client.get_list_database()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if our database exists
            db_exists = any(db['name'] == self.config.influx_database for db in databases)
            
            # Check data flow - count recent data points
            data_flow_healthy = False
            recent_data_count = 0
            data_gap_minutes = 0
            
            try:
                # Query for data in last 5 minutes
                query = 'SELECT COUNT(*) FROM "aggr_1m"."trades_1m" WHERE time > now() - 5m'
                result = list(self.influx_client.query(query).get_points())
                if result:
                    recent_data_count = result[0].get('count', 0)
                    data_flow_healthy = recent_data_count > 100  # Expect at least 100 points in 5 min
                
                # Check for data gaps
                query_latest = 'SELECT * FROM "aggr_1m"."trades_1m" ORDER BY time DESC LIMIT 1'
                result_latest = list(self.influx_client.query(query_latest).get_points())
                if result_latest:
                    from dateutil import parser
                    latest_time = parser.parse(result_latest[0]['time'])
                    gap = datetime.now(latest_time.tzinfo) - latest_time
                    data_gap_minutes = gap.total_seconds() / 60
                    
            except Exception as e:
                self.logger.warning(f"Could not check data flow: {e}")
            
            # Determine status based on all factors
            if not db_exists:
                status = HealthStatus.CRITICAL
                message = f"Database '{self.config.influx_database}' not found"
            elif data_gap_minutes > 5:
                status = HealthStatus.UNHEALTHY
                message = f"Data gap detected: {data_gap_minutes:.1f} minutes since last write"
            elif not data_flow_healthy:
                status = HealthStatus.DEGRADED
                message = f"Low data flow: only {recent_data_count} points in last 5 min"
            elif response_time > 2000:  # 2 seconds
                status = HealthStatus.DEGRADED
                message = f"InfluxDB responding slowly ({response_time:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"InfluxDB healthy, {recent_data_count} recent points"
            
            details = {
                'ping_result': ping_result,
                'database_exists': db_exists,
                'total_databases': len(databases),
                'target_database': self.config.influx_database,
                'recent_data_count': recent_data_count,
                'data_flow_healthy': data_flow_healthy,
                'data_gap_minutes': round(data_gap_minutes, 2)
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.CRITICAL
            message = f"InfluxDB check failed: {str(e)}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name='influxdb',
            status=status,
            response_time_ms=response_time,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    async def _check_aggr_server_health(self) -> HealthCheck:
        """Check aggr-server health using HTTP endpoint and data writing status"""
        start_time = time.time()
        
        # Check if aggr-server is responding via HTTP
        service_responding = False
        http_status = None
        http_error = None
        
        if AIOHTTP_AVAILABLE:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    # Try to connect to aggr-server on port 3000
                    # Most aggr-server implementations have a health or status endpoint
                    for endpoint in ['/health', '/status', '/', '/api/status']:
                        try:
                            async with session.get(f'http://aggr-server:3000{endpoint}') as response:
                                http_status = response.status
                                if response.status < 500:  # Accept 2xx, 3xx, 4xx as "service responding"
                                    service_responding = True
                                    break
                        except Exception:
                            continue
                    
                    # If no endpoint worked, try a simple TCP connection test
                    if not service_responding:
                        try:
                            async with session.get('http://aggr-server:3000') as response:
                                http_status = response.status
                                service_responding = True  # Any response means service is up
                        except Exception as e:
                            http_error = str(e)
                            
            except Exception as e:
                http_error = str(e)
                self.logger.debug(f"HTTP check failed, falling back to data flow monitoring: {e}")
        else:
            http_error = "aiohttp not available - using data flow monitoring only"
            self.logger.debug("aiohttp not available, relying on InfluxDB data flow monitoring for aggr-server health")
        
        try:
            
            # Check data flow by querying InfluxDB (primary indicator)
            data_flow_healthy = False
            recent_data_count = 0
            data_gap_minutes = 0
            last_write_time = None
            
            try:
                # Query for recent data writes
                query = 'SELECT COUNT(*) FROM "aggr_1m"."trades_1m" WHERE time > now() - 5m'
                result = list(self.influx_client.query(query).get_points())
                if result:
                    recent_data_count = result[0].get('count', 0)
                    data_flow_healthy = recent_data_count > 100
                
                # Check latest write time
                query_latest = 'SELECT * FROM "aggr_1m"."trades_1m" ORDER BY time DESC LIMIT 1'
                result_latest = list(self.influx_client.query(query_latest).get_points())
                if result_latest:
                    from dateutil import parser
                    latest_time = parser.parse(result_latest[0]['time'])
                    last_write_time = latest_time.isoformat()
                    gap = datetime.now(latest_time.tzinfo) - latest_time
                    data_gap_minutes = gap.total_seconds() / 60
                    
            except Exception as e:
                self.logger.warning(f"Could not check aggr-server data flow: {e}")
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine health status - prioritize data flow over HTTP response
            if data_gap_minutes > 10:
                status = HealthStatus.CRITICAL
                message = f"Aggr-server not writing data for {data_gap_minutes:.1f} minutes"
            elif data_gap_minutes > 5:
                status = HealthStatus.UNHEALTHY
                message = f"Data gap detected: {data_gap_minutes:.1f} minutes since last write"
            elif not data_flow_healthy and AIOHTTP_AVAILABLE and not service_responding:
                status = HealthStatus.CRITICAL
                message = "Aggr-server not responding and no data flow detected"
            elif not data_flow_healthy:
                status = HealthStatus.DEGRADED
                message = f"Low data flow: only {recent_data_count} points in last 5 min"
            elif AIOHTTP_AVAILABLE and not service_responding:
                status = HealthStatus.DEGRADED
                message = f"HTTP endpoint not responding, but data flow OK ({recent_data_count} points/5min)"
            else:
                status = HealthStatus.HEALTHY
                if AIOHTTP_AVAILABLE:
                    message = f"Aggr-server healthy, writing {recent_data_count} points/5min"
                else:
                    message = f"Data flow healthy: {recent_data_count} points/5min (HTTP check unavailable)"
            
            details = {
                'service_responding': service_responding,
                'http_status': http_status,
                'http_error': http_error,
                'http_check_available': AIOHTTP_AVAILABLE,
                'recent_data_count': recent_data_count,
                'data_flow_healthy': data_flow_healthy,
                'data_gap_minutes': round(data_gap_minutes, 2),
                'last_write_time': last_write_time
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.CRITICAL
            message = f"Aggr-server check failed: {str(e)}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name='aggr_server',
            status=status,
            response_time_ms=response_time,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    async def _check_system_health(self) -> HealthCheck:
        """Check system resource health"""
        start_time = time.time()
        
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on thresholds
            if (cpu_percent > self.alert_thresholds['cpu_percent'] or
                memory.percent > self.alert_thresholds['memory_percent'] or
                disk.percent > self.alert_thresholds['disk_percent']):
                status = HealthStatus.DEGRADED
                message = "System resources under pressure"
            elif (cpu_percent > 95 or memory.percent > 95 or disk.percent > 95):
                status = HealthStatus.CRITICAL
                message = "System resources critically low"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            details = {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory.percent, 1),
                'memory_used_gb': round(memory.used / 1024**3, 2),
                'memory_available_gb': round(memory.available / 1024**3, 2),
                'disk_percent': round(disk.percent, 1),
                'disk_free_gb': round(disk.free / 1024**3, 2),
                'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            message = f"System check failed: {str(e)}"
            details = {'error': str(e)}
        
        return HealthCheck(
            name='system',
            status=status,
            response_time_ms=response_time,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    async def _collect_system_metrics(self):
        """Collect detailed system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            self.last_system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024**2,
                memory_available_mb=memory.available / 1024**2,
                disk_percent=disk.percent,
                disk_free_gb=disk.free / 1024**3,
                load_average=load_avg,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _record_health_check(self, service_name: str, health_check: HealthCheck):
        """Record health check result"""
        
        self.health_checks[service_name] = health_check
        
        # Add to history
        if service_name not in self.check_history:
            self.check_history[service_name] = []
        
        self.check_history[service_name].append(health_check)
        
        # Limit history size
        if len(self.check_history[service_name]) > self.max_history_size:
            self.check_history[service_name] = self.check_history[service_name][-self.max_history_size:]
        
        # Update performance metrics
        self.performance_metrics['total_checks_performed'] += 1
        
        if health_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            self.performance_metrics['failed_checks'] += 1
    
    def _record_failed_health_check(self, service_name: str, error_message: str):
        """Record a failed health check"""
        
        health_check = HealthCheck(
            name=service_name,
            status=HealthStatus.CRITICAL,
            response_time_ms=0,
            message=f"Health check execution failed: {error_message}",
            details={'error': error_message},
            timestamp=datetime.now()
        )
        
        self._record_health_check(service_name, health_check)
    
    async def _check_alert_conditions(self):
        """Check if any alert conditions are met and handle auto-restart"""
        
        alerts_triggered = 0
        
        for service_name, health_check in self.health_checks.items():
            service_config = self.monitored_services.get(service_name, {})
            is_critical = service_config.get('critical', False)
            auto_restart = service_config.get('auto_restart', False)
            
            # Check for auto-restart conditions
            if auto_restart and health_check.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                # Check if we should restart (avoid restart loops)
                should_restart = await self._should_restart_service(service_name, health_check)
                if should_restart:
                    await self._restart_service(service_name, service_config, health_check)
                    alerts_triggered += 1
            
            # Alert on critical service failures
            if is_critical and health_check.status == HealthStatus.CRITICAL:
                await self._trigger_alert(f"CRITICAL: {service_name} service is down", health_check)
                alerts_triggered += 1
            
            # Alert on high response times
            if health_check.response_time_ms > self.alert_thresholds['response_time_ms']:
                await self._trigger_alert(f"HIGH LATENCY: {service_name} response time is {health_check.response_time_ms:.0f}ms", health_check)
                alerts_triggered += 1
        
        # System resource alerts
        if self.last_system_metrics:
            metrics = self.last_system_metrics
            
            if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                await self._trigger_alert(f"HIGH CPU: {metrics.cpu_percent:.1f}%", None)
                alerts_triggered += 1
            
            if metrics.memory_percent > self.alert_thresholds['memory_percent']:
                await self._trigger_alert(f"HIGH MEMORY: {metrics.memory_percent:.1f}%", None)
                alerts_triggered += 1
            
            if metrics.disk_percent > self.alert_thresholds['disk_percent']:
                await self._trigger_alert(f"HIGH DISK USAGE: {metrics.disk_percent:.1f}%", None)
                alerts_triggered += 1
        
        if alerts_triggered > 0:
            self.performance_metrics['alerts_triggered'] += alerts_triggered
            self.performance_metrics['last_alert_time'] = datetime.now().isoformat()
    
    async def _trigger_alert(self, message: str, health_check: Optional[HealthCheck]):
        """Trigger an alert"""
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': 'critical' if 'CRITICAL' in message else 'warning',
            'health_check': health_check.to_dict() if health_check else None
        }
        
        # Log alert
        if alert_data['severity'] == 'critical':
            self.logger.critical(f"ALERT: {message}")
        else:
            self.logger.warning(f"ALERT: {message}")
        
        # In production, you would send alerts to:
        # - Slack/Discord webhooks
        # - Email notifications
        # - PagerDuty/similar
        # - Push to monitoring systems
        
        # For now, just publish to Redis for other services to consume
        try:
            alert_channel = f"{self.config.redis_key_prefix}:alerts"
            self.redis_client.publish(alert_channel, json.dumps(alert_data))
        except Exception as e:
            self.logger.error(f"Failed to publish alert: {e}")
    
    def _update_performance_metrics(self, loop_time_ms: float):
        """Update performance metrics"""
        
        # Update average check time
        total_checks = self.performance_metrics['total_checks_performed']
        current_avg = self.performance_metrics['average_check_time_ms']
        
        if total_checks > 0:
            self.performance_metrics['average_check_time_ms'] = (
                (current_avg * (total_checks - 1) + loop_time_ms) / total_checks
            )
    
    def _log_monitoring_status(self):
        """Log periodic monitoring status"""
        
        total_checks = self.performance_metrics['total_checks_performed']
        failed_checks = self.performance_metrics['failed_checks']
        success_rate = ((total_checks - failed_checks) / max(1, total_checks)) * 100
        avg_time = self.performance_metrics['average_check_time_ms']
        
        healthy_services = sum(1 for hc in self.health_checks.values() if hc.status == HealthStatus.HEALTHY)
        total_services = len(self.health_checks)
        
        self.logger.info(
            f"Health Monitor Status: {healthy_services}/{total_services} services healthy, "
            f"Success rate: {success_rate:.1f}%, Avg check time: {avg_time:.1f}ms, "
            f"Total checks: {total_checks}"
        )
    
    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        if not self.health_checks:
            return {
                'status': 'unknown',
                'message': 'No health checks performed yet',
                'timestamp': datetime.now().isoformat()
            }
        
        # Count statuses
        status_counts = {}
        for health_check in self.health_checks.values():
            status = health_check.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        if status_counts.get('critical', 0) > 0:
            overall_status = 'critical'
            message = f"{status_counts['critical']} critical issues detected"
        elif status_counts.get('unhealthy', 0) > 0:
            overall_status = 'unhealthy'
            message = f"{status_counts['unhealthy']} services unhealthy"
        elif status_counts.get('degraded', 0) > 0:
            overall_status = 'degraded'
            message = f"{status_counts['degraded']} services degraded"
        else:
            overall_status = 'healthy'
            message = "All services healthy"
        
        return {
            'status': overall_status,
            'message': message,
            'service_count': len(self.health_checks),
            'status_breakdown': status_counts,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        
        return {
            'overall_status': self.get_overall_health_status(),
            'services': {name: hc.to_dict() for name, hc in self.health_checks.items()},
            'system_metrics': self.last_system_metrics.to_dict() if self.last_system_metrics else None,
            'performance_metrics': self.performance_metrics.copy(),
            'alert_thresholds': self.alert_thresholds.copy(),
            'monitoring_config': {
                'interval_seconds': self.monitoring_interval,
                'monitored_services': len(self.monitored_services),
                'is_monitoring': self.is_monitoring
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_prometheus_metrics(self) -> str:
        """Generate Prometheus-style metrics"""
        
        metrics = []
        
        # Service health metrics
        for service_name, health_check in self.health_checks.items():
            status_value = {
                HealthStatus.HEALTHY: 1,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0,
                HealthStatus.CRITICAL: 0
            }.get(health_check.status, 0)
            
            metrics.append(f'squeezeflow_service_health{{service="{service_name}"}} {status_value}')
            metrics.append(f'squeezeflow_service_response_time_ms{{service="{service_name}"}} {health_check.response_time_ms}')
        
        # System metrics
        if self.last_system_metrics:
            metrics.extend([
                f'squeezeflow_cpu_percent {self.last_system_metrics.cpu_percent}',
                f'squeezeflow_memory_percent {self.last_system_metrics.memory_percent}',
                f'squeezeflow_disk_percent {self.last_system_metrics.disk_percent}',
                f'squeezeflow_memory_used_mb {self.last_system_metrics.memory_used_mb}',
                f'squeezeflow_disk_free_gb {self.last_system_metrics.disk_free_gb}'
            ])
        
        # Performance metrics
        metrics.extend([
            f'squeezeflow_health_checks_total {self.performance_metrics["total_checks_performed"]}',
            f'squeezeflow_health_checks_failed {self.performance_metrics["failed_checks"]}',
            f'squeezeflow_alerts_triggered {self.performance_metrics["alerts_triggered"]}',
            f'squeezeflow_avg_check_time_ms {self.performance_metrics["average_check_time_ms"]}'
        ])
        
        return '\n'.join(metrics) + '\n'
    
    def register_custom_service(self, service_name: str, check_function, critical: bool = False, timeout_seconds: int = 5):
        """Register a custom service for monitoring"""
        
        self.monitored_services[service_name] = {
            'type': ServiceType.STRATEGY_RUNNER,  # Default type
            'check_function': check_function,
            'critical': critical,
            'timeout_seconds': timeout_seconds
        }
        
        self.logger.info(f"Registered custom service for monitoring: {service_name}")
    
    def get_service_health_history(self, service_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get health check history for a service"""
        
        if service_name not in self.check_history:
            return []
        
        history = self.check_history[service_name][-limit:]
        return [hc.to_dict() for hc in history]
    
    async def _should_restart_service(self, service_name: str, health_check: HealthCheck) -> bool:
        """Check if we should restart a service (avoid restart loops)"""
        
        # Get restart history for this service
        if service_name not in self.restart_history:
            self.restart_history[service_name] = []
        
        # Clean up old restart entries (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.restart_history[service_name] = [
            restart_time for restart_time in self.restart_history[service_name]
            if restart_time > cutoff_time
        ]
        
        # Check if we've exceeded max restarts
        recent_restarts = len(self.restart_history[service_name])
        if recent_restarts >= self.max_restarts_per_hour:
            self.logger.warning(
                f"Service {service_name} has been restarted {recent_restarts} times in the last hour. "
                f"Skipping restart to avoid loop."
            )
            return False
        
        # Check specific conditions for aggr-server
        if service_name == 'aggr_server':
            # Only restart if data gap is significant (>10 minutes)
            data_gap = health_check.details.get('data_gap_minutes', 0)
            if data_gap < 10:
                return False  # Don't restart for small gaps
        
        return True
    
    async def _restart_service(self, service_name: str, service_config: Dict[str, Any], 
                              health_check: HealthCheck):
        """Restart a service using Docker Compose"""
        
        restart_command = service_config.get('restart_command')
        if not restart_command:
            self.logger.error(f"No restart command configured for {service_name}")
            return
        
        try:
            import subprocess
            
            self.logger.warning(f"Restarting {service_name} due to {health_check.status.value} status")
            
            # Execute restart command
            result = subprocess.run(
                restart_command.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Record successful restart
                self.restart_history[service_name].append(datetime.now())
                
                # Send alert about restart
                await self._trigger_alert(
                    f"SERVICE RESTARTED: {service_name} was automatically restarted",
                    health_check
                )
                
                self.logger.info(f"Successfully restarted {service_name}")
                
                # Wait a bit for service to come up
                await asyncio.sleep(10)
                
            else:
                self.logger.error(
                    f"Failed to restart {service_name}: {result.stderr}"
                )
                await self._trigger_alert(
                    f"RESTART FAILED: Could not restart {service_name}",
                    health_check
                )
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Restart command timed out for {service_name}")
        except Exception as e:
            self.logger.error(f"Error restarting {service_name}: {e}")


async def main():
    """Main entry point for running the health monitor service"""
    
    # Initialize service
    config_manager = ConfigManager()
    monitor = HealthMonitor(config_manager)
    
    try:
        # Start monitoring
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    except Exception as e:
        print(f"Health Monitor error: {e}")
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
    
    logger = logging.getLogger('health_monitor_main')
    logger.info("Starting SqueezeFlow Health Monitor")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.fatal(f"Failed to start Health Monitor: {e}")
        sys.exit(1)