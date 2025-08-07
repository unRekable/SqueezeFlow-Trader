#!/usr/bin/env python3
"""
Optimized Redis Client for SqueezeFlow Trader
Performance optimization for Redis operations with connection pooling and batching
"""

import redis
import redis.connection
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging


class OptimizedRedisClient:
    """High-performance Redis client with connection pooling and batching"""
    
    def __init__(self, host='localhost', port=6379, db=0, max_connections=20):
        """
        Initialize optimized Redis client
        
        Args:
            host: Redis host
            port: Redis port  
            db: Redis database number
            max_connections: Maximum connections in pool
        """
        self.logger = logging.getLogger(__name__)
        
        # Create optimized connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=max_connections,
            retry_on_timeout=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30
        )
        
        # Create Redis client with pool
        self.client = redis.Redis(
            connection_pool=self.pool,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Batch processing configuration
        self.batch_size = 50
        self.batch_timeout = 2.0  # seconds
        self._pending_batch = []
        self._batch_lock = asyncio.Lock()
        self._batch_task = None
        
        # Performance metrics
        self.metrics = {
            'operations': 0,
            'batch_operations': 0,
            'avg_response_time': 0.0,
            'errors': 0,
            'pool_usage': 0
        }
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=30,
            expected_exception=redis.RedisError
        )
        
    async def ping_async(self) -> bool:
        """Async ping with circuit breaker"""
        try:
            with self.circuit_breaker:
                start_time = time.time()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.client.ping
                )
                self._update_metrics(time.time() - start_time, success=True)
                return result
        except Exception as e:
            self.logger.error(f"Redis ping failed: {e}")
            self._update_metrics(0, success=False)
            return False
    
    async def set_with_ttl_async(self, key: str, value: str, ttl: int) -> bool:
        """Optimized async SET with TTL"""
        try:
            with self.circuit_breaker:
                start_time = time.time()
                
                # Use pipeline for atomic operations
                pipe = self.client.pipeline()
                pipe.setex(key, ttl, value)
                results = await asyncio.get_event_loop().run_in_executor(
                    None, pipe.execute
                )
                
                self._update_metrics(time.time() - start_time, success=True)
                return bool(results[0])
                
        except Exception as e:
            self.logger.error(f"Redis SET failed for key {key}: {e}")
            self._update_metrics(0, success=False)
            return False
    
    async def get_async(self, key: str) -> Optional[str]:
        """Optimized async GET"""
        try:
            with self.circuit_breaker:
                start_time = time.time()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.client.get, key
                )
                self._update_metrics(time.time() - start_time, success=True)
                return result
                
        except Exception as e:
            self.logger.error(f"Redis GET failed for key {key}: {e}")
            self._update_metrics(0, success=False)
            return None
    
    async def batch_publish_signals(self, signals: List[Dict]) -> Dict[str, bool]:
        """Batch publish multiple signals with pipeline"""
        results = {}
        
        if not signals:
            return results
        
        try:
            with self.circuit_breaker:
                start_time = time.time()
                
                # Use pipeline for batch operations
                pipe = self.client.pipeline()
                
                for signal in signals:
                    key = f"squeeze_signal:{signal['symbol']}"
                    signal_json = json.dumps(signal, default=str)
                    
                    # Batch multiple operations per signal
                    pipe.setex(key, signal.get('ttl', 300), signal_json)
                    pipe.publish('signals', signal_json)
                    
                    # Add to history with limited size
                    history_key = f"signal_history:{signal['symbol']}"
                    pipe.lpush(history_key, signal_json)
                    pipe.ltrim(history_key, 0, 99)  # Keep last 100
                    pipe.expire(history_key, 86400)  # 24 hour expiry
                
                # Execute entire batch atomically
                batch_results = await asyncio.get_event_loop().run_in_executor(
                    None, pipe.execute
                )
                
                # Process results
                for i, signal in enumerate(signals):
                    signal_id = signal.get('signal_id', f'signal_{i}')
                    # Each signal generates 5 operations, check first one (setex)
                    results[signal_id] = bool(batch_results[i * 5])
                
                batch_time = time.time() - start_time
                self._update_metrics(batch_time, success=True)
                self.metrics['batch_operations'] += len(signals)
                
                self.logger.info(
                    f"Batch published {len(signals)} signals in {batch_time:.3f}s "
                    f"({len(signals)/batch_time:.1f} signals/sec)"
                )
                
        except Exception as e:
            self.logger.error(f"Batch publish failed: {e}")
            self._update_metrics(0, success=False)
            for signal in signals:
                signal_id = signal.get('signal_id', 'unknown')
                results[signal_id] = False
        
        return results
    
    async def get_signals_by_pattern(self, pattern: str) -> List[Dict]:
        """Efficiently get multiple signals by pattern"""
        try:
            with self.circuit_breaker:
                start_time = time.time()
                
                # Get keys matching pattern
                keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.client.keys, pattern
                )
                
                if not keys:
                    return []
                
                # Batch get all values using pipeline
                pipe = self.client.pipeline()
                for key in keys:
                    pipe.get(key)
                
                values = await asyncio.get_event_loop().run_in_executor(
                    None, pipe.execute
                )
                
                # Parse JSON results
                signals = []
                for value in values:
                    if value:
                        try:
                            signal = json.loads(value)
                            signals.append(signal)
                        except json.JSONDecodeError:
                            continue
                
                self._update_metrics(time.time() - start_time, success=True)
                return signals
                
        except Exception as e:
            self.logger.error(f"Pattern get failed for {pattern}: {e}")
            self._update_metrics(0, success=False)
            return []
    
    async def cleanup_expired_keys(self, pattern: str) -> int:
        """Clean up expired keys efficiently"""
        try:
            start_time = time.time()
            
            # Use SCAN for memory-efficient key iteration
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.client.scan, cursor, pattern, 1000
                )
                
                if keys:
                    # Check TTL and delete expired keys
                    pipe = self.client.pipeline()
                    for key in keys:
                        pipe.ttl(key)
                    
                    ttls = await asyncio.get_event_loop().run_in_executor(
                        None, pipe.execute
                    )
                    
                    expired_keys = [key for key, ttl in zip(keys, ttls) if ttl <= 0]
                    
                    if expired_keys:
                        deleted = await asyncio.get_event_loop().run_in_executor(
                            None, self.client.delete, *expired_keys
                        )
                        deleted_count += deleted
                
                if cursor == 0:
                    break
            
            cleanup_time = time.time() - start_time
            self.logger.info(f"Cleaned up {deleted_count} expired keys in {cleanup_time:.2f}s")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        self.metrics['operations'] += 1
        
        if success:
            # Update rolling average response time
            current_avg = self.metrics['avg_response_time']
            operations = self.metrics['operations']
            self.metrics['avg_response_time'] = (
                (current_avg * (operations - 1) + response_time) / operations
            )
        else:
            self.metrics['errors'] += 1
        
        # Update pool usage metrics
        self.metrics['pool_usage'] = len(self.pool._available_connections)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        total_ops = self.metrics['operations']
        error_rate = (self.metrics['errors'] / total_ops * 100) if total_ops > 0 else 0
        
        return {
            'total_operations': total_ops,
            'batch_operations': self.metrics['batch_operations'],
            'avg_response_time_ms': round(self.metrics['avg_response_time'] * 1000, 2),
            'error_rate_percent': round(error_rate, 2),
            'pool_utilization': self.metrics['pool_usage'],
            'max_connections': self.pool.max_connections,
            'circuit_breaker_state': self.circuit_breaker.state
        }
    
    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        health = {
            'redis_ping': False,
            'pool_available': False,
            'circuit_breaker_ok': True,
            'response_time_ok': True,
            'error_rate_ok': True
        }
        
        # Ping test
        health['redis_ping'] = await self.ping_async()
        
        # Pool availability
        health['pool_available'] = len(self.pool._available_connections) > 0
        
        # Circuit breaker state
        health['circuit_breaker_ok'] = self.circuit_breaker.state == 'closed'
        
        # Response time check
        avg_response = self.metrics['avg_response_time'] * 1000
        health['response_time_ok'] = avg_response < 100  # < 100ms
        
        # Error rate check
        total_ops = self.metrics['operations']
        error_rate = (self.metrics['errors'] / total_ops * 100) if total_ops > 0 else 0
        health['error_rate_ok'] = error_rate < 5  # < 5%
        
        health['overall_healthy'] = all(health.values())
        
        return health
    
    async def close(self):
        """Clean shutdown"""
        try:
            if self._batch_task and not self._batch_task.done():
                self._batch_task.cancel()
            
            # Close connection pool
            self.pool.disconnect()
            self.logger.info("Redis client closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing Redis client: {e}")


class CircuitBreaker:
    """Circuit breaker pattern for Redis resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 30, 
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
    
    def __enter__(self):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise self.expected_exception("Circuit breaker is OPEN")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
        else:
            # Success - reset failure count
            self.failure_count = 0
            if self.state == 'half-open':
                self.state = 'closed'
        
        return False


# Factory function for easy integration
def create_optimized_redis_client(config: Dict) -> OptimizedRedisClient:
    """Create optimized Redis client from configuration"""
    return OptimizedRedisClient(
        host=config.get('redis_host', 'localhost'),
        port=config.get('redis_port', 6379),
        db=config.get('redis_db', 0),
        max_connections=config.get('redis_max_connections', 20)
    )