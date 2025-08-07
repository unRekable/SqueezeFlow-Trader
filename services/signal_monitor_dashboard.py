#!/usr/bin/env python3
"""
Signal Monitor Dashboard - Real-time monitoring of Redis signal publishing

This dashboard provides real-time monitoring of the enhanced Redis signal
publishing system with metrics, health status, and live signal feeds.

Updated for current SqueezeFlow system workflow (January 2025):
- Compatible with strategy_runner.py signal format
- Monitors Redis channels: squeezeflow:signals
- Tracks signal_id, base_symbol, trading pairs
- Integrates with health monitoring
"""

import asyncio
import json
import logging
import redis
import aioredis
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading

from services.config.unified_config import ConfigManager


@dataclass
class DashboardMetrics:
    """Dashboard metrics tracking"""
    signals_received: int = 0
    signals_per_minute: int = 0
    unique_symbols: int = 0
    avg_signal_score: float = 0.0
    last_signal_time: Optional[datetime] = None
    uptime_seconds: int = 0
    long_signals: int = 0
    short_signals: int = 0
    high_score_signals: int = 0  # Score >= 7
    signals_published_count: int = 0


class SignalMonitorDashboard:
    """Real-time dashboard for monitoring signal publishing system"""
    
    def __init__(self):
        self.logger = logging.getLogger('signal_monitor')
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Redis connections
        self._redis_client: Optional[redis.Redis] = None
        self._aioredis_client: Optional[aioredis.Redis] = None
        
        # Dashboard state
        self.is_running = False
        self.start_time = datetime.now()
        self.metrics = DashboardMetrics()
        
        # Signal tracking
        self.recent_signals: List[Dict] = []
        self.signal_history: Dict[str, List] = {}  # Symbol -> signals
        self.symbols_seen: set = set()
        
        # Performance tracking
        self.minute_counters = {}  # Minute -> signal count
        self.health_checks = []
        
        # Display settings
        self.max_recent_signals = 50
        self.max_history_per_symbol = 20
        self.refresh_interval = 5  # seconds
        
        self.logger.info("Signal Monitor Dashboard initialized")
    
    @property
    def redis_client(self) -> redis.Redis:
        """Redis client for dashboard operations"""
        if self._redis_client is None:
            redis_config = self.config_manager.get_redis_config()
            self._redis_client = redis.Redis(**redis_config)
        return self._redis_client
    
    async def get_aioredis_client(self) -> aioredis.Redis:
        """Async Redis client for pub/sub operations"""
        if self._aioredis_client is None:
            redis_config = self.config_manager.get_redis_config()
            # Convert sync Redis config to aioredis format
            aioredis_url = f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}/{redis_config.get('db', 0)}"
            self._aioredis_client = aioredis.from_url(aioredis_url, decode_responses=True)
            # Test connection
            await self._aioredis_client.ping()
            self.logger.info("AioRedis connection established")
        return self._aioredis_client
    
    async def start_monitoring(self):
        """Start the real-time monitoring dashboard"""
        
        self.logger.info("Starting Signal Monitor Dashboard...")
        self.is_running = True
        
        try:
            # Test connections
            await self._test_connections()
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._signal_subscriber()),
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._metrics_updater()),
                asyncio.create_task(self._display_dashboard())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            raise
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        
        self.logger.info("Stopping Signal Monitor Dashboard...")
        self.is_running = False
        
        # Close connections
        if self._redis_client:
            self._redis_client.close()
        if self._aioredis_client:
            await self._aioredis_client.close()
            self.logger.info("AioRedis connection closed")
    
    async def _test_connections(self):
        """Test Redis connections"""
        
        try:
            # Test main Redis connection
            self.redis_client.ping()
            self.logger.info("Redis main connection successful")
            
            # Test async Redis connection
            await self.get_aioredis_client()
            self.logger.info("Async Redis connection successful")
            
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            raise
    
    async def _signal_subscriber(self):
        """Subscribe to Redis signals and process them"""
        
        self.logger.info("Starting async signal subscriber...")
        
        try:
            # Get async Redis client
            aioredis_client = await self.get_aioredis_client()
            
            # Subscribe to signal channels (current Redis pattern from strategy_runner)
            signal_channel = f"{self.config.redis_key_prefix}:signals"
            alerts_channel = f"{self.config.redis_key_prefix}:alerts"
            performance_alerts_channel = f"{self.config.redis_key_prefix}:performance_alerts"
            
            # Create pub/sub object
            pubsub = aioredis_client.pubsub()
            await pubsub.subscribe(signal_channel, alerts_channel, performance_alerts_channel)
            
            self.logger.info(f"Subscribed to channels: {signal_channel}, {alerts_channel}, {performance_alerts_channel}")
            
            # Async message processing loop
            async for message in pubsub.listen():
                if not self.is_running:
                    self.logger.info("Stopping signal subscriber...")
                    break
                
                if message['type'] == 'message':
                    await self._process_signal_message(message)
            
            # Cleanup
            await pubsub.unsubscribe(signal_channel, alerts_channel, performance_alerts_channel)
            await pubsub.close()
                    
        except Exception as e:
            self.logger.error(f"Signal subscriber error: {e}")
            # Add retry logic
            if self.is_running:
                self.logger.info("Retrying signal subscriber in 5 seconds...")
                await asyncio.sleep(5)
                if self.is_running:  # Check again after sleep
                    await self._signal_subscriber()
    
    async def _process_signal_message(self, message: Dict):
        """Process received signal message"""
        
        try:
            # With aioredis decode_responses=True, data should already be decoded
            channel = message['channel']
            data = message['data']
            
            # Parse signal data
            signal_data = json.loads(data)
            
            # Only process actual trading signals (skip alerts)
            if 'signal_id' in signal_data and 'action' in signal_data:
                # Update metrics
                self._update_signal_metrics(signal_data)
                
                # Store signal
                self._store_signal(signal_data)
                
                # Log signal (debug level)
                symbol = signal_data.get('base_symbol', signal_data.get('symbol', 'UNKNOWN'))
                self.logger.debug(f"Received signal: {symbol} {signal_data.get('action')} score={signal_data.get('score')}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal message: {e}")
    
    def _update_signal_metrics(self, signal: Dict):
        """Update dashboard metrics with new signal"""
        
        # Basic metrics
        self.metrics.signals_received += 1
        self.metrics.last_signal_time = datetime.now()
        
        # Symbol tracking (handle both base_symbol and symbol formats)
        symbol = signal.get('base_symbol', signal.get('symbol', 'UNKNOWN'))
        self.symbols_seen.add(symbol)
        self.metrics.unique_symbols = len(self.symbols_seen)
        
        # Action tracking
        action = signal.get('action', 'UNKNOWN')
        if action == 'LONG':
            self.metrics.long_signals += 1
        elif action == 'SHORT':
            self.metrics.short_signals += 1
        
        # Score tracking
        score = signal.get('score', 0)
        if score >= 7.0:
            self.metrics.high_score_signals += 1
            
        if score > 0:
            current_avg = self.metrics.avg_signal_score
            total_signals = self.metrics.signals_received
            self.metrics.avg_signal_score = (
                (current_avg * (total_signals - 1) + score) / total_signals
            )
        
        # Minute-based counting for rate calculation
        current_minute = datetime.now().replace(second=0, microsecond=0)
        if current_minute not in self.minute_counters:
            self.minute_counters[current_minute] = 0
        self.minute_counters[current_minute] += 1
        
        # Clean old minute counters (keep last 5 minutes)
        cutoff_time = current_minute - timedelta(minutes=5)
        self.minute_counters = {
            minute: count for minute, count in self.minute_counters.items()
            if minute > cutoff_time
        }
        
        # Calculate signals per minute (last minute)
        last_minute = current_minute - timedelta(minutes=1)
        self.metrics.signals_per_minute = self.minute_counters.get(last_minute, 0)
    
    def _store_signal(self, signal: Dict):
        """Store signal in recent signals and history"""
        
        # Add to recent signals
        signal_with_timestamp = {
            **signal,
            'received_at': datetime.now().isoformat()
        }
        
        self.recent_signals.append(signal_with_timestamp)
        
        # Limit recent signals
        if len(self.recent_signals) > self.max_recent_signals:
            self.recent_signals.pop(0)
        
        # Add to symbol history (use base_symbol for consistency)
        symbol = signal.get('base_symbol', signal.get('symbol', 'UNKNOWN'))
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append(signal_with_timestamp)
        
        # Limit history per symbol
        if len(self.signal_history[symbol]) > self.max_history_per_symbol:
            self.signal_history[symbol].pop(0)
    
    async def _health_monitor(self):
        """Monitor system health and performance"""
        
        self.logger.info("Starting health monitor...")
        
        while self.is_running:
            try:
                # Get health status from Strategy Runner if available
                health_data = await self._get_system_health()
                
                # Store health check
                health_check = {
                    'timestamp': datetime.now().isoformat(),
                    'health_data': health_data,
                    'dashboard_metrics': {
                        'signals_received': self.metrics.signals_received,
                        'unique_symbols': self.metrics.unique_symbols,
                        'signals_per_minute': self.metrics.signals_per_minute,
                        'avg_signal_score': self.metrics.avg_signal_score,
                        'long_signals': self.metrics.long_signals,
                        'short_signals': self.metrics.short_signals,
                        'high_score_signals': self.metrics.high_score_signals
                    }
                }
                
                self.health_checks.append(health_check)
                
                # Limit health check history
                if len(self.health_checks) > 100:
                    self.health_checks.pop(0)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _get_system_health(self) -> Dict:
        """Get system health from Redis and other sources"""
        
        try:
            health_data = {
                'redis_connection': True,
                'pubsub_connection': True,
                'redis_info': {},
                'signal_stats': {}
            }
            
            # Redis info
            try:
                redis_info = self.redis_client.info()
                health_data['redis_info'] = {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0)
                }
            except Exception as e:
                health_data['redis_connection'] = False
                health_data['redis_error'] = str(e)
            
            # Signal statistics from Redis (current pattern)
            try:
                counter_key = f"{self.config.redis_key_prefix}:stats:signals_published"
                published_count = self.redis_client.get(counter_key)
                health_data['signal_stats']['published_count'] = int(published_count) if published_count else 0
                
                # Check for strategy runner health key
                runner_health_key = f"{self.config.redis_key_prefix}:health:strategy_runner"
                runner_health = self.redis_client.get(runner_health_key)
                health_data['strategy_runner_active'] = runner_health is not None
                
                # Get signal history count
                history_keys = self.redis_client.keys(f"{self.config.redis_key_prefix}:history:*")
                health_data['signal_stats']['symbols_with_history'] = len(history_keys)
                
            except Exception as e:
                health_data['signal_stats']['published_count'] = 0
                health_data['strategy_runner_active'] = False
                health_data['signal_stats']['symbols_with_history'] = 0
            
            return health_data
            
        except Exception as e:
            return {'error': str(e), 'redis_connection': False}
    
    async def _metrics_updater(self):
        """Update calculated metrics periodically"""
        
        while self.is_running:
            try:
                # Update uptime
                self.metrics.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
                
                # Wait before next update
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Metrics updater error: {e}")
                await asyncio.sleep(1)
    
    async def _display_dashboard(self):
        """Display real-time dashboard"""
        
        while self.is_running:
            try:
                # Clear screen (ANSI escape code)
                print("\033[2J\033[H", end="")
                
                # Display dashboard
                self._render_dashboard()
                
                # Wait before next refresh
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard display error: {e}")
                await asyncio.sleep(self.refresh_interval)
    
    def _render_dashboard(self):
        """Render the dashboard display"""
        
        now = datetime.now()
        uptime = timedelta(seconds=self.metrics.uptime_seconds)
        
        # Header
        print("═" * 80)
        print(f"📊 SQUEEZEFLOW SIGNAL MONITOR DASHBOARD")
        print(f"⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} | ⏱️  Uptime: {uptime}")
        print("═" * 80)
        
        # Metrics Summary
        print(f"📈 LIVE METRICS")
        print(f"   Signals Received: {self.metrics.signals_received:,}")
        print(f"   Unique Symbols: {self.metrics.unique_symbols}")
        print(f"   Signals/Minute: {self.metrics.signals_per_minute}")
        print(f"   Avg Signal Score: {self.metrics.avg_signal_score:.2f}")
        print(f"   Long/Short: {self.metrics.long_signals}/{self.metrics.short_signals}")
        print(f"   High Quality (≥7): {self.metrics.high_score_signals}")
        
        if self.metrics.last_signal_time:
            time_since_last = now - self.metrics.last_signal_time
            print(f"   Last Signal: {time_since_last.total_seconds():.0f}s ago")
        else:
            print(f"   Last Signal: Never")
        
        print()
        
        # Recent Signals
        print(f"🔔 RECENT SIGNALS (Last {min(10, len(self.recent_signals))})")
        print("   Time     | Symbol    | Action | Score | Size | Lev | Price    | ID")
        print("   " + "-" * 75)
        
        for signal in self.recent_signals[-10:]:
            received_time = datetime.fromisoformat(signal['received_at']).strftime('%H:%M:%S')
            # Handle both symbol formats (trading pair or base symbol)
            symbol = signal.get('base_symbol', signal.get('symbol', 'N/A'))[:8]
            action = signal.get('action', 'N/A')[:5]
            score = signal.get('score', 0)
            size = signal.get('position_size_factor', 0)
            leverage = signal.get('leverage', 0)
            price = signal.get('entry_price', 0)
            signal_id = signal.get('signal_id', 'N/A')[:8]
            
            print(f"   {received_time} | {symbol:9} | {action:6} | {score:5.1f} | {size:4.1f} | {leverage:3} | {price:8.0f} | {signal_id}")
        
        if not self.recent_signals:
            print("   No signals received yet...")
        
        print()
        
        # Symbol Activity
        print(f"💹 SYMBOL ACTIVITY")
        symbol_counts = {}
        for signal in self.recent_signals[-20:]:  # Last 20 signals
            # Use base_symbol for consistency
            symbol = signal.get('base_symbol', signal.get('symbol', 'UNKNOWN'))
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        for symbol, count in sorted_symbols[:10]:  # Top 10
            bar_length = min(20, count * 2)
            bar = "█" * bar_length
            print(f"   {symbol:8} | {count:2} | {bar}")
        
        if not sorted_symbols:
            print("   No symbol activity...")
        
        print()
        
        # System Health
        print(f"🏥 SYSTEM HEALTH")
        if self.health_checks:
            latest_health = self.health_checks[-1]
            health_data = latest_health.get('health_data', {})
            
            redis_status = "🟢 Connected" if health_data.get('redis_connection') else "🔴 Disconnected"
            print(f"   Redis: {redis_status}")
            
            if 'redis_info' in health_data:
                redis_info = health_data['redis_info']
                print(f"   Clients: {redis_info.get('connected_clients', 0)}")
                print(f"   Memory: {redis_info.get('used_memory_human', 'N/A')}")
                print(f"   Commands: {redis_info.get('total_commands_processed', 0):,}")
            
            published_count = health_data.get('signal_stats', {}).get('published_count', 0)
            symbols_with_history = health_data.get('signal_stats', {}).get('symbols_with_history', 0)
            strategy_runner_active = health_data.get('strategy_runner_active', False)
            
            print(f"   Published Signals: {published_count:,}")
            print(f"   Symbols w/ History: {symbols_with_history}")
            
            runner_status = "🟢 Active" if strategy_runner_active else "🟡 Unknown"
            print(f"   Strategy Runner: {runner_status}")
        else:
            print("   Waiting for health data...")
        
        print("\n" + "═" * 80)
        print(f"Press Ctrl+C to stop monitoring | Refresh: {self.refresh_interval}s")
    
    def get_dashboard_summary(self) -> Dict:
        """Get dashboard summary for API/external access"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.metrics.uptime_seconds,
            'metrics': {
                'signals_received': self.metrics.signals_received,
                'unique_symbols': self.metrics.unique_symbols,
                'signals_per_minute': self.metrics.signals_per_minute,
                'avg_signal_score': self.metrics.avg_signal_score,
                'long_signals': self.metrics.long_signals,
                'short_signals': self.metrics.short_signals,
                'high_score_signals': self.metrics.high_score_signals,
                'last_signal_time': self.metrics.last_signal_time.isoformat() if self.metrics.last_signal_time else None
            },
            'recent_signals_count': len(self.recent_signals),
            'symbols_tracked': len(self.signal_history),
            'health_checks_count': len(self.health_checks),
            'is_running': self.is_running
        }


async def main():
    """Run the signal monitor dashboard"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/signal_dashboard.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('main')
    logger.info("Starting Signal Monitor Dashboard")
    
    try:
        dashboard = SignalMonitorDashboard()
        await dashboard.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
        logger.info("Dashboard stopped by user")
    except Exception as e:
        print(f"\nDashboard error: {e}")
        logger.error(f"Dashboard error: {e}")


if __name__ == "__main__":
    asyncio.run(main())