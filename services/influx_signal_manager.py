#!/usr/bin/env python3
"""
Enhanced InfluxDB Signal Manager
Provides advanced signal storage, analytics, and monitoring capabilities for the SqueezeFlow Trader

This module enhances the existing basic signal storage with:
- Signal performance tracking and analytics
- Query helpers for signal analysis
- Aggregation views and statistics
- Signal archival and long-term analysis
- Monitoring queries for signal quality
- Comprehensive retention policy management

Compatible with existing InfluxDB 1.8.10 and 'significant_trades' database structure.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from influxdb import InfluxDBClient
try:
    from influxdb.exceptions import InfluxDBError
except ImportError:
    # Fallback for different InfluxDB client versions
    InfluxDBError = Exception
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class SignalOutcome(Enum):
    """Signal outcome tracking"""
    PENDING = "pending"
    PROFITABLE = "profitable"
    UNPROFITABLE = "unprofitable"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class SignalPerformanceMetrics:
    """Signal performance metrics"""
    signal_id: str
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    outcome: SignalOutcome
    holding_time_minutes: Optional[int]
    score: float
    confidence: float


@dataclass
class SignalAnalytics:
    """Comprehensive signal analytics"""
    total_signals: int
    profitable_signals: int
    unprofitable_signals: int
    pending_signals: int
    expired_signals: int
    win_rate: float
    average_pnl: float
    average_holding_time: float
    total_pnl: float
    max_profit: float
    max_loss: float
    profit_factor: float
    sharpe_ratio: Optional[float]
    avg_score: float
    avg_confidence: float


class InfluxSignalManager:
    """
    Enhanced InfluxDB Signal Manager
    
    Provides comprehensive signal management with analytics, performance tracking,
    and monitoring capabilities while maintaining compatibility with existing infrastructure.
    """
    
    def __init__(self, influx_client: InfluxDBClient, database: str = "significant_trades"):
        """
        Initialize Enhanced InfluxDB Signal Manager
        
        Args:
            influx_client: Configured InfluxDB client instance
            database: Database name (default: significant_trades)
        """
        self.client = influx_client
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.operation_metrics = {
            'writes_performed': 0,
            'queries_executed': 0,
            'errors_encountered': 0,
            'avg_write_time': 0.0,
            'avg_query_time': 0.0
        }
        
        # Validate connection and setup
        self._validate_setup()
        
        self.logger.info(f"Enhanced InfluxDB Signal Manager initialized for database: {database}")
    
    def _validate_setup(self):
        """Validate InfluxDB connection and database setup"""
        try:
            # Test connection
            self.client.ping()
            
            # Verify database exists
            databases = [db['name'] for db in self.client.get_list_database()]
            if self.database not in databases:
                raise ValueError(f"Database '{self.database}' does not exist")
            
            # Check retention policies
            retention_policies = self.client.get_list_retention_policies(self.database)
            self.logger.info(f"Found {len(retention_policies)} retention policies")
            
            # Log existing measurements for reference
            measurements_result = self.client.query(f'SHOW MEASUREMENTS ON "{self.database}"')
            measurements = [point['name'] for point in measurements_result.get_points()]
            self.logger.info(f"Available measurements: {measurements}")
            
        except Exception as e:
            self.logger.error(f"InfluxDB setup validation failed: {e}")
            raise
    
    def store_enhanced_signal(self, signal: Dict, additional_fields: Dict = None) -> bool:
        """
        Store signal with enhanced tracking capabilities
        
        Args:
            signal: Base signal dictionary from strategy_runner
            additional_fields: Additional fields for enhanced tracking
            
        Returns:
            bool: Success status
        """
        start_time = datetime.now()
        
        try:
            # Create enhanced point structure
            point = {
                'measurement': 'strategy_signals',
                'tags': {
                    'symbol': signal['symbol'],
                    'action': signal['action'],
                    'strategy': signal.get('strategy', 'SqueezeFlowStrategy'),
                    'service': signal.get('service', 'strategy_runner'),
                    'score_tier': self._get_score_tier(signal.get('score', 0)),
                    'confidence_tier': self._get_confidence_tier(signal.get('confidence', 0))
                },
                'time': signal['timestamp'],
                'fields': {
                    # Core signal fields
                    'signal_id': signal['signal_id'],
                    'score': float(signal.get('score', 0)),
                    'position_size_factor': float(signal.get('position_size_factor', 1.0)),
                    'leverage': int(signal.get('leverage', 1)),
                    'entry_price': float(signal.get('entry_price', 0)),
                    'confidence': float(signal.get('confidence', 0)),
                    'reasoning': signal.get('reasoning', ''),
                    
                    # Enhanced tracking fields
                    'ttl_seconds': int(signal.get('ttl', 300)),
                    'outcome': SignalOutcome.PENDING.value,
                    'created_at': datetime.now().isoformat(),
                    'tracking_id': str(uuid.uuid4())[:8]
                }
            }
            
            # Add additional fields if provided
            if additional_fields:
                point['fields'].update(additional_fields)
            
            # Write to InfluxDB
            success = self.client.write_points([point], time_precision='ms')
            
            if success:
                write_time = (datetime.now() - start_time).total_seconds()
                self._update_operation_metrics('write', write_time, True)
                self.logger.debug(f"Enhanced signal stored: {signal['signal_id']}")
                return True
            else:
                self._update_operation_metrics('write', 0, False)
                self.logger.error(f"Failed to store signal: {signal['signal_id']}")
                return False
                
        except Exception as e:
            write_time = (datetime.now() - start_time).total_seconds()
            self._update_operation_metrics('write', write_time, False)
            self.logger.error(f"Error storing enhanced signal: {e}")
            return False
    
    def update_signal_outcome(self, signal_id: str, outcome: SignalOutcome, 
                            exit_price: float = None, pnl: float = None,
                            pnl_percentage: float = None) -> bool:
        """
        Update signal outcome after trade completion
        
        Args:
            signal_id: Signal identifier
            outcome: Final outcome of the signal
            exit_price: Exit price if trade was executed
            pnl: Profit/Loss amount
            pnl_percentage: Profit/Loss percentage
            
        Returns:
            bool: Success status
        """
        try:
            # Calculate holding time
            entry_signal = self.get_signal_by_id(signal_id)
            if not entry_signal:
                self.logger.warning(f"Signal {signal_id} not found for outcome update")
                return False
            
            entry_time = pd.to_datetime(entry_signal.get('time'))
            exit_time = datetime.now()
            holding_time_minutes = int((exit_time - entry_time).total_seconds() / 60)
            
            # Create update point
            point = {
                'measurement': 'strategy_signals',
                'tags': {
                    'symbol': entry_signal['symbol'],
                    'action': entry_signal['action'],
                    'strategy': entry_signal.get('strategy', 'SqueezeFlowStrategy'),
                    'service': entry_signal.get('service', 'strategy_runner'),
                    'score_tier': self._get_score_tier(entry_signal.get('score', 0)),
                    'confidence_tier': self._get_confidence_tier(entry_signal.get('confidence', 0))
                },
                'time': exit_time,
                'fields': {
                    # Reference to original signal
                    'signal_id': signal_id,
                    'tracking_id': entry_signal.get('tracking_id'),
                    'original_entry_time': entry_signal.get('time'),
                    
                    # Copy original signal data
                    'score': float(entry_signal.get('score', 0)),
                    'confidence': float(entry_signal.get('confidence', 0)),
                    'entry_price': float(entry_signal.get('entry_price', 0)),
                    
                    # Outcome data
                    'outcome': outcome.value,
                    'exit_time': exit_time.isoformat(),
                    'holding_time_minutes': holding_time_minutes,
                    'updated_at': exit_time.isoformat()
                }
            }
            
            # Add financial metrics if provided
            if exit_price is not None:
                point['fields']['exit_price'] = float(exit_price)
            if pnl is not None:
                point['fields']['pnl'] = float(pnl)
            if pnl_percentage is not None:
                point['fields']['pnl_percentage'] = float(pnl_percentage)
            
            # Write outcome update
            success = self.client.write_points([point], time_precision='ms')
            
            if success:
                self.logger.info(f"Signal outcome updated: {signal_id} -> {outcome.value}")
                return True
            else:
                self.logger.error(f"Failed to update signal outcome: {signal_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating signal outcome: {e}")
            return False
    
    def get_signal_analytics(self, symbol: str = None, hours_back: int = 24,
                           strategy: str = None) -> SignalAnalytics:
        """
        Get comprehensive signal analytics
        
        Args:
            symbol: Filter by symbol (optional)
            hours_back: Hours to look back for analysis
            strategy: Filter by strategy (optional)
            
        Returns:
            SignalAnalytics: Comprehensive analytics object
        """
        try:
            # Build query conditions
            conditions = [f"time >= now() - {hours_back}h"]
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if strategy:
                conditions.append(f"strategy = '{strategy}'")
            
            where_clause = " AND ".join(conditions)
            
            # Execute analytics queries
            analytics_queries = {
                'overview': f'''
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable,
                        COUNT(CASE WHEN outcome = 'unprofitable' THEN 1 END) as unprofitable,
                        COUNT(CASE WHEN outcome = 'pending' THEN 1 END) as pending,
                        COUNT(CASE WHEN outcome = 'expired' THEN 1 END) as expired,
                        MEAN(score) as avg_score,
                        MEAN(confidence) as avg_confidence
                    FROM strategy_signals 
                    WHERE {where_clause}
                ''',
                
                'financial': f'''
                    SELECT 
                        MEAN(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as max_profit,
                        MIN(pnl) as max_loss,
                        MEAN(holding_time_minutes) as avg_holding_time
                    FROM strategy_signals 
                    WHERE {where_clause} AND pnl IS NOT NULL
                '''
            }
            
            results = {}
            for query_name, query in analytics_queries.items():
                try:
                    result = self.client.query(query)
                    points = list(result.get_points())
                    results[query_name] = points[0] if points else {}
                except Exception as e:
                    self.logger.warning(f"Analytics query '{query_name}' failed: {e}")
                    results[query_name] = {}
            
            # Build analytics object
            overview = results.get('overview', {})
            financial = results.get('financial', {})
            
            total_signals = overview.get('total_signals', 0)
            profitable = overview.get('profitable', 0)
            unprofitable = overview.get('unprofitable', 0)
            
            # Calculate derived metrics
            win_rate = (profitable / max(1, profitable + unprofitable)) * 100
            avg_profit = financial.get('avg_pnl', 0) if profitable > 0 else 0
            max_loss = abs(financial.get('max_loss', 0))
            profit_factor = (avg_profit * profitable) / max(1, max_loss * unprofitable) if unprofitable > 0 else float('inf')
            
            analytics = SignalAnalytics(
                total_signals=total_signals,
                profitable_signals=profitable,
                unprofitable_signals=unprofitable,
                pending_signals=overview.get('pending', 0),
                expired_signals=overview.get('expired', 0),
                win_rate=win_rate,
                average_pnl=financial.get('avg_pnl', 0),
                average_holding_time=financial.get('avg_holding_time', 0),
                total_pnl=financial.get('total_pnl', 0),
                max_profit=financial.get('max_profit', 0),
                max_loss=financial.get('max_loss', 0),
                profit_factor=profit_factor,
                sharpe_ratio=None,  # Would need price data for calculation
                avg_score=overview.get('avg_score', 0),
                avg_confidence=overview.get('avg_confidence', 0)
            )
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting signal analytics: {e}")
            return SignalAnalytics(
                total_signals=0, profitable_signals=0, unprofitable_signals=0,
                pending_signals=0, expired_signals=0, win_rate=0, average_pnl=0,
                average_holding_time=0, total_pnl=0, max_profit=0, max_loss=0,
                profit_factor=0, sharpe_ratio=None, avg_score=0, avg_confidence=0
            )
    
    def get_signal_performance_by_score(self, hours_back: int = 168) -> Dict[str, Dict]:
        """
        Get signal performance breakdown by score tiers
        
        Args:
            hours_back: Hours to analyze (default: 1 week)
            
        Returns:
            Dict: Performance metrics by score tier
        """
        try:
            query = f'''
                SELECT 
                    score_tier,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable,
                    COUNT(CASE WHEN outcome = 'unprofitable' THEN 1 END) as unprofitable,
                    MEAN(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    MEAN(holding_time_minutes) as avg_holding_time,
                    MEAN(score) as avg_score
                FROM strategy_signals 
                WHERE time >= now() - {hours_back}h 
                AND outcome != 'pending'
                GROUP BY score_tier
            '''
            
            result = self.client.query(query)
            points = list(result.get_points())
            
            performance_by_tier = {}
            for point in points:
                tier = point.get('score_tier', 'unknown')
                total = point.get('total_signals', 0)
                profitable = point.get('profitable', 0)
                unprofitable = point.get('unprofitable', 0)
                
                win_rate = (profitable / max(1, profitable + unprofitable)) * 100
                
                performance_by_tier[tier] = {
                    'total_signals': total,
                    'profitable_signals': profitable,
                    'unprofitable_signals': unprofitable,
                    'win_rate_percent': round(win_rate, 2),
                    'avg_pnl': point.get('avg_pnl', 0),
                    'total_pnl': point.get('total_pnl', 0),
                    'avg_holding_time_minutes': point.get('avg_holding_time', 0),
                    'avg_score': point.get('avg_score', 0)
                }
            
            return performance_by_tier
            
        except Exception as e:
            self.logger.error(f"Error getting performance by score: {e}")
            return {}
    
    def get_recent_signals(self, symbol: str = None, limit: int = 100,
                         outcome_filter: SignalOutcome = None) -> List[Dict]:
        """
        Get recent signals with optional filtering
        
        Args:
            symbol: Filter by symbol
            limit: Maximum signals to return
            outcome_filter: Filter by outcome
            
        Returns:
            List[Dict]: List of signal records
        """
        try:
            conditions = []
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if outcome_filter:
                conditions.append(f"outcome = '{outcome_filter.value}'")
            
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            
            query = f'''
                SELECT * FROM strategy_signals 
                {where_clause}
                ORDER BY time DESC 
                LIMIT {limit}
            '''
            
            result = self.client.query(query)
            signals = list(result.get_points())
            
            # Convert timestamps for better readability
            for signal in signals:
                if 'time' in signal:
                    signal['time'] = pd.to_datetime(signal['time']).strftime('%Y-%m-%d %H:%M:%S')
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_signal_by_id(self, signal_id: str) -> Optional[Dict]:
        """
        Get specific signal by ID
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Optional[Dict]: Signal record or None
        """
        try:
            query = f'''
                SELECT * FROM strategy_signals 
                WHERE signal_id = '{signal_id}'
                ORDER BY time DESC 
                LIMIT 1
            '''
            
            result = self.client.query(query)
            points = list(result.get_points())
            
            return points[0] if points else None
            
        except Exception as e:
            self.logger.error(f"Error getting signal by ID: {e}")
            return None
    
    def create_signal_aggregation_view(self, timeframe: str = '1h') -> bool:
        """
        Create continuous query for signal aggregation
        
        Args:
            timeframe: Aggregation timeframe (1h, 4h, 1d)
            
        Returns:
            bool: Success status
        """
        try:
            cq_name = f"cq_signal_aggregation_{timeframe}"
            target_measurement = f"signal_analytics_{timeframe}"
            
            # Drop existing continuous query if exists
            try:
                self.client.query(f'DROP CONTINUOUS QUERY "{cq_name}" ON "{self.database}"')
            except:
                pass  # CQ might not exist
            
            # Create new continuous query
            cq_query = f'''
                CREATE CONTINUOUS QUERY "{cq_name}" ON "{self.database}"
                BEGIN
                    SELECT 
                        COUNT(*) as signal_count,
                        COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_count,
                        COUNT(CASE WHEN outcome = 'unprofitable' THEN 1 END) as unprofitable_count,
                        MEAN(score) as avg_score,
                        MEAN(confidence) as avg_confidence,
                        MEAN(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl
                    INTO "{target_measurement}"
                    FROM "strategy_signals"
                    GROUP BY time({timeframe}), symbol, strategy
                END
            '''
            
            self.client.query(cq_query)
            self.logger.info(f"Created signal aggregation view: {target_measurement}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating aggregation view: {e}")
            return False
    
    def get_signal_quality_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get signal quality and monitoring metrics
        
        Args:
            hours_back: Hours to analyze
            
        Returns:
            Dict: Quality metrics and health indicators
        """
        try:
            # Signal volume and frequency
            volume_query = f'''
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(*) / {hours_back} as signals_per_hour,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT strategy) as unique_strategies
                FROM strategy_signals 
                WHERE time >= now() - {hours_back}h
            '''
            
            # Signal outcome distribution
            outcome_query = f'''
                SELECT 
                    outcome,
                    COUNT(*) as count
                FROM strategy_signals 
                WHERE time >= now() - {hours_back}h
                GROUP BY outcome
            '''
            
            # Signal timing metrics
            timing_query = f'''
                SELECT 
                    MEAN(holding_time_minutes) as avg_holding_time,
                    MAX(holding_time_minutes) as max_holding_time,
                    MIN(holding_time_minutes) as min_holding_time,
                    STDDEV(holding_time_minutes) as holding_time_stddev
                FROM strategy_signals 
                WHERE time >= now() - {hours_back}h 
                AND holding_time_minutes IS NOT NULL
            '''
            
            # Execute queries
            volume_result = self.client.query(volume_query)
            outcome_result = self.client.query(outcome_query)
            timing_result = self.client.query(timing_query)
            
            # Process results
            volume_data = list(volume_result.get_points())[0] if volume_result else {}
            outcome_data = list(outcome_result.get_points())
            timing_data = list(timing_result.get_points())[0] if timing_result else {}
            
            # Build outcome distribution
            outcome_distribution = {}
            for point in outcome_data:
                outcome_distribution[point['outcome']] = point['count']
            
            # Calculate health indicators
            total_signals = volume_data.get('total_signals', 0)
            pending_signals = outcome_distribution.get('pending', 0)
            expired_signals = outcome_distribution.get('expired', 0)
            
            pending_ratio = (pending_signals / max(1, total_signals)) * 100
            expired_ratio = (expired_signals / max(1, total_signals)) * 100
            
            health_indicators = {
                'healthy_signal_volume': total_signals >= hours_back * 0.5,  # At least 0.5 signals/hour
                'low_pending_ratio': pending_ratio < 20,  # Less than 20% pending
                'low_expired_ratio': expired_ratio < 10,  # Less than 10% expired
                'multiple_symbols': volume_data.get('unique_symbols', 0) > 1,
                'consistent_strategy': volume_data.get('unique_strategies', 0) >= 1
            }
            
            return {
                'volume_metrics': volume_data,
                'outcome_distribution': outcome_distribution,
                'timing_metrics': timing_data,
                'quality_indicators': {
                    'pending_ratio_percent': round(pending_ratio, 2),
                    'expired_ratio_percent': round(expired_ratio, 2),
                    'signals_per_hour': round(volume_data.get('signals_per_hour', 0), 2)
                },
                'health_indicators': health_indicators,
                'overall_health_score': sum(health_indicators.values()) / len(health_indicators) * 100,
                'analysis_period_hours': hours_back,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal quality metrics: {e}")
            return {'error': str(e)}
    
    def archive_old_signals(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Archive old signals to reduce database size
        
        Args:
            days_to_keep: Days of signals to keep in main measurement
            
        Returns:
            Dict: Archive operation results
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # Export old signals to archive measurement
            export_query = f'''
                SELECT * INTO "strategy_signals_archive" 
                FROM "strategy_signals" 
                WHERE time < '{cutoff_time.isoformat()}'
            '''
            
            # Execute archive operation
            export_result = self.client.query(export_query)
            
            # Count archived signals
            count_query = f'''
                SELECT COUNT(*) FROM "strategy_signals_archive"
                WHERE time < '{cutoff_time.isoformat()}'
            '''
            
            count_result = self.client.query(count_query)
            archived_count = list(count_result.get_points())[0]['count'] if count_result else 0
            
            # Delete old signals from main measurement
            delete_query = f'''
                DELETE FROM "strategy_signals" 
                WHERE time < '{cutoff_time.isoformat()}'
            '''
            
            self.client.query(delete_query)
            
            self.logger.info(f"Archived {archived_count} signals older than {days_to_keep} days")
            
            return {
                'archived_signals': archived_count,
                'cutoff_date': cutoff_time.isoformat(),
                'days_kept': days_to_keep,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error archiving old signals: {e}")
            return {
                'archived_signals': 0,
                'error': str(e),
                'success': False
            }
    
    def get_retention_policy_info(self) -> List[Dict]:
        """
        Get current retention policy information
        
        Returns:
            List[Dict]: Retention policy details
        """
        try:
            policies = self.client.get_list_retention_policies(self.database)
            
            # Convert durations to human readable format
            for policy in policies:
                duration = policy.get('duration', '0s')
                if duration == '0s':
                    policy['duration_readable'] = 'Infinite'
                else:
                    # Parse duration (e.g., "720h0m0s" -> "30 days")
                    hours = int(duration.split('h')[0]) if 'h' in duration else 0
                    days = hours // 24
                    if days > 0:
                        policy['duration_readable'] = f"{days} days"
                    else:
                        policy['duration_readable'] = f"{hours} hours"
            
            return policies
            
        except Exception as e:
            self.logger.error(f"Error getting retention policies: {e}")
            return []
    
    def get_operation_metrics(self) -> Dict[str, Any]:
        """
        Get manager operation metrics
        
        Returns:
            Dict: Operation performance metrics
        """
        return {
            'operation_metrics': self.operation_metrics.copy(),
            'health_status': {
                'low_error_rate': self.operation_metrics['errors_encountered'] < 10,
                'reasonable_write_time': self.operation_metrics['avg_write_time'] < 0.1,
                'reasonable_query_time': self.operation_metrics['avg_query_time'] < 0.5
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_operation_metrics(self):
        """Reset operation metrics"""
        self.operation_metrics = {
            'writes_performed': 0,
            'queries_executed': 0,
            'errors_encountered': 0,
            'avg_write_time': 0.0,
            'avg_query_time': 0.0
        }
        self.logger.info("Operation metrics reset")
    
    def _get_score_tier(self, score: float) -> str:
        """Categorize signal score into tiers"""
        if score >= 8.0:
            return 'high'
        elif score >= 6.0:
            return 'medium'
        elif score >= 4.0:
            return 'low'
        else:
            return 'very_low'
    
    def _get_confidence_tier(self, confidence: float) -> str:
        """Categorize confidence into tiers"""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _update_operation_metrics(self, operation_type: str, duration: float, success: bool):
        """Update operation performance metrics"""
        if operation_type == 'write':
            self.operation_metrics['writes_performed'] += 1
            if success:
                # Update average write time
                count = self.operation_metrics['writes_performed']
                current_avg = self.operation_metrics['avg_write_time']
                self.operation_metrics['avg_write_time'] = (
                    (current_avg * (count - 1) + duration) / count
                )
            else:
                self.operation_metrics['errors_encountered'] += 1
        
        elif operation_type == 'query':
            self.operation_metrics['queries_executed'] += 1
            if success:
                # Update average query time
                count = self.operation_metrics['queries_executed']
                current_avg = self.operation_metrics['avg_query_time']
                self.operation_metrics['avg_query_time'] = (
                    (current_avg * (count - 1) + duration) / count
                )
            else:
                self.operation_metrics['errors_encountered'] += 1


def create_signal_manager_from_config(config_manager) -> InfluxSignalManager:
    """
    Factory function to create InfluxSignalManager from ConfigManager
    
    Args:
        config_manager: ConfigManager instance
        
    Returns:
        InfluxSignalManager: Configured signal manager instance
    """
    try:
        influx_config = config_manager.get_influx_config()
        influx_client = InfluxDBClient(**influx_config)
        
        database = influx_config.get('database', 'significant_trades')
        signal_manager = InfluxSignalManager(influx_client, database)
        
        return signal_manager
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating signal manager: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    import sys
    from services.config.unified_config import ConfigManager
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing Enhanced InfluxDB Signal Manager")
    
    try:
        # Initialize with config
        config_manager = ConfigManager()
        signal_manager = create_signal_manager_from_config(config_manager)
        
        # Test basic functionality
        test_signal = {
            'signal_id': f'test_{uuid.uuid4().hex[:8]}',
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            'action': 'LONG',
            'score': 7.5,
            'confidence': 0.85,
            'entry_price': 45000.0,
            'position_size_factor': 1.0,
            'leverage': 3,
            'strategy': 'SqueezeFlowStrategy',
            'service': 'test_runner',
            'reasoning': 'Test signal for enhanced manager'
        }
        
        # Store test signal
        success = signal_manager.store_enhanced_signal(test_signal)
        if success:
            logger.info("Test signal stored successfully")
            
            # Get analytics
            analytics = signal_manager.get_signal_analytics(hours_back=1)
            logger.info(f"Current analytics: {analytics.total_signals} total signals")
            
            # Get quality metrics
            quality = signal_manager.get_signal_quality_metrics(hours_back=1)
            logger.info(f"Quality score: {quality.get('overall_health_score', 0):.1f}%")
            
        else:
            logger.error("Failed to store test signal")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    
    logger.info("Enhanced InfluxDB Signal Manager test completed")