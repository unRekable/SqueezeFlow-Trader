#!/usr/bin/env python3
"""
CVD Baseline Manager - Tracks CVD levels at position entry for exit decisions

Located in strategies/squeezeflow/ for strategy-specific functionality.
Stores and retrieves CVD baseline values when positions are opened/closed
to enable intelligent exit strategies based on CVD flow changes.
"""

import redis
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict


@dataclass
class CVDBaseline:
    """CVD baseline data stored when position opens"""
    symbol: str
    trade_id: int
    signal_id: str
    entry_time: datetime
    spot_cvd: float
    futures_cvd: float
    cvd_divergence: float
    entry_price: float
    side: str  # 'long' or 'short'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CVDBaseline':
        """Create from dictionary"""
        data = data.copy()
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)


class CVDBaselineManager:
    """Manages CVD baseline storage and retrieval"""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "squeezeflow"):
        """
        Initialize CVD baseline manager
        
        Args:
            redis_client: Redis client instance
            key_prefix: Redis key prefix
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.baseline_key = f"{key_prefix}:cvd_baselines"
        
        # Setup logging
        self.logger = logging.getLogger('cvd_baseline_manager')
        
        # Performance tracking
        self.metrics = {
            'baselines_stored': 0,
            'baselines_retrieved': 0,
            'baselines_deleted': 0,
            'storage_errors': 0,
            'retrieval_errors': 0
        }
        
        self.logger.info("CVD Baseline Manager initialized")
    
    def store_baseline(self, signal_id: str, trade_id: int, symbol: str, side: str,
                      entry_price: float, spot_cvd: float, futures_cvd: float) -> bool:
        """
        Store CVD baseline when position opens
        
        Args:
            signal_id: Signal ID that triggered the trade
            trade_id: FreqTrade trade ID
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            entry_price: Entry price
            spot_cvd: Spot CVD at entry
            futures_cvd: Futures CVD at entry
            
        Returns:
            bool: Storage success
        """
        try:
            # Calculate CVD divergence
            cvd_divergence = futures_cvd - spot_cvd
            
            # Create baseline object
            baseline = CVDBaseline(
                symbol=symbol,
                trade_id=trade_id,
                signal_id=signal_id,
                entry_time=datetime.now(),
                spot_cvd=spot_cvd,
                futures_cvd=futures_cvd,
                cvd_divergence=cvd_divergence,
                entry_price=entry_price,
                side=side
            )
            
            # Store in Redis with trade_id as key
            baseline_key = f"{self.baseline_key}:trade:{trade_id}"
            baseline_data = json.dumps(baseline.to_dict(), default=str)
            
            # Store with 7-day expiration (trades shouldn't last longer)
            success = self.redis_client.setex(
                baseline_key, 
                int(timedelta(days=7).total_seconds()),  # Convert to integer
                baseline_data
            )
            
            if success:
                self.metrics['baselines_stored'] += 1
                self.logger.info(f"Stored CVD baseline for trade {trade_id} ({symbol}): "
                               f"spot={spot_cvd:.2f}, futures={futures_cvd:.2f}, divergence={cvd_divergence:.2f}")
                
                # Also store in symbol index for quick lookups
                self._update_symbol_index(symbol, trade_id, baseline_key)
                
                return True
            else:
                self.metrics['storage_errors'] += 1
                self.logger.error(f"Failed to store CVD baseline for trade {trade_id}")
                return False
                
        except Exception as e:
            self.metrics['storage_errors'] += 1
            self.logger.error(f"Error storing CVD baseline: {e}")
            return False
    
    def get_baseline(self, trade_id: int) -> Optional[CVDBaseline]:
        """
        Retrieve CVD baseline for a trade
        
        Args:
            trade_id: FreqTrade trade ID
            
        Returns:
            CVDBaseline object or None if not found
        """
        try:
            baseline_key = f"{self.baseline_key}:trade:{trade_id}"
            baseline_data = self.redis_client.get(baseline_key)
            
            if baseline_data:
                data = json.loads(baseline_data)
                baseline = CVDBaseline.from_dict(data)
                self.metrics['baselines_retrieved'] += 1
                self.logger.debug(f"Retrieved CVD baseline for trade {trade_id}")
                return baseline
            else:
                self.logger.debug(f"No CVD baseline found for trade {trade_id}")
                return None
                
        except Exception as e:
            self.metrics['retrieval_errors'] += 1
            self.logger.error(f"Error retrieving CVD baseline: {e}")
            return None
    
    def get_symbol_baselines(self, symbol: str) -> List[CVDBaseline]:
        """
        Get all baselines for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of CVDBaseline objects
        """
        try:
            symbol_index_key = f"{self.baseline_key}:symbol:{symbol}"
            trade_ids = self.redis_client.smembers(symbol_index_key)
            
            baselines = []
            for trade_id_str in trade_ids:
                try:
                    trade_id = int(trade_id_str)
                    baseline = self.get_baseline(trade_id)
                    if baseline:
                        baselines.append(baseline)
                except (ValueError, TypeError):
                    continue
            
            return baselines
            
        except Exception as e:
            self.logger.error(f"Error getting symbol baselines: {e}")
            return []
    
    def remove_baseline(self, trade_id: int) -> bool:
        """
        Remove CVD baseline when position closes
        
        Args:
            trade_id: FreqTrade trade ID
            
        Returns:
            bool: Removal success
        """
        try:
            # Get baseline first to get symbol for index cleanup
            baseline = self.get_baseline(trade_id)
            
            # Remove from Redis
            baseline_key = f"{self.baseline_key}:trade:{trade_id}"
            deleted = self.redis_client.delete(baseline_key)
            
            # Fix: Only update symbol index if baseline exists and has symbol attribute
            if deleted and baseline is not None:
                # Remove from symbol index
                symbol_index_key = f"{self.baseline_key}:symbol:{baseline.symbol}"
                self.redis_client.srem(symbol_index_key, trade_id)
                
                self.metrics['baselines_deleted'] += 1
                self.logger.info(f"Removed CVD baseline for closed trade {trade_id}")
                return True
            
            return deleted > 0
            
        except Exception as e:
            self.logger.error(f"Error removing CVD baseline: {e}")
            return False
    
    def _update_symbol_index(self, symbol: str, trade_id: int, baseline_key: str):
        """Update symbol index for quick symbol-based lookups"""
        try:
            symbol_index_key = f"{self.baseline_key}:symbol:{symbol}"
            self.redis_client.sadd(symbol_index_key, trade_id)
            # Set expiration on index
            self.redis_client.expire(symbol_index_key, int(timedelta(days=7).total_seconds()))
            
        except Exception as e:
            self.logger.error(f"Error updating symbol index: {e}")
    
    def calculate_cvd_flow_change(self, trade_id: int, current_spot_cvd: float, 
                                current_futures_cvd: float) -> Optional[Dict[str, float]]:
        """
        Calculate CVD flow change since position entry
        
        Args:
            trade_id: FreqTrade trade ID
            current_spot_cvd: Current spot CVD
            current_futures_cvd: Current futures CVD
            
        Returns:
            Dict with flow change metrics or None if baseline not found
        """
        try:
            baseline = self.get_baseline(trade_id)
            if not baseline:
                return None
            
            # Calculate changes
            spot_change = current_spot_cvd - baseline.spot_cvd
            futures_change = current_futures_cvd - baseline.futures_cvd
            current_divergence = current_futures_cvd - current_spot_cvd
            divergence_change = current_divergence - baseline.cvd_divergence
            
            # Calculate flow metrics
            flow_metrics = {
                'spot_cvd_change': spot_change,
                'futures_cvd_change': futures_change,
                'divergence_change': divergence_change,
                'current_divergence': current_divergence,
                'baseline_divergence': baseline.cvd_divergence,
                'time_since_entry_hours': (datetime.now() - baseline.entry_time).total_seconds() / 3600,
                'flow_direction_maintained': self._is_flow_direction_maintained(baseline.side, divergence_change)
            }
            
            return flow_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating CVD flow change: {e}")
            return None
    
    def _is_flow_direction_maintained(self, side: str, divergence_change: float) -> bool:
        """
        Check if CVD flow direction is maintained
        
        Args:
            side: Position side ('long' or 'short')
            divergence_change: Change in CVD divergence
            
        Returns:
            bool: True if flow direction is maintained
        """
        if side == 'long':
            # For long positions, we want increasing or stable divergence (futures leading up)
            return divergence_change >= -0.1  # Allow small negative changes
        else:  # short
            # For short positions, we want decreasing or stable divergence (futures leading down)
            return divergence_change <= 0.1   # Allow small positive changes
    
    def cleanup_expired_baselines(self) -> int:
        """
        Clean up expired baselines (Redis TTL should handle this, but manual cleanup for safety)
        
        Returns:
            int: Number of baselines cleaned up
        """
        try:
            cleaned = 0
            pattern = f"{self.baseline_key}:trade:*"
            
            # Scan for all baseline keys
            for key in self.redis_client.scan_iter(match=pattern):
                try:
                    # Check if key exists and get data
                    data = self.redis_client.get(key)
                    if data:
                        baseline_data = json.loads(data)
                        entry_time = datetime.fromisoformat(baseline_data['entry_time'])
                        
                        # Remove baselines older than 7 days
                        if datetime.now() - entry_time > timedelta(days=7):
                            trade_id = int(key.split(':')[-1])
                            if self.remove_baseline(trade_id):
                                cleaned += 1
                                
                except (ValueError, json.JSONDecodeError, KeyError):
                    # Invalid data, remove it
                    self.redis_client.delete(key)
                    cleaned += 1
            
            if cleaned > 0:
                self.logger.info(f"Cleaned up {cleaned} expired CVD baselines")
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error during baseline cleanup: {e}")
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CVD baseline manager metrics"""
        return {
            **self.metrics,
            'total_active_baselines': len(list(self.redis_client.scan_iter(match=f"{self.baseline_key}:trade:*"))),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'baselines_stored': 0,
            'baselines_retrieved': 0,
            'baselines_deleted': 0,
            'storage_errors': 0,
            'retrieval_errors': 0
        }
        self.logger.info("CVD baseline manager metrics reset")


def create_cvd_baseline_manager_from_config(config_manager) -> CVDBaselineManager:
    """
    Factory function to create CVD baseline manager from configuration
    
    Args:
        config_manager: ConfigManager instance
        
    Returns:
        CVDBaselineManager instance
    """
    from services.config.service_config import ConfigManager
    
    # Create Redis client
    redis_config = config_manager.get_redis_config()
    redis_client = redis.Redis(**redis_config)
    
    # Get key prefix from config
    config = config_manager.get_config()
    key_prefix = config.redis_key_prefix
    
    manager = CVDBaselineManager(redis_client, key_prefix)
    return manager