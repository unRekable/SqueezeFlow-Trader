#!/usr/bin/env python3
"""
FakeRedis - In-memory Redis replacement for backtest environment

Provides Redis-compatible interface for CVD baseline tracking during backtests
without requiring actual Redis server connection.
"""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta


class FakeRedis:
    """
    In-memory Redis replacement for backtesting
    
    Implements minimal Redis interface needed for CVD baseline management
    """
    
    def __init__(self):
        """Initialize in-memory storage"""
        self.data: Dict[str, str] = {}
        self.expiration: Dict[str, datetime] = {}
        self.sets: Dict[str, set] = {}
        self.lists: Dict[str, List[str]] = {}  # For list operations
        
    def setex(self, key: str, time: Union[int, float, timedelta], value: str) -> bool:
        """Set key with expiration time"""
        try:
            self.data[key] = value
            
            # Handle time parameter
            if isinstance(time, timedelta):
                expire_time = datetime.now() + time
            else:
                expire_time = datetime.now() + timedelta(seconds=float(time))
                
            self.expiration[key] = expire_time
            return True
            
        except Exception:
            return False
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        # Check expiration - Fix: Collect expired keys first to avoid modifying dict during iteration
        if key in self.expiration:
            if datetime.now() > self.expiration[key]:
                # Key expired - delete it
                expired_keys = [key]
                for k in expired_keys:
                    self.delete(k)
                return None
                
        return self.data.get(key)
    
    def delete(self, key: str) -> int:
        """Delete key"""
        deleted = 0
        
        if key in self.data:
            del self.data[key]
            deleted += 1
            
        if key in self.expiration:
            del self.expiration[key]
            
        if key in self.sets:
            del self.sets[key]
            
        if key in self.lists:
            del self.lists[key]
            
        return deleted
    
    def sadd(self, key: str, *values) -> int:
        """Add values to set"""
        if key not in self.sets:
            self.sets[key] = set()
            
        added = 0
        for value in values:
            if value not in self.sets[key]:
                self.sets[key].add(str(value))
                added += 1
                
        return added
    
    def srem(self, key: str, *values) -> int:
        """Remove values from set"""
        if key not in self.sets:
            return 0
            
        removed = 0
        for value in values:
            if str(value) in self.sets[key]:
                self.sets[key].remove(str(value))
                removed += 1
                
        # Clean up empty set
        if not self.sets[key]:
            del self.sets[key]
            
        return removed
    
    def smembers(self, key: str) -> set:
        """Get all members of set"""
        # Check expiration
        if key in self.expiration:
            if datetime.now() > self.expiration[key]:
                self.delete(key)
                return set()
                
        return self.sets.get(key, set()).copy()
    
    def expire(self, key: str, time: Union[int, float, timedelta]) -> bool:
        """Set expiration for key"""
        try:
            # Handle time parameter
            if isinstance(time, timedelta):
                expire_time = datetime.now() + time
            else:
                expire_time = datetime.now() + timedelta(seconds=float(time))
                
            self.expiration[key] = expire_time
            return True
            
        except Exception:
            return False
    
    def scan_iter(self, match: str = "*", count: int = 10) -> List[str]:
        """Scan keys matching pattern"""
        import fnmatch
        
        # Clean expired keys first
        self._cleanup_expired()
        
        keys = []
        for key in self.data.keys():
            if fnmatch.fnmatch(key, match):
                keys.append(key)
                
        return keys
    
    def _cleanup_expired(self):
        """Clean up expired keys"""
        now = datetime.now()
        expired_keys = []
        
        for key, expire_time in self.expiration.items():
            if now > expire_time:
                expired_keys.append(key)
                
        for key in expired_keys:
            self.delete(key)
    
    def lpush(self, key: str, *values) -> int:
        """Push values to left of list"""
        if key not in self.lists:
            self.lists[key] = []
            
        # Add values to left (front) of list
        for value in reversed(values):
            self.lists[key].insert(0, str(value))
            
        return len(self.lists[key])
    
    def lrange(self, key: str, start: int, stop: int) -> List[str]:
        """Get range of elements from list"""
        if key not in self.lists:
            return []
            
        # Check expiration
        if key in self.expiration:
            if datetime.now() > self.expiration[key]:
                self.delete(key)
                return []
        
        lst = self.lists[key]
        
        # Handle negative indices
        if stop == -1:
            return lst[start:]
        else:
            return lst[start:stop+1]
    
    def ltrim(self, key: str, start: int, stop: int) -> bool:
        """Trim list to specified range"""
        if key not in self.lists:
            return True
            
        lst = self.lists[key]
        
        # Handle negative indices  
        if stop == -1:
            self.lists[key] = lst[start:]
        else:
            self.lists[key] = lst[start:stop+1]
            
        return True
    
    def flushall(self):
        """Clear all data (for testing)"""
        self.data.clear()
        self.expiration.clear()
        self.sets.clear()
        self.lists.clear()
    
    def info(self) -> Dict[str, Any]:
        """Get info about fake redis instance"""
        self._cleanup_expired()
        
        return {
            'redis_version': 'FakeRedis-1.0.0',
            'connected_clients': 1,
            'used_memory': len(str(self.data)) + len(str(self.sets)),
            'used_memory_human': f"{len(str(self.data)) + len(str(self.sets))} bytes",
            'total_keys': len(self.data) + len(self.sets),
            'expired_keys': 0,
            'keyspace_hits': 0,
            'keyspace_misses': 0
        }
    
    def ping(self) -> str:
        """Ping response"""
        return "PONG"