#!/usr/bin/env python3
"""
Signal Validator - Comprehensive validation module for trading signals

Provides validation, deduplication, and quality control for signals before
Redis publishing. Ensures signal integrity and prevents duplicate signals.
"""

import hashlib
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class ValidationResult(Enum):
    """Signal validation results"""
    VALID = "valid"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"


@dataclass
class ValidationError:
    """Signal validation error details"""
    code: str
    message: str
    field: Optional[str] = None
    severity: str = "error"  # error, warning, info


@dataclass
class SignalMetrics:
    """Signal processing metrics for monitoring"""
    total_validated: int = 0
    valid_signals: int = 0
    invalid_signals: int = 0
    duplicate_signals: int = 0
    rate_limited_signals: int = 0
    validation_errors: Dict[str, int] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = {}


class SignalValidator:
    """Comprehensive signal validation and deduplication system"""
    
    def __init__(self, config: Dict):
        """
        Initialize Signal Validator
        
        Args:
            config: Validation configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('signal_validator')
        
        # Deduplication tracking
        self._signal_hashes: Set[str] = set()
        self._signal_timestamps: Dict[str, datetime] = {}
        self._symbol_last_signal: Dict[str, datetime] = {}
        
        # Metrics tracking
        self.metrics = SignalMetrics()
        
        # Validation rules configuration
        self.validation_rules = self._initialize_validation_rules()
        
        # Rate limiting configuration - removed cooldown for unlimited trading
        self.rate_limits = {
            'max_signals_per_minute': config.get('max_signals_per_minute', 100),  # Increased for unlimited trading
            'max_signals_per_symbol_per_hour': config.get('max_signals_per_symbol_per_hour', 100),  # Increased for unlimited trading
            'cooldown_minutes': config.get('signal_cooldown_minutes', 0)  # No cooldown - trade when conditions are met
        }
        
        # Cleanup configuration
        self.cleanup_interval = timedelta(hours=config.get('cleanup_interval_hours', 24))
        self.last_cleanup = datetime.now()
        
        self.logger.info("Signal Validator initialized with comprehensive validation rules")
    
    def _initialize_validation_rules(self) -> Dict:
        """Initialize signal validation rules"""
        return {
            'required_fields': [
                'signal_id', 'timestamp', 'symbol', 'action', 'score',
                'position_size_factor', 'leverage', 'entry_price', 'ttl'
            ],
            'symbol_pattern': r'^[A-Z]{2,10}$',  # 2-10 uppercase letters
            'valid_actions': ['LONG', 'SHORT', 'CLOSE'],
            'score_range': (0.0, 10.0),
            'position_size_range': (0.1, 2.0),
            'leverage_range': (1, 10),
            'min_entry_price': 0.000001,
            'max_entry_price': 1000000,
            'min_ttl': 60,  # 1 minute
            'max_ttl': 3600,  # 1 hour
            'max_confidence': 1.0,
            'min_confidence': 0.0
        }
    
    def validate_signal(self, signal: Dict) -> Tuple[ValidationResult, List[ValidationError]]:
        """
        Comprehensive signal validation
        
        Args:
            signal: Signal dictionary to validate
            
        Returns:
            Tuple of (ValidationResult, List of ValidationErrors)
        """
        self.metrics.total_validated += 1
        errors = []
        
        try:
            # 1. Basic structure validation
            structure_errors = self._validate_structure(signal)
            errors.extend(structure_errors)
            
            # 2. Field type and range validation
            field_errors = self._validate_fields(signal)
            errors.extend(field_errors)
            
            # 3. Business logic validation
            business_errors = self._validate_business_logic(signal)
            errors.extend(business_errors)
            
            # If basic validation fails, don't proceed with deduplication
            if errors:
                self.metrics.invalid_signals += 1
                self._update_error_metrics(errors)
                return ValidationResult.INVALID, errors
            
            # 4. Deduplication check
            if self._is_duplicate_signal(signal):
                self.metrics.duplicate_signals += 1
                return ValidationResult.DUPLICATE, [
                    ValidationError("DUPLICATE_SIGNAL", "Signal already processed recently")
                ]
            
            # 5. Rate limiting check
            if self._is_rate_limited(signal):
                self.metrics.rate_limited_signals += 1
                return ValidationResult.RATE_LIMITED, [
                    ValidationError("RATE_LIMITED", "Signal rate limit exceeded")
                ]
            
            # 6. Expiry check
            if self._is_signal_expired(signal):
                return ValidationResult.EXPIRED, [
                    ValidationError("SIGNAL_EXPIRED", "Signal timestamp is too old")
                ]
            
            # Signal is valid - track it
            self._track_valid_signal(signal)
            self.metrics.valid_signals += 1
            
            return ValidationResult.VALID, []
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            error = ValidationError("VALIDATION_EXCEPTION", f"Validation exception: {str(e)}")
            self.metrics.invalid_signals += 1
            return ValidationResult.INVALID, [error]
    
    def _validate_structure(self, signal: Dict) -> List[ValidationError]:
        """Validate signal structure and required fields"""
        errors = []
        
        # Check if signal is a dictionary
        if not isinstance(signal, dict):
            errors.append(ValidationError(
                "INVALID_TYPE", "Signal must be a dictionary"
            ))
            return errors
        
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in signal:
                errors.append(ValidationError(
                    "MISSING_FIELD", f"Required field '{field}' is missing", field
                ))
            elif signal[field] is None:
                errors.append(ValidationError(
                    "NULL_FIELD", f"Required field '{field}' cannot be null", field
                ))
        
        return errors
    
    def _validate_fields(self, signal: Dict) -> List[ValidationError]:
        """Validate individual field types and ranges"""
        errors = []
        
        # Symbol validation
        symbol = signal.get('symbol')
        if symbol and not isinstance(symbol, str):
            errors.append(ValidationError(
                "INVALID_SYMBOL_TYPE", "Symbol must be a string", "symbol"
            ))
        elif symbol and not symbol.isupper():
            errors.append(ValidationError(
                "INVALID_SYMBOL_FORMAT", "Symbol must be uppercase", "symbol"
            ))
        elif symbol and len(symbol) < 2:
            errors.append(ValidationError(
                "INVALID_SYMBOL_LENGTH", "Symbol must be at least 2 characters", "symbol"
            ))
        
        # Action validation
        action = signal.get('action')
        if action and action not in self.validation_rules['valid_actions']:
            errors.append(ValidationError(
                "INVALID_ACTION", f"Action must be one of {self.validation_rules['valid_actions']}", "action"
            ))
        
        # Score validation
        score = signal.get('score')
        if score is not None:
            if not isinstance(score, (int, float)):
                errors.append(ValidationError(
                    "INVALID_SCORE_TYPE", "Score must be a number", "score"
                ))
            else:
                min_score, max_score = self.validation_rules['score_range']
                if not (min_score <= score <= max_score):
                    errors.append(ValidationError(
                        "INVALID_SCORE_RANGE", 
                        f"Score must be between {min_score} and {max_score}", "score"
                    ))
        
        # Position size factor validation
        pos_size = signal.get('position_size_factor')
        if pos_size is not None:
            if not isinstance(pos_size, (int, float)):
                errors.append(ValidationError(
                    "INVALID_POSITION_SIZE_TYPE", "Position size factor must be a number", "position_size_factor"
                ))
            else:
                min_pos, max_pos = self.validation_rules['position_size_range']
                if not (min_pos <= pos_size <= max_pos):
                    errors.append(ValidationError(
                        "INVALID_POSITION_SIZE_RANGE",
                        f"Position size factor must be between {min_pos} and {max_pos}", "position_size_factor"
                    ))
        
        # Leverage validation
        leverage = signal.get('leverage')
        if leverage is not None:
            if not isinstance(leverage, int):
                errors.append(ValidationError(
                    "INVALID_LEVERAGE_TYPE", "Leverage must be an integer", "leverage"
                ))
            else:
                min_lev, max_lev = self.validation_rules['leverage_range']
                if not (min_lev <= leverage <= max_lev):
                    errors.append(ValidationError(
                        "INVALID_LEVERAGE_RANGE",
                        f"Leverage must be between {min_lev} and {max_lev}", "leverage"
                    ))
        
        # Entry price validation
        entry_price = signal.get('entry_price')
        if entry_price is not None:
            if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                errors.append(ValidationError(
                    "INVALID_ENTRY_PRICE", "Entry price must be a positive number", "entry_price"
                ))
            else:
                min_price = self.validation_rules['min_entry_price']
                max_price = self.validation_rules['max_entry_price']
                if not (min_price <= entry_price <= max_price):
                    errors.append(ValidationError(
                        "INVALID_ENTRY_PRICE_RANGE",
                        f"Entry price must be between {min_price} and {max_price}", "entry_price"
                    ))
        
        # TTL validation
        ttl = signal.get('ttl')
        if ttl is not None:
            if not isinstance(ttl, int) or ttl <= 0:
                errors.append(ValidationError(
                    "INVALID_TTL", "TTL must be a positive integer", "ttl"
                ))
            else:
                min_ttl = self.validation_rules['min_ttl']
                max_ttl = self.validation_rules['max_ttl']
                if not (min_ttl <= ttl <= max_ttl):
                    errors.append(ValidationError(
                        "INVALID_TTL_RANGE",
                        f"TTL must be between {min_ttl} and {max_ttl} seconds", "ttl"
                    ))
        
        # Confidence validation (optional field)
        confidence = signal.get('confidence')
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                errors.append(ValidationError(
                    "INVALID_CONFIDENCE_TYPE", "Confidence must be a number", "confidence"
                ))
            else:
                min_conf = self.validation_rules['min_confidence']
                max_conf = self.validation_rules['max_confidence']
                if not (min_conf <= confidence <= max_conf):
                    errors.append(ValidationError(
                        "INVALID_CONFIDENCE_RANGE",
                        f"Confidence must be between {min_conf} and {max_conf}", "confidence"
                    ))
        
        # Timestamp validation
        timestamp = signal.get('timestamp')
        if timestamp:
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                errors.append(ValidationError(
                    "INVALID_TIMESTAMP", "Timestamp must be valid ISO format", "timestamp"
                ))
        
        return errors
    
    def _validate_business_logic(self, signal: Dict) -> List[ValidationError]:
        """Validate business logic rules"""
        errors = []
        
        # Score-based validation
        score = signal.get('score', 0)
        position_size = signal.get('position_size_factor', 1.0)
        leverage = signal.get('leverage', 1)
        
        # High score should have reasonable position sizing
        if score >= 8.0 and position_size < 1.0:
            errors.append(ValidationError(
                "INCONSISTENT_HIGH_SCORE", 
                "High score signals should have position size >= 1.0", 
                "position_size_factor",
                "warning"
            ))
        
        # Low score should have conservative position sizing
        if score < 4.0 and position_size > 1.0:
            errors.append(ValidationError(
                "RISKY_LOW_SCORE", 
                "Low score signals should have conservative position sizing", 
                "position_size_factor",
                "warning"
            ))
        
        # Leverage should correlate with score
        if score >= 8.0 and leverage < 3:
            errors.append(ValidationError(
                "CONSERVATIVE_HIGH_SCORE", 
                "High score signals could use higher leverage", 
                "leverage",
                "info"
            ))
        
        # Confidence and score consistency
        confidence = signal.get('confidence', 0)
        if confidence and abs(confidence * 10 - score) > 2.0:
            errors.append(ValidationError(
                "INCONSISTENT_CONFIDENCE_SCORE", 
                "Confidence and score values seem inconsistent", 
                "confidence",
                "warning"
            ))
        
        return errors
    
    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """Check if signal is a duplicate"""
        
        # Create signal fingerprint for deduplication
        signal_hash = self._create_signal_hash(signal)
        
        # Check if we've seen this exact signal recently
        if signal_hash in self._signal_hashes:
            return True
        
        # Check if we have a recent signal for this symbol with same action
        symbol = signal.get('symbol')
        action = signal.get('action')
        dedup_key = f"{symbol}:{action}"
        
        if dedup_key in self._signal_timestamps:
            last_signal_time = self._signal_timestamps[dedup_key]
            time_diff = datetime.now() - last_signal_time
            
            # Consider duplicate if same symbol+action within cooldown period
            cooldown = timedelta(minutes=self.rate_limits['cooldown_minutes'])
            if time_diff < cooldown:
                return True
        
        return False
    
    def _is_rate_limited(self, signal: Dict) -> bool:
        """Check if signal exceeds rate limits"""
        now = datetime.now()
        symbol = signal.get('symbol')
        
        # Check global rate limit (signals per minute)
        minute_ago = now - timedelta(minutes=1)
        recent_signals = len([
            ts for ts in self._signal_timestamps.values() 
            if ts > minute_ago
        ])
        
        if recent_signals >= self.rate_limits['max_signals_per_minute']:
            return True
        
        # Check per-symbol rate limit (signals per hour)
        hour_ago = now - timedelta(hours=1)
        symbol_signals = len([
            ts for key, ts in self._signal_timestamps.items()
            if key.startswith(f"{symbol}:") and ts > hour_ago
        ])
        
        if symbol_signals >= self.rate_limits['max_signals_per_symbol_per_hour']:
            return True
        
        return False
    
    def _is_signal_expired(self, signal: Dict) -> bool:
        """Check if signal is too old to be valid"""
        timestamp_str = signal.get('timestamp')
        if not timestamp_str:
            return False
        
        try:
            signal_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if signal_time.tzinfo:
                signal_time = signal_time.replace(tzinfo=None)
            
            # Signal is expired if older than 5 minutes
            age = datetime.now() - signal_time
            return age > timedelta(minutes=5)
            
        except (ValueError, AttributeError):
            return True  # Invalid timestamp is considered expired
    
    def _track_valid_signal(self, signal: Dict):
        """Track valid signal for deduplication and rate limiting"""
        
        # Store signal hash
        signal_hash = self._create_signal_hash(signal)
        self._signal_hashes.add(signal_hash)
        
        # Store timestamp for rate limiting
        symbol = signal.get('symbol')
        action = signal.get('action')
        dedup_key = f"{symbol}:{action}"
        self._signal_timestamps[dedup_key] = datetime.now()
        
        # Store symbol last signal time
        self._symbol_last_signal[symbol] = datetime.now()
        
        # Periodic cleanup
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_tracking_data()
    
    def _create_signal_hash(self, signal: Dict) -> str:
        """Create unique hash for signal deduplication"""
        
        # Use key fields for hash generation
        hash_fields = {
            'symbol': signal.get('symbol'),
            'action': signal.get('action'),
            'entry_price': round(signal.get('entry_price', 0), 8),  # Round to avoid float precision issues
            'position_size_factor': signal.get('position_size_factor'),
            'leverage': signal.get('leverage')
        }
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(hash_fields, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _cleanup_tracking_data(self):
        """Clean up old tracking data to prevent memory leaks"""
        
        now = datetime.now()
        cleanup_threshold = now - timedelta(hours=24)
        
        # Clean up old timestamps
        old_keys = [
            key for key, timestamp in self._signal_timestamps.items()
            if timestamp < cleanup_threshold
        ]
        
        for key in old_keys:
            del self._signal_timestamps[key]
        
        # Clean up old symbol tracking
        old_symbols = [
            symbol for symbol, timestamp in self._symbol_last_signal.items()
            if timestamp < cleanup_threshold
        ]
        
        for symbol in old_symbols:
            del self._symbol_last_signal[symbol]
        
        # Clear signal hashes (they're time-based anyway)
        self._signal_hashes.clear()
        
        self.last_cleanup = now
        self.logger.info(f"Cleaned up {len(old_keys)} old signal records")
    
    def _update_error_metrics(self, errors: List[ValidationError]):
        """Update error metrics for monitoring"""
        
        for error in errors:
            error_code = error.code
            if error_code in self.metrics.validation_errors:
                self.metrics.validation_errors[error_code] += 1
            else:
                self.metrics.validation_errors[error_code] = 1
    
    def get_validation_metrics(self) -> Dict:
        """Get comprehensive validation metrics"""
        
        total = self.metrics.total_validated
        if total == 0:
            return {'status': 'no_signals_processed'}
        
        return {
            'total_signals_validated': total,
            'valid_signals': self.metrics.valid_signals,
            'invalid_signals': self.metrics.invalid_signals,
            'duplicate_signals': self.metrics.duplicate_signals,
            'rate_limited_signals': self.metrics.rate_limited_signals,
            'validation_success_rate': (self.metrics.valid_signals / total) * 100,
            'error_breakdown': self.metrics.validation_errors.copy(),
            'tracking_stats': {
                'tracked_hashes': len(self._signal_hashes),
                'tracked_timestamps': len(self._signal_timestamps),
                'tracked_symbols': len(self._symbol_last_signal)
            },
            'rate_limiting_config': self.rate_limits.copy(),
            'last_cleanup': self.last_cleanup.isoformat()
        }
    
    def reset_metrics(self):
        """Reset validation metrics"""
        self.metrics = SignalMetrics()
        self.logger.info("Validation metrics reset")
    
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        
        if symbol not in self._symbol_last_signal:
            return False
        
        last_signal_time = self._symbol_last_signal[symbol]
        cooldown = timedelta(minutes=self.rate_limits['cooldown_minutes'])
        
        return datetime.now() - last_signal_time < cooldown
    
    def get_symbol_cooldown_remaining(self, symbol: str) -> int:
        """Get remaining cooldown time for symbol in seconds"""
        
        if not self.is_symbol_in_cooldown(symbol):
            return 0
        
        last_signal_time = self._symbol_last_signal[symbol]
        cooldown = timedelta(minutes=self.rate_limits['cooldown_minutes'])
        elapsed = datetime.now() - last_signal_time
        remaining = cooldown - elapsed
        
        return max(0, int(remaining.total_seconds()))


class BatchSignalValidator:
    """Batch validation for high-frequency signal processing"""
    
    def __init__(self, validator: SignalValidator):
        self.validator = validator
        self.logger = logging.getLogger('batch_signal_validator')
    
    def validate_batch(self, signals: List[Dict]) -> Dict:
        """
        Validate a batch of signals efficiently
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Dictionary with validation results and statistics
        """
        if not signals:
            return {'valid_signals': [], 'invalid_signals': [], 'batch_stats': {}}
        
        valid_signals = []
        invalid_signals = []
        validation_stats = {
            'total_processed': len(signals),
            'valid_count': 0,
            'invalid_count': 0,
            'duplicate_count': 0,
            'rate_limited_count': 0,
            'expired_count': 0
        }
        
        for i, signal in enumerate(signals):
            try:
                result, errors = self.validator.validate_signal(signal)
                
                if result == ValidationResult.VALID:
                    valid_signals.append(signal)
                    validation_stats['valid_count'] += 1
                else:
                    invalid_signals.append({
                        'signal': signal,
                        'result': result.value,
                        'errors': [{'code': e.code, 'message': e.message} for e in errors],
                        'batch_index': i
                    })
                    validation_stats['invalid_count'] += 1
                    
                    # Update specific counters
                    if result == ValidationResult.DUPLICATE:
                        validation_stats['duplicate_count'] += 1
                    elif result == ValidationResult.RATE_LIMITED:
                        validation_stats['rate_limited_count'] += 1
                    elif result == ValidationResult.EXPIRED:
                        validation_stats['expired_count'] += 1
                        
            except Exception as e:
                self.logger.error(f"Batch validation error at index {i}: {e}")
                invalid_signals.append({
                    'signal': signal,
                    'result': 'exception',
                    'errors': [{'code': 'BATCH_VALIDATION_ERROR', 'message': str(e)}],
                    'batch_index': i
                })
                validation_stats['invalid_count'] += 1
        
        return {
            'valid_signals': valid_signals,
            'invalid_signals': invalid_signals,
            'batch_stats': validation_stats
        }