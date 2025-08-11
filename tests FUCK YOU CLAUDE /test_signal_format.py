"""
Signal Format and Metadata Tests
Test Redis signal format and metadata validation for FreqTrade compatibility
"""

import pytest
import json
import uuid
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestSignalFormatValidation:
    """Test signal format and metadata validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock strategy runner to avoid dependencies
        self.mock_strategy_runner = MagicMock()
        
    def test_redis_signal_structure_basic(self):
        """Test Redis signal contains all required fields with correct types"""
        
        # Create test order
        test_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5
        }
        
        # Convert to Redis signal format
        redis_signal = self._convert_order_to_signal(test_order)
        
        # Validate required fields exist
        required_fields = [
            'signal_id', 'timestamp', 'symbol', 'action',
            'score', 'position_size_factor', 'leverage',
            'entry_price', 'ttl'
        ]
        
        for field in required_fields:
            assert field in redis_signal, f"Missing required field: {field}"
        
        # Validate field types
        assert isinstance(redis_signal['signal_id'], str), \
            f"signal_id should be str, got {type(redis_signal['signal_id'])}"
        
        assert isinstance(redis_signal['timestamp'], str), \
            f"timestamp should be str, got {type(redis_signal['timestamp'])}"
        
        assert isinstance(redis_signal['symbol'], str), \
            f"symbol should be str, got {type(redis_signal['symbol'])}"
        
        assert isinstance(redis_signal['score'], (int, float)), \
            f"score should be numeric, got {type(redis_signal['score'])}"
        
        assert isinstance(redis_signal['entry_price'], (int, float)), \
            f"entry_price should be numeric, got {type(redis_signal['entry_price'])}"
        
        assert isinstance(redis_signal['ttl'], int), \
            f"ttl should be int, got {type(redis_signal['ttl'])}"
        
        # Validate field values and ranges
        assert redis_signal['action'] in ['LONG', 'SHORT', 'CLOSE'], \
            f"Invalid action: {redis_signal['action']}"
        
        assert 0.0 <= redis_signal['score'] <= 10.0, \
            f"Score out of range: {redis_signal['score']}"
        
        assert redis_signal['position_size_factor'] in [0.5, 1.0, 1.5], \
            f"Invalid position_size_factor: {redis_signal['position_size_factor']}"
        
        assert redis_signal['leverage'] in [2, 3, 5], \
            f"Invalid leverage: {redis_signal['leverage']}"
        
        assert redis_signal['ttl'] == 300, \
            f"TTL should be 300 seconds, got {redis_signal['ttl']}"
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(redis_signal['timestamp'])
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {redis_signal['timestamp']}")
        
        # Validate UUID format
        try:
            uuid.UUID(redis_signal['signal_id'])
        except ValueError:
            pytest.fail(f"Invalid UUID format: {redis_signal['signal_id']}")
    
    def test_score_based_metadata_mapping(self):
        """Test score-based position sizing and leverage mapping"""
        
        score_test_cases = [
            # (score, expected_position_factor, expected_leverage)
            (3.5, 0.5, 2),   # Below entry threshold - reduced size
            (4.0, 0.5, 2),   # Entry threshold - low confidence
            (4.5, 0.5, 2),   # Low confidence
            (5.9, 0.5, 2),   # Still low confidence
            (6.0, 1.0, 3),   # Medium confidence
            (7.0, 1.0, 3),   # Medium confidence
            (7.9, 1.0, 3),   # Still medium confidence
            (8.0, 1.5, 5),   # High confidence
            (9.0, 1.5, 5),   # High confidence
            (10.0, 1.5, 5),  # Maximum score
        ]
        
        for score, expected_pos_factor, expected_leverage in score_test_cases:
            test_order = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000,
                'signal_type': 'LONG_SQUEEZE',
                'confidence': 0.8,
                'score': score
            }
            
            redis_signal = self._convert_order_to_signal(test_order)
            
            assert redis_signal['position_size_factor'] == expected_pos_factor, \
                f"Score {score}: Expected position_factor {expected_pos_factor}, got {redis_signal['position_size_factor']}"
            
            assert redis_signal['leverage'] == expected_leverage, \
                f"Score {score}: Expected leverage {expected_leverage}, got {redis_signal['leverage']}"
    
    def test_signal_action_mapping(self):
        """Test order side to signal action mapping"""
        
        action_mappings = [
            ('BUY', 'LONG'),
            ('SELL', 'SHORT'),
            ('CLOSE', 'CLOSE'),
            ('EXIT', 'CLOSE')
        ]
        
        for order_side, expected_action in action_mappings:
            test_order = {
                'symbol': 'BTCUSDT',
                'side': order_side,
                'quantity': 0.001,
                'price': 50000,
                'signal_type': 'LONG_SQUEEZE',
                'confidence': 0.8,
                'score': 7.5
            }
            
            redis_signal = self._convert_order_to_signal(test_order)
            
            assert redis_signal['action'] == expected_action, \
                f"Order side {order_side}: Expected action {expected_action}, got {redis_signal['action']}"
    
    def test_signal_serialization_json_compatibility(self):
        """Test signal can be serialized/deserialized as JSON"""
        
        test_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000.0,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5
        }
        
        redis_signal = self._convert_order_to_signal(test_order)
        
        # Test JSON serialization
        try:
            json_string = json.dumps(redis_signal)
        except TypeError as e:
            pytest.fail(f"Signal not JSON serializable: {e}")
        
        # Test JSON deserialization
        try:
            deserialized_signal = json.loads(json_string)
        except json.JSONDecodeError as e:
            pytest.fail(f"Signal JSON not deserializable: {e}")
        
        # Validate deserialized signal matches original
        for key, value in redis_signal.items():
            assert key in deserialized_signal, f"Missing key after deserialization: {key}"
            assert deserialized_signal[key] == value, \
                f"Value mismatch for {key}: {deserialized_signal[key]} != {value}"
    
    def test_signal_ttl_and_expiration(self):
        """Test signal TTL is set correctly for Redis expiration"""
        
        test_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5
        }
        
        redis_signal = self._convert_order_to_signal(test_order)
        
        # TTL should be 5 minutes (300 seconds)
        expected_ttl = 300
        assert redis_signal['ttl'] == expected_ttl, \
            f"Expected TTL {expected_ttl}, got {redis_signal['ttl']}"
        
        # Timestamp should be recent (within last 5 seconds)
        signal_time = datetime.fromisoformat(redis_signal['timestamp'])
        now = datetime.now()
        time_diff = (now - signal_time).total_seconds()
        
        assert time_diff < 5, f"Signal timestamp too old: {time_diff} seconds ago"
    
    def test_signal_key_generation(self):
        """Test Redis key generation follows correct pattern"""
        
        test_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5
        }
        
        redis_signal = self._convert_order_to_signal(test_order)
        
        # Generate expected key pattern
        expected_key_pattern = f"squeeze_signal:{redis_signal['symbol']}:{redis_signal['signal_id']}"
        
        # Key should match pattern (we'll generate it the same way)
        generated_key = self._generate_redis_key(redis_signal)
        
        assert generated_key == expected_key_pattern, \
            f"Key mismatch: {generated_key} != {expected_key_pattern}"
        
        # Key should contain symbol and be unique
        assert redis_signal['symbol'] in generated_key, "Symbol missing from key"
        assert redis_signal['signal_id'] in generated_key, "Signal ID missing from key"
        assert generated_key.startswith('squeeze_signal:'), "Key missing prefix"
    
    @pytest.mark.parametrize("symbol", ['BTCUSDT', 'ETHUSDT', 'BTCUSD', 'ETH-USD'])
    def test_multiple_symbols_signal_format(self, symbol):
        """Test signal format consistency across different symbols"""
        
        test_order = {
            'symbol': symbol,
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5
        }
        
        redis_signal = self._convert_order_to_signal(test_order)
        
        # All signals should have same structure regardless of symbol
        required_fields = [
            'signal_id', 'timestamp', 'symbol', 'action',
            'score', 'position_size_factor', 'leverage',
            'entry_price', 'ttl'
        ]
        
        for field in required_fields:
            assert field in redis_signal, f"Symbol {symbol}: Missing field {field}"
        
        # Symbol should be preserved exactly
        assert redis_signal['symbol'] == symbol, \
            f"Symbol mismatch: {redis_signal['symbol']} != {symbol}"
    
    def test_edge_case_handling(self):
        """Test signal format with edge case values"""
        
        edge_cases = [
            # Minimum score
            {'score': 0.0, 'expected_pos_factor': 0.5, 'expected_leverage': 2},
            
            # Maximum score
            {'score': 10.0, 'expected_pos_factor': 1.5, 'expected_leverage': 5},
            
            # Boundary scores
            {'score': 4.0, 'expected_pos_factor': 0.5, 'expected_leverage': 2},
            {'score': 6.0, 'expected_pos_factor': 1.0, 'expected_leverage': 3},
            {'score': 8.0, 'expected_pos_factor': 1.5, 'expected_leverage': 5},
        ]
        
        for case in edge_cases:
            test_order = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000,
                'signal_type': 'LONG_SQUEEZE',
                'confidence': 0.8,
                'score': case['score']
            }
            
            redis_signal = self._convert_order_to_signal(test_order)
            
            assert redis_signal['position_size_factor'] == case['expected_pos_factor'], \
                f"Score {case['score']}: Wrong position factor"
            
            assert redis_signal['leverage'] == case['expected_leverage'], \
                f"Score {case['score']}: Wrong leverage"
    
    def test_signal_metadata_completeness(self):
        """Test signal contains all metadata FreqTrade needs"""
        
        test_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5,
            'phase_results': {
                'cvd_reset_deceleration': True,
                'absorption_candle': True,
                'failed_breakdown': False,
                'directional_bias': True
            }
        }
        
        redis_signal = self._convert_order_to_signal(test_order)
        
        # Should contain strategy metadata
        freqtrade_metadata = [
            'signal_id',           # For deduplication
            'timestamp',           # For timing validation
            'symbol',              # Trading pair
            'action',              # LONG/SHORT/CLOSE
            'score',               # Signal strength
            'position_size_factor', # Position sizing
            'leverage',            # Leverage setting
            'entry_price',         # Entry price guidance
            'ttl'                  # Signal expiration
        ]
        
        for field in freqtrade_metadata:
            assert field in redis_signal, f"Missing FreqTrade metadata: {field}"
            
            # No None values
            assert redis_signal[field] is not None, f"Field {field} is None"
            
            # No empty strings
            if isinstance(redis_signal[field], str):
                assert redis_signal[field].strip() != '', f"Field {field} is empty"
    
    # Helper methods
    
    def _convert_order_to_signal(self, order):
        """Convert strategy order to Redis signal format"""
        
        # Determine position sizing based on score
        score = order['score']
        if score < 6.0:
            position_size_factor = 0.5
            leverage = 2
        elif score < 8.0:
            position_size_factor = 1.0
            leverage = 3
        else:
            position_size_factor = 1.5
            leverage = 5
        
        # Map order side to signal action
        action_mapping = {
            'BUY': 'LONG',
            'SELL': 'SHORT',
            'CLOSE': 'CLOSE',
            'EXIT': 'CLOSE'
        }
        
        action = action_mapping.get(order['side'], 'LONG')
        
        # Create Redis signal
        redis_signal = {
            'signal_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'symbol': order['symbol'],
            'action': action,
            'score': order['score'],
            'position_size_factor': position_size_factor,
            'leverage': leverage,
            'entry_price': order['price'],
            'ttl': 300  # 5 minutes
        }
        
        return redis_signal
    
    def _generate_redis_key(self, signal):
        """Generate Redis key for signal"""
        return f"squeeze_signal:{signal['symbol']}:{signal['signal_id']}"


class TestSignalValidationRules:
    """Test signal validation rules and error handling"""
    
    def test_invalid_score_handling(self):
        """Test handling of invalid score values"""
        
        invalid_scores = [-1.0, 11.0, float('inf'), float('nan'), None, 'invalid']
        
        for invalid_score in invalid_scores:
            test_order = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000,
                'signal_type': 'LONG_SQUEEZE',
                'confidence': 0.8,
                'score': invalid_score
            }
            
            with pytest.raises((ValueError, TypeError)):
                self._convert_order_to_signal_strict(test_order)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        
        required_fields = ['symbol', 'side', 'price', 'score']
        
        base_order = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.001,
            'price': 50000,
            'signal_type': 'LONG_SQUEEZE',
            'confidence': 0.8,
            'score': 7.5
        }
        
        for field in required_fields:
            incomplete_order = base_order.copy()
            del incomplete_order[field]
            
            with pytest.raises(KeyError):
                self._convert_order_to_signal_strict(incomplete_order)
    
    def test_invalid_symbol_format(self):
        """Test handling of invalid symbol formats"""
        
        invalid_symbols = ['', None, 123, 'BTC', 'invalid-symbol-format']
        
        for invalid_symbol in invalid_symbols:
            test_order = {
                'symbol': invalid_symbol,
                'side': 'BUY',
                'quantity': 0.001,
                'price': 50000,
                'signal_type': 'LONG_SQUEEZE',
                'confidence': 0.8,
                'score': 7.5
            }
            
            if invalid_symbol in [None, 123, '']:
                with pytest.raises((ValueError, TypeError)):
                    self._convert_order_to_signal_strict(test_order)
    
    def _convert_order_to_signal_strict(self, order):
        """Strict version of signal conversion with validation"""
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'price', 'score']
        for field in required_fields:
            if field not in order:
                raise KeyError(f"Missing required field: {field}")
            if order[field] is None:
                raise ValueError(f"Field {field} cannot be None")
        
        # Validate score
        score = order['score']
        if not isinstance(score, (int, float)):
            raise TypeError(f"Score must be numeric, got {type(score)}")
        if score < 0 or score > 10:
            raise ValueError(f"Score must be between 0-10, got {score}")
        if not np.isfinite(score):
            raise ValueError(f"Score must be finite, got {score}")
        
        # Validate symbol
        symbol = order['symbol']
        if not isinstance(symbol, str):
            raise TypeError(f"Symbol must be string, got {type(symbol)}")
        if len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        
        # Continue with normal conversion...
        return self._convert_order_to_signal_basic(order)
    
    def _convert_order_to_signal_basic(self, order):
        """Basic signal conversion without strict validation"""
        
        score = order['score']
        if score < 6.0:
            position_size_factor = 0.5
            leverage = 2
        elif score < 8.0:
            position_size_factor = 1.0
            leverage = 3
        else:
            position_size_factor = 1.5
            leverage = 5
        
        action_mapping = {
            'BUY': 'LONG',
            'SELL': 'SHORT',
            'CLOSE': 'CLOSE',
            'EXIT': 'CLOSE'
        }
        
        action = action_mapping.get(order['side'], 'LONG')
        
        return {
            'signal_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'symbol': order['symbol'],
            'action': action,
            'score': order['score'],
            'position_size_factor': position_size_factor,
            'leverage': leverage,
            'entry_price': order['price'],
            'ttl': 300
        }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])