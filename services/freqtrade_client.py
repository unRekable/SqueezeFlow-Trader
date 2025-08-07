#!/usr/bin/env python3
"""
FreqTrade API Client - Integration with FreqTrade REST API

Provides position tracking, trade status, and portfolio information
for the SqueezeFlow Strategy Runner Service.
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a FreqTrade position"""
    trade_id: int
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    quantity: float  # Amount in base currency
    entry_time: datetime
    pnl: float
    pnl_percentage: float
    is_open: bool


class FreqTradeAPIClient:
    """FreqTrade REST API client for position tracking"""
    
    def __init__(self, api_url: str, username: str, password: str = None, timeout: int = 10):
        """
        Initialize FreqTrade API client with JWT authentication
        
        Args:
            api_url: FreqTrade API base URL
            username: API username (for JWT login)
            password: API password (required for JWT login)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.username = username
        self.password = password or ""
        self.timeout = timeout
        
        # Setup logging
        self.logger = logging.getLogger('freqtrade_client')
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # JWT token storage
        self.access_token = None
        self.refresh_token = None
        
        # Authenticate and get JWT tokens
        if not self._authenticate():
            self.logger.warning("Failed to authenticate with FreqTrade API - will retry on first request")
            # Don't raise exception here - allow retry during first request
        
        self.logger.info(f"FreqTrade API client initialized with JWT auth: {api_url}")
    
    def _authenticate(self) -> bool:
        """
        Authenticate with FreqTrade API using JWT token authentication
        
        Returns:
            bool: Authentication success
        """
        try:
            # Login endpoint uses basic auth with username and password
            login_url = f"{self.api_url}/api/v1/token/login"
            
            # Create basic auth with username and password
            import base64
            auth_string = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            
            headers = {
                'Authorization': f'Basic {auth_string}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            self.logger.info(f"Attempting JWT login for user: {self.username} to {login_url}")
            response = requests.post(login_url, headers=headers, timeout=self.timeout)
            
            self.logger.debug(f"Login response status: {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"Login response content: {response.text}")
            
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens.get('access_token')
                self.refresh_token = tokens.get('refresh_token')
                
                if self.access_token:
                    # Set JWT token in session headers for all future requests
                    self.session.headers['Authorization'] = f'Bearer {self.access_token}'
                    self.logger.info("JWT authentication successful")
                    return True
                else:
                    self.logger.error("No access token received from login")
                    return False
            elif response.status_code == 401:
                # Try fallback to basic auth for older FreqTrade versions
                self.logger.info("JWT auth failed, trying basic auth fallback")
                return self._setup_basic_auth()
            else:
                self.logger.error(f"Login failed: {response.status_code} - {response.text}")
                # Try fallback to basic auth
                self.logger.info("Trying basic auth fallback after login failure")
                return self._setup_basic_auth()
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            # Try fallback to basic auth
            self.logger.info("Trying basic auth fallback after exception")
            return self._setup_basic_auth()
    
    def _setup_basic_auth(self) -> bool:
        """
        Setup basic authentication as fallback
        
        Returns:
            bool: Setup success
        """
        try:
            import base64
            auth_string = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            
            # Set basic auth in session headers
            self.session.headers['Authorization'] = f'Basic {auth_string}'
            self.logger.info("Basic authentication setup complete")
            
            # Test the connection
            test_response = self.session.get(f"{self.api_url}/api/v1/ping", timeout=self.timeout)
            if test_response.status_code == 200:
                self.logger.info("Basic auth connection test successful")
                return True
            else:
                self.logger.error(f"Basic auth test failed: {test_response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Basic auth setup error: {e}")
            return False
    
    def _refresh_token_if_needed(self) -> bool:
        """
        Refresh JWT token if needed
        
        Returns:
            bool: Token refresh success
        """
        if not self.refresh_token:
            self.logger.warning("No refresh token available, need to re-authenticate")
            return self._authenticate()
        
        try:
            refresh_url = f"{self.api_url}/api/v1/token/refresh"
            headers = {
                'Authorization': f'Bearer {self.refresh_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.post(refresh_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens.get('access_token')
                # Update session with new token
                self.session.headers['Authorization'] = f'Bearer {self.access_token}'
                self.logger.debug("JWT token refreshed successfully")
                return True
            else:
                self.logger.error(f"Token refresh failed: {response.status_code}")
                # Try full re-authentication
                return self._authenticate()
                
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            # Try full re-authentication
            return self._authenticate()
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict = None, retry_auth: bool = True) -> Optional[Dict]:
        """
        Make authenticated request to FreqTrade API using JWT token or basic auth
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data for POST/PUT requests
            retry_auth: Whether to retry with token refresh on 401
            
        Returns:
            Response data or None on error
        """
        try:
            url = f"{self.api_url}{endpoint}"
            self.logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            self.logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401 and retry_auth:
                # Authentication failed, try to re-authenticate and retry
                self.logger.warning("Received 401, attempting re-authentication")
                if self._authenticate():
                    # Retry the request once with new auth
                    return self._make_request(endpoint, method, data, retry_auth=False)
                else:
                    self.logger.error("Re-authentication failed")
                    return None
            elif response.status_code == 401:
                self.logger.error("Authentication failed after retry attempt")
                return None
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"API request error for {endpoint}: {e}")
            return None
    
    def _parse_balance_value(self, raw_value) -> float:
        """
        Parse balance value with unit conversion handling
        
        Args:
            raw_value: Raw value from FreqTrade API (could be string, float, dict)
            
        Returns:
            float: Parsed balance value in USD
        """
        try:
            self.logger.debug(f"Parsing balance value: {raw_value} (type: {type(raw_value).__name__})")
            
            # Handle different value types
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
            elif isinstance(raw_value, str):
                # Remove any currency symbols and convert
                clean_value = raw_value.replace('$', '').replace(',', '').strip()
                value = float(clean_value)
            elif isinstance(raw_value, dict):
                # If it's a dict, look for common value keys
                for key in ['value', 'total', 'balance', 'amount']:
                    if key in raw_value:
                        return self._parse_balance_value(raw_value[key])
                value = 0.0
            else:
                self.logger.warning(f"Unknown value type: {type(raw_value)} for value: {raw_value}")
                value = 0.0
            
            # Always apply scale detection - it will handle different value ranges appropriately
            converted_value = self._apply_scale_detection(value)
            
            self.logger.debug(f"Balance parsing result: {raw_value} -> {value} -> ${converted_value:.2f}")
            return converted_value
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error parsing balance value '{raw_value}': {e}")
            return 0.0
    
    def _apply_scale_detection(self, value: float) -> float:
        """
        Apply scale detection and conversion for FreqTrade balance formatting issues
        
        Based on real FreqTrade UI data:
        - Available USDT: 293,789 (needs /1000 = $293.79) 
        - ETH/USDT:USDT: 230,354 (needs /1000 = $230.35)
        - BTC/USDT:USDT: 465,857 (needs /1000 = $465.86)
        - Total: 990 (this is $990, not 990k)
        
        Args:
            value: Raw numeric value
            
        Returns:
            float: Corrected value with proper scaling
        """
        try:
            self.logger.debug(f"Scale detection input: {value} (type: {type(value).__name__})")
            
            # Case 1: Total field shows 990 and means $990 (no conversion needed)
            if 900 <= value <= 1100:  # The total field range (990)
                self.logger.debug(f"Total field value detected: {value} -> ${value:.2f} (no scaling)")
                return value  # No conversion needed - 990 is $990
                
            # Case 2: Large values that represent micro-units needing division by 1000
            # This covers the 293789, 230354, 465857 cases
            elif 100000 <= value <= 999999:  # 100k to 999k range
                converted_value = value / 1000  # Convert to dollars
                self.logger.info(f"Large balance scale detected: {value} -> ${converted_value:.2f} (divided by 1000)")
                return converted_value
                
            # Case 3: Values in millions (likely micro-units) 
            elif value >= 1000000:
                # Try different scale factors based on magnitude
                if value >= 1000000000:  # Billion+ range (nano-units)
                    converted = value / 1000000000
                    self.logger.info(f"Nano-unit scale detected: {value} -> ${converted:.2f} (divided by 1B)")
                    return converted
                elif value >= 1000000:  # Million+ range (micro-units)
                    converted = value / 1000000  
                    self.logger.info(f"Micro-unit scale detected: {value} -> ${converted:.2f} (divided by 1M)")
                    return converted
            
            # Case 4: Values that look like they're in cents (end in 00)
            elif value > 10000 and str(int(value)).endswith('00'):
                potential_dollars = value / 100
                if 100 <= potential_dollars <= 100000:  # Reasonable account range
                    self.logger.info(f"Cent-to-dollar conversion: {value} -> ${potential_dollars:.2f} (divided by 100)")
                    return potential_dollars
            
            # Case 5: Values in reasonable dollar range - no conversion
            elif 0 <= value <= 100000:
                self.logger.debug(f"Normal range value: {value} -> ${value:.2f} (no scaling)")
                return value
            
            # Default: return as-is with warning
            self.logger.warning(f"Unknown scale for value {value}, returning as-is")
            return value
            
        except Exception as e:
            self.logger.error(f"Error in scale detection: {e}")
            return value
    
    def _calculate_total_from_currencies(self, currencies_data) -> Tuple[float, bool]:
        """
        Calculate total balance from currencies array with proper conversion
        
        Args:
            currencies_data: Array of currency data from FreqTrade API
            
        Returns:
            tuple: (total_balance, success_flag)
        """
        try:
            total_balance = 0.0
            currencies_found = 0
            
            self.logger.info(f"Calculating total from {len(currencies_data)} currencies")
            
            for currency_data in currencies_data:
                currency = currency_data.get('currency', '').upper()
                balance_raw = currency_data.get('balance', 0)
                bot_owned_raw = currency_data.get('bot_owned', 0)
                
                self.logger.info(f"Processing currency {currency}: balance={balance_raw}, bot_owned={bot_owned_raw}")
                
                # Use bot_owned if available, otherwise use balance
                value_to_use = bot_owned_raw if bot_owned_raw and bot_owned_raw != 0 else balance_raw
                
                if value_to_use and value_to_use != 0:
                    balance_value = self._parse_balance_value(value_to_use)
                    
                    # Convert to USD equivalent
                    if currency == 'USDT' or currency == 'USD':
                        # Direct USD equivalent
                        usd_value = balance_value
                    elif currency in ['BTC', 'ETH', 'SOL']:  # Major cryptos
                        # For crypto pairs like ETH/USDT, the balance often represents USDT value
                        usd_value = balance_value  # This goes through scale detection in _parse_balance_value
                    else:
                        # Unknown currency - skip or use a default rate
                        self.logger.info(f"Unknown currency {currency} with balance {balance_value}, skipping")
                        continue
                    
                    total_balance += usd_value
                    currencies_found += 1
                    self.logger.info(f"Added {currency}: {value_to_use} -> ${usd_value:.2f} (total now: ${total_balance:.2f})")
            
            success = currencies_found > 0
            if success:
                self.logger.info(f"Final calculated total from {currencies_found} currencies: ${total_balance:.2f}")
            else:
                self.logger.warning("No valid currencies found for total calculation")
            
            return total_balance, success
            
        except Exception as e:
            self.logger.error(f"Error calculating total from currencies: {e}")
            return 0.0, False
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions from FreqTrade
        
        Returns:
            List of Position objects
        """
        try:
            # Get open trades from FreqTrade
            response = self._make_request('/api/v1/status')
            
            if response is None:
                self.logger.warning("Failed to get open positions")
                return []
            
            positions = []
            trades = response if isinstance(response, list) else []
            
            for trade_data in trades:
                try:
                    # Parse trade data
                    position = self._parse_trade_to_position(trade_data)
                    if position:
                        positions.append(position)
                        
                except Exception as e:
                    self.logger.error(f"Error parsing trade data: {e}")
                    continue
            
            self.logger.debug(f"Retrieved {len(positions)} open positions")
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
    
    def _parse_trade_to_position(self, trade_data: Dict) -> Optional[Position]:
        """
        Parse FreqTrade trade data to Position object
        
        Args:
            trade_data: Raw trade data from FreqTrade API
            
        Returns:
            Position object or None if parsing fails
        """
        try:
            # Extract required fields
            trade_id = trade_data.get('trade_id')
            pair = trade_data.get('pair', '')
            
            # Convert pair format (BTC/USDT:USDT -> BTCUSDT)
            if '/' in pair:
                symbol = pair.split('/')[0] + pair.split('/')[1].split(':')[0]
            else:
                symbol = pair
            
            # Determine side from FreqTrade's is_short field (correct method)
            amount = trade_data.get('amount', 0)
            is_short = trade_data.get('is_short', False)
            side = 'short' if is_short else 'long'
            
            # Entry information
            entry_rate = float(trade_data.get('open_rate', 0))
            current_rate = float(trade_data.get('current_rate', entry_rate))
            quantity = abs(float(amount))
            
            # Entry time
            open_date = trade_data.get('open_date')
            if open_date:
                entry_time = datetime.fromisoformat(open_date.replace('Z', '+00:00'))
            else:
                entry_time = datetime.now()
            
            # PnL information
            profit_abs = float(trade_data.get('profit_abs', 0))
            profit_pct = float(trade_data.get('profit_pct', 0))
            
            # Create Position object
            position = Position(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=entry_rate,
                current_price=current_rate,
                quantity=quantity,
                entry_time=entry_time,
                pnl=profit_abs,
                pnl_percentage=profit_pct,
                is_open=True
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error parsing trade data: {e}")
            return None
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state compatible with strategy interface
        
        Returns:
            Portfolio state dict with positions and total_value
        """
        try:
            # Get open positions
            positions = self.get_open_positions()
            
            # Get balance information
            balance_response = self._make_request('/api/v1/balance')
            balance_data = balance_response or {}
            
            # Calculate total value with proper unit handling
            total_balance = 0
            available_balance = 0
            balance_found = False
            balance_source = "unknown"
            
            # COMPREHENSIVE DEBUG LOGGING
            self.logger.debug("=== BALANCE DEBUG START ===")
            self.logger.debug(f"Raw balance response keys: {list(balance_data.keys()) if balance_data else 'None'}")
            
            # Only log full response at debug level to avoid spam
            import json
            self.logger.debug(f"Full raw balance response: {json.dumps(balance_data, indent=2, default=str) if balance_data else 'None'}")
            
            # Log only important fields at debug level
            for key, value in balance_data.items():
                if key in ['total', 'total_bot', 'value_bot', 'free', 'available', 'currencies']:
                    self.logger.debug(f"Field '{key}': {value if key != 'currencies' else f'{len(value)} currencies'}")
            
            # Parse balance fields correctly - Fix: Use the correct FreqTrade fields
            # FreqTrade returns: total_bot (managed balance), total (including unmanaged), currencies with bot_owned
            
            # Priority 1: Use total_bot (this is the managed balance, should be $990)
            if 'total_bot' in balance_data:
                raw_total = balance_data['total_bot']
                total_balance = self._parse_balance_value(raw_total)  # Apply proper parsing
                balance_found = True
                balance_source = "total_bot field"
                self.logger.debug(f"Found total balance from 'total_bot' field: {raw_total} -> ${total_balance}")
            # Fallback: Use total field
            elif 'total' in balance_data:
                raw_total = balance_data['total']
                total_balance = self._parse_balance_value(raw_total)  # Apply proper parsing
                balance_found = True
                balance_source = "total field"
                self.logger.info(f"Found total balance from 'total' field: {raw_total} -> ${total_balance}")
            elif 'value_bot' in balance_data:
                raw_value = balance_data['value_bot']
                total_balance = self._parse_balance_value(raw_value)  # Apply proper parsing
                balance_found = True
                balance_source = "value_bot field"
                self.logger.info(f"Found total balance from 'value_bot' field: {raw_value} -> ${total_balance}")
            elif 'currencies' in balance_data:
                # Parse currencies and sum USDT equivalents
                total_balance, balance_found = self._calculate_total_from_currencies(balance_data['currencies'])
                balance_source = "currencies calculation"
                self.logger.info(f"Calculated total from currencies: ${total_balance}")
            
            # Get available balance (what's actually available for trading)
            if balance_found:
                # Fix: Look for USDT currency with bot_owned field (this is the available balance)
                available_balance = 0
                if 'currencies' in balance_data:
                    self.logger.debug(f"Processing {len(balance_data['currencies'])} currencies:")
                    for currency_data in balance_data['currencies']:
                        currency = currency_data.get('currency')
                        
                        if currency == 'USDT':
                            # Use bot_owned field - this is what's available for trading
                            bot_owned = currency_data.get('bot_owned', 0)
                            balance = currency_data.get('balance', 0)
                            free = currency_data.get('free', 0)
                            
                            self.logger.debug(f"USDT currency found - bot_owned: {bot_owned}, balance: {balance}, free: {free}")
                            
                            if bot_owned and bot_owned > 0:
                                available_balance = self._parse_balance_value(bot_owned)
                                self.logger.debug(f"Using USDT bot_owned as available: {bot_owned} -> ${available_balance}")
                                break
                            elif free and free > 0:
                                available_balance = self._parse_balance_value(free)
                                self.logger.info(f"Using USDT free as available: {free} -> ${available_balance}")
                                break
                            elif balance and balance > 0:
                                available_balance = self._parse_balance_value(balance)
                                self.logger.info(f"Using USDT balance as available: {balance} -> ${available_balance}")
                                break
                
                # Fallback: Try to get free balance from API response
                if available_balance == 0:
                    if 'free' in balance_data:
                        raw_free = balance_data['free']
                        available_balance = self._parse_balance_value(raw_free)
                        self.logger.info(f"Found free balance: {raw_free} -> ${available_balance}")
                    elif 'available' in balance_data:
                        raw_available = balance_data['available']
                        available_balance = self._parse_balance_value(raw_available)
                        self.logger.info(f"Found available balance: {raw_available} -> ${available_balance}")
                    else:
                        # Calculate available balance by subtracting position values
                        used_in_positions = sum(
                            abs(pos.entry_price * pos.quantity) for pos in positions
                        ) if positions else 0
                        available_balance = total_balance - used_in_positions
                        self.logger.info(f"Calculated available balance: ${total_balance} - ${used_in_positions} = ${available_balance}")
                
                # Ensure available balance is not negative
                available_balance = max(0, available_balance)
                self.logger.debug(f"Final available balance: ${available_balance}")
            
            self.logger.debug("=== BALANCE DEBUG END ===")
            
            # Use real balance if found and greater than 0
            if balance_found and total_balance > 0:
                # Calculate used balance for better display
                used_balance = max(0, total_balance - available_balance)
                
                self.logger.debug(f"Final balance calculations: Total: ${total_balance:.2f}, Available: ${available_balance:.2f}, Used: ${used_balance:.2f}, Source: {balance_source}")
                
                # Fixed: Don't convert to "k" format - display actual dollar amounts
                total_display = f"${total_balance:.2f}"
                available_display = f"${available_balance:.2f}"
                used_display = f"${used_balance:.2f}"
                
                self.logger.debug(f"Using FreqTrade balance - Available: {available_display}, Total: {total_display}, Used: {used_display} (source: {balance_source})")
            else:
                # Only use fallback if we couldn't get real balance
                self.logger.warning(f"No valid balance found (balance_found: {balance_found}, total_balance: {total_balance}), using fallback")
                total_balance = 100000  # Fallback for testing
                available_balance = 100000
                balance_source = "fallback mock"
            
            # Convert positions to strategy format
            strategy_positions = []
            for position in positions:
                strategy_positions.append({
                    'trade_id': position.trade_id,
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'quantity': position.quantity,
                    'entry_time': position.entry_time.isoformat(),
                    'pnl': position.pnl,
                    'pnl_percentage': position.pnl_percentage
                })
            
            portfolio_state = {
                'total_value': total_balance,  # Total portfolio value including positions
                'positions': strategy_positions,
                'cash': available_balance,  # FIXED: Available balance for new trades (not total)
                'timestamp': datetime.now().isoformat(),
                'balance_source': balance_source,  # Track where balance came from
                'raw_balance_data': balance_data if balance_data else {}  # For debugging
            }
            
            self.logger.debug(f"Portfolio state retrieved: {len(strategy_positions)} positions, ${total_balance:.2f} total value, ${available_balance:.2f} available cash")
            return portfolio_state
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio state: {e}")
            # Return fallback with error indicator
            return {
                'total_value': 0,  # Signal error to Strategy Runner
                'positions': [], 
                'cash': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trade_history(self, days: int = 7) -> List[Dict]:
        """
        Get recent trade history
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of completed trades
        """
        try:
            response = self._make_request('/api/v1/trades')
            
            if not response:
                return []
            
            trades = response.get('trades', [])
            
            # Filter by date if needed
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = []
            
            for trade in trades:
                open_date = trade.get('open_date')
                if open_date:
                    trade_date = datetime.fromisoformat(open_date.replace('Z', '+00:00'))
                    if trade_date >= cutoff_date:
                        recent_trades.append(trade)
            
            return recent_trades
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to FreqTrade API
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Test API call
            response = self._make_request('/api/v1/ping')
            if response and response.get('status') == 'pong':
                auth_type = "JWT" if self.access_token else "Basic"
                return True, f"Connection successful with {auth_type} authentication"
            else:
                return False, "API connection failed - ping response invalid"
                
        except Exception as e:
            return False, f"Connection error: {e}"
    
    def get_balance_debug_info(self) -> Dict[str, Any]:
        """
        Get detailed balance information for debugging balance calculation issues
        
        Returns:
            Dict: Comprehensive balance debug information
        """
        try:
            # Get raw balance response
            balance_response = self._make_request('/api/v1/balance')
            if not balance_response:
                return {'error': 'Failed to get balance response'}
            
            debug_info = {
                'raw_response': balance_response,
                'response_keys': list(balance_response.keys()),
                'parsed_balances': {},
                'total_calculations': [],
                'recommendations': []
            }
            
            # Parse different balance fields
            for key in ['total', 'value', 'balance']:
                if key in balance_response:
                    raw_value = balance_response[key]
                    parsed_value = self._parse_balance_value(raw_value)
                    debug_info['parsed_balances'][key] = {
                        'raw': raw_value,
                        'parsed': parsed_value,
                        'type': type(raw_value).__name__
                    }
            
            # Parse currencies if available
            if 'currencies' in balance_response:
                currencies = balance_response['currencies']
                debug_info['currencies_detail'] = []
                total_from_currencies, success = self._calculate_total_from_currencies(currencies)
                
                for currency_data in currencies:
                    currency = currency_data.get('currency', 'unknown')
                    balance_raw = currency_data.get('balance', 0)
                    balance_parsed = self._parse_balance_value(balance_raw)
                    
                    debug_info['currencies_detail'].append({
                        'currency': currency,
                        'raw_balance': balance_raw,
                        'parsed_balance': balance_parsed,
                        'balance_type': type(balance_raw).__name__
                    })
                
                debug_info['total_from_currencies'] = {
                    'calculated_total': total_from_currencies,
                    'calculation_success': success
                }
            
            # Add recommendations based on the data
            if 'total' in balance_response:
                total_value = debug_info['parsed_balances']['total']['parsed']
                if total_value > 100000:
                    debug_info['recommendations'].append(
                        f"Large total value ({total_value}) detected - may need scale conversion"
                    )
                elif total_value == 0:
                    debug_info['recommendations'].append("Zero total value - check balance calculation logic")
                    
            if 'currencies' in balance_response and len(balance_response['currencies']) > 1:
                debug_info['recommendations'].append("Multiple currencies detected - verify conversion logic")
            
            return debug_info
            
        except Exception as e:
            return {'error': f'Error getting balance debug info: {e}'}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get client health status
        
        Returns:
            Health status information
        """
        connection_ok, message = self.test_connection()
        
        return {
            'client': 'freqtrade_api',
            'connected': connection_ok,
            'message': message,
            'api_url': self.api_url,
            'auth_method': 'jwt_token_auth',
            'username': self.username,
            'has_access_token': bool(self.access_token),
            'has_refresh_token': bool(self.refresh_token),
            'timestamp': datetime.now().isoformat()
        }
    
    def close(self):
        """Close HTTP session"""
        if hasattr(self, 'session'):
            self.session.close()
            self.logger.info("FreqTrade API client session closed")


def create_freqtrade_client_from_config(config_manager) -> FreqTradeAPIClient:
    """
    Factory function to create FreqTrade client from configuration
    
    Args:
        config_manager: ConfigManager instance
        
    Returns:
        FreqTradeAPIClient instance
    """
    freqtrade_config = config_manager.get_freqtrade_config()
    
    if not freqtrade_config['enabled']:
        raise RuntimeError("FreqTrade integration is disabled in configuration")
    
    client = FreqTradeAPIClient(
        api_url=freqtrade_config['api_url'],
        username=freqtrade_config['username'],
        password=freqtrade_config['password'],
        timeout=freqtrade_config['timeout']
    )
    
    return client