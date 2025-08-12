"""System constants for Veloce"""

# Version
VERSION = "1.0.0"
SYSTEM_NAME = "Veloce"

# Exchanges
SUPPORTED_EXCHANGES = [
    "BINANCE",
    "BINANCE_FUTURES",
    "BYBIT",
    "OKX",
    "DERIBIT",
    "COINBASE",
    "KRAKEN"
]

# Markets
DEFAULT_QUOTE_CURRENCY = "USDT"
SPOT_SUFFIX = "usdt"
PERP_SUFFIX = "usdt-perp"

# Timeframes (in seconds)
TIMEFRAME_SECONDS = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}

# Indicator defaults
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_MULT = 2.0
DEFAULT_KC_PERIOD = 20
DEFAULT_KC_MULT = 1.5
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9

# Strategy phases
PHASE_NAMES = [
    "Context & Squeeze",
    "Divergence Analysis",
    "Reset Detection",
    "Scoring",
    "Exit Management"
]

# Risk management
MIN_POSITION_SIZE = 0.001  # 0.1% minimum
MAX_POSITION_SIZE = 0.5    # 50% maximum
DEFAULT_STOP_LOSS = 0.02   # 2%
DEFAULT_TAKE_PROFIT = 2.0  # 2:1 RR

# Performance thresholds
MIN_WIN_RATE = 0.3         # 30% minimum win rate
MIN_PROFIT_FACTOR = 1.2    # Minimum profit factor
MAX_DRAWDOWN = 0.20        # 20% max drawdown

# Cache settings
DEFAULT_CACHE_TTL = 60     # 60 seconds
MAX_CACHE_SIZE = 1000      # Maximum cache entries

# API limits
MAX_API_REQUESTS_PER_MINUTE = 100
MAX_WEBSOCKET_CONNECTIONS = 10
API_TIMEOUT = 30           # seconds

# Logging
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT_JSON = '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
LOG_FORMAT_TEXT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Dashboard
CHART_COLORS = {
    "green": "#26a69a",
    "red": "#ef5350",
    "blue": "#42a5f5",
    "orange": "#ffa726",
    "purple": "#ab47bc",
    "gray": "#78909c"
}

CHART_THEMES = {
    "dark": {
        "background": "#1e1e1e",
        "text": "#e0e0e0",
        "grid": "#424242"
    },
    "light": {
        "background": "#ffffff",
        "text": "#212121",
        "grid": "#e0e0e0"
    }
}

# Error messages
ERROR_MESSAGES = {
    "NO_DATA": "No data available for the specified parameters",
    "CONNECTION_FAILED": "Failed to connect to data source",
    "INVALID_CONFIG": "Invalid configuration provided",
    "INSUFFICIENT_BALANCE": "Insufficient balance for trade",
    "SIGNAL_VALIDATION_FAILED": "Signal validation failed",
    "STRATEGY_ERROR": "Strategy execution error"
}