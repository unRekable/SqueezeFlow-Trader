"""Custom exceptions for Veloce system"""


class VeloceException(Exception):
    """Base exception for all Veloce errors"""
    pass


class ConfigurationError(VeloceException):
    """Raised when configuration is invalid"""
    pass


class DataProviderError(VeloceException):
    """Raised when data provider encounters an error"""
    pass


class ConnectionError(DataProviderError):
    """Raised when connection to data source fails"""
    pass


class NoDataError(DataProviderError):
    """Raised when no data is available"""
    pass


class StrategyError(VeloceException):
    """Raised when strategy encounters an error"""
    pass


class SignalValidationError(StrategyError):
    """Raised when signal validation fails"""
    pass


class IndicatorError(StrategyError):
    """Raised when indicator calculation fails"""
    pass


class ExecutionError(VeloceException):
    """Raised when order execution fails"""
    pass


class InsufficientBalanceError(ExecutionError):
    """Raised when balance is insufficient for trade"""
    pass


class PositionError(ExecutionError):
    """Raised when position management fails"""
    pass


class BacktestError(VeloceException):
    """Raised when backtesting encounters an error"""
    pass


class DashboardError(VeloceException):
    """Raised when dashboard generation fails"""
    pass


class APIError(VeloceException):
    """Raised when API encounters an error"""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails"""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    pass