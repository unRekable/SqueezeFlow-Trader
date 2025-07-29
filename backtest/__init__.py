"""
SqueezeFlow Backtest Package
Industrial-standard backtesting framework for cryptocurrency trading strategies

Architecture:
- core/: Core functionality (portfolio, fees, strategy interfaces)
- visualization/: Plotting and visualization systems
- strategy_logging/: Comprehensive logging framework
- analysis/: CVD and market analysis tools
- strategies/: Trading strategy implementations
- results/: All generated outputs (images, logs, debug files)
- tests/: Unit tests for all components

Main Entry Point: engine.py
"""

__version__ = "2.0.0"
__author__ = "SqueezeFlow Trading System"

# Import main components for easy access
from .engine import SqueezeFlowBacktestEngine

# Core components
from .core.portfolio import PortfolioManager, Position, PositionType, RiskLimits
from .core.fees import FeeCalculator, TradingCosts
from .core.strategy import BaseStrategy, TradingSignal, SignalStrength

# Visualization
from .visualization.plotter import BacktestPlotter

__all__ = [
    "SqueezeFlowBacktestEngine",
    "PortfolioManager", "Position", "PositionType", "RiskLimits",
    "FeeCalculator", "TradingCosts", 
    "BaseStrategy", "TradingSignal", "SignalStrength",
    "BacktestPlotter"
]