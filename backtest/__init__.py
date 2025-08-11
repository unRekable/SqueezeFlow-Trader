#!/usr/bin/env python3
"""
SqueezeFlow Backtest Engine - Clean Modular Architecture

This package provides a clean, professional backtest engine with complete
separation between strategy logic and execution orchestration.

Architecture:
- engine.py: Pure orchestration - loads data and executes orders
- core/: Core trading components (strategy, portfolio)
- reporting/: Reporting and visualization components

Key Principles:
- Strategy has COMPLETE authority over all calculations and trading decisions
- Engine ONLY orchestrates - loads data and executes orders
- Clean separation ensures same strategy works in backtest AND live trading
- Professional logging, visualization, and reporting
"""

from .engine import BacktestEngine
from .core import Portfolio, Position
from .reporting import BacktestLogger, BacktestVisualizer

__version__ = "2.0.0"
__author__ = "SqueezeFlow Trader Team"

__all__ = [
    "BacktestEngine",
    "Portfolio",
    "Position",
    "BacktestLogger",
    "BacktestVisualizer"
]