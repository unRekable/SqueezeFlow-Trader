"""
Unit Tests for Backtest Framework
Comprehensive test suite for all backtest components

Test Modules:
- test_portfolio: Portfolio management and risk limit tests
- test_fees: Fee calculation and trading cost tests  
- test_strategies: Strategy interface and implementation tests
- test_plotter: Visualization system tests
- test_engine: Integration tests for backtest engine

Usage:
    # Run all tests
    python -m pytest backtest/tests/
    
    # Run specific test module
    python -m pytest backtest/tests/test_portfolio.py
    
    # Run with coverage
    python -m pytest backtest/tests/ --cov=backtest
"""

import os
import sys

# Add backtest directory to path for testing
backtest_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)

__all__ = []