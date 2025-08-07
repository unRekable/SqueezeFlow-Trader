#!/usr/bin/env python3
"""
SqueezeFlow Data Pipeline - Unified data coordination system

This package provides a clean, unified data pipeline that coordinates
all data loading, processing, and validation for the SqueezeFlow Trader.

Components:
- loaders/: Data loading components (InfluxDB, symbol/market discovery)
- processors/: Data processing components (CVD calculation, exchange mapping)
- pipeline.py: Main coordination layer for unified data access

Key Benefits:
- Eliminates data inconsistencies across components
- Provides quality validation and caching
- Clean separation between raw data loading and processing
- Supports both backtest and live trading workflows
"""

from .pipeline import DataPipeline, create_data_pipeline

__version__ = "1.0.0"
__author__ = "SqueezeFlow Trader Team"

__all__ = [
    "DataPipeline",
    "create_data_pipeline"
]