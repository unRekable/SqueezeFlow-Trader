#!/usr/bin/env python3
"""
Simple Test Strategy - No external dependencies
Just for testing FreqTrade API authentication
"""

import pandas as pd
from freqtrade.strategy import IStrategy


class SimpleTest(IStrategy):
    """Simple test strategy without external dependencies"""
    
    # Strategy metadata
    INTERFACE_VERSION = 3
    
    # Basic strategy settings
    timeframe = '5m'
    stoploss = -0.10
    minimal_roi = {"0": 0.10}
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Add minimal indicators"""
        dataframe['sma_20'] = dataframe['close'].rolling(20).mean()
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define entry conditions - very simple test"""
        dataframe.loc[
            (dataframe['close'] > dataframe['sma_20']),
            'enter_long'
        ] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Define exit conditions"""
        dataframe.loc[
            (dataframe['close'] < dataframe['sma_20']),
            'exit_long'
        ] = 1
        return dataframe