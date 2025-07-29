#!/usr/bin/env python3
"""
Industrial-Standard Strategy Logging Framework
Comprehensive logging system for SqueezeFlow strategy debugging and analysis
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class LogLevel(Enum):
    """Strategy-specific log levels"""
    STRATEGY_DEBUG = "STRATEGY_DEBUG"
    SIGNAL_GENERATION = "SIGNAL_GENERATION"
    DATA_VALIDATION = "DATA_VALIDATION"
    STATE_TRANSITION = "STATE_TRANSITION"
    THRESHOLD_CHECK = "THRESHOLD_CHECK"
    CVD_ANALYSIS = "CVD_ANALYSIS"
    MARKET_REGIME = "MARKET_REGIME"
    ERROR_ANALYSIS = "ERROR_ANALYSIS"


class StrategyLogger:
    """
    Industrial-standard logging framework for trading strategies
    Provides structured, comprehensive logging with multiple output channels
    """
    
    def __init__(self, strategy_name: str, backtest_session_id: str = None):
        """
        Initialize strategy logger with industrial-standard configuration
        
        Args:
            strategy_name: Name of the trading strategy
            backtest_session_id: Unique session identifier for this backtest run
        """
        self.strategy_name = strategy_name
        self.session_id = backtest_session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory structure
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configure multiple loggers for different purposes
        self._setup_loggers()
        
        # Strategy execution tracking
        self.signal_count = 0
        self.error_count = 0
        self.state_transitions = []
        
        self.info(f"🚀 Strategy Logger initialized for {strategy_name} (Session: {self.session_id})")
    
    def _setup_loggers(self):
        """Setup industrial-standard logging configuration"""
        
        # Main strategy logger
        self.logger = logging.getLogger(f"{self.strategy_name}_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for comprehensive logging
        log_file = os.path.join(self.logs_dir, f"{self.strategy_name}_{self.session_id}.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Debug file handler for detailed analysis
        debug_file = os.path.join(self.logs_dir, f"{self.strategy_name}_{self.session_id}_debug.log")
        debug_handler = logging.FileHandler(debug_file, mode='w')
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s\n'
            'Context: %(pathname)s\n'
            '---'
        )
        debug_handler.setFormatter(debug_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(debug_handler)
        
        # Create CSV logger for structured data analysis
        self._setup_csv_logger()
        
        self.log_file_path = log_file
        self.debug_file_path = debug_file
    
    def _setup_csv_logger(self):
        """Setup CSV logger for structured data analysis"""
        csv_file = os.path.join(self.logs_dir, f"{self.strategy_name}_{self.session_id}_signals.csv")
        self.csv_logger = logging.getLogger(f"{self.strategy_name}_csv_{self.session_id}")
        self.csv_logger.setLevel(logging.INFO)
        
        csv_handler = logging.FileHandler(csv_file, mode='w')
        csv_formatter = logging.Formatter('%(message)s')
        csv_handler.setFormatter(csv_formatter)
        self.csv_logger.addHandler(csv_handler)
        
        # Write CSV header
        self.csv_logger.info("timestamp,symbol,signal_type,confidence,price,state,market_regime,data_points,spot_cvd,perp_cvd,divergence,error_message")
        
        self.csv_file_path = csv_file
    
    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Log info message with optional structured data"""
        formatted_msg = self._format_message(message, extra_data)
        self.logger.info(formatted_msg)
    
    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """Log debug message with optional structured data"""
        formatted_msg = self._format_message(message, extra_data)
        self.logger.debug(formatted_msg)
    
    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """Log warning message with optional structured data"""
        formatted_msg = self._format_message(message, extra_data)
        self.logger.warning(formatted_msg)
    
    def error(self, message: str, error: Exception = None, extra_data: Dict[str, Any] = None):
        """Log error message with exception details"""
        self.error_count += 1
        formatted_msg = self._format_message(message, extra_data)
        
        if error:
            formatted_msg += f"\n🔥 Exception: {type(error).__name__}: {str(error)}"
            import traceback
            formatted_msg += f"\n📋 Traceback:\n{traceback.format_exc()}"
        
        self.logger.error(formatted_msg)
    
    def log_signal_generation(self, symbol: str, signal_type: str, confidence: float, 
                            price: float, state: str, market_regime: str = "Unknown",
                            data_points: int = 0, spot_cvd: float = 0, perp_cvd: float = 0,
                            divergence: float = 0, error_message: str = ""):
        """Log signal generation with structured data for analysis"""
        self.signal_count += 1
        timestamp = datetime.now().isoformat()
        
        # Log to CSV for structured analysis
        csv_row = f"{timestamp},{symbol},{signal_type},{confidence},{price},{state},{market_regime},{data_points},{spot_cvd},{perp_cvd},{divergence},\"{error_message}\""
        self.csv_logger.info(csv_row)
        
        # Log to main logger
        self.info(
            f"🎯 SIGNAL: {symbol} → {signal_type} (Confidence: {confidence:.2f}, Price: ${price:,.2f})",
            {
                'state': state,
                'market_regime': market_regime,
                'data_points': data_points,
                'spot_cvd': spot_cvd,
                'perp_cvd': perp_cvd,
                'divergence': divergence
            }
        )
    
    def log_state_transition(self, symbol: str, old_state: str, new_state: str, reason: str):
        """Log strategy state transitions"""
        self.state_transitions.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'old_state': old_state,
            'new_state': new_state,
            'reason': reason
        })
        
        self.info(f"🔄 STATE: {symbol} {old_state} → {new_state} (Reason: {reason})")
    
    def log_data_validation(self, symbol: str, timeframe: str, success: bool, 
                          data_points: int = 0, error_message: str = ""):
        """Log data validation results"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.debug(
            f"🔍 DATA: {symbol} {timeframe} → {status} ({data_points} points)",
            {'error': error_message if not success else None}
        )
    
    def log_cvd_analysis(self, symbol: str, timeframe: str, spot_cvd: float, 
                        perp_cvd: float, divergence: float, analysis: str):
        """Log CVD analysis details"""
        self.debug(
            f"📊 CVD: {symbol} {timeframe} → Divergence: {divergence:,.0f}",
            {
                'spot_cvd': spot_cvd,
                'perp_cvd': perp_cvd,
                'analysis': analysis
            }
        )
    
    def log_threshold_check(self, symbol: str, check_name: str, value: float, 
                          threshold: float, passed: bool, context: str = ""):
        """Log threshold validation checks"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.debug(
            f"🎚️ THRESHOLD: {symbol} {check_name} → {status} ({value:.2f} vs {threshold:.2f})",
            {'context': context}
        )
    
    def log_market_regime(self, symbol: str, timeframe: str, regime: str, 
                         confidence: float, analysis: Dict[str, Any]):
        """Log market regime detection"""
        self.debug(
            f"🏛️ REGIME: {symbol} {timeframe} → {regime} (Confidence: {confidence:.2f})",
            analysis
        )
    
    def _format_message(self, message: str, extra_data: Dict[str, Any] = None) -> str:
        """Format log message with optional structured data"""
        if not extra_data:
            return message
        
        formatted_data = []
        for key, value in extra_data.items():
            if value is not None:
                if isinstance(value, float):
                    formatted_data.append(f"{key}={value:.2f}")
                else:
                    formatted_data.append(f"{key}={value}")
        
        if formatted_data:
            return f"{message} | {', '.join(formatted_data)}"
        return message
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        return {
            'session_id': self.session_id,
            'strategy_name': self.strategy_name,
            'total_signals': self.signal_count,
            'total_errors': self.error_count,
            'state_transitions': len(self.state_transitions),
            'log_file': self.log_file_path,
            'debug_file': self.debug_file_path,
            'csv_file': self.csv_file_path,
            'latest_states': self.state_transitions[-5:] if self.state_transitions else []
        }
    
    def close(self):
        """Close all logging handlers and write session summary"""
        summary = self.get_session_summary()
        self.info(f"📋 Session Summary: {summary['total_signals']} signals, {summary['total_errors']} errors")
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        for handler in self.csv_logger.handlers[:]:
            handler.close()
            self.csv_logger.removeHandler(handler)
        
        return summary


def create_strategy_logger(strategy_name: str, session_id: str = None) -> StrategyLogger:
    """
    Factory function to create strategy logger
    
    Args:
        strategy_name: Name of the trading strategy
        session_id: Optional session identifier
        
    Returns:
        Configured StrategyLogger instance
    """
    return StrategyLogger(strategy_name, session_id)