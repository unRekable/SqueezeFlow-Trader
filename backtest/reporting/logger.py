#!/usr/bin/env python3
"""
Backtest Logger - Comprehensive logging system
Structured logging with multiple output formats and rotation
"""

import logging
import logging.handlers
import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class BacktestLogger:
    """Professional logging system for backtest operations"""
    
    def __init__(self, log_dir: str = "backtest/results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup loggers
        self._setup_main_logger()
        self._setup_trade_logger()
        self._setup_signal_logger()
        
        # Storage for structured data
        self.trades_data = []
        self.signals_data = []
        self.performance_data = []
    
    def _setup_main_logger(self):
        """Setup main application logger"""
        self.main_logger = logging.getLogger(f'backtest_{self.session_id}')
        self.main_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.main_logger.handlers.clear()
        
        # File handler with rotation
        log_file = self.log_dir / f"backtest_{self.session_id}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.main_logger.addHandler(file_handler)
        self.main_logger.addHandler(console_handler)
    
    def _setup_trade_logger(self):
        """Setup trade execution logger"""
        trades_file = self.log_dir / f"trades_{self.session_id}.csv"
        self.trades_file = open(trades_file, 'w', newline='')
        self.trades_writer = csv.writer(self.trades_file)
        
        # Write CSV header
        self.trades_writer.writerow([
            'timestamp', 'symbol', 'side', 'quantity', 'price', 
            'pnl', 'fees', 'signal_type', 'confidence', 'portfolio_value'
        ])
        self.trades_file.flush()
    
    def _setup_signal_logger(self):
        """Setup signal analysis logger"""
        signals_file = self.log_dir / f"signals_{self.session_id}.csv"
        self.signals_file = open(signals_file, 'w', newline='')
        self.signals_writer = csv.writer(self.signals_file)
        
        # Write CSV header
        self.signals_writer.writerow([
            'timestamp', 'symbol', 'signal_type', 'confidence', 
            'price', 'squeeze_score', 'cvd_divergence', 'executed'
        ])
        self.signals_file.flush()
    
    def info(self, message: str):
        """Log info message"""
        self.main_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.main_logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.main_logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.main_logger.debug(message)
    
    def log_trade(self, trade_data: Dict):
        """Log trade execution"""
        try:
            row = [
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('quantity', 0),
                trade_data.get('price', 0),
                trade_data.get('pnl', 0),
                trade_data.get('fees', 0),
                trade_data.get('signal_type', ''),
                trade_data.get('confidence', 0),
                trade_data.get('portfolio_value', 0)
            ]
            
            self.trades_writer.writerow(row)
            self.trades_file.flush()
            self.trades_data.append(trade_data)
            
            self.info(f"Trade executed: {trade_data.get('side')} {trade_data.get('quantity')} {trade_data.get('symbol')} @ ${trade_data.get('price', 0):.2f}")
            
        except Exception as e:
            self.error(f"Failed to log trade: {e}")
    
    def log_signal(self, signal_data: Dict, executed: bool = False):
        """Log trading signal"""
        try:
            row = [
                signal_data.get('timestamp', datetime.now()),
                signal_data.get('symbol', ''),
                signal_data.get('signal_type', ''),
                signal_data.get('confidence', 0),
                signal_data.get('price', 0),
                signal_data.get('squeeze_score', 0),
                signal_data.get('cvd_divergence', 0),
                executed
            ]
            
            self.signals_writer.writerow(row)
            self.signals_file.flush()
            
            signal_data['executed'] = executed
            self.signals_data.append(signal_data)
            
            if executed:
                self.info(f"Signal executed: {signal_data.get('signal_type')} for {signal_data.get('symbol')}")
            else:
                self.debug(f"Signal generated: {signal_data.get('signal_type')} for {signal_data.get('symbol')}")
                
        except Exception as e:
            self.error(f"Failed to log signal: {e}")
    
    def log_performance(self, performance_data: Dict):
        """Log performance metrics"""
        try:
            self.performance_data.append({
                'timestamp': datetime.now(),
                **performance_data
            })
            
            # Log key metrics
            self.info(f"Performance update - Total Return: {performance_data.get('total_return', 0):.2f}%")
            self.info(f"Win Rate: {performance_data.get('win_rate', 0):.1f}%, Trades: {performance_data.get('total_trades', 0)}")
            
        except Exception as e:
            self.error(f"Failed to log performance: {e}")
    
    def log_backtest_start(self, config: Dict):
        """Log backtest start configuration"""
        self.info("=" * 60)
        self.info("BACKTEST SESSION STARTED")
        self.info("=" * 60)
        self.info(f"Session ID: {self.session_id}")
        self.info(f"Symbol: {config.get('symbol', 'N/A')}")
        self.info(f"Strategy: {config.get('strategy_name', 'N/A')}")
        self.info(f"Date Range: {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}")
        self.info(f"Initial Balance: ${config.get('initial_balance', 0):,.2f}")
        self.info(f"Timeframe: {config.get('timeframe', 'N/A')}")
        self.info("-" * 60)
    
    def log_backtest_end(self, results: Dict):
        """Log backtest completion"""
        self.info("-" * 60)
        self.info("BACKTEST SESSION COMPLETED")
        self.info(f"Final Balance: ${results.get('final_balance', 0):,.2f}")
        self.info(f"Total Return: {results.get('total_return', 0):.2f}%")
        self.info(f"Total Trades: {results.get('total_trades', 0)}")
        self.info(f"Win Rate: {results.get('win_rate', 0):.1f}%")
        self.info("=" * 60)
    
    def save_session_summary(self, results: Dict):
        """Save complete session summary to JSON"""
        try:
            summary_file = self.log_dir / f"summary_{self.session_id}.json"
            
            summary = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'trades_count': len(self.trades_data),
                'signals_count': len(self.signals_data),
                'performance_snapshots': len(self.performance_data)
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.info(f"Session summary saved: {summary_file}")
            
        except Exception as e:
            self.error(f"Failed to save session summary: {e}")
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        return {
            'session_id': self.session_id,
            'trades_logged': len(self.trades_data),
            'signals_logged': len(self.signals_data),
            'performance_snapshots': len(self.performance_data),
            'log_directory': str(self.log_dir)
        }
    
    def close(self):
        """Close all file handlers"""
        try:
            if hasattr(self, 'trades_file'):
                self.trades_file.close()
            if hasattr(self, 'signals_file'):
                self.signals_file.close()
                
            # Close logger handlers
            for handler in self.main_logger.handlers[:]:
                handler.close()
                self.main_logger.removeHandler(handler)
                
        except Exception as e:
            print(f"Error closing logger: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()