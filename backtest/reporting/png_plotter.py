#!/usr/bin/env python3
"""
PNG Plotter - Professional chart generation for backtest analysis
Creates high-quality PNG charts with matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class PNGPlotter:
    """Professional PNG chart generator for backtest results"""
    
    def __init__(self):
        # Set matplotlib style
        plt.style.use('default')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'neutral': '#6C757D',
            'background': '#F8F9FA'
        }
    
    def create_all_charts(self, results: Dict, dataset: Dict, 
                         executed_orders: List[Dict], output_dir: Path) -> Dict[str, str]:
        """
        Create all backtest charts
        
        Returns:
            Dict mapping chart names to filenames
        """
        charts = {}
        
        try:
            # 1. Equity curve
            charts['equity_curve'] = self.create_equity_curve(
                results, executed_orders, output_dir
            )
            
            # 2. Price action with signals
            charts['price_signals'] = self.create_price_signals_chart(
                dataset, executed_orders, output_dir
            )
            
            # 3. CVD analysis
            charts['cvd_analysis'] = self.create_cvd_analysis_chart(
                dataset, output_dir
            )
            
            # 4. Performance metrics
            charts['performance'] = self.create_performance_chart(
                results, output_dir
            )
            
            # 5. Trade analysis
            charts['trades'] = self.create_trade_analysis_chart(
                executed_orders, output_dir
            )
            
        except Exception as e:
            print(f"Warning: Chart generation error: {e}")
        
        return charts
    
    def create_equity_curve(self, results: Dict, executed_orders: List[Dict], 
                           output_dir: Path) -> str:
        """Create equity curve chart"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Build equity curve from transactions
        initial_balance = results.get('initial_balance', 10000)
        equity_curve = [initial_balance]
        dates = [datetime.now()]  # Placeholder
        
        running_balance = initial_balance
        for order in executed_orders:
            pnl = order.get('pnl', 0)
            running_balance += pnl
            equity_curve.append(running_balance)
            dates.append(order.get('timestamp', datetime.now()))
        
        # Plot equity curve
        ax.plot(dates, equity_curve, linewidth=2, color=self.colors['primary'], label='Portfolio Value')
        ax.axhline(y=initial_balance, color=self.colors['neutral'], linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Formatting
        ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        if len(dates) > 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'equity_curve.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)
    
    def create_price_signals_chart(self, dataset: Dict, executed_orders: List[Dict], 
                                  output_dir: Path) -> str:
        """Create price action with trading signals"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        ohlcv = dataset.get('ohlcv', pd.DataFrame())
        if ohlcv.empty:
            # Create placeholder chart
            ax1.text(0.5, 0.5, 'No price data available', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Price Action & Trading Signals')
            chart_path = output_dir / 'price_signals.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(chart_path.name)
        
        # Extract price data
        if 'close' in ohlcv.columns:
            prices = ohlcv['close']
            times = ohlcv.index if hasattr(ohlcv.index, '__getitem__') else range(len(prices))
        else:
            prices = ohlcv.iloc[:, 4] if len(ohlcv.columns) > 4 else ohlcv.iloc[:, -1]
            times = range(len(prices))
        
        # Plot price
        ax1.plot(times, prices, linewidth=1.5, color=self.colors['primary'], label='Price')
        
        # Plot trading signals
        for order in executed_orders:
            price = order.get('price', 0)
            side = order.get('side', '')
            timestamp = order.get('timestamp', datetime.now())
            
            if side.upper() == 'BUY':
                ax1.scatter(timestamp, price, color=self.colors['success'], 
                           marker='^', s=100, label='Buy Signal', zorder=5)
            elif side.upper() == 'SELL':
                ax1.scatter(timestamp, price, color=self.colors['danger'], 
                           marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Price Action & Trading Signals', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Volume subplot
        if 'volume' in ohlcv.columns:
            volumes = ohlcv['volume']
            ax2.bar(times, volumes, alpha=0.6, color=self.colors['secondary'])
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Time')
        else:
            ax2.text(0.5, 0.5, 'No volume data available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'price_signals.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)
    
    def create_cvd_analysis_chart(self, dataset: Dict, output_dir: Path) -> str:
        """Create CVD analysis visualization"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        spot_cvd = dataset.get('spot_cvd', pd.Series())
        futures_cvd = dataset.get('futures_cvd', pd.Series())
        cvd_divergence = dataset.get('cvd_divergence', pd.Series())
        
        if spot_cvd.empty and futures_cvd.empty:
            fig.suptitle('CVD Analysis - No Data Available')
            chart_path = output_dir / 'cvd_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(chart_path.name)
        
        times = range(len(spot_cvd)) if not spot_cvd.empty else range(len(futures_cvd))
        
        # SPOT CVD
        if not spot_cvd.empty:
            ax1.plot(times, spot_cvd, color='red', linewidth=2, label='SPOT CVD', alpha=0.8)
        ax1.set_ylabel('SPOT CVD')
        ax1.set_title('SPOT CVD (Cumulative Volume Delta)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # FUTURES CVD
        if not futures_cvd.empty:
            ax2.plot(times, futures_cvd, color='orange', linewidth=2, label='FUTURES CVD', alpha=0.8)
        ax2.set_ylabel('FUTURES CVD')
        ax2.set_title('FUTURES CVD (Cumulative Volume Delta)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # CVD Divergence
        if not cvd_divergence.empty:
            ax3.plot(times, cvd_divergence, color='purple', linewidth=2, label='CVD Divergence', alpha=0.8)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('CVD Divergence')
        ax3.set_xlabel('Time')
        ax3.set_title('CVD Divergence (SPOT - FUTURES)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        fig.suptitle('CVD Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'cvd_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)
    
    def create_performance_chart(self, results: Dict, output_dir: Path) -> str:
        """Create performance metrics visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract metrics
        total_return = results.get('total_return', 0)
        win_rate = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)
        winning_trades = results.get('winning_trades', 0)
        losing_trades = results.get('losing_trades', 0)
        
        # 1. Return bar chart
        ax1.bar(['Total Return'], [total_return], 
                color=self.colors['success'] if total_return > 0 else self.colors['danger'])
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Total Return')
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss pie chart
        if total_trades > 0:
            sizes = [winning_trades, losing_trades]
            labels = ['Winning Trades', 'Losing Trades']
            colors = [self.colors['success'], self.colors['danger']]
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Trade Distribution')
        
        # 3. Win rate gauge
        ax3.bar(['Win Rate'], [win_rate], color=self.colors['primary'])
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Win Rate')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade count
        ax4.bar(['Total Trades'], [total_trades], color=self.colors['secondary'])
        ax4.set_ylabel('Number of Trades')
        ax4.set_title('Total Trades')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'performance.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)
    
    def create_trade_analysis_chart(self, executed_orders: List[Dict], output_dir: Path) -> str:
        """Create trade analysis chart"""
        
        if not executed_orders:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trades executed', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Trade Analysis')
            
            chart_path = output_dir / 'trades.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(chart_path.name)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract PnL data
        pnls = [order.get('pnl', 0) for order in executed_orders]
        trade_numbers = list(range(1, len(pnls) + 1))
        
        # 1. PnL per trade
        colors = [self.colors['success'] if pnl > 0 else self.colors['danger'] for pnl in pnls]
        ax1.bar(trade_numbers, pnls, color=colors)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('PnL ($)')
        ax1.set_title('PnL per Trade')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative PnL
        cumulative_pnl = np.cumsum(pnls)
        ax2.plot(trade_numbers, cumulative_pnl, linewidth=2, color=self.colors['primary'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative PnL ($)')
        ax2.set_title('Cumulative PnL')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'trades.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)
    
    def create_squeeze_signals_chart(self, signals: List[Dict], output_dir: Path) -> str:
        """Create squeeze signals visualization"""
        
        if not signals:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No signals generated', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Squeeze Signals Analysis')
            
            chart_path = output_dir / 'squeeze_signals.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(chart_path.name)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Extract signal data
        timestamps = [s.get('timestamp', datetime.now()) for s in signals]
        squeeze_scores = [s.get('squeeze_score', 0) for s in signals]
        confidences = [s.get('confidence', 0) for s in signals]
        
        # Plot squeeze scores
        ax1.plot(timestamps, squeeze_scores, linewidth=2, color=self.colors['primary'], label='Squeeze Score')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axhline(y=0.3, color=self.colors['success'], linestyle='--', alpha=0.7, label='Long Threshold')
        ax1.axhline(y=-0.3, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Short Threshold')
        
        ax1.set_ylabel('Squeeze Score')
        ax1.set_title('Squeeze Score Timeline')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot confidence levels
        ax2.bar(timestamps, confidences, alpha=0.6, color=self.colors['secondary'], label='Signal Confidence')
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time')
        ax2.set_title('Signal Confidence Levels')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        fig.suptitle('Squeeze Signals Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'squeeze_signals.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)