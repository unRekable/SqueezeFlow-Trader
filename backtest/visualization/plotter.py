#!/usr/bin/env python3
"""
Backtest Plotter - Comprehensive plotting system for trading results
Extracted from monolithic engine.py for clean modular architecture
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Optional, Any


class BacktestPlotter:
    """
    Comprehensive plotting system for backtest results
    Creates multi-panel plots showing price action, CVD analysis, and performance metrics
    """
    
    def __init__(self, save_directory: str = None):
        # Use the new results/images directory by default
        if save_directory is None:
            # Get the path relative to the backtest directory
            backtest_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_directory = os.path.join(backtest_dir, "results", "images")
        self.save_directory = save_directory
        self.logger = logging.getLogger('BacktestPlotter')
        
        # Debug and display options
        self.show_full_range = False
        self.requested_start_date = None
        self.requested_end_date = None
        self.debug_mode = False
        
        # Ensure save directory exists
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Plot styling configuration
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Setup consistent plot styling"""
        plt.style.use('default')
        
        # Configure matplotlib for better plots
        plt.rcParams.update({
            'figure.figsize': (18, 12),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })
    
    def create_comprehensive_plot(self, symbol: str, historical_data: Dict,
                                 trades: List[Dict], filename: str = None,
                                 portfolio_metrics: Dict = None) -> str:
        """
        Create comprehensive plot exactly like final_optimized_strategy.png
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            historical_data: Dict with 'price', 'spot_cvd', 'perp_cvd' DataFrames
            trades: List of trade dictionaries
            filename: Output filename (auto-generated if None)
            portfolio_metrics: Optional portfolio performance metrics
            
        Returns:
            Path to saved plot file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{symbol.lower()}_backtest_results_{timestamp}.png'
            
            # Create figure with 4 subplots - FIXED layout
            fig = plt.figure(figsize=(18, 12))
            
            # FIXED: Use GridSpec for proper layout control  
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(3, 2, height_ratios=[2, 1, 1], figure=fig)
            
            ax1 = fig.add_subplot(gs[0, :])    # Top full width - price chart
            ax2 = fig.add_subplot(gs[1, 0])    # Middle left - spot CVD
            ax3 = fig.add_subplot(gs[1, 1])    # Middle right - perp CVD  
            ax4 = fig.add_subplot(gs[2, 0])    # Bottom left - performance stats
            ax5 = fig.add_subplot(gs[2, 1])    # Bottom right - cumulative P&L
            
            # Prepare data
            price_data = historical_data['price']
            spot_cvd = historical_data['spot_cvd'] / 1_000_000  # Convert to millions
            perp_cvd = historical_data['perp_cvd'] / 1_000_000  # Convert to millions
            
            # Filter out NaN values for plotting
            valid_mask = price_data['price'].notna() & spot_cvd.notna() & perp_cvd.notna()
            
            if not valid_mask.any():
                self.logger.warning(f"No valid data points for plotting {symbol}")
                return ""
            
            # Get time range for plotting
            plot_mask = valid_mask
            
            if self.show_full_range and self.requested_start_date and self.requested_end_date:
                # Show full requested date range
                if self.debug_mode:
                    self.logger.info(f"🔍 DEBUG: Showing full requested range: {self.requested_start_date.date()} to {self.requested_end_date.date()}")
                
                time_mask = (price_data.index >= self.requested_start_date) & (price_data.index <= self.requested_end_date)
                plot_mask = valid_mask & time_mask
                
                # Add visual indicators for data gaps
                self._add_data_gap_indicators = True
                
            elif trades:
                # Default behavior: focus around trades
                trade_times = []
                for t in trades:
                    if isinstance(t['entry_time'], str):
                        trade_times.append(pd.to_datetime(t['entry_time']))
                    else:
                        trade_times.append(t['entry_time'])
                    if isinstance(t['exit_time'], str):
                        trade_times.append(pd.to_datetime(t['exit_time']))
                    else:
                        trade_times.append(t['exit_time'])
                
                if trade_times:
                    start_time = min(trade_times) - pd.Timedelta(hours=4)
                    end_time = max(trade_times) + pd.Timedelta(hours=4)
                    
                    # Filter data for plotting
                    time_mask = (price_data.index >= start_time) & (price_data.index <= end_time)
                    plot_mask = valid_mask & time_mask
            
            # Get plot data
            plot_times = price_data.index[plot_mask]
            plot_prices = price_data['price'][plot_mask]
            plot_spot_cvd = spot_cvd[plot_mask]
            plot_perp_cvd = perp_cvd[plot_mask]
            
            # Plot 1: PRICE WITH TRADES
            self._plot_price_with_trades(ax1, symbol, plot_times, plot_prices, trades)
            
            # Plot 2: SPOT CVD
            self._plot_cvd_data(ax2, "Spot Market CVD", plot_times, plot_spot_cvd, 
                               trades, 'blue', 'red')
            
            # Plot 3: PERP CVD  
            self._plot_cvd_data(ax3, "Perpetual Market CVD", plot_times, plot_perp_cvd,
                               trades, 'red', 'blue')
            
            # Plot 4: PERFORMANCE STATS
            self._plot_performance_stats(ax4, trades, portfolio_metrics)
            
            # Plot 5: CUMULATIVE P&L
            self._plot_cumulative_pnl(ax5, trades)
            
            # Format x-axis for time plots
            for ax in [ax2, ax3, ax5]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.tick_params(axis='x', rotation=45, labelsize=8)
            
            plt.tight_layout()
            
            # Save plot
            filepath = os.path.join(self.save_directory, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"📊 Plot saved: {filepath}")
            
            # Show plot (comment out for batch processing)
            plt.show()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive plot for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _plot_price_with_trades(self, ax, symbol: str, plot_times, plot_prices, trades: List[Dict]):
        """Plot price chart with trade markers and annotations"""
        # Main price line
        ax.plot(plot_times, plot_prices, color='black', linewidth=2, 
               label=f'{symbol} Price', alpha=0.9)
        
        # Mark trades
        for i, trade in enumerate(trades):
            if trade['symbol'] == symbol:
                entry_time = pd.to_datetime(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time']
                exit_time = pd.to_datetime(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                side = trade['side']
                pnl_pct = trade.get('pnl_pct', 0.0)
                
                # Trade line (green if profit, red if loss)
                line_color = 'green' if pnl_pct > 0 else 'red'
                ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                       color=line_color, linewidth=4, alpha=0.8, zorder=3)
                
                # Entry marker
                entry_color = 'green' if side == 'long' else 'red'
                ax.scatter(entry_time, entry_price, color=entry_color, s=200, 
                          marker='^' if side == 'long' else 'v', zorder=5,
                          edgecolors='black', linewidth=2)
                
                # Exit marker  
                exit_color = 'red' if side == 'long' else 'green'
                ax.scatter(exit_time, exit_price, color=exit_color, s=200,
                          marker='v' if side == 'long' else '^', zorder=5,
                          edgecolors='black', linewidth=2)
                
                # Trade annotation
                self._add_trade_annotation(ax, trade, i+1, plot_prices)
        
        ax.set_title(f'Enhanced SqueezeFlow: {symbol} Multi-Timeframe Strategy', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{symbol} Price ($)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _add_trade_annotation(self, ax, trade: Dict, trade_number: int, plot_prices):
        """Add trade annotation with details"""
        entry_time = pd.to_datetime(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time']
        exit_time = pd.to_datetime(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        side = trade['side']
        pnl_pct = trade.get('pnl_pct', 0.0)
        
        # Position annotation at appropriate height
        mid_time = entry_time + (exit_time - entry_time) / 2
        price_range = plot_prices.max() - plot_prices.min()
        mid_price = max(entry_price, exit_price) + price_range * 0.02
        
        # Get strategy info if available
        strategy_info = "ENHANCED"
        if 'winning_pattern' in trade and trade['winning_pattern']:
            strategy_info = "WINNING_PATTERN"
        elif 'strategy_type' in trade:
            strategy_info = trade['strategy_type']
        
        trade_text = f"T{trade_number}: {side.upper()} {pnl_pct:+.1f}%\n{strategy_info}"
        
        ax.annotate(trade_text, xy=(mid_time, mid_price), ha='center',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='lightgreen' if pnl_pct > 0 else 'lightcoral',
                           alpha=0.9, edgecolor='black', linewidth=1))
    
    def _plot_cvd_data(self, ax, title: str, plot_times, cvd_data, trades: List[Dict],
                      line_color: str, marker_color: str):
        """Plot CVD data with divergence markers"""
        ax.plot(plot_times, cvd_data, color=line_color, linewidth=2, 
               label=title.split()[0] + ' CVD')
        
        # Mark divergences at trade entry points
        for trade in trades:
            entry_time = pd.to_datetime(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time']
            
            # Find closest CVD value at trade time
            if len(plot_times) > 0:
                time_diffs = np.abs(plot_times - entry_time)
                if len(time_diffs) > 0:
                    closest_idx = time_diffs.argmin()
                    if closest_idx < len(cvd_data):
                        cvd_value = cvd_data.iloc[closest_idx]
                        ax.scatter(entry_time, cvd_value, color=marker_color, s=100, 
                                  marker='*', zorder=10, edgecolors='black', linewidth=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('CVD (Million $)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_performance_stats(self, ax, trades: List[Dict], portfolio_metrics: Dict = None):
        """Plot performance statistics box"""
        if not trades:
            ax.text(0.5, 0.5, 'No trades executed', transform=ax.transAxes,
                   fontsize=14, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('pnl_pct', 0) > 0)
        total_pnl = sum(t.get('pnl_pct', 0) for t in trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        best_trade = max(t.get('pnl_pct', 0) for t in trades) if trades else 0
        worst_trade = min(t.get('pnl_pct', 0) for t in trades) if trades else 0
        
        # Calculate average duration
        avg_duration = 0
        if trades and 'duration_minutes' in trades[0]:
            avg_duration = sum(t.get('duration_minutes', 0) for t in trades) / total_trades / 60
        elif trades and 'holding_hours' in trades[0]:
            avg_duration = sum(t.get('holding_hours', 0) for t in trades) / total_trades
        
        # Count strategy types
        enhanced_trades = total_trades
        winning_pattern_trades = 0
        for t in trades:
            if t.get('winning_pattern', False) or t.get('strategy_type') == 'WINNING_PATTERN':
                winning_pattern_trades += 1
                enhanced_trades -= 1
        
        # Portfolio metrics
        portfolio_text = ""
        if portfolio_metrics:
            portfolio_text = f"\nPORTFOLIO:\nBalance: ${portfolio_metrics.get('current_balance', 0):,.0f}\nReturn: {portfolio_metrics.get('total_return', 0):.1f}%"
        
        stats_text = f"""ENHANCED RESULTS:
Total P&L: {total_pnl:+.2f}%
Win Rate: {win_rate:.1f}%
Total Trades: {total_trades}
Avg Duration: {avg_duration:.1f}h
Best: {best_trade:+.1f}%
Worst: {worst_trade:+.1f}%

TRADE BREAKDOWN:
Winning Patterns: {winning_pattern_trades} trades
Enhanced: {enhanced_trades} trades

STRATEGY EVOLUTION:
✓ Enhanced sensitivity
✓ Real market fees
✓ Multi-timeframe analysis
✓ CVD divergence signals{portfolio_text}"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                        alpha=0.9, edgecolor='darkgreen', linewidth=2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_cumulative_pnl(self, ax, trades: List[Dict]):
        """Plot cumulative P&L curve"""
        if not trades:
            ax.text(0.5, 0.5, 'No trades to plot', transform=ax.transAxes,
                   fontsize=12, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return
        
        # Sort trades by exit time and calculate cumulative P&L
        sorted_trades = sorted(trades, key=lambda x: pd.to_datetime(x['exit_time']) if isinstance(x['exit_time'], str) else x['exit_time'])
        
        cumulative_times = []
        cumulative_pnl = []
        running_total = 0
        
        for trade in sorted_trades:
            exit_time = pd.to_datetime(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time']
            cumulative_times.append(exit_time)
            running_total += trade.get('pnl_pct', 0)
            cumulative_pnl.append(running_total)
        
        if cumulative_times and cumulative_pnl:
            ax.plot(cumulative_times, cumulative_pnl, 
                   color='darkgreen', linewidth=3, marker='o', markersize=6)
            ax.fill_between(cumulative_times, cumulative_pnl, 
                           alpha=0.3, color='lightgreen')
        
        ax.set_title('Cumulative P&L: Enhanced Strategy Performance', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative P&L (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def create_equity_curve_plot(self, portfolio_manager, filename: str = None) -> str:
        """
        Create standalone equity curve plot
        
        Args:
            portfolio_manager: PortfolioManager instance with completed trades
            filename: Output filename
            
        Returns:
            Path to saved plot file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'equity_curve_{timestamp}.png'
            
            # Get equity curve data
            equity_df = portfolio_manager.get_equity_curve()
            
            if equity_df.empty:
                self.logger.warning("No equity data to plot")
                return ""
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Equity curve
            ax1.plot(equity_df['timestamp'], equity_df['balance'], 
                    color='darkblue', linewidth=2, label='Portfolio Balance')
            ax1.axhline(y=portfolio_manager.initial_balance, color='gray', 
                       linestyle='--', alpha=0.7, label='Initial Balance')
            
            ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Balance ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Trade P&L bars (FIXED)
            colors = ['green' if pnl > 0 else 'red' for pnl in equity_df['trade_pnl']]
            # Calculate appropriate bar width based on time spacing
            if len(equity_df) > 1:
                time_diff = (equity_df['timestamp'].iloc[1] - equity_df['timestamp'].iloc[0]).total_seconds() / 86400  # Convert to days
                bar_width = time_diff * 0.8  # 80% of time spacing
            else:
                bar_width = 0.01  # Default small width
            ax2.bar(equity_df['timestamp'], equity_df['trade_pnl'], 
                   color=colors, alpha=0.7, width=bar_width)
            
            ax2.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Trade P&L ($)', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            filepath = os.path.join(self.save_directory, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"📈 Equity curve saved: {filepath}")
            
            plt.show()
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating equity curve plot: {e}")
            return ""
    
    def create_performance_summary_plot(self, metrics: Dict, filename: str = None) -> str:
        """
        Create performance summary dashboard
        
        Args:
            metrics: Performance metrics dictionary
            filename: Output filename
            
        Returns:
            Path to saved plot file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'performance_summary_{timestamp}.png'
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            
            # Plot 1: Key metrics
            key_metrics = [
                f"Total Return: {metrics.get('total_return', 0):.2f}%",
                f"Win Rate: {metrics.get('win_rate', 0):.1f}%",
                f"Total Trades: {metrics.get('total_trades', 0)}",
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
                f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%",
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}"
            ]
            
            ax1.text(0.05, 0.95, '\n'.join(key_metrics), transform=ax1.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            ax1.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            
            # Plot 2: Win/Loss distribution (FIXED)
            wins = metrics.get('winning_trades', 0)
            losses = metrics.get('losing_trades', 0)
            
            if wins > 0 or losses > 0:
                # Ensure we have positive values for pie chart
                pie_values = [max(wins, 0.1), max(losses, 0.1)]  # Minimum 0.1 to prevent empty slices
                pie_labels = [f'Wins ({wins})', f'Losses ({losses})']
                ax2.pie(pie_values, labels=pie_labels, 
                       colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
                ax2.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No trade data available', transform=ax2.transAxes,
                        fontsize=12, ha='center', va='center')
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
            
            # Plot 3: Return distribution (if available)
            if 'trade_returns' in metrics and metrics['trade_returns']:
                ax3.hist(metrics['trade_returns'], bins=20, alpha=0.7, 
                        color='blue', edgecolor='black')
                ax3.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Return (%)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Monthly returns (if available)
            if 'monthly_returns' in metrics and metrics['monthly_returns']:
                months = list(metrics['monthly_returns'].keys())
                returns = list(metrics['monthly_returns'].values())
                colors = ['green' if r > 0 else 'red' for r in returns]
                
                ax4.bar(months, returns, color=colors, alpha=0.7)
                ax4.set_title('Monthly Returns', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Return (%)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filepath = os.path.join(self.save_directory, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"📊 Performance summary saved: {filepath}")
            
            plt.show()
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating performance summary plot: {e}")
            return ""


# Global plotter instance for easy import
backtest_plotter = BacktestPlotter()


def create_comprehensive_plot(symbol: str, historical_data: Dict, trades: List[Dict],
                             filename: str = None, portfolio_metrics: Dict = None) -> str:
    """
    Convenience function to create comprehensive plot
    
    Args:
        symbol: Trading symbol
        historical_data: Historical price and CVD data
        trades: List of trade dictionaries
        filename: Output filename
        portfolio_metrics: Optional portfolio metrics
        
    Returns:
        Path to saved plot file
    """
    return backtest_plotter.create_comprehensive_plot(
        symbol, historical_data, trades, filename, portfolio_metrics
    )


def create_equity_curve(portfolio_manager, filename: str = None) -> str:
    """
    Convenience function to create equity curve plot
    
    Args:
        portfolio_manager: PortfolioManager instance
        filename: Output filename
        
    Returns:
        Path to saved plot file
    """
    return backtest_plotter.create_equity_curve_plot(portfolio_manager, filename)


if __name__ == "__main__":
    # Test the plotting system
    print("🎨 Backtest Plotter Testing")
    print("=" * 40)
    
    # This would be called from the backtest engine with real data
    print("Plotter initialized successfully!")
    print(f"Save directory: {backtest_plotter.save_directory}")
    print("Ready to create comprehensive trading plots!")