#!/usr/bin/env python3
"""
PNG Plotter - Professional chart generation for backtest analysis
Creates high-quality PNG charts with matplotlib

Enhanced Features (v2.0):
- Signal categorization: Entry vs Exit signals with distinct markers and colors
- CVD divergence detection: Automatic detection of significant SPOT/FUTURES divergences
- CVD reset detection: Identify when divergence crosses back through mean
- Enhanced equity curve: Drawdown analysis, peaks/troughs, cumulative returns
- Improved timezone handling: Proper UTC timezone support throughout
- Signal quality annotations: Show signal scores and confidence levels
- Advanced legend management: Single entries per signal type, optimal positioning
- Shaded regions for extreme market conditions and drawdown periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import pytz
from typing import Dict, List, Optional, Tuple
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
            'background': '#F8F9FA',
            'entry_buy': '#00C851',     # Green for buy entries
            'entry_sell': '#FF4444',    # Red for sell entries  
            'exit': '#9E9E9E',          # Gray for exits
            'divergence_bull': '#4CAF50', # Green star for bullish divergence
            'divergence_bear': '#F44336', # Red star for bearish divergence
            'reset': '#2196F3'          # Blue dot for resets
        }
        # Set default timezone
        self.tz = pytz.UTC
    
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
    
    def create_equity_curve_chart(self, results: Dict, executed_orders: List[Dict], 
                                 output_dir: Path) -> str:
        """Create enhanced equity curve chart with drawdown analysis"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        # Build equity curve from transactions
        initial_balance = results.get('initial_balance', 10000)
        equity_curve = [initial_balance]
        dates = [datetime.now(tz=self.tz)]  # UTC timezone
        
        running_balance = initial_balance
        for order in executed_orders:
            pnl = order.get('pnl', 0)
            running_balance += pnl
            equity_curve.append(running_balance)
            timestamp = order.get('timestamp', datetime.now(tz=self.tz))
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(tz=self.tz)
            dates.append(timestamp)
        
        # Calculate returns and drawdowns
        equity_array = np.array(equity_curve)
        returns = (equity_array - initial_balance) / initial_balance * 100
        drawdowns, peaks, troughs = self._calculate_drawdowns(equity_array)
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Main equity curve
        line1 = ax1.plot(dates, equity_curve, linewidth=3, color=self.colors['primary'], label='Portfolio Value', zorder=3)
        ax1.axhline(y=initial_balance, color=self.colors['neutral'], linestyle='--', alpha=0.7, label='Initial Balance')
        
        # Highlight drawdown periods
        if len(drawdowns) > 0:
            drawdown_mask = drawdowns < -0.05  # 5% drawdown threshold
            if np.any(drawdown_mask):
                ax1.fill_between(dates, equity_curve, initial_balance, 
                               where=drawdown_mask, color=self.colors['danger'], 
                               alpha=0.2, label='Drawdown > 5%')
        
        # Add peak and trough annotations
        if len(peaks) > 0:
            peak_dates = [dates[i] for i in peaks]
            peak_values = [equity_curve[i] for i in peaks]
            ax1.scatter(peak_dates, peak_values, color='green', marker='^', s=50, 
                       alpha=0.7, zorder=4, label='Peaks')
        
        if len(troughs) > 0:
            trough_dates = [dates[i] for i in troughs]
            trough_values = [equity_curve[i] for i in troughs]
            ax1.scatter(trough_dates, trough_values, color='red', marker='v', s=50,
                       alpha=0.7, zorder=4, label='Troughs')
        
        # Secondary y-axis for returns
        ax1_right = ax1.twinx()
        ax1_right.plot(dates, returns, linewidth=2, color=self.colors['secondary'], 
                      alpha=0.7, linestyle=':', label='Cumulative Return %')
        ax1_right.set_ylabel('Cumulative Return (%)', color=self.colors['secondary'])
        ax1_right.tick_params(axis='y', labelcolor=self.colors['secondary'])
        
        # Formatting main chart
        title = f'Portfolio Equity Curve (Max DD: {max_drawdown:.1%})'
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_right.legend(loc='upper right')
        
        # Drawdown chart
        if len(drawdowns) > 0:
            ax2.fill_between(dates, drawdowns * 100, 0, color=self.colors['danger'], 
                           alpha=0.6, label='Drawdown %')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.set_title('Portfolio Drawdown')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis
        if len(dates) > 1:
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / 'equity_curve.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(chart_path.name)
    
    def create_equity_curve(self, results: Dict, executed_orders: List[Dict], 
                           output_dir: Path) -> str:
        """Legacy method - redirects to enhanced version"""
        return self.create_equity_curve_chart(results, executed_orders, output_dir)
    
    def create_price_signals_chart(self, dataset: Dict, executed_orders: List[Dict], 
                                  output_dir: Path) -> str:
        """Create enhanced price action chart with categorized trading signals"""
        
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
        ax1.plot(times, prices, linewidth=1.5, color=self.colors['primary'], label='Price', zorder=1)
        
        # Categorize and plot signals
        signal_types = {'entry_buy': [], 'entry_sell': [], 'exit_sell': [], 'exit_buy': []}
        signal_prices = {'entry_buy': [], 'entry_sell': [], 'exit_sell': [], 'exit_buy': []}
        signal_times = {'entry_buy': [], 'entry_sell': [], 'exit_sell': [], 'exit_buy': []}
        signal_scores = {'entry_buy': [], 'entry_sell': [], 'exit_sell': [], 'exit_buy': []}
        
        for order in executed_orders:
            price = order.get('price', 0)
            side = order.get('side', '')
            timestamp = order.get('timestamp', datetime.now(tz=self.tz))
            signal_type = self._get_signal_type(order)
            score = order.get('signal_score', order.get('confidence', 0))
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(tz=self.tz)
            
            if signal_type in signal_types:
                signal_types[signal_type].append(order)
                signal_prices[signal_type].append(price)
                signal_times[signal_type].append(timestamp)
                signal_scores[signal_type].append(score)
        
        # Plot each signal type with distinct markers
        legend_added = set()
        
        # Entry BUY signals - green up triangles
        if signal_times['entry_buy']:
            scatter = ax1.scatter(signal_times['entry_buy'], signal_prices['entry_buy'], 
                                color=self.colors['entry_buy'], marker='^', s=120, 
                                label='Entry BUY', zorder=5, edgecolors='white', linewidth=1)
            legend_added.add('entry_buy')
            
            # Add signal score annotations
            for i, (time, price, score) in enumerate(zip(signal_times['entry_buy'], 
                                                        signal_prices['entry_buy'], 
                                                        signal_scores['entry_buy'])):
                if score > 0:
                    ax1.annotate(f'{score:.2f}', (time, price), xytext=(5, 10), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Entry SELL signals - red down triangles  
        if signal_times['entry_sell']:
            ax1.scatter(signal_times['entry_sell'], signal_prices['entry_sell'], 
                       color=self.colors['entry_sell'], marker='v', s=120, 
                       label='Entry SELL', zorder=5, edgecolors='white', linewidth=1)
            legend_added.add('entry_sell')
            
            # Add signal score annotations
            for i, (time, price, score) in enumerate(zip(signal_times['entry_sell'], 
                                                        signal_prices['entry_sell'], 
                                                        signal_scores['entry_sell'])):
                if score > 0:
                    ax1.annotate(f'{score:.2f}', (time, price), xytext=(5, -15), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Exit signals - gray X markers
        if signal_times['exit_sell']:
            ax1.scatter(signal_times['exit_sell'], signal_prices['exit_sell'], 
                       color=self.colors['exit'], marker='x', s=100, 
                       label='Exit SELL (Close Long)', zorder=5, linewidth=2)
            legend_added.add('exit_sell')
        
        if signal_times['exit_buy']:
            ax1.scatter(signal_times['exit_buy'], signal_prices['exit_buy'], 
                       color=self.colors['exit'], marker='x', s=100, 
                       label='Exit BUY (Close Short)', zorder=5, linewidth=2)
            legend_added.add('exit_buy')
        
        ax1.set_title('Price Action & Trading Signals', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        if legend_added:
            ax1.legend(loc='upper left')
        
        # Volume subplot
        if 'volume' in ohlcv.columns:
            volumes = ohlcv['volume']
            ax2.bar(times, volumes, alpha=0.6, color=self.colors['secondary'], width=0.8)
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)
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
        """Create enhanced CVD analysis with divergence and reset detection"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
        
        spot_cvd = dataset.get('spot_cvd', pd.Series())
        futures_cvd = dataset.get('futures_cvd', pd.Series())
        cvd_divergence = dataset.get('cvd_divergence', pd.Series())
        executed_orders = dataset.get('executed_orders', [])
        
        if spot_cvd.empty and futures_cvd.empty:
            fig.suptitle('CVD Analysis - No Data Available')
            chart_path = output_dir / 'cvd_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(chart_path.name)
        
        # Use proper time index if available
        if hasattr(spot_cvd, 'index') and len(spot_cvd.index) > 0:
            times = spot_cvd.index
        elif hasattr(futures_cvd, 'index') and len(futures_cvd.index) > 0:
            times = futures_cvd.index
        else:
            max_len = max(len(spot_cvd) if not spot_cvd.empty else 0,
                         len(futures_cvd) if not futures_cvd.empty else 0)
            times = range(max_len)
        
        # Detect divergences and resets
        divergences = self._detect_divergences(spot_cvd, futures_cvd)
        resets = self._detect_resets(cvd_divergence) if not cvd_divergence.empty else []
        
        # SPOT CVD with enhanced features
        if not spot_cvd.empty:
            ax1.plot(times, spot_cvd, color='red', linewidth=2, label='SPOT CVD', alpha=0.8, zorder=2)
            
            # Add trade entry markers
            if executed_orders:
                for order in executed_orders:
                    timestamp = order.get('timestamp', datetime.now(tz=self.tz))
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except:
                            continue
                    
                    # Find closest time index
                    if hasattr(times, '__getitem__'):
                        try:
                            # Try to match timestamp to data
                            if timestamp in times:
                                idx = times.get_loc(timestamp)
                                cvd_value = spot_cvd.iloc[idx] if idx < len(spot_cvd) else 0
                                ax1.axvline(x=timestamp, color='blue', alpha=0.5, linestyle=':', linewidth=1)
                                ax1.annotate(f'Trade: {order.get("side", "?").upper()}', 
                                           (timestamp, cvd_value), 
                                           xytext=(10, 10), textcoords='offset points', 
                                           fontsize=8, alpha=0.7, rotation=90)
                        except:
                            pass
        
        ax1.set_ylabel('SPOT CVD')
        ax1.set_title('SPOT CVD Analysis with Trade Entries', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # FUTURES CVD with enhanced features  
        if not futures_cvd.empty:
            ax2.plot(times, futures_cvd, color='orange', linewidth=2, label='FUTURES CVD', alpha=0.8, zorder=2)
            
            # Add trade entry markers
            if executed_orders:
                for order in executed_orders:
                    timestamp = order.get('timestamp', datetime.now(tz=self.tz))
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except:
                            continue
                    
                    if hasattr(times, '__getitem__'):
                        try:
                            if timestamp in times:
                                ax2.axvline(x=timestamp, color='blue', alpha=0.5, linestyle=':', linewidth=1)
                        except:
                            pass
        
        ax2.set_ylabel('FUTURES CVD')
        ax2.set_title('FUTURES CVD Analysis with Trade Entries', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Enhanced CVD Divergence chart
        if not cvd_divergence.empty:
            # Main divergence line
            ax3.plot(times, cvd_divergence, color='purple', linewidth=2, 
                    label='CVD Divergence', alpha=0.8, zorder=2)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=1)
            
            # Calculate statistics for divergence detection
            divergence_mean = cvd_divergence.mean()
            divergence_std = cvd_divergence.std()
            upper_threshold = divergence_mean + 2 * divergence_std
            lower_threshold = divergence_mean - 2 * divergence_std
            
            # Add threshold lines
            ax3.axhline(y=upper_threshold, color=self.colors['divergence_bear'], 
                       linestyle='--', alpha=0.5, label='Bearish Threshold (+2σ)')
            ax3.axhline(y=lower_threshold, color=self.colors['divergence_bull'], 
                       linestyle='--', alpha=0.5, label='Bullish Threshold (-2σ)')
            
            # Shade extreme divergence periods
            extreme_bullish = cvd_divergence < lower_threshold
            extreme_bearish = cvd_divergence > upper_threshold
            
            if np.any(extreme_bullish):
                ax3.fill_between(times, cvd_divergence, lower_threshold, 
                               where=extreme_bullish, color=self.colors['divergence_bull'], 
                               alpha=0.2, label='Extreme Bullish Divergence')
            
            if np.any(extreme_bearish):
                ax3.fill_between(times, upper_threshold, cvd_divergence, 
                               where=extreme_bearish, color=self.colors['divergence_bear'], 
                               alpha=0.2, label='Extreme Bearish Divergence')
            
            # Add divergence markers
            if divergences['bullish']:
                bull_times = [times[i] for i in divergences['bullish'] if i < len(times)]
                bull_values = [cvd_divergence.iloc[i] for i in divergences['bullish'] if i < len(cvd_divergence)]
                if bull_times and bull_values:
                    ax3.scatter(bull_times, bull_values, color=self.colors['divergence_bull'], 
                              marker='*', s=200, label='Bullish Divergence', zorder=5, 
                              edgecolors='white', linewidth=1)
            
            if divergences['bearish']:
                bear_times = [times[i] for i in divergences['bearish'] if i < len(times)]
                bear_values = [cvd_divergence.iloc[i] for i in divergences['bearish'] if i < len(cvd_divergence)]
                if bear_times and bear_values:
                    ax3.scatter(bear_times, bear_values, color=self.colors['divergence_bear'], 
                              marker='*', s=200, label='Bearish Divergence', zorder=5,
                              edgecolors='white', linewidth=1)
            
            # Add reset markers
            if resets:
                reset_times = [times[i] for i in resets if i < len(times)]
                reset_values = [cvd_divergence.iloc[i] for i in resets if i < len(cvd_divergence)]
                if reset_times and reset_values:
                    ax3.scatter(reset_times, reset_values, color=self.colors['reset'], 
                              marker='o', s=60, label='CVD Reset', zorder=5, alpha=0.8)
        
        ax3.set_ylabel('CVD Divergence')
        ax3.set_xlabel('Time')
        ax3.set_title('CVD Divergence Analysis (SPOT - FUTURES)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        fig.suptitle('Enhanced CVD Analysis Dashboard', fontsize=16, fontweight='bold')
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
    
    def _detect_divergences(self, spot_cvd: pd.Series, futures_cvd: pd.Series) -> Dict[str, List[int]]:
        """Detect significant CVD divergences between SPOT and FUTURES"""
        divergences = {'bullish': [], 'bearish': []}
        
        if spot_cvd.empty or futures_cvd.empty:
            return divergences
        
        # Ensure same length
        min_len = min(len(spot_cvd), len(futures_cvd))
        spot_data = spot_cvd.iloc[:min_len]
        futures_data = futures_cvd.iloc[:min_len]
        
        # Calculate divergence
        divergence = spot_data - futures_data
        
        if len(divergence) < 10:  # Need minimum data points
            return divergences
        
        # Calculate rolling statistics
        window = min(20, len(divergence) // 4)
        rolling_mean = divergence.rolling(window=window, center=True).mean()
        rolling_std = divergence.rolling(window=window, center=True).std()
        
        # Detect significant divergences (>2 standard deviations)
        for i in range(window, len(divergence) - window):
            if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_std.iloc[i]):
                continue
            
            current_div = divergence.iloc[i]
            mean_val = rolling_mean.iloc[i]
            std_val = rolling_std.iloc[i]
            
            if abs(current_div - mean_val) > 2 * std_val:
                if current_div > mean_val:  # SPOT > FUTURES
                    divergences['bullish'].append(i)
                else:  # FUTURES > SPOT
                    divergences['bearish'].append(i)
        
        # Remove duplicates that are too close together
        divergences['bullish'] = self._filter_close_signals(divergences['bullish'], min_distance=5)
        divergences['bearish'] = self._filter_close_signals(divergences['bearish'], min_distance=5)
        
        return divergences
    
    def _detect_resets(self, cvd_divergence: pd.Series) -> List[int]:
        """Detect CVD reset points where divergence crosses back through mean"""
        resets = []
        
        if cvd_divergence.empty or len(cvd_divergence) < 10:
            return resets
        
        # Calculate rolling mean
        window = min(20, len(cvd_divergence) // 4)
        rolling_mean = cvd_divergence.rolling(window=window, center=True).mean()
        
        # Detect crossings through mean
        for i in range(1, len(cvd_divergence) - 1):
            if pd.isna(rolling_mean.iloc[i]):
                continue
            
            prev_val = cvd_divergence.iloc[i-1] - rolling_mean.iloc[i-1] if not pd.isna(rolling_mean.iloc[i-1]) else 0
            curr_val = cvd_divergence.iloc[i] - rolling_mean.iloc[i]
            
            # Check for zero crossing (reset)
            if (prev_val > 0 and curr_val <= 0) or (prev_val < 0 and curr_val >= 0):
                resets.append(i)
        
        # Filter close resets
        resets = self._filter_close_signals(resets, min_distance=10)
        
        return resets
    
    def _calculate_drawdowns(self, equity_curve: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
        """Calculate portfolio drawdowns and identify peaks/troughs"""
        if len(equity_curve) < 2:
            return np.array([]), [], []
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown as percentage from peak
        drawdowns = (equity_curve - running_max) / running_max
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(1, len(equity_curve) - 1):
            # Peak: higher than both neighbors and creates new running max
            if (equity_curve[i] > equity_curve[i-1] and 
                equity_curve[i] > equity_curve[i+1] and
                equity_curve[i] == running_max[i]):
                peaks.append(i)
            
            # Trough: lower than both neighbors
            elif (equity_curve[i] < equity_curve[i-1] and 
                  equity_curve[i] < equity_curve[i+1]):
                troughs.append(i)
        
        return drawdowns, peaks, troughs
    
    def _get_signal_type(self, order: Dict) -> str:
        """Categorize signal as entry or exit type"""
        side = order.get('side', '').upper()
        order_type = order.get('type', '').upper()
        action = order.get('action', '').lower()
        
        # Check for explicit action field
        if 'entry' in action:
            return 'entry_buy' if side == 'BUY' else 'entry_sell'
        elif 'exit' in action or 'close' in action:
            return 'exit_sell' if side == 'SELL' else 'exit_buy'
        
        # Check order type
        if 'entry' in order_type:
            return 'entry_buy' if side == 'BUY' else 'entry_sell'
        elif 'exit' in order_type:
            return 'exit_sell' if side == 'SELL' else 'exit_buy'
        
        # Default categorization based on side
        # Assume BUY = entry long or exit short
        # Assume SELL = entry short or exit long
        if side == 'BUY':
            return 'entry_buy'  # Default to entry
        elif side == 'SELL':
            return 'entry_sell'  # Default to entry
        else:
            return 'entry_buy'  # Fallback
    
    def _filter_close_signals(self, signals: List[int], min_distance: int = 5) -> List[int]:
        """Filter out signals that are too close together"""
        if not signals:
            return signals
        
        filtered = [signals[0]]
        
        for signal in signals[1:]:
            if signal - filtered[-1] >= min_distance:
                filtered.append(signal)
        
        return filtered
    
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
        timestamps = [s.get('timestamp', datetime.now(tz=self.tz)) for s in signals]
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