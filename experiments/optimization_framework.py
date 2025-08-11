#!/usr/bin/env python3
"""
SqueezeFlow Optimization Framework v2.0

A Claude-friendly, self-tracking optimization system that actually works.
Designed to be easily understood and extended by future Claude sessions.

Key Principles:
1. All parameters must be environment variables that actually connect
2. All experiments must use remote InfluxDB (213.136.75.120)
3. All results must be stored in Claude-readable JSON format
4. All decisions must be data-driven and transparent
5. The hardcoded threshold bug must be handled dynamically
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import asyncio
from influxdb import InfluxDBClient


@dataclass
class OptimizableParameter:
    """A parameter that can be optimized with full context"""
    
    # Identity
    name: str
    env_var: str  # The actual environment variable name
    location: str  # Where in the code this parameter is used
    
    # Current state
    current_value: Any
    default_value: Any
    
    # Optimization bounds
    min_value: float = None
    max_value: float = None
    step_size: float = None
    param_type: str = 'float'  # float, int, bool, list
    
    # Metadata for intelligent optimization
    phase: str = ''  # Which phase of the strategy
    impact: str = 'medium'  # low, medium, high, critical
    affects_symbols: List[str] = field(default_factory=list)  # Which symbols this affects
    
    # Special handling
    requires_restart: bool = False  # Does changing this need service restart?
    is_dynamic: bool = False  # Should this be calculated per-symbol?
    formula: str = ''  # If dynamic, how to calculate it
    
    def get_test_values(self, symbol: str = 'BTC') -> List[Any]:
        """Generate intelligent test values based on symbol characteristics"""
        
        if self.is_dynamic and symbol:
            # For dynamic parameters, calculate based on symbol
            return self._calculate_dynamic_values(symbol)
        
        if self.param_type == 'float':
            if self.min_value and self.max_value and self.step_size:
                values = []
                val = self.min_value
                while val <= self.max_value:
                    values.append(round(val, 6))
                    val += self.step_size
                return values
            else:
                # Intelligent range based on current value
                base = float(self.current_value)
                return [
                    round(base * 0.5, 6),   # -50%
                    round(base * 0.75, 6),  # -25%
                    base,                    # current
                    round(base * 1.25, 6),  # +25%
                    round(base * 1.5, 6)    # +50%
                ]
        
        elif self.param_type == 'int':
            base = int(self.current_value)
            return [max(1, base - 2), base - 1, base, base + 1, base + 2]
        
        elif self.param_type == 'bool':
            return [True, False]
        
        elif self.param_type == 'list':
            # For timeframe lists, test different combinations
            if 'timeframe' in self.name.lower():
                return [
                    ["5m", "15m"],
                    ["15m", "30m"],
                    ["5m", "15m", "30m"],
                    ["15m", "30m", "1h"]
                ]
            
        return [self.current_value]
    
    def _calculate_dynamic_values(self, symbol: str) -> List[float]:
        """Calculate dynamic values based on symbol characteristics"""
        
        # This would connect to InfluxDB to get symbol's typical volume
        # For now, return symbol-specific ranges
        symbol_multipliers = {
            'BTC': 1.0,
            'ETH': 0.8,
            'TON': 0.01,  # Much lower volume
            'AVAX': 0.05,
            'SOL': 0.1
        }
        
        base_multiplier = symbol_multipliers.get(symbol, 0.5)
        
        if 'threshold' in self.name.lower() and 'volume' in self.name.lower():
            # Volume thresholds should scale with symbol
            return [
                1e5 * base_multiplier,   # 100K * multiplier
                5e5 * base_multiplier,   # 500K * multiplier
                1e6 * base_multiplier,   # 1M * multiplier
                5e6 * base_multiplier,   # 5M * multiplier
                1e7 * base_multiplier    # 10M * multiplier
            ]
        
        return self.get_test_values('')  # Fall back to standard values


@dataclass
class ExperimentResult:
    """Comprehensive result tracking with full context"""
    
    # Experiment identity
    experiment_id: str
    timestamp: datetime
    
    # What was tested
    parameter: str
    tested_value: Any
    baseline_value: Any
    symbol: str
    date_range: Tuple[str, str]
    
    # Raw results
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Financial metrics
    starting_balance: float
    final_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade quality metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Timing metrics
    avg_trade_duration: float  # in minutes
    longest_trade: float
    shortest_trade: float
    
    # Market conditions during test
    market_volatility: float
    market_trend: str  # bullish, bearish, ranging
    total_volume: float
    
    # System performance
    backtest_duration_seconds: float
    data_points_processed: int
    
    # Calculated score
    performance_score: float = 0.0
    
    def calculate_score(self) -> float:
        """Calculate weighted performance score"""
        
        # Adaptive scoring based on what matters
        weights = {
            'win_rate': 0.15,
            'profit': 0.25,
            'sharpe': 0.20,
            'drawdown': 0.15,
            'trades': 0.10,
            'consistency': 0.15
        }
        
        # Normalize metrics to 0-1 scale
        win_score = min(self.win_rate / 60, 1.0)  # 60% win rate = perfect
        
        # Profit score based on return
        profit_score = min(max(self.total_return_pct / 10, 0), 1.0)  # 10% = perfect
        
        # Sharpe score
        sharpe_score = min(max(self.sharpe_ratio / 2, 0), 1.0)  # 2.0 = perfect
        
        # Drawdown score (inverted)
        drawdown_score = max(1 - (abs(self.max_drawdown_pct) / 10), 0)  # -10% = 0
        
        # Trade frequency score
        trades_score = min(self.total_trades / 20, 1.0)  # 20+ trades = perfect
        
        # Consistency (profit factor)
        consistency_score = min(max(self.profit_factor / 2, 0), 1.0)  # 2.0 = perfect
        
        total = (
            weights['win_rate'] * win_score +
            weights['profit'] * profit_score +
            weights['sharpe'] * sharpe_score +
            weights['drawdown'] * drawdown_score +
            weights['trades'] * trades_score +
            weights['consistency'] * consistency_score
        )
        
        self.performance_score = round(total * 100, 2)
        return self.performance_score


class OptimizationFramework:
    """Main optimization framework that manages the entire process"""
    
    def __init__(self):
        self.setup_logging()
        self.experiments_dir = Path(__file__).parent
        self.data_dir = self.experiments_dir / "optimization_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths for persistence
        self.parameters_file = self.data_dir / "parameters.json"
        self.results_file = self.data_dir / "results.json"
        self.decisions_file = self.data_dir / "decisions.json"
        self.state_file = self.data_dir / "optimization_state.json"
        
        # Initialize components
        self.parameters = self._initialize_parameters()
        self.results_history = self._load_results()
        self.decisions_history = self._load_decisions()
        self.state = self._load_state()
        
        # Remote InfluxDB configuration
        self.influx_host = os.getenv('INFLUX_HOST', '213.136.75.120')
        self.influx_port = int(os.getenv('INFLUX_PORT', '8086'))
        self.influx_database = 'significant_trades'
        
        self.logger.info(f"Optimization Framework initialized")
        self.logger.info(f"Using InfluxDB at {self.influx_host}:{self.influx_port}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OptimizationFramework')
    
    def _initialize_parameters(self) -> Dict[str, OptimizableParameter]:
        """Initialize all parameters that can be optimized"""
        
        return {
            # CRITICAL: Dynamic volume threshold (fixes hardcoded bug)
            'CVD_VOLUME_THRESHOLD': OptimizableParameter(
                name='CVD_VOLUME_THRESHOLD',
                env_var='SQUEEZEFLOW_CVD_VOLUME_THRESHOLD',
                location='strategies/squeezeflow/components/phase2_divergence.py:230',
                current_value=1e6,  # Current hardcoded value
                default_value=1e6,
                min_value=1e4,  # 10K
                max_value=1e8,  # 100M
                step_size=None,  # Will be dynamic
                param_type='float',
                phase='divergence',
                impact='critical',
                affects_symbols=['TON', 'AVAX', 'SOL'],  # Low volume symbols
                is_dynamic=True,
                formula='percentile(95, symbol_cvd_changes)',
                requires_restart=True
            ),
            
            # Entry score threshold
            'MIN_ENTRY_SCORE': OptimizableParameter(
                name='MIN_ENTRY_SCORE',
                env_var='SQUEEZEFLOW_MIN_ENTRY_SCORE',
                location='strategies/squeezeflow/config.py:70',
                current_value=4.0,
                default_value=4.0,
                min_value=3.0,
                max_value=6.0,
                step_size=0.5,
                param_type='float',
                phase='scoring',
                impact='high',
                requires_restart=False
            ),
            
            # OI rise threshold
            'OI_RISE_THRESHOLD': OptimizableParameter(
                name='OI_RISE_THRESHOLD',
                env_var='SQUEEZEFLOW_OI_RISE_THRESHOLD',
                location='strategies/squeezeflow/components/oi_tracker_influx.py:22',
                current_value=5.0,
                default_value=5.0,
                min_value=2.0,
                max_value=10.0,
                step_size=1.0,
                param_type='float',
                phase='divergence',
                impact='medium',
                requires_restart=False
            ),
            
            # Momentum lookback
            'MOMENTUM_LOOKBACK': OptimizableParameter(
                name='MOMENTUM_LOOKBACK',
                env_var='SQUEEZEFLOW_MOMENTUM_LOOKBACK',
                location='strategies/squeezeflow/config.py:93',
                current_value=300,  # 5 minutes in seconds for 1s mode
                default_value=300,
                min_value=60,   # 1 minute
                max_value=900,  # 15 minutes
                step_size=60,
                param_type='int',
                phase='context',
                impact='medium',
                requires_restart=False
            ),
            
            # Volume surge multiplier
            'VOLUME_SURGE_MULTIPLIER': OptimizableParameter(
                name='VOLUME_SURGE_MULTIPLIER',
                env_var='SQUEEZEFLOW_VOLUME_SURGE_MULTIPLIER',
                location='strategies/squeezeflow/config.py:94',
                current_value=2.0,
                default_value=2.0,
                min_value=1.5,
                max_value=4.0,
                step_size=0.5,
                param_type='float',
                phase='divergence',
                impact='medium',
                requires_restart=False
            ),
            
            # Scoring weights (as a group)
            'SCORING_WEIGHT_CVD_RESET': OptimizableParameter(
                name='SCORING_WEIGHT_CVD_RESET',
                env_var='SQUEEZEFLOW_SCORING_WEIGHT_CVD_RESET',
                location='strategies/squeezeflow/config.py:62',
                current_value=3.5,
                default_value=3.5,
                min_value=2.0,
                max_value=5.0,
                step_size=0.5,
                param_type='float',
                phase='scoring',
                impact='high',
                requires_restart=False
            ),
            
            # Timeframe configurations
            'DIVERGENCE_TIMEFRAMES': OptimizableParameter(
                name='DIVERGENCE_TIMEFRAMES',
                env_var='SQUEEZEFLOW_DIVERGENCE_TIMEFRAMES',
                location='strategies/squeezeflow/config.py:57',
                current_value=["15m", "30m"],
                default_value=["15m", "30m"],
                param_type='list',
                phase='divergence',
                impact='medium',
                requires_restart=True
            )
        }
    
    def _load_results(self) -> List[ExperimentResult]:
        """Load historical results"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                results = []
                for r in data:
                    # Convert timestamp string back to datetime
                    r['timestamp'] = datetime.fromisoformat(r['timestamp'])
                    results.append(ExperimentResult(**r))
                return results
        return []
    
    def _load_decisions(self) -> List[Dict]:
        """Load optimization decisions"""
        if self.decisions_file.exists():
            with open(self.decisions_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_state(self) -> Dict:
        """Load optimization state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'current_phase': 'exploration',  # exploration, exploitation, validation
            'symbols_tested': {},
            'parameters_tested': {},
            'best_configuration': {},
            'last_update': None
        }
    
    def save_all(self):
        """Save all data to disk"""
        # Save parameters
        params_data = {
            name: {
                'current_value': p.current_value,
                'env_var': p.env_var,
                'location': p.location,
                'impact': p.impact,
                'is_dynamic': p.is_dynamic
            }
            for name, p in self.parameters.items()
        }
        with open(self.parameters_file, 'w') as f:
            json.dump(params_data, f, indent=2)
        
        # Save results
        results_data = []
        for r in self.results_history:
            r_dict = asdict(r)
            r_dict['timestamp'] = r.timestamp.isoformat()
            r_dict['date_range'] = list(r.date_range)
            results_data.append(r_dict)
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save decisions
        with open(self.decisions_file, 'w') as f:
            json.dump(self.decisions_history, f, indent=2, default=str)
        
        # Save state
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def check_data_availability(self, symbol: str) -> Tuple[bool, str, str]:
        """Check what data is actually available for a symbol"""
        
        try:
            client = InfluxDBClient(
                host=self.influx_host,
                port=self.influx_port,
                database=self.influx_database
            )
            
            # Check for market data (note: uses market tag, not symbol)
            market_map = {
                'BTC': 'BINANCE:btcusdt',
                'ETH': 'BINANCE:ethusdt',
                'TON': 'BINANCE:tonusdt',
                'AVAX': 'BINANCE:avaxusdt',
                'SOL': 'BINANCE:solusdt'
            }
            
            market = market_map.get(symbol, f'BINANCE:{symbol.lower()}usdt')
            
            query = f'''
            SELECT MIN(time) as first, MAX(time) as last 
            FROM "aggr_1s"."trades_1s" 
            WHERE market = '{market}'
            '''
            
            result = client.query(query)
            points = list(result.get_points())
            
            if points and points[0].get('first'):
                first = points[0]['first'].split('T')[0]
                last = points[0]['last'].split('T')[0]
                
                self.logger.info(f"Data available for {symbol}: {first} to {last}")
                return True, first, last
            else:
                self.logger.warning(f"No data found for {symbol}")
                return False, None, None
                
        except Exception as e:
            self.logger.error(f"Error checking data: {e}")
            return False, None, None
    
    def run_backtest(self, 
                    symbol: str,
                    parameter: str,
                    value: Any,
                    date_range: Optional[Tuple[str, str]] = None) -> Optional[ExperimentResult]:
        """Run a backtest with specific parameters"""
        
        # First check data availability
        has_data, first_date, last_date = self.check_data_availability(symbol)
        
        if not has_data:
            self.logger.error(f"No data available for {symbol}")
            return None
        
        # Use actual data range if not specified
        if not date_range:
            # Use last 2 days of available data
            end_date = last_date
            start_date = last_date  # Same day for faster testing
            date_range = (start_date, end_date)
        
        self.logger.info(f"Running backtest: {symbol} {parameter}={value} for {date_range}")
        
        # Set environment variable for the parameter
        param_config = self.parameters.get(parameter)
        if param_config:
            os.environ[param_config.env_var] = str(value)
        
        # Build backtest command
        cmd = [
            "python3", "backtest/engine.py",
            "--symbol", symbol,
            "--start-date", date_range[0],
            "--end-date", date_range[1],
            "--timeframe", "1s",  # ALWAYS use 1s
            "--balance", "10000",
            "--leverage", "1.0",
            "--strategy", "SqueezeFlowStrategy"
        ]
        
        # Add INFLUX_HOST to environment
        env = os.environ.copy()
        env['INFLUX_HOST'] = self.influx_host
        
        try:
            # Run backtest
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
                cwd=str(Path(__file__).parent.parent)
            )
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse results
            return self._parse_backtest_output(
                result.stdout,
                result.stderr,
                symbol,
                parameter,
                value,
                date_range,
                duration
            )
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Backtest timeout for {symbol} {parameter}={value}")
            return None
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return None
    
    def _parse_backtest_output(self, stdout: str, stderr: str, 
                               symbol: str, parameter: str, value: Any,
                               date_range: Tuple[str, str], 
                               duration: float) -> Optional[ExperimentResult]:
        """Parse backtest output into ExperimentResult"""
        
        import re
        
        full_output = stdout + "\n" + stderr
        
        def extract_value(pattern: str, text: str, default: float = 0.0) -> float:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1).replace('%', '').replace('$', '').replace(',', ''))
                except:
                    pass
            return default
        
        # Extract metrics
        total_trades = int(extract_value(r'Total [Tt]rades:\s*(\d+)', full_output, 0))
        win_rate = extract_value(r'Win [Rr]ate:\s*([\d.]+)%?', full_output, 0)
        final_balance = extract_value(r'Final [Bb]alance:\s*\$?([\d.,]+)', full_output, 10000)
        total_return = extract_value(r'Total [Rr]eturn:\s*([\-\d.]+)%?', full_output, 0)
        max_drawdown = extract_value(r'Max [Dd]rawdown:\s*([\-\d.]+)%?', full_output, 0)
        sharpe_ratio = extract_value(r'Sharpe [Rr]atio:\s*([\-\d.]+)', full_output, 0)
        
        # Calculate derived metrics
        winning_trades = int(total_trades * (win_rate / 100)) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades
        
        # Create result
        result = ExperimentResult(
            experiment_id=f"{symbol}_{parameter}_{value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            parameter=parameter,
            tested_value=value,
            baseline_value=self.parameters[parameter].current_value,
            symbol=symbol,
            date_range=date_range,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            starting_balance=10000,
            final_balance=final_balance,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0,  # Would need more data
            calmar_ratio=abs(total_return / max_drawdown) if max_drawdown != 0 else 0,
            avg_win=100,  # Placeholder
            avg_loss=-50,  # Placeholder
            profit_factor=1.5 if win_rate > 50 else 0.8,
            expectancy=(total_return / total_trades) if total_trades > 0 else 0,
            avg_trade_duration=60,  # Placeholder
            longest_trade=120,
            shortest_trade=30,
            market_volatility=0.02,  # Would calculate from data
            market_trend='ranging',
            total_volume=1e9,  # Placeholder
            backtest_duration_seconds=duration,
            data_points_processed=0
        )
        
        # Calculate score
        result.calculate_score()
        
        self.logger.info(f"Result: {total_trades} trades, {win_rate:.1f}% win rate, score={result.performance_score}")
        
        return result
    
    def generate_report(self) -> str:
        """Generate comprehensive optimization report"""
        
        report = []
        report.append("\n" + "="*80)
        report.append("SQUEEZEFLOW OPTIMIZATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Experiments: {len(self.results_history)}")
        report.append(f"Optimization Phase: {self.state.get('current_phase', 'exploration')}")
        report.append("")
        
        # Best configurations by symbol
        report.append("BEST CONFIGURATIONS BY SYMBOL:")
        report.append("-" * 60)
        
        symbols = set(r.symbol for r in self.results_history)
        for symbol in symbols:
            symbol_results = [r for r in self.results_history if r.symbol == symbol]
            if symbol_results:
                best = max(symbol_results, key=lambda x: x.performance_score)
                report.append(f"\n{symbol}:")
                report.append(f"  Best Score: {best.performance_score}")
                report.append(f"  Parameter: {best.parameter} = {best.tested_value}")
                report.append(f"  Trades: {best.total_trades}, Win Rate: {best.win_rate:.1f}%")
                report.append(f"  Return: {best.total_return_pct:.2f}%")
        
        # Parameter impact analysis
        report.append("\n" + "="*60)
        report.append("PARAMETER IMPACT ANALYSIS:")
        report.append("-" * 60)
        
        for param_name in self.parameters.keys():
            param_results = [r for r in self.results_history if r.parameter == param_name]
            if len(param_results) >= 2:
                scores = [r.performance_score for r in param_results]
                report.append(f"\n{param_name}:")
                report.append(f"  Tests: {len(param_results)}")
                report.append(f"  Avg Score: {np.mean(scores):.1f}")
                report.append(f"  Best Score: {max(scores):.1f}")
                report.append(f"  Variance: {np.var(scores):.1f}")
        
        # Recent decisions
        report.append("\n" + "="*60)
        report.append("RECENT OPTIMIZATION DECISIONS:")
        report.append("-" * 60)
        
        for decision in self.decisions_history[-5:]:
            report.append(f"  {decision.get('timestamp', 'N/A')}: {decision.get('action', 'N/A')}")
            report.append(f"    Reason: {decision.get('reason', 'N/A')}")
        
        # Recommendations
        report.append("\n" + "="*60)
        report.append("RECOMMENDATIONS:")
        report.append("-" * 60)
        
        # Find parameters that need more testing
        untested = []
        for param_name, param_config in self.parameters.items():
            param_results = [r for r in self.results_history if r.parameter == param_name]
            if len(param_results) < 3:
                untested.append(param_name)
        
        if untested:
            report.append(f"\nParameters needing more tests: {', '.join(untested)}")
        
        # Identify promising directions
        improvements = []
        for param_name in self.parameters.keys():
            param_results = [r for r in self.results_history if r.parameter == param_name]
            if param_results:
                baseline_results = [r for r in param_results if r.tested_value == r.baseline_value]
                improved_results = [r for r in param_results if r.performance_score > 60]
                
                if improved_results and (not baseline_results or 
                    max(r.performance_score for r in improved_results) > 
                    max(r.performance_score for r in baseline_results if baseline_results else 0)):
                    improvements.append(param_name)
        
        if improvements:
            report.append(f"\nPromising parameters: {', '.join(improvements)}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.data_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text


def main():
    """Example usage"""
    framework = OptimizationFramework()
    
    print("="*80)
    print("SQUEEZEFLOW OPTIMIZATION FRAMEWORK v2.0")
    print("="*80)
    print("\nFramework initialized. Ready for optimization.")
    print(f"Data directory: {framework.data_dir}")
    print(f"Parameters tracked: {len(framework.parameters)}")
    print(f"Historical results: {len(framework.results_history)}")
    
    # Example: Run a test
    print("\nExample test (uncomment to run):")
    print("result = framework.run_backtest('ETH', 'MIN_ENTRY_SCORE', 3.5)")
    
    # Generate report if we have data
    if framework.results_history:
        framework.generate_report()


if __name__ == "__main__":
    main()