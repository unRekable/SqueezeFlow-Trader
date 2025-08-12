#!/usr/bin/env python3
"""
SqueezeFlow Optimization Framework v3.0
ADAPTED for the current system - integrates with visual validation

This version:
1. Works with the new TradingView unified dashboard
2. Integrates visual validation for self-debugging
3. Respects the actual data flow documented in the system
4. Does NOT break any existing functionality
5. Uses the adaptive learning from previous sessions
"""

import os
import sys
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
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import visual validator
from backtest.reporting.visual_validator import DashboardVisualValidator

# Import adaptive learner
from experiments.adaptive_learner import AdaptiveLearner


@dataclass
class OptimizationResult:
    """Results from a single optimization run"""
    parameter_name: str
    parameter_value: Any
    symbol: str
    date_range: Tuple[str, str]
    
    # Backtest results
    total_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Visual validation results
    dashboard_path: str = ""
    screenshot_path: str = ""
    visual_validation: Dict = field(default_factory=dict)
    
    # Timing
    backtest_duration: float = 0.0
    
    # Score from visual analysis
    optimization_score: float = 0.0
    
    def calculate_score(self) -> float:
        """Calculate optimization score based on multiple factors"""
        # Weight different metrics
        trade_score = min(self.total_trades / 10, 1.0) * 0.2  # Want some trades
        win_score = self.win_rate * 0.3  # Win rate important
        return_score = min(max(self.total_return / 5, 0), 1.0) * 0.3  # 5% return = good
        sharpe_score = min(max(self.sharpe_ratio / 2, 0), 1.0) * 0.2  # Sharpe 2.0 = good
        
        self.optimization_score = (trade_score + win_score + return_score + sharpe_score) * 100
        return self.optimization_score


class OptimizationFrameworkV3:
    """
    Optimization framework adapted for the current system
    
    Key adaptations:
    1. Uses the actual backtest engine command structure
    2. Parses results from the new dashboard format
    3. Integrates visual validation for self-debugging
    4. Works with the adaptive learner for continuity
    5. Respects the documented data flow
    """
    
    def __init__(self):
        self.setup_logging()
        
        # Paths
        self.experiments_dir = Path(__file__).parent
        self.project_root = self.experiments_dir.parent
        self.results_dir = self.project_root / "results"
        
        # Data storage
        self.optimization_data_dir = self.experiments_dir / "optimization_data_v3"
        self.optimization_data_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results_file = self.optimization_data_dir / "results.json"
        self.state_file = self.optimization_data_dir / "state.json"
        
        # Load previous results
        self.results_history = self._load_results()
        self.state = self._load_state()
        
        # Initialize components
        self.visual_validator = DashboardVisualValidator(str(self.results_dir))
        self.adaptive_learner = AdaptiveLearner()
        
        # CRITICAL: Use remote InfluxDB
        self.influx_host = "213.136.75.120"
        
        self.logger.info("🚀 Optimization Framework V3 initialized")
        self.logger.info(f"📊 Remote InfluxDB: {self.influx_host}")
        
    def setup_logging(self):
        """Setup logging for the framework"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimization_v3.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OptimizationV3')
        
    def _load_results(self) -> List[OptimizationResult]:
        """Load previous optimization results"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                return [OptimizationResult(**r) for r in data]
        return []
        
    def _load_state(self) -> Dict:
        """Load optimization state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'last_run': None,
            'current_cycle': 0,
            'best_parameters': {}
        }
        
    def _save_results(self):
        """Save optimization results"""
        data = [asdict(r) for r in self.results_history]
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    def _save_state(self):
        """Save optimization state"""
        self.state['last_run'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
            
    async def run_backtest(self, symbol: str, parameter_name: str, 
                          parameter_value: Any, date_range: Tuple[str, str]) -> OptimizationResult:
        """
        Run a single backtest with specified parameters
        
        CRITICAL: This must match the actual system's backtest command structure
        """
        start_date, end_date = date_range
        
        # Build the backtest command - MUST use remote InfluxDB and 1s timeframe
        env = os.environ.copy()
        env['INFLUX_HOST'] = self.influx_host
        env['INFLUX_PORT'] = '8086'
        
        # Set the parameter in environment
        if parameter_name == 'MIN_ENTRY_SCORE':
            env['SQUEEZEFLOW_MIN_ENTRY_SCORE'] = str(parameter_value)
        elif parameter_name == 'CVD_VOLUME_THRESHOLD':
            env['SQUEEZEFLOW_CVD_VOLUME_THRESHOLD'] = str(parameter_value)
        # Add more parameter mappings as needed
        
        # Build command - ALWAYS use 1s timeframe
        cmd = [
            'python3', 
            'backtest/engine.py',
            '--symbol', symbol,
            '--start-date', start_date,
            '--end-date', end_date,
            '--timeframe', '1s',  # ALWAYS 1s
            '--balance', '10000',
            '--leverage', '1.0',
            '--strategy', 'SqueezeFlowStrategy'
        ]
        
        self.logger.info(f"🚀 Running backtest: {symbol} with {parameter_name}={parameter_value}")
        
        # Run the backtest
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode != 0:
                self.logger.error(f"❌ Backtest failed: {result.stderr}")
                return OptimizationResult(
                    parameter_name=parameter_name,
                    parameter_value=parameter_value,
                    symbol=symbol,
                    date_range=date_range,
                    backtest_duration=duration
                )
                
            # Parse the output to extract results
            output = result.stdout
            
            # Create result object
            opt_result = OptimizationResult(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                symbol=symbol,
                date_range=date_range,
                backtest_duration=duration
            )
            
            # Extract metrics from output (parse the backtest output)
            opt_result = self._parse_backtest_output(output, opt_result)
            
            # Find the dashboard that was created
            dashboard_path = self._find_latest_dashboard()
            if dashboard_path:
                opt_result.dashboard_path = str(dashboard_path)
                
                # Run visual validation
                self.logger.info("📸 Running visual validation...")
                validation = self.visual_validator.capture_dashboard(dashboard_path)
                
                if validation.get('success'):
                    opt_result.screenshot_path = validation.get('screenshot_path', '')
                    opt_result.visual_validation = validation
                    self.logger.info(f"✅ Visual validation complete: {opt_result.screenshot_path}")
                    
                    # Extract additional metrics from visual analysis
                    opt_result = await self._analyze_screenshot(opt_result)
                    
            # Calculate optimization score
            opt_result.calculate_score()
            
            return opt_result
            
        except subprocess.TimeoutExpired:
            self.logger.error("⏱️ Backtest timed out")
            return OptimizationResult(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                symbol=symbol,
                date_range=date_range,
                backtest_duration=300.0
            )
        except Exception as e:
            self.logger.error(f"💥 Backtest error: {e}")
            return OptimizationResult(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                symbol=symbol,
                date_range=date_range,
                backtest_duration=duration if 'duration' in locals() else 0
            )
            
    def _parse_backtest_output(self, output: str, result: OptimizationResult) -> OptimizationResult:
        """Parse backtest console output to extract metrics"""
        
        lines = output.split('\n')
        
        for line in lines:
            # Look for key metrics in the output
            if 'Total trades:' in line:
                try:
                    result.total_trades = int(line.split(':')[-1].strip())
                except:
                    pass
            elif 'Win rate:' in line:
                try:
                    win_rate_str = line.split(':')[-1].strip().replace('%', '')
                    result.win_rate = float(win_rate_str) / 100
                except:
                    pass
            elif 'Total return:' in line:
                try:
                    return_str = line.split(':')[-1].strip().replace('%', '')
                    result.total_return = float(return_str)
                except:
                    pass
            elif 'Sharpe ratio:' in line:
                try:
                    result.sharpe_ratio = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'Max drawdown:' in line:
                try:
                    dd_str = line.split(':')[-1].strip().replace('%', '')
                    result.max_drawdown = float(dd_str)
                except:
                    pass
                    
        return result
        
    def _find_latest_dashboard(self) -> Optional[Path]:
        """Find the most recently created dashboard"""
        dashboards = list(self.results_dir.glob("backtest_*/dashboard.html"))
        if dashboards:
            return max(dashboards, key=lambda p: p.stat().st_mtime)
        return None
        
    async def _analyze_screenshot(self, result: OptimizationResult) -> OptimizationResult:
        """
        Analyze the screenshot to extract visual information
        This is where Claude can "see" the actual dashboard output
        """
        if not result.screenshot_path or not Path(result.screenshot_path).exists():
            return result
            
        # Here we would use visual analysis to extract:
        # - Whether charts are properly rendered
        # - Trade markers on the chart
        # - CVD patterns visible
        # - Score values shown
        
        # For now, just note that we have visual validation
        self.logger.info(f"👁️ Visual analysis available at: {result.screenshot_path}")
        self.logger.info("   Claude can now analyze this screenshot to verify results")
        
        return result
        
    async def optimize_parameter(self, parameter_name: str, test_values: List[Any],
                                symbol: str = 'ETH', date_range: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Optimize a single parameter by testing multiple values
        """
        if not date_range:
            # Use a known good date range for ETH
            date_range = ('2025-08-10', '2025-08-10')
            
        self.logger.info(f"🔧 Optimizing {parameter_name} for {symbol}")
        self.logger.info(f"📊 Testing values: {test_values}")
        
        results = []
        
        for value in test_values:
            self.logger.info(f"Testing {parameter_name}={value}...")
            
            result = await self.run_backtest(
                symbol=symbol,
                parameter_name=parameter_name,
                parameter_value=value,
                date_range=date_range
            )
            
            results.append(result)
            self.results_history.append(result)
            
            # Save after each test
            self._save_results()
            
            # Log result
            self.logger.info(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1%}, "
                           f"Return: {result.total_return:.2f}%, Score: {result.optimization_score:.1f}")
            
        # Find best value
        best_result = max(results, key=lambda r: r.optimization_score)
        
        # Update state
        if symbol not in self.state['best_parameters']:
            self.state['best_parameters'][symbol] = {}
        self.state['best_parameters'][symbol][parameter_name] = {
            'value': best_result.parameter_value,
            'score': best_result.optimization_score,
            'trades': best_result.total_trades,
            'win_rate': best_result.win_rate
        }
        self._save_state()
        
        # Record learning
        self.adaptive_learner.record_learning(
            symbol=symbol,
            concept=f"{parameter_name}_optimization",
            finding=f"Best value: {best_result.parameter_value} (score: {best_result.optimization_score:.1f})",
            confidence=0.8
        )
        
        return {
            'parameter': parameter_name,
            'best_value': best_result.parameter_value,
            'best_score': best_result.optimization_score,
            'all_results': results
        }
        
    async def run_optimization_cycle(self, symbols: List[str] = None, 
                                   parameters: Dict[str, List] = None) -> Dict:
        """
        Run a complete optimization cycle
        """
        if not symbols:
            symbols = ['ETH']  # Start with ETH which we know has data
            
        if not parameters:
            # Start with the most critical parameter
            parameters = {
                'MIN_ENTRY_SCORE': [2.0, 3.0, 4.0, 5.0, 6.0]
            }
            
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION CYCLE STARTING")
        self.logger.info("=" * 80)
        
        cycle_results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n🎯 Optimizing {symbol}...")
            cycle_results[symbol] = {}
            
            for param_name, test_values in parameters.items():
                result = await self.optimize_parameter(
                    parameter_name=param_name,
                    test_values=test_values,
                    symbol=symbol
                )
                
                cycle_results[symbol][param_name] = result
                
        # Generate report
        self._generate_report(cycle_results)
        
        # Update cycle count
        self.state['current_cycle'] += 1
        self._save_state()
        
        return cycle_results
        
    def _generate_report(self, cycle_results: Dict):
        """Generate optimization report"""
        report_path = self.optimization_data_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("OPTIMIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for symbol, params in cycle_results.items():
                f.write(f"\n{symbol} RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                for param_name, result in params.items():
                    f.write(f"\n{param_name}:\n")
                    f.write(f"  Best Value: {result['best_value']}\n")
                    f.write(f"  Best Score: {result['best_score']:.1f}\n")
                    f.write(f"  All Results:\n")
                    
                    for r in result['all_results']:
                        f.write(f"    {r.parameter_value}: Score={r.optimization_score:.1f}, "
                               f"Trades={r.total_trades}, Win={r.win_rate:.1%}\n")
                        
        self.logger.info(f"📊 Report saved to: {report_path}")
        
    def get_status(self) -> Dict:
        """Get current optimization status"""
        return {
            'last_run': self.state.get('last_run'),
            'cycles_completed': self.state.get('current_cycle', 0),
            'results_collected': len(self.results_history),
            'best_parameters': self.state.get('best_parameters', {}),
            'learnings': self.adaptive_learner.generate_status_report()
        }


async def main():
    """Main entry point for testing"""
    framework = OptimizationFrameworkV3()
    
    # Show status
    status = framework.get_status()
    print("\n📊 Current Status:")
    print(f"  Cycles completed: {status['cycles_completed']}")
    print(f"  Results collected: {status['results_collected']}")
    
    # Run a small test optimization
    print("\n🚀 Running test optimization...")
    
    results = await framework.run_optimization_cycle(
        symbols=['ETH'],
        parameters={
            'MIN_ENTRY_SCORE': [3.0, 4.0, 5.0]
        }
    )
    
    print("\n✅ Optimization complete!")
    print(f"Best parameters found: {framework.state['best_parameters']}")


if __name__ == "__main__":
    asyncio.run(main())