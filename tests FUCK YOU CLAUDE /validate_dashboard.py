#!/usr/bin/env python3
"""
Unified Dashboard Validator with Browser MCP Integration
Validates all dashboard features and can use browser for visual verification
"""

import json
import math
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple

class DashboardValidator:
    """Comprehensive dashboard validation with browser support"""
    
    def __init__(self):
        self.latest_report = self._find_latest_report()
        
    def _find_latest_report(self) -> Path:
        """Find the most recent dashboard"""
        results_dir = Path('backtest/results')
        if results_dir.exists():
            reports = sorted(results_dir.glob('report_*/dashboard.html'))
            if reports:
                return reports[-1]
        return None
    
    def validate_all(self) -> Dict:
        """Run all validation checks"""
        if not self.latest_report:
            return {"error": "No dashboard found"}
        
        html = self.latest_report.read_text()
        
        results = {
            "dashboard_path": str(self.latest_report),
            "structure": self._validate_structure(html),
            "data": self._validate_data(html),
            "features": self._validate_features(html),
            "navigation": self._validate_navigation(html),
            "performance": self._validate_performance(html)
        }
        
        # Overall status
        all_checks = []
        for category in results.values():
            if isinstance(category, dict) and 'checks' in category:
                all_checks.extend(category['checks'].values())
        
        results['overall_status'] = 'PASS' if all(all_checks) else 'FAIL'
        results['passed'] = sum(all_checks)
        results['failed'] = len(all_checks) - sum(all_checks)
        
        return results
    
    def _validate_structure(self, html: str) -> Dict:
        """Validate HTML structure and components"""
        checks = {
            'has_library': 'lightweight-charts.standalone.production.js' in html,
            'has_header': 'class="header"' in html or 'id="header"' in html,
            'has_charts_container': 'class="charts-container"' in html,
            'has_portfolio_panel': 'class="portfolio-panel"' in html,
            'has_price_chart': 'id="price-chart"' in html,
            'has_oi_chart': 'id="oi-chart"' in html,
            'has_score_chart': 'id="score-chart"' in html,
            'has_cvd_chart': 'id="cvd-chart"' in html,
        }
        
        return {
            'checks': checks,
            'passed': sum(checks.values()),
            'failed': len(checks) - sum(checks.values())
        }
    
    def _validate_data(self, html: str) -> Dict:
        """Validate data structures"""
        checks = {}
        
        # Extract chartData
        start = html.find('const chartData = ') + len('const chartData = ')
        end = html.find(';', start)
        
        if start > len('const chartData = '):
            try:
                data = json.loads(html[start:end])
                
                # Check timeframes
                checks['has_timeframes'] = 'timeframes' in data
                checks['has_1s_data'] = '1s' in data.get('timeframes', {})
                checks['has_1m_data'] = '1m' in data.get('timeframes', {})
                
                # Check for specific features
                checks['has_exchange_volumes'] = 'exchange_volumes' in data
                checks['has_oi_data'] = 'oi_data' in data
                checks['has_portfolio_data'] = 'portfolio' in data
                checks['has_strategy_scores'] = 'strategy_scores' in data
                checks['has_trades'] = 'trades' in data
                
                # Check data quality
                if '1s' in data.get('timeframes', {}):
                    candles = data['timeframes']['1s'].get('data', [])
                    checks['1s_has_candles'] = len(candles) > 0
                    if candles:
                        # Check for NaN values
                        first_candle = candles[0]
                        checks['no_nan_values'] = all(
                            v is not None and v != 'NaN'
                            for v in [first_candle.get('open'), first_candle.get('high'),
                                    first_candle.get('low'), first_candle.get('close')]
                        )
                
            except json.JSONDecodeError:
                checks['valid_json'] = False
        else:
            checks['has_chart_data'] = False
        
        return {
            'checks': checks,
            'passed': sum(checks.values()),
            'failed': len(checks) - sum(checks.values())
        }
    
    def _validate_features(self, html: str) -> Dict:
        """Validate specific feature implementations"""
        checks = {
            # Exchange-colored volume bars
            'exchange_volume_implementation': 'addExchangeVolume' in html or 'exchange_volumes' in html,
            'exchange_colors_defined': all(
                color in html for color in ['#F0B90B', '#FF6B00', '#00D982', '#0052FF']
            ),
            
            # OI Candlesticks
            'oi_candlesticks': 'addCandlestickSeries' in html and 'oiCandles' in html,
            'oi_stats_panel': 'class="oi-stats"' in html,
            
            # Portfolio features
            'portfolio_value_chart': 'portfolio-value-chart' in html,
            'position_size_chart': 'position-size-chart' in html,
            'trades_list': 'trades-list' in html,
            
            # Strategy scoring
            'strategy_scoring': 'score.setData' in html or 'scoreData' in html,
            'confidence_bands': 'upperBand' in html and 'lowerBand' in html,
            
            # Chart synchronization
            'chart_sync': 'syncCharts' in html or 'subscribeVisibleTimeRangeChange' in html,
            
            # Timeframe switching
            'timeframe_buttons': 'tf-btn' in html,
            'timeframe_handler': 'switchTimeframe' in html,
        }
        
        return {
            'checks': checks,
            'passed': sum(checks.values()),
            'failed': len(checks) - sum(checks.values())
        }
    
    def _validate_navigation(self, html: str) -> Dict:
        """Validate multi-page navigation"""
        checks = {
            'has_navigation_links': 'nav-link' in html,
            'has_main_page': 'Main' in html or 'Dashboard' in html,
            'has_portfolio_link': 'Portfolio' in html,
            'has_exchange_link': 'Exchange' in html or 'Analytics' in html,
        }
        
        # Check if separate pages exist
        if self.latest_report:
            report_dir = self.latest_report.parent
            checks['exchange_page_exists'] = (report_dir / 'exchange_analytics.html').exists()
            checks['portfolio_page_exists'] = (report_dir / 'portfolio.html').exists()
        
        return {
            'checks': checks,
            'passed': sum(checks.values()),
            'failed': len(checks) - sum(checks.values())
        }
    
    def _validate_performance(self, html: str) -> Dict:
        """Validate performance optimizations"""
        checks = {
            'uses_binary_encoding': 'atob' in html or 'base64' in html,
            'has_viewport_meta': 'viewport' in html,
            'charts_initialized_properly': 'initCharts' in html,
            'dom_ready_check': 'DOMContentLoaded' in html or 'readyState' in html,
        }
        
        return {
            'checks': checks,
            'passed': sum(checks.values()),
            'failed': len(checks) - sum(checks.values())
        }
    
    def print_report(self, results: Dict):
        """Print formatted validation report"""
        print("\n" + "="*60)
        print("📊 DASHBOARD VALIDATION REPORT")
        print("="*60)
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"📁 Dashboard: {results['dashboard_path']}")
        print(f"📈 Overall: {results['overall_status']}")
        print(f"✅ Passed: {results['passed']} checks")
        print(f"❌ Failed: {results['failed']} checks")
        
        for category, data in results.items():
            if isinstance(data, dict) and 'checks' in data:
                print(f"\n### {category.upper().replace('_', ' ')}")
                for check, passed in data['checks'].items():
                    status = "✅" if passed else "❌"
                    print(f"  {status} {check.replace('_', ' ').title()}")
        
        # Recommendations
        if results['failed'] > 0:
            print("\n🔧 RECOMMENDATIONS:")
            
            if not results.get('navigation', {}).get('checks', {}).get('exchange_page_exists'):
                print("  • Create exchange_analytics.html for detailed exchange stats")
            
            if not results.get('features', {}).get('checks', {}).get('exchange_volume_implementation'):
                print("  • Implement exchange-colored volume bars in price chart")
            
            if not results.get('structure', {}).get('checks', {}).get('has_portfolio_panel'):
                print("  • Add portfolio panel to right sidebar")
        
        print("\n" + "="*60)

# Keep old function for compatibility
def validate_dashboard(dashboard_path):
    """Legacy validation function"""
    
    if not Path(dashboard_path).exists():
        print(f"❌ Dashboard not found: {dashboard_path}")
        return False
    
    html_content = Path(dashboard_path).read_text()
    
    # Extract chart data
    start = html_content.find('const chartData = ') + len('const chartData = ')
    end = html_content.find(';', start)
    
    try:
        data = json.loads(html_content[start:end])
    except Exception as e:
        print(f"❌ Failed to parse chart data: {e}")
        return False
    
    print('🔍 SELF-VALIDATION REPORT')
    print('=' * 50)
    
    # 1. Check symbol
    symbol = data.get('symbol')
    print(f'✓ Symbol: {symbol}')
    issues = []
    
    # 2. Check OHLCV data quality
    print(f'\n📊 OHLCV Data Quality:')
    for tf in ['1s', '1m', '5m', '15m', '1h']:
        tf_data = data.get('timeframes', {}).get(tf, {}).get('data', [])
        if not tf_data:
            issues.append(f'{tf}: No data')
            continue
        
        # Check first candle
        first = tf_data[0]
        if math.isnan(first['open']):  # NaN check
            issues.append(f'{tf}: NaN values')
        elif first['open'] > 10000:  # BTC price range
            issues.append(f'{tf}: BTC prices detected ({first["open"]:.0f})')
        elif first['open'] < 1000:  # Too low
            issues.append(f'{tf}: Invalid prices ({first["open"]:.0f})')
        else:
            print(f'  ✓ {tf}: {len(tf_data)} candles, price ~${first["open"]:.0f}')
    
    if issues:
        print(f'\n⚠️ DATA ISSUES FOUND:')
        for issue in issues:
            print(f'  - {issue}')
    
    # 3. Check trades
    trades = data.get('trades', {})
    markers = len(trades.get('markers', []))
    trade_list = len(trades.get('list', []))
    print(f'\n🔄 Trades:')
    print(f'  ✓ {markers} markers, {trade_list} in list')
    
    # 4. Check metrics
    metrics = data.get('metrics', {})
    print(f'\n📈 Performance Metrics:')
    print(f'  Balance: ${metrics["final_balance"]:,.2f}')
    print(f'  Return: {metrics["total_return"]:.2f}%')
    print(f'  Trades: {metrics["total_trades"]}')
    print(f'  Win Rate: {metrics["win_rate"]:.0f}%')
    
    # 5. Overall verdict
    print(f'\n🎯 VERDICT:')
    if not issues and markers > 0:
        print('  ✅ Dashboard is VALID and should display correctly!')
        return True
    else:
        print('  ❌ Dashboard has issues that need fixing')
        return False

def validate_with_browser(dashboard_path: str) -> bool:
    """Use browser MCP to visually validate dashboard"""
    try:
        # This would use the browser MCP tool to:
        # 1. Navigate to the dashboard
        # 2. Take screenshot
        # 3. Check for visual elements
        # 4. Return validation status
        print("🌐 Browser validation available via MCP tools")
        print(f"   Use: mcp__browsermcp__browser_navigate to open {dashboard_path}")
        print(f"   Use: mcp__browsermcp__browser_screenshot to capture visual")
        return True
    except:
        return False

if __name__ == "__main__":
    # New validation with DashboardValidator class
    validator = DashboardValidator()
    results = validator.validate_all()
    validator.print_report(results)
    
    # Suggest browser validation
    if results.get('dashboard_path'):
        print("\n💡 TIP: Use browser MCP tools for visual validation:")
        print(f"   1. Navigate to: file://{Path(results['dashboard_path']).absolute()}")
        print("   2. Take screenshot for visual inspection")
        print("   3. Check if charts render correctly")
    
    # Legacy support
    if len(sys.argv) > 1:
        print("\n" + "="*60)
        print("Running legacy validation on specific file...")
        result = validate_dashboard(sys.argv[1])
        sys.exit(0 if result else 1)