#!/usr/bin/env python3
"""
Test framework to debug why charts don't render in dashboard
Based on lessons learned: simple test works, complex dashboard doesn't
"""

from pathlib import Path
import re
import json

class DashboardChartTester:
    def __init__(self):
        self.results_dir = Path('backtest/results')
        self.tests_dir = Path('tests')
        self.tests_dir.mkdir(exist_ok=True)
        
    def find_latest_dashboards(self, count=3):
        """Find the most recent dashboards"""
        reports = sorted(self.results_dir.glob('report_*/dashboard.html'))
        return reports[-count:] if len(reports) >= count else reports
    
    def extract_chart_code(self, dashboard_path):
        """Extract just the chart initialization code"""
        html = Path(dashboard_path).read_text()
        
        # Find the script section
        script_start = html.find('<script>')
        script_end = html.find('</script>', script_start)
        
        if script_start == -1 or script_end == -1:
            return None
            
        script_content = html[script_start:script_end]
        
        # Extract key parts
        analysis = {
            'has_library': 'lightweight-charts' in html,
            'has_chart_data': 'const chartData' in script_content,
            'has_init_charts': 'function initCharts' in script_content,
            'has_window_load': 'window.addEventListener' in script_content,
            'script_in_same_tag': False  # Check if script src and content in same tag
        }
        
        # Critical check: script tag structure
        if '<script src="https://unpkg.com/lightweight-charts' in html:
            # Find this script tag
            lib_tag_start = html.find('<script src="https://unpkg.com/lightweight-charts')
            lib_tag_end = html.find('</script>', lib_tag_start)
            lib_tag_content = html[lib_tag_start:lib_tag_end]
            
            # Check if there's content between opening and closing
            if lib_tag_end - lib_tag_start > 200:  # More than just the tag
                analysis['script_in_same_tag'] = True
        
        return analysis
    
    def create_minimal_test(self, dashboard_path, test_name):
        """Create a minimal test from dashboard data"""
        html = Path(dashboard_path).read_text()
        
        # Extract chart data
        data_match = re.search(r'const chartData = ({.*?});', html, re.DOTALL)
        if not data_match:
            return None
            
        chart_data = data_match.group(1)
        
        # Get symbol
        symbol_match = re.search(r'"symbol":\s*"([^"]+)"', chart_data)
        symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
        
        # Create minimal test that WILL work
        test_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Chart Test - {symbol}</title>
    <!-- CRITICAL: Library in its own tag -->
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{ background: #131722; color: white; margin: 0; padding: 20px; }}
        .status {{ padding: 10px; background: #1e222d; margin-bottom: 10px; }}
        .chart {{ height: 400px; background: #1e222d; border: 1px solid #2a2e39; }}
    </style>
</head>
<body>
    <div class="status" id="status">Initializing {symbol} chart...</div>
    <div class="chart" id="chart"></div>
    
    <!-- CRITICAL: Separate script tag for code -->
    <script>
        // Wait for everything to load
        window.addEventListener('load', function() {{
            const status = document.getElementById('status');
            const container = document.getElementById('chart');
            
            console.log('Window loaded, creating chart...');
            
            try {{
                // Create chart with explicit dimensions
                const chart = LightweightCharts.createChart(container, {{
                    width: container.offsetWidth,
                    height: 400,
                    layout: {{
                        background: {{ type: 'solid', color: '#1e222d' }},
                        textColor: '#d1d4dc',
                    }},
                    grid: {{
                        vertLines: {{ color: '#2a2e39' }},
                        horzLines: {{ color: '#2a2e39' }},
                    }}
                }});
                
                // Add series
                const series = chart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350'
                }});
                
                // Use actual data from dashboard
                const chartData = {chart_data};
                
                // Set data from actual backtest
                if (chartData.timeframes && chartData.timeframes['1s']) {{
                    const data = chartData.timeframes['1s'].data.slice(0, 100); // First 100 candles
                    series.setData(data);
                    status.innerHTML = '✅ {symbol} chart loaded with ' + data.length + ' candles';
                    status.style.color = '#26a69a';
                }} else {{
                    status.innerHTML = '⚠️ No data found for {symbol}';
                    status.style.color = '#ffa726';
                }}
                
            }} catch(error) {{
                status.innerHTML = '❌ Error: ' + error.message;
                status.style.color = '#ef5350';
                console.error('Chart error:', error);
            }}
        }});
    </script>
</body>
</html>"""
        
        # Save test
        test_path = self.tests_dir / f'test_{test_name}.html'
        test_path.write_text(test_html)
        return test_path
    
    def diagnose_dashboard(self, dashboard_path):
        """Diagnose why a dashboard doesn't work"""
        print(f"\n🔍 Diagnosing: {dashboard_path.name}")
        print("=" * 60)
        
        analysis = self.extract_chart_code(dashboard_path)
        
        if not analysis:
            print("❌ Could not analyze dashboard")
            return
        
        issues = []
        
        # Check each requirement
        print("\n📋 Checklist:")
        print(f"  {'✅' if analysis['has_library'] else '❌'} Library included")
        print(f"  {'✅' if analysis['has_chart_data'] else '❌'} Chart data present")
        print(f"  {'✅' if analysis['has_init_charts'] else '❌'} Init function exists")
        print(f"  {'✅' if analysis['has_window_load'] else '❌'} Window load listener")
        print(f"  {'✅' if not analysis['script_in_same_tag'] else '❌'} Script tags separated")
        
        # Critical issue check
        if analysis['script_in_same_tag']:
            issues.append("CRITICAL: Script tag has both src and content - must separate!")
        
        if issues:
            print(f"\n⚠️ Issues found:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("\n✅ Structure looks correct")
        
        return analysis
    
    def run_tests(self):
        """Run all tests"""
        print("🧪 DASHBOARD CHART TESTING FRAMEWORK")
        print("=" * 60)
        
        # Get latest dashboards
        dashboards = self.find_latest_dashboards(3)
        
        if not dashboards:
            print("❌ No dashboards found")
            return
        
        print(f"\n📊 Found {len(dashboards)} recent dashboards")
        
        # Test each dashboard
        for i, dashboard in enumerate(dashboards):
            # Extract symbol from path
            report_dir = dashboard.parent.name
            
            # Diagnose the dashboard
            self.diagnose_dashboard(dashboard)
            
            # Create minimal test
            test_path = self.create_minimal_test(dashboard, f"{report_dir}")
            
            if test_path:
                print(f"\n✅ Created test: {test_path}")
                print(f"   Open this to verify charts work: open {test_path}")
        
        print("\n" + "=" * 60)
        print("📝 SUMMARY:")
        print("  • Simple tests work because script tags are separated")
        print("  • Dashboards fail if script has both src and content")
        print("  • Solution: Always use separate <script> tags")
        
        return dashboards

if __name__ == "__main__":
    tester = DashboardChartTester()
    dashboards = tester.run_tests()
    
    # Apply fix to latest dashboard
    if dashboards:
        latest = dashboards[-1]
        print(f"\n🔧 Applying fix to latest dashboard: {latest}")
        
        # Read dashboard
        html = latest.read_text()
        
        # Check for the issue
        if '<script src="https://unpkg.com/lightweight-charts' in html:
            # Find and fix
            pattern = r'(<script src="https://unpkg.com/lightweight-charts[^>]+>)(.*?)(</script>)'
            
            def fix_script_tag(match):
                opening = match.group(1)
                content = match.group(2)
                closing = match.group(3)
                
                if content.strip():  # Has content
                    # Separate into two tags
                    return f"{opening}{closing}\n    <script>{content}</script>"
                else:
                    return match.group(0)  # No change needed
            
            fixed_html = re.sub(pattern, fix_script_tag, html, flags=re.DOTALL)
            
            if fixed_html != html:
                latest.write_text(fixed_html)
                print("   ✅ Fixed script tag issue!")
            else:
                print("   ℹ️ No script tag issue found")