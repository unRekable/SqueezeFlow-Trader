#!/usr/bin/env python3
"""
Debug why initCharts isn't working even though structure is correct
"""

from pathlib import Path
import re

def debug_init_sequence(dashboard_path):
    """Debug the exact initialization sequence"""
    
    html = Path(dashboard_path).read_text()
    
    print("🔍 DEBUGGING CHART INITIALIZATION")
    print("=" * 60)
    
    # Find where initCharts is defined
    init_def = html.find('function initCharts()')
    print(f"\n1. initCharts defined at position: {init_def}")
    
    # Find where it's called
    init_calls = []
    for match in re.finditer(r'initCharts\(\)', html):
        init_calls.append(match.start())
    
    print(f"2. initCharts called at positions: {init_calls}")
    
    # Find window.addEventListener
    window_load = html.find("window.addEventListener('load'")
    print(f"3. window.load listener at: {window_load}")
    
    # Check if chartOptions is defined
    chart_options = html.find('const chartOptions')
    print(f"4. chartOptions defined at: {chart_options}")
    
    # Check the actual window.load code
    if window_load > 0:
        # Extract the listener code
        listener_end = html.find('});', window_load) + 3
        listener_code = html[window_load:listener_end]
        
        print("\n5. Window load listener code:")
        print("-" * 40)
        # Show first 500 chars
        print(listener_code[:500])
        print("-" * 40)
    
    # Check for console.log statements
    console_logs = []
    for match in re.finditer(r'console\.log\([^)]+\)', html):
        console_logs.append(html[match.start():match.end()])
    
    print(f"\n6. Console.log statements found: {len(console_logs)}")
    if console_logs:
        print("   First few logs:")
        for log in console_logs[:3]:
            print(f"   • {log}")
    
    # Check for try-catch blocks
    try_blocks = html.count('try {')
    print(f"\n7. Try-catch blocks: {try_blocks}")
    
    # Look for the actual chart creation
    chart_creates = []
    for match in re.finditer(r'LightweightCharts\.createChart', html):
        # Get surrounding context
        start = max(0, match.start() - 50)
        end = min(len(html), match.end() + 100)
        context = html[start:end]
        chart_creates.append(context)
    
    print(f"\n8. Chart creation calls: {len(chart_creates)}")
    
    # Check if charts variable is properly scoped
    charts_var_declarations = []
    for match in re.finditer(r'(let|const|var)\s+charts\s*=', html):
        charts_var_declarations.append(match.start())
    
    print(f"\n9. 'charts' variable declarations at: {charts_var_declarations}")
    
    # Check for the critical issue: are we inside a function that's not called?
    # Find all function definitions
    functions = []
    for match in re.finditer(r'function\s+(\w+)\s*\(', html):
        func_name = match.group(1)
        func_pos = match.start()
        functions.append((func_name, func_pos))
    
    print(f"\n10. Functions defined: {[f[0] for f in functions]}")
    
    # Create a fixed version
    print("\n" + "=" * 60)
    print("🔧 CREATING FIXED VERSION...")
    
    # The issue might be that initCharts is called but charts aren't created
    # Let's inline everything in window.load
    
    fixed_html = html
    
    # Remove the setTimeout wrapper if it exists
    fixed_html = fixed_html.replace(
        "setTimeout(() => {",
        "// Direct execution\n            "
    )
    fixed_html = fixed_html.replace(
        "}, 200);",
        "// No delay"
    )
    
    # Make sure console.log statements are there for debugging
    if "console.log('Window loaded" not in fixed_html:
        fixed_html = fixed_html.replace(
            "window.addEventListener('load', () => {",
            """window.addEventListener('load', () => {
            console.log('Window loaded, charts available:', typeof LightweightCharts);
            console.log('Container exists:', document.getElementById('price-chart'));"""
        )
    
    # Save the fixed version
    fixed_path = Path(dashboard_path).parent / 'dashboard_fixed.html'
    fixed_path.write_text(fixed_html)
    
    print(f"✅ Created fixed version: {fixed_path}")
    print("\nOpen both to compare:")
    print(f"  Original: {dashboard_path}")
    print(f"  Fixed: {fixed_path}")
    
    return fixed_path

if __name__ == "__main__":
    # Debug the latest dashboard
    latest = sorted(Path('backtest/results').glob('report_*/dashboard.html'))[-1]
    print(f"Debugging: {latest}\n")
    
    fixed = debug_init_sequence(latest)
    
    # Open both for comparison
    import os
    os.system(f"open {latest}")
    os.system(f"open {fixed}")