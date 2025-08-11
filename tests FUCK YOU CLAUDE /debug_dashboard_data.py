"""Debug script to check what data the dashboard is generating"""

import json
import re
from pathlib import Path

# Get latest dashboard
dashboards = sorted(Path("backtest/results").glob("*/dashboard.html"))
if not dashboards:
    print("No dashboards found")
    exit(1)

latest = dashboards[-1]
print(f"Checking: {latest}")

# Read dashboard HTML
html = latest.read_text()

# Extract the chartData JSON
match = re.search(r'const chartData = ({.*?});', html, re.DOTALL)
if not match:
    print("Could not find chartData in dashboard")
    exit(1)

# Parse the JSON
try:
    data = json.loads(match.group(1))
    
    print("\n=== DATA STRUCTURE ===")
    print(f"Keys: {list(data.keys())}")
    print(f"Symbol: {data.get('symbol', 'MISSING')}")
    
    if 'candles' in data:
        candles = data['candles']
        print(f"\n=== CANDLES ===")
        print(f"Total candles: {len(candles)}")
        if candles:
            print(f"First candle: {candles[0]}")
            print(f"Last candle: {candles[-1]}")
            
            # Check for issues
            issues = []
            for i, candle in enumerate(candles[:10]):
                if 'time' not in candle:
                    issues.append(f"Candle {i} missing 'time'")
                if 'open' not in candle:
                    issues.append(f"Candle {i} missing 'open'")
                if not isinstance(candle.get('time'), (int, float)):
                    issues.append(f"Candle {i} time is not numeric: {candle.get('time')}")
                    
            if issues:
                print("\n⚠️ ISSUES FOUND:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ Data structure looks correct")
    else:
        print("❌ No 'candles' key in data!")
        
    if 'cvd' in data:
        print(f"\n=== CVD ===")
        print(f"CVD points: {len(data['cvd'])}")
        if data['cvd']:
            print(f"First CVD: {data['cvd'][0]}")
            
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    print("Raw JSON snippet:", match.group(1)[:500])