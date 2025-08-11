"""
Validate if TradingView charts are actually rendering using browser automation
"""

import time
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
except ImportError:
    print("Installing selenium...")
    import subprocess
    subprocess.check_call(["pip3", "install", "selenium"])
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

def check_chart_rendering(html_path):
    """Check if a chart actually renders with content"""
    
    # Setup Chrome in headless mode
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=options)
    except:
        print("Chrome driver not found. Trying Safari...")
        driver = webdriver.Safari()
    
    try:
        # Load the HTML file
        file_url = f"file://{Path(html_path).absolute()}"
        print(f"Loading: {file_url}")
        driver.get(file_url)
        
        # Wait for chart to initialize
        time.sleep(3)
        
        # Check for canvas elements (TradingView creates canvas for charts)
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        print(f"Found {len(canvases)} canvas elements")
        
        # Check if canvases have actual content
        for i, canvas in enumerate(canvases):
            width = canvas.get_attribute("width")
            height = canvas.get_attribute("height")
            print(f"Canvas {i+1}: {width}x{height}")
            
            # Execute JavaScript to check if canvas has been drawn on
            has_content = driver.execute_script("""
                const canvas = arguments[0];
                const ctx = canvas.getContext('2d');
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                
                // Check if any pixel is not transparent/black
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i] > 0 || data[i+1] > 0 || data[i+2] > 0) {
                        return true;
                    }
                }
                return false;
            """, canvas)
            
            print(f"  Has visible content: {has_content}")
        
        # Check console errors
        logs = driver.get_log('browser')
        errors = [log for log in logs if log['level'] == 'SEVERE']
        if errors:
            print("\n⚠️ JavaScript Errors found:")
            for error in errors:
                print(f"  {error['message']}")
        
        # Take a screenshot for visual verification
        screenshot_path = f"{Path(html_path).stem}_screenshot.png"
        driver.save_screenshot(screenshot_path)
        print(f"\n📸 Screenshot saved: {screenshot_path}")
        
        return len(canvases) > 0
        
    finally:
        driver.quit()

# Check test files
test_files = [
    "test_tradingview_working.html",
    "test_future_timestamps.html"
]

for file in test_files:
    if Path(file).exists():
        print(f"\n{'='*50}")
        print(f"Checking: {file}")
        print('='*50)
        check_chart_rendering(file)

# Check latest dashboard
dashboards = sorted(Path("backtest/results").glob("*/dashboard.html"))
if dashboards:
    latest = dashboards[-1]
    print(f"\n{'='*50}")
    print(f"Checking Dashboard: {latest.name}")
    print('='*50)
    check_chart_rendering(str(latest))