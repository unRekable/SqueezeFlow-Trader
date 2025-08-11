#!/usr/bin/env python3
"""
HTML to Image Converter using Python libraries
Fallback method for capturing dashboard screenshots
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import base64
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required packages if missing"""
    try:
        import playwright
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.info("Installing playwright for screenshot capture...")
        os.system("pip3 install playwright")
        os.system("python3 -m playwright install chromium")
        
def capture_with_playwright(html_path: Path, output_path: Path) -> bool:
    """Use Playwright to capture screenshot"""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            
            # Load the HTML file
            file_url = f"file://{html_path.absolute()}"
            page.goto(file_url)
            
            # Wait for content to load
            page.wait_for_timeout(3000)
            
            # Take full page screenshot
            page.screenshot(path=str(output_path), full_page=True)
            
            browser.close()
            
        logger.info(f"✅ Screenshot saved with Playwright: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Playwright screenshot failed: {e}")
        return False

def capture_with_selenium(html_path: Path, output_path: Path) -> bool:
    """Use Selenium to capture screenshot"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        driver = webdriver.Chrome(options=options)
        
        file_url = f"file://{html_path.absolute()}"
        driver.get(file_url)
        
        # Wait for page to load
        import time
        time.sleep(3)
        
        # Take screenshot
        driver.save_screenshot(str(output_path))
        driver.quit()
        
        logger.info(f"✅ Screenshot saved with Selenium: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Selenium screenshot failed: {e}")
        return False

def extract_data_from_html(html_path: Path) -> dict:
    """Extract data and structure from HTML for text-based analysis"""
    try:
        from bs4 import BeautifulSoup
        
        with open(html_path, 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            
        # Extract key information
        data = {
            "title": soup.title.string if soup.title else "Dashboard",
            "charts": [],
            "metrics": {},
            "tables": []
        }
        
        # Find Plotly charts
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'Plotly.newPlot' in script.string:
                # Extract chart data
                import re
                chart_matches = re.findall(r'Plotly\.newPlot\([\'"](\w+)[\'"],\s*(\[.*?\]),\s*(\{.*?\})', script.string, re.DOTALL)
                for div_id, chart_data, layout in chart_matches:
                    data["charts"].append({
                        "id": div_id,
                        "has_data": len(chart_data) > 10,  # Simple check
                        "type": "plotly"
                    })
                    
        # Extract metrics from divs with specific classes
        metric_divs = soup.find_all('div', class_=['metric', 'stat', 'kpi'])
        for div in metric_divs:
            label = div.find(['h3', 'h4', 'label', 'span'])
            value = div.find(['h2', 'p', 'span', 'div'])
            if label and value:
                data["metrics"][label.text.strip()] = value.text.strip()
                
        # Extract tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if rows:
                data["tables"].append({
                    "rows": len(rows),
                    "headers": [th.text.strip() for th in rows[0].find_all(['th', 'td'])] if rows else []
                })
                
        return data
        
    except Exception as e:
        logger.error(f"HTML parsing failed: {e}")
        return {}

def main():
    """Main capture function"""
    if len(sys.argv) < 2:
        # Find latest report
        reports = list(Path("backtest/results").glob("report_*/dashboard.html"))
        if not reports:
            logger.error("No HTML reports found")
            return 1
        html_path = max(reports, key=lambda p: p.stat().st_mtime)
    else:
        html_path = Path(sys.argv[1])
        
    if not html_path.exists():
        logger.error(f"HTML file not found: {html_path}")
        return 1
        
    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = html_path.parent / f"screenshot_{timestamp}.png"
    
    # Try different methods
    success = (
        capture_with_playwright(html_path, output_path) or
        capture_with_selenium(html_path, output_path)
    )
    
    if success:
        print(f"\n✅ Screenshot saved: {output_path}")
        print(f"📸 Claude can now read this image file")
    else:
        print("\n⚠️ Screenshot methods failed, extracting text data...")
        data = extract_data_from_html(html_path)
        
        # Save as JSON for Claude to read
        json_path = html_path.parent / f"dashboard_data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"📊 Data extracted to: {json_path}")
        print(f"📖 Claude can read this JSON file")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())