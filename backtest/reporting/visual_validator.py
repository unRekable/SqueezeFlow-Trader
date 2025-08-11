#!/usr/bin/env python3
"""
Visual Validator for HTML Dashboards
Captures screenshots of generated HTML reports so Claude can analyze them
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import subprocess
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardVisualValidator:
    """Takes screenshots of HTML dashboards for Claude to analyze"""
    
    def __init__(self, results_dir: str = "backtest/results"):
        self.results_dir = Path(results_dir)
        self.reports_found = []
        self.screenshots_taken = []
        
    def find_latest_report(self) -> Optional[Path]:
        """Find the most recent HTML report"""
        html_files = list(self.results_dir.glob("report_*/dashboard.html"))
        if not html_files:
            logger.warning("No HTML reports found")
            return None
            
        latest = max(html_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest report: {latest}")
        return latest
        
    def take_screenshot_with_webkit(self, html_path: Path, output_path: Path) -> bool:
        """Use webkit2png to capture screenshot (macOS)"""
        try:
            # Convert to file:// URL
            file_url = f"file://{html_path.absolute()}"
            
            # Use webkit2png if available
            cmd = [
                "webkit2png",
                "-F",  # Full size
                "-o", output_path.stem,  # Output name without extension
                "-D", str(output_path.parent),  # Output directory
                "--delay=3",  # Wait for JS to render
                file_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Screenshot saved: {output_path}")
                return True
            else:
                logger.error(f"webkit2png failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning("webkit2png not found, trying alternative method...")
            return False
            
    def take_screenshot_with_chrome(self, html_path: Path, output_path: Path) -> bool:
        """Use Chrome headless to capture screenshot"""
        try:
            file_url = f"file://{html_path.absolute()}"
            
            # Try Chrome/Chromium headless mode
            chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/usr/bin/google-chrome",
                "/usr/bin/chromium",
                "chrome",
                "chromium"
            ]
            
            chrome_exe = None
            for path in chrome_paths:
                if subprocess.run(["which", path], capture_output=True).returncode == 0:
                    chrome_exe = path
                    break
                elif os.path.exists(path):
                    chrome_exe = path
                    break
                    
            if not chrome_exe:
                logger.error("Chrome/Chromium not found")
                return False
                
            cmd = [
                chrome_exe,
                "--headless",
                "--disable-gpu",
                "--window-size=1920,1080",
                f"--screenshot={output_path}",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                file_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 or output_path.exists():
                logger.info(f"✅ Screenshot saved: {output_path}")
                return True
            else:
                logger.error(f"Chrome screenshot failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Chrome screenshot error: {e}")
            return False
            
    def take_screenshot_with_safari(self, html_path: Path, output_path: Path) -> bool:
        """Use Safari via AppleScript to capture screenshot (macOS fallback)"""
        try:
            file_url = f"file://{html_path.absolute()}"
            
            # AppleScript to open in Safari and screenshot
            script = f'''
            tell application "Safari"
                activate
                open location "{file_url}"
                delay 3
            end tell
            
            do shell script "screencapture -l $(osascript -e 'tell app \\"Safari\\" to id of window 1') {output_path}"
            
            tell application "Safari"
                close (every tab of window 1 whose URL contains "{html_path.name}")
            end tell
            '''
            
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
            if output_path.exists():
                logger.info(f"✅ Screenshot saved via Safari: {output_path}")
                return True
            else:
                logger.error("Safari screenshot failed")
                return False
                
        except Exception as e:
            logger.error(f"Safari screenshot error: {e}")
            return False
            
    def capture_dashboard(self, html_path: Optional[Path] = None) -> Dict[str, Any]:
        """Capture screenshot of dashboard and return analysis"""
        if html_path is None:
            html_path = self.find_latest_report()
            
        if not html_path or not html_path.exists():
            return {
                "success": False,
                "error": "No HTML report found"
            }
            
        # Generate screenshot filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_name = f"dashboard_screenshot_{timestamp}.png"
        screenshot_path = html_path.parent / screenshot_name
        
        # Try different screenshot methods
        success = (
            self.take_screenshot_with_chrome(html_path, screenshot_path) or
            self.take_screenshot_with_webkit(html_path, screenshot_path) or
            self.take_screenshot_with_safari(html_path, screenshot_path)
        )
        
        if success:
            self.screenshots_taken.append(screenshot_path)
            
            # Generate validation report
            report = {
                "success": True,
                "html_path": str(html_path),
                "screenshot_path": str(screenshot_path),
                "timestamp": timestamp,
                "validation_notes": [
                    "Screenshot captured successfully",
                    f"Claude can now read: {screenshot_path}",
                    "Use Read tool on the screenshot to analyze the dashboard"
                ],
                "file_size": screenshot_path.stat().st_size,
                "dimensions": self._get_image_dimensions(screenshot_path)
            }
            
            # Save validation report
            report_path = html_path.parent / f"validation_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"📸 Screenshot ready for Claude to analyze: {screenshot_path}")
            logger.info(f"📊 Validation report: {report_path}")
            
            return report
        else:
            return {
                "success": False,
                "error": "Failed to capture screenshot with all methods",
                "attempted_methods": ["chrome", "webkit2png", "safari"]
            }
            
    def _get_image_dimensions(self, image_path: Path) -> Optional[Dict[str, int]]:
        """Get image dimensions if possible"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return {"width": img.width, "height": img.height}
        except ImportError:
            # Try with sips (macOS)
            try:
                result = subprocess.run(
                    ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(image_path)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    width = int(lines[1].split()[-1])
                    height = int(lines[2].split()[-1])
                    return {"width": width, "height": height}
            except:
                pass
        return None
        
    def validate_all_recent_reports(self, limit: int = 5) -> list:
        """Validate multiple recent reports"""
        html_files = sorted(
            self.results_dir.glob("report_*/dashboard.html"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        results = []
        for html_path in html_files:
            logger.info(f"Validating: {html_path}")
            result = self.capture_dashboard(html_path)
            results.append(result)
            time.sleep(1)  # Avoid overwhelming the system
            
        return results

def main():
    """Main entry point for visual validation"""
    validator = DashboardVisualValidator()
    
    # Check for specific report or use latest
    if len(sys.argv) > 1:
        html_path = Path(sys.argv[1])
        result = validator.capture_dashboard(html_path)
    else:
        result = validator.capture_dashboard()
        
    if result["success"]:
        print("\n" + "="*60)
        print("✅ VISUAL VALIDATION SUCCESSFUL")
        print("="*60)
        print(f"📸 Screenshot: {result['screenshot_path']}")
        print(f"📄 HTML Report: {result['html_path']}")
        print("\n🔍 Claude can now analyze the dashboard by reading:")
        print(f"   {result['screenshot_path']}")
        print("\n💡 Next step: Use Read tool on the screenshot file")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ VISUAL VALIDATION FAILED")
        print("="*60)
        print(f"Error: {result.get('error', 'Unknown error')}")
        print("\n🔧 Troubleshooting:")
        print("1. Install Chrome: brew install --cask google-chrome")
        print("2. Install webkit2png: brew install webkit2png")
        print("3. Or ensure Safari is accessible")
        print("="*60)
        
    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())