# Visual Validation Process for Dashboard Development

## 🎯 The Claude Code Visual Validation Framework

This document outlines the **MANDATORY** visual validation process that must be followed when developing or debugging dashboards. This process was established after multiple dashboard failures that could have been prevented with proper visual inspection.

## 📋 The 4-Step Process

### Step 1: Execute
Write and deploy your dashboard code.

### Step 2: Validate Visually
**NEVER ASSUME THE DASHBOARD WORKS** - Always use visual validation:

```bash
# Option 1: Use MCP Playwright Browser Tools
mcp__playwright__browser_navigate - Navigate to the dashboard
mcp__playwright__browser_take_screenshot - Capture what's actually displayed
mcp__playwright__browser_console_messages - Check for JavaScript errors

# Option 2: Use visual_validator.py (if available)
python3 backtest/reporting/visual_validator.py
```

### Step 3: Debug Using Visual Feedback
1. **Read the screenshot** - See exactly what the user sees
2. **Check console errors** - Identify JavaScript issues
3. **Inspect page elements** - Use browser_snapshot for DOM structure
4. **Test interactions** - Click buttons, check if they work

### Step 4: Document and Learn
Update relevant documentation with findings and solutions.

## 🔍 Visual Validation Checklist

- [ ] Navigate to the dashboard URL
- [ ] Take a screenshot (both viewport and full page)
- [ ] Check console for errors
- [ ] Verify all expected elements are visible
- [ ] Test interactive elements (buttons, dropdowns)
- [ ] Compare against expected output
- [ ] Document any issues found

## 💡 Example: TradingView Implementation Debug Session

### Problem: Dashboard shows blank page

**Step 1: Navigate and Check Console**
```javascript
mcp__playwright__browser_navigate(url: "file:///path/to/dashboard.html")
mcp__playwright__browser_console_messages()
// Result: "TypeError: chart.addCandlestickSeries is not a function"
```

**Step 2: Take Screenshot**
```javascript
mcp__playwright__browser_take_screenshot(fullPage: true)
// Visual confirmation: Page is blank except header
```

**Step 3: Research the Error**
```javascript
WebFetch(url: "https://tradingview.github.io/lightweight-charts/docs")
// Learn: API changed from addCandlestickSeries to addSeries(SeriesType, options)
```

**Step 4: Test API Methods**
```javascript
mcp__playwright__browser_evaluate(function: "() => {
    const chart = LightweightCharts.createChart(document.createElement('div'));
    return Object.getOwnPropertyNames(Object.getPrototypeOf(chart)).filter(m => m.startsWith('add'));
}")
// Result: "addCustomSeries, addSeries, addPane"
```

**Step 5: Fix and Verify**
- Update code to use new API
- Regenerate dashboard
- Take new screenshot
- Verify all panes are visible

## 🚨 Common Visual Issues and Solutions

### Issue: JavaScript Errors Not Visible
**Solution**: Always check console messages before assuming code works
```javascript
mcp__playwright__browser_console_messages()
```

### Issue: Elements Present but Not Visible
**Solution**: Take full page screenshot to check if content is below fold
```javascript
mcp__playwright__browser_take_screenshot(fullPage: true)
```

### Issue: Partial Rendering
**Solution**: Check DOM structure to see what's actually loaded
```javascript
mcp__playwright__browser_snapshot()
```

### Issue: Interactive Elements Not Working
**Solution**: Test clicks and observe console for errors
```javascript
mcp__playwright__browser_click(element: "button", ref: "e12")
mcp__playwright__browser_console_messages()
```

## 📊 Visual Validation for Multi-Pane Charts

When implementing complex visualizations like TradingView with multiple panes:

1. **Verify each pane individually**
   - Take screenshot after each pane is added
   - Confirm data is displayed correctly
   - Check pane separation is visible

2. **Check synchronization**
   - Zoom/pan main chart
   - Verify all panes stay synchronized

3. **Validate indicators**
   - Each indicator should be in its designated pane
   - Labels and values should be visible
   - Threshold lines should be rendered

## 🎓 Lessons Learned

### From TradingView Implementation (2025-08-11)

1. **API Documentation is Critical**
   - Always check official docs when errors occur
   - APIs change between versions
   - Test available methods directly in browser console

2. **Visual Feedback Accelerates Debugging**
   - Console errors immediately revealed the issue
   - Screenshots confirmed what was/wasn't working
   - Iterative visual testing led to quick resolution

3. **Never Skip Visual Validation**
   - Even when console shows "Dashboard generated"
   - Even when file size looks correct
   - Even when no errors in Python logs

## 🔧 Tools Reference

### MCP Playwright Browser Tools
- `mcp__playwright__browser_navigate` - Load pages
- `mcp__playwright__browser_take_screenshot` - Capture visuals
- `mcp__playwright__browser_console_messages` - Get JS errors
- `mcp__playwright__browser_evaluate` - Run JS in page context
- `mcp__playwright__browser_snapshot` - Get DOM structure
- `mcp__playwright__browser_click` - Interact with elements

### Helper Scripts
- `visual_validator.py` - Automated visual validation
- `test_dashboard.py` - Dashboard testing framework

## 📝 Documentation to Update After Visual Validation

1. **DASHBOARD_PROGRESS.md** - Track what's working/broken
2. **SYSTEM_TRUTH.md** - Document actual vs expected behavior
3. **LESSONS_LEARNED.md** - Record debugging insights
4. **Component-specific docs** - Update with API changes

## ⚡ Quick Commands

```bash
# Generate dashboard with TradingView panes
USE_TRADINGVIEW_PANES=true python3 backtest/engine.py --symbol BTC

# Navigate to latest dashboard
latest=$(ls -t results/backtest_* | head -1)
file:///Users/u/PycharmProjects/SqueezeFlow Trader/$latest/dashboard.html

# Check if using TradingView implementation
grep -c "paneIndex" $latest/dashboard.html
```

## 🎯 Remember

**"If you can't see it, it doesn't work"** - Always validate visually!

The visual validation process is not optional. It's the difference between assuming something works and KNOWING it works.