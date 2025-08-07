---
description: Check SqueezeFlow system health and status
allowed-tools: ["Bash", "Read"]
---

# System Check Command

Comprehensive health check for SqueezeFlow Trader system.

## Usage:
- `/system-check` - Full system health check

## What it checks:
1. **Docker Services**: All containers running and healthy
2. **Database Connectivity**: InfluxDB and Redis accessibility
3. **Data Pipeline**: Recent data collection and signal generation
4. **API Endpoints**: FreqTrade and monitoring interfaces
5. **Log Files**: Recent errors and warnings
6. **Disk Space**: Available storage for data and logs
7. **Python Environment**: Virtual environment and dependencies

## Output:
- ✅ Green: Component is healthy
- ⚠️  Yellow: Component has warnings
- ❌ Red: Component has critical issues
- 📊 Summary: Overall system health score

The command provides actionable recommendations for any issues found.