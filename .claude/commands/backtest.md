---
description: Run backtest with specified parameters
argument-hint: "[timeframe] [balance]"
allowed-tools: Bash, Read, Write
---

# Backtest Command

Run a backtest for SqueezeFlow Trader with flexible parameters.

## Usage Examples:
- `/backtest` - Run default backtest (last week, 10k balance)
- `/backtest last_month` - Run last month with default balance
- `/backtest last_week 50000` - Run last week with 50k balance
- `/backtest last_month 100000 squeezeflow_strategy` - Full custom backtest

## Parameters:
1. **timeframe** (optional): `last_week`, `last_month`, `last_3months`, or custom date range
2. **balance** (optional): Starting balance in USD (default: 10000)
3. **strategy** (optional): Strategy name (default: squeezeflow_strategy)

The command will:
1. Validate parameters and show backtest plan
2. Execute the backtest with proper logging
3. Generate visualization results (HTML + PNG)
4. Show performance summary
5. Open results in browser if successful