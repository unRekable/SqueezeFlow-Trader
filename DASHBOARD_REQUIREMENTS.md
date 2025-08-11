# Dashboard Requirements - FINAL CONSOLIDATED SPEC

## 🎯 Core Purpose
**Create an interactive dashboard that visualizes what the SqueezeFlow strategy is seeing and thinking**

## ⚠️ CRITICAL REQUIREMENTS FROM USER

### 1. Volume Bars with Exchange Colors
- **MUST show volume bars colored by exchange** (Binance yellow, Bybit orange, OKX green, etc.)
- **Stacked bars** showing which exchange contributed what volume
- **Inside the TradingView chart** (not separate)
- Each bar segment shows exchange name and percentage

### 2. Portfolio Panel (Right Sidebar)
- Real-time portfolio value tracking
- Position size evolution
- Drawdown visualization  
- Recent trades with PnL
- Key performance metrics (Return, Balance, Win Rate, Max DD, Sharpe)

### 3. Exchange-Specific Analytics Page
- Separate HTML page for detailed exchange statistics
- Market coverage by exchange
- Volume distribution charts
- Buy/sell pressure by exchange
- Trade execution analysis

### 4. Open Interest Features
- **OI Candlesticks** (OHLC of OI values)
- OI trend indicator (POSITIVE/NEGATIVE/NEUTRAL)
- 1h and 24h percentage changes
- Stats panel below OI chart

### 5. Strategy Insights
- Strategy scoring visualization (0-1 normalized)
- Signal confidence bands
- Phase indicators
- Price/CVD divergence strength

### 6. Technical Requirements
- Handle 86,400 data points (1s data for 24 hours)
- Use TradingView lightweight-charts (NOT Plotly)
- Binary encoding for data transfer
- WebGL acceleration
- All charts synchronized (zoom/pan together)

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header: SqueezeFlow | ETH/USDT | Price | Change | [1s][1m] │
├───────────────────────────────────────┬─────────────────────┤
│                                       │                     │
│  Price Chart with Volume              │  Portfolio Panel    │
│  - Candlesticks                      │  - Total Return     │
│  - Trade markers                      │  - Current Balance  │
│  - Exchange-colored volume bars      │  - Max Drawdown     │
│                                       │  - Win Rate         │
├───────────────────────────────────────┤  - Portfolio Chart  │
│  Open Interest (Candlesticks)         │  - Position Chart   │
│  [Trend: POSITIVE] [1h: +2%] [24h: +5%]  - Recent Trades X (XXXXXXXXX remove this from here Claude)  │
├───────────────────────────────────────┤                     │
│  Strategy Scoring & Confidence        │                     │
├───────────────────────────────────────┤                     │
│  CVD Analysis (Spot vs Futures)       │                     │
├───────────────────────────────────────┤                     │
│  Price/CVD Divergence Strength        │                     │
└───────────────────────────────────────┴─────────────────────┘
```

## 🔑 Implementation Strategy
(XXXXXXXXXX CLAUDE FUCKING KEEP TRACK OF PROGRESS IN DESIGNATED FILES! I WILL HAVE YOU READ THIS ONE BUT YOU NEED TO KEEP BETTER DOCS STRUCTURE)
### Phase 1: Fix Current Visualizer
1. Remove NaN values from 1s data ✅
2. Ensure proper symbol (ETH not BTC) (XXXXXXXXX:ATTENTION CLAUDE: THIS LOGIC NEEDS TO BE REWORKED SO THAT YOU FIGURE OUT EACH TIME WHICH CRYPTO IS THE BACKTEST ACTUALLY ON. MAY BE ETH, MAY BE OTHERS)
3. Clean up test files

### Phase 2: Add Missing Features
1. **Exchange-colored volume bars** (CRITICAL)
2. **Portfolio panel** on right side
3. **OI candlesticks** (not just line)
4. **Strategy scoring** visualization

### Phase 3: Create Exchange Analytics Page
1. Separate HTML for exchange stats
2. Market coverage tables
3. Volume distribution charts
4. Trade routing analysis

## 📁 Final File Structure

```
backtest/reporting/
├── visualizer.py               # Main visualizer (KEEP & ENHANCE)
├── dashboard_generator.py      # Dashboard HTML generation
├── data_serializer.py         # Binary encoding for performance
└── templates/
    ├── dashboard.html         # Main dashboard template
    └── exchange_analytics.html # Exchange stats page
```

## ❌ Files to Delete/Consolidate

### Delete These Documentation Files:
- BETTER_VISUALIZATION_PLAN.md (obsolete)
- ELEGANT_VISUALIZATION_SOLUTION.md (obsolete)
- PERFORMANCE_IMPROVEMENTS.md (merged here)
- backtest/reporting/INTERACTIVE_PLOTTING_IMPLEMENTATION.md (merged here)
- OI_DISABLED_NOTE.md (OI works from server now)

### Delete These Test Files:
- All test_*.py files in root
- All validate_*.py files
- All create_*.py files
- All debug_*.py files
- backtest/reporting/old_mess/* (entire folder)

## ✅ Current State (SYSTEM_TRUTH)

### What Works:
- Basic dashboard generation
- 1m, 5m, 15m, 1h timeframes (ETH data)
- Trade markers
- Portfolio metrics
- Self-validation system

### What's Missing/Broken:
- **1s timeframe has NaN values** (fixing)
- **No exchange-colored volume bars** (CRITICAL)
- **No portfolio panel** (just metrics)
- **No OI candlesticks** (needs implementation)
- **No exchange analytics page**
- **No strategy scoring visualization**

## 🎯 Success Criteria

The dashboard is complete when:
1. ✅ All 86,400 1s data points display without NaN
2. ✅ Volume bars show exchange colors in stacked format
3. ✅ Portfolio panel shows real-time evolution
4. ✅ OI displays as candlesticks with stats
5. ✅ Exchange analytics page shows detailed stats
6. ✅ All charts synchronized
7. ✅ Performance is smooth (<2s load time)

## 🚀 Next Steps

1. Fix the NaN issue in 1s data
2. Implement exchange-colored volume bars
3. Add portfolio panel to right side
4. Create OI candlestick visualization
5. Build exchange analytics page
6. Clean up all test files
7. Update documentation

## 📝 Notes

- OI data is available from server (213.136.75.120)
- Use BacktestDataLoader.aggregate_to_timeframe() for aggregation
- TradingView lightweight-charts for all visualizations
- Binary encoding for performance (base64 + struct)
- Keep visualization separate from strategy logic