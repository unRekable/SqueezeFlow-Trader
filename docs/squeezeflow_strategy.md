# SqueezeFlow Trading Strategy - Complete Manual Trading Methodology

## Overview

SqueezeFlow is a sophisticated CVD (Cumulative Volume Delta) divergence trading strategy that identifies market squeeze conditions through the analysis of spot vs perpetual CVD patterns. The strategy focuses on detecting market state transitions from chaos → equilibrium → directional bias.

## Core Philosophy

The strategy does NOT trade levels or thresholds - it trades **market state transitions** through **continuous opportunity evaluation**. Rather than blocking on insufficient conditions, it uses an additive 10-point scoring system to **determine when market conditions justify entry** based on current market conditions and CVD flow analysis.

## Market State Model

### Five Sequential Analysis Phases:

1. **CONTEXT ASSESSMENT** - Larger timeframe regime analysis (1h/4h)
2. **DIVERGENCE DETECTION** - Price-CVD imbalances identification (15m/30m)  
3. **RESET MONITORING** - Convergence exhaustion and explosive pattern detection (5m/15m)
4. **ABSORPTION CONFIRMATION** - Price action validation with volume support
5. **POSITION MANAGEMENT** - Flow-following exit strategy execution

**Phase Integration**: Each phase contributes scoring points (0-2.5 each) to a total opportunity assessment (0-10 points), ensuring the strategy **always returns the next best available trade** regardless of individual phase results.

### Strategy Flow Visualization

```
Market Analysis → Divergence Detection → Reset Monitoring → Entry Signal → Position Management
     ↓                    ↓                    ↓              ↓              ↓
Phase 1 (1h/4h)      Phase 2 (15m/30m)    Phase 3 (5m/15m)  Phase 4 (5m)   Phase 5 (All TF)
Context Assessment   Pattern Recognition    Exhaustion Watch   Scoring Decision   Flow Following
(Intel Gathering)    (Intel Gathering)     (Intel Gathering)  (0-10 Points)     (Exit Logic)
   
   ↓ ANALYZE           ↓ IDENTIFY            ↓ DETECT          ↓ SCORE        ↓ MANAGE
   
Squeeze Environment → CVD Imbalance → Convergence Exhaustion → Entry Decision → Trade Management
(Market Context)     (Opportunity)      (Timing Signal)       (0-10 Points)    (Exit Rules)

Total Score: 0-10 Points → Trade Signal if ≥4 Points, Otherwise Wait
```

**Continuous Evaluation Rules:**
- Phases 1-3 gather market intelligence without scoring
- Phase 4 consolidates all findings into a 0-10 point decision
- No blocking gates - all phases execute to build complete picture
- Total score determines trade quality tier and confidence level
- Strategy only returns signals when scoring meets minimum thresholds (4+ points)
- Exit conditions monitored continuously during position management

## Continuous Evaluation Scoring System

### 10-Point Opportunity Assessment

The strategy uses an additive scoring system where Phase 4 consolidates all intelligence to determine the next best trade available:

**Intelligence Gathering (Phases 1-3):**
- **Phase 1 - Context Assessment**: Identifies market regime (bull/bear squeeze environment)
- **Phase 2 - Divergence Detection**: Locates price-CVD imbalances (opportunities)
- **Phase 3 - Reset Detection**: Monitors convergence exhaustion (timing signals)

**Scoring Decision (Phase 4):**
All intelligence from Phases 1-3 is consolidated into four scored criteria (10 points total):
- **CVD Reset Deceleration**: 3.5 points (validates reset quality from Phase 3)
- **Absorption Confirmation**: 2.5 points (price action validation)
- **Failed Breakdown Pattern**: 2.0 points (pattern strength from Phase 2)
- **Directional Bias Confirmation**: 2.0 points (incorporates Phase 1 context)

**Signal Generation Logic:**
- **8-10 points**: Premium quality signals (high confidence entries)
- **6-7 points**: High quality signals (standard confidence entries)
- **4-5 points**: Medium quality signals (constructed from best available assessment)
- **0-3 points**: No signal - insufficient market conditions

**Key Principle**: The strategy only generates signals when market conditions meet minimum quality thresholds. Low-scoring setups (0-3 points) indicate conditions are insufficient for trading and should be passed over.

## Strategy Workflow

### Phase 1: Larger Context Assessment ("Zoom Out")

**Objective**: Determine the dominant squeeze environment

**Process**:
- Examine longer timeframes (30m, 1h, 4h) to identify overall market bias
- Determine if market is in SHORT_SQUEEZE or LONG_SQUEEZE environment
- Look for sustained volume accumulation patterns in one direction
- Larger volume imbalances = longer squeeze duration potential

**Key Question**: "Is this a short squeeze environment or long squeeze environment?"

### Phase 2: Divergence Detection

**Objective**: Identify price-CVD imbalances that create trading opportunities

**Pattern Recognition**:
- **Long Setup**: Price stable/rising + Spot CVD up + Perp CVD down (shorts accumulating, price not falling)
- **Short Setup**: Price stable/falling + Spot CVD down + Perp CVD up (longs accumulating, price not rising)
- **Volume Pattern**: Look for divergences that are notably larger than the last few hours of activity (e.g., if recent swings were 200M, a 400M+ divergence is significant)

**Critical Insight**: Looking for moments when "price is unbalanced relative to what CVD is saying"

### Phase 3: Reset Detection - Convergence-Based Exhaustion System

**Objective**: Identify market exhaustion through convergence patterns and equilibrium restoration

**Reset Type A - Convergence Exhaustion Pattern (CVD-Predictive)**:
- **Setup**: Price movement driven by unbalanced CVD (PERP-dominated or SPOT-dominated)
- **Convergence Signal**: SPOT and PERP CVD begin converging as the lagging side catches up
- **Key Pattern**: Price stagnation despite active CVD convergence
- **Market Psychology**: Market seeking equilibrium after imbalanced trend exhaustion
- **Detection Criteria**: CVD convergence + price momentum loss = Reset detected

**Reset Type B - Explosive Confirmation**:
- **Trigger**: Large price movement following convergence exhaustion pattern
- **Pattern**: Major price move accompanied by supporting CVD movement
- **Market Psychology**: Equilibrium restoration through volatility shakeout
- **Detection Criteria**: Significant price/CVD movement = Reset confirmed

**Reset Recognition Logic**:
- Monitor for unbalanced movement patterns (one CVD type dominates)
- Detect convergence while price momentum exhausts
- Either convergence exhaustion OR explosive follow-up triggers reset
- Both patterns indicate equilibrium restoration moments

**Multi-Timeframe Integration**:
- **Cross-Validation**: Checks 5m, 15m, 30m timeframes for aligned convergence
- **Signal Amplification**: Multi-timeframe alignment creates stronger entry signals
- **Momentum Analysis**: Sustained momentum patterns and deceleration detection

### 📊 Trading Interpretation - What Reset Detection Looks Like on Your Chart

**Type A in Practice**: You'll see price going sideways after a directional move, while the CVD lines start coming together. It looks like the market is "taking a breather" before deciding the next direction.

**Type B in Practice**: After Type A convergence, you might see a sudden spike/dump that quickly reverses. This is the market's way of shaking out weak hands before the real move.

**Quick Recognition**: If PERP CVD led a rally but now SPOT is catching up while price stalls = Reset incoming.

### Phase 4: Scoring Decision & Entry Signal

**Objective**: Enter when reset momentum exhausts and new directional bias begins

**Important Note**: Phase 4 is where all market intelligence gathered in Phases 1-3 gets consolidated into a single scored decision. The four criteria below reference and validate findings from the earlier analytical phases.

**Entry Criteria Priority System (Total Score: 10 points)**:

### Scoring Contribution System (All Conditions Contribute Points)

1. **CVD Reset Deceleration** (3.5 points) - **CRITICAL**
   - CVD stops moving as much during reset attempts
   - Second test of low shows less CVD movement than first test
   - Both CVD and price showing signs of "compression down there"
   - **Why Important**: Indicates exhaustion momentum is ending

2. **Absorption Candle Confirmation** (2.5 points) - **HIGH PRIORITY**
   - Candles that don't close under reset low
   - Close higher than open (buyers stepping in)  
   - Wicks below but close above (selling pressure absorbed)
   - Volume confirmation on wick rejection
   - **Why Important**: Confirms real participants stepping in

### Supporting Criteria (Nice to Have - Additional confidence)

3. **Failed Breakdown Pattern** (2.0 points) - **MEDIUM PRIORITY**
   - Multiple candles try to break range but fail (create wicks)
   - Each attempt shows decreasing CVD movement
   - Market equilibrium forming after chaos
   - **Why Important**: Shows multiple failed attempts strengthen setup

4. **Directional Bias Confirmation** (2.0 points) - **SUPPORTING**
   - CVD must start going up "a bit"
   - Price must follow CVD direction  
   - Both CVDs moving in same direction acceptable
   - **Why Important**: Early indication of new trend direction

### Scoring Decision Matrix (Phase 4):
Based on consolidated intelligence from all phases:
- **8+ points**: Premium entries (all phases align perfectly)
- **6-7 points**: High-confidence entries (strong market setups)
- **4-5 points**: Medium-confidence entries (decent opportunities constructed from available conditions)
- **0-3 points**: No trade - conditions insufficient (wait for better setup)

**Trading Decision Rules:**
- **Score < 4**: No trade - wait for better conditions
- **Score 4-5**: Consider entry with reduced position size
- **Score 6-7**: Standard entry with normal position size  
- **Score 8+**: High-conviction entry, consider larger size

**Note**: The strategy requires a minimum score of 4 points to generate a trade signal. Below this threshold, market conditions are considered insufficient for entry.

**Entry Timing**: Enter when both CVD and price stop falling and begin to stabilize - this marks the exhaustion point and potential reversal

### Complete SqueezeFlow Trading Workflow - Convergence Exhaustion Cycle

**Step 1: Unbalanced Movement Recognition**
- Identify price trends driven primarily by one CVD type (SPOT or PERP dominance)
- Track which market segment is driving the current move
- Monitor for sustained imbalanced flow patterns

**Step 2: Convergence Exhaustion Detection**
- **Convergence Pattern**: Lagging CVD type begins catching up to dominant type
- **Exhaustion Signal**: Price momentum stalls despite ongoing convergence
- **Reset Type A**: This convergence-while-stalling pattern = Predictive reset
- **Market Psychology**: Equilibrium seeking after imbalanced trend completion

**Step 3: Explosive Confirmation (Optional)**
- **Reset Type B**: Large price movement may follow convergence exhaustion
- **Pattern**: Major move with supporting CVD activity
- **Confirmation**: Validates the equilibrium restoration process

**Step 4: Market Rebalancing Assessment**
- Previous CVD imbalance has been corrected through convergence
- Weak participants (who created dominance) have been shaken out
- Market establishing fresh equilibrium conditions

**Step 5: New Opportunity Evaluation**
- Assess longer-term conditions from newly balanced starting point
- Evaluate if fresh CVD patterns support new directional bias
- Check setup alignment with dominant squeeze environment

**Step 6: Entry Execution**
- Enter positions from equilibrium-restored conditions
- Trade new trend development from balanced foundation
- Monitor ongoing CVD patterns for trend continuation validation

**Example Cycle**:
```
PERP-Driven Rally → SPOT Convergence → Price Stagnation → Reset Detected
→ Possible Explosive Move → Market Rebalanced → New Long Opportunity
```

### Phase 5: Position Management

**Objective**: Follow the order flow until conditions change

**Management Principles**:
- **NO fixed profit targets** - follow the squeeze until invalidation
- **NO fixed stop losses** - no predetermined % or $ loss limits, exit based on condition changes
- **Dynamic exits only** - exit when market structure invalidates entry thesis (e.g., range breaks)
- Track CVD trends from entry point continuously
- Compare current CVD to entry baseline
- Monitor larger timeframe validity

**Exit Conditions**:

1. **Flow Reversal Pattern**:
   - SPOT CVD declining (selling) while PERP CVD elevated (buying)
   - Price struggling despite CVD divergence
   - "Dangerous situation" - real money leaving, leverage will get squeezed

2. **Range Break**:
   - Price breaks below entry range/reset low
   - Signals reset wasn't complete, needs bigger move
   - Note: This is structural invalidation, not a fixed stop loss

3. **Larger Timeframe Invalidation**:
   - Dominant squeeze environment changes
   - Exit or secure profits if longer timeframe turns against position

4. **CVD Trend Invalidation**:
   - Both CVDs start declining together
   - Squeeze conditions weakening/reversing

## Timeframe Hierarchy

### Timeframe Assignments by Phase:
- **Phase 1 (Context)**: 1h/4h - Identify dominant squeeze environment
- **Phase 2 (Divergence)**: 15m/30m - Spot significant CVD imbalances
- **Phase 3 (Reset)**: 5m/15m - Detect exhaustion patterns
- **Phase 4 (Entry)**: 1m/5m - Precise entry timing and absorption patterns
- **Phase 5 (Management)**: All timeframes - Monitor for exit conditions

### How to Use Multiple Timeframes:
1. **Start Wide**: Begin analysis with 4h/1h for market context
2. **Zoom In**: Progress through shorter timeframes as you move through phases
3. **Confirm Alignment**: Best trades show agreement across multiple timeframes
4. **Monitor All**: During position management, watch all timeframes for changes

### Practical Application:
- **Strong Setup**: All timeframes align (4h context + 30m divergence + 5m reset)
- **Medium Setup**: Most timeframes align, one may be neutral
- **Weak Setup**: Only short timeframes show signals (be cautious)

## Qualitative Pattern Recognition System

### Dynamic Market Baseline Discovery

The strategy employs qualitative pattern recognition that adapts to each market's current personality, eliminating fixed numerical thresholds entirely.

**Core Principle**: Read pattern shapes and behaviors relative to current market conditions, not absolute CVD values.

### Pattern Recognition Approach

**Daily Market Baseline Discovery**:
1. Analyze recent CVD activity patterns (last 24 hours)
2. Determine current market "normal" behavior and typical swing sizes
3. Identify current noise level vs significant movement patterns
4. Establish pattern recognition context for current trading session

**Qualitative Pattern Analysis**:
- **Trend Shape Recognition**: Focus on convergence curvature and momentum characteristics
- **Relative Momentum Assessment**: Compare current behavior to recent behavior patterns
- **Context-Aware Evaluation**: Pattern significance relative to current market activity
- **Adaptive Pattern Scoring**: Score patterns based on current market baseline

### Market Adaptation Examples

**BTC Market Analysis**:
- **Quiet Day**: Normal CVD swings 100-300M → Look for convergence patterns within this range
- **Volatile Day**: Normal CVD swings 400-800M → Same convergence patterns, different scale
- **Pattern Focus**: Convergence speed, exhaustion signs, momentum decline relative to recent activity

**ETH Market Analysis**:
- **Low Activity**: CVD patterns around 50-150M → Smaller scale, same pattern recognition
- **High Activity**: CVD patterns around 200-500M → Larger scale, identical pattern logic
- **Key Insight**: Pattern shape and behavior matter, not the absolute numbers

**Universal Pattern Recognition Principles**:
- Monitor recent typical CVD activity to establish daily baseline
- Identify when current patterns deviate meaningfully from baseline
- Look for convergence exhaustion relative to current market scale
- Focus on trend behavior, momentum changes, and pattern completion timing

**Benefits**:
- **True Market Adaptation**: Responds to each market's current personality
- **No Fixed Thresholds**: Pure qualitative pattern recognition
- **Scale Independence**: Same logic works across all market sizes
- **Daily Calibration**: Fresh baseline discovery each trading session


## CVD Calculation Methodology

### Technical Implementation
The strategy uses industry-standard Cumulative Volume Delta calculation:

1. **Per-Minute Volume Delta**: Buy Volume - Sell Volume for each minute
2. **Cumulative CVD**: Running total (cumsum) of all minute deltas over time
3. **Multi-Exchange Aggregation**: 
   - All SPOT exchanges aggregated into single SPOT CVD
   - All PERP exchanges aggregated into single FUTURES CVD
   - Uses `exchange_mapper.py` for automatic SPOT/PERP classification

### Pre-Aggregated Timeframes in InfluxDB
The database automatically maintains higher timeframes via Continuous Queries:

**Base Data**: `aggr_1m.trades_1m` - Raw 1-minute data from exchanges

**Continuous Queries** (Auto-updated):
- `cq_5m`: Creates `aggr_5m.trades_5m` - 5-minute aggregation
- `cq_15m`: Creates `aggr_15m.trades_15m` - 15-minute aggregation
- `cq_30m`: Creates `aggr_30m.trades_30m` - 30-minute aggregation
- `cq_1h`: Creates `aggr_1h.trades_1h` - 1-hour aggregation
- `cq_4h`: Creates `aggr_4h.trades_4h` - 4-hour aggregation

**Key Fields Maintained**:
- OHLC: `open`, `high`, `low`, `close`
- Volume: `vbuy`, `vsell` (for CVD calculation)
- Counts: `cbuy`, `csell`, `lbuy`, `lsell`

### Implementation References
- **CVD Calculation**: See `utils/cvd_analysis_tool.py`
- **Exchange Classification**: See `data/processors/exchange_mapper.py`
- **Database Structure**: InfluxDB with retention policies and continuous queries

### Why This Matters
- **Performance**: Direct queries to pre-aggregated data (10x faster)
- **Consistency**: Same CVD values in backtest and live trading
- **No Resampling**: Query `aggr_15m.trades_15m` directly for 15m CVD
- **Automatic Updates**: Continuous queries maintain all timeframes

## Risk Management

### Position Sizing:
- Based on confidence in larger timeframe context
- Reduce size if only short-term setups available
- Increase size when all timeframes align

### Exit Priority:
1. **Secure Profits**: If larger timeframe suggests holding but local conditions deteriorating
2. **Flow Following**: Primary exit method - follow order flow until conditions change
3. **Range Defense**: Exit if price breaks entry range significantly

## Key Success Factors

### Pattern Recognition Skills:
- Visual identification of CVD divergence/convergence patterns
- Recognition of absorption candles and wick patterns
- Understanding market state transitions and exhaustion patterns
- Reading order flow narratives and equilibrium seeking behavior

### Contextual Awareness:
- Larger timeframe squeeze environment assessment
- Volume accumulation patterns and imbalanced flow recognition
- Market participant behavior analysis (SPOT vs PERP dynamics)
- Momentum exhaustion signals and convergence timing

### Adaptive Execution:
- Flexible entry based on available convergence exhaustion patterns
- Dynamic exit based on changing flow conditions
- Reset pattern recognition across multiple timeframes
- No rigid rules - trade the current market narrative

## Success Metrics

### Trade Quality Indicators:
- Entry near local bottoms/tops after reset patterns
- Holding trades until natural condition changes
- High win rate through selective convergence exhaustion criteria
- Large winners relative to small losers

### Reset Detection Quality Indicators:
- Accurate convergence exhaustion pattern identification
- Proper distinction between equilibrium-seeking vs random convergence
- Timing precision for both predictive and confirmatory reset types
- Market rebalancing assessment accuracy after reset patterns
- Successful baseline reset timing for optimal measurement reference points

### System Performance:
- Consistent profitability across different market conditions
- Effective risk management through flow following
- Minimal drawdowns through selective entries
- Successful reset pattern recognition and exploitation

## Common Mistakes to Avoid

### 1. **Fixed Threshold Trading** ❌
**Mistake**: Using static CVD levels like "buy when SPOT CVD > 50M"
**Why Wrong**: Markets have different personalities each day - 50M could be noise today but significant tomorrow
**Correct Approach**: Use qualitative pattern recognition that adapts to current market conditions
**Example**: Same convergence pattern works whether today's range is 100M or 500M

### 2. **Premature Exits** ❌
**Mistake**: Exiting on first +2% profit instead of following flow changes
**Why Wrong**: Misses the large moves that make the strategy profitable long-term
**Correct Approach**: Exit only when CVD flow changes or larger timeframe invalidates
**Example**: Hold LONG until SPOT CVD starts declining or range breaks below entry

### 3. **Missing Reset Patterns** ❌
**Mistake**: Entering immediately on CVD divergence without waiting for reset
**Why Wrong**: Divergence can continue much longer than expected - premature entries
**Correct Approach**: Wait for convergence exhaustion or explosive confirmation
**Example**: See PERP-driven rally, wait for SPOT convergence + price stagnation

### 4. **Misreading Convergence** ❌
**Mistake**: Trading any CVD convergence as a reset signal
**Why Wrong**: Random convergence during sideways action ≠ exhaustion-indicating convergence
**Correct Approach**: Only trade convergence after unbalanced moves with price stagnation
**Example**: SPOT/PERP converging during sideways chop = noise, not meaningful reset

### 5. **Skipping Phase Validation** ❌
**Mistake**: Rushing to entry without completing all 5 phases methodically
**Why Wrong**: Incomplete analysis leads to low-quality setups and poor results
**Correct Approach**: Systematically validate each phase before advancing to next
**Example**: Don't skip absorption candle confirmation even if CVD patterns look perfect

### 6. **Quantitative Thinking** ❌
**Mistake**: Expecting same numerical thresholds to work across different market conditions
**Why Wrong**: This is qualitative pattern recognition, not quantitative algorithm
**Correct Approach**: Read market personality daily and adapt pattern recognition accordingly
**Example**: BTC quiet day (100M swings) vs volatile day (500M swings) - same patterns, different scales

### 7. **Trading Poor Setups** ❌
**Mistake**: Taking trades when score is below 4 points just to be in the market
**Why Wrong**: Low scores indicate insufficient edge - better to wait for quality setups
**Correct Approach**: Only trade when scoring meets minimum threshold (4+ points)
**Example**: 2/10 score = no trade, wait. 5/10 score = small position. 8/10 score = full position

## Data Infrastructure

### Multi-Timeframe Architecture

The strategy utilizes a sophisticated multi-timeframe data infrastructure built on InfluxDB with parallel query execution for optimal performance.

**Parallel Query System**:
- Multiple timeframe queries executed concurrently
- ~80% performance improvement over sequential queries
- Full InfluxDB compatibility with robust error recovery

**Automated Data Aggregation**:
- Base Data: 1-minute trading data from aggregation server
- Continuous Queries: Automated aggregation into 5m, 15m, 30m, 1h, 4h timeframes
- Complete OHLCV + Volume data maintenance for CVD calculations

**Performance Benefits**:
- 10x faster queries through pre-aggregated data
- Memory efficiency with no real-time resampling overhead
- Data consistency between backtest and live trading
- Automatic scaling for new timeframes

## Debug Infrastructure

### Comprehensive Logging System

The strategy includes industrial-standard debug logging for systematic analysis:

**Multi-Channel Logging**:
- Console logging for real-time monitoring
- File logging with detailed context
- Debug logging with technical implementation details
- CSV logging for structured signal analysis

**Strategy Phase Tracking**:
- Market regime detection with timeframe analysis
- CVD divergence detection with database query logging
- Reset detection with convergence exhaustion analysis
- Entry signal generation with scoring details
- Exit signal monitoring with flow analysis

**Troubleshooting Support**:
- Zero trades debugging workflow
- Data pipeline verification procedures
- Strategy state analysis capabilities
- Database connectivity validation

---

*This methodology represents the complete manual trading approach distilled from extensive market observation and convergence exhaustion pattern recognition. Success requires developing visual pattern recognition skills and understanding market participant behavior through CVD analysis and equilibrium restoration timing.*

*For implementation changes and version history, see `SqueezeFlow_Changelog.md`.*