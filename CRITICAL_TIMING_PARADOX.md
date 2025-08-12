# The Aggregation-Scoring Timing Paradox
> Critical architectural decision for Veloce and understanding of SqueezeFlow's approach

## The Fundamental Problem

When we aggregate data (into candles, timeframes, or chunks), we create an inherent timing conflict between signal quality and execution speed.

### The Core Paradox

**Complete Data = High Confidence but Late Entry**
**Incomplete Data = Fast Entry but Low Confidence**

## Part 1: How Aggregation Affects Trade Execution Timing

### The Latency Formula
```
Execution Latency = Aggregation Period + Processing Time

For 1-minute candles:
- Best case: Event at second 59 → 1 second lag
- Worst case: Event at second 1 → 59 second lag  
- Average: 30 second lag
```

### Concrete Examples

#### Using 1-Minute Aggregation
```python
# Timeline of a trade signal:
12:00:00 - Divergence starts forming
12:00:30 - Divergence clearly visible  
12:00:59 - Candle almost complete
12:01:00 - Candle closes, strategy runs
12:01:01 - Signal generated, trade executes
# Result: 61 seconds from divergence start to trade!
```

#### Using 1-Second Data
```python
# Timeline of same signal:
12:00:00 - Divergence starts forming
12:00:01 - Strategy checks (partial data)
12:00:02 - Probabilistic entry triggered
# Result: 2 seconds from divergence to trade!
```

### The Hidden Impact on Strategy Performance

This timing difference compounds:
- 30-second average delay on entry
- 30-second average delay on exit  
- = 60 seconds of missed movement per trade
- On a 2% move, that could be 0.5% of profit lost!

---

## Part 2: How This Affects Scoring Systems

### The Scoring Timing Paradox

Our scoring systems typically require completed patterns:

```python
def calculate_traditional_score():
    # Each component needs different amounts of history:
    squeeze_score = detect_squeeze()        # Needs 20 completed candles
    divergence_score = detect_divergence()  # Needs 100 completed candles
    reset_score = detect_reset()            # Needs confirmed reversal
    
    total_score = squeeze_score + divergence_score + reset_score
    # By the time score = 8... the move already happened!
```

### The Fundamental Conflict

1. **Wait for Confirmation:**
   - ✅ High score accuracy (8/10)
   - ❌ Entry after 1-2% move already happened
   - ❌ Worse risk/reward ratio

2. **Act on Partial Data:**
   - ✅ Better entry price
   - ✅ Capture full move
   - ❌ Lower score confidence (5/10)
   - ❌ More false signals

---

## How SqueezeFlow Currently Handles This

After analyzing the codebase:

### SqueezeFlow's Approach: "Complete Candle Confirmation"

```python
# From backtest/engine.py and strategy.py
def process_candle(self, candle, historical_data):
    # SqueezeFlow waits for COMPLETE candles
    # It gets ALL historical data up to current candle
    # But only acts AFTER candle closes
    
    # This means:
    # - In 1s mode: 1 second delay (good!)
    # - In 5m mode: Up to 5 minute delay (bad!)
    # - Strategy sees complete patterns only
```

**SqueezeFlow's Scoring:**
```python
# From strategies/squeezeflow/strategy.py:398-420
# Calculate position sizing based on score
position_size_factor = self.config.get_position_size_factor(total_score)

# Score thresholds (from config.py:74-79):
position_size_by_score = {
    "0-3.9": 0.0,    # No trade
    "4-5": 0.5,      # 50% size
    "6-7": 1.0,      # 100% size
    "8+": 1.5        # 150% size
}
```

**Key Issues with SqueezeFlow's Approach:**
1. **Binary decision** - Either enter or don't, based on threshold
2. **No temporal consideration** - Fresh vs stale signals treated same
3. **Waits for complete confirmation** - Misses optimal entry
4. **Fixed position sizing** - Doesn't scale smoothly with confidence

---

## Professional Solutions to This Paradox

### Solution 1: "Confidence Decay" (Renaissance Technologies Style)
```python
class ConfidenceDecayScoring:
    """Score decreases as pattern ages"""
    
    def calculate_entry_score(self, pattern_age_seconds):
        base_score = 8.0
        decay_factor = 0.5 ** (pattern_age_seconds / 10)  # Half-life: 10s
        return base_score * decay_factor
        
        # At 0s: Score = 8.0 (ENTER NOW!)
        # At 10s: Score = 4.0 (Getting stale)
        # At 20s: Score = 2.0 (Too late)
```

### Solution 2: "Probabilistic Entry" (Two Sigma Approach)
```python
class ProbabilisticEntry:
    """Enter based on probability of pattern completing"""
    
    def on_tick(self, tick):
        # Calculate probabilities, not certainties
        p_squeeze = self.probability_squeeze_completes()      # 0.7
        p_divergence = self.probability_divergence_holds()    # 0.6
        p_reset = self.probability_reset_confirms()           # 0.8
        
        expected_score = (p_squeeze * 3.0 + 
                         p_divergence * 3.0 + 
                         p_reset * 2.0)
        
        if expected_score > 5.0:  # Enter on EXPECTED value
            position_size = expected_score / 10.0
            self.enter(size=position_size)
```

### Solution 3: "Cascade Entry" (Citadel Style)
```python
class CascadeEntry:
    """Build position in stages as confirmation increases"""
    
    def evaluate_entry(self):
        if immediate_score > 3 and position == 0:
            self.enter(size=0.33)  # Fast entry, low confidence
            
        elif partial_score > 5 and position == 0.33:
            self.add(size=0.33)    # Add on partial confirmation
            
        elif full_score > 7 and position == 0.66:
            self.add(size=0.34)    # Complete on full confirmation
```

### Solution 4: "Dual-Track Scoring" (Jane Street Method)
```python
class DualTrackScoring:
    """Separate systems for entry (fast) and confirmation (slow)"""
    
    def on_tick(self, tick):
        # Fast ML model for entry decision
        entry_score = self.fast_predictor.predict(tick)  # 100ms
        
        if entry_score > 6:
            self.enter()
            self.schedule_confirmation_check(5_seconds)
    
    def confirm_entry(self):
        # Full analysis for position management
        confirm_score = self.full_analysis.calculate()  # 5 seconds
        
        if confirm_score < 4:
            self.exit()  # Exit if not confirmed
```

---

## Recommended Solution for Veloce: "Temporal Score Decomposition"

### The Elegant Approach
```python
class VeloceTemporalScoring:
    """
    Decompose score into time-sensitive components
    Weight them differently for entry vs sizing
    """
    
    def calculate_composite_score(self, data):
        # FAST COMPONENTS (calculated on incomplete data)
        momentum_score = self.calc_momentum(data.last_10_ticks)     # 0-3 pts
        volume_spike = self.detect_volume_spike(data.current_tick)  # 0-2 pts
        
        # MEDIUM COMPONENTS (probabilistic on partial data)  
        divergence_forming = self.calc_divergence_probability(data.last_100_ticks)  # 0-3 pts
        
        # SLOW COMPONENTS (need full confirmation)
        squeeze_confirmed = self.calc_squeeze(data.last_20_candles)  # 0-2 pts
        
        # Entry decision weights fast components more
        entry_score = (momentum_score * 1.0 + 
                      volume_spike * 1.0 +
                      divergence_forming * 0.7 +
                      squeeze_confirmed * 0.3)
        
        # Position sizing uses all components equally
        confidence_score = (momentum_score + volume_spike + 
                           divergence_forming + squeeze_confirmed)
        
        return {
            'entry_trigger': entry_score > 4,        # Fast decision
            'position_size': confidence_score / 10.0  # Size by full score
        }
```

### Why This Solves Both Problems

1. **Respects time value** - Fresh signals acted on quickly
2. **Maintains risk management** - Position sized by confidence
3. **Reduces latency** - Entry on partial confirmation
4. **Preserves strategy logic** - Still uses all 5 phases
5. **Backtestable** - Can simulate on historical tick data

---

## Key Decision Points for Veloce

### Critical Questions to Answer:

1. **Entry Timing Philosophy**
   - Option A: Wait for full confirmation (SqueezeFlow approach)
   - Option B: Probabilistic early entry (Professional approach)
   - Option C: Cascade/staged entry (Hybrid approach)

2. **Score Calculation Method**
   - Option A: Binary thresholds (current)
   - Option B: Continuous probability (modern)
   - Option C: Temporal weighting (recommended)

3. **Position Sizing Approach**
   - Option A: Fixed tiers (current: 0%, 50%, 100%, 150%)
   - Option B: Continuous scaling (0-150% based on exact score)
   - Option C: Dynamic based on signal age and confirmation

4. **Data Processing Mode**
   - Option A: Complete candles only (current)
   - Option B: Tick-by-tick with running candles
   - Option C: Hybrid - ticks for entry, candles for confirmation

---

## Implementation Implications

### If we choose "Temporal Score Decomposition":

**Pros:**
- 10-50x faster entry (sub-second vs 30-60 seconds)
- Better entry prices
- Maintains risk management through position sizing
- Compatible with existing 5-phase strategy

**Cons:**
- More complex scoring system
- Harder to explain/debug
- Requires careful backtest validation
- May increase false signal rate (mitigated by smaller initial positions)

### If we keep "Complete Candle Confirmation":

**Pros:**
- Simple to understand and implement
- Already working in SqueezeFlow
- Lower false signal rate
- Easier to backtest

**Cons:**
- 30-60 second average latency
- Misses optimal entries
- Competitive disadvantage vs modern algos
- Limits effectiveness of 1-second data collection

---

## The Core Insight

**The aggregation period fundamentally determines minimum reaction time.**

But more subtly: **The scoring method determines whether we can act before that minimum.**

Professional trading systems solve this by:
1. **Separating entry decision from position sizing**
2. **Using probabilistic rather than binary scoring**
3. **Weighting time-sensitive components differently**
4. **Building positions gradually as confirmation increases**

The choice for Veloce is whether to:
- Maintain SqueezeFlow's simplicity but accept the latency
- Adopt professional methods for competitive advantage
- Find a middle ground that's both effective and maintainable