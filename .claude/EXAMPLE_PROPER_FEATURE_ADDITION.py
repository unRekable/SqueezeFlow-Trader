"""
EXAMPLE: The RIGHT Way to Add a New Feature/Indicator
This shows how to add a hypothetical "Market Momentum Index" feature
that all components can use without manual updates everywhere
"""

# ============================================
# STEP 1: Add to indicator_config.py
# ============================================
"""
In /backtest/indicator_config.py:

@dataclass
class IndicatorConfig:
    # ... existing fields ...
    
    # New feature flag
    enable_market_momentum: bool = True
    momentum_threshold: float = 0.7  # Config parameters too!
    
    @classmethod
    def from_env(cls):
        return cls(
            # ... existing ...
            enable_market_momentum=os.getenv('BACKTEST_ENABLE_MOMENTUM', 'true').lower() == 'true',
            momentum_threshold=float(os.getenv('BACKTEST_MOMENTUM_THRESHOLD', '0.7'))
        )
"""

# ============================================
# STEP 2: Update data/pipeline.py to load it
# ============================================
"""
In /data/pipeline.py:

def calculate_momentum_data(self, ohlcv_df):
    # Only calculate if enabled
    if not self.config.enable_market_momentum:
        return pd.Series(dtype=float)  # Return empty
    
    # Calculate momentum
    return calculate_momentum(ohlcv_df)

def get_complete_dataset(self):
    # ... existing code ...
    
    # Add momentum to dataset if enabled
    dataset['market_momentum'] = pd.Series(dtype=float)
    if self.config.enable_market_momentum:
        dataset['market_momentum'] = self.calculate_momentum_data(ohlcv_df)
    
    return dataset
"""

# ============================================
# STEP 3: Use in strategy phases
# ============================================
"""
In /strategies/squeezeflow/components/phase4_scoring.py:

def calculate_score(self):
    # Get config (could be passed in or imported)
    config = get_indicator_config()
    
    # Use momentum if enabled
    momentum_bonus = 0
    if config.enable_market_momentum:
        momentum = dataset.get('market_momentum', pd.Series())
        if not momentum.empty:
            latest_momentum = momentum.iloc[-1]
            if latest_momentum > config.momentum_threshold:
                momentum_bonus = 1.0  # Add to score
    
    scores['momentum'] = momentum_bonus
"""

# ============================================
# STEP 4: Handle in visualization
# ============================================
"""
In /backtest/reporting/interactive_strategy_visualizer.py:

def prepare_visualization_data(self):
    config = get_indicator_config()
    
    # Only process momentum if enabled
    if config.enable_market_momentum:
        momentum_data = self._prepare_momentum_chart(dataset)
    else:
        momentum_data = None
    
    return {
        'momentum': momentum_data,
        # ... other data
    }
"""

# ============================================
# STEP 5: Add integration test
# ============================================
"""
In test_config_integration.py:

# Test momentum can be disabled
os.environ['BACKTEST_ENABLE_MOMENTUM'] = 'false'
config = get_indicator_config()
assert config.enable_market_momentum == False

# Test pipeline respects it
pipeline = DataPipeline()
dataset = pipeline.get_complete_dataset(...)
assert dataset['market_momentum'].empty  # Should be empty when disabled
"""

# ============================================
# KEY PRINCIPLES DEMONSTRATED:
# ============================================
"""
1. ONE place to control feature (indicator_config.py)
2. ALL components check config before using feature
3. Sensible defaults when disabled (empty series, 0 scores)
4. Environment variables follow naming convention
5. Integration test ensures it works

Now when you want to disable momentum:
    export BACKTEST_ENABLE_MOMENTUM=false
    
And EVERYTHING automatically adapts - no hunting through files!
"""

# ============================================
# ANTI-PATTERN (What NOT to do):
# ============================================
"""
❌ WRONG - Hardcoding in multiple places:

# In phase2_divergence.py
if MOMENTUM_ENABLED:  # Where does this come from?
    calculate_momentum()

# In phase4_scoring.py  
USE_MOMENTUM = True  # Different variable!
if USE_MOMENTUM:
    add_momentum_score()

# In visualizer.py
show_momentum = True  # Yet another flag!

This leads to:
- Inconsistent behavior
- Hours of debugging
- Forgotten updates
- Angry users (and developers)
"""

print("This is an example file showing the RIGHT way to add features.")
print("See .claude/ARCHITECTURAL_PRINCIPLES.md for more details.")