# Backtest Performance Analysis

## Date: 2025-08-09
## Status: 🔴 CRITICAL PERFORMANCE ISSUE

---

## 🎯 THE PROBLEM

**1-minute data, 15 days**: ~2 minutes (180 windows/second)
**1-second data, 1 day**: ~100 minutes projected (14 windows/second)

That's **12x slower** despite having only 4x more data points!

---

## 🔍 ROOT CAUSE ANALYSIS

### Data Volume Comparison

#### Previous (1m data, 15 days):
- Windows: 21,600 (15 × 24 × 60 / 5-minute steps)
- Window size: 4 hours = 240 data points
- Total data points: ~5.4 million operations

#### Current (1s data, 1 day):
- Windows: 86,399 (24 × 60 × 60 / 1-second steps)
- Window size: 1 hour = 3,600 data points
- Total data points: ~311 million operations (57x more!)

### The REAL Problem: Redundant Data Slicing

With 1-hour windows and 1-second steps:
- Window 1: 00:00:00 - 01:00:00 (3,600 points)
- Window 2: 00:00:01 - 01:00:01 (3,599 same points + 1 new)
- Window 3: 00:00:02 - 01:00:02 (3,598 same points + 2 new)

**We're re-slicing 99.97% identical data 86,399 times!**

---

## 📊 PERFORMANCE BOTTLENECKS

### 1. Data Slicing Overhead (engine.py:1342)
```python
# This happens 86,399 times!
windowed_data = full_data.loc[window_start_tz:window_end_tz]
```
- Each slice creates a new pandas object
- Timezone conversions on every slice
- Memory allocation/deallocation overhead

### 2. Strategy Processing Overhead
- Phase 1-4 run on nearly identical data
- CVD calculations on 3,600 points each time
- Pattern detection on overlapping windows

### 3. Memory Churn
- Creating/destroying pandas objects
- Garbage collection overhead
- Cache misses due to data copying

---

## ✅ SOLUTIONS

### Option 1: Sliding Window Buffer (BEST)
Instead of re-slicing, maintain a rolling buffer:
```python
# Initialize once
buffer = deque(maxlen=3600)

# Each step, just add one point
buffer.append(new_data_point)
```
- O(1) updates instead of O(n) slicing
- No memory allocation per window
- 100x+ speedup potential

### Option 2: Larger Step Sizes
- Use 10-second or 30-second steps
- Reduces windows from 86,399 to 8,640 or 2,880
- Still evaluates all data, just less frequently

### Option 3: Smart Caching
- Cache overlapping data between windows
- Only update the changed portion
- Reuse calculations from previous window

### Option 4: Vectorized Window Processing
- Process multiple windows in parallel
- Batch similar operations
- Use numpy stride tricks for efficient windowing

---

## 📈 EXPECTED IMPROVEMENTS

### Current Performance:
- 14 windows/second
- 100+ minutes for 1 day

### With Optimizations:
- 500+ windows/second (sliding buffer)
- 2-3 minutes for 1 day
- Similar to 1m performance

---

## 💡 WHY THE DIFFERENCE?

### 1-Minute with 5-Minute Steps:
- Each window has 0% overlap with previous
- Clean, distinct data slices
- Efficient processing

### 1-Second with 1-Second Steps:
- Each window has 99.97% overlap
- Massive redundancy
- Inefficient re-processing

---

## 🚀 IMMEDIATE ACTIONS

1. **Quick Fix**: Increase step size to 10-30 seconds
   - Reduces windows by 10-30x
   - Immediate speedup

2. **Proper Fix**: Implement sliding window buffer
   - Eliminates redundant slicing
   - O(1) window updates

3. **Strategy Optimization**: Cache calculations
   - Reuse CVD calculations
   - Incremental pattern updates

---

## 📝 IMPLEMENTATION PRIORITY

1. **First**: Run with 30-second steps (immediate relief)
2. **Second**: Test scoring threshold changes  
3. **Third**: Implement sliding window buffer for production

---

## 🎯 KEY INSIGHT

The backtest wasn't designed for 1-second stepping with hour-long windows. The architecture assumes non-overlapping or minimally overlapping windows. With 99.97% overlap, we need a different approach - either larger steps or smarter windowing.