# CRITICAL: Open Interest Field Names

## ⚠️ IMPORTANT - READ THIS FIRST

The Open Interest data in InfluxDB uses different field names than expected:

### ✅ CORRECT Field Names
- **`open_interest`** - Contains the actual OI values in BTC/ETH/etc
- **Example**: 2,638,099 BTC for BTC on OKX

### ❌ INCORRECT Field Names (DO NOT USE)
- **`open_interest_usd`** - This field exists but is ALWAYS NULL
- **`open_interest_coin`** - This field doesn't exist

## Database Structure

```sql
-- Measurement
open_interest

-- Tags
symbol: 'BTC', 'ETH', etc.
exchange: 'OKX', 'BINANCE_FUTURES', 'BYBIT', 'TOTAL_AGG'

-- Fields
open_interest: float (the actual OI value in coin units)
open_interest_usd: float (always NULL - don't use)
funding_rate: float
```

## Correct Query Examples

```sql
-- ✅ CORRECT - Get BTC OI from OKX
SELECT mean(open_interest) as oi 
FROM open_interest 
WHERE symbol='BTC' AND exchange='OKX'
GROUP BY time(5m)

-- ❌ WRONG - This will return NULL
SELECT mean(open_interest_usd) as oi 
FROM open_interest 
WHERE symbol='BTC' AND exchange='OKX'
GROUP BY time(5m)
```

## Code Usage

```python
# ✅ CORRECT
query = f"""
SELECT mean(open_interest) as oi
FROM open_interest 
WHERE symbol = '{symbol}'
"""

# ❌ WRONG - Will always return None
query = f"""
SELECT mean(open_interest_usd) as oi_usd
FROM open_interest 
WHERE symbol = '{symbol}'
"""
```

## Files Already Fixed
- `/data/loaders/influx_client.py` - All OI queries use `open_interest`
- `/strategies/squeezeflow/components/oi_tracker_influx.py` - Fixed field references
- `/experiments/concept_validator.py` - Updated queries
- `/data/loaders/symbol_discovery.py` - Fixed COUNT queries

## Remember
**ALWAYS use `open_interest` field, NEVER use `open_interest_usd`**