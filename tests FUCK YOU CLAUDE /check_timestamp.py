"""Check timestamp issues"""

from datetime import datetime

# The timestamp from dashboard
ts = 1754819594

# Convert to datetime
dt = datetime.fromtimestamp(ts)
print(f"Timestamp {ts} converts to: {dt}")
print(f"Year: {dt.year}")

# This looks like year 2025 timestamps!
# Let's check what August 10, 2025 should be
aug_10_2025 = datetime(2025, 8, 10, 0, 0, 0)
print(f"\nAugust 10, 2025 timestamp should be: {aug_10_2025.timestamp()}")

# The issue: timestamp looks like it's from year 2025 but interpreted wrong!
print(f"\nISSUE: Dashboard timestamp {ts} is from year {dt.year}")
print("This is a future date - charts might not handle this correctly!")