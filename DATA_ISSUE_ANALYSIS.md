# Data Issue Analysis: pH and Time Column Swapping

## Problem Identified

Based on user feedback and data analysis, there are **TWO critical issues** in the data processing:

### Issue 1: Column Misalignment for Materials with Fraction Column

**Affected Materials:** Barium (Ba) and potentially others

**Excel Structure:**
```
Fraction | pH | Time (days) | Cumulative release (mg/m²)
1        | 12.08 | 0.08 | 0.180315819
```

**What Code Currently Reads:**
- Column 2 (Fraction=1.0) → stored as pH ❌
- Column 3 (pH=12.08) → stored as Time_days ❌
- Column 4 (Time=0.08) → stored as Cumulative_Release_mg_m2 ❌
- Column 5 (Release=0.180) → LOST ❌

**What Code Should Read:**
- Column 2 (Fraction=1.0) → skip or map to time
- Column 3 (pH=12.08) → store as pH ✅
- Column 4 (Time=0.08) → store as Time_days ✅
- Column 5 (Release=0.180) → store as Cumulative_Release_mg_m2 ✅

### Issue 2: pH and Time Columns May Be Swapped

**Potential Issue:** Even for materials WITHOUT Fraction column, pH and Time might be in swapped order.

**Current Code Assumption:**
```python
if has_time_column:
    # Assumes: Column 2 = pH, Column 3 = Time, Column 4 = Release
    ph_val = data_row.iloc[2]
    time_val = data_row.iloc[3]
    cumulative_release = data_row.iloc[4]
```

**Possible Actual Excel Structure:**
```
pH | Time (days) | Cumulative release (mg/m²)
12.08 | 0.08 | 0.303
```

**OR (if swapped):**
```
Time (days) | pH | Cumulative release (mg/m²)
0.08 | 12.08 | 0.303
```

## Evidence from Data

### Aluminum (Al) - Appears Correct
```
pH: 12.08, 11.91, 11.84 (reasonable pH values for cement)
Time: 0.08, 1.0, 2.25 (reasonable time progression)
```

### Barium (Ba) - Definitely Wrong
```
pH: 1.0, 2.0, 3.0 (these are Fraction values!)
Time: 12.08, 11.91, 11.84 (these are pH values!)
Release: 0.08, 1.0, 2.25 (these are Time values!)
```

## Root Cause Analysis

The data processing code in `src/data_processing.py` has these issues:

1. **Column Detection Logic:**
   - Checks if "Time" is in header to determine structure
   - But doesn't verify the actual column positions
   - Assumes fixed column order (2, 3, 4) without checking

2. **Fraction Column Handling:**
   - When Fraction column exists, code should skip it
   - But code reads Fraction as pH when Time column is detected

3. **No Column Position Verification:**
   - Code doesn't find actual column indices for pH, Time, Release
   - Relies on fixed positions which may be wrong

## Solution Required

1. **Fix Column Detection:**
   - Find actual column indices for pH, Time, Fraction, Cumulative Release
   - Don't assume fixed positions
   - Handle both with and without Fraction column

2. **Verify Column Order:**
   - Check if pH comes before Time or vice versa
   - May need to swap based on actual Excel structure

3. **Reprocess All Data:**
   - Fix the data processing code
   - Reprocess entire Excel file
   - Verify all materials have correct data

4. **Retrain Model:**
   - Model was trained on incorrect data
   - Must retrain after data fix

## Next Steps

1. ✅ Document the issue (this file)
2. ⏳ Examine Excel file structure manually or with proper tools
3. ⏳ Fix data processing code to find columns dynamically
4. ⏳ Reprocess data
5. ⏳ Validate against Excel file
6. ⏳ Retrain model

## Verification Method

To verify the fix:
1. Process one material sheet from Excel
2. Compare first few rows with Excel file manually
3. Ensure pH, Time, and Release values match exactly
4. Repeat for multiple materials (with and without Fraction column)

