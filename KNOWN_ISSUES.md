# Known Issues

## Data Processing Issue: Column Misalignment for Materials with Fraction Column

### Problem Description

Some materials in the processed CSV file have misaligned columns, causing incorrect predictions. The issue occurs when the Excel file contains a "Fraction" column before the pH, Time, and Cumulative Release columns.

### Affected Materials

**Confirmed Affected:**
- **Barium (Ba)** - All records have misaligned data

**Potentially Affected:**
- Any material where the Excel sheet structure includes a Fraction column before pH/Time/Release columns

### Symptoms

When using the web application or API to predict leaching for affected materials:
- Predictions do not match the actual values from the Excel file
- Example: For Barium with pH=12.08, Time=0.08 days, the model predicts incorrectly because it was trained on misaligned data

### Root Cause

The data processing code in `src/data_processing.py` has a logic issue when handling Excel sheets that contain a "Fraction" column. 

**Expected Excel Structure:**
```
Fraction | pH | Time (days) | Cumulative release (mg/m²)
1        | 12.08 | 0.08 | 0.180315819
```

**What the Code Currently Reads:**
- Fraction value (1.0) → stored as pH
- pH value (12.08) → stored as Time_days  
- Time value (0.08) → stored as Cumulative_Release_mg_m2
- Cumulative Release value → lost/misplaced

**Actual CSV Data (Incorrect):**
```csv
Material,pH,Time_days,Cumulative_Release_mg_m2
Ba,1.0,12.08,0.08
```

**What It Should Be:**
```csv
Material,pH,Time_days,Cumulative_Release_mg_m2
Ba,12.08,0.08,0.180315819
```

### Impact

1. **Model Training**: The model was trained on incorrect data for affected materials
2. **Predictions**: All predictions for affected materials will be inaccurate
3. **Data Integrity**: The processed CSV file contains incorrect values for these materials

### Example: Barium First Record

**From Excel File:**
- Fraction: 1
- pH: 12.08
- Time (days): 0.08
- Cumulative release: 0.180315819 mg/m²

**In Processed CSV:**
- pH: 1.0 (incorrect - this is the Fraction)
- Time_days: 12.08 (incorrect - this is the pH)
- Cumulative_Release_mg_m2: 0.08 (incorrect - this is the Time)

**Model Prediction:**
- When user inputs: Material=Ba, pH=12.08, Time=0.08
- Model was trained on: pH=1.0, Time=12.08, Release=0.08
- Result: Prediction will be incorrect

### Technical Details

**Location of Issue:**
- File: `src/data_processing.py`
- Function: `extract_leaching_data_from_sheet()`
- Lines: 54-98

**Current Logic:**
The code checks if "Time" is in the header row to determine column structure:
- If "Time" present → Standard format: pH, Time, Cumulative Release
- If "Time" not present → Special format: Fraction, pH, Cumulative Release (maps Fraction to Time)

**Problem:**
For some materials (like Barium), the Excel has: Fraction, pH, Time, Cumulative Release
But the code doesn't detect "Time" correctly or the column order is different, causing it to read Fraction as pH.

### Fix Required

1. **Update Data Processing Logic:**
   - Improve detection of column structure
   - Handle Fraction column correctly when Time column is also present
   - Verify column order and mapping

2. **Reprocess Data:**
   - Re-run data processing on the original Excel file
   - Verify all materials have correct column alignment
   - Generate new consolidated CSV file

3. **Retrain Model:**
   - Train new model with corrected data
   - Validate predictions match Excel values
   - Update production model files

### Workaround

Until the issue is fixed:
- **Do not use the model for predictions on affected materials** (Barium confirmed)
- For other materials, verify predictions against Excel data before trusting results
- Check the processed CSV file to identify which materials have misaligned columns

### Verification Steps

To check if a material is affected:
```python
import pandas as pd
df = pd.read_csv('data/processed/consolidated_leaching_data_FINAL.csv')

# Check for suspicious patterns (pH values that look like fractions)
material = 'Ba'
material_data = df[df['Material'] == material]
print(material_data[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].head())

# If pH values are 1.0, 2.0, 3.0, etc. (fraction numbers), the data is misaligned
```

### Status

- **Date Identified**: November 2024
- **Severity**: High - Affects model accuracy for specific materials
- **Priority**: High - Needs fixing before production use
- **Status**: Documented, pending fix

### Next Steps

1. ✅ Document the issue (this file)
2. ⏳ Fix data processing code
3. ⏳ Reprocess data from Excel
4. ⏳ Retrain model
5. ⏳ Validate predictions
6. ⏳ Update production deployment

---

## Other Issues

_Add other known issues here as they are discovered._

