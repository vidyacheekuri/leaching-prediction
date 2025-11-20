# Feedback Response and Action Plan

## Summary of Observations

This document addresses the testing feedback and outlines solutions for each observation.

---

## 1. Unable to Reproduce Excel Results

**Observation:** When inputting time and pH values from the Excel file, results don't match.

**Root Cause:** 
- Data processing issue documented in `KNOWN_ISSUES.md`
- Column misalignment for materials with Fraction column (Barium confirmed, others potentially affected)
- Model was trained on misaligned data

**Status:** ✅ Documented, ⏳ Needs fixing

**Action Items:**
1. Fix data processing code in `src/data_processing.py`
2. Reprocess data from Excel
3. Retrain model
4. Validate predictions match Excel values

---

## 2. Statistical Anomaly: Mean/Median Lower Than CL-

**Observation:** Sometimes mean (X) or median (Med) values are lower than CL-, which is statistically surprising.

**Analysis:**
- CL- (Confidence Limit Minus) should typically be the lower bound
- Mean/Median being lower suggests:
  - Possible data processing error
  - Incorrect statistical measure extraction
  - Or legitimate data distribution (need to verify)

**Status:** ⏳ Needs investigation

**Action Items:**
1. Review how CL-, CL+, X, Med, LPx, UPx are extracted from Excel
2. Verify statistical calculations
3. Check if this is a data quality issue or model limitation
4. Add validation warnings in the app if this occurs

---

## 3. Feature Request: Predict pH at Time t

**Request:** Predict both leaching (mg/m²) AND pH at time t, using initial pH as input.

**Use Case:** Run short test (0.08 or 1 day), predict result at 64 days.

**Current State:**
- Model only predicts leaching
- pH is an input variable (assumed constant)
- No pH prediction capability

**Status:** 🆕 New feature request

**Implementation Plan:**
1. **Data Analysis:**
   - Check if pH changes over time in the dataset
   - Analyze pH evolution patterns
   - Determine if pH prediction is feasible

2. **Model Options:**
   - **Option A:** Train separate pH prediction model
   - **Option B:** Multi-output model (predict both leaching and pH)
   - **Option C:** Sequential model (predict pH first, then use predicted pH for leaching)

3. **Interface Updates:**
   - Add "Initial pH" input field
   - Add "Target Time" input field
   - Display both predictions: leaching and pH at target time

**Challenges:**
- Need to verify if pH data shows time-dependent patterns
- May require significant model retraining
- Need to validate pH prediction accuracy

---

## 4. Time Limit Validation

**Observation:** Time can be entered up to 100 days, but tests only go to 64 days.

**Current State:**
- Validation allows: 0.01 to 100 days
- Training data goes up to: 64 days
- Predictions beyond 64 days are extrapolations (less reliable)

**Status:** ⚠️ Needs validation update

**Action Items:**
1. Update validation in `app.py` to warn/restrict beyond 64 days
2. Add warning message for predictions > 64 days
3. Consider adding confidence intervals that widen beyond training range

**Proposed Solution:**
```python
if time_days > 64:
    warning = "Warning: Prediction beyond 64 days (training data limit). Results may be less reliable."
```

---

## 5. Feature Request: More Specific Cement Types

**Request:** Add detailed cement type classifications from Excel:
- CEM I (Portland cement, Portland cement filler)
- CEM II/A-L, CEM II/B-L (Portland limestone cement)
- CEM II/A-V, CEM II/B-V (Portland fly ash cement)
- CEM II/B-Q (Portland pozzolanic cement)
- CEM II/B-S, CEM II/A-S (Portland-slag cement)
- CEM II/B-M (Portland composite cement)
- CEM III/B (Blastfurnace cement)
- CEM V/A (Composite cement)

**Current State:**
- Model uses simplified cement types: CEM_I, CEM_II, CEM_III, CEM_V, Unknown
- Detailed subtypes not captured

**Status:** 🆕 New feature request

**Implementation Plan:**
1. **Data Analysis:**
   - Extract detailed cement type information from Excel
   - Map Excel samples to cement types:
     - N1, H6, H9 → CEM I (Portland cement)
     - H1, H3, D1, D2, H7, N2 → CEM I (Portland cement filler)
     - H5 → CEM II/A-L
     - etc.

2. **Data Processing Update:**
   - Update `clean_material_condition()` in `src/data_processing.py`
   - Add mapping for all cement subtypes
   - Update label encoders

3. **Model Retraining:**
   - Retrain with detailed cement types
   - Evaluate if this improves accuracy

4. **Interface Update:**
   - Add dropdown with all cement subtypes
   - Group by main type (CEM I, CEM II, etc.)

**Challenges:**
- Need to map Excel sample codes to cement types
- May need to verify cement type information in Excel
- Model may need more data for some rare cement types

---

## 6. Feature Request: Dutch Standard Thresholds Integration

**Request:** Integrate NEN 7375 leaching thresholds at 64 days for direct comparison.

**Thresholds Table:**
| Substance | Test | Threshold (mg/m²) at 64 days |
|-----------|------|------------------------------|
| Antimony | NEN 7375 | 8.7 |
| Arsenic | NEN 7375 | 260 |
| Barium | NEN 7375 | 1500 |
| Cadmium | NEN 7375 | 3.8 |
| Chromium | NEN 7375 | 120 |
| Cobalt | NEN 7375 | 60 |
| Copper | NEN 7375 | 98 |
| Tin | NEN 7375 | 20 |
| Mercury | NEN 7375 | 1.4 |
| Molybdenum | NEN 7375 | 1 |
| Nickel | NEN 7375 | 81 |
| Lead | NEN 7375 | 40 |
| Selenium | NEN 7375 | 4.8 |
| Vanadium | NEN 7375 | 110 |
| Zinc | NEN 7375 | 800 |
| Bromides | NEN 7375 | 670 |
| Fluorides | NEN 7375 | 2200 |
| Chlorides | NEN 7375 | 110000 |
| Sulfates | NEN 7375 | 165000 |

**Status:** 🆕 New feature request

**Implementation Plan:**
1. **Add Threshold Data:**
   - Create `config/regulatory_thresholds.yaml` with all thresholds
   - Map materials to thresholds (e.g., Br → Bromides, SO4 → Sulfates)

2. **Interface Updates:**
   - When time = 64 days, show threshold comparison
   - Display: "Predicted: X mg/m² | Threshold: Y mg/m² | Status: PASS/FAIL"
   - Color coding: Green (pass), Red (fail)
   - Add threshold table in a collapsible section

3. **API Updates:**
   - Add threshold comparison to API response
   - Include pass/fail status

**Proposed UI Enhancement:**
```
Prediction Result: 45.2 mg/m²

Regulatory Compliance (NEN 7375 at 64 days):
Threshold: 1500 mg/m²
Status: ✅ PASS (Prediction is below threshold)
Margin: 1454.8 mg/m² below threshold
```

---

## Priority Action Plan

### Immediate (High Priority)
1. ✅ Document data processing issue
2. ⏳ Fix data processing code for Fraction column
3. ⏳ Add time validation warning (>64 days)
4. ⏳ Add Dutch standard thresholds integration

### Short Term (Medium Priority)
5. ⏳ Investigate mean/median < CL- issue
6. ⏳ Add detailed cement type extraction
7. ⏳ Reprocess data and retrain model

### Long Term (Feature Requests)
8. ⏳ Implement pH prediction at time t
9. ⏳ Add confidence intervals for extrapolations
10. ⏳ Enhanced validation and warnings

---

## Next Steps

1. **Review this document** - Confirm priorities
2. **Start with high-priority items** - Fix data processing, add thresholds
3. **Test and validate** - Ensure fixes work correctly
4. **Deploy updates** - Push to production

---

## Questions for Clarification

1. **pH Prediction:** Do you have data showing how pH changes over time in your tests? This will determine if pH prediction is feasible.

2. **Cement Type Mapping:** Do you have a complete mapping of Excel sample codes (N1, H6, H9, etc.) to cement types? This is needed for implementation.

3. **Statistical Measures:** Should we prioritize fixing the CL- vs Mean/Median issue, or is this acceptable for now?

4. **Threshold Display:** Should thresholds only show at exactly 64 days, or also for predictions extrapolated to 64 days from shorter tests?

---

**Document Created:** November 2024  
**Status:** Ready for implementation

