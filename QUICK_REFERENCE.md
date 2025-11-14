# Cement Leaching Model - Quick Reference

## 🎯 Performance Summary
- **R² Score:** 0.8582 (85.82%) - Very Good
- **Accuracy:** 93.0% of predictions <10% error
- **Test Samples:** 674
- **Materials:** 20 elements

## 🚀 Quick Start

### Web App
```bash
python app.py
# Open: http://localhost:8080
```

### Python API
```python
from simple_predict import predict_leaching
result = predict_leaching('Al', 12.0, 1.0)
```

### Batch Predictions
```bash
python quick_predictions.py
```

## 📊 Key Results
- **93.0%** of test samples have <10% error
- **92.3%** of test samples have <5% error
- **Only 3.1%** have high error (≥50%)

## ⚠️ Challenging Materials
- **Aluminum (Al):** Higher prediction error (~40% of samples >10% error)
- **Bromine (Br):** Higher prediction error (~54.5% of samples >10% error)
- Other materials: Excellent accuracy (<5% error rate)

## 🔬 Best Performance
- **pH Range:** 8-12 (optimal)
- **Time Range:** 1-30 days
- **Materials:** SO4, Pb, Na, Cl, Cr (excellent)
- **Materials:** Al, Br (challenging)

## 📁 Key Files
- `MODEL_PERFORMANCE_REPORT.md` - Full report
- `SUMMARY_STATISTICS.csv` - All metrics
- `results/test_predictions.csv` - Test results
- `results/plots/` - Visualizations
