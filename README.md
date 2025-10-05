# 21-Elements Monolithic Cement Leaching Prediction

A machine learning project for predicting elemental leaching behavior from monolithic cement materials under different pH and time conditions.

## Project Overview

This project analyzes leaching data from 21 different elements (Al, As, Ba, Br, Ca, Cd, Cl, Cr, Cu, F, Fe, K, Mg, Mo, Na, P, Pb, Si, SO4, Zn) in monolithic cement materials. The goal is to predict cumulative release values (mg/m²) based on material properties, pH levels, and time exposure.

## Key Features

- **Comprehensive Data Processing**: Extracts and consolidates leaching data from Excel files with multiple material sheets
- **Advanced Feature Engineering**: Creates 25+ engineered features including interactions, transformations, and domain-specific metrics
- **Multi-Model Approach**: Implements XGBoost, LightGBM, Random Forest, Neural Networks, and ensemble methods
- **Material-Specific Models**: Specialized models for problematic materials (Al, Br) with conditional prediction logic
- **Model Serialization**: Save and load trained models for deployment and reuse
- **Web Application**: Beautiful Flask web interface for making predictions
- **REST API**: Programmatic access to predictions via JSON API
- **High Performance**: Achieves R² > 0.85 with median absolute percentage error < 1%

## Dataset

- **Total Samples**: 3,368 data points
- **Materials**: 20 different elements
- **Features**: 25 engineered features including material properties, pH, time, and interactions
- **Target**: Cumulative release (mg/m²) with log transformation

## Model Performance

- **Best Model**: XGBoost with ensemble approach
- **R² Score**: 0.8588
- **RMSE**: 31.26 mg/m²
- **Median Error**: 0.12%
- **High Error Reduction**: From 7.9% to 2.4% samples with >10% error using conditional models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 21-ELEMENTS-Monolithic_cement_leaching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the Excel data file (`LXS-Monolithe-21.xlsx`) in the project directory.

## Usage

### Quick Start

#### 1. Train Model and Start Web App
```bash
# Train model (automatically saves to models/ directory)
python main.py

# Start web application
python app.py
# Open browser to: http://localhost:5000
```

#### 2. Use Python API
```python
from src.data_processing import DataProcessor
from src.ml_pipeline import MLPipeline

# Initialize processor
processor = DataProcessor('LXS-Monolithe-21.xlsx')
df = processor.load_and_consolidate_data()

# Train models
pipeline = MLPipeline()
best_model, results = pipeline.train_models(df)

# Make predictions
prediction = pipeline.predict_leaching(
    material='Al', 
    ph=12.0, 
    time_days=1.0,
    cement_type='CEM_I',
    form_type='Concrete',
    stat_measure='CL_Minus'
)
print(f"Predicted leaching: {prediction['predicted_leaching_mg_m2']} mg/m²")
```

#### 3. Use REST API
```python
import requests

response = requests.post('http://localhost:5000/api/predict', json={
    'material': 'Al',
    'ph': 12.0,
    'time_days': 1.0,
    'cement_type': 'CEM_I',
    'form_type': 'Concrete',
    'stat_measure': 'CL_Minus'
})

result = response.json()
print(f"Prediction: {result['prediction']} mg/m²")
```

### Available Materials

- **Metals**: Al, Fe, Si, As, Cr, Mo, Ba, P, Ca, K, Mg, Na, Cd, Cu, Pb, Zn
- **Non-metals**: Br, F, Cl, SO4

### Supported Cement Types

- CEM I, CEM II/A, CEM II/B, CEM III, CEM V

### Form Types

- Concrete, Mortar, Sewage_Sludge

### Statistical Measures

- CL_Minus, CL_Plus, LPx, Med, UPx, X

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── model_config.yaml
├── data/
│   ├── LXS-Monolithe-21.xlsx
│   └── processed/
│       └── consolidated_leaching_data_FINAL.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── ml_pipeline.py
│   ├── model_serializer.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       └── conditional_ensemble.py
├── app.py                     # Flask web application
├── demo_new_features.py       # Demo script for new features
├── notebooks/
│   └── 21_Leaching.ipynb
└── tests/
    ├── __init__.py
    ├── test_data_processing.py
    └── test_ml_pipeline.py
```

## Key Components

### Data Processing (`src/data_processing.py`)
- Excel file parsing and data extraction
- Feature engineering and categorical encoding
- Data validation and cleaning

### ML Pipeline (`src/ml_pipeline.py`)
- Model training and evaluation
- Cross-validation and hyperparameter tuning
- Prediction functions with error handling

### Conditional Ensemble (`src/models/conditional_ensemble.py`)
- Material-specific model selection
- Improved prediction accuracy for problematic materials

### Model Serialization (`src/model_serializer.py`)
- Save and load trained models using pickle
- Model metadata and versioning
- Easy deployment and model management

### Web Application (`app.py`)
- Beautiful Flask web interface
- RESTful API endpoints
- Input validation and error handling
- Responsive design for all devices

## Configuration

Model parameters and settings can be adjusted in `config/model_config.yaml`:

```yaml
models:
  xgboost:
    n_estimators: 300
    max_depth: 7
    learning_rate: 0.05
  features:
    include_interactions: true
    include_polynomial: true
  data:
    test_size: 0.2
    random_state: 42
```

## Results and Analysis

The project includes comprehensive error analysis:
- Residual plots and error distributions
- Material-specific performance metrics
- Cross-validation results
- Feature importance analysis

High-error outliers are identified and can be reviewed in `high_error_outliers.csv`.
