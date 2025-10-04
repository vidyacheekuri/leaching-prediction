#!/usr/bin/env python3
"""
Generate predictions specifically for the test dataset.

This script loads the test dataset and generates predictions to verify
model performance on unseen data.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import with explicit path to help IDE
from src.model_serializer import ModelSerializer


def main():
    """Generate predictions for the test dataset."""
    print("🧪 Test Dataset Predictions")
    print("=" * 40)
    
    # Load test dataset
    test_file = 'data/processed/test_dataset.csv'
    if not os.path.exists(test_file):
        print("❌ Test dataset not found. Please run split_datasets.py first.")
        return
    
    print("📂 Loading test dataset...")
    df_test = pd.read_csv(test_file)
    print(f"   Test samples: {len(df_test)}")
    
    # Load trained model
    print("🤖 Loading trained model...")
    serializer = ModelSerializer('models')
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = serializer.load_model_components('production_model')
    
    # Get feature columns (exclude target and categorical columns)
    feature_cols = [col for col in feature_columns if col in df_test.columns]
    
    # Prepare features
    X_test = df_test[feature_cols]
    y_test = df_test['Cumulative_Release_mg_m2']
    
    # Generate predictions
    print("🔮 Generating predictions...")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    # Ensure non-negative predictions (leaching values can never be negative)
    y_pred = np.maximum(y_pred, 0.0)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Create output dataframe
    output = df_test.copy()
    output['model_prediction'] = y_pred
    output['prediction_error'] = y_pred - y_test
    output['absolute_error'] = np.abs(output['prediction_error'])
    output['relative_error'] = (output['prediction_error'] / y_test) * 100
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/test_predictions.csv'
    
    # Select key columns for output
    output_cols = [
        'Material', 'pH', 'Time_days', 'Cement_Type', 'Form_Type', 'Stat_Measure',
        'model_prediction', 'Cumulative_Release_mg_m2', 'prediction_error', 
        'absolute_error', 'relative_error'
    ]
    
    result_df = output[output_cols].copy()
    result_df.to_csv(output_file, index=False)
    
    print(f"✅ Test predictions saved to {output_file}")
    
    # Print performance metrics
    print(f"\n📊 Test Set Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f} mg/m²")
    print(f"   RMSE: {rmse:.2f} mg/m²")
    print(f"   Total test samples: {len(df_test)}")
    
    # Performance by material
    print(f"\n🔬 Performance by Material:")
    material_stats = result_df.groupby('Material').agg({
        'Cumulative_Release_mg_m2': 'count',
        'absolute_error': 'mean',
        'relative_error': 'mean'
    }).round(2)
    
    material_stats.columns = ['Count', 'MAE', 'MRE_%']
    print(material_stats.sort_values('MAE'))
    
    # Show some sample predictions
    print(f"\n📋 Sample Predictions:")
    sample_cols = ['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2', 'model_prediction', 'absolute_error']
    print(result_df[sample_cols].head(10).to_string(index=False))
    
    # Performance assessment
    if r2 > 0.9:
        performance = "Excellent"
    elif r2 > 0.8:
        performance = "Very Good"
    elif r2 > 0.7:
        performance = "Good"
    elif r2 > 0.6:
        performance = "Fair"
    else:
        performance = "Poor"
    
    print(f"\n🎯 Overall Performance: {performance} (R² = {r2:.4f})")
    
    return output_file


if __name__ == "__main__":
    main()
