#!/usr/bin/env python3
"""
Quick script to generate predictions for all data and compare with actual values.

This is a simplified version that matches the user's original code structure.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import with explicit path to help IDE
from src.model_serializer import ModelSerializer
from src.data_processing import DataProcessor
from src.utils import load_config


def main():
    """Generate predictions for all data and save to CSV."""
    print("🧪 Quick Prediction Generation")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Load processed data
    processed_data_path = 'data/processed/consolidated_leaching_data_FINAL.csv'
    if os.path.exists(processed_data_path):
        print("📂 Loading processed data...")
        df = pd.read_csv(processed_data_path)
    else:
        print("📂 Processing raw data...")
        excel_file = config.get('paths', {}).get('data_file', 'LXS-Monolithe-21.xlsx')
        processor = DataProcessor(excel_file)
        df = processor.load_and_consolidate_data()
        df, feature_columns, label_encoders = processor.create_features(df)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(processed_data_path, index=False)
    
    # Load trained model
    print("🤖 Loading trained model...")
    serializer = ModelSerializer('models')
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = serializer.load_model_components('production_model')
    
    # 1. Prepare data you want to predict (here all data; replace with test split as needed)
    df_pred = df.copy()
    
    # 2. Build feature DataFrame and transform
    X = df_pred[feature_columns]
    
    # 3. Model prediction (XGBoost was trained on original features, not transformed)
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    
    # Ensure non-negative predictions (leaching values can never be negative)
    y_pred = np.maximum(y_pred, 0.0)
    
    # 4. Create output DataFrame
    output = df_pred.copy()
    output['model_prediction'] = y_pred
    output['actual'] = df_pred['Cumulative_Release_mg_m2']
    
    # 5. Save to CSV
    output_cols = ['Material', 'pH', 'Time_days', 'Cement_Type', 'Form_Type', 'Stat_Measure',
                   'model_prediction', 'actual']
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    output_file = 'results/model_predictions_vs_actual.csv'
    output[output_cols].to_csv(output_file, index=False)
    
    print(f"✅ Saved predictions to {output_file}")
    
    # Print some statistics
    mae = np.mean(np.abs(output['model_prediction'] - output['actual']))
    rmse = np.sqrt(np.mean((output['model_prediction'] - output['actual'])**2))
    
    # Calculate R² properly
    from sklearn.metrics import r2_score
    r2 = r2_score(output['actual'], output['model_prediction'])
    
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE: {mae:.2f} mg/m²")
    print(f"   RMSE: {rmse:.2f} mg/m²")
    print(f"   R²: {r2:.4f}")
    print(f"   Total predictions: {len(output)}")


if __name__ == "__main__":
    main()
