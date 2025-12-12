#!/usr/bin/env python3
"""
Comprehensive error analysis for training and testing datasets.

This script calculates error metrics, percentage errors, and identifies
what percentage of predictions have <10% error.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.model_serializer import ModelSerializer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model_and_data():
    """Load the trained model and datasets."""
    print("📂 Loading model and datasets...")
    
    # Load model
    serializer = ModelSerializer('models')
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = serializer.load_model_components('production_model')
    
    # Load datasets
    train_df = pd.read_csv('data/processed/train_dataset.csv')
    test_df = pd.read_csv('data/processed/test_dataset.csv')
    
    print(f"✅ Model loaded: {metadata.get('model_type', 'Unknown')}")
    print(f"✅ Train dataset: {len(train_df)} records")
    print(f"✅ Test dataset: {len(test_df)} records")
    
    return model, label_encoders, power_transformer, scaler, feature_columns, train_df, test_df


def make_predictions(model, power_transformer, scaler, feature_columns, df):
    """Make predictions on a dataset."""
    # Extract features
    X = df[feature_columns].copy()
    
    # Apply transformations (same as training)
    X_transformed = power_transformer.transform(X)
    X_scaled = scaler.transform(X_transformed)
    
    # Make predictions
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)
    
    # Get true values
    y_true = df['Cumulative_Release_mg_m2'].values
    
    return y_true, y_pred


def calculate_error_metrics(y_true, y_pred, dataset_name):
    """Calculate comprehensive error metrics."""
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Percentage errors
    abs_error = np.abs(y_pred - y_true)
    
    # Handle division by zero for percentage error
    # For true values = 0, we can't calculate percentage error
    # Use absolute error instead or mark as NaN
    pct_error = np.where(y_true != 0, (abs_error / y_true) * 100, np.nan)
    
    # Mean Absolute Percentage Error (MAPE) - excluding zeros
    mape = np.nanmean(pct_error)
    
    # Median Absolute Percentage Error
    median_ape = np.nanmedian(pct_error)
    
    # Percentage of predictions with <10% error
    pct_under_10 = np.sum(pct_error < 10) / len(pct_error[~np.isnan(pct_error)]) * 100
    
    # Percentage of predictions with <5% error
    pct_under_5 = np.sum(pct_error < 5) / len(pct_error[~np.isnan(pct_error)]) * 100
    
    # Percentage of predictions with <20% error
    pct_under_20 = np.sum(pct_error < 20) / len(pct_error[~np.isnan(pct_error)]) * 100
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Absolute_Error': abs_error,
        'Percentage_Error': pct_error,
        'Error_Less_Than_10pct': pct_error < 10
    })
    
    # Summary statistics
    summary = {
        'Dataset': dataset_name,
        'Total_Records': len(y_true),
        'MAE_mg_m2': round(mae, 2),
        'RMSE_mg_m2': round(rmse, 2),
        'R2_Score': round(r2, 4),
        'MAPE_%': round(mape, 2),
        'Median_APE_%': round(median_ape, 2),
        'Pct_Under_5pct_Error': round(pct_under_5, 2),
        'Pct_Under_10pct_Error': round(pct_under_10, 2),
        'Pct_Under_20pct_Error': round(pct_under_20, 2),
        'Mean_Pct_Error': round(np.nanmean(pct_error), 2),
        'Std_Pct_Error': round(np.nanstd(pct_error), 2),
        'Min_Pct_Error': round(np.nanmin(pct_error), 2),
        'Max_Pct_Error': round(np.nanmax(pct_error), 2),
        'P25_Pct_Error': round(np.nanpercentile(pct_error, 25), 2),
        'P50_Pct_Error': round(np.nanpercentile(pct_error, 50), 2),
        'P75_Pct_Error': round(np.nanpercentile(pct_error, 75), 2),
        'P90_Pct_Error': round(np.nanpercentile(pct_error, 90), 2),
        'P95_Pct_Error': round(np.nanpercentile(pct_error, 95), 2)
    }
    
    return summary, results_df


def analyze_by_material(df, y_true, y_pred):
    """Analyze errors by material type."""
    if 'Material' not in df.columns:
        return None
    
    material_results = []
    
    for material in df['Material'].unique():
        mask = df['Material'] == material
        mat_y_true = y_true[mask]
        mat_y_pred = y_pred[mask]
        
        if len(mat_y_true) == 0:
            continue
        
        abs_error = np.abs(mat_y_pred - mat_y_true)
        pct_error = np.where(mat_y_true != 0, (abs_error / mat_y_true) * 100, np.nan)
        
        pct_under_10 = np.sum(pct_error < 10) / len(pct_error[~np.isnan(pct_error)]) * 100 if len(pct_error[~np.isnan(pct_error)]) > 0 else 0
        
        material_results.append({
            'Material': material,
            'Count': len(mat_y_true),
            'MAE': round(mean_absolute_error(mat_y_true, mat_y_pred), 2),
            'RMSE': round(np.sqrt(mean_squared_error(mat_y_true, mat_y_pred)), 2),
            'R2': round(r2_score(mat_y_true, mat_y_pred), 4),
            'MAPE_%': round(np.nanmean(pct_error), 2),
            'Pct_Under_10pct_Error': round(pct_under_10, 2)
        })
    
    return pd.DataFrame(material_results)


def main():
    """Main function to run error analysis."""
    print("=" * 70)
    print("📊 COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 70)
    print()
    
    # Load model and data
    model, label_encoders, power_transformer, scaler, feature_columns, train_df, test_df = load_model_and_data()
    
    print("\n" + "=" * 70)
    print("🔍 ANALYZING TRAINING DATASET")
    print("=" * 70)
    
    # Training dataset predictions
    train_y_true, train_y_pred = make_predictions(model, power_transformer, scaler, feature_columns, train_df)
    train_summary, train_results = calculate_error_metrics(train_y_true, train_y_pred, 'Training')
    
    # Print training summary
    print("\n📈 Training Dataset Error Summary:")
    print("-" * 70)
    for key, value in train_summary.items():
        print(f"  {key:25s}: {value}")
    
    # Analyze by material for training
    train_material_analysis = analyze_by_material(train_df, train_y_true, train_y_pred)
    
    print("\n" + "=" * 70)
    print("🔍 ANALYZING TESTING DATASET")
    print("=" * 70)
    
    # Testing dataset predictions
    test_y_true, test_y_pred = make_predictions(model, power_transformer, scaler, feature_columns, test_df)
    test_summary, test_results = calculate_error_metrics(test_y_true, test_y_pred, 'Testing')
    
    # Print testing summary
    print("\n📈 Testing Dataset Error Summary:")
    print("-" * 70)
    for key, value in test_summary.items():
        print(f"  {key:25s}: {value}")
    
    # Analyze by material for testing
    test_material_analysis = analyze_by_material(test_df, test_y_true, test_y_pred)
    
    # Add material info to results
    train_results['Material'] = train_df['Material'].values
    test_results['Material'] = test_df['Material'].values
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    train_results.to_csv(results_dir / 'train_error_analysis.csv', index=False)
    test_results.to_csv(results_dir / 'test_error_analysis.csv', index=False)
    
    # Save summaries
    summary_df = pd.DataFrame([train_summary, test_summary])
    summary_df.to_csv(results_dir / 'error_summary.csv', index=False)
    
    # Save material analysis
    if train_material_analysis is not None:
        train_material_analysis.to_csv(results_dir / 'train_error_by_material.csv', index=False)
        print("\n📊 Training Error by Material:")
        print(train_material_analysis.to_string(index=False))
    
    if test_material_analysis is not None:
        test_material_analysis.to_csv(results_dir / 'test_error_by_material.csv', index=False)
        print("\n📊 Testing Error by Material:")
        print(test_material_analysis.to_string(index=False))
    
    # Print key findings
    print("\n" + "=" * 70)
    print("🎯 KEY FINDINGS")
    print("=" * 70)
    print(f"\n✅ Training Dataset:")
    print(f"   • {train_summary['Pct_Under_10pct_Error']:.2f}% of predictions have <10% error")
    print(f"   • Mean Absolute Error: {train_summary['MAE_mg_m2']} mg/m²")
    print(f"   • R² Score: {train_summary['R2_Score']:.4f}")
    
    print(f"\n✅ Testing Dataset:")
    print(f"   • {test_summary['Pct_Under_10pct_Error']:.2f}% of predictions have <10% error")
    print(f"   • Mean Absolute Error: {test_summary['MAE_mg_m2']} mg/m²")
    print(f"   • R² Score: {test_summary['R2_Score']:.4f}")
    
    print(f"\n💾 Results saved to:")
    print(f"   • results/train_error_analysis.csv")
    print(f"   • results/test_error_analysis.csv")
    print(f"   • results/error_summary.csv")
    if train_material_analysis is not None:
        print(f"   • results/train_error_by_material.csv")
        print(f"   • results/test_error_by_material.csv")
    
    print("\n" + "=" * 70)
    print("✅ Error analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

