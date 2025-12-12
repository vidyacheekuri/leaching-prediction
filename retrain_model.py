"""
Comprehensive model retraining script with improved hyperparameters and evaluation.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, 'src')

from src.data_processing import DataProcessor
from src.ml_pipeline import MLPipeline
from src.model_serializer import ModelSerializer
from src.utils import load_config
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model_performance(y_true, y_pred, dataset_name="Dataset"):
    """Calculate comprehensive performance metrics."""
    # Avoid division by zero
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove zeros for percentage calculations
    non_zero_mask = y_true > 0
    
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Percentage errors (only for non-zero actuals)
    if non_zero_mask.sum() > 0:
        ape = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100
        mape = np.mean(ape)
        median_ape = np.median(ape)
        
        # Percentage with <10% error
        pct_under_10 = (ape < 10).sum() / len(ape) * 100
        pct_under_20 = (ape < 20).sum() / len(ape) * 100
    else:
        mape = np.nan
        median_ape = np.nan
        pct_under_10 = np.nan
        pct_under_20 = np.nan
    
    print(f"\n📊 {dataset_name} Performance Metrics:")
    print("=" * 70)
    print(f"   R² Score:           {r2:.6f}")
    print(f"   RMSE:               {rmse:.6f} mg/m²")
    print(f"   MAE:                {mae:.6f} mg/m²")
    if not np.isnan(mape):
        print(f"   MAPE:               {mape:.2f}%")
        print(f"   Median APE:          {median_ape:.2f}%")
        print(f"   Predictions <10% error: {pct_under_10:.2f}%")
        print(f"   Predictions <20% error: {pct_under_20:.2f}%")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'median_ape': median_ape,
        'pct_under_10': pct_under_10,
        'pct_under_20': pct_under_20
    }

def main():
    """Retrain model with verified data and improved hyperparameters."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL RETRAINING")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Step 1: Load verified data
    print("\n📊 Step 1: Loading verified data...")
    processed_file = 'data/processed/consolidated_leaching_data_FINAL.csv'
    df = pd.read_csv(processed_file)
    print(f"   ✅ Loaded {len(df)} records")
    print(f"   Materials: {df['Material'].nunique()}")
    print(f"   Time range: {df['Time_days'].min():.2f} - {df['Time_days'].max():.2f} days")
    
    # Step 2: Initialize pipeline
    print("\n🤖 Step 2: Initializing ML Pipeline...")
    pipeline = MLPipeline(random_state=config.get('data', {}).get('random_state', 42))
    
    # Step 3: Prepare data
    print("\n📊 Step 3: Preparing data for training...")
    # Load feature columns and encoders from existing model if available
    feature_columns = None
    label_encoders = None
    
    model_dir = Path('models')
    if (model_dir / 'production_model_metadata.pkl').exists():
        print("   📂 Loading existing feature columns and encoders...")
        serializer = ModelSerializer('models')
        try:
            _, label_encoders, _, _, feature_columns, _ = serializer.load_model_components('production_model')
            print(f"   ✅ Loaded {len(feature_columns)} feature columns")
        except Exception as e:
            print(f"   ⚠️  Could not load existing model: {e}")
            print("   🔄 Will create new feature columns...")
    
    # Create features if needed
    if feature_columns is None:
        processor = DataProcessor('LXS-Monolithe-21.xlsx')
        df, feature_columns, label_encoders = processor.create_features(df)
        print(f"   ✅ Created {len(feature_columns)} features")
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = pipeline.prepare_data(
        df, feature_columns, label_encoders,
        test_size=config.get('data', {}).get('test_size', 0.2)
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Step 4: Train models with improved hyperparameters
    print("\n🎯 Step 4: Training models with optimized hyperparameters...")
    best_model, best_model_name, results, models, pt, scaler = pipeline.train_models(
        X_train, X_test, y_train, y_test, y_train_orig, y_test_orig
    )
    
    # Step 5: Evaluate on training set
    print("\n📊 Step 5: Evaluating on training set...")
    y_train_pred_log = best_model.predict(pt.transform(scaler.transform(X_train)))
    y_train_pred = np.expm1(y_train_pred_log)
    train_metrics = evaluate_model_performance(y_train_orig, y_train_pred, "Training Set")
    
    # Step 6: Evaluate on test set
    print("\n📊 Step 6: Evaluating on test set...")
    y_test_pred_log = best_model.predict(pt.transform(scaler.transform(X_test)))
    y_test_pred = np.expm1(y_test_pred_log)
    test_metrics = evaluate_model_performance(y_test_orig, y_test_pred, "Test Set")
    
    # Step 7: Material-specific evaluation
    print("\n📊 Step 7: Material-Specific Performance...")
    test_df = df.loc[X_test.index].copy()
    test_df['Predicted'] = y_test_pred
    test_df['Actual'] = y_test_orig
    test_df['Error_Pct'] = np.abs((test_df['Actual'] - test_df['Predicted']) / test_df['Actual']) * 100
    
    material_performance = []
    for material in test_df['Material'].unique():
        mat_data = test_df[test_df['Material'] == material]
        if len(mat_data) > 0:
            non_zero = mat_data[mat_data['Actual'] > 0]
            if len(non_zero) > 0:
                pct_under_10 = (non_zero['Error_Pct'] < 10).sum() / len(non_zero) * 100
                median_error = non_zero['Error_Pct'].median()
                material_performance.append({
                    'Material': material,
                    'Samples': len(mat_data),
                    'Median_Error_Pct': median_error,
                    'Pct_Under_10_Error': pct_under_10,
                    'R2': r2_score(mat_data['Actual'], mat_data['Predicted'])
                })
    
    material_df = pd.DataFrame(material_performance).sort_values('Pct_Under_10_Error', ascending=False)
    print("\n   Material Performance (sorted by % with <10% error):")
    print(material_df.to_string(index=False))
    
    # Step 8: Test specific problematic case (Cd)
    print("\n🔍 Step 8: Testing problematic case (Cd)...")
    cd_test = test_df[
        (test_df['Material'] == 'Cd') &
        (np.isclose(test_df['pH'], 11.88, rtol=1e-3)) &
        (np.isclose(test_df['Time_days'], 9.0, rtol=1e-3))
    ]
    if len(cd_test) > 0:
        cd_row = cd_test.iloc[0]
        print(f"   Cd test case:")
        print(f"      Actual: {cd_row['Actual']:.6f} mg/m²")
        print(f"      Predicted: {cd_row['Predicted']:.6f} mg/m²")
        print(f"      Error: {cd_row['Error_Pct']:.2f}%")
    else:
        print("   ⚠️  Cd test case not found in test set")
        # Try to predict it manually
        try:
            pred = pipeline.predict_leaching('Cd', 11.88, 9.0, 'Unknown', 'Concrete', 'LPx')
            print(f"   Manual prediction: {pred:.6f} mg/m²")
        except Exception as e:
            print(f"   ❌ Could not make manual prediction: {e}")
    
    # Step 9: Save model
    print("\n💾 Step 9: Saving model...")
    serializer = ModelSerializer('models')
    serializer.save_model(
        best_model, label_encoders, pt, scaler, feature_columns,
        {'model_type': best_model_name, 'feature_count': len(feature_columns)}
    )
    print(f"   ✅ Model saved as 'production_model'")
    
    # Step 10: Summary
    print("\n" + "=" * 80)
    print("RETRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Model: {best_model_name}")
    print(f"Test R²: {test_metrics['r2']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f} mg/m²")
    if not np.isnan(test_metrics['pct_under_10']):
        print(f"Test Predictions <10% error: {test_metrics['pct_under_10']:.2f}%")
        print(f"Test Predictions <20% error: {test_metrics['pct_under_20']:.2f}%")
    
    # Save performance report
    performance_report = {
        'best_model': best_model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'material_performance': material_df.to_dict('records')
    }
    
    import json
    with open('results/retraining_performance.json', 'w') as f:
        json.dump(performance_report, f, indent=2, default=str)
    print(f"\n✅ Performance report saved to: results/retraining_performance.json")
    
    material_df.to_csv('results/material_performance_retrained.csv', index=False)
    print(f"✅ Material performance saved to: results/material_performance_retrained.csv")

if __name__ == "__main__":
    main()

