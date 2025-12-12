#!/usr/bin/env python3
"""
Verification script to check if predictions match Excel data.

This script loads specific test cases from the Excel file and compares
model predictions with actual values to identify any mismatches.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.model_serializer import ModelSerializer
from src.data_processing import DataProcessor


def load_model():
    """Load the trained model."""
    serializer = ModelSerializer('models')
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = serializer.load_model_components('production_model')
    return model, label_encoders, power_transformer, scaler, feature_columns


def predict_leaching(material, ph, time_days, cement_type, form_type, stat_measure,
                    model, label_encoders, power_transformer, scaler, feature_columns):
    """Make a prediction using the model."""
    def safe_encode(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else 0
    
    material_groups = {'Al': 0, 'Fe': 0, 'Si': 0, 'As': 1, 'Cr': 1, 'Mo': 1,
                      'Ba': 2, 'P': 2, 'Br': 3, 'F': 3, 'Cl': 3,
                      'Ca': 4, 'K': 4, 'Mg': 4, 'Na': 4,
                      'Cd': 5, 'Cu': 5, 'Pb': 5, 'Zn': 5, 'SO4': 6}
    
    material_enc = safe_encode(label_encoders['Material'], material)
    cement_enc = safe_encode(label_encoders['Cement_Type'], cement_type)
    form_enc = safe_encode(label_encoders['Form_Type'], form_type)
    stat_enc = safe_encode(label_encoders['Stat_Measure'], stat_measure)
    
    feat = {
        'Material_encoded': material_enc,
        'Cement_Type_encoded': cement_enc,
        'Form_Type_encoded': form_enc,
        'Stat_Measure_encoded': stat_enc,
        'pH': ph,
        'Time_days': time_days,
        'Cement_Content': 80,
        'Additives_Count': 1,
        'log_Time': np.log1p(time_days),
        'log_pH': np.log(ph),
        'sqrt_Time': np.sqrt(time_days),
        'sqrt_pH': np.sqrt(ph),
        'pH_squared': ph ** 2,
        'pH_cubed': ph ** 3,
        'Time_squared': time_days ** 2,
        'Time_pH_interaction': time_days * ph,
        'log_Time_pH': np.log1p(time_days) * ph,
        'Material_pH_interaction': material_enc * ph,
        'Material_Time_interaction': material_enc * time_days,
        'pH_normalized': (ph - 7.5) / 2.0,
        'Time_normalized': (time_days - 10.0) / 15.0,
        'Alkalinity_index': ph - 7,
        'Reactivity_score': ph * np.log1p(time_days),
        'Leaching_potential': (ph ** 2) * np.sqrt(time_days),
        'Material_group': material_groups.get(material, 0)
    }
    
    X_df = pd.DataFrame([feat])[feature_columns]
    X_transformed = power_transformer.transform(X_df)
    X_scaled = scaler.transform(X_transformed)
    y_log = model.predict(X_scaled)[0]
    prediction = np.expm1(y_log)
    return max(float(prediction), 0.0)


def verify_predictions():
    """Verify predictions against Excel data."""
    print("=" * 70)
    print("🔍 PREDICTION VERIFICATION")
    print("=" * 70)
    print()
    
    # Load model
    print("📂 Loading model...")
    model, label_encoders, power_transformer, scaler, feature_columns = load_model()
    print("✅ Model loaded")
    print()
    
    # Load Excel data
    print("📂 Loading Excel data...")
    processor = DataProcessor('LXS-Monolithe-21.xlsx')
    df = processor.load_and_consolidate_data()
    df, _, _ = processor.create_features(df)
    print("✅ Excel data loaded")
    print()
    
    # Test first record for each material
    print("=" * 70)
    print("🧪 Testing First Record for Each Material")
    print("=" * 70)
    print()
    
    mismatches = []
    matches = []
    
    for material in df['Material'].unique():
        material_df = df[df['Material'] == material].copy()
        if len(material_df) == 0:
            continue
        
        # Get first record
        first_record = material_df.iloc[0]
        
        actual = first_record['Cumulative_Release_mg_m2']
        ph = first_record['pH']
        time_days = first_record['Time_days']
        cement_type = first_record['Cement_Type']
        form_type = first_record['Form_Type']
        stat_measure = first_record['Stat_Measure']
        
        # Make prediction
        predicted = predict_leaching(
            material, ph, time_days, cement_type, form_type, stat_measure,
            model, label_encoders, power_transformer, scaler, feature_columns
        )
        
        # Calculate error
        error_pct = abs(predicted - actual) / actual * 100 if actual > 0 else np.nan
        
        result = {
            'Material': material,
            'pH': ph,
            'Time_days': time_days,
            'Actual': actual,
            'Predicted': predicted,
            'Error_%': error_pct,
            'Cement_Type': cement_type,
            'Form_Type': form_type,
            'Stat_Measure': stat_measure,
            'Material_Condition': first_record.get('Material_Condition', 'N/A')
        }
        
        if error_pct > 20:  # Consider >20% error as mismatch
            mismatches.append(result)
        else:
            matches.append(result)
        
        print(f"{material:3s} | pH={ph:5.2f} | Time={time_days:6.2f} | "
              f"Actual={actual:10.2f} | Predicted={predicted:10.2f} | "
              f"Error={error_pct:6.2f}%")
    
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Matches (<20% error): {len(matches)}")
    print(f"❌ Mismatches (>20% error): {len(mismatches)}")
    print()
    
    if mismatches:
        print("⚠️  MISMATCHES FOUND:")
        print("-" * 70)
        mismatch_df = pd.DataFrame(mismatches)
        print(mismatch_df.to_string(index=False))
        print()
        mismatch_df.to_csv('results/prediction_mismatches.csv', index=False)
        print("💾 Saved to: results/prediction_mismatches.csv")
    
    # Save all results
    all_results = matches + mismatches
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/prediction_verification.csv', index=False)
    print("💾 All results saved to: results/prediction_verification.csv")
    
    return results_df


if __name__ == "__main__":
    verify_predictions()

