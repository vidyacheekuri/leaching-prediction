#!/usr/bin/env python3
"""
Simple prediction script - Direct function call version.

Usage:
    python simple_predict.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_serializer import ModelSerializer

# Global variables for loaded model
_model = None
_label_encoders = None
_power_transformer = None
_scaler = None
_feature_columns = None

def load_model_once():
    """Load model once and cache it globally."""
    global _model, _label_encoders, _power_transformer, _scaler, _feature_columns
    
    if _model is None:
        print("📂 Loading model (first time)...")
        serializer = ModelSerializer('models')
        _model, _label_encoders, _power_transformer, _scaler, _feature_columns, _ = serializer.load_model_components('production_model')
        print("✅ Model loaded and cached!")
    
    return _model, _label_encoders, _power_transformer, _scaler, _feature_columns


def predict_leaching(material, ph, time_days, cement_type='CEM_I', form_type='Concrete', stat_measure='CL_Minus'):
    """
    Predict cement leaching for given parameters.
    
    Args:
        material (str): Material name (e.g., 'Al', 'Pb', 'Zn')
        ph (float): pH value (1-14)
        time_days (float): Time in days (0.01-100)
        cement_type (str): Type of cement (default: 'CEM_I')
        form_type (str): Form type (default: 'Concrete')
        stat_measure (str): Statistical measure (default: 'CL_Minus')
    
    Returns:
        float: Predicted leaching in mg/m²
    """
    
    # Load model
    model, label_encoders, power_transformer, scaler, feature_columns = load_model_once()
    
    # Safe encoding function
    def safe_encode(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else 0
    
    # Material grouping
    material_groups = {'Al': 0, 'Fe': 0, 'Si': 0, 'As': 1, 'Cr': 1, 'Mo': 1,
                      'Ba': 2, 'P': 2, 'Br': 3, 'F': 3, 'Cl': 3,
                      'Ca': 4, 'K': 4, 'Mg': 4, 'Na': 4,
                      'Cd': 5, 'Cu': 5, 'Pb': 5, 'Zn': 5, 'SO4': 6}
    
    # Encode categorical features
    material_enc = safe_encode(label_encoders['Material'], material)
    cement_enc = safe_encode(label_encoders['Cement_Type'], cement_type)
    form_enc = safe_encode(label_encoders['Form_Type'], form_type)
    stat_enc = safe_encode(label_encoders['Stat_Measure'], stat_measure)
    
    # Create feature dictionary
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
        'Time_pH_interaction': ph * time_days,
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
    
    # Create DataFrame and transform
    X_df = pd.DataFrame([feat])[feature_columns]
    X_transformed = power_transformer.transform(X_df)
    X_scaled = scaler.transform(X_transformed)
    
    # Make prediction
    y_log = model.predict(X_scaled)[0]
    prediction = np.expm1(y_log)
    
    # Ensure non-negative prediction (leaching values can never be negative)
    prediction = max(prediction, 0.0)
    
    return float(prediction)


def main():
    """Example usage of the prediction function."""
    
    print("🧪 CEMENT LEACHING PREDICTION")
    print("=" * 40)
    
    # Example 1: Simple prediction
    print("\n📊 Example 1: Basic prediction")
    prediction1 = predict_leaching('Al', 12.0, 1.0)
    print(f"Al at pH 12.0 for 1.0 days → {prediction1:.2f} mg/m²")
    
    # Example 2: With specific parameters
    print("\n📊 Example 2: With specific parameters")
    prediction2 = predict_leaching('Pb', 11.5, 9.0, 'CEM_I', 'Concrete', 'CL_Plus')
    print(f"Pb at pH 11.5 for 9.0 days (CEM_I, Concrete, CL_Plus) → {prediction2:.2f} mg/m²")
    
    # Example 3: Multiple predictions
    print("\n📊 Example 3: Multiple materials")
    materials = ['Al', 'Pb', 'Zn', 'Br', 'Cu']
    ph = 12.0
    time_days = 1.0
    
    for material in materials:
        pred = predict_leaching(material, ph, time_days)
        print(f"{material:2s} at pH {ph} for {time_days} days → {pred:.2f} mg/m²")
    
    # Example 4: Different pH values
    print("\n📊 Example 4: Effect of pH on Al leaching")
    material = 'Al'
    time_days = 1.0
    ph_values = [8.0, 10.0, 11.0, 12.0, 12.5]
    
    for ph in ph_values:
        pred = predict_leaching(material, ph, time_days)
        print(f"{material} at pH {ph:4.1f} for {time_days} days → {pred:.2f} mg/m²")
    
    # Example 5: Different time periods
    print("\n📊 Example 5: Effect of time on Pb leaching")
    material = 'Pb'
    ph = 11.5
    time_values = [0.08, 1.0, 4.0, 9.0, 16.0, 28.0, 64.0]
    
    for time_days in time_values:
        pred = predict_leaching(material, ph, time_days)
        print(f"{material} at pH {ph} for {time_days:5.2f} days → {pred:.2f} mg/m²")
    
    print("\n✅ All predictions completed!")


if __name__ == "__main__":
    main()
