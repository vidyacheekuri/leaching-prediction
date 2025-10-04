#!/usr/bin/env python3
"""
Sensitivity Analysis for Cement Leaching Predictions.

This script performs comprehensive sensitivity analysis to understand how
different parameters affect leaching predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.model_serializer import ModelSerializer


def load_model():
    """Load the trained model and components."""
    print("🤖 Loading trained model...")
    serializer = ModelSerializer('models')
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = serializer.load_model_components('production_model')
    return model, label_encoders, power_transformer, scaler, feature_columns, metadata


def predict_leaching(material, ph, time_days, cement_type='CEM_I', form_type='Concrete', stat_measure='CL_Minus',
                    model=None, label_encoders=None, power_transformer=None, scaler=None, feature_columns=None):
    """Make a single leaching prediction."""
    
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
    
    # Create DataFrame and predict
    X_df = pd.DataFrame([feat])[feature_columns]
    y_pred_log = model.predict(X_df)[0]
    y_pred = np.expm1(y_pred_log)
    
    # Ensure non-negative prediction
    y_pred = max(y_pred, 0.0)
    
    return float(y_pred)


def ph_sensitivity_analysis(model, label_encoders, power_transformer, scaler, feature_columns):
    """Analyze sensitivity to pH changes."""
    print("\n🧪 pH Sensitivity Analysis")
    print("=" * 40)
    
    # Test materials
    materials = ['Al', 'Pb', 'Zn', 'Br', 'Cu']
    ph_range = np.linspace(1.0, 14.0, 50)
    time_days = 1.0
    
    results = []
    
    for material in materials:
        for ph in ph_range:
            pred = predict_leaching(material, ph, time_days, 
                                  model=model, label_encoders=label_encoders,
                                  power_transformer=power_transformer, scaler=scaler,
                                  feature_columns=feature_columns)
            results.append({
                'Material': material,
                'pH': ph,
                'Time_days': time_days,
                'Prediction': pred
            })
    
    df_ph = pd.DataFrame(results)
    
    # Save results
    df_ph.to_csv('results/ph_sensitivity_analysis.csv', index=False)
    print("✅ pH sensitivity results saved to results/ph_sensitivity_analysis.csv")
    
    # Print summary
    print(f"\n📊 pH Sensitivity Summary:")
    for material in materials:
        material_data = df_ph[df_ph['Material'] == material]
        min_pred = material_data['Prediction'].min()
        max_pred = material_data['Prediction'].max()
        ph_at_min = material_data.loc[material_data['Prediction'].idxmin(), 'pH']
        ph_at_max = material_data.loc[material_data['Prediction'].idxmax(), 'pH']
        
        print(f"   {material}: {min_pred:.2f} - {max_pred:.2f} mg/m²")
        print(f"     Min at pH {ph_at_min:.1f}, Max at pH {ph_at_max:.1f}")
    
    return df_ph


def time_sensitivity_analysis(model, label_encoders, power_transformer, scaler, feature_columns):
    """Analyze sensitivity to time changes."""
    print("\n⏱️ Time Sensitivity Analysis")
    print("=" * 40)
    
    # Test materials
    materials = ['Al', 'Pb', 'Zn', 'Br', 'Cu']
    time_range = np.logspace(-1, 2, 50)  # 0.1 to 100 days
    ph = 12.0
    
    results = []
    
    for material in materials:
        for time_days in time_range:
            pred = predict_leaching(material, ph, time_days,
                                  model=model, label_encoders=label_encoders,
                                  power_transformer=power_transformer, scaler=scaler,
                                  feature_columns=feature_columns)
            results.append({
                'Material': material,
                'pH': ph,
                'Time_days': time_days,
                'Prediction': pred
            })
    
    df_time = pd.DataFrame(results)
    
    # Save results
    df_time.to_csv('results/time_sensitivity_analysis.csv', index=False)
    print("✅ Time sensitivity results saved to results/time_sensitivity_analysis.csv")
    
    # Print summary
    print(f"\n📊 Time Sensitivity Summary:")
    for material in materials:
        material_data = df_time[df_time['Material'] == material]
        min_pred = material_data['Prediction'].min()
        max_pred = material_data['Prediction'].max()
        time_at_min = material_data.loc[material_data['Prediction'].idxmin(), 'Time_days']
        time_at_max = material_data.loc[material_data['Prediction'].idxmax(), 'Time_days']
        
        print(f"   {material}: {min_pred:.2f} - {max_pred:.2f} mg/m²")
        print(f"     Min at {time_at_min:.2f} days, Max at {time_at_max:.2f} days")
    
    return df_time


def material_comparison_analysis(model, label_encoders, power_transformer, scaler, feature_columns):
    """Compare leaching behavior across different materials."""
    print("\n🔬 Material Comparison Analysis")
    print("=" * 40)
    
    # Test conditions
    ph_values = [8.0, 10.0, 12.0]
    time_values = [0.1, 1.0, 10.0, 30.0]
    
    # Get all available materials
    materials = list(label_encoders['Material'].classes_)
    
    results = []
    
    for material in materials:
        for ph in ph_values:
            for time_days in time_values:
                pred = predict_leaching(material, ph, time_days,
                                      model=model, label_encoders=label_encoders,
                                      power_transformer=power_transformer, scaler=scaler,
                                      feature_columns=feature_columns)
                results.append({
                    'Material': material,
                    'pH': ph,
                    'Time_days': time_days,
                    'Prediction': pred
                })
    
    df_material = pd.DataFrame(results)
    
    # Save results
    df_material.to_csv('results/material_comparison_analysis.csv', index=False)
    print("✅ Material comparison results saved to results/material_comparison_analysis.csv")
    
    # Print summary by material
    print(f"\n📊 Material Leaching Summary (pH=12, Time=1 day):")
    baseline_data = df_material[(df_material['pH'] == 12.0) & (df_material['Time_days'] == 1.0)]
    baseline_sorted = baseline_data.sort_values('Prediction', ascending=False)
    
    for _, row in baseline_sorted.head(10).iterrows():
        print(f"   {row['Material']}: {row['Prediction']:.2f} mg/m²")
    
    return df_material


def interaction_analysis(model, label_encoders, power_transformer, scaler, feature_columns):
    """Analyze pH-Time interactions."""
    print("\n🔄 pH-Time Interaction Analysis")
    print("=" * 40)
    
    # Create pH-Time grid
    ph_range = np.linspace(8.0, 13.0, 20)
    time_range = np.logspace(-0.5, 1.5, 20)  # 0.3 to 31.6 days
    
    materials = ['Al', 'Pb', 'Zn']
    
    results = []
    
    for material in materials:
        for ph in ph_range:
            for time_days in time_range:
                pred = predict_leaching(material, ph, time_days,
                                      model=model, label_encoders=label_encoders,
                                      power_transformer=power_transformer, scaler=scaler,
                                      feature_columns=feature_columns)
                results.append({
                    'Material': material,
                    'pH': ph,
                    'Time_days': time_days,
                    'Prediction': pred
                })
    
    df_interaction = pd.DataFrame(results)
    
    # Save results
    df_interaction.to_csv('results/ph_time_interaction_analysis.csv', index=False)
    print("✅ pH-Time interaction results saved to results/ph_time_interaction_analysis.csv")
    
    # Print summary
    print(f"\n📊 Interaction Analysis Summary:")
    for material in materials:
        material_data = df_interaction[df_interaction['Material'] == material]
        max_pred = material_data['Prediction'].max()
        max_idx = material_data['Prediction'].idxmax()
        max_ph = material_data.loc[max_idx, 'pH']
        max_time = material_data.loc[max_idx, 'Time_days']
        
        print(f"   {material}: Max {max_pred:.2f} mg/m² at pH {max_ph:.1f}, {max_time:.1f} days")
    
    return df_interaction


def create_visualizations(df_ph, df_time, df_material, df_interaction):
    """Create visualization plots for sensitivity analysis."""
    print("\n📊 Creating Visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. pH Sensitivity Plot
    plt.figure(figsize=(12, 8))
    for material in df_ph['Material'].unique():
        material_data = df_ph[df_ph['Material'] == material]
        plt.plot(material_data['pH'], material_data['Prediction'], 
                label=material, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('pH')
    plt.ylabel('Predicted Leaching (mg/m²)')
    plt.title('pH Sensitivity Analysis\n(Time = 1 day)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/ph_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time Sensitivity Plot
    plt.figure(figsize=(12, 8))
    for material in df_time['Material'].unique():
        material_data = df_time[df_time['Material'] == material]
        plt.loglog(material_data['Time_days'], material_data['Prediction'], 
                  label=material, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Predicted Leaching (mg/m²)')
    plt.title('Time Sensitivity Analysis\n(pH = 12.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/time_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Material Comparison Heatmap
    plt.figure(figsize=(14, 10))
    pivot_data = df_material.pivot_table(values='Prediction', 
                                        index='Material', 
                                        columns=['pH', 'Time_days'], 
                                        aggfunc='mean')
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Predicted Leaching (mg/m²)'})
    plt.title('Material Leaching Comparison\n(Heatmap of pH-Time combinations)')
    plt.tight_layout()
    plt.savefig('results/plots/material_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to results/plots/")


def main():
    """Run comprehensive sensitivity analysis."""
    print("🔬 Cement Leaching Sensitivity Analysis")
    print("=" * 50)
    
    # Load model
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = load_model()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run analyses
    df_ph = ph_sensitivity_analysis(model, label_encoders, power_transformer, scaler, feature_columns)
    df_time = time_sensitivity_analysis(model, label_encoders, power_transformer, scaler, feature_columns)
    df_material = material_comparison_analysis(model, label_encoders, power_transformer, scaler, feature_columns)
    df_interaction = interaction_analysis(model, label_encoders, power_transformer, scaler, feature_columns)
    
    # Create visualizations
    try:
        create_visualizations(df_ph, df_time, df_material, df_interaction)
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        print("   (This is optional - analysis results are still saved)")
    
    # Summary
    print(f"\n🎉 Sensitivity Analysis Complete!")
    print(f"📁 Results saved:")
    print(f"   - results/ph_sensitivity_analysis.csv")
    print(f"   - results/time_sensitivity_analysis.csv")
    print(f"   - results/material_comparison_analysis.csv")
    print(f"   - results/ph_time_interaction_analysis.csv")
    print(f"   - results/plots/ (visualizations)")


if __name__ == "__main__":
    main()
