#!/usr/bin/env python3
"""
Simple prediction script for cement leaching.

This script loads the trained model and allows users to make predictions
by inputting material properties interactively.
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


def load_model():
    """Load the trained model and its components."""
    print("📂 Loading trained model...")
    
    try:
        serializer = ModelSerializer('models')
        model, label_encoders, power_transformer, scaler, feature_columns, metadata = serializer.load_model_components('production_model')
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"🔧 Features: {metadata.get('feature_count', 0)}")
        print(f"📈 Materials: {len(metadata.get('materials', []))}")
        
        return model, label_encoders, power_transformer, scaler, feature_columns, metadata
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None, None, None


def get_user_input():
    """Get prediction inputs from user."""
    print("\n🧪 CEMENT LEACHING PREDICTION")
    print("=" * 40)
    
    # Load model first to get available options
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = load_model()
    
    if model is None:
        return None, None, None, None, None, None
    
    print(f"\n📋 Available Materials: {', '.join(metadata.get('materials', []))}")
    print(f"📋 Available Cement Types: {', '.join(metadata.get('cement_types', []))}")
    print(f"📋 Available Form Types: {', '.join(metadata.get('form_types', []))}")
    print(f"📋 Available Stat Measures: {', '.join(metadata.get('stat_measures', []))}")
    
    print("\n" + "=" * 40)
    
    try:
        # Get material
        material = input("Enter Material (e.g., Al, Pb, Zn): ").strip()
        
        # Get pH
        ph = float(input("Enter pH value (1-14): "))
        
        # Get time
        time_days = float(input("Enter time in days (0.01-100): "))
        
        # Get cement type
        cement_type = input("Enter Cement Type (e.g., CEM_I, CEM_II): ").strip()
        
        # Get form type
        form_type = input("Enter Form Type (e.g., Concrete, Mortar): ").strip()
        
        # Get stat measure
        stat_measure = input("Enter Statistical Measure (e.g., CL_Minus, CL_Plus): ").strip()
        
        return material, ph, time_days, cement_type, form_type, stat_measure, model, label_encoders, power_transformer, scaler, feature_columns
        
    except ValueError:
        print("❌ Invalid input. Please enter valid numbers for pH and time.")
        return None, None, None, None, None, None, None, None, None, None, None
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return None, None, None, None, None, None, None, None, None, None, None


def predict_leaching(material, ph, time_days, cement_type, form_type, stat_measure,
                    model, label_encoders, power_transformer, scaler, feature_columns):
    """Make leaching prediction."""
    
    # Safe encoding function
    def safe_encode(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else 0
    
    # Material grouping
    material_groups = {'Al': 0, 'Fe': 0, 'Si': 0, 'As': 1, 'Cr': 1, 'Mo': 1,
                      'Ba': 2, 'P': 2, 'Br': 3, 'F': 3, 'Cl': 3,
                      'Ca': 4, 'K': 4, 'Mg': 4, 'Na': 4,
                      'Cd': 5, 'Cu': 5, 'Pb': 5, 'Zn': 5, 'SO4': 6}
    
    try:
        # Encode categorical features
        material_enc = safe_encode(label_encoders['Material'], material)
        cement_enc = safe_encode(label_encoders['Cement_Type'], cement_type)
        form_enc = safe_encode(label_encoders['Form_Type'], form_type)
        stat_enc = safe_encode(label_encoders['Stat_Measure'], stat_measure)
        
        # Create complete feature dictionary
        feat = {
            'Material_encoded': material_enc,
            'Cement_Type_encoded': cement_enc,
            'Form_Type_encoded': form_enc,
            'Stat_Measure_encoded': stat_enc,
            'pH': ph,
            'Time_days': time_days,
            'Cement_Content': 80,  # Default
            'Additives_Count': 1,  # Default
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
        
        return float(prediction)
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def main():
    """Main function to run the prediction script."""
    
    # Get user inputs
    inputs = get_user_input()
    if inputs[0] is None:  # Check if any input is None (error case)
        return
    
    material, ph, time_days, cement_type, form_type, stat_measure, model, label_encoders, power_transformer, scaler, feature_columns = inputs
    
    # Validate inputs
    if not (1 <= ph <= 14):
        print("❌ Error: pH must be between 1 and 14")
        return
    
    if not (0.01 <= time_days <= 100):
        print("❌ Error: Time must be between 0.01 and 100 days")
        return
    
    # Make prediction
    try:
        print("\n🔮 Making prediction...")
        prediction = predict_leaching(material, ph, time_days, cement_type, form_type, stat_measure,
                                    model, label_encoders, power_transformer, scaler, feature_columns)
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 PREDICTION RESULTS")
        print("=" * 50)
        print(f"📊 Material: {material}")
        print(f"🧪 pH: {ph}")
        print(f"⏱️  Time: {time_days} days")
        print(f"🏗️  Cement Type: {cement_type}")
        print(f"🔧 Form Type: {form_type}")
        print(f"📈 Statistical Measure: {stat_measure}")
        print(f"\n🎯 PREDICTED LEACHING: {prediction:.2f} mg/m²")
        
        # Confidence assessment
        if prediction < 10:
            confidence = "High"
        elif prediction < 100:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        print(f"🎖️  Confidence: {confidence}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if prediction < 1:
            print("   Very low leaching - excellent performance")
        elif prediction < 10:
            print("   Low leaching - good performance")
        elif prediction < 50:
            print("   Moderate leaching - acceptable performance")
        else:
            print("   High leaching - may need attention")
        
        print("\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def quick_predict_example():
    """Example of how to use the prediction function programmatically."""
    
    print("\n🚀 QUICK PREDICTION EXAMPLE")
    print("=" * 40)
    
    # Load model
    model, label_encoders, power_transformer, scaler, feature_columns, metadata = load_model()
    
    if model is None:
        return
    
    # Example predictions
    examples = [
        ("Al", 12.0, 1.0, "CEM_I", "Concrete", "CL_Minus"),
        ("Pb", 11.5, 9.0, "CEM_I", "Concrete", "CL_Plus"),
        ("Zn", 10.8, 36.0, "CEM_II", "Mortar", "Med"),
        ("Br", 12.2, 4.0, "CEM_I", "Concrete", "CL_Minus"),
        ("Cu", 11.0, 16.0, "CEM_III", "Sewage_Sludge", "LPx")
    ]
    
    print("\n🧪 Example Predictions:")
    print("-" * 50)
    
    for i, (material, ph, time_days, cement_type, form_type, stat_measure) in enumerate(examples, 1):
        try:
            prediction = predict_leaching(material, ph, time_days, cement_type, form_type, stat_measure,
                                        model, label_encoders, power_transformer, scaler, feature_columns)
            print(f"{i}. {material} at pH {ph} for {time_days} days → {prediction:.2f} mg/m²")
        except Exception as e:
            print(f"{i}. {material} → Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cement Leaching Prediction')
    parser.add_argument('--examples', action='store_true', help='Run example predictions')
    args = parser.parse_args()
    
    if args.examples:
        quick_predict_example()
    else:
        main()
