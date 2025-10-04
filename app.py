#!/usr/bin/env python3
"""
Flask web application for cement leaching prediction.

This application provides a web interface for making predictions using the trained model.
Users can input material properties and get leaching predictions through a simple web form.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from src.model_serializer import ModelSerializer

# Initialize Flask app
app = Flask(__name__)

# Global variables for model components
model = None
label_encoders = None
power_transformer = None
scaler = None
feature_columns = None
model_metadata = None

def load_model():
    """Load the trained model and its components."""
    global model, label_encoders, power_transformer, scaler, feature_columns, model_metadata
    
    try:
        serializer = ModelSerializer('models')
        model, label_encoders, power_transformer, scaler, feature_columns, model_metadata = serializer.load_model_components('production_model')
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure you have trained and saved a model first.")
        return False

def safe_transform(le, val):
    """
    Safely transform a value using a label encoder.
    
    Args:
        le: Label encoder
        val: Value to transform
        
    Returns:
        Encoded value or 0 if value not found
    """
    return le.transform([val])[0] if val in le.classes_ else 0

def predict_leaching(material, ph, time_days, cement_type, form_type, stat_measure):
    """
    Make a leaching prediction using the trained model.
    
    Args:
        material: Material name
        ph: pH value
        time_days: Time in days
        cement_type: Type of cement
        form_type: Form type
        stat_measure: Statistical measure
        
    Returns:
        Predicted leaching value in mg/m²
    """
    global model, label_encoders, power_transformer, scaler, feature_columns, model_metadata
    
    # Load model if not already loaded
    if model is None:
        if not load_model():
            return None
    
    try:
        # Encode categorical features
        mat_enc = safe_transform(label_encoders['Material'], material)
        cement_enc = safe_transform(label_encoders['Cement_Type'], cement_type)
        form_enc = safe_transform(label_encoders['Form_Type'], form_type)
        stat_enc = safe_transform(label_encoders['Stat_Measure'], stat_measure)
        
        # Material grouping
        material_groups = {'Al': 0, 'Fe': 0, 'Si': 0, 'As': 1, 'Cr': 1, 'Mo': 1,
                          'Ba': 2, 'P': 2, 'Br': 3, 'F': 3, 'Cl': 3,
                          'Ca': 4, 'K': 4, 'Mg': 4, 'Na': 4,
                          'Cd': 5, 'Cu': 5, 'Pb': 5, 'Zn': 5, 'SO4': 6}
        
        # Create complete feature dictionary
        feat = {
            'Material_encoded': mat_enc,
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
            'Material_pH_interaction': mat_enc * ph,
            'Material_Time_interaction': mat_enc * time_days,
            'pH_normalized': (ph - 7.5) / 2.0,  # Approximate normalization
            'Time_normalized': (time_days - 10.0) / 15.0,  # Approximate normalization
            'Alkalinity_index': ph - 7,
            'Reactivity_score': ph * np.log1p(time_days),
            'Leaching_potential': (ph ** 2) * np.sqrt(time_days),
            'Material_group': material_groups.get(material, 0)
        }
        
        # Create DataFrame and ensure correct column order
        X_df = pd.DataFrame([feat])[feature_columns]
        
        # Make prediction (XGBoost was trained on original features, not transformed)
        y_log = model.predict(X_df)[0]
        prediction = np.expm1(y_log)  # Transform back to original scale
        
        # Ensure non-negative prediction (leaching values can never be negative)
        prediction = max(prediction, 0.0)
        
        return round(prediction, 2)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cement Leaching Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        select, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
        }
        .submit-btn {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 20px;
        }
        .submit-btn:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
        }
        .error {
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .info {
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .form-row {
            display: flex;
            gap: 20px;
        }
        .form-row .form-group {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Cement Leaching Prediction</h1>
        
        {% if model_loaded %}
        <div class="info">
            <strong>✅ Model Status:</strong> Ready for predictions<br>
            <strong>📊 Model Type:</strong> {{ model_metadata.model_type }}<br>
            <strong>🔧 Features:</strong> {{ model_metadata.feature_count }}<br>
            <strong>📈 Materials:</strong> {{ model_metadata.materials | length }} supported
        </div>
        
        <form method="POST">
            <h2>Input Parameters</h2>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="material">Material:</label>
                    <select name="material" id="material" required>
                        <option value="">Select Material</option>
                        {% for m in model_metadata.materials %}
                        <option value="{{ m }}" {% if request.form.material == m %}selected{% endif %}>{{ m }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="ph">pH Value:</label>
                    <input type="number" name="ph" id="ph" step="0.01" min="1" max="14" 
                           value="{{ request.form.ph if request.form.ph else '12.0' }}" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="time_days">Time (days):</label>
                    <input type="number" name="time_days" id="time_days" step="0.01" min="0.01" 
                           value="{{ request.form.time_days if request.form.time_days else '1.0' }}" required>
                </div>
                
                <div class="form-group">
                    <label for="cement_type">Cement Type:</label>
                    <select name="cement_type" id="cement_type" required>
                        <option value="">Select Cement Type</option>
                        {% for c in model_metadata.cement_types %}
                        <option value="{{ c }}" {% if request.form.cement_type == c %}selected{% endif %}>{{ c }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="form_type">Form Type:</label>
                    <select name="form_type" id="form_type" required>
                        <option value="">Select Form Type</option>
                        {% for f in model_metadata.form_types %}
                        <option value="{{ f }}" {% if request.form.form_type == f %}selected{% endif %}>{{ f }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="stat_measure">Statistical Measure:</label>
                    <select name="stat_measure" id="stat_measure" required>
                        <option value="">Select Statistical Measure</option>
                        {% for s in model_metadata.stat_measures %}
                        <option value="{{ s }}" {% if request.form.stat_measure == s %}selected{% endif %}>{{ s }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            
            <button type="submit" class="submit-btn">🔮 Predict Leaching</button>
        </form>
        
        {% if prediction %}
        <div class="result">
            <h3>Prediction Result</h3>
            <div class="prediction">{{ prediction }} mg/m²</div>
            <p><strong>Input Summary:</strong> {{ request.form.material }} at pH {{ request.form.ph }} for {{ request.form.time_days }} days</p>
            <p><strong>Conditions:</strong> {{ request.form.cement_type }}, {{ request.form.form_type }}, {{ request.form.stat_measure }}</p>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="error">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}
        
        {% else %}
        <div class="error">
            <strong>❌ Model Not Available</strong><br>
            Please ensure you have trained and saved a model first.<br>
            Run the training pipeline and save the model using the model serializer.
        </div>
        {% endif %}
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 14px;">
            <p><strong>21-Elements Monolithic Cement Leaching Prediction</strong></p>
            <p>This application uses machine learning to predict elemental leaching from cement materials.</p>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    """Main route for the web application."""
    prediction = None
    error = None
    
    if request.method == "POST" and model is not None:
        try:
            # Get form data
            material = request.form["material"]
            ph = float(request.form["ph"])
            time_days = float(request.form["time_days"])
            cement_type = request.form["cement_type"]
            form_type = request.form["form_type"]
            stat_measure = request.form["stat_measure"]
            
            # Validate inputs
            if not (1 <= ph <= 14):
                error = "pH must be between 1 and 14"
            elif not (0.01 <= time_days <= 100):
                error = "Time must be between 0.01 and 100 days"
            elif not all([material, cement_type, form_type, stat_measure]):
                error = "All fields are required"
            else:
                # Make prediction
                prediction = predict_leaching(material, ph, time_days, cement_type, form_type, stat_measure)
                if prediction is None:
                    error = "Prediction failed. Please check your inputs."
                    
        except ValueError as e:
            error = "Invalid input values. Please check your inputs."
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template_string(
        HTML_TEMPLATE,
        model_loaded=model is not None,
        model_metadata=model_metadata or {},
        prediction=prediction,
        error=error
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for programmatic predictions."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["material", "ph", "time_days", "cement_type", "form_type", "stat_measure"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Make prediction
        prediction = predict_leaching(
            data["material"], data["ph"], data["time_days"],
            data["cement_type"], data["form_type"], data["stat_measure"]
        )
        
        if prediction is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        return jsonify({
            "prediction": float(prediction),
            "unit": "mg/m²",
            "input": data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def api_status():
    """API endpoint to check model status."""
    return jsonify({
        "model_loaded": model is not None,
        "model_type": model_metadata.get("model_type", "Unknown") if model_metadata else "Unknown",
        "available_materials": model_metadata.get("materials", []) if model_metadata else [],
        "feature_count": model_metadata.get("feature_count", 0) if model_metadata else 0
    })

@app.route("/api/materials")
def api_materials():
    """API endpoint to get available materials and options."""
    if not model_metadata:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "materials": model_metadata.get("materials", []),
        "cement_types": model_metadata.get("cement_types", []),
        "form_types": model_metadata.get("form_types", []),
        "stat_measures": model_metadata.get("stat_measures", [])
    })

if __name__ == "__main__":
    print("🚀 Starting Cement Leaching Prediction Web App")
    print("=" * 50)
    
    # Load model on startup
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"📊 Available materials: {len(model_metadata.get('materials', []))}")
        print(f"🔧 Model type: {model_metadata.get('model_type', 'Unknown')}")
        print("\n🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:8080")
        print("🔗 API endpoint: http://localhost:8080/api/predict")
        print("\n" + "=" * 50)
        
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("❌ Failed to load model. Please train and save a model first.")
        print("\nTo train a model:")
        print("1. Run: python main.py")
        print("2. The model will be automatically saved")
        print("3. Then run: python app.py")
