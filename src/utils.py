"""
Utility functions for the cement leaching prediction project.

This module contains helper functions for data validation, visualization,
model evaluation, and other common tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import yaml
import os
from pathlib import Path


def load_config(config_path: str = 'config/model_config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YamlError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


def validate_inputs(material: str, ph: float, time_days: float, 
                   available_materials: List[str]) -> Dict[str, Any]:
    """
    Validate input parameters for prediction.
    
    Args:
        material: Material name
        ph: pH value
        time_days: Time in days
        available_materials: List of available materials
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Validate material
    if material not in available_materials:
        errors.append(f"Material '{material}' not supported. Available: {available_materials}")
    
    # Validate pH
    if not isinstance(ph, (int, float)) or not (1 <= ph <= 14):
        errors.append("pH must be a number between 1 and 14")
    elif ph < 3 or ph > 13:
        warnings.append(f"pH value {ph} is outside typical range (3-13)")
    
    # Validate time
    if not isinstance(time_days, (int, float)) or not (0.01 <= time_days <= 100):
        errors.append("Time must be a number between 0.01 and 100 days")
    elif time_days > 64:
        warnings.append(f"Time value {time_days} exceeds standard leaching test duration")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def calculate_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive error metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing error metrics
    """
    # Handle zero values for percentage error
    mask = y_true != 0
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    metrics = {
        'r2_score': float(np.corrcoef(y_true, y_pred)[0, 1] ** 2),
        'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        'mae': float(np.mean(np.abs(y_true - y_pred))),
        'mape': float(np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100),
        'median_ape': float(np.median(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100),
        'max_error': float(np.max(np.abs(y_true - y_pred))),
        'mean_error': float(np.mean(y_pred - y_true)),
        'std_error': float(np.std(y_pred - y_true))
    }
    
    return metrics


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                  title: str = "Residual Plot", figsize: Tuple[int, int] = (10, 6)):
    """
    Create residual plots for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    # Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, 
                           top_n: int = 15, figsize: Tuple[int, int] = (10, 8)):
    """
    Plot feature importance from tree-based models.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show
        figsize: Figure size
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.title(f"Top {top_n} Feature Importance")
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


def create_error_analysis_report(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                                material_col: str = 'Material') -> Dict[str, Any]:
    """
    Create comprehensive error analysis report.
    
    Args:
        df: DataFrame containing predictions and actual values
        y_true_col: Column name for true values
        y_pred_col: Column name for predicted values
        material_col: Column name for material
        
    Returns:
        Dictionary containing error analysis results
    """
    df = df.copy()
    df['abs_error'] = np.abs(df[y_pred_col] - df[y_true_col])
    df['pct_error'] = np.where(df[y_true_col] != 0, 
                               df['abs_error'] / df[y_true_col] * 100, np.nan)
    
    # Overall statistics
    overall_stats = calculate_error_metrics(df[y_true_col].values, df[y_pred_col].values)
    
    # Material-specific statistics
    material_stats = {}
    for material in df[material_col].unique():
        material_data = df[df[material_col] == material]
        material_stats[material] = calculate_error_metrics(
            material_data[y_true_col].values, 
            material_data[y_pred_col].values
        )
        material_stats[material]['sample_count'] = len(material_data)
    
    # High error samples
    high_error_threshold = 10  # 10% error
    high_error_samples = df[df['pct_error'] > high_error_threshold].copy()
    
    return {
        'overall_stats': overall_stats,
        'material_stats': material_stats,
        'high_error_samples': high_error_samples,
        'high_error_count': len(high_error_samples),
        'high_error_percentage': len(high_error_samples) / len(df) * 100
    }


def save_model_artifacts(model, scaler, transformer, label_encoders, 
                        feature_columns, output_dir: str = 'models'):
    """
    Save model artifacts for deployment.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        transformer: Fitted transformer
        label_encoders: Dictionary of label encoders
        feature_columns: List of feature columns
        output_dir: Output directory
    """
    import joblib
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save artifacts
    joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(transformer, os.path.join(output_dir, 'transformer.pkl'))
    joblib.dump(label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
    
    # Save feature columns as text file
    with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    print(f"Model artifacts saved to {output_dir}")


def load_model_artifacts(model_dir: str = 'models'):
    """
    Load model artifacts for prediction.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Tuple of (model, scaler, transformer, label_encoders, feature_columns)
    """
    import joblib
    
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    transformer = joblib.load(os.path.join(model_dir, 'transformer.pkl'))
    label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    
    # Load feature columns
    with open(os.path.join(model_dir, 'feature_columns.txt'), 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    return model, scaler, transformer, label_encoders, feature_columns


def print_model_summary(results_df: pd.DataFrame, best_model_name: str):
    """
    Print formatted model performance summary.
    
    Args:
        results_df: DataFrame containing model results
        best_model_name: Name of the best performing model
    """
    print("\n" + "="*60)
    print("🏆 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Display all results
    print(results_df.round(4))
    
    # Highlight best model
    best_score = results_df.loc[best_model_name, 'R²_score']
    print(f"\n🥇 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_score:.4f}")
    print(f"   RMSE: {results_df.loc[best_model_name, 'RMSE']:.2f} mg/m²")
    print(f"   MAE: {results_df.loc[best_model_name, 'MAE']:.2f} mg/m²")
    
    # Performance assessment
    if best_score > 0.9:
        print("\n🎉 OUTSTANDING! R² > 0.90 - Excellent model performance!")
    elif best_score > 0.8:
        print("\n✅ EXCELLENT! R² > 0.80 - Very good model performance!")
    elif best_score > 0.7:
        print("\n👍 GOOD! R² > 0.70 - Acceptable model performance!")
    else:
        print("\n⚠️  Model needs improvement. Consider more feature engineering.")


def create_prediction_report(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a formatted report of predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        DataFrame with formatted prediction results
    """
    report_data = []
    
    for i, pred in enumerate(predictions, 1):
        if 'error' not in pred:
            report_data.append({
                'Prediction_ID': i,
                'Material': pred.get('input_summary', '').split()[0],
                'Predicted_Leaching_mg_m2': pred['predicted_leaching_mg_m2'],
                'Model_Used': pred['model_used'],
                'Model_R2': pred['model_r2_score'],
                'Confidence': pred['confidence'],
                'Input_Summary': pred['input_summary']
            })
        else:
            report_data.append({
                'Prediction_ID': i,
                'Material': 'ERROR',
                'Predicted_Leaching_mg_m2': np.nan,
                'Model_Used': 'ERROR',
                'Model_R2': np.nan,
                'Confidence': 'ERROR',
                'Input_Summary': pred['error']
            })
    
    return pd.DataFrame(report_data)
