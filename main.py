#!/usr/bin/env python3
"""
Main execution script for the cement leaching prediction project.

This script demonstrates the complete workflow from data loading to model training
and prediction for the cement leaching prediction project.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DataProcessor
from src.ml_pipeline import MLPipeline
from src.models.conditional_ensemble import ConditionalEnsemble
from src.model_serializer import ModelSerializer
from src.utils import load_config, print_model_summary, create_prediction_report
import numpy as np


def main():
    """Main execution function."""
    print("🚀 21-Elements Monolithic Cement Leaching Prediction")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize data processor
    excel_file = config.get('paths', {}).get('data_file', 'data/LXS-Monolithe-21.xlsx')
    processor = DataProcessor(excel_file)
    
    try:
        # Load and consolidate data
        print("\n📊 Step 1: Loading and processing data...")
        df = processor.load_and_consolidate_data()
        
        # Create features
        print("\n🔧 Step 2: Feature engineering...")
        df, feature_columns, label_encoders = processor.create_features(df)
        
        # Save processed data
        output_dir = config.get('paths', {}).get('output_dir', 'data/processed')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        processor.save_processed_data(df, os.path.join(output_dir, 'consolidated_leaching_data_FINAL.csv'))
        
        # Get data summary
        summary = processor.get_data_summary(df)
        print(f"\n📈 Dataset Summary:")
        print(f"   Total samples: {summary['total_samples']}")
        print(f"   Materials: {summary['num_materials']}")
        print(f"   pH range: {summary['ph_range'][0]:.2f} - {summary['ph_range'][1]:.2f}")
        print(f"   Time range: {summary['time_range'][0]:.2f} - {summary['time_range'][1]:.2f} days")
        
        # Initialize ML pipeline
        print("\n🤖 Step 3: Training machine learning models...")
        pipeline = MLPipeline(random_state=config.get('data', {}).get('random_state', 42))
        
        # Prepare data for training
        X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = pipeline.prepare_data(
            df, feature_columns, label_encoders, 
            test_size=config.get('data', {}).get('test_size', 0.2)
        )
        
        # Train models
        best_model, best_model_name, results, models, pt, scaler = pipeline.train_models(
            X_train, X_test, y_train, y_test, y_train_orig, y_test_orig
        )
        
        # Print results
        print_model_summary(results, best_model_name)
        
        # Train conditional ensemble for improved accuracy
        print("\n🎯 Step 4: Training conditional ensemble for problematic materials...")
        conditional_ensemble = ConditionalEnsemble(random_state=42)
        
        # Train specialized models
        specialized_results = conditional_ensemble.train_specialized_models(
            df, feature_columns, label_encoders
        )
        
        # Set global model
        conditional_ensemble.set_global_model(best_model, pt, scaler)
        
        # Evaluate improvement
        improvement = conditional_ensemble.evaluate_improvement(
            df.loc[y_test.index], y_test_orig.values, 
            np.expm1(best_model.predict(X_test if best_model_name not in ['Neural Network', 'Elastic Net'] else scaler.transform(pt.transform(X_test))))
        )
        
        print(f"\n📊 Conditional Ensemble Improvement:")
        print(f"   High error reduction: {improvement['improvement']['high_error_reduction']} samples")
        print(f"   RMSE reduction: {improvement['improvement']['rmse_reduction']:.1f}%")
        print(f"   MAE reduction: {improvement['improvement']['mae_reduction']:.1f}%")
        
        # Save the trained model automatically
        print("\n💾 Step 4.5: Saving trained model...")
        serializer = ModelSerializer('models')
        saved_files = serializer.save_model_components(
            model=best_model,
            label_encoders=label_encoders,
            power_transformer=pt,
            scaler=scaler,
            feature_columns=feature_columns,
            model_name='production_model'
        )
        print("✅ Model saved successfully!")
        
        # Demonstration predictions
        print("\n🧪 Step 5: Demonstration predictions...")
        test_cases = [
            {'material': 'Al', 'ph': 12.0, 'time_days': 1.0},
            {'material': 'Pb', 'ph': 11.5, 'time_days': 9.0},
            {'material': 'Zn', 'ph': 10.8, 'time_days': 36.0},
            {'material': 'Br', 'ph': 12.2, 'time_days': 4.0},
            {'material': 'Cu', 'ph': 11.0, 'time_days': 16.0}
        ]
        
        predictions = []
        for i, test_case in enumerate(test_cases, 1):
            result = pipeline.predict_leaching(**test_case)
            predictions.append(result)
            
            print(f"\nTest {i}: {test_case}")
            if 'error' not in result:
                print(f"   → Predicted: {result['predicted_leaching_mg_m2']} mg/m²")
                print(f"   → Confidence: {result['confidence']}")
                print(f"   → Model: {result['model_used']} (R²={result['model_r2_score']})")
            else:
                print(f"   → Error: {result['error']}")
        
        # Create prediction report
        report_df = create_prediction_report(predictions)
        
        # Save results
        results_dir = config.get('paths', {}).get('results_dir', 'results')
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        report_df.to_csv(os.path.join(results_dir, 'prediction_report.csv'), index=False)
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pipeline.get_feature_importance(top_n=10)
            print(f"\n📊 Top 10 Most Important Features:")
            print("-" * 40)
            for idx, row in feature_importance.iterrows():
                print(f"{row['feature']:25s}: {row['importance']:.4f}")
        
        # Final summary
        model_summary = pipeline.get_model_summary()
        print(f"\n🎉 PROJECT COMPLETE!")
        print(f"🏆 Best Model: {model_summary['best_model']}")
        print(f"📊 R² Score: {model_summary['r2_score']:.4f}")
        print(f"🎯 RMSE: {model_summary['rmse']:.2f} mg/m²")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"💾 Ready for Production: ✅")
        
        if model_summary['r2_score'] > 0.85:
            print("\n🚀 MISSION ACCOMPLISHED! High-quality predictive model ready!")
        elif model_summary['r2_score'] > 0.75:
            print("\n✅ SUCCESS! Good predictive model achieved!")
        else:
            print("\n⚠️  Consider additional feature engineering or data preprocessing.")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Excel file '{excel_file}' not found.")
        print("Please ensure the data file is in the project directory.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
