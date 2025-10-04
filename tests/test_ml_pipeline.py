"""
Tests for ML pipeline module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_pipeline import MLPipeline


class TestMLPipeline:
    """Test cases for MLPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = MLPipeline(random_state=42)
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'Material': np.random.choice(['Al', 'Pb', 'Zn'], n_samples),
            'Material_encoded': np.random.randint(0, 3, n_samples),
            'Cement_Type_encoded': np.random.randint(0, 3, n_samples),
            'Form_Type_encoded': np.random.randint(0, 2, n_samples),
            'Stat_Measure_encoded': np.random.randint(0, 3, n_samples),
            'pH': np.random.uniform(8, 13, n_samples),
            'Time_days': np.random.uniform(0.1, 64, n_samples),
            'log_Time': np.random.uniform(0, 4, n_samples),
            'pH_squared': np.random.uniform(64, 169, n_samples),
            'Time_pH_interaction': np.random.uniform(0.8, 832, n_samples),
            'log_Release': np.random.uniform(0, 7, n_samples),
            'Cumulative_Release_mg_m2': np.random.uniform(0.1, 1000, n_samples)
        })
        
        self.feature_columns = [
            'Material_encoded', 'Cement_Type_encoded', 'Form_Type_encoded', 
            'Stat_Measure_encoded', 'pH', 'Time_days', 'log_Time', 
            'pH_squared', 'Time_pH_interaction'
        ]
        
        self.label_encoders = {
            'Material': type('MockEncoder', (), {'classes_': ['Al', 'Pb', 'Zn']})(),
            'Cement_Type': type('MockEncoder', (), {'classes_': ['CEM_I', 'CEM_II', 'CEM_III']})(),
            'Form_Type': type('MockEncoder', (), {'classes_': ['Concrete', 'Mortar']})(),
            'Stat_Measure': type('MockEncoder', (), {'classes_': ['CL_Minus', 'CL_Plus', 'Med']})()
        }
    
    def test_prepare_data(self):
        """Test data preparation for ML."""
        X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = self.pipeline.prepare_data(
            self.sample_data, self.feature_columns, self.label_encoders, test_size=0.2
        )
        
        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == len(self.sample_data)
        assert y_train.shape[0] + y_test.shape[0] == len(self.sample_data)
        assert X_train.shape[1] == len(self.feature_columns)
        
        # Check that test size is approximately correct
        assert abs(len(X_test) / len(self.sample_data) - 0.2) < 0.05
    
    def test_predict_leaching_valid_input(self):
        """Test prediction with valid inputs."""
        # Set up pipeline with mock components
        self.pipeline.label_encoders = self.label_encoders
        self.pipeline.feature_columns = self.feature_columns
        self.pipeline.best_model_name = 'XGBoost'
        self.pipeline.results = pd.DataFrame({
            'R²_score': [0.85]
        }, index=['XGBoost'])
        
        # Mock the model
        class MockModel:
            def predict(self, X):
                return np.array([np.log(50.0)])  # log of 50
        
        self.pipeline.best_model = MockModel()
        
        # Test prediction
        result = self.pipeline.predict_leaching('Al', 12.0, 1.0)
        
        assert 'error' not in result
        assert 'predicted_leaching_mg_m2' in result
        assert result['model_used'] == 'XGBoost'
    
    def test_predict_leaching_invalid_material(self):
        """Test prediction with invalid material."""
        self.pipeline.label_encoders = self.label_encoders
        
        result = self.pipeline.predict_leaching('InvalidMaterial', 12.0, 1.0)
        
        assert 'error' in result
        assert 'not supported' in result['error']
    
    def test_predict_leaching_invalid_ph(self):
        """Test prediction with invalid pH."""
        self.pipeline.label_encoders = self.label_encoders
        
        result = self.pipeline.predict_leaching('Al', 15.0, 1.0)
        
        assert 'error' in result
        assert 'pH must be between 1 and 14' in result['error']
    
    def test_predict_leaching_invalid_time(self):
        """Test prediction with invalid time."""
        self.pipeline.label_encoders = self.label_encoders
        
        result = self.pipeline.predict_leaching('Al', 12.0, 200.0)
        
        assert 'error' in result
        assert 'Time must be between 0.01 and 100 days' in result['error']
    
    def test_assess_performance(self):
        """Test performance assessment."""
        assert self.pipeline._assess_performance(0.95) == "Outstanding"
        assert self.pipeline._assess_performance(0.85) == "Excellent"
        assert self.pipeline._assess_performance(0.75) == "Good"
        assert self.pipeline._assess_performance(0.65) == "Needs Improvement"
