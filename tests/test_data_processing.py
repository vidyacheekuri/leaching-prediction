"""
Tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataProcessor("test_file.xlsx")
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Material': ['Al', 'Al', 'Pb', 'Pb'],
            'Material_Condition': [
                'CEM I Cement concrete Monolith-[CL-]',
                'CEM II Cement mortar Monolith-[Med]',
                'CEM I Cement concrete Monolith-[CL+]',
                'CEM III Sewage sludge Monolith-[LPx]'
            ],
            'pH': [12.0, 11.5, 10.8, 9.2],
            'Time_days': [1.0, 4.0, 9.0, 16.0],
            'Cumulative_Release_mg_m2': [25.5, 45.2, 78.9, 123.4]
        })
    
    def test_clean_material_condition(self):
        """Test material condition cleaning function."""
        # Test CEM I extraction
        cement_type, form_type, stat_measure = self.processor.clean_material_condition(
            'CEM I Cement concrete Monolith-[CL-]'
        )
        assert cement_type == 'CEM_I'
        assert form_type == 'Concrete'
        assert stat_measure == 'CL_Minus'
        
        # Test CEM II extraction
        cement_type, form_type, stat_measure = self.processor.clean_material_condition(
            'CEM II Cement mortar Monolith-[Med]'
        )
        assert cement_type == 'CEM_II'
        assert form_type == 'Mortar'
        assert stat_measure == 'Med'
        
        # Test unknown values
        cement_type, form_type, stat_measure = self.processor.clean_material_condition(
            'Unknown condition'
        )
        assert cement_type == 'Unknown'
        assert form_type == 'Unknown'
        assert stat_measure == 'Unknown'
    
    def test_create_features(self):
        """Test feature creation."""
        df, feature_columns, label_encoders = self.processor.create_features(self.sample_data)
        
        # Check that encoded columns were created
        assert 'Material_encoded' in df.columns
        assert 'Cement_Type_encoded' in df.columns
        assert 'Form_Type_encoded' in df.columns
        assert 'Stat_Measure_encoded' in df.columns
        
        # Check that engineered features were created
        assert 'log_Time' in df.columns
        assert 'pH_squared' in df.columns
        assert 'Time_pH_interaction' in df.columns
        
        # Check that label encoders were created
        assert 'Material' in label_encoders
        assert 'Cement_Type' in label_encoders
        assert 'Form_Type' in label_encoders
        assert 'Stat_Measure' in label_encoders
        
        # Check feature columns list
        assert len(feature_columns) > 0
        assert 'Material_encoded' in feature_columns
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        summary = self.processor.get_data_summary(self.sample_data)
        
        assert summary['total_samples'] == 4
        assert summary['num_materials'] == 2
        assert summary['materials'] == ['Al', 'Pb']
        assert 'ph_range' in summary
        assert 'time_range' in summary
        assert 'leaching_range' in summary
        assert 'material_distribution' in summary
    
    def test_feature_engineering_values(self):
        """Test that engineered features have correct values."""
        df, _, _ = self.processor.create_features(self.sample_data)
        
        # Test log transformation
        expected_log_time = np.log1p(1.0)
        assert np.isclose(df.loc[0, 'log_Time'], expected_log_time)
        
        # Test pH squared
        expected_pH_squared = 12.0 ** 2
        assert df.loc[0, 'pH_squared'] == expected_pH_squared
        
        # Test interaction
        expected_interaction = 1.0 * 12.0
        assert df.loc[0, 'Time_pH_interaction'] == expected_interaction
