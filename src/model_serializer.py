"""
Model serialization utilities for the cement leaching prediction project.

This module provides functions to save and load trained models and their components
using pickle for easy deployment and reuse.
"""

import pickle
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler


class ModelSerializer:
    """Handles saving and loading of trained models and their components."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the model serializer.
        
        Args:
            models_dir: Directory to save/load model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_components(self, model, label_encoders: Dict, 
                            power_transformer: PowerTransformer, scaler: StandardScaler,
                            feature_columns: List[str], model_name: str = 'best_model') -> Dict[str, str]:
        """
        Save all model components using pickle.
        
        Args:
            model: Trained model object
            label_encoders: Dictionary of label encoders
            power_transformer: Fitted power transformer
            scaler: Fitted scaler
            feature_columns: List of feature column names
            model_name: Name for the model files
            
        Returns:
            Dictionary with file paths of saved components
        """
        print(f"💾 Saving model components to {self.models_dir}...")
        
        saved_files = {}
        
        # Save the model
        model_path = self.models_dir / f'{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        saved_files['model'] = str(model_path)
        print(f"  ✅ Model saved: {model_path}")
        
        # Save label encoders
        encoders_path = self.models_dir / f'{model_name}_label_encoders.pkl'
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        saved_files['label_encoders'] = str(encoders_path)
        print(f"  ✅ Label encoders saved: {encoders_path}")
        
        # Save power transformer and scaler together
        transformers_path = self.models_dir / f'{model_name}_transformers.pkl'
        with open(transformers_path, 'wb') as f:
            pickle.dump((power_transformer, scaler), f)
        saved_files['transformers'] = str(transformers_path)
        print(f"  ✅ Transformers saved: {transformers_path}")
        
        # Save feature columns
        features_path = self.models_dir / f'{model_name}_feature_columns.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        saved_files['feature_columns'] = str(features_path)
        print(f"  ✅ Feature columns saved: {features_path}")
        
        # Save metadata
        metadata = {
            'model_type': type(model).__name__,
            'feature_count': len(feature_columns),
            'materials': list(label_encoders['Material'].classes_),
            'cement_types': list(label_encoders['Cement_Type'].classes_),
            'form_types': list(label_encoders['Form_Type'].classes_),
            'stat_measures': list(label_encoders['Stat_Measure'].classes_),
            'feature_columns': feature_columns
        }
        
        metadata_path = self.models_dir / f'{model_name}_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        saved_files['metadata'] = str(metadata_path)
        print(f"  ✅ Metadata saved: {metadata_path}")
        
        print(f"✅ All model components saved successfully!")
        return saved_files
    
    def load_model_components(self, model_name: str = 'best_model') -> Tuple[Any, Dict, PowerTransformer, StandardScaler, List[str], Dict]:
        """
        Load all model components from pickle files.
        
        Args:
            model_name: Name of the model files to load
            
        Returns:
            Tuple of (model, label_encoders, power_transformer, scaler, feature_columns, metadata)
        """
        print(f"📂 Loading model components from {self.models_dir}...")
        
        try:
            # Load the model
            model_path = self.models_dir / f'{model_name}.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"  ✅ Model loaded: {model_path}")
            
            # Load label encoders
            encoders_path = self.models_dir / f'{model_name}_label_encoders.pkl'
            with open(encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
            print(f"  ✅ Label encoders loaded: {encoders_path}")
            
            # Load power transformer and scaler
            transformers_path = self.models_dir / f'{model_name}_transformers.pkl'
            with open(transformers_path, 'rb') as f:
                power_transformer, scaler = pickle.load(f)
            print(f"  ✅ Transformers loaded: {transformers_path}")
            
            # Load feature columns
            features_path = self.models_dir / f'{model_name}_feature_columns.pkl'
            with open(features_path, 'rb') as f:
                feature_columns = pickle.load(f)
            print(f"  ✅ Feature columns loaded: {features_path}")
            
            # Load metadata
            metadata_path = self.models_dir / f'{model_name}_metadata.pkl'
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"  ✅ Metadata loaded: {metadata_path}")
            
            print(f"✅ All model components loaded successfully!")
            return model, label_encoders, power_transformer, scaler, feature_columns, metadata
            
        except FileNotFoundError as e:
            print(f"❌ Error loading model components: {e}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error loading model components: {e}")
            raise
    
    def list_available_models(self) -> List[str]:
        """
        List all available saved models.
        
        Returns:
            List of available model names
        """
        model_files = list(self.models_dir.glob('*.pkl'))
        model_names = set()
        
        for file_path in model_files:
            name = file_path.stem
            if '_' in name:
                base_name = name.split('_')[0]
                model_names.add(base_name)
            else:
                model_names.add(name)
        
        return sorted(list(model_names))
    
    def get_model_info(self, model_name: str = 'best_model') -> Dict[str, Any]:
        """
        Get information about a saved model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        metadata_path = self.models_dir / f'{model_name}_metadata.pkl'
        
        if not metadata_path.exists():
            return {'error': f'Model {model_name} not found'}
        
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        except Exception as e:
            return {'error': f'Error loading model info: {e}'}
    
    def delete_model(self, model_name: str = 'best_model') -> bool:
        """
        Delete a saved model and all its components.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            files_to_delete = [
                f'{model_name}.pkl',
                f'{model_name}_label_encoders.pkl',
                f'{model_name}_transformers.pkl',
                f'{model_name}_feature_columns.pkl',
                f'{model_name}_metadata.pkl'
            ]
            
            deleted_count = 0
            for filename in files_to_delete:
                file_path = self.models_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
                    print(f"  ✅ Deleted: {filename}")
            
            if deleted_count > 0:
                print(f"✅ Deleted model '{model_name}' ({deleted_count} files)")
                return True
            else:
                print(f"⚠️  No files found for model '{model_name}'")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting model: {e}")
            return False


def save_model_quick(model, label_encoders: Dict, power_transformer: PowerTransformer, 
                    scaler: StandardScaler, feature_columns: List[str], 
                    models_dir: str = 'models') -> Dict[str, str]:
    """
    Quick function to save model components (backward compatibility).
    
    Args:
        model: Trained model object
        label_encoders: Dictionary of label encoders
        power_transformer: Fitted power transformer
        scaler: Fitted scaler
        feature_columns: List of feature column names
        models_dir: Directory to save files
        
    Returns:
        Dictionary with file paths of saved components
    """
    serializer = ModelSerializer(models_dir)
    return serializer.save_model_components(
        model, label_encoders, power_transformer, scaler, feature_columns
    )


def load_model_quick(models_dir: str = 'models') -> Tuple[Any, Dict, PowerTransformer, StandardScaler, List[str]]:
    """
    Quick function to load model components (backward compatibility).
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (model, label_encoders, power_transformer, scaler, feature_columns)
    """
    serializer = ModelSerializer(models_dir)
    model, label_encoders, power_transformer, scaler, feature_columns, _ = serializer.load_model_components()
    return model, label_encoders, power_transformer, scaler, feature_columns


# Example usage
if __name__ == "__main__":
    # This would be used after training a model
    print("Model Serializer Example")
    print("=" * 30)
    
    # Example of how to use after training
    example_code = '''
    # After training your model:
    from model_serializer import ModelSerializer
    
    serializer = ModelSerializer('models')
    
    # Save model
    saved_files = serializer.save_model_components(
        model=your_trained_model,
        label_encoders=label_encoders,
        power_transformer=power_transformer,
        scaler=scaler,
        feature_columns=feature_columns,
        model_name='production_model'
    )
    
    # Load model later
    model, encoders, pt, scaler, features, metadata = serializer.load_model_components('production_model')
    
    # List available models
    available_models = serializer.list_available_models()
    print(f"Available models: {available_models}")
    '''
    
    print("Example usage:")
    print(example_code)
