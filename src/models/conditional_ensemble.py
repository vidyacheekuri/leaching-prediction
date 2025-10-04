"""
Conditional ensemble model for improved prediction accuracy.

This module implements a conditional ensemble approach that uses specialized
models for problematic materials (Al, Br) and the global model for others.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb


class ConditionalEnsemble:
    """
    Conditional ensemble that selects appropriate model based on material type.
    
    Uses specialized models for materials with high prediction errors (Al, Br)
    and the global model for all other materials.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the conditional ensemble.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.global_model = None
        self.specialized_models = {}
        self.transformers = {}
        self.feature_columns = None
        self.label_encoders = None
        self.problematic_materials = ['Al', 'Br']  # Materials with high errors
        
    def train_specialized_models(self, df: pd.DataFrame, feature_columns: List[str],
                                label_encoders: Dict) -> Dict[str, Any]:
        """
        Train specialized models for problematic materials.
        
        Args:
            df: Full dataset
            feature_columns: List of feature column names
            label_encoders: Dictionary of label encoders
            
        Returns:
            Dictionary containing training results for each specialized model
        """
        print("🔧 Training specialized models for problematic materials...")
        
        self.feature_columns = feature_columns
        self.label_encoders = label_encoders
        
        results = {}
        
        for material in self.problematic_materials:
            print(f"\n🔄 Training specialized model for {material}...")
            
            # Filter data for this material
            df_material = df[df['Material'] == material].copy()
            
            if len(df_material) < 10:  # Need minimum samples
                print(f"   Insufficient data for {material}: {len(df_material)} samples")
                continue
                
            # Prepare features and target
            X_material = df_material[feature_columns]
            y_material = np.log1p(df_material['Cumulative_Release_mg_m2'])
            
            # Split data
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_material, y_material, test_size=0.2, random_state=self.random_state
            )
            
            # Preprocessing
            pt_material = PowerTransformer(method='yeo-johnson')
            scaler_material = StandardScaler()
            
            X_tr_transformed = pt_material.fit_transform(X_tr)
            X_tr_scaled = scaler_material.fit_transform(X_tr_transformed)
            X_te_transformed = pt_material.transform(X_te)
            X_te_scaled = scaler_material.transform(X_te_transformed)
            
            # Train model
            model_material = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            )
            
            model_material.fit(X_tr_scaled, y_tr)
            
            # Evaluate
            y_pred_log = model_material.predict(X_te_scaled)
            y_pred_orig = np.expm1(y_pred_log)
            y_true_orig = np.expm1(y_te)
            
            r2 = r2_score(y_true_orig, y_pred_orig)
            mae = mean_absolute_error(y_true_orig, y_pred_orig)
            
            # Store model and transformers
            self.specialized_models[material] = model_material
            self.transformers[material] = {
                'power_transformer': pt_material,
                'scaler': scaler_material
            }
            
            results[material] = {
                'r2_score': r2,
                'mae': mae,
                'sample_count': len(df_material),
                'test_samples': len(y_te)
            }
            
            print(f"   R² Score: {r2:.4f}")
            print(f"   MAE: {mae:.2f} mg/m²")
            print(f"   Samples: {len(df_material)}")
        
        return results
    
    def set_global_model(self, model, transformer, scaler):
        """
        Set the global model and its transformers.
        
        Args:
            model: Trained global model
            transformer: Fitted power transformer
            scaler: Fitted scaler
        """
        self.global_model = model
        self.transformers['global'] = {
            'power_transformer': transformer,
            'scaler': scaler
        }
    
    def predict(self, X: pd.DataFrame, materials: pd.Series) -> np.ndarray:
        """
        Make predictions using conditional ensemble.
        
        Args:
            X: Feature matrix
            materials: Series containing material names for each sample
            
        Returns:
            Array of predictions
        """
        predictions = np.zeros(len(X))
        
        for i, material in enumerate(materials):
            if material in self.problematic_materials and material in self.specialized_models:
                # Use specialized model
                model = self.specialized_models[material]
                transformer = self.transformers[material]['power_transformer']
                scaler = self.transformers[material]['scaler']
                
                # Transform features
                X_sample = X.iloc[[i]]
                X_transformed = transformer.transform(X_sample)
                X_scaled = scaler.transform(X_transformed)
                
                # Predict
                pred_log = model.predict(X_scaled)[0]
                predictions[i] = np.expm1(pred_log)
                
            else:
                # Use global model
                model = self.global_model
                transformer = self.transformers['global']['power_transformer']
                scaler = self.transformers['global']['scaler']
                
                # Transform features
                X_sample = X.iloc[[i]]
                X_transformed = transformer.transform(X_sample)
                X_scaled = scaler.transform(X_transformed)
                
                # Predict
                pred_log = model.predict(X_scaled)[0]
                predictions[i] = np.expm1(pred_log)
        
        return predictions
    
    def predict_single(self, material: str, features_dict: Dict[str, float]) -> float:
        """
        Predict leaching for a single sample.
        
        Args:
            material: Material name
            features_dict: Dictionary of feature values
            
        Returns:
            Predicted leaching value
        """
        # Create DataFrame from features
        X_pred = pd.DataFrame([features_dict])[self.feature_columns]
        
        if material in self.problematic_materials and material in self.specialized_models:
            # Use specialized model
            model = self.specialized_models[material]
            transformer = self.transformers[material]['power_transformer']
            scaler = self.transformers[material]['scaler']
        else:
            # Use global model
            model = self.global_model
            transformer = self.transformers['global']['power_transformer']
            scaler = self.transformers['global']['scaler']
        
        # Transform and predict
        X_transformed = transformer.transform(X_pred)
        X_scaled = scaler.transform(X_transformed)
        pred_log = model.predict(X_scaled)[0]
        
        return np.expm1(pred_log)
    
    def evaluate_improvement(self, df_test: pd.DataFrame, y_true: np.ndarray, 
                           y_pred_global: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate improvement of conditional ensemble over global model.
        
        Args:
            df_test: Test DataFrame with material information
            y_true: True values
            y_pred_global: Global model predictions
            
        Returns:
            Dictionary containing improvement metrics
        """
        # Get conditional ensemble predictions
        y_pred_conditional = self.predict(df_test[self.feature_columns], df_test['Material'])
        
        # Calculate errors
        abs_error_global = np.abs(y_pred_global - y_true)
        abs_error_conditional = np.abs(y_pred_conditional - y_true)
        pct_error_global = np.where(y_true != 0, abs_error_global / y_true * 100, np.nan)
        pct_error_conditional = np.where(y_true != 0, abs_error_conditional / y_true * 100, np.nan)
        
        # Overall improvement
        improvement = {
            'global_model': {
                'rmse': np.sqrt(np.mean((y_pred_global - y_true) ** 2)),
                'mae': np.mean(abs_error_global),
                'median_ape': np.nanmedian(pct_error_global),
                'high_error_samples': np.sum(pct_error_global > 10)
            },
            'conditional_ensemble': {
                'rmse': np.sqrt(np.mean((y_pred_conditional - y_true) ** 2)),
                'mae': np.mean(abs_error_conditional),
                'median_ape': np.nanmedian(pct_error_conditional),
                'high_error_samples': np.sum(pct_error_conditional > 10)
            }
        }
        
        # Calculate improvement percentages
        improvement['improvement'] = {
            'rmse_reduction': (improvement['global_model']['rmse'] - improvement['conditional_ensemble']['rmse']) / improvement['global_model']['rmse'] * 100,
            'mae_reduction': (improvement['global_model']['mae'] - improvement['conditional_ensemble']['mae']) / improvement['global_model']['mae'] * 100,
            'high_error_reduction': improvement['global_model']['high_error_samples'] - improvement['conditional_ensemble']['high_error_samples']
        }
        
        # Material-specific improvement
        material_improvement = {}
        for material in self.problematic_materials:
            mask = df_test['Material'] == material
            if np.sum(mask) > 0:
                material_improvement[material] = {
                    'global_mae': np.mean(abs_error_global[mask]),
                    'conditional_mae': np.mean(abs_error_conditional[mask]),
                    'improvement': (np.mean(abs_error_global[mask]) - np.mean(abs_error_conditional[mask])) / np.mean(abs_error_global[mask]) * 100,
                    'sample_count': np.sum(mask)
                }
        
        improvement['material_specific'] = material_improvement
        
        return improvement
