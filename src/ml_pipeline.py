"""
Machine learning pipeline for cement leaching prediction.

This module handles model training, evaluation, and prediction for the 
cement leaching prediction project.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


class MLPipeline:
    """Machine learning pipeline for cement leaching prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ML pipeline.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = None
        self.power_transformer = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str], 
                    label_encoders: Dict, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for machine learning.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            label_encoders: Dictionary of label encoders
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, y_train_orig, y_test_orig)
        """
        print("📊 Preparing data for machine learning...")
        
        # Store for later use
        self.feature_columns = feature_columns
        self.label_encoders = label_encoders
        
        # Prepare features and targets
        X = df[feature_columns]
        y_log = df['log_Release']
        y_original = df['Cumulative_Release_mg_m2']
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.random_state, 
            stratify=df['Material']
        )
        
        _, _, y_train_orig, y_test_orig = train_test_split(
            X, y_original, test_size=test_size, random_state=self.random_state, 
            stratify=df['Material']
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig

    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series, 
                    y_train_orig: pd.Series, y_test_orig: pd.Series) -> Tuple:
        """
        Train multiple advanced models and return best performer.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets (log scale)
            y_train_orig, y_test_orig: Training and test targets (original scale)
            
        Returns:
            Tuple of (best_model, best_model_name, results, models, pt, scaler)
        """
        print("🤖 Training advanced ML models...")

        # Power transformation and scaling
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        X_train_transformed = self.power_transformer.fit_transform(X_train)
        X_test_transformed = self.power_transformer.transform(X_test)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_transformed)
        X_test_scaled = self.scaler.transform(X_test_transformed)

        # Define optimized models
        self.models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(150, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=self.random_state
            ),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        }

        # Train and evaluate models
        results = {}
        predictions = {}

        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")

            # Use scaled data for NN and Elastic Net
            if name in ['Neural Network', 'Elastic Net']:
                model.fit(X_train_scaled, y_train)
                y_pred_log = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
            else:
                model.fit(X_train, y_train)
                y_pred_log = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

            # Transform back to original scale
            y_pred_original = np.expm1(y_pred_log)

            # Calculate metrics
            r2 = r2_score(y_test_orig, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_original))
            mae = mean_absolute_error(y_test_orig, y_pred_original)

            results[name] = {
                'R²_score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV_R²_mean': cv_scores.mean(),
                'CV_R²_std': cv_scores.std()
            }

            predictions[name] = y_pred_original

            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.2f} mg/m²")
            print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Create ensemble of top 3 models
        results_df = pd.DataFrame(results).T.sort_values('R²_score', ascending=False)
        top_3_models = results_df.head(3).index.tolist()

        print(f"\n🎯 Creating ensemble from top 3 models: {top_3_models}")

        ensemble_models = [(name, self.models[name]) for name in top_3_models]
        voting_regressor = VotingRegressor(ensemble_models)

        # Train ensemble
        if any('Neural Network' in name or 'Elastic Net' in name for name, _ in ensemble_models):
            voting_regressor.fit(X_train_scaled, y_train)
            y_pred_ensemble_log = voting_regressor.predict(X_test_scaled)
        else:
            voting_regressor.fit(X_train, y_train)
            y_pred_ensemble_log = voting_regressor.predict(X_test)

        y_pred_ensemble = np.expm1(y_pred_ensemble_log)

        # Evaluate ensemble
        r2_ensemble = r2_score(y_test_orig, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test_orig, y_pred_ensemble))
        mae_ensemble = mean_absolute_error(y_test_orig, y_pred_ensemble)

        results['Ensemble'] = {
            'R²_score': r2_ensemble,
            'RMSE': rmse_ensemble,
            'MAE': mae_ensemble,
            'CV_R²_mean': np.nan,
            'CV_R²_std': np.nan
        }

        print(f"\n🏆 ENSEMBLE RESULTS:")
        print(f"   R² Score: {r2_ensemble:.4f}")
        print(f"   RMSE: {rmse_ensemble:.2f} mg/m²")

        # Store results and best model
        self.results = pd.DataFrame(results).T.sort_values('R²_score', ascending=False)
        self.best_model_name = self.results.index[0]
        self.best_model = voting_regressor if self.best_model_name == 'Ensemble' else self.models[self.best_model_name]

        return self.best_model, self.best_model_name, self.results, self.models, self.power_transformer, self.scaler

    def predict_leaching(self, material: str, ph: float, time_days: float,
                        cement_type: str = 'Unknown', form_type: str = 'Unknown', 
                        stat_measure: str = 'Unknown') -> Dict[str, Any]:
        """
        Production-ready prediction function.
        
        Args:
            material: Material name (e.g., 'Al', 'Pb', 'Zn')
            ph: pH value (1-14)
            time_days: Time in days (0.01-100)
            cement_type: Type of cement (default: 'Unknown')
            form_type: Form type (default: 'Unknown')
            stat_measure: Statistical measure (default: 'Unknown')
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        try:
            # Input validation
            if material not in self.label_encoders['Material'].classes_:
                available_materials = list(self.label_encoders['Material'].classes_)
                return {'error': f'Material {material} not supported. Available: {available_materials}'}

            if not (1 <= ph <= 14):
                return {'error': 'pH must be between 1 and 14'}

            if not (0.01 <= time_days <= 100):
                return {'error': 'Time must be between 0.01 and 100 days'}

            # Encode categorical features with safe handling
            def safe_encode(encoder, value):
                if value in encoder.classes_:
                    return encoder.transform([value])[0]
                else:
                    # Return the first encoded value as default
                    return 0
            
            material_enc = safe_encode(self.label_encoders['Material'], material)
            cement_enc = safe_encode(self.label_encoders['Cement_Type'], cement_type)
            form_enc = safe_encode(self.label_encoders['Form_Type'], form_type)
            stat_enc = safe_encode(self.label_encoders['Stat_Measure'], stat_measure)

            # Create all features
            features = {
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
                'Time_pH_interaction': time_days * ph,
                'log_Time_pH': np.log1p(time_days) * ph,
                'Material_pH_interaction': material_enc * ph,
                'Material_Time_interaction': material_enc * time_days,
                'pH_normalized': (ph - 7.5) / 2.0,  # Approximate normalization
                'Time_normalized': (time_days - 10.0) / 15.0,  # Approximate normalization
                'Alkalinity_index': ph - 7,
                'Reactivity_score': ph * np.log1p(time_days),
                'Leaching_potential': (ph ** 2) * np.sqrt(time_days)
            }

            # Material grouping
            material_groups = {'Al': 0, 'Fe': 0, 'Si': 0, 'As': 1, 'Cr': 1, 'Mo': 1,
                              'Ba': 2, 'P': 2, 'Br': 3, 'F': 3, 'Cl': 3,
                              'Ca': 4, 'K': 4, 'Mg': 4, 'Na': 4,
                              'Cd': 5, 'Cu': 5, 'Pb': 5, 'Zn': 5, 'SO4': 6}
            features['Material_group'] = material_groups.get(material, 0)

            # Create prediction input
            X_pred = pd.DataFrame([features])[self.feature_columns]

            # Make prediction
            if self.best_model_name in ['Neural Network', 'Elastic Net']:
                X_pred_transformed = self.power_transformer.transform(X_pred)
                X_pred_scaled = self.scaler.transform(X_pred_transformed)
                pred_log = self.best_model.predict(X_pred_scaled)[0]
            else:
                pred_log = self.best_model.predict(X_pred)[0]

            # Transform back to original scale
            prediction = np.expm1(pred_log)

            # Confidence assessment
            confidence = 'High'
            if prediction > 500:
                confidence = 'Low'
            elif prediction > 100:
                confidence = 'Medium'

            return {
                'predicted_leaching_mg_m2': round(prediction, 2),
                'model_used': self.best_model_name,
                'model_r2_score': round(self.results.loc[self.best_model_name, 'R²_score'], 4),
                'confidence': confidence,
                'input_summary': f"{material} at pH {ph} for {time_days} days"
            }

        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the best model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.best_model, 'feature_importances_'):
            return pd.DataFrame({'feature': [], 'importance': []})
            
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of model performance.
        
        Returns:
            Dictionary containing model performance metrics
        """
        if self.results is None:
            return {}
            
        best_score = self.results.loc[self.best_model_name, 'R²_score']
        
        return {
            'best_model': self.best_model_name,
            'r2_score': best_score,
            'rmse': self.results.loc[self.best_model_name, 'RMSE'],
            'mae': self.results.loc[self.best_model_name, 'MAE'],
            'performance_level': self._assess_performance(best_score),
            'all_results': self.results.to_dict()
        }

    def _assess_performance(self, r2_score: float) -> str:
        """Assess model performance level based on R² score."""
        if r2_score > 0.9:
            return "Outstanding"
        elif r2_score > 0.8:
            return "Excellent"
        elif r2_score > 0.7:
            return "Good"
        else:
            return "Needs Improvement"
