"""Main predictor class for customer churn prediction."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from loguru import logger

# Optional XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will be disabled.")


class ChurnPredictor:
    """
    Main predictor class for customer churn prediction.
    """
    
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize the ChurnPredictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'xgboost', 'logistic', 'svm', 'auto')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'churn'
        self.is_trained = False
        logger.info(f"ChurnPredictor initialized with model_type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            df: Input DataFrame
            fit_encoders: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            Prepared DataFrame with encoded features
        """
        df_prepared = df.copy()
        
        # Remove customer_id if present (non-predictive)
        if 'customer_id' in df_prepared.columns:
            df_prepared = df_prepared.drop(columns=['customer_id'])
        
        # Handle datetime columns by converting to numeric features
        datetime_columns = df_prepared.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            # Convert to timestamp
            df_prepared[f"{col}_timestamp"] = pd.to_datetime(df_prepared[col]).astype(int) / 10**9
            df_prepared = df_prepared.drop(columns=[col])
            logger.info(f"Converted datetime column {col} to timestamp")
        
        # Handle mixed-type object columns that might contain dates
        object_columns = df_prepared.select_dtypes(include=['object']).columns
        for col in object_columns:
            if col == self.target_column:
                continue
            
            # Try to convert to datetime first, but be more restrictive
            try:
                # Only attempt datetime conversion if values look like dates
                sample_values = df_prepared[col].dropna().head(10).astype(str)
                if any(any(char.isdigit() and ('-' in val or '/' in val or ':' in val)) for val in sample_values for char in val):
                    datetime_series = pd.to_datetime(df_prepared[col], errors='coerce')
                    if datetime_series.notna().sum() > len(df_prepared) * 0.8:  # If more than 80% are valid dates
                        df_prepared[f"{col}_timestamp"] = datetime_series.astype(int) / 10**9
                        df_prepared = df_prepared.drop(columns=[col])
                        logger.info(f"Converted object column {col} to datetime timestamp")
                        continue
            except:
                pass
        
        # Handle categorical columns
        categorical_columns = df_prepared.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != self.target_column]
        
        for col in categorical_columns:
            if fit_encoders:
                # Fit new encoder
                le = LabelEncoder()
                # Handle any missing values
                df_prepared[col] = df_prepared[col].fillna('Unknown')
                df_prepared[col] = le.fit_transform(df_prepared[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    df_prepared[col] = df_prepared[col].fillna('Unknown')
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df_prepared[col] = df_prepared[col].astype(str)
                    
                    # Map unseen categories to 'Unknown' if it exists, otherwise to the first class
                    mask = ~df_prepared[col].isin(le.classes_)
                    if 'Unknown' in le.classes_:
                        df_prepared.loc[mask, col] = 'Unknown'
                    else:
                        df_prepared.loc[mask, col] = le.classes_[0]
                    
                    df_prepared[col] = le.transform(df_prepared[col])
                else:
                    # If encoder doesn't exist, fill with 0
                    df_prepared[col] = 0
        
        # Handle boolean columns
        bool_columns = df_prepared.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df_prepared[col] = df_prepared[col].astype(int)
        
        # Ensure all remaining columns are numeric
        for col in df_prepared.columns:
            if col == self.target_column:
                continue
            try:
                df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce').fillna(0)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {e}")
                # Drop problematic columns
                df_prepared = df_prepared.drop(columns=[col])
        
        # Final check: ensure all feature columns are numeric
        feature_columns = [col for col in df_prepared.columns if col != self.target_column]
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(df_prepared[col]):
                logger.warning(f"Dropping non-numeric column: {col}")
                df_prepared = df_prepared.drop(columns=[col])
        
        logger.info(f"Prepared features: {df_prepared.shape[1]-1} columns")
        return df_prepared
    
    def _get_model(self, model_type: str) -> Any:
        """Get model instance based on type."""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, falling back to RandomForest")
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif model_type == 'logistic':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, df: pd.DataFrame, target_column: str = 'churn', 
              test_size: float = 0.2, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the churn prediction model.
        
        Args:
            df: Training DataFrame
            target_column: Name of target column
            test_size: Size of test set
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training churn prediction model on {len(df)} samples")
        
        self.target_column = target_column
        
        # Prepare features
        df_prepared = self.prepare_features(df, fit_encoders=True)
        
        # Separate features and target
        if target_column not in df_prepared.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        X = df_prepared.drop(columns=[target_column])
        y = df_prepared[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model(s)
        if self.model_type == 'auto':
            # Train multiple models and select the best
            model_results = self._train_multiple_models(
                X_train_scaled, X_test_scaled, y_train, y_test, tune_hyperparameters
            )
            best_model_type = max(model_results.keys(), 
                                 key=lambda k: model_results[k]['roc_auc'])
            self.model = model_results[best_model_type]['model']
            self.model_type = best_model_type
            logger.info(f"Best model selected: {best_model_type}")
        else:
            # Train single model
            self.model = self._get_model(self.model_type)
            
            if tune_hyperparameters:
                self.model = self._tune_hyperparameters(
                    self.model, X_train_scaled, y_train
                )
            
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        results = {
            'model_type': self.model_type,
            'roc_auc': roc_auc,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self._get_feature_importance(),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        logger.info(f"Model training completed. ROC-AUC: {roc_auc:.4f}")
        return results
    
    def _train_multiple_models(self, X_train, X_test, y_train, y_test, 
                              tune_hyperparameters: bool) -> Dict[str, Dict]:
        """Train multiple models and compare performance."""
        model_types = ['random_forest', 'logistic', 'svm']
        if XGBOOST_AVAILABLE:
            model_types.append('xgboost')
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model")
            
            try:
                model = self._get_model(model_type)
                
                if tune_hyperparameters and model_type in ['random_forest', 'xgboost']:
                    model = self._tune_hyperparameters(model, X_train, y_train)
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                results[model_type] = {
                    'model': model,
                    'roc_auc': roc_auc
                }
                
                logger.info(f"{model_type} ROC-AUC: {roc_auc:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue
        
        return results
    
    def _tune_hyperparameters(self, model, X_train, y_train):
        """Perform hyperparameter tuning."""
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'XGBClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        model_name = type(model).__name__
        if model_name in param_grids:
            logger.info(f"Tuning hyperparameters for {model_name}")
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        return model
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained or self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = abs(self.model.coef_[0])
        else:
            return {}
        
        feature_importance = dict(zip(self.feature_columns, importances))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make churn predictions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        df_prepared = self.prepare_features(df, fit_encoders=False)
        
        # Select only the features used in training
        X = df_prepared[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get churn prediction probabilities.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of probabilities for churn (class 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        df_prepared = self.prepare_features(df, fit_encoders=False)
        
        # Select only the features used in training
        X = df_prepared[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.target_column = model_data.get('target_column', 'churn')
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"error": "No trained model available"}
        
        info = {
            'model_type': self.model_type,
            'target_column': self.target_column,
            'feature_count': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'label_encoders': list(self.label_encoders.keys()),
            'is_trained': self.is_trained
        }
        
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        return info
