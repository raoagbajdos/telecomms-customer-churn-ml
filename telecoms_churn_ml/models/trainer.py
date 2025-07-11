"""Model training utilities and orchestration."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from .predictor import ChurnPredictor
from .evaluator import ModelEvaluator
from loguru import logger


class ModelTrainer:
    """
    High-level model training orchestration.
    """
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.predictor = None
        self.evaluator = ModelEvaluator()
        logger.info("ModelTrainer initialized")
    
    def train_and_evaluate(self, data_path: str, target_column: str = 'churn',
                          model_type: str = 'auto', test_size: float = 0.2,
                          tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            data_path: Path to the training data CSV
            target_column: Name of the target column
            model_type: Type of model to train
            test_size: Fraction of data to use for testing
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Training and evaluation results
        """
        logger.info(f"Starting training pipeline with data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Initialize predictor
        self.predictor = ChurnPredictor(model_type=model_type)
        
        # Train model
        training_results = self.predictor.train(
            df=df,
            target_column=target_column,
            test_size=test_size,
            tune_hyperparameters=tune_hyperparameters
        )
        
        logger.info("Training completed successfully")
        
        return training_results
    
    def evaluate_on_holdout(self, holdout_data_path: str) -> Dict[str, Any]:
        """
        Evaluate trained model on holdout data.
        
        Args:
            holdout_data_path: Path to holdout data CSV
            
        Returns:
            Evaluation results
        """
        if self.predictor is None or not self.predictor.is_trained:
            raise ValueError("No trained model available. Train a model first.")
        
        logger.info(f"Evaluating on holdout data from {holdout_data_path}")
        
        # Load holdout data
        holdout_df = pd.read_csv(holdout_data_path)
        
        # Get predictions
        predictions = self.predictor.predict(holdout_df)
        probabilities = self.predictor.predict_proba(holdout_df)
        
        # Extract true labels
        y_true = holdout_df[self.predictor.target_column].values
        
        # Evaluate
        evaluation_results = self.evaluator.evaluate_model(
            y_true=y_true,
            y_pred=predictions,
            y_pred_proba=probabilities[:, 1] if probabilities is not None else None,
            model_name=f"{self.predictor.model_type}_holdout"
        )
        
        return evaluation_results
    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.predictor is None or not self.predictor.is_trained:
            raise ValueError("No trained model available. Train a model first.")
        
        self.predictor.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        self.predictor = ChurnPredictor()
        self.predictor.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        if self.predictor is None:
            return {"error": "No model available"}
        
        return self.predictor.get_model_info()
    
    def generate_training_report(self) -> str:
        """
        Generate a comprehensive training report.
        
        Returns:
            Formatted training report
        """
        if self.predictor is None:
            return "No model available for reporting"
        
        model_info = self.predictor.get_model_info()
        
        report = f"""
Training Report
{'=' * 50}

Model Information:
- Model Type: {model_info.get('model_type', 'Unknown')}
- Training Status: {'Trained' if model_info.get('is_trained', False) else 'Not Trained'}
- Target Column: {model_info.get('target_column', 'Unknown')}
- Feature Count: {model_info.get('feature_count', 0)}

Features Used:
{', '.join(model_info.get('feature_columns', []))}

Label Encoders Applied:
{', '.join(model_info.get('label_encoders', []))}
"""
        
        if 'n_estimators' in model_info:
            report += f"\nModel Parameters:\n- Number of Estimators: {model_info['n_estimators']}"
        
        return report
