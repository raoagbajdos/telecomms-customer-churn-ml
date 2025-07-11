"""Model evaluation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score
)
from loguru import logger

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Plotting functions will be disabled.")


class ModelEvaluator:
    """
    Model evaluation and visualization utilities.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.results = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: Optional[np.ndarray] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'precision': classification_rep['1']['precision'],
            'recall': classification_rep['1']['recall'],
            'f1_score': classification_rep['1']['f1-score']
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            results['roc_auc'] = roc_auc
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        # Store results
        self.results[model_name] = results
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1-Score: {results['f1_score']:.4f}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare multiple evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No models to compare")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            if 'roc_auc' in results:
                row['ROC-AUC'] = results['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.round(4)
    
    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None):
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Cannot plot confusion matrix.")
            return
            
        if model_name not in self.results:
            logger.error(f"Model {model_name} not found in results")
            return
        
        conf_matrix = np.array(self.results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, model_name: str) -> str:
        """
        Generate a text report for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted text report
        """
        if model_name not in self.results:
            return f"Model {model_name} not found in results"
        
        results = self.results[model_name]
        
        report = f"""
Model Evaluation Report: {model_name}
{'=' * 50}

Overall Metrics:
- Accuracy: {results['accuracy']:.4f}
- Precision: {results['precision']:.4f}
- Recall: {results['recall']:.4f}
- F1-Score: {results['f1_score']:.4f}
"""
        
        if 'roc_auc' in results:
            report += f"- ROC-AUC: {results['roc_auc']:.4f}\n"
        
        report += f"""
Confusion Matrix:
{np.array(results['confusion_matrix'])}

Classification Report:
{pd.DataFrame(results['classification_report']).round(4).to_string()}
"""
        
        return report
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()
        logger.info("Evaluation results cleared")
