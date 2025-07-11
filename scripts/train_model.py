#!/usr/bin/env python3
"""
Main training script for telecoms customer churn prediction.

This script orchestrates the complete ML pipeline:
1. Generate or load sample data
2. Clean and preprocess the data
3. Train multiple models
4. Evaluate and select the best model
5. Save the final model as model.pkl
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telecoms_churn_ml.data import DataProcessor, SampleDataGenerator
from telecoms_churn_ml.models import ChurnPredictor


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=log_level
    )


def generate_sample_data(output_dir: str, n_customers: int = 2000) -> None:
    """Generate sample telecoms data."""
    logger.info(f"Generating sample data for {n_customers} customers")
    
    generator = SampleDataGenerator(seed=42)
    generator.save_sample_data(
        output_dir=output_dir,
        n_customers=n_customers,
        churn_rate=0.26,
        format='csv'
    )
    
    logger.info(f"Sample data saved to {output_dir}")


def load_and_process_data(data_dir: str) -> pd.DataFrame:
    """Load and process the raw data."""
    logger.info(f"Loading data from {data_dir}")
    
    processor = DataProcessor()
    
    # Load and clean data
    unified_data = processor.clean_and_unify_data(data_dir)
    
    # Save processed data
    processed_dir = Path(data_dir).parent / "processed"
    processor.save_processed_data(
        unified_data, 
        processed_dir / "unified_telecoms_data.csv"
    )
    
    # Log data summary
    summary = processor.get_data_summary(unified_data)
    logger.info(f"Processed data shape: {summary['shape']}")
    logger.info(f"Columns: {summary['columns']}")
    
    return unified_data


def train_model(data: pd.DataFrame, model_dir: str) -> dict:
    """Train the churn prediction model."""
    logger.info("Training churn prediction model")
    
    # Initialize predictor
    predictor = ChurnPredictor(model_type='auto')
    
    # Train model
    results = predictor.train(
        data, 
        target_column='churn',
        test_size=0.2,
        tune_hyperparameters=True
    )
    
    # Save model
    model_path = Path(model_dir) / "model.pkl"
    predictor.save_model(model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Best model: {results['model_type']}")
    logger.info(f"ROC-AUC Score: {results['roc_auc']:.4f}")
    
    return results


def print_results(results: dict) -> None:
    """Print training results."""
    print("\n" + "="*60)
    print("🎯 TELECOMS CUSTOMER CHURN PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n📊 Model Performance:")
    print(f"   • Best Model: {results['model_type'].upper()}")
    print(f"   • ROC-AUC Score: {results['roc_auc']:.4f}")
    print(f"   • Training Samples: {results['train_size']:,}")
    print(f"   • Test Samples: {results['test_size']:,}")
    
    # Classification metrics
    cr = results['classification_report']
    if '1' in cr:  # Churn class
        precision = cr['1']['precision']
        recall = cr['1']['recall']
        f1 = cr['1']['f1-score']
        
        print(f"\n📈 Churn Detection Metrics:")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall: {recall:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
    
    # Feature importance
    if results.get('feature_importance'):
        print(f"\n🔍 Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(
            list(results['feature_importance'].items())[:5], 1
        ):
            print(f"   {i}. {feature}: {importance:.4f}")
    
    print(f"\n✅ Model successfully saved as 'model.pkl'")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train telecoms customer churn prediction model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample data if no data files found"
    )
    parser.add_argument(
        "--n-customers",
        type=int,
        default=2000,
        help="Number of customers for sample data generation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger.info("🚀 Starting telecoms customer churn prediction pipeline")
    
    try:
        # Check if data exists
        data_dir = Path(args.data_dir)
        data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx"))
        
        if not data_files and args.generate_sample:
            logger.info("No data files found. Generating sample data...")
            generate_sample_data(str(data_dir), args.n_customers)
        elif not data_files:
            logger.error(f"No data files found in {data_dir}")
            logger.info("Use --generate-sample to create sample data")
            return 1
        
        # Load and process data
        processed_data = load_and_process_data(str(data_dir))
        
        # Check if target column exists
        if 'churn' not in processed_data.columns:
            logger.error("Target column 'churn' not found in data")
            logger.info("Available columns: " + ", ".join(processed_data.columns))
            return 1
        
        # Train model
        results = train_model(processed_data, args.model_dir)
        
        # Print results
        print_results(results)
        
        logger.info("✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
