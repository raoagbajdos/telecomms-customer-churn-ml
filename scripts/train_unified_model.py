#!/usr/bin/env python3
"""
Complete model training pipeline from generated multi-table data.
This script will:
1. Unify the multi-table data into a single dataset
2. Process and clean the data
3. Train a machine learning model
4. Save the trained model as a pickle file
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telecoms_churn_ml.data.processor import DataProcessor
from telecoms_churn_ml.models.predictor import ChurnPredictor


def main():
    """Main training pipeline."""
    logger.info("Starting complete model training pipeline")
    
    # Define paths
    data_dir = project_root / "data" / "generated"
    model_output_path = project_root / "models" / "model.pkl"
    processed_data_path = project_root / "data" / "processed" / "unified_data.csv"
    
    # Ensure output directories exist
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Unify the multi-table data
        logger.info("Step 1: Unifying multi-table data")
        processor = DataProcessor()
        unified_df = processor.unify_generated_data(data_dir)
        
        # Save the unified dataset
        unified_df.to_csv(processed_data_path, index=False)
        logger.info(f"Unified dataset saved to {processed_data_path}")
        
        # Step 2: Check data quality
        logger.info("Step 2: Checking data quality")
        logger.info(f"Dataset shape: {unified_df.shape}")
        
        # Find the churn column (might be prefixed)
        churn_column = None
        possible_churn_cols = [col for col in unified_df.columns if 'churn' in col.lower()]
        if possible_churn_cols:
            churn_column = possible_churn_cols[0]  # Use the first churn column found
            logger.info(f"Using target column: {churn_column}")
            logger.info("Target variable distribution:")
            print(unified_df[churn_column].value_counts())
        else:
            logger.error("No churn column found in dataset")
            return
        
        # Step 3: Process the data for ML
        logger.info("Step 3: Processing data for machine learning")
        processed_df = processor.clean_dataframe(unified_df, "unified_dataset")
        
        # Step 4: Train the model
        logger.info("Step 4: Training machine learning model")
        predictor = ChurnPredictor(model_type='auto')  # Use auto to select best model
        
        training_results = predictor.train(
            df=processed_df,
            target_column=churn_column,
            test_size=0.2,
            tune_hyperparameters=True
        )
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Model type selected: {predictor.model_type}")
        logger.info(f"Training accuracy: {training_results.get('accuracy', 'N/A')}")
        logger.info(f"ROC-AUC score: {training_results.get('roc_auc', 'N/A')}")
        logger.info(f"F1 score: {training_results.get('f1_score', 'N/A')}")
        
        # Step 5: Save the trained model
        logger.info("Step 5: Saving trained model")
        predictor.save_model(str(model_output_path))
        logger.info(f"Model saved to {model_output_path}")
        
        # Step 6: Generate model summary
        model_info = predictor.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            if key != 'feature_columns':  # Skip long feature list
                logger.info(f"  {key}: {value}")
        
        logger.info(f"Model uses {len(model_info.get('feature_columns', []))} features")
        
        # Step 7: Test loading the saved model
        logger.info("Step 7: Testing model loading")
        test_predictor = ChurnPredictor()
        test_predictor.load_model(str(model_output_path))
        logger.info("Model loading test successful!")
        
        logger.success("Complete pipeline executed successfully!")
        logger.info(f"Trained model available at: {model_output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
