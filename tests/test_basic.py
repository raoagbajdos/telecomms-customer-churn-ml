"""Basic tests for the telecoms churn ML package."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_import_data_processor(self):
        """Test that DataProcessor can be imported."""
        from telecoms_churn_ml.data import DataProcessor
        processor = DataProcessor()
        assert processor is not None
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        from telecoms_churn_ml.data import DataProcessor
        
        processor = DataProcessor()
        df = pd.DataFrame({
            'Customer ID': [1, 2, 3],
            'Total Charges': [100, 200, 300],
            'Monthly-Charges': [50, 60, 70]
        })
        
        cleaned_df = processor._clean_column_names(df)
        
        expected_columns = ['customer_id', 'total_charges', 'monthly_charges']
        assert list(cleaned_df.columns) == expected_columns
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        from telecoms_churn_ml.data import DataProcessor
        
        processor = DataProcessor()
        df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', np.nan, 'A', 'B']
        })
        
        cleaned_df = processor._handle_missing_values(df)
        
        # Should have no missing values
        assert cleaned_df.isnull().sum().sum() == 0


class TestSampleDataGenerator:
    """Test sample data generation."""
    
    def test_import_sample_generator(self):
        """Test that SampleDataGenerator can be imported."""
        from telecoms_churn_ml.data import SampleDataGenerator
        generator = SampleDataGenerator(seed=42)
        assert generator is not None
    
    def test_generate_customer_data(self):
        """Test customer data generation."""
        from telecoms_churn_ml.data import SampleDataGenerator
        
        generator = SampleDataGenerator(seed=42)
        customer_data = generator.generate_customer_data(100)
        
        assert len(customer_data) == 100
        assert 'customer_id' in customer_data.columns
        assert 'gender' in customer_data.columns
        assert customer_data['customer_id'].nunique() == 100
    
    def test_generate_complete_dataset(self):
        """Test complete dataset generation."""
        from telecoms_churn_ml.data import SampleDataGenerator
        
        generator = SampleDataGenerator(seed=42)
        dataset = generator.generate_complete_dataset(50, churn_rate=0.3)
        
        assert 'customers' in dataset
        assert 'billing' in dataset
        assert 'usage' in dataset
        
        # Check data consistency
        assert len(dataset['customers']) == 50
        assert len(dataset['billing']) == 50
        assert len(dataset['usage']) == 50


class TestChurnPredictor:
    """Test churn prediction functionality."""
    
    def test_import_churn_predictor(self):
        """Test that ChurnPredictor can be imported."""
        from telecoms_churn_ml.models import ChurnPredictor
        predictor = ChurnPredictor()
        assert predictor is not None
    
    def test_prepare_features(self):
        """Test feature preparation."""
        from telecoms_churn_ml.models import ChurnPredictor
        
        predictor = ChurnPredictor()
        df = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'gender': ['Male', 'Female', 'Male'],
            'tenure': [12, 24, 6],
            'monthly_charges': [50.0, 75.0, 30.0],
            'churn': [0, 1, 0]
        })
        
        prepared_df = predictor.prepare_features(df, fit_encoders=True)
        
        # Check that categorical features are encoded
        assert prepared_df['gender'].dtype in [np.int64, np.int32]
        assert 'customer_id' in prepared_df.columns


class TestModelTraining:
    """Test model training with sample data."""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        from telecoms_churn_ml.data import SampleDataGenerator
        from telecoms_churn_ml.models import ChurnPredictor
        
        # Generate small sample dataset
        generator = SampleDataGenerator(seed=42)
        dataset = generator.generate_complete_dataset(100, churn_rate=0.3)
        
        # Combine datasets
        combined_data = dataset['customers']
        combined_data = combined_data.merge(dataset['billing'], on='customer_id', how='left')
        combined_data = combined_data.merge(dataset['usage'], on='customer_id', how='left')
        
        # Train model
        predictor = ChurnPredictor(model_type='logistic')  # Use simple model for test
        results = predictor.train(
            combined_data, 
            target_column='churn',
            test_size=0.3,
            tune_hyperparameters=False
        )
        
        # Check results
        assert 'roc_auc' in results
        assert results['roc_auc'] > 0.5  # Should be better than random
        assert predictor.is_trained
        
        # Test prediction
        test_data = combined_data.head(5).drop(columns=['churn'])
        predictions = predictor.predict(test_data)
        probabilities = predictor.predict_proba(test_data)
        
        assert len(predictions) == 5
        assert len(probabilities) == 5
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities)


class TestModelSerialization:
    """Test model saving and loading."""
    
    def test_model_save_load(self):
        """Test model serialization."""
        from telecoms_churn_ml.data import SampleDataGenerator
        from telecoms_churn_ml.models import ChurnPredictor
        
        # Generate and train model
        generator = SampleDataGenerator(seed=42)
        dataset = generator.generate_complete_dataset(50, churn_rate=0.3)
        
        combined_data = dataset['customers']
        combined_data = combined_data.merge(dataset['billing'], on='customer_id', how='left')
        
        predictor = ChurnPredictor(model_type='logistic')
        predictor.train(
            combined_data, 
            target_column='churn',
            test_size=0.3,
            tune_hyperparameters=False
        )
        
        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            # Save model
            predictor.save_model(model_path)
            assert model_path.exists()
            
            # Load model
            new_predictor = ChurnPredictor()
            new_predictor.load_model(model_path)
            
            assert new_predictor.is_trained
            assert new_predictor.model_type == predictor.model_type
            
            # Test that loaded model can make predictions
            test_data = combined_data.head(3).drop(columns=['churn'])
            predictions1 = predictor.predict(test_data)
            predictions2 = new_predictor.predict(test_data)
            
            # Predictions should be identical
            assert np.array_equal(predictions1, predictions2)


if __name__ == "__main__":
    pytest.main([__file__])
