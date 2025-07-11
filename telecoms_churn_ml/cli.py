"""Command-line interface for telecoms churn prediction."""

import sys
import argparse
from pathlib import Path
import pandas as pd
import click
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telecoms_churn_ml.models import ChurnPredictor


@click.group()
def cli():
    """Telecoms Customer Churn Prediction CLI."""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV file with customer data')
@click.option('--output', '-o', required=True, help='Output CSV file for predictions')
@click.option('--model', '-m', default='models/model.pkl', help='Path to trained model')
@click.option('--probability', '-p', is_flag=True, help='Include churn probabilities')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def predict(input, output, model, probability, verbose):
    """Make churn predictions on new customer data."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stdout, level=log_level)
    
    try:
        # Load model
        logger.info(f"Loading model from {model}")
        predictor = ChurnPredictor()
        predictor.load_model(model)
        
        # Load data
        logger.info(f"Loading data from {input}")
        data = pd.read_csv(input)
        logger.info(f"Loaded {len(data)} customers")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = predictor.predict(data)
        
        # Prepare output
        output_data = data.copy()
        output_data['churn_prediction'] = predictions
        
        if probability:
            probabilities = predictor.predict_proba(data)
            output_data['churn_probability'] = probabilities
        
        # Save results
        logger.info(f"Saving predictions to {output}")
        output_data.to_csv(output, index=False)
        
        # Summary
        churn_count = sum(predictions)
        churn_rate = churn_count / len(predictions) * 100
        
        logger.info(f"Prediction Summary:")
        logger.info(f"  • Total customers: {len(predictions)}")
        logger.info(f"  • Predicted churners: {churn_count}")
        logger.info(f"  • Predicted churn rate: {churn_rate:.2f}%")
        
        click.echo(f"✅ Predictions saved to {output}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', default='models/model.pkl', help='Path to trained model')
def info(model):
    """Display information about the trained model."""
    
    try:
        predictor = ChurnPredictor()
        predictor.load_model(model)
        
        model_info = predictor.get_model_info()
        
        click.echo("\n" + "="*50)
        click.echo("📋 MODEL INFORMATION")
        click.echo("="*50)
        
        click.echo(f"Model Type: {model_info['model_type'].upper()}")
        click.echo(f"Target Column: {model_info['target_column']}")
        click.echo(f"Feature Count: {model_info['feature_count']}")
        click.echo(f"Training Status: {'✅ Trained' if model_info['is_trained'] else '❌ Not Trained'}")
        
        if model_info.get('n_estimators'):
            click.echo(f"Estimators: {model_info['n_estimators']}")
        
        click.echo(f"\n🔍 Features Used:")
        for i, feature in enumerate(model_info['feature_columns'][:10], 1):
            click.echo(f"  {i:2d}. {feature}")
        
        if len(model_info['feature_columns']) > 10:
            remaining = len(model_info['feature_columns']) - 10
            click.echo(f"      ... and {remaining} more features")
        
        if model_info['label_encoders']:
            click.echo(f"\n📊 Categorical Features Encoded:")
            for encoder in model_info['label_encoders']:
                click.echo(f"  • {encoder}")
        
        click.echo("="*50)
        
    except Exception as e:
        click.echo(f"❌ Failed to load model info: {e}")
        sys.exit(1)


@cli.command()
@click.option('--data-dir', default='data/raw', help='Directory with raw data')
@click.option('--output', '-o', default='data/processed/processed_data.csv', help='Output file')
def process(data_dir, output):
    """Process raw telecoms data."""
    
    try:
        from telecoms_churn_ml.data import DataProcessor
        
        logger.info(f"Processing data from {data_dir}")
        
        processor = DataProcessor()
        processed_data = processor.clean_and_unify_data(data_dir)
        
        # Save processed data
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data.to_csv(output_path, index=False)
        
        logger.info(f"Processed data saved to {output}")
        logger.info(f"Shape: {processed_data.shape}")
        
        click.echo(f"✅ Data processing completed")
        click.echo(f"📁 Output: {output}")
        click.echo(f"📊 Shape: {processed_data.shape}")
        
    except Exception as e:
        click.echo(f"❌ Data processing failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--output-dir', default='data/raw', help='Output directory for sample data')
@click.option('--customers', '-n', default=1000, help='Number of customers to generate')
@click.option('--churn-rate', default=0.26, help='Expected churn rate (0.0-1.0)')
def generate(output_dir, customers, churn_rate):
    """Generate sample telecoms data for testing."""
    
    try:
        from telecoms_churn_ml.data import SampleDataGenerator
        
        logger.info(f"Generating sample data for {customers} customers")
        
        generator = SampleDataGenerator(seed=42)
        generator.save_sample_data(
            output_dir=output_dir,
            n_customers=customers,
            churn_rate=churn_rate,
            format='csv'
        )
        
        click.echo(f"✅ Sample data generated")
        click.echo(f"📁 Location: {output_dir}")
        click.echo(f"👥 Customers: {customers}")
        click.echo(f"📈 Churn Rate: {churn_rate*100:.1f}%")
        
    except Exception as e:
        click.echo(f"❌ Sample data generation failed: {e}")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
