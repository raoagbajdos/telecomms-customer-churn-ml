# Telecoms Customer Churn ML Pipeline

A comprehensive machine learning pipeline for predicting customer churn in the telecommunications industry. This project follows data science best practices with proper data cleaning, feature engineering, model training, and evaluation.

## Features

- **Data Cleaning & Preprocessing**: Handles messy telecoms and billing data
- **Data Unification**: Merges multiple data sources into a unified dataset
- **Feature Engineering**: Creates meaningful features for churn prediction
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive evaluation metrics and visualizations
- **Production Ready**: Serialized model output (model.pkl) for deployment

## Project Structure

```
telecoms-customer-churn-ml/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Cleaned and processed data
│   └── sample/                 # Sample datasets for testing
├── notebooks/                  # Jupyter notebooks for exploration
├── telecoms_churn_ml/         # Main package
│   ├── data/                  # Data processing modules
│   ├── features/              # Feature engineering
│   ├── models/                # Model training and evaluation
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── config/                    # Configuration files
├── models/                    # Trained models
├── reports/                   # Generated reports and visualizations
└── scripts/                   # Standalone scripts
```

## Installation

This project uses `uv` for fast dependency management. First, install `uv`:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
# Install dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

1. **Prepare your data**: Place raw telecoms and billing data in `data/raw/`
2. **Run the pipeline**:
   ```bash
   python scripts/train_model.py
   ```
3. **Make predictions**:
   ```bash
   churn-predict --input data/new_customers.csv --output predictions.csv
   ```

## Usage

### Data Processing
```python
from telecoms_churn_ml.data import DataProcessor

processor = DataProcessor()
clean_data = processor.clean_and_unify_data('data/raw/')
```

### Model Training
```python
from telecoms_churn_ml.models import ChurnPredictor

predictor = ChurnPredictor()
model = predictor.train(clean_data)
predictor.save_model(model, 'models/model.pkl')
```

### Making Predictions
```python
from telecoms_churn_ml.models import ChurnPredictor

predictor = ChurnPredictor()
predictor.load_model('models/model.pkl')
predictions = predictor.predict(new_data)
```

## Development

### Setup Development Environment
```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Code formatting
black telecoms_churn_ml/
isort telecoms_churn_ml/

# Type checking
mypy telecoms_churn_ml/
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=telecoms_churn_ml --cov-report=html
```

## Model Performance

The pipeline trains multiple models and selects the best performer:
- Random Forest
- XGBoost
- Logistic Regression
- Support Vector Machine

Key metrics tracked:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Configuration

Configure the pipeline using `config/config.yaml`:
- Data sources and formats
- Feature engineering parameters
- Model hyperparameters
- Evaluation metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
