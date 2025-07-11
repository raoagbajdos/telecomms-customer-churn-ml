# 🚀 Quick Start Guide

Welcome to the Telecoms Customer Churn ML project! This guide will get you up and running in minutes.

## 📋 Prerequisites

- Python 3.9 or higher
- Git (optional, for cloning)

## ⚡ Quick Setup (2 minutes)

### Option 1: Automatic Setup (Recommended)

```bash
# Navigate to the project directory
cd telecoms-customer-churn-ml

# Run the setup script (installs everything and generates sample data)
python setup.py
```

### Option 2: Manual Setup

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

## 🎯 Quick Demo (1 minute)

Run the demo to see everything in action:

```bash
python scripts/demo.py
```

This will:
- ✅ Generate sample telecoms data (1000 customers)
- ✅ Clean and process the data  
- ✅ Train a machine learning model
- ✅ Save the model as `models/model.pkl`
- ✅ Make sample predictions

## 📊 Quick Training

Train a model with the full pipeline:

```bash
# Generate sample data and train model
python scripts/train_model.py --generate-sample --verbose

# Or with make (if available)
make train
```

## 🔮 Quick Predictions

Use the trained model to make predictions:

```bash
# Using the CLI
churn-predict --input data/raw/customers.csv --output predictions.csv --probability

# Or generate new sample data first
churn-predict generate --customers 500 --output-dir data/raw
churn-predict --input data/raw/customers.csv --output predictions.csv
```

## 📁 Project Structure

```
telecoms-customer-churn-ml/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Cleaned data
├── models/
│   └── model.pkl         # Trained model (after running demo)
├── scripts/
│   ├── demo.py          # Quick demo script
│   └── train_model.py   # Full training pipeline
├── telecoms_churn_ml/   # Main package
└── config/              # Configuration files
```

## 🛠️ Available Commands

### Using the CLI:
```bash
churn-predict --help              # Show help
churn-predict info                # Model information
churn-predict generate            # Generate sample data
churn-predict process             # Process raw data
churn-predict --input X --output Y # Make predictions
```

### Using Make (if available):
```bash
make help                # Show all commands
make demo                # Run demo
make train               # Train model
make test                # Run tests
make format              # Format code
```

### Using Python scripts:
```bash
python scripts/demo.py                    # Quick demo
python scripts/train_model.py --help      # Training options
```

## 📈 Understanding the Output

After training, you'll see results like:

```
🎯 TELECOMS CUSTOMER CHURN PREDICTION RESULTS
==============================================

📊 Model Performance:
   • Best Model: RANDOM_FOREST
   • ROC-AUC Score: 0.8547
   • Training Samples: 1,600
   • Test Samples: 400

📈 Churn Detection Metrics:
   • Precision: 0.7234
   • Recall: 0.6891
   • F1-Score: 0.7058

🔍 Top 5 Most Important Features:
   1. monthly_charges: 0.2341
   2. total_charges: 0.1876
   3. tenure: 0.1654
   4. contract_type: 0.1223
   5. internet_service: 0.0987

✅ Model successfully saved as 'model.pkl'
```

## 🎯 Next Steps

1. **Explore the Data**: Check the generated sample data in `data/raw/`
2. **Customize Training**: Modify `config/config.yaml` for different settings
3. **Add Your Data**: Replace sample data with real telecoms data
4. **Experiment**: Try different models and hyperparameters
5. **Deploy**: Use the trained model in production

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure dependencies are installed
uv pip install -e .
```

### No Data Found
```bash
# Generate sample data
python scripts/train_model.py --generate-sample
```

### Model Not Found
```bash
# Train a model first
python scripts/demo.py
```

## 📚 Learn More

- **README.md** - Comprehensive documentation
- **config/config.yaml** - All configuration options
- **tests/** - Example usage and tests
- **Makefile** - Available commands

## 🎉 Success!

If you see the model.pkl file in the `models/` directory, you're all set! 

Your telecoms customer churn prediction system is ready to use. The model can predict which customers are likely to churn based on their usage patterns, billing information, and service preferences.

**Happy predicting! 🔮**
