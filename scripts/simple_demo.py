#!/usr/bin/env python3
"""
Simple demo script for telecoms customer churn ML project.
This version uses only built-in Python libraries for testing.
"""

import sys
import os
import random
import pickle
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Simple data generation without external dependencies
def generate_simple_data(n_customers=100):
    """Generate simple sample data for testing."""
    print(f"📊 Generating sample data for {n_customers} customers...")
    
    data = []
    for i in range(n_customers):
        customer = {
            'customer_id': f'CUST_{i:04d}',
            'tenure': random.randint(1, 72),
            'monthly_charges': round(random.uniform(20, 120), 2),
            'total_charges': 0,  # Will calculate
            'contract_type': random.choice(['Month-to-month', 'One year', 'Two year']),
            'payment_method': random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']),
            'internet_service': random.choice(['DSL', 'Fiber optic', 'No']),
            'gender': random.choice(['Male', 'Female']),
            'senior_citizen': random.choice([0, 1]),
            'partner': random.choice(['Yes', 'No']),
            'dependents': random.choice(['Yes', 'No']),
        }
        
        # Calculate total charges
        customer['total_charges'] = round(customer['tenure'] * customer['monthly_charges'], 2)
        
        # Simple churn logic (higher probability for month-to-month, high charges)
        churn_prob = 0.1
        if customer['contract_type'] == 'Month-to-month':
            churn_prob += 0.3
        if customer['monthly_charges'] > 80:
            churn_prob += 0.2
        if customer['tenure'] < 12:
            churn_prob += 0.2
            
        customer['churn'] = 1 if random.random() < churn_prob else 0
        
        data.append(customer)
    
    return data

# Simple model class
class SimpleChurnModel:
    """Simple churn prediction model using basic rules."""
    
    def __init__(self):
        self.is_trained = False
        self.rules = {}
    
    def train(self, data):
        """Train the simple model."""
        print("🤖 Training simple rule-based model...")
        
        # Analyze patterns in training data
        churners = [d for d in data if d['churn'] == 1]
        total_customers = len(data)
        total_churners = len(churners)
        
        print(f"   Training on {total_customers} customers")
        print(f"   Churners: {total_churners} ({total_churners/total_customers*100:.1f}%)")
        
        # Simple rules based on data analysis
        self.rules = {
            'high_charges_threshold': 80,
            'low_tenure_threshold': 12,
            'month_to_month_risk': 0.4,
            'base_risk': 0.1
        }
        
        self.is_trained = True
        return {
            'accuracy': 0.75,  # Simulated
            'total_samples': total_customers,
            'churn_rate': total_churners/total_customers
        }
    
    def predict(self, customer_data):
        """Make predictions for customers."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = []
        probabilities = []
        
        for customer in customer_data:
            risk_score = self.rules['base_risk']
            
            # Apply rules
            if customer['monthly_charges'] > self.rules['high_charges_threshold']:
                risk_score += 0.3
            
            if customer['tenure'] < self.rules['low_tenure_threshold']:
                risk_score += 0.2
            
            if customer['contract_type'] == 'Month-to-month':
                risk_score += self.rules['month_to_month_risk']
            
            # Clip probability
            risk_score = min(risk_score, 0.9)
            
            prediction = 1 if risk_score > 0.5 else 0
            
            predictions.append(prediction)
            probabilities.append(risk_score)
        
        return predictions, probabilities
    
    def save(self, filepath):
        """Save the model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load the model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def main():
    """Run the simple demo."""
    print("🚀 Simple Telecoms Customer Churn ML Demo")
    print("=" * 50)
    
    try:
        # Set random seed for reproducibility
        random.seed(42)
        
        # Step 1: Generate data
        data = generate_simple_data(1000)
        print(f"   ✅ Generated {len(data)} customer records")
        
        # Step 2: Split data
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(train_data)} customers")
        print(f"   Testing: {len(test_data)} customers")
        
        # Step 3: Train model
        print(f"\n🤖 Training model...")
        model = SimpleChurnModel()
        results = model.train(train_data)
        
        print(f"   ✅ Model trained successfully")
        print(f"   Simulated accuracy: {results['accuracy']:.2f}")
        
        # Step 4: Make predictions
        print(f"\n🔮 Making predictions...")
        predictions, probabilities = model.predict(test_data)
        
        # Calculate metrics
        actual_churn = [d['churn'] for d in test_data]
        correct_predictions = sum(1 for p, a in zip(predictions, actual_churn) if p == a)
        accuracy = correct_predictions / len(test_data)
        
        predicted_churners = sum(predictions)
        actual_churners = sum(actual_churn)
        
        print(f"   Test accuracy: {accuracy:.3f}")
        print(f"   Predicted churners: {predicted_churners}")
        print(f"   Actual churners: {actual_churners}")
        
        # Step 5: Save model
        print(f"\n💾 Saving model...")
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "model.pkl"
        model.save(model_path)
        
        print(f"   ✅ Model saved to: {model_path}")
        
        # Step 6: Sample predictions
        print(f"\n📋 Sample predictions:")
        for i in range(5):
            customer = test_data[i]
            pred = predictions[i]
            prob = probabilities[i]
            actual = customer['churn']
            
            status = "✅" if pred == actual else "❌"
            print(f"   {status} {customer['customer_id']}: {'CHURN' if pred == 1 else 'STAY'} "
                  f"(prob: {prob:.3f}, actual: {'CHURN' if actual == 1 else 'STAY'})")
        
        # Step 7: Create sample data files
        print(f"\n📁 Creating sample data files...")
        data_dir = project_root / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as simple CSV format
        csv_content = []
        if data:
            # Header
            headers = list(data[0].keys())
            csv_content.append(','.join(headers))
            
            # Data rows
            for row in data[:100]:  # Save first 100 for testing
                values = [str(row[h]) for h in headers]
                csv_content.append(','.join(values))
        
        csv_file = data_dir / "sample_customers.csv"
        with open(csv_file, 'w') as f:
            f.write('\n'.join(csv_content))
        
        print(f"   ✅ Sample data saved to: {csv_file}")
        
        print(f"\n🎉 Demo completed successfully!")
        print("=" * 50)
        print("✅ Your model.pkl file is ready for use!")
        print(f"📁 Model location: {model_path}")
        print(f"📊 Sample data: {csv_file}")
        print(f"\n📋 Summary:")
        print(f"   • Generated 1,000 customer records")
        print(f"   • Trained a simple rule-based model")
        print(f"   • Achieved {accuracy:.1%} accuracy on test data")
        print(f"   • Model saved as model.pkl")
        print(f"\n💡 Next steps:")
        print(f"   • Install dependencies: pip install pandas scikit-learn")
        print(f"   • Run full pipeline: python scripts/train_model.py")
        print(f"   • Check README.md for more details")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
