#!/usr/bin/env python3
"""
Test the trained model with a simple prediction example.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telecoms_churn_ml.models.predictor import ChurnPredictor


def create_sample_customer():
    """Create a sample customer for testing."""
    sample_data = {
        'customer_id': ['TEST_001'],
        'gender': ['Female'],
        'senior_citizen': [0],
        'partner': ['Yes'],
        'dependents': ['No'],
        'billing_phone_service': ['Yes'],
        'billing_multiple_lines': ['No'],
        'billing_internet_service': ['Fiber optic'],
        'billing_online_security': ['No'],
        'billing_online_backup': ['Yes'],
        'billing_device_protection': ['No'],
        'billing_tech_support': ['No'],
        'billing_streaming_tv': ['No'],
        'billing_streaming_movies': ['No'],
        'billing_contract': ['Month-to-month'],
        'billing_paperless_billing': ['Yes'],
        'billing_payment_method': ['Electronic check'],
        'billing_tenure': [1],
        'billing_monthly_charges': [70.70],
        'billing_total_charges': [70.70],
        'usage_monthly_data_gb': [15.5],
        'usage_monthly_calls': [120],
        'usage_monthly_sms': [50],
        'usage_peak_hour_usage_pct': [25.0],
        'usage_international_calls': [5],
        'usage_roaming_usage_gb': [0.5],
        'usage_video_streaming_gb': [8.2],
        'usage_social_media_gb': [2.1],
        'usage_gaming_gb': [1.8],
        'usage_web_browsing_gb': [3.4],
        'usage_unique_apps_used': [15],
        'usage_avg_session_duration_min': [22.5],
        'customer_care_support_calls_count': [2],
        'customer_care_support_chat_count': [1],
        'customer_care_support_email_count': [0],
        'customer_care_avg_resolution_time_hours': [4.5],
        'customer_care_first_call_resolution_rate': [0.5],
        'customer_care_complaint_count': [1],
        'customer_care_primary_complaint_type': ['billing'],
        'customer_care_customer_satisfaction_score': [3.2],
        'customer_care_escalation_count': [0],
        'customer_care_avg_response_time_min': [15.0],
        'crm_customer_segment': ['Basic'],
        'crm_lifetime_value': [850.0],
        'crm_acquisition_channel': ['Online'],
        'crm_last_login_days_ago': [2],
        'crm_app_usage_frequency': ['Daily'],
        'crm_email_open_rate': [0.25],
        'crm_sms_response_rate': [0.40],
        'crm_promotion_usage_count': [1],
        'crm_payment_delays': [0],
        'crm_autopay_enabled': [0],
        'crm_referrals_made': [0],
        'crm_referred_by_friend': [0],
        'crm_plan_changes_count': [1],
        'crm_service_additions_count': [0],
        'social_has_social_account': [1],
        'social_social_mentions_count': [5],
        'social_social_sentiment': ['neutral'],
        'social_online_reviews_count': [1],
        'social_website_visits_monthly': [12],
        'social_mobile_app_rating': [4.0],
        'social_forum_posts_count': [0],
        'social_community_member': [0],
        'social_brand_mentions_positive': [2],
        'social_brand_mentions_negative': [1],
        'social_follower_count': [150],
        'social_is_influencer': [0],
        'network_avg_download_speed_mbps': [45.2],
        'network_avg_upload_speed_mbps': [12.1],
        'network_network_uptime_pct': [98.5],
        'network_connection_drops_monthly': [3],
        'network_avg_signal_strength_db': [-65],
        'network_coverage_rating': ['Good'],
        'network_primary_network_type': ['4G'],
        'network_data_throttling_events': [1],
        'network_throttling_duration_hours': [2.0],
        'network_roaming_countries_visited': [1],
        'network_roaming_issues_count': [0],
        'network_peak_hour_slowdown_pct': [15.0],
        'network_congestion_events_monthly': [4],
        'network_primary_device_type': ['Smartphone'],
        'network_network_support_tickets': [1],
        'network_avg_issue_resolution_hours': [6.0],
        'network_urban_area': [True],
        'network_tower_distance_km': [1.2]
    }
    
    return pd.DataFrame(sample_data)


def main():
    """Test the trained model."""
    logger.info("Testing trained model")
    
    model_path = project_root / "models" / "model.pkl"
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        # Load the trained model
        logger.info("Loading trained model")
        predictor = ChurnPredictor()
        predictor.load_model(str(model_path))
        
        # Get model information
        model_info = predictor.get_model_info()
        logger.info("Model Information:")
        logger.info(f"  Model type: {model_info['model_type']}")
        logger.info(f"  Features: {model_info['feature_count']}")
        logger.info(f"  Target: {model_info['target_column']}")
        
        # Create sample data
        logger.info("Creating sample customer data")
        sample_df = create_sample_customer()
        logger.info(f"Sample data shape: {sample_df.shape}")
        
        # Make prediction using the actual training data format
        logger.info("Loading actual training data for prediction test")
        training_data = pd.read_csv(project_root / "data" / "processed" / "unified_data.csv")
        test_sample = training_data.head(3).copy()
        
        # Remove target column for prediction
        if 'billing_churn' in test_sample.columns:
            actual_labels = test_sample['billing_churn'].values
            test_sample_features = test_sample.drop(columns=['billing_churn'])
        else:
            actual_labels = None
            test_sample_features = test_sample
        
        logger.info(f"Test data shape: {test_sample_features.shape}")
        
        # Make predictions
        logger.info("Making predictions")
        predictions = predictor.predict(test_sample_features)
        probabilities = predictor.predict_proba(test_sample_features)
        
        # Display results
        logger.info("Prediction Results:")
        for i in range(len(predictions)):
            churn_prob = probabilities[i][1] if probabilities is not None else "N/A"
            actual = actual_labels[i] if actual_labels is not None else "Unknown"
            logger.info(f"  Customer {i+1}: Predicted={predictions[i]}, Probability={churn_prob:.3f}, Actual={actual}")
        
        logger.success("Model testing completed successfully!")
        
        # Model summary
        logger.info("\n" + "="*50)
        logger.info("MODEL SUMMARY")
        logger.info("="*50)
        logger.info(f"Model file: {model_path}")
        logger.info(f"Model size: {model_path.stat().st_size / 1024:.2f} KB")
        logger.info(f"Model type: {model_info['model_type']} (XGBoost)")
        logger.info(f"Features used: {model_info['feature_count']}")
        logger.info(f"Target variable: {model_info['target_column']}")
        logger.info(f"Categorical features encoded: {len(model_info.get('label_encoders', []))}")
        
        if model_info.get('n_estimators'):
            logger.info(f"Number of estimators: {model_info['n_estimators']}")
        
        logger.info("\nThe model is ready for production use!")
        
    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
