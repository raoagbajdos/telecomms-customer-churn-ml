"""Sample data generation for testing and development."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import random
from datetime import datetime, timedelta
from loguru import logger


class SampleDataGenerator:
    """
    Generates realistic sample telecoms and billing data for testing.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the sample data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"SampleDataGenerator initialized with seed {seed}")
    
    def generate_customer_data(self, n_customers: int = 1000) -> pd.DataFrame:
        """
        Generate customer demographic data.
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            DataFrame with customer data
        """
        logger.info(f"Generating customer data for {n_customers} customers")
        
        # Generate customer IDs
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        
        # Demographics
        genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.51, 0.49])
        senior_citizens = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
        partners = np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48])
        dependents = np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])
        
        # Add some noise to make data "messy"
        genders = self._add_categorical_noise(genders, ['M', 'F', 'MALE', 'FEMALE', ''])
        
        customer_data = pd.DataFrame({
            'customer_id': customer_ids,
            'gender': genders,
            'senior_citizen': senior_citizens,
            'partner': partners,
            'dependents': dependents
        })
        
        return customer_data
    
    def generate_billing_data(self, customer_ids: list, 
                            churn_rate: float = 0.26) -> pd.DataFrame:
        """
        Generate billing and service data.
        
        Args:
            customer_ids: List of customer IDs
            churn_rate: Expected churn rate
            
        Returns:
            DataFrame with billing data
        """
        n_customers = len(customer_ids)
        logger.info(f"Generating billing data for {n_customers} customers")
        
        # Service features
        phone_services = np.random.choice(['Yes', 'No'], n_customers, p=[0.90, 0.10])
        multiple_lines = np.where(
            phone_services == 'Yes',
            np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.42, 0.48, 0.10]),
            'No phone service'
        )
        
        internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])
        
        # Internet-dependent services
        internet_dependent_services = ['online_security', 'online_backup', 'device_protection', 
                                     'tech_support', 'streaming_tv', 'streaming_movies']
        service_data = {}
        
        for service in internet_dependent_services:
            service_data[service] = np.where(
                internet_services == 'No',
                'No internet service',
                np.random.choice(['Yes', 'No'], n_customers, p=[0.35, 0.65])
            )
        
        # Contract and payment
        contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                        n_customers, p=[0.55, 0.21, 0.24])
        paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])
        payment_methods = np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer (automatic)', 'Credit card (automatic)'],
                                         n_customers, p=[0.34, 0.19, 0.22, 0.25])
        
        # Tenure and charges
        tenure = np.random.exponential(scale=24, size=n_customers).astype(int)
        tenure = np.clip(tenure, 1, 72)  # Cap at 72 months
        
        # Monthly charges based on services
        base_charges = np.random.normal(50, 15, n_customers)
        
        # Adjust charges based on services
        fiber_bonus = np.where(internet_services == 'Fiber optic', 
                              np.random.normal(25, 5, n_customers), 0)
        service_bonus = sum([np.where(service_data[service] == 'Yes', 
                                    np.random.normal(5, 2, n_customers), 0) 
                           for service in internet_dependent_services])
        
        monthly_charges = np.clip(base_charges + fiber_bonus + service_bonus, 18.25, 118.75)
        
        # Total charges with some missing/corrupted values
        total_charges = tenure * monthly_charges + np.random.normal(0, 50, n_customers)
        total_charges = np.clip(total_charges, 18.80, 8684.80)
        
        # Generate churn (target variable)
        # Higher churn probability for month-to-month, high charges, low tenure
        churn_prob = 0.15  # Base probability
        churn_prob += np.where(contract_types == 'Month-to-month', 0.20, 0)
        churn_prob += np.where(tenure < 12, 0.15, 0)
        churn_prob += np.where(monthly_charges > 80, 0.10, 0)
        churn_prob += np.where(internet_services == 'Fiber optic', 0.05, 0)
        
        churn = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_customers)
        
        # Add noise to make data messy
        monthly_charges = self._add_numeric_noise(monthly_charges)
        total_charges = self._add_missing_values(total_charges, 0.1)
        contract_types = self._add_categorical_noise(contract_types, 
                                                   ['Monthly', 'Yearly', 'Two years', ''])
        
        billing_data = pd.DataFrame({
            'customer_id': customer_ids,
            'phone_service': phone_services,
            'multiple_lines': multiple_lines,
            'internet_service': internet_services,
            'online_security': service_data['online_security'],
            'online_backup': service_data['online_backup'],
            'device_protection': service_data['device_protection'],
            'tech_support': service_data['tech_support'],
            'streaming_tv': service_data['streaming_tv'],
            'streaming_movies': service_data['streaming_movies'],
            'contract': contract_types,
            'paperless_billing': paperless_billing,
            'payment_method': payment_methods,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'churn': churn
        })
        
        return billing_data
    
    def generate_usage_data(self, customer_ids: list) -> pd.DataFrame:
        """
        Generate comprehensive usage/activity data.
        
        Args:
            customer_ids: List of customer IDs
            
        Returns:
            DataFrame with usage data
        """
        n_customers = len(customer_ids)
        logger.info(f"Generating usage data for {n_customers} customers")
        
        # Basic usage metrics
        monthly_data_gb = np.random.lognormal(mean=3, sigma=1, size=n_customers)
        monthly_calls = np.random.poisson(lam=45, size=n_customers)
        monthly_sms = np.random.poisson(lam=120, size=n_customers)
        
        # Voice usage patterns
        peak_hour_usage_pct = np.random.beta(2, 5, n_customers) * 100
        international_calls = np.random.poisson(lam=3, size=n_customers)
        roaming_usage_gb = np.random.exponential(scale=0.5, size=n_customers)
        
        # Data usage patterns
        video_streaming_gb = np.random.gamma(2, 5, n_customers)
        social_media_gb = np.random.gamma(1.5, 2, n_customers)
        gaming_gb = np.random.exponential(scale=1, size=n_customers)
        web_browsing_gb = np.random.gamma(1, 3, n_customers)
        
        # App usage metrics
        unique_apps_used = np.random.poisson(lam=25, size=n_customers)
        avg_session_duration_min = np.random.lognormal(mean=3, sigma=0.5, size=n_customers)
        
        # Add missing values and outliers
        monthly_data_gb = self._add_missing_values(monthly_data_gb, 0.05)
        monthly_data_gb = self._add_outliers(monthly_data_gb, 0.02)
        
        usage_data = pd.DataFrame({
            'customer_id': customer_ids,
            'monthly_data_gb': monthly_data_gb,
            'monthly_calls': monthly_calls,
            'monthly_sms': monthly_sms,
            'peak_hour_usage_pct': peak_hour_usage_pct,
            'international_calls': international_calls,
            'roaming_usage_gb': roaming_usage_gb,
            'video_streaming_gb': video_streaming_gb,
            'social_media_gb': social_media_gb,
            'gaming_gb': gaming_gb,
            'web_browsing_gb': web_browsing_gb,
            'unique_apps_used': unique_apps_used,
            'avg_session_duration_min': avg_session_duration_min
        })
        
        return usage_data
    
    def generate_customer_care_data(self, customer_ids: list) -> pd.DataFrame:
        """
        Generate customer care interaction data.
        
        Args:
            customer_ids: List of customer IDs
            
        Returns:
            DataFrame with customer care data
        """
        n_customers = len(customer_ids)
        logger.info(f"Generating customer care data for {n_customers} customers")
        
        # Support interactions
        support_calls_count = np.random.poisson(lam=2, size=n_customers)
        support_chat_count = np.random.poisson(lam=1.5, size=n_customers)
        support_email_count = np.random.poisson(lam=0.8, size=n_customers)
        
        # Resolution metrics
        avg_resolution_time_hours = np.random.lognormal(mean=2, sigma=0.8, size=n_customers)
        first_call_resolution_rate = np.random.beta(3, 2, n_customers)
        
        # Complaint data
        complaint_count = np.random.poisson(lam=0.5, size=n_customers)
        complaint_types = []
        satisfaction_scores = []
        
        for i in range(n_customers):
            # Complaint categories
            complaint_categories = ['Billing', 'Network', 'Service', 'Technical', 'None']
            weights = [0.25, 0.20, 0.15, 0.15, 0.25]
            complaint_types.append(np.random.choice(complaint_categories, p=weights))
            
            # Customer satisfaction (1-5 scale, with some bias towards complaints)
            if complaint_count[i] > 2:
                sat_score = np.random.normal(2.5, 0.8)
            elif support_calls_count[i] > 5:
                sat_score = np.random.normal(3.2, 0.6)
            else:
                sat_score = np.random.normal(4.1, 0.7)
            satisfaction_scores.append(np.clip(sat_score, 1, 5))
        
        # Escalation data
        escalation_count = np.random.poisson(lam=0.3, size=n_customers)
        
        # Response times
        avg_response_time_min = np.random.lognormal(mean=3, sigma=0.5, size=n_customers)
        
        care_data = pd.DataFrame({
            'customer_id': customer_ids,
            'support_calls_count': support_calls_count,
            'support_chat_count': support_chat_count,
            'support_email_count': support_email_count,
            'avg_resolution_time_hours': avg_resolution_time_hours,
            'first_call_resolution_rate': first_call_resolution_rate,
            'complaint_count': complaint_count,
            'primary_complaint_type': complaint_types,
            'customer_satisfaction_score': satisfaction_scores,
            'escalation_count': escalation_count,
            'avg_response_time_min': avg_response_time_min
        })
        
        return care_data
    
    def generate_crm_data(self, customer_ids: list) -> pd.DataFrame:
        """
        Generate CRM and customer relationship data.
        
        Args:
            customer_ids: List of customer IDs
            
        Returns:
            DataFrame with CRM data
        """
        n_customers = len(customer_ids)
        logger.info(f"Generating CRM data for {n_customers} customers")
        
        # Customer segmentation
        segments = ['Premium', 'Standard', 'Basic', 'Enterprise']
        segment_weights = [0.15, 0.45, 0.30, 0.10]
        customer_segments = np.random.choice(segments, n_customers, p=segment_weights)
        
        # Lifetime value
        lifetime_value = np.random.lognormal(mean=7, sigma=1, size=n_customers)
        
        # Acquisition data
        acquisition_channels = ['Online', 'Store', 'Referral', 'Call Center', 'Partner']
        acquisition_weights = [0.35, 0.25, 0.15, 0.15, 0.10]
        acquisition_channel = np.random.choice(acquisition_channels, n_customers, p=acquisition_weights)
        
        # Account activity
        last_login_days_ago = np.random.exponential(scale=7, size=n_customers)
        app_usage_frequency = np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], 
                                             n_customers, p=[0.3, 0.4, 0.2, 0.1])
        
        # Marketing interactions
        email_open_rate = np.random.beta(2, 3, n_customers)
        sms_response_rate = np.random.beta(1.5, 4, n_customers)
        promotion_usage_count = np.random.poisson(lam=3, size=n_customers)
        
        # Payment behavior
        payment_delays = np.random.poisson(lam=1, size=n_customers)
        autopay_enabled = np.random.choice([True, False], n_customers, p=[0.65, 0.35])
        
        # Referral data
        referrals_made = np.random.poisson(lam=0.5, size=n_customers)
        referred_by_friend = np.random.choice([True, False], n_customers, p=[0.20, 0.80])
        
        # Account changes
        plan_changes_count = np.random.poisson(lam=1.2, size=n_customers)
        service_additions_count = np.random.poisson(lam=0.8, size=n_customers)
        
        crm_data = pd.DataFrame({
            'customer_id': customer_ids,
            'customer_segment': customer_segments,
            'lifetime_value': lifetime_value,
            'acquisition_channel': acquisition_channel,
            'last_login_days_ago': last_login_days_ago,
            'app_usage_frequency': app_usage_frequency,
            'email_open_rate': email_open_rate,
            'sms_response_rate': sms_response_rate,
            'promotion_usage_count': promotion_usage_count,
            'payment_delays': payment_delays,
            'autopay_enabled': autopay_enabled,
            'referrals_made': referrals_made,
            'referred_by_friend': referred_by_friend,
            'plan_changes_count': plan_changes_count,
            'service_additions_count': service_additions_count
        })
        
        return crm_data
    
    def generate_social_data(self, customer_ids: list) -> pd.DataFrame:
        """
        Generate social media and digital engagement data.
        
        Args:
            customer_ids: List of customer IDs
            
        Returns:
            DataFrame with social data
        """
        n_customers = len(customer_ids)
        logger.info(f"Generating social data for {n_customers} customers")
        
        # Social media presence
        has_social_account = np.random.choice([True, False], n_customers, p=[0.75, 0.25])
        
        # Social engagement metrics
        social_mentions_count = np.where(has_social_account,
                                       np.random.poisson(lam=2, size=n_customers), 0)
        
        social_sentiment = []
        for i in range(n_customers):
            if has_social_account[i]:
                sentiment_options = ['Positive', 'Neutral', 'Negative']
                sentiment_weights = [0.4, 0.4, 0.2]
                social_sentiment.append(np.random.choice(sentiment_options, p=sentiment_weights))
            else:
                social_sentiment.append('No Data')
        
        # Online reviews
        online_reviews_count = np.random.poisson(lam=0.3, size=n_customers)
        avg_review_rating = np.where(online_reviews_count > 0,
                                   np.random.normal(3.5, 1.2, n_customers), np.nan)
        avg_review_rating = np.clip(avg_review_rating, 1, 5)
        
        # Digital engagement
        website_visits_monthly = np.random.poisson(lam=5, size=n_customers)
        mobile_app_rating = np.where(np.random.random(n_customers) < 0.6,
                                   np.random.normal(4.0, 0.8, n_customers), np.nan)
        mobile_app_rating = np.clip(mobile_app_rating, 1, 5)
        
        # Community participation
        forum_posts_count = np.random.poisson(lam=0.8, size=n_customers)
        community_member = np.random.choice([True, False], n_customers, p=[0.25, 0.75])
        
        # Brand advocacy
        brand_mentions_positive = np.where(has_social_account,
                                         np.random.poisson(lam=1, size=n_customers), 0)
        brand_mentions_negative = np.where(has_social_account,
                                         np.random.poisson(lam=0.3, size=n_customers), 0)
        
        # Influencer status
        follower_count = np.where(has_social_account,
                                np.random.lognormal(mean=6, sigma=2, size=n_customers), 0)
        is_influencer = follower_count > 10000
        
        social_data = pd.DataFrame({
            'customer_id': customer_ids,
            'has_social_account': has_social_account,
            'social_mentions_count': social_mentions_count,
            'social_sentiment': social_sentiment,
            'online_reviews_count': online_reviews_count,
            'avg_review_rating': avg_review_rating,
            'website_visits_monthly': website_visits_monthly,
            'mobile_app_rating': mobile_app_rating,
            'forum_posts_count': forum_posts_count,
            'community_member': community_member,
            'brand_mentions_positive': brand_mentions_positive,
            'brand_mentions_negative': brand_mentions_negative,
            'follower_count': follower_count,
            'is_influencer': is_influencer
        })
        
        return social_data
    
    def generate_network_data(self, customer_ids: list) -> pd.DataFrame:
        """
        Generate network performance and technical data.
        
        Args:
            customer_ids: List of customer IDs
            
        Returns:
            DataFrame with network data
        """
        n_customers = len(customer_ids)
        logger.info(f"Generating network data for {n_customers} customers")
        
        # Network quality metrics
        avg_download_speed_mbps = np.random.lognormal(mean=4, sigma=0.8, size=n_customers)
        avg_upload_speed_mbps = avg_download_speed_mbps * np.random.uniform(0.1, 0.3, n_customers)
        
        # Connection reliability
        network_uptime_pct = np.random.beta(20, 2, n_customers) * 100
        connection_drops_monthly = np.random.poisson(lam=3, size=n_customers)
        
        # Coverage and signal strength
        avg_signal_strength_db = np.random.normal(-75, 10, n_customers)
        avg_signal_strength_db = np.clip(avg_signal_strength_db, -120, -40)
        
        coverage_rating = []
        for signal in avg_signal_strength_db:
            if signal > -60:
                rating = 'Excellent'
            elif signal > -70:
                rating = 'Good'
            elif signal > -80:
                rating = 'Fair'
            else:
                rating = 'Poor'
            coverage_rating.append(rating)
        
        # Network technology
        network_types = ['5G', '4G LTE', '4G', '3G']
        network_weights = [0.25, 0.55, 0.15, 0.05]
        primary_network_type = np.random.choice(network_types, n_customers, p=network_weights)
        
        # Data throttling
        data_throttling_events = np.random.poisson(lam=1, size=n_customers)
        throttling_duration_hours = np.where(data_throttling_events > 0,
                                           np.random.exponential(scale=8, size=n_customers), 0)
        
        # Roaming performance
        roaming_countries_visited = np.random.poisson(lam=2, size=n_customers)
        roaming_issues_count = np.where(roaming_countries_visited > 0,
                                      np.random.poisson(lam=0.5, size=n_customers), 0)
        
        # Network congestion
        peak_hour_slowdown_pct = np.random.beta(2, 8, n_customers) * 100
        congestion_events_monthly = np.random.poisson(lam=5, size=n_customers)
        
        # Device compatibility
        device_types = ['Smartphone', 'Tablet', 'IoT Device', 'Mobile Hotspot']
        device_weights = [0.70, 0.15, 0.10, 0.05]
        primary_device_type = np.random.choice(device_types, n_customers, p=device_weights)
        
        # Network support issues
        network_support_tickets = np.random.poisson(lam=1.5, size=n_customers)
        avg_issue_resolution_hours = np.random.lognormal(mean=3, sigma=0.8, size=n_customers)
        
        # Location-based metrics
        urban_area = np.random.choice([True, False], n_customers, p=[0.75, 0.25])
        tower_distance_km = np.where(urban_area,
                                   np.random.exponential(scale=2, size=n_customers),
                                   np.random.exponential(scale=8, size=n_customers))
        
        network_data = pd.DataFrame({
            'customer_id': customer_ids,
            'avg_download_speed_mbps': avg_download_speed_mbps,
            'avg_upload_speed_mbps': avg_upload_speed_mbps,
            'network_uptime_pct': network_uptime_pct,
            'connection_drops_monthly': connection_drops_monthly,
            'avg_signal_strength_db': avg_signal_strength_db,
            'coverage_rating': coverage_rating,
            'primary_network_type': primary_network_type,
            'data_throttling_events': data_throttling_events,
            'throttling_duration_hours': throttling_duration_hours,
            'roaming_countries_visited': roaming_countries_visited,
            'roaming_issues_count': roaming_issues_count,
            'peak_hour_slowdown_pct': peak_hour_slowdown_pct,
            'congestion_events_monthly': congestion_events_monthly,
            'primary_device_type': primary_device_type,
            'network_support_tickets': network_support_tickets,
            'avg_issue_resolution_hours': avg_issue_resolution_hours,
            'urban_area': urban_area,
            'tower_distance_km': tower_distance_km
        })
        
        return network_data
    
    def _add_categorical_noise(self, data: np.ndarray, 
                              noise_values: list, noise_rate: float = 0.05) -> np.ndarray:
        """Add noise to categorical data."""
        noisy_data = data.copy()
        n_noise = int(len(data) * noise_rate)
        noise_indices = np.random.choice(len(data), n_noise, replace=False)
        
        for idx in noise_indices:
            noisy_data[idx] = np.random.choice(noise_values)
        
        return noisy_data
    
    def _add_numeric_noise(self, data: np.ndarray, noise_rate: float = 0.02) -> np.ndarray:
        """Add noise to numeric data."""
        noisy_data = data.copy()
        n_noise = int(len(data) * noise_rate)
        noise_indices = np.random.choice(len(data), n_noise, replace=False)
        
        # Add random values or multiply by random factor
        for idx in noise_indices:
            if np.random.random() < 0.5:
                noisy_data[idx] *= np.random.uniform(0.8, 1.2)
            else:
                noisy_data[idx] += np.random.normal(0, data.std() * 0.1)
        
        return noisy_data
    
    def _add_missing_values(self, data: np.ndarray, missing_rate: float = 0.05) -> np.ndarray:
        """Add missing values to data."""
        noisy_data = data.copy()
        n_missing = int(len(data) * missing_rate)
        missing_indices = np.random.choice(len(data), n_missing, replace=False)
        noisy_data[missing_indices] = np.nan
        return noisy_data
    
    def _add_outliers(self, data: np.ndarray, outlier_rate: float = 0.02) -> np.ndarray:
        """Add outliers to numeric data."""
        noisy_data = data.copy()
        n_outliers = int(len(data) * outlier_rate)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        
        # Create extreme values
        for idx in outlier_indices:
            if np.random.random() < 0.5:
                noisy_data[idx] = data.mean() + 5 * data.std()
            else:
                noisy_data[idx] = max(0, data.mean() - 5 * data.std())
        
        return noisy_data
    
    def generate_complete_dataset(self, n_customers: int = 1000, 
                                churn_rate: float = 0.26) -> Dict[str, pd.DataFrame]:
        """
        Generate complete comprehensive dataset with multiple tables including:
        - Customer demographics
        - Billing and service data
        - Usage patterns and behavior
        - Customer care interactions
        - CRM and relationship data
        - Social media and digital engagement
        - Network performance and technical data
        
        Args:
            n_customers: Number of customers to generate
            churn_rate: Expected churn rate
            
        Returns:
            Dictionary of DataFrames with all data sources
        """
        logger.info(f"Generating complete comprehensive dataset for {n_customers} customers")
        
        # Generate customer data
        customer_data = self.generate_customer_data(n_customers)
        customer_ids = customer_data['customer_id'].tolist()
        
        # Generate billing data
        billing_data = self.generate_billing_data(customer_ids, churn_rate)
        
        # Generate usage data
        usage_data = self.generate_usage_data(customer_ids)
        
        # Generate customer care data
        care_data = self.generate_customer_care_data(customer_ids)
        
        # Generate CRM data
        crm_data = self.generate_crm_data(customer_ids)
        
        # Generate social data
        social_data = self.generate_social_data(customer_ids)
        
        # Generate network data
        network_data = self.generate_network_data(customer_ids)
        
        dataset = {
            'customers': customer_data,
            'billing': billing_data,
            'usage': usage_data,
            'customer_care': care_data,
            'crm': crm_data,
            'social': social_data,
            'network': network_data
        }
        
        logger.info("Complete comprehensive dataset generation finished")
        logger.info(f"Generated {len(dataset)} data tables:")
        for table_name, df in dataset.items():
            logger.info(f"  • {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return dataset
    
    def save_sample_data(self, output_dir: str, n_customers: int = 1000,
                        churn_rate: float = 0.26, format: str = 'csv') -> None:
        """
        Generate and save sample data to files.
        
        Args:
            output_dir: Output directory path
            n_customers: Number of customers to generate
            churn_rate: Expected churn rate
            format: Output format ('csv', 'excel', 'parquet')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset
        dataset = self.generate_complete_dataset(n_customers, churn_rate)
        
        # Save each table
        for table_name, df in dataset.items():
            if format == 'csv':
                file_path = output_path / f"{table_name}.csv"
                df.to_csv(file_path, index=False)
            elif format == 'excel':
                file_path = output_path / f"{table_name}.xlsx"
                df.to_excel(file_path, index=False)
            elif format == 'parquet':
                file_path = output_path / f"{table_name}.parquet"
                df.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved {table_name} data to {file_path}")
        
        # Create data dictionary
        self._create_data_dictionary(output_path)
    
    def _create_data_dictionary(self, output_path: Path) -> None:
        """
        Create a data dictionary documenting all fields in the generated datasets.
        
        Args:
            output_path: Directory where to save the data dictionary
        """
        data_dictionary = {
            'customers': {
                'customer_id': 'Unique customer identifier (string)',
                'gender': 'Customer gender (Male/Female, with some data quality issues)',
                'senior_citizen': 'Whether customer is senior citizen (0/1)',
                'partner': 'Whether customer has partner (Yes/No)',
                'dependents': 'Whether customer has dependents (Yes/No)'
            },
            'billing': {
                'customer_id': 'Unique customer identifier (string)',
                'phone_service': 'Whether customer has phone service (Yes/No)',
                'multiple_lines': 'Whether customer has multiple phone lines (Yes/No/No phone service)',
                'internet_service': 'Type of internet service (DSL/Fiber optic/No)',
                'online_security': 'Whether customer has online security (Yes/No/No internet service)',
                'online_backup': 'Whether customer has online backup (Yes/No/No internet service)',
                'device_protection': 'Whether customer has device protection (Yes/No/No internet service)',
                'tech_support': 'Whether customer has tech support (Yes/No/No internet service)',
                'streaming_tv': 'Whether customer has streaming TV (Yes/No/No internet service)',
                'streaming_movies': 'Whether customer has streaming movies (Yes/No/No internet service)',
                'contract': 'Contract type (Month-to-month/One year/Two year, with data quality issues)',
                'paperless_billing': 'Whether customer uses paperless billing (Yes/No)',
                'payment_method': 'Payment method (Electronic check/Mailed check/Bank transfer/Credit card)',
                'tenure': 'Number of months customer has stayed with company (integer)',
                'monthly_charges': 'Monthly charges amount (float, with some noise)',
                'total_charges': 'Total charges amount (float, with missing values)',
                'churn': 'Whether customer churned (0/1) - TARGET VARIABLE'
            },
            'usage': {
                'customer_id': 'Unique customer identifier (string)',
                'monthly_data_gb': 'Monthly data usage in GB (float, with missing values and outliers)',
                'monthly_calls': 'Number of monthly calls (integer)',
                'monthly_sms': 'Number of monthly SMS (integer)',
                'peak_hour_usage_pct': 'Percentage of usage during peak hours (float)',
                'international_calls': 'Number of international calls (integer)',
                'roaming_usage_gb': 'Roaming data usage in GB (float)',
                'video_streaming_gb': 'Video streaming data usage in GB (float)',
                'social_media_gb': 'Social media data usage in GB (float)',
                'gaming_gb': 'Gaming data usage in GB (float)',
                'web_browsing_gb': 'Web browsing data usage in GB (float)',
                'unique_apps_used': 'Number of unique apps used (integer)',
                'avg_session_duration_min': 'Average session duration in minutes (float)'
            },
            'customer_care': {
                'customer_id': 'Unique customer identifier (string)',
                'support_calls_count': 'Number of support calls (integer)',
                'support_chat_count': 'Number of support chat sessions (integer)',
                'support_email_count': 'Number of support emails (integer)',
                'avg_resolution_time_hours': 'Average resolution time in hours (float)',
                'first_call_resolution_rate': 'First call resolution rate (float 0-1)',
                'complaint_count': 'Number of complaints (integer)',
                'primary_complaint_type': 'Primary complaint category (Billing/Network/Service/Technical/None)',
                'customer_satisfaction_score': 'Customer satisfaction score 1-5 (float)',
                'escalation_count': 'Number of escalations (integer)',
                'avg_response_time_min': 'Average response time in minutes (float)'
            },
            'crm': {
                'customer_id': 'Unique customer identifier (string)',
                'customer_segment': 'Customer segment (Premium/Standard/Basic/Enterprise)',
                'lifetime_value': 'Customer lifetime value (float)',
                'acquisition_channel': 'Customer acquisition channel (Online/Store/Referral/Call Center/Partner)',
                'last_login_days_ago': 'Days since last login (float)',
                'app_usage_frequency': 'App usage frequency (Daily/Weekly/Monthly/Rarely)',
                'email_open_rate': 'Email open rate (float 0-1)',
                'sms_response_rate': 'SMS response rate (float 0-1)',
                'promotion_usage_count': 'Number of promotions used (integer)',
                'payment_delays': 'Number of payment delays (integer)',
                'autopay_enabled': 'Whether autopay is enabled (boolean)',
                'referrals_made': 'Number of referrals made (integer)',
                'referred_by_friend': 'Whether referred by friend (boolean)',
                'plan_changes_count': 'Number of plan changes (integer)',
                'service_additions_count': 'Number of service additions (integer)'
            },
            'social': {
                'customer_id': 'Unique customer identifier (string)',
                'has_social_account': 'Whether customer has social media account (boolean)',
                'social_mentions_count': 'Number of social media mentions (integer)',
                'social_sentiment': 'Overall social sentiment (Positive/Neutral/Negative/No Data)',
                'online_reviews_count': 'Number of online reviews (integer)',
                'avg_review_rating': 'Average review rating 1-5 (float, with missing values)',
                'website_visits_monthly': 'Monthly website visits (integer)',
                'mobile_app_rating': 'Mobile app rating 1-5 (float, with missing values)',
                'forum_posts_count': 'Number of forum posts (integer)',
                'community_member': 'Whether customer is community member (boolean)',
                'brand_mentions_positive': 'Number of positive brand mentions (integer)',
                'brand_mentions_negative': 'Number of negative brand mentions (integer)',
                'follower_count': 'Social media follower count (integer)',
                'is_influencer': 'Whether customer is influencer (boolean)'
            },
            'network': {
                'customer_id': 'Unique customer identifier (string)',
                'avg_download_speed_mbps': 'Average download speed in Mbps (float)',
                'avg_upload_speed_mbps': 'Average upload speed in Mbps (float)',
                'network_uptime_pct': 'Network uptime percentage (float)',
                'connection_drops_monthly': 'Monthly connection drops (integer)',
                'avg_signal_strength_db': 'Average signal strength in dB (float)',
                'coverage_rating': 'Coverage quality rating (Excellent/Good/Fair/Poor)',
                'primary_network_type': 'Primary network type (5G/4G LTE/4G/3G)',
                'data_throttling_events': 'Number of data throttling events (integer)',
                'throttling_duration_hours': 'Total throttling duration in hours (float)',
                'roaming_countries_visited': 'Number of roaming countries visited (integer)',
                'roaming_issues_count': 'Number of roaming issues (integer)',
                'peak_hour_slowdown_pct': 'Peak hour slowdown percentage (float)',
                'congestion_events_monthly': 'Monthly network congestion events (integer)',
                'primary_device_type': 'Primary device type (Smartphone/Tablet/IoT Device/Mobile Hotspot)',
                'network_support_tickets': 'Number of network support tickets (integer)',
                'avg_issue_resolution_hours': 'Average issue resolution time in hours (float)',
                'urban_area': 'Whether customer is in urban area (boolean)',
                'tower_distance_km': 'Distance to nearest tower in km (float)'
            }
        }
        
        # Save data dictionary as JSON
        import json
        dict_path = output_path / "data_dictionary.json"
        with open(dict_path, 'w') as f:
            json.dump(data_dictionary, f, indent=2)
        logger.info(f"Saved data dictionary to {dict_path}")
        
        # Create a readable markdown version
        md_path = output_path / "data_dictionary.md"
        with open(md_path, 'w') as f:
            f.write("# Telecoms Customer Churn Dataset - Data Dictionary\n\n")
            f.write("This document describes all fields in the generated telecoms customer churn dataset.\n\n")
            
            for table_name, fields in data_dictionary.items():
                f.write(f"## {table_name.title()} Table\n\n")
                f.write("| Field | Description |\n")
                f.write("|-------|-------------|\n")
                for field_name, description in fields.items():
                    f.write(f"| `{field_name}` | {description} |\n")
                f.write("\n")
            
            f.write("## Data Quality Notes\n\n")
            f.write("- This dataset intentionally includes realistic data quality issues:\n")
            f.write("  - Missing values in some fields\n")
            f.write("  - Categorical data inconsistencies (e.g., 'Male' vs 'M')\n")
            f.write("  - Outliers in numeric fields\n")
            f.write("  - Data entry errors and typos\n")
            f.write("- The `churn` field in the billing table is the target variable for prediction\n")
            f.write("- All tables can be joined using the `customer_id` field\n\n")
            
        logger.info(f"Saved readable data dictionary to {md_path}")
