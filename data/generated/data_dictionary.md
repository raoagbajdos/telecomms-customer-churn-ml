# Telecoms Customer Churn Dataset - Data Dictionary

This document describes all fields in the generated telecoms customer churn dataset.

## Customers Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `gender` | Customer gender (Male/Female, with some data quality issues) |
| `senior_citizen` | Whether customer is senior citizen (0/1) |
| `partner` | Whether customer has partner (Yes/No) |
| `dependents` | Whether customer has dependents (Yes/No) |

## Billing Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `phone_service` | Whether customer has phone service (Yes/No) |
| `multiple_lines` | Whether customer has multiple phone lines (Yes/No/No phone service) |
| `internet_service` | Type of internet service (DSL/Fiber optic/No) |
| `online_security` | Whether customer has online security (Yes/No/No internet service) |
| `online_backup` | Whether customer has online backup (Yes/No/No internet service) |
| `device_protection` | Whether customer has device protection (Yes/No/No internet service) |
| `tech_support` | Whether customer has tech support (Yes/No/No internet service) |
| `streaming_tv` | Whether customer has streaming TV (Yes/No/No internet service) |
| `streaming_movies` | Whether customer has streaming movies (Yes/No/No internet service) |
| `contract` | Contract type (Month-to-month/One year/Two year, with data quality issues) |
| `paperless_billing` | Whether customer uses paperless billing (Yes/No) |
| `payment_method` | Payment method (Electronic check/Mailed check/Bank transfer/Credit card) |
| `tenure` | Number of months customer has stayed with company (integer) |
| `monthly_charges` | Monthly charges amount (float, with some noise) |
| `total_charges` | Total charges amount (float, with missing values) |
| `churn` | Whether customer churned (0/1) - TARGET VARIABLE |

## Usage Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `monthly_data_gb` | Monthly data usage in GB (float, with missing values and outliers) |
| `monthly_calls` | Number of monthly calls (integer) |
| `monthly_sms` | Number of monthly SMS (integer) |
| `peak_hour_usage_pct` | Percentage of usage during peak hours (float) |
| `international_calls` | Number of international calls (integer) |
| `roaming_usage_gb` | Roaming data usage in GB (float) |
| `video_streaming_gb` | Video streaming data usage in GB (float) |
| `social_media_gb` | Social media data usage in GB (float) |
| `gaming_gb` | Gaming data usage in GB (float) |
| `web_browsing_gb` | Web browsing data usage in GB (float) |
| `unique_apps_used` | Number of unique apps used (integer) |
| `avg_session_duration_min` | Average session duration in minutes (float) |

## Customer_Care Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `support_calls_count` | Number of support calls (integer) |
| `support_chat_count` | Number of support chat sessions (integer) |
| `support_email_count` | Number of support emails (integer) |
| `avg_resolution_time_hours` | Average resolution time in hours (float) |
| `first_call_resolution_rate` | First call resolution rate (float 0-1) |
| `complaint_count` | Number of complaints (integer) |
| `primary_complaint_type` | Primary complaint category (Billing/Network/Service/Technical/None) |
| `customer_satisfaction_score` | Customer satisfaction score 1-5 (float) |
| `escalation_count` | Number of escalations (integer) |
| `avg_response_time_min` | Average response time in minutes (float) |

## Crm Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `customer_segment` | Customer segment (Premium/Standard/Basic/Enterprise) |
| `lifetime_value` | Customer lifetime value (float) |
| `acquisition_channel` | Customer acquisition channel (Online/Store/Referral/Call Center/Partner) |
| `last_login_days_ago` | Days since last login (float) |
| `app_usage_frequency` | App usage frequency (Daily/Weekly/Monthly/Rarely) |
| `email_open_rate` | Email open rate (float 0-1) |
| `sms_response_rate` | SMS response rate (float 0-1) |
| `promotion_usage_count` | Number of promotions used (integer) |
| `payment_delays` | Number of payment delays (integer) |
| `autopay_enabled` | Whether autopay is enabled (boolean) |
| `referrals_made` | Number of referrals made (integer) |
| `referred_by_friend` | Whether referred by friend (boolean) |
| `plan_changes_count` | Number of plan changes (integer) |
| `service_additions_count` | Number of service additions (integer) |

## Social Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `has_social_account` | Whether customer has social media account (boolean) |
| `social_mentions_count` | Number of social media mentions (integer) |
| `social_sentiment` | Overall social sentiment (Positive/Neutral/Negative/No Data) |
| `online_reviews_count` | Number of online reviews (integer) |
| `avg_review_rating` | Average review rating 1-5 (float, with missing values) |
| `website_visits_monthly` | Monthly website visits (integer) |
| `mobile_app_rating` | Mobile app rating 1-5 (float, with missing values) |
| `forum_posts_count` | Number of forum posts (integer) |
| `community_member` | Whether customer is community member (boolean) |
| `brand_mentions_positive` | Number of positive brand mentions (integer) |
| `brand_mentions_negative` | Number of negative brand mentions (integer) |
| `follower_count` | Social media follower count (integer) |
| `is_influencer` | Whether customer is influencer (boolean) |

## Network Table

| Field | Description |
|-------|-------------|
| `customer_id` | Unique customer identifier (string) |
| `avg_download_speed_mbps` | Average download speed in Mbps (float) |
| `avg_upload_speed_mbps` | Average upload speed in Mbps (float) |
| `network_uptime_pct` | Network uptime percentage (float) |
| `connection_drops_monthly` | Monthly connection drops (integer) |
| `avg_signal_strength_db` | Average signal strength in dB (float) |
| `coverage_rating` | Coverage quality rating (Excellent/Good/Fair/Poor) |
| `primary_network_type` | Primary network type (5G/4G LTE/4G/3G) |
| `data_throttling_events` | Number of data throttling events (integer) |
| `throttling_duration_hours` | Total throttling duration in hours (float) |
| `roaming_countries_visited` | Number of roaming countries visited (integer) |
| `roaming_issues_count` | Number of roaming issues (integer) |
| `peak_hour_slowdown_pct` | Peak hour slowdown percentage (float) |
| `congestion_events_monthly` | Monthly network congestion events (integer) |
| `primary_device_type` | Primary device type (Smartphone/Tablet/IoT Device/Mobile Hotspot) |
| `network_support_tickets` | Number of network support tickets (integer) |
| `avg_issue_resolution_hours` | Average issue resolution time in hours (float) |
| `urban_area` | Whether customer is in urban area (boolean) |
| `tower_distance_km` | Distance to nearest tower in km (float) |

## Data Quality Notes

- This dataset intentionally includes realistic data quality issues:
  - Missing values in some fields
  - Categorical data inconsistencies (e.g., 'Male' vs 'M')
  - Outliers in numeric fields
  - Data entry errors and typos
- The `churn` field in the billing table is the target variable for prediction
- All tables can be joined using the `customer_id` field

