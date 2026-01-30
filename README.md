# AI Email-Campaign Lead Scorer
### Smart Email Targeting and Conversion Engine v1.0

An AI-powered lead scoring system designed to filter and prioritize email marketing lists. This engine analyzes demographic data and user behavior to predict which email leads are most likely to convert, enabling businesses to target high-value prospects and minimize email bounce/spam rates.

## Core Functionality
- Lead Prioritization: Automatically ranks email leads from highest to lowest conversion probability.
- Smart Targeting: Identifies high-potential users based on historical engagement patterns.
- Bulk Processing: Filter entire CSV email lists to discover "Gold Leads" instantly.

## Model Performance
- Overall Classification Accuracy: 87.75%
- Algorithm: XGBoost Classifier
- Key Predictive Drivers: 
  1. Device Type (52.8%)
  2. Payment Method (22.6%)
  3. Ad Test Group (16.8%)

## Tech Stack
- Backend: Python 3.12
- ML Framework: XGBoost, Scikit-Learn
- Interface: Streamlit (Email Simulator and Bulk List Processor)
- Serialization: Joblib / Pickle

## Project Structure
- app.py: The Lead Scoring dashboard.
- model_training.py: Pipeline for training the email conversion model.
- user_demographics_model.pkl: The trained AI classifier.
- label_encoder_dict.pkl: Encoders for processing user attributes.
- user_behavior_data.csv: Dataset used for training and validation.

## How to Run
1. Install requirements:
   pip install streamlit pandas xgboost scikit-learn

2. Run the engine:
   streamlit run app.py

---
2026 AI Marketing Intelligence | Engineered by Mohammed.
