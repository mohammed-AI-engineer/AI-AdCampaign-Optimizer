# AI Ad-Campaign Optimizer
### High-Conversion Predictive Engine v1.0

An AI-driven system designed to process marketing leads and discover high-conversion opportunities. This project uses Machine Learning to predict whether a lead will "Succeed" or "Fail" based on user demographics and behavior.

## Model Performance
- Overall Accuracy: 87.75%
- Algorithm: XGBoost Classifier
- Key Predictive Drivers: 
  1. Device Type (52.8%)
  2. Payment Method (22.6%)
  3. Test Group (16.8%)

## Tech Stack
- Backend: Python 3.12
- ML Framework: XGBoost, Scikit-Learn
- Interface: Streamlit (Manual Simulator and Bulk Processor)
- Serialization: Joblib / Pickle

## Project Structure
- app.py: The interactive Streamlit dashboard.
- model_training.py: The training pipeline logic.
- user_demographics_model.pkl: The trained AI brain.
- label_encoder_dict.pkl: Categorical data encoders.
- user_behavior_data.csv: Historical campaign data.

## How to Run
1. Install requirements:
   pip install streamlit pandas xgboost scikit-learn

2. Run the app:
   streamlit run app.py

---
2026 AI Marketing Intelligence | Engineered by Mohammed.