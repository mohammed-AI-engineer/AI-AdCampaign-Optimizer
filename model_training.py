import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# 1. Path Configuration
# Using relative paths for better portability on GitHub and remote servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "user_behavior_data.csv")

def run_performance_analysis():
    # 2. Data Loading
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
    
    data = pd.read_csv(DATA_PATH)

    # 3. Data Cleaning
    # Filtering records to maintain high-quality data standards
    if 'Email' in data.columns:
        data = data[data['Email'] != "Not Available"]

    # 4. Feature Encoding
    # Converting categorical data into numerical format for XGBoost optimization
    categorical_cols = ['Gender', 'City', 'Device_Type', 'Payment_Method', 'Test_Group', 'Campaign_Result']
    le_dict = {}
    
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le

    # 5. Dataset Preparation
    feature_cols = ['Age', 'Gender', 'City', 'Device_Type', 'Payment_Method', 'Test_Group']
    X = data[feature_cols]
    y = data['Campaign_Result']

    # 6. Class Imbalance Correction (SMOTE)
    # Applying Synthetic Minority Over-sampling to balance the target classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 7. Training and Validation Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # 8. Model Training
    # Initializing XGBoost with high-capacity hyperparameters
    clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=1,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # 9. Evaluation Metrics
    y_pred = clf.predict(X_test)
    
    print("-" * 50)
    print(f"Model Training Status: Completed")
    print(f"Overall Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("-" * 50)

    # Classification Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=le_dict['Campaign_Result'].classes_,
        columns=le_dict['Campaign_Result'].classes_
    )
    print("\nConfusion Matrix Breakdown:")
    print(cm_df)

    # Feature Importance Score
    # Ranking features based on their predictive power
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importances (Predictive Power):")
    print(importances)

    print("\nNote: Process finished. No external model files were generated.")

if __name__ == "__main__":
    run_performance_analysis()