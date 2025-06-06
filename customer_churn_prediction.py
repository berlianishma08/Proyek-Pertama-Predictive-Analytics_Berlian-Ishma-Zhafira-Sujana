import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# 1. SET YOUR DATASET PATH HERE
DATASET_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data():
    # Try multiple possible locations
    possible_locations = [
        DATASET_PATH
    ]
    
    for path in possible_locations:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            return pd.read_csv(path)
    
    exit(1)

# Load data
df = load_data()

# Data preparation
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))