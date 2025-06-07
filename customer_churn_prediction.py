import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Konversi TotalCharges ke float dan isi nilai NaN dengan median
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 3. Drop kolom customerID karena tidak informatif
df.drop(columns=['customerID'], inplace=True)

# 4. Encoding target kolom Churn
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 5. One-hot encoding untuk kolom kategorikal
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaymentMethod', 'PaperlessBilling']
df = pd.get_dummies(df, columns=categorical_cols)

# 6. Definisikan fitur dan target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Bangun model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Evaluasi model
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
