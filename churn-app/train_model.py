import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ── 1. Load Data ──────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv(r'../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"  Shape: {data.shape}")

# ── 2. Clean Data ─────────────────────────────────────────────
print("Cleaning data...")

# Fix TotalCharges
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(subset=['TotalCharges'], inplace=True)
data.reset_index(drop=True, inplace=True)

# Drop customerID
data.drop(columns=['customerID'], inplace=True)

# Encode target
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# ── 3. Encode Categorical Features ───────────────────────────
print("Encoding features...")

categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# ── 4. Split ─────────────────────────────────────────────────
print("Splitting data...")

X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Scale (fit on train only) ─────────────────────────────
print("Scaling numerical features...")

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ── 6. Train Model ───────────────────────────────────────────
print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 7. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ── 8. Save Artifacts ────────────────────────────────────────
os.makedirs('model', exist_ok=True)

with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model/feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)

print("\nModel artifacts saved to ./model/")
print("  - churn_model.pkl")
print("  - scaler.pkl")
print("  - feature_columns.pkl")
print("\nDone! You can now run: uvicorn main:app --reload")
