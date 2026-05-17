import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import warnings
warnings.filterwarnings('ignore')


# =============================================================
# SECTION 1 — LOAD DATASET
# =============================================================
print("=" * 60)
print("  UCI Heart Disease Dataset — Cleveland Clinic")
# UPDATED: Reading from the file created in the previous step
print("  Loading from: heart_disease_for_excel.csv")
print("=" * 60)

# Use skiprows=1 to ignore the 'sep=,' line at the top of the file
df = pd.read_csv('heart_disease_for_excel.csv', skiprows=1)

# The original UCI target can have values 0-4.
# 0 = no disease, 1-4 = disease present → binarise to 0/1
df['target'] = (df['target'] > 0).astype(int)

print(f"\nDataset loaded successfully!")
print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Target : 0 = No Disease, 1 = Has Disease")
print(f"\n  Patients WITH heart disease    : {(df['target'] == 1).sum()}")
print(f"  Patients WITHOUT heart disease : {(df['target'] == 0).sum()}")
print(f"\nFirst 5 rows:")
print(df.head())


# =============================================================
# SECTION 2 — DATA PREPROCESSING & EDA
# =============================================================
print("\n" + "=" * 60)
print("  SECTION 2 — DATA PREPROCESSING & EDA")
print("=" * 60)

# --- Missing Values ---
print("\n--- Handling Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  No missing values found.")
if missing.sum() > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("  Filled missing values with column median.")

# --- Outlier Detection (IQR Method) ---
print("\n--- Outlier Detection (IQR Method) ---")
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    print(f"  {col:<12}: {n_out} outlier(s) detected")

# --- Feature Encoding ---
print("\n--- Feature Encoding ---")
print("  All features are already numeric.")

# --- EDA: Basic Statistics ---
print("\n--- EDA: Dataset Statistics ---")
print(df.describe().round(2))

# --- EDA: Feature Correlation with Target ---
print("\n--- EDA: Feature Correlation with Target ---")
corr = df.corr(numeric_only=True)['target'].drop('target').sort_values(ascending=False)
print(corr.round(4))


# =============================================================
# SECTION 3 — TRAIN / TEST SPLIT (80% | 20%)
# =============================================================
print("\n" + "=" * 60)
print("  SECTION 3 — TRAIN / TEST SPLIT")
print("=" * 60)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Total samples : {len(df)}")
print(f"  Training samples : {X_train.shape[0]} (80%)")
print(f"  Testing samples : {X_test.shape[0]} (20%)")


# =============================================================
# SECTION 4 — FEATURE SCALING
# =============================================================
print("\n--- Feature Scaling (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("  StandardScaler applied.")


# =============================================================
# SECTION 5 — EVALUATION HELPER
# =============================================================
results = []

def evaluate_model(name, y_test, y_pred):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    
    return {'Model': name, 'Accuracy': round(acc,4), 'Precision': round(prec,4),
            'Recall': round(rec,4), 'F1-Score': round(f1,4)}


# =============================================================
# SECTION 6 — BASELINE MODELS
# =============================================================
print("\n" + "=" * 60)
print("  SECTION 6 — BASELINE MODELS")
print("=" * 60)

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
results.append(evaluate_model('Logistic Regression', y_test, lr_model.predict(X_test_scaled)))

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
results.append(evaluate_model('Naive Bayes', y_test, nb_model.predict(X_test_scaled)))


# =============================================================
# SECTION 7 — ADVANCED MODELS
# =============================================================
print("\n" + "=" * 60)
print("  SECTION 7 — ADVANCED MODELS")
print("=" * 60)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)
results.append(evaluate_model('KNN', y_test, knn_model.predict(X_test_scaled)))

# SVM
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train_scaled, y_train)
results.append(evaluate_model('SVM', y_test, svm_model.predict(X_test_scaled)))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)
results.append(evaluate_model('Random Forest', y_test, rf_model.predict(X_test_scaled)))


# =============================================================
# SECTION 8 — FINAL COMPARATIVE REPORT
# =============================================================
comparison_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

print("\n\n" + "=" * 72)
print("                    FINAL MODEL COMPARISON REPORT")
print("=" * 72)
print(comparison_df.to_string(index=False))
print("=" * 72)

best_name = comparison_df.iloc[0]['Model']
print(f"\n  Best Model : {best_name}")
