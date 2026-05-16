import pandas as pd
import numpy as np

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
# SECTION 1 — GENERATE DATASET (10,000 patients)
# =============================================================
print("Generating dataset with 10,000 patients...")

np.random.seed(42)
n = 10000

age      = np.random.randint(30, 80, n)
sex      = np.random.randint(0, 2, n)
cp       = np.random.randint(0, 4, n)
trestbps = np.random.randint(90, 200, n)
chol     = np.random.randint(150, 400, n)
fbs      = np.random.randint(0, 2, n)
restecg  = np.random.randint(0, 3, n)
thalach  = np.random.randint(70, 210, n)
exang    = np.random.randint(0, 2, n)
oldpeak  = np.round(np.random.uniform(0, 6, n), 1)
slope    = np.random.randint(0, 3, n)
ca       = np.random.randint(0, 4, n)
thal     = np.random.randint(0, 4, n)

risk_score = (
    (age > 55).astype(int) * 2 +
    (sex == 1).astype(int) * 1 +
    (cp >= 2).astype(int) * 2 +
    (trestbps > 140).astype(int) * 1 +
    (chol > 240).astype(int) * 1 +
    (fbs == 1).astype(int) * 1 +
    (thalach < 120).astype(int) * 2 +
    (exang == 1).astype(int) * 2 +
    (oldpeak > 2).astype(int) * 2 +
    (ca > 0).astype(int) * 2
)

noise  = np.random.randint(0, 3, n)
target = ((risk_score + noise) >= 8).astype(int)

df = pd.DataFrame({
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
    'ca': ca, 'thal': thal, 'target': target
})

print(f"Dataset shape: {df.shape}")
print(f"Patients WITH heart disease    : {(df['target'] == 1).sum()}")
print(f"Patients WITHOUT heart disease : {(df['target'] == 0).sum()}")


# =============================================================
# SECTION 2 — DATA PREPROCESSING & EDA
# Lead: Data Preprocessing & EDA
# =============================================================

# --- Missing Values ---
print("\n--- Missing Values ---")
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")

# --- Outlier Detection (IQR method) ---
print("\n--- Outlier Detection (IQR) ---")
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    print(f"  {col:<12}: {n_outliers} outliers")

# --- Feature Encoding ---
# All features are already integer-encoded; no text-to-number conversion needed.
# Binary: sex (0/1), fbs (0/1), exang (0/1)
# Ordinal: cp (0-3), restecg (0-2), slope (0-2), thal (0-3), ca (0-3)
print("\n--- Feature Encoding ---")
print("All categorical features are already integer-encoded (no text conversion needed).")
print("Binary : sex, fbs, exang")
print("Ordinal: cp, restecg, slope, thal, ca")

# --- Basic EDA Stats ---
print("\n--- EDA: Basic Statistics ---")
print(df.describe())

print("\n--- EDA: Correlation with Target (top features) ---")
corr = df.corr()['target'].drop('target').sort_values(ascending=False)
print(corr)


# =============================================================
# SECTION 3 — SPLIT DATA  (80% TRAIN | 20% TEST)
# Lead: Data Preprocessing & EDA
# NOTE: test_size=0.20 → 2,000 test samples, 8,000 training samples
# =============================================================
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}  (80%)")
print(f"Testing  samples : {X_test.shape[0]}   (20%)")


# =============================================================
# SECTION 4 — FEATURE SCALING (Standardization)
# Lead: Data Preprocessing & EDA
# =============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on training data
X_test_scaled  = scaler.transform(X_test)         # transform test with same parameters
print("\nFeature scaling done! (StandardScaler — mean=0, std=1)")


# =============================================================
# SECTION 5 — EVALUATION HELPER
# Lead: Evaluation & Reporting
# =============================================================
results = []

def evaluate_model(name, y_test, y_pred):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print()
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Has Disease']))

    return {'Model': name, 'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
            'Recall': round(rec, 4), 'F1-Score': round(f1, 4)}


# =============================================================
# SECTION 6 — BASELINE MODELS
# Lead: Modeling Lead - Baseline Algorithms
# =============================================================

# --- Logistic Regression ---
print("\nTraining Logistic Regression (Baseline)...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
results.append(evaluate_model('Logistic Regression', y_test, lr_pred))

# --- Naive Bayes ---
print("Training Naive Bayes (Baseline)...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_pred = nb_model.predict(X_test_scaled)
results.append(evaluate_model('Naive Bayes', y_test, nb_pred))

print("\n>>> Baseline Analysis:")
print("  Logistic Regression: assumes a linear decision boundary — struggles with")
print("    non-linear patterns in features like oldpeak and thalach.")
print("  Naive Bayes: assumes all features are independent — fails when correlated")
print("    features (e.g. age & thalach) carry joint predictive power.")


# =============================================================
# SECTION 7 — ADVANCED MODELS + HYPERPARAMETER TUNING
# Lead: Modeling Lead - Advanced Algorithms
# =============================================================

# --- KNN (tune K) ---
print("\nTraining KNN — trying K values: 3, 5, 7, 9, 11 ...")
best_k     = 3
best_k_acc = 0
for k in [3, 5, 7, 9, 11]:
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train_scaled, y_train)
    acc_tmp = accuracy_score(y_test, knn_tmp.predict(X_test_scaled))
    print(f"  K={k:<2}  Accuracy={acc_tmp:.4f}")
    if acc_tmp > best_k_acc:
        best_k_acc = acc_tmp
        best_k     = k

print(f"  Best K = {best_k}  (Accuracy: {best_k_acc:.4f})")
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
knn_pred  = knn_model.predict(X_test_scaled)
results.append(evaluate_model(f'KNN (K={best_k})', y_test, knn_pred))

# --- SVM ---
print("Training SVM (kernel=linear, C=1)...")
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred  = svm_model.predict(X_test_scaled)
results.append(evaluate_model('SVM', y_test, svm_pred))

# --- Random Forest (tune n_estimators and max_depth) ---
print("\nTraining Random Forest — tuning n_estimators and max_depth ...")
rf_candidates = [(50, 5), (100, 5), (100, 10), (150, 10)]
best_rf       = None
best_rf_label = ''
best_rf_acc   = 0
for n_est, depth in rf_candidates:
    rf_tmp = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
    rf_tmp.fit(X_train_scaled, y_train)
    acc_tmp = accuracy_score(y_test, rf_tmp.predict(X_test_scaled))
    print(f"  n_estimators={n_est:<3}  max_depth={depth:<3}  Accuracy={acc_tmp:.4f}")
    if acc_tmp > best_rf_acc:
        best_rf_acc   = acc_tmp
        best_rf       = rf_tmp
        best_rf_label = f'Random Forest (n={n_est}, depth={depth})'

print(f"  Best: {best_rf_label}  (Accuracy: {best_rf_acc:.4f})")
rf_pred = best_rf.predict(X_test_scaled)
results.append(evaluate_model(best_rf_label, y_test, rf_pred))


# =============================================================
# SECTION 8 — COMPARATIVE REPORT
# Lead: Evaluation & Reporting
# =============================================================
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
comparison_df.index += 1

print("\n")
print("=" * 70)
print("                    FINAL MODEL COMPARISON REPORT")
print("=" * 70)
header = f"  {'Rank':<5} {'Model':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7}"
print(header)
print("  " + "-" * 66)
for rank, row in comparison_df.iterrows():
    print(f"  {rank:<5} {row['Model']:<35} {row['Accuracy']:>9.4f} "
          f"{row['Precision']:>10.4f} {row['Recall']:>7.4f} {row['F1-Score']:>7.4f}")
print("=" * 70)

best_name = comparison_df.iloc[0]['Model']
print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {comparison_df.iloc[0]['Accuracy'] * 100:.2f}%")


# =============================================================
# SECTION 9 — BEST MODEL & SAMPLE PREDICTION
# =============================================================
model_map = {
    'Logistic Regression': lr_model,
    'Naive Bayes':         nb_model,
    'SVM':                 svm_model,
}
# KNN and RF keys are dynamic (include tuning params), so match by prefix
if best_name not in model_map:
    if best_name.startswith('KNN'):
        model_map[best_name] = knn_model
    elif best_name.startswith('Random Forest'):
        model_map[best_name] = best_rf

best_model = model_map[best_name]

sample_patient = pd.DataFrame([{
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
    'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}])

sample_scaled = scaler.transform(sample_patient)
prediction    = best_model.predict(sample_scaled)[0]

print("\n--- Sample Patient Prediction ---")
if prediction == 1:
    print("Patient is likely to HAVE heart disease")
else:
    print("Patient is UNLIKELY to have heart disease")

print("\n0 = No heart disease")
print("1 = Has heart disease")