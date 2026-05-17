# Model Comparison Report — Heart Disease Prediction

**Dataset:** 1,025 patients | **Training:** 820 (80%) | **Testing:** 205 (20%)  
**Scaling:** StandardScaler (fit on training data only)  
**Missing Values:** None | **Target:** Binary (0 = No Disease, 1 = Has Disease)  
**Class Distribution:** 526 with disease (51.3%) · 499 without disease (48.7%)

---

## Final Rankings

| Rank | Model                    | Accuracy | Precision | Recall | F1-Score |
|------|--------------------------|----------|-----------|--------|----------|
| 1    | Random Forest (n=100, depth=5) | 0.9268   | 0.8947    | 0.9714 | 0.9315   |
| 2    | KNN (K=7)                | 0.8439   | 0.8230    | 0.8857 | 0.8532   |
| 3    | Naive Bayes              | 0.8293   | 0.8070    | 0.8762 | 0.8402   |
| 4    | SVM                      | 0.8146   | 0.7638    | 0.9238 | 0.8362   |
| 5    | Logistic Regression      | 0.8098   | 0.7619    | 0.9143 | 0.8312   |

---

## Model-by-Model Breakdown

### 1. Logistic Regression (Baseline)

**Type:** Baseline | **Accuracy: 80.98%**

Draws a linear decision boundary between classes. Simple, fast, and interpretable, but assumes the relationship between features and the target is linear — which is not always the case.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8098 |
| Precision | 0.7619 |
| Recall    | 0.9143 |
| F1-Score  | 0.8312 |

**Confusion Matrix:** TN=70 · FP=30 · FN=9 · TP=96

**Where it fails:** Misses non-linear patterns in features like `oldpeak`, `exang`, and `ca` (the three strongest negative correlators). Patients with complex multi-factor risk profiles are harder for this linear model to classify correctly, which explains the lower Precision — it generates more false positives (30) than the advanced models.

---

### 2. Naive Bayes (Baseline)

**Type:** Baseline | **Accuracy: 82.93%**

Uses Bayes' theorem and assumes all features are statistically independent of each other. Fast and works well with small datasets, but the independence assumption rarely holds in medical data.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8293 |
| Precision | 0.8070 |
| Recall    | 0.8762 |
| F1-Score  | 0.8402 |

**Confusion Matrix:** TN=78 · FP=22 · FN=13 · TP=92

**Where it fails:** The independence assumption breaks down when features are correlated — for example, `age` and `thalach` are naturally linked (older patients typically have lower max heart rates). This leads to overconfident probability estimates and slightly more false negatives (13) than SVM and Logistic Regression.

---

### 3. KNN — K=7 (Advanced)

**Type:** Distance-based | **Accuracy: 84.39%**

Classifies a patient by looking at the 7 nearest patients in the training set and assigning the most common label. Feature scaling is mandatory — KNN uses Euclidean distance, so unscaled features would produce meaningless distance calculations.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8439 |
| Precision | 0.8230 |
| Recall    | 0.8857 |
| F1-Score  | 0.8532 |

**Confusion Matrix:** TN=80 · FP=20 · FN=12 · TP=93

**Notes:** KNN achieves the second-best Accuracy and the highest Precision among all non-ensemble models. Using k=7 (an odd number) avoids tie votes. A limitation of KNN is that it is computationally expensive at prediction time on larger datasets, since it must compute distances to all training points for every new sample.

---

### 4. SVM (Advanced)

**Type:** Margin-based | **Accuracy: 81.46%**

Finds the hyperplane that maximises the margin between the two classes in the scaled feature space. Uses a linear kernel with regularisation parameter C=1.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8146 |
| Precision | 0.7638 |
| Recall    | 0.9238 |
| F1-Score  | 0.8362 |

**Confusion Matrix:** TN=70 · FP=30 · FN=8 · TP=97

**Notes:** SVM achieves the second-highest Recall (0.9238) — it misses only 8 actual disease cases, making it a strong candidate when minimising False Negatives is the priority. However, it generates 30 false positives (matching Logistic Regression), which keeps its Precision lower. Performance could potentially be improved by testing an RBF kernel, but training time would increase.

---

### 5. Random Forest (Advanced — Best Model)

**Type:** Ensemble | **Accuracy: 92.68%**

Builds 100 decision trees on random subsets of the data and features, then combines their votes by majority. The ensemble approach reduces variance and captures complex non-linear feature interactions that the baseline models miss.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9268 |
| Precision | 0.8947 |
| Recall    | 0.9714 |
| F1-Score  | 0.9315 |

**Confusion Matrix:** TN=88 · FP=12 · FN=3 · TP=102

**Notes:** Random Forest is the clear winner — it outperforms all other models on Accuracy, Precision, and F1-Score, and achieves the highest Recall (0.9714), missing only 3 actual disease cases across 205 test patients. The combination of `n_estimators=100` and `max_depth=5` provides strong generalisation without overfitting. The accuracy gap between Random Forest (92.68%) and the next best model KNN (84.39%) is 8.29 percentage points — a substantial improvement that justifies the ensemble approach.

---

## Outlier Summary (IQR Method)

| Feature    | Outliers Detected |
|------------|-------------------|
| age        | 0                 |
| trestbps   | 30                |
| chol       | 16                |
| thalach    | 4                 |
| oldpeak    | 7                 |

Outliers were reported but not removed — clinical measurements may be extreme but still medically valid.

---

## Feature Correlation with Target

| Feature   | Correlation | Direction |
|-----------|-------------|-----------|
| cp        | +0.4349     | Positive  |
| thalach   | +0.4229     | Positive  |
| slope     | +0.3455     | Positive  |
| restecg   | +0.1345     | Positive  |
| fbs       | −0.0412     | Negative  |
| chol      | −0.1000     | Negative  |
| trestbps  | −0.1388     | Negative  |
| age       | −0.2293     | Negative  |
| sex       | −0.2795     | Negative  |
| thal      | −0.3378     | Negative  |
| ca        | −0.3821     | Negative  |
| exang     | −0.4380     | Negative  |
| oldpeak   | −0.4384     | Negative  |

The strongest positive predictors are `cp` (chest pain type) and `thalach` (max heart rate). The strongest negative predictors are `oldpeak` (ST depression) and `exang` (exercise-induced angina) — higher values of these indicate reduced cardiovascular function and strongly associate with disease presence in the binarised target.

---

## Metric Definitions

| Metric    | Formula                             | What it tells you                                      |
|-----------|-------------------------------------|--------------------------------------------------------|
| Accuracy  | (TP + TN) / Total                   | Overall correct predictions                            |
| Precision | TP / (TP + FP)                      | Of predicted disease cases, how many are actually sick |
| Recall    | TP / (TP + FN)                      | Of actual disease cases, how many were caught          |
| F1-Score  | 2 × (Precision × Recall) / (P + R)  | Harmonic mean — balances Precision and Recall          |

**In medical diagnosis, Recall is especially important** — a missed disease case (False Negative) is more dangerous than a false alarm (False Positive).

---

## Key Takeaways

- **Random Forest** is the best overall model with 92.68% accuracy, the highest F1-Score (0.9315), and the highest Recall (0.9714) — missing only 3 disease cases out of 105.
- **KNN (k=7)** is the second-best model at 84.39% accuracy and the most precise among non-ensemble models (0.8230 Precision).
- **SVM** achieves the second-highest Recall (0.9238), making it worth considering in scenarios where catching every case matters more than overall accuracy.
- **Baseline models** (Logistic Regression and Naive Bayes) perform at ~81–83%, providing a reasonable floor given the dataset's moderate size (1,025 patients).
- The dataset is well-balanced (51.3% / 48.7% split), so Accuracy is a reliable metric here — no class imbalance skewing the results.
- All five models correctly identify the majority of disease cases (Recall ≥ 0.876), confirming the dataset has strong predictive signal.
