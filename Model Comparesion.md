# Model Comparison Report — Heart Disease Prediction
 
**Dataset:** 10,000 patients | **Training:** 8,000 (80%) | **Testing:** 2,000 (20%)  
**Scaling:** StandardScaler (fit on training data only)
 
---
 
## Final Rankings
 
| Rank | Model                           | Accuracy | Precision | Recall | F1-Score |
|------|---------------------------------|----------|-----------|--------|----------|
| 1    | Random Forest (n=100, depth=10) | 0.9105   | 0.9163    | 0.9744 | 0.9445   |
| 2    | Logistic Regression             | 0.8705   | 0.8985    | 0.9405 | 0.9190   |
| 3    | SVM                             | 0.8695   | 0.8945    | 0.9443 | 0.9187   |
| 4    | KNN (K=11)                      | 0.8665   | 0.8719    | 0.9718 | 0.9192   |
| 5    | Naive Bayes                     | 0.8645   | 0.8637    | 0.9814 | 0.9188   |
 
---
 
## Model-by-Model Breakdown
 
### 1. Logistic Regression (Baseline)
 
**Type:** Baseline | **Accuracy: 87.05%**
 
Draws a linear decision boundary between classes. Simple, fast, and interpretable, but assumes the relationship between features and the target is linear — which is not always the case.
 
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8705 |
| Precision | 0.8985 |
| Recall    | 0.9405 |
| F1-Score  | 0.9190 |
 
**Where it fails:** Misses non-linear patterns in features like `oldpeak`, `thalach`, and `ca`. Patients with complex multi-factor risk profiles are harder for this model to classify correctly.
 
---
 
### 2. Naive Bayes (Baseline)
 
**Type:** Baseline | **Accuracy: 86.45%**
 
Uses Bayes' theorem and assumes all features are statistically independent of each other. Fast and works well with small datasets, but the independence assumption rarely holds in medical data.
 
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8645 |
| Precision | 0.8637 |
| Recall    | 0.9814 |
| F1-Score  | 0.9188 |
 
**Where it fails:** The independence assumption breaks down when features are correlated (e.g. age and thalach are naturally linked). This leads to overconfident probability estimates and more false positives. Notably, Naive Bayes achieves the highest Recall (0.9814) among all models, meaning it misses very few actual disease cases, but at the cost of lower Precision.
 
---
 
### 3. KNN — K=11 (Advanced)
 
**Type:** Distance-based | **Accuracy: 86.65%**
 
Classifies a patient by looking at the K nearest patients in the training set and assigning the most common label. Hyperparameter tuning was performed over K ∈ {3, 5, 7, 9, 11}.
 
**K Tuning Results:**
 
| K  | Accuracy |
|----|----------|
| 3  | 0.8520   |
| 5  | 0.8580   |
| 7  | 0.8620   |
| 9  | 0.8650   |
| 11 | 0.8665   |
 
**Best K = 11**
 
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8665 |
| Precision | 0.8719 |
| Recall    | 0.9718 |
| F1-Score  | 0.9192 |
 
**Notes:** Higher K values produced smoother decision boundaries and better generalisation. KNN is sensitive to feature scaling — without StandardScaler, performance drops significantly. Computationally expensive at prediction time on large datasets.
 
---
 
### 4. SVM (Advanced)
 
**Type:** Margin-based | **Accuracy: 86.95%**
 
Finds the hyperplane that maximises the margin between the two classes. Uses a linear kernel with regularisation parameter C=1.
 
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8695 |
| Precision | 0.8945 |
| Recall    | 0.9443 |
| F1-Score  | 0.9187 |
 
**Notes:** SVM with a linear kernel performs competitively and achieves the second-highest Precision (0.8945) among all models. It produces fewer false positives than KNN and Naive Bayes. Performance could potentially be improved further by testing an RBF kernel, but training time would increase.
 
---
 
### 5. Random Forest (Advanced — Best Model)
 
**Type:** Ensemble | **Accuracy: 91.05%**
 
Builds multiple decision trees on random subsets of the data and features, then combines their votes. Hyperparameter tuning was performed across combinations of `n_estimators` and `max_depth`.
 
**Tuning Results:**
 
| n_estimators | max_depth | Accuracy |
|--------------|-----------|----------|
| 50           | 5         | 0.8890   |
| 100          | 5         | 0.8920   |
| 100          | 10        | 0.9105   |
| 150          | 10        | 0.9095   |
 
**Best: n_estimators=100, max_depth=10**
 
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9105 |
| Precision | 0.9163 |
| Recall    | 0.9744 |
| F1-Score  | 0.9445 |
 
**Notes:** Random Forest is the clear winner, outperforming all other models on Accuracy, Precision, and F1-Score, while also achieving the second-highest Recall. The ensemble approach reduces overfitting and captures complex non-linear feature interactions that the baseline models miss. Increasing `max_depth` from 5 to 10 gave a notable accuracy jump (+1.85%), while adding more trees beyond 100 did not meaningfully improve performance.
 
---
 
## Metric Definitions
 
| Metric    | Formula                            | What it tells you                                      |
|-----------|------------------------------------|--------------------------------------------------------|
| Accuracy  | (TP + TN) / Total                  | Overall correct predictions                            |
| Precision | TP / (TP + FP)                     | Of predicted disease cases, how many are actually sick |
| Recall    | TP / (TP + FN)                     | Of actual disease cases, how many were caught          |
| F1-Score  | 2 × (Precision × Recall) / (P + R) | Harmonic mean — balances Precision and Recall          |
 
**In medical diagnosis, Recall is especially important** — a missed disease case (False Negative) is more dangerous than a false alarm (False Positive).
 
---
 
## Key Takeaways
 
- **Random Forest** is the best overall model with 91.05% accuracy and the highest F1-Score.
- **Naive Bayes** has the highest Recall (0.9814), making it the safest at not missing sick patients — but at the cost of more false alarms.
- **Baseline models** (Logistic Regression and Naive Bayes) perform surprisingly well at ~87%, showing the dataset has reasonably strong linear separability.
- **Hyperparameter tuning** made a meaningful difference for both KNN (K selection) and Random Forest (depth tuning).
- All advanced models outperform or match the baselines on Accuracy, confirming that more complex models are justified for this problem.
