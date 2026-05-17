
# 📊 Final Model Comparison Report

**Dataset:** UCI Heart Disease (1,025 patients) | **Training:** 820 (80%) | **Testing:** 205 (20%)

**Scaling:** StandardScaler (fit on training data only)

---

## Final Rankings

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- |
| **1** | **Random Forest (Best)** | **0.9268** | **0.8947** | **0.9714** | **0.9315** |
| **2** | KNN (K=7) | 0.8439 | 0.8230 | 0.8857 | 0.8532 |
| **3** | Naive Bayes | 0.8293 | 0.8070 | 0.8762 | 0.8402 |
| **4** | SVM (Linear Kernel) | 0.8146 | 0.7638 | 0.9238 | 0.8362 |
| **5** | Logistic Regression | 0.8098 | 0.7619 | 0.9143 | 0.8312 |

---

## Model-by-Model Breakdown

### 1. Logistic Regression (Baseline)

**Type:** Baseline | **Accuracy: 80.98%** Draws a linear decision boundary between classes. While fast and interpretable, it assumes the relationship between features and the target is linear.

* **Strengths:** Easy to explain; shows the "weight" of each feature.
* **Weaknesses:** Misses non-linear interactions between risk factors like age, max heart rate (`thalach`), and ST depression (`oldpeak`).

---

### 2. Naive Bayes (Baseline)

**Type:** Baseline | **Accuracy: 82.93%** Uses Bayes' theorem assuming all features are independent.

* **Observation:** Performed surprisingly well for a baseline, achieving a balanced F1-score of 0.8402. It is highly efficient for medical screening but can be "overconfident" if features are correlated (like age and blood pressure).

---

### 3. KNN — K=7 (Advanced)

**Type:** Distance-based | **Accuracy: 84.39%** Classifies a patient by looking at the 7 most similar patients in the data.

* **Best K found:** 7.
* **Notes:** KNN is highly dependent on `StandardScaler`. Without scaling, features with large numbers (like Cholesterol) would drown out smaller features (like `oldpeak`), leading to poor accuracy.

---

### 4. SVM (Advanced)

**Type:** Margin-based | **Accuracy: 81.46%** Finds the "widest road" (hyperplane) that separates healthy patients from sick ones.

* **Observation:** Achieved a high **Recall (0.9238)**, meaning it is very sensitive at catching disease, though it had more "false alarms" (lower precision) than the Random Forest.

---

### 5. Random Forest (Best Model)

**Type:** Ensemble | **Accuracy: 92.68%** The clear winner. It combines 100 decision trees and takes a "majority vote" for the final diagnosis.

* **Why it won:** It captures complex, non-linear patterns that other models miss.
* **Clinical Value:** It achieved the highest **Recall (0.9714)**. In a medical context, this is critical because it means the model only missed ~3% of actual heart disease cases.

---

## Metric Definitions (Oral Exam Cheat Sheet)

| Metric | What it tells the Doctor | Importance |
| --- | --- | --- |
| **Accuracy** | Overall, how many patients were correctly diagnosed? | Good for a general overview. |
| **Precision** | If the model says "Disease," how sure are we the patient is actually sick? | Reduces "False Alarms" and unnecessary stress. |
| **Recall** | Of all the sick patients, how many did we actually catch? | **CRITICAL:** High recall ensures we don't send a sick person home (False Negative). |
| **F1-Score** | The balance between Precision and Recall. | The best metric if you want a "middle ground" model. |

---

## Key Takeaways for Presentation

1. **Ensemble Power:** Random Forest outperformed simple models by 10% because it handles the variety of clinical features (binary, ordinal, and continuous) much better.
2. **Recall is King:** In heart disease prediction, a **False Negative** (missing a sick patient) is much more dangerous than a **False Positive** (a false alarm). Our best model (Random Forest) has a 97% Recall rate.
3. **Data Quality:** Using `StandardScaler` was essential to ensure all clinical measurements were evaluated on the same scale (0 to 1).
