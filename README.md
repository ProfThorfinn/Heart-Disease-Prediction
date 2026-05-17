Here is the updated `README.md` reflecting the real-world dataset, the corrected file names, and the specific results from your latest model run.

---

# Heart Disease Prediction using Machine Learning

A machine learning project that predicts whether a patient has heart disease using five classification algorithms trained on the **UCI Heart Disease Dataset (Cleveland Clinic Foundation)**.

* `0` = No heart disease present
* `1` = Heart disease present

---

## Project Structure

```
├── create_csv.py                    # Script to generate the dataset file
├── heart_disease_prediction.py      # Main machine learning script
├── heart_disease_for_excel.csv      # Real-world dataset (1,025 records)
└── README.md                        # Project overview (this file)

```

---

## Dataset

The project uses the real-world **UCI Heart Disease Dataset**. It contains clinical data from 1,025 patients with 13 clinical features and 1 target variable.

| Feature | Description | Type |
| --- | --- | --- |
| age | Age of the patient | Integer |
| sex | Sex (0 = female, 1 = male) | Binary |
| cp | Chest pain type (0–3) | Ordinal |
| trestbps | Resting blood pressure (mmHg) | Integer |
| chol | Serum cholesterol (mg/dl) | Integer |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0–2) | Ordinal |
| thalach | Maximum heart rate achieved | Integer |
| exang | Exercise-induced angina (0/1) | Binary |
| oldpeak | ST depression induced by exercise | Float |
| slope | Slope of peak exercise ST segment | Ordinal |
| ca | Number of major vessels (0–4) | Ordinal |
| thal | Thalassemia type (0–3) | Ordinal |

**Train/Test Split:** 80% training (820 samples) / 20% testing (205 samples).

---

## Workflow

### 1. Data Preprocessing & EDA

* **Dataset Loading:** Handled CSV metadata (`sep=,`) and binarized target values.
* **Outlier Detection:** Identified outliers in `trestbps`, `chol`, `thalach`, and `oldpeak` using the IQR method.
* **Correlation Analysis:** Identified `cp` (Chest Pain) and `thalach` (Max Heart Rate) as top positive predictors, while `oldpeak` and `exang` showed strong negative correlations.
* **Feature Scaling:** Applied `StandardScaler` to normalize the feature ranges.

### 2. Modeling & Evaluation

* **Baseline Models:** Logistic Regression and Naive Bayes.
* **Advanced Models:** - **KNN:** Tuned for the best K-value.
* **SVM:** Utilized a linear kernel.
* **Random Forest:** Ensemble method with depth constraints to prevent overfitting.


* **Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-Score.

---

## Model Performance Summary

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- |
| 1 | **Random Forest** | **92.68%** | 0.8947 | 0.9714 | 0.9315 |
| 2 | KNN | 84.39% | 0.8230 | 0.8857 | 0.8532 |
| 3 | Naive Bayes | 82.93% | 0.8070 | 0.8762 | 0.8402 |
| 4 | SVM | 81.46% | 0.7638 | 0.9238 | 0.8362 |
| 5 | Logistic Regression | 80.98% | 0.7619 | 0.9143 | 0.8312 |

**Best model: Random Forest — 92.68% accuracy**

---

## How to Run

1. **Prepare the Data:**
Run the CSV creation script to generate the dataset:
```bash
python create_csv.py

```


2. **Train and Evaluate:**
Run the main prediction script:
```bash
python heart_disease_prediction.py

```



**Requirements:**

```
pandas
numpy
scikit-learn

```

Install dependencies via pip:

```bash
pip install pandas numpy scikit-learn

```
