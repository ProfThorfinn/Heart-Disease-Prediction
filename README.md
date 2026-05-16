# Heart Disease Prediction using Machine Learning
 
A machine learning project that predicts whether a patient has heart disease using five classification algorithms trained on a synthetic dataset of 10,000 patients.
 
- `0` = Patient does **NOT** have heart disease
- `1` = Patient **DOES** have heart disease
---
 
## Project Structure
 
```
├── heartDisease.py                  # Main script
├── Heart_Disease_Prediction.ipynb   # Jupyter notebook version
├── README.md                        # Project overview (this file)
└── model_comparison.md              # Detailed model comparison report
```
 
---
 
## Dataset
 
The dataset is synthetically generated using a clinically-motivated risk scoring model with 10,000 patients and 13 features.
 
| Feature    | Description                          | Type    |
|------------|--------------------------------------|---------|
| age        | Age of the patient                   | Integer |
| sex        | Sex (0 = female, 1 = male)           | Binary  |
| cp         | Chest pain type (0–3)                | Ordinal |
| trestbps   | Resting blood pressure (mmHg)        | Integer |
| chol       | Serum cholesterol (mg/dl)            | Integer |
| fbs        | Fasting blood sugar > 120 mg/dl      | Binary  |
| restecg    | Resting ECG results (0–2)            | Ordinal |
| thalach    | Maximum heart rate achieved          | Integer |
| exang      | Exercise-induced angina (0/1)        | Binary  |
| oldpeak    | ST depression induced by exercise    | Float   |
| slope      | Slope of peak exercise ST segment    | Ordinal |
| ca         | Number of major vessels (0–3)        | Ordinal |
| thal       | Thalassemia type (0–3)               | Ordinal |
 
**Train/Test Split:** 80% training (8,000 samples) / 20% testing (2,000 samples)
 
---
 
## Work Distribution
 
### Data Preprocessing & EDA
- Missing value detection
- Outlier detection using the IQR method
- Exploratory Data Analysis — basic statistics and feature correlation with target
- Feature encoding verification (all features are integer-encoded)
- Feature scaling using `StandardScaler` (fitted on training data only)
### Modeling — Baseline Algorithms
- Train/test split (80/20)
- Logistic Regression
- Naive Bayes
- Analysis of where baseline models fail
### Modeling — Advanced Algorithms
- KNN with hyperparameter tuning (K = 3, 5, 7, 9, 11)
- SVM with linear kernel
- Random Forest with hyperparameter tuning (n_estimators × max_depth combinations)
### Evaluation & Reporting
- Accuracy, Precision, Recall, and F1-Score for all models
- Confusion matrix (TN / FP / FN / TP) per model
- Final ranked comparison table
---
 
## Models Trained
 
| # | Model               | Type       |
|---|---------------------|------------|
| 1 | Logistic Regression | Baseline   |
| 2 | Naive Bayes         | Baseline   |
| 3 | KNN                 | Advanced   |
| 4 | SVM                 | Advanced   |
| 5 | Random Forest       | Ensemble   |
 
---
 
## Results Summary
 
| Rank | Model                          | Accuracy | F1-Score |
|------|--------------------------------|----------|----------|
| 1    | Random Forest (n=100, depth=10)| 91.05%   | 0.9445   |
| 2    | Logistic Regression            | 87.05%   | 0.9190   |
| 3    | SVM                            | 86.95%   | 0.9187   |
| 4    | KNN (K=11)                     | 86.65%   | 0.9192   |
| 5    | Naive Bayes                    | 86.45%   | 0.9188   |
 
**Best model: Random Forest (n=100, depth=10) — 91.05% accuracy**
 
See `model_comparison.md` for the full breakdown.
 
---
 
## How to Run
 
```bash
# Run the Python script
python heartDisease.py
 
# Or open the notebook
jupyter notebook Heart_Disease_Prediction.ipynb
```
 
**Requirements:**
```
pandas
numpy
scikit-learn
```
 
Install with:
```bash
pip install pandas numpy scikit-learn
```
