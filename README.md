# ❤️ Heart Disease Prediction — Machine Learning Project
 
## 🏗️ Project Folder Structure
 
```
heart_disease_prediction/
│
├── 📓 heart_disease_prediction.ipynb   ← MAIN NOTEBOOK (start here)
├── 📄 requirements.txt                 ← Python dependencies
├── 📊 model_comparison_report.md       ← Written analysis & report
├── 📋 model_comparison_report.csv      ← Auto-generated on notebook run
├── 📖 README.md                        ← This file
│
└── 📁 plots/                           ← All visualisations (auto-created)
    ├── 01_outlier_boxplots.png
    ├── 02_target_distribution.png
    ├── 03_histograms.png
    ├── 04_correlation_heatmap.png
    ├── 05_categorical_vs_target.png
    ├── 06_continuous_kde.png
    ├── 07_feature_correlation_target.png
    ├── 08_knn_tuning.png
    ├── 09_confusion_matrices.png
    ├── 10_roc_curves.png
    ├── 11_feature_importance.png
    ├── 12_model_comparison.png
    └── 13_cv_boxplot.png
```
 
---
 
## 🚀 Quick Start
 
### Step 1: Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
 
### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```
 
### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook heart_disease_prediction.ipynb
```
 
### Step 4: Run all cells
`Kernel → Restart & Run All`
 
---
 
## 📦 Libraries Used
 
| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Core language |
| pandas | ≥ 1.5 | Data manipulation |
| numpy | ≥ 1.23 | Numerical computing |
| scikit-learn | ≥ 1.2 | ML algorithms & evaluation |
| matplotlib | ≥ 3.6 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisations |
| jupyter | ≥ 1.0 | Notebook environment |
 
---
 
## 📚 Notebook Sections
 
| # | Section | What it covers |
|---|---------|---------------|
| 1 | Project Introduction | Goals, dataset overview, algorithm list |
| 2 | Import Libraries | All imports with explanations |
| 3 | Load Dataset | UCI Cleveland dataset loading + offline fallback |
| 4 | Data Exploration | Shape, dtypes, missing values, class balance |
| 5 | Data Cleaning | Missing values, duplicates, type fixes, outlier detection |
| 6 | EDA | Histograms, correlation heatmap, count plots, KDE plots |
| 7 | Feature Selection | Correlation with target, feature ranking |
| 8 | Feature Scaling | StandardScaler — why and how |
| 9 | Model Training | All 5 algorithms with explanations |
| 10 | Hyperparameter Tuning | GridSearchCV for KNN, SVM, Random Forest |
| 11 | Model Evaluation | Confusion matrices, ROC curves, feature importance |
| 12 | Model Comparison | Table, grouped bar chart, CV boxplot |
| 13 | Final Conclusion | Algorithm analysis, clinical implications |
 
---
