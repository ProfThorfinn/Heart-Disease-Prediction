❤️ Heart Disease Prediction — Machine Learning Project
🏗️ Project Folder Structure
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

🚀 Quick Start
Step 1: Create a virtual environment (recommended)
bashpython -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Step 2: Install dependencies
bashpip install -r requirements.txt
Step 3: Launch Jupyter Notebook
bashjupyter notebook heart_disease_prediction.ipynb
Step 4: Run all cells
Kernel → Restart & Run All

📦 Libraries Used
LibraryVersionPurposePython≥ 3.8Core languagepandas≥ 1.5Data manipulationnumpy≥ 1.23Numerical computingscikit-learn≥ 1.2ML algorithms & evaluationmatplotlib≥ 3.6Plottingseaborn≥ 0.12Statistical visualisationsjupyter≥ 1.0Notebook environment

📚 Notebook Sections
#SectionWhat it covers1Project IntroductionGoals, dataset overview, algorithm list2Import LibrariesAll imports with explanations3Load DatasetUCI Cleveland dataset loading + offline fallback4Data ExplorationShape, dtypes, missing values, class balance5Data CleaningMissing values, duplicates, type fixes, outlier detection6EDAHistograms, correlation heatmap, count plots, KDE plots7Feature SelectionCorrelation with target, feature ranking8Feature ScalingStandardScaler — why and how9Model TrainingAll 5 algorithms with explanations10Hyperparameter TuningGridSearchCV for KNN, SVM, Random Forest11Model EvaluationConfusion matrices, ROC curves, feature importance12Model ComparisonTable, grouped bar chart, CV boxplot13Final ConclusionAlgorithm analysis, clinical implications
