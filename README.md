# Explainable Heart Disease Risk Predictor

End-to-end **machine learning** project that predicts heart disease risk from clinical features, compares multiple models, and explains predictions with **SHAP**.


## Problem

Clinicians and screening tools need a fast, interpretable estimate of cardiovascular risk from routine measurements (age, blood pressure, cholesterol, ECG-related features, etc.). A black-box score is not enough — stakeholders need **why** a patient was flagged.

## Solution

1. Train and compare **Logistic Regression**, **Random Forest**, and **XGBoost**
2. Select the best model by **ROC-AUC** (with Accuracy, Precision, Recall, F1)
3. Explain predictions with **SHAP** (global + local)
4. Serve an interactive **Streamlit** demo for live risk scoring

## Dataset

UCI-style **Heart Disease** tabular dataset (303 patients, 13 clinical features + binary target).

| Feature | Meaning |
|---------|---------|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) colored by fluoroscopy |
| thal | Thalassemia status |
| target | 1 = disease, 0 = no disease |

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.787 | 0.750 | 0.909 | 0.822 | 0.880 |
| **Random Forest** | **0.852** | **0.800** | **0.970** | **0.877** | **0.917** |
| XGBoost | 0.803 | 0.769 | 0.909 | 0.833 | 0.874 |

**Best model:** Random Forest (selected by test ROC-AUC)

## Project structure

```
heart-disease-risk-ml/
├── app/
│   └── streamlit_app.py      # Live demo
├── data/
│   ├── raw/                  # Downloaded CSV
│   └── processed/            # Train/test splits
├── models/                   # Saved pipelines + scaler
├── notebooks/
│   └── 01_eda.ipynb          # Optional EDA notebook
├── reports/
│   ├── figures/              # ROC, confusion matrix, SHAP plots
│   └── metrics.json
├── src/
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── requirements.txt
└── README.md
```

## Quick start (Windows)

```bash
cd heart-disease-risk-ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Train models + generate metrics & SHAP plots
python -m src.train

# Launch demo
streamlit run app/streamlit_app.py
```

## 1-week plan

| Day | Focus |
|-----|--------|
| 1 | Setup, download data, EDA |
| 2 | Preprocessing + baseline Logistic Regression |
| 3 | Random Forest + XGBoost + hyperparameter search |
| 4 | Metrics, ROC, confusion matrices |
| 5 | SHAP global + local explanations |
| 6 | Streamlit UI polish |
| 7 | README, GitHub, resume bullet polish |

## Resume bullet examples

- Developed an explainable ML pipeline for heart disease risk prediction using Logistic Regression, Random Forest, and XGBoost; selected the best model by ROC-AUC and validated with precision/recall/F1.
- Applied SHAP to explain individual risk predictions and global feature importance for clinical interpretability.
- Built and deployed a Streamlit demo for interactive risk scoring from patient clinical features.

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · Matplotlib/Seaborn · Streamlit

## Disclaimer

This is an educational ML portfolio project — **not** a medical device and **not** for clinical decision-making.
