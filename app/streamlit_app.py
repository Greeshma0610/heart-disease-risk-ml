"""Streamlit demo — Explainable Heart Disease Risk Predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BEST_MODEL_PATH, FEATURE_COLUMNS, FIGURES_DIR, METRICS_JSON

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        return None
    return joblib.load(BEST_MODEL_PATH)


def load_metrics():
    if not METRICS_JSON.exists():
        return None
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


st.title("Explainable Heart Disease Risk Predictor")
st.caption(
    "ML portfolio demo — Logistic Regression / Random Forest / XGBoost + SHAP. "
    "Not for clinical use."
)

model = load_model()
metrics = load_metrics()

if model is None:
    st.error(
        "Model not found. From the project root run:\n\n"
        "`python -m src.train`"
    )
    st.stop()

col_a, col_b = st.columns([1.1, 1])

with col_a:
    st.subheader("Patient features")
    age = st.slider("Age", 29, 77, 55)
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "Chest pain type (cp)",
        options=[0, 1, 2, 3],
        help="0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic",
    )
    trestbps = st.slider("Resting BP (trestbps)", 90, 200, 130)
    chol = st.slider("Cholesterol (chol)", 120, 560, 240)
    fbs = st.selectbox("Fasting blood sugar > 120 (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.slider("Max heart rate (thalach)", 70, 210, 150)
    exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
    slope = st.selectbox("ST slope", [0, 1, 2])
    ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], index=2)

    row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ]
    )[FEATURE_COLUMNS]

    if st.button("Predict risk", type="primary"):
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)
        st.metric("Disease probability", f"{proba * 100:.1f}%")
        if pred == 1:
            st.error("Model prediction: **Higher risk** (class = 1)")
        else:
            st.success("Model prediction: **Lower risk** (class = 0)")

with col_b:
    st.subheader("Model performance")
    if metrics:
        st.write(f"**Best model:** `{metrics.get('best_model')}`")
        table = (
            pd.DataFrame(metrics["models"])
            .T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"]]
            .round(3)
        )
        st.dataframe(table, use_container_width=True)
    else:
        st.info("Run training to populate metrics.")

    roc_path = FIGURES_DIR / "roc_curves.png"
    shap_bar = FIGURES_DIR / "shap_bar.png"
    shap_sum = FIGURES_DIR / "shap_summary.png"

    if roc_path.exists():
        st.image(str(roc_path), caption="ROC curves (test set)")
    if shap_bar.exists():
        st.image(str(shap_bar), caption="SHAP global feature importance")
    if shap_sum.exists():
        st.image(str(shap_sum), caption="SHAP summary (beeswarm)")

st.divider()
st.markdown(
    "Educational portfolio project only. Do **not** use for medical diagnosis or treatment decisions."
)
