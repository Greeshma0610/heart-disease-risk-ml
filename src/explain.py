"""SHAP explainability for the best model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FEATURE_COLUMNS, FIGURES_DIR


def _get_estimator(pipeline) -> Any:
    """Unwrap sklearn Pipeline to the final classifier when possible."""
    if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps:
        return pipeline.named_steps["clf"]
    return pipeline


def _transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "named_steps") and "scaler" in pipeline.named_steps:
        return pipeline.named_steps["scaler"].transform(X)
    return X.values


def generate_shap_plots(pipeline, X_sample: pd.DataFrame, model_name: str) -> None:
    """
    Create global SHAP summary + bar importance plots.
    Works for tree models; falls back to KernelExplainer for linear models.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    estimator = _get_estimator(pipeline)
    X_trans = _transform_features(pipeline, X_sample)
    feature_names = list(FEATURE_COLUMNS)

    # Prefer TreeExplainer for RF / XGB
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans)
        # Binary classifiers sometimes return list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        # Linear / fallback — KernelExplainer on a small background set
        background = shap.sample(X_trans, min(50, len(X_trans)))
        explainer = shap.KernelExplainer(
            lambda data: pipeline.predict_proba(data)[:, 1]
            if not hasattr(pipeline, "named_steps")
            else estimator.predict_proba(data)[:, 1],
            background,
        )
        # For scaled pipelines, KernelExplainer should score in original space
        # Re-run using original-feature predict_proba for correctness
        bg_df = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(
            lambda data: pipeline.predict_proba(
                pd.DataFrame(data, columns=feature_names)
            )[:, 1],
            bg_df,
        )
        shap_values = explainer.shap_values(
            X_sample.iloc[: min(80, len(X_sample))], nsamples=100
        )
        X_trans = X_sample.iloc[: min(80, len(X_sample))].values

    # Summary beeswarm
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        show=False,
        max_display=13,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot of mean |SHAP|
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=13,
    )
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots to {FIGURES_DIR}")
