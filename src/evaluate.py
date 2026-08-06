"""Model evaluation helpers and plot generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, METRICS_JSON


def classification_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No disease", "Disease"],
        yticklabels=["No disease", "Disease"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(curves: Dict[str, tuple], out_path: Path) -> None:
    """curves: name -> (fpr, tpr, auc)."""
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Heart Disease Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test, name: str) -> Dict[str, Any]:
    """Score a fitted classifier and write confusion matrix plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = classification_metrics(y_test, y_prob, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_confusion_matrix(
        y_test,
        y_pred,
        f"{name} Confusion Matrix",
        FIGURES_DIR / f"cm_{name.lower().replace(' ', '_')}.png",
    )
    return {
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "y_prob": y_prob,
    }


def save_metrics_report(all_metrics: Dict[str, Dict[str, float]], best_model: str) -> None:
    payload = {"best_model": best_model, "models": all_metrics}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
