"""Train Logistic Regression, Random Forest, and XGBoost; save best model."""
from __future__ import annotations

import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_NAMES_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data import prepare_splits
from src.evaluate import evaluate_model, plot_roc_curves, save_metrics_report
from src.explain import generate_shap_plots

warnings.filterwarnings("ignore", category=UserWarning)


def build_models():
    """Return name -> (pipeline, optional grid)."""
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }

    rf = Pipeline(
        steps=[
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        ]
    )
    rf_grid = {
        "clf__max_depth": [4, 6, 10, None],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    xgb = Pipeline(
        steps=[
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    use_label_encoder=False,
                ),
            )
        ]
    )
    xgb_grid = {
        "clf__max_depth": [3, 4, 6],
        "clf__min_child_weight": [1, 3],
    }

    return {
        "Logistic Regression": (lr, lr_grid),
        "Random Forest": (rf, rf_grid),
        "XGBoost": (xgb, xgb_grid),
    }


def fit_with_cv(pipeline, grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def main() -> None:
    print("Loading data and creating train/test split...")
    X_train, X_test, y_train, y_test = prepare_splits()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    roc_payload = {}
    fitted = {}

    for name, (pipe, grid) in build_models().items():
        print(f"\n=== Training {name} (GridSearchCV ROC-AUC) ===")
        model, cv_auc, best_params = fit_with_cv(pipe, grid, X_train, y_train)
        print(f"Best CV ROC-AUC: {cv_auc:.4f}")
        print(f"Best params: {best_params}")

        result = evaluate_model(model, X_test, y_test, name)
        metrics = result["metrics"]
        metrics["cv_roc_auc"] = cv_auc
        all_metrics[name] = metrics
        fitted[name] = model
        roc_payload[name] = (
            np.array(result["fpr"]),
            np.array(result["tpr"]),
            metrics["roc_auc"],
        )
        print(
            "Test — "
            f"Acc={metrics['accuracy']:.3f}  "
            f"P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"AUC={metrics['roc_auc']:.3f}"
        )

    plot_roc_curves(roc_payload, FIGURES_DIR / "roc_curves.png")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\nBest model by test ROC-AUC: {best_name}")

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURE_NAMES_PATH)
    save_metrics_report(all_metrics, best_name)
    print(f"Saved model -> {BEST_MODEL_PATH}")

    print("\nGenerating SHAP explanations on a test sample...")
    sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
    generate_shap_plots(best_model, sample, best_name)

    print("\nDone. Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
