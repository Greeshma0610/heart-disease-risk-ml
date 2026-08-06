"""Paths and training configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/"
    "master/dataset.csv"
)
# Fallback mirror (same schema)
DATA_URL_FALLBACK = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/"
    "master/datasets/heart_disease/heart.csv"
)

RAW_CSV = DATA_RAW / "heart.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
