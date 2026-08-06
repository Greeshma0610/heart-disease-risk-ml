"""Download, clean, and split the heart disease dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    DATA_URL,
    DATA_URL_FALLBACK,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COL,
    TEST_SIZE,
)


def download_dataset(force: bool = False) -> Path:
    """Download heart disease CSV into data/raw if missing."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not force:
        return RAW_CSV

    last_error = None
    for url in (DATA_URL, DATA_URL_FALLBACK):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            RAW_CSV.write_bytes(resp.content)
            return RAW_CSV
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not download dataset: {last_error}")


def load_raw() -> pd.DataFrame:
    """Load and lightly clean the raw dataset."""
    download_dataset()
    df = pd.read_csv(RAW_CSV)

    # Normalize column names across common mirrors
    rename_map = {
        "HeartDisease": TARGET_COL,
        "target": TARGET_COL,
        "condition": TARGET_COL,
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if TARGET_COL not in df.columns:
        # Some CSVs use last column as label
        df = df.rename(columns={df.columns[-1]: TARGET_COL})

    # Keep known features when present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    df = df[FEATURE_COLUMNS + [TARGET_COL]].copy()
    df = df.dropna()
    # Binary target (some datasets have 0-4 severity)
    df[TARGET_COL] = (df[TARGET_COL].astype(float) > 0).astype(int)
    return df


def prepare_splits(
    df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split and persist CSVs."""
    if df is None:
        df = load_raw()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_out = DATA_PROCESSED / "train.csv"
    test_out = DATA_PROCESSED / "test.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_out, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_out, index=False)
    return X_train, X_test, y_train, y_test
