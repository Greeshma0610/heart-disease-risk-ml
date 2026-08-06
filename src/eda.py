"""Quick EDA script — run: python -m src.eda"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL
from src.data import load_raw


def main() -> None:
    df = load_raw()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Shape:", df.shape)
    print("\nTarget balance:")
    print(df[TARGET_COL].value_counts(normalize=True).round(3))
    print("\nMissing values:", int(df.isna().sum().sum()))
    print("\nDescribe:")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))

    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df, palette="Blues")
    plt.title("Target distribution (0=no disease, 1=disease)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_balance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nSaved EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
