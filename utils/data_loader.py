"""
utils/data_loader.py
--------------------
Shared data-loading and cleaning logic.
Imported by train_model.py and test_model.py so the logic lives in one place.
"""

import sys
import pandas as pd

from utils.paths import DATASET_PATH

# Label mapping shared by all scripts
LABEL_MAP = {
    "Safe Email": 0,
    "Phishing Email": 1,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}   # reverse: 0 → "Safe Email"


def load_and_clean_dataset() -> pd.DataFrame:
    """
    Load the phishing email CSV, clean it, and return a tidy DataFrame
    with columns ['email', 'label'] where label is 0 (safe) or 1 (phishing).

    Raises SystemExit with a user-friendly message if the file is missing.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("   Please ensure the dataset file is in place before running.")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

    # ── Rename & basic cleaning ───────────────────────────────────────────────
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    df.columns = ["email", "label"]

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Label mapping ────────────────────────────────────────────────────────
    # Strip whitespace first so "Safe Email " still maps correctly.
    df["label"] = df["label"].str.strip().map(LABEL_MAP)

    # Drop rows whose label didn't match either known category.
    rows_before = len(df)
    df.dropna(subset=["label"], inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"⚠️  Dropped {rows_dropped} row(s) with unrecognised label values.")

    df["label"] = df["label"].astype(int)

    return df
