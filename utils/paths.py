"""
utils/paths.py
--------------
Central path configuration for the project.
All other modules import paths from here — no hardcoded strings anywhere else.
"""

from pathlib import Path

# ── Root of the project (one level above this file) ──────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATASET_PATH = BASE_DIR / "dataset" / "Phishing_Email.csv"

# ── Saved artefacts ───────────────────────────────────────────────────────────
MODEL_DIR      = BASE_DIR / "model"
MODEL_PATH     = MODEL_DIR / "phishing_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# ── Output figures ────────────────────────────────────────────────────────────
SCREENSHOTS_DIR       = BASE_DIR / "screenshots"
CONFUSION_MATRIX_PATH = SCREENSHOTS_DIR / "confusion_matrix.png"
MODEL_COMPARISON_PATH = SCREENSHOTS_DIR / "model_comparison.png"
