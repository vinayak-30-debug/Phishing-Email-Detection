import sys
import joblib

from utils.paths import MODEL_PATH, VECTORIZER_PATH
from utils.data_loader import LABEL_NAMES, load_and_clean_dataset


def main():
    # ======================================
    # Load Model & Vectorizer
    # ======================================

    try:
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print("❌ Model files not found.")
        print("   Please run 'python train_model.py' first to train the model.")
        sys.exit(1)

    # ======================================
    # Load Dataset
    # ======================================

    print("Loading dataset…")
    df = load_and_clean_dataset()
    print(f"Dataset loaded.  ({len(df):,} rows)\n")

    print("=" * 70)

    # ======================================
    # Test First 10 Emails
    # ======================================

    for i in range(10):
        email  = df.iloc[i]["email"]
        actual = df.iloc[i]["label"]

        prediction  = model.predict(vectorizer.transform([email]))[0]
        confidence  = max(model.predict_proba(vectorizer.transform([email]))[0]) * 100

        actual_label    = LABEL_NAMES[int(actual)]
        predicted_label = LABEL_NAMES[int(prediction)]

        print(f"\nEmail {i + 1}")
        print("-" * 70)
        print(email[:150] + "…")
        print(f"\nActual Label    : {actual_label}")
        print(f"Predicted Label : {predicted_label}  (confidence: {confidence:.1f}%)")

        match = "✅" if actual == prediction else "❌"
        print(f"Match           : {match}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()