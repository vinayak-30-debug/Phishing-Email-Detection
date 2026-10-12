import sys
import joblib

from utils.paths import MODEL_PATH, VECTORIZER_PATH


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
    # Get Email Input from User
    # ======================================

    print("=" * 60)
    print("       🛡️  Phishing Email Detector")
    print("=" * 60)
    print("\nPaste the email text below.")
    print("When done, press Enter on a blank line:\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)

    email = "\n".join(lines).strip()

    if not email:
        print("\n⚠️  No email text provided. Exiting.")
        sys.exit(1)

    # ======================================
    # Predict
    # ======================================

    email_vector = vectorizer.transform([email])
    prediction   = model.predict(email_vector)[0]
    confidence   = max(model.predict_proba(email_vector)[0]) * 100

    print("\n" + "=" * 60)

    if prediction == 1:
        print("🚨 Result     : Phishing Email")
    else:
        print("✅ Result     : Safe Email")

    print(f"📊 Confidence : {confidence:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()