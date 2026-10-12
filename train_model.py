import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from utils.paths import (
    MODEL_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    SCREENSHOTS_DIR,
    CONFUSION_MATRIX_PATH,
    MODEL_COMPARISON_PATH,
)
from utils.data_loader import load_and_clean_dataset


def main():
    # ======================================
    # Load & Clean Dataset
    # ======================================

    print("Loading dataset…")
    df = load_and_clean_dataset()
    print(f"Dataset Loaded Successfully!  ({len(df):,} rows)\n")

    # ======================================
    # Features & Labels
    # ======================================

    X = df["email"]
    y = df["label"]

    # ======================================
    # TF-IDF Vectorization
    # ======================================

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,     # More features for better coverage
        ngram_range=(1, 2),     # Unigrams + bigrams (e.g. "click here", "verify account")
        sublinear_tf=True       # Log-scale TF to reduce impact of very frequent terms
    )

    X = vectorizer.fit_transform(X)

    print("TF-IDF Vectorization Completed!")
    print("Feature Matrix Shape:", X.shape)

    # ======================================
    # Train / Test Split
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y              # Preserve class ratio in both splits
    )

    print(f"\nTraining Samples : {X_train.shape[0]:,}")
    print(f"Testing  Samples : {X_test.shape[0]:,}")

    # ======================================
    # Machine Learning Models
    # ======================================

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression":     LogisticRegression(max_iter=1000),
        "Random Forest":           RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_model = None
    best_f1    = 0

    accuracies = {}
    f1_scores  = {}

    # 5-fold stratified CV — stable estimates regardless of how the data was split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n========== MODEL EVALUATION ==========\n")

    for name, model in models.items():

        print("=" * 60)
        print(f"Training: {name}")

        # ── Cross-Validation (on full data, before the hold-out test) ─────────
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"CV F1 (5-fold) : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # ── Hold-out Evaluation ───────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted")
        auc      = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        accuracies[name] = accuracy
        f1_scores[name]  = f1

        print(f"Accuracy       : {accuracy:.4f}")
        print(f"F1 (weighted)  : {f1:.4f}")
        print(f"ROC-AUC        : {auc:.4f}")

        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Select best model by F1 — robust against class imbalance
        if f1 > best_f1:
            best_f1    = f1
            best_model = model

    print("\n" + "=" * 60)
    print(f"Best Model     : {best_model.__class__.__name__}")
    print(f"Best F1 Score  : {best_f1:.4f}")

    # ======================================
    # Save Best Model
    # ======================================

    MODEL_DIR.mkdir(exist_ok=True)
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel and Vectorizer saved successfully!")

    # ======================================
    # Confusion Matrix Plot
    # ======================================

    y_pred = best_model.predict(X_test)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Safe", "Phishing"],
        cmap="Blues",
    )

    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.show()

    # ======================================
    # Model Performance Comparison (F1 + Accuracy)
    # ======================================

    x      = range(len(f1_scores))
    width  = 0.35
    labels = list(f1_scores.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_f1  = ax.bar([i - width / 2 for i in x], f1_scores.values(),  width, label="F1 Score", color="steelblue")
    bars_acc = ax.bar([i + width / 2 for i in x], accuracies.values(), width, label="Accuracy",  color="lightcoral")

    ax.set_title("Model Performance Comparison (F1 vs Accuracy)")
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.90, 1.00)
    ax.legend()

    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PATH)
    plt.show()


if __name__ == "__main__":
    main()