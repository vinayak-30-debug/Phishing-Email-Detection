import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("dataset/Phishing_Email.csv")

print("Dataset Loaded Successfully!\n")

# ======================================
# Data Cleaning
# ======================================

df.drop(columns=["Unnamed: 0"], inplace=True)

df.columns = ["email", "label"]

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

df["label"] = df["label"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

print("Data Cleaning Completed!\n")

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
    max_features=5000
)

X = vectorizer.fit_transform(X)

print("TF-IDF Vectorization Completed!")
print("Feature Matrix Shape:", X.shape)

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ======================================
# Machine Learning Models
# ======================================

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_model = None
best_accuracy = 0

accuracies = {}

print("\n========== MODEL EVALUATION ==========\n")

for name, model in models.items():

    print("=" * 60)
    print(f"Training: {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies[name] = accuracy

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\n" + "=" * 60)
print(f"Best Model : {best_model.__class__.__name__}")
print(f"Best Accuracy : {best_accuracy:.4f}")

# ======================================
# Save Best Model
# ======================================

os.makedirs("model", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

joblib.dump(best_model, "model/phishing_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and Vectorizer saved successfully!")

# ======================================
# Confusion Matrix
# ======================================

y_pred = best_model.predict(X_test)

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Safe", "Phishing"],
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig("screenshots/confusion_matrix.png")

plt.show()

# ======================================
# Model Accuracy Comparison
# ======================================

plt.figure(figsize=(8,5))

plt.bar(
    accuracies.keys(),
    accuracies.values()
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.ylim(0.90, 1.00)

for i, value in enumerate(accuracies.values()):
    plt.text(i, value + 0.002, f"{value:.4f}", ha="center")

plt.tight_layout()

plt.savefig("screenshots/model_comparison.png")

plt.show()