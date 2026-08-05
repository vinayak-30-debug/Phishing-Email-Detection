import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load dataset
df = pd.read_csv("dataset/Phishing_Email.csv")

# Remove unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

# Rename columns
df.columns = ["email", "label"]

print("=" * 70)

# Test first 10 emails
for i in range(10):

    email = df.iloc[i]["email"]
    actual = df.iloc[i]["label"]

    prediction = model.predict(vectorizer.transform([email]))[0]

    predicted = "Phishing Email" if prediction == 1 else "Safe Email"

    print(f"\nEmail {i+1}")
    print("-" * 70)
    print(email[:150] + "...")
    print(f"\nActual Label    : {actual}")
    print(f"Predicted Label : {predicted}")