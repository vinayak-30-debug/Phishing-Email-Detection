import joblib

model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

with open("test_email.txt", "r", encoding="utf-8") as f:
    email = f.read()

email_vector = vectorizer.transform([email])

prediction = model.predict(email_vector)

if prediction[0] == 1:
    print("⚠️ Phishing Email")
else:
    print("✅ Safe Email")