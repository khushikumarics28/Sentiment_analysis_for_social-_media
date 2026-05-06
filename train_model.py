import pandas as pd

# Load dataset
df = pd.read_csv("sentimentdataset.csv", encoding='latin-1')

# Keep only required columns
df = df[["text", "sentiment"]]

# Clean data
df = df.dropna()
df["sentiment"] = df["sentiment"].str.lower().str.strip()

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    return text

df["text"] = df["text"].apply(clean_text)

print("Sentiment Distribution:\n")
print(df["sentiment"].value_counts())

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Features & labels
X = df["text"]
y = df["sentiment"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        max_features=8000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9
    )),
    ("clf", LogisticRegression(max_iter=2000, C=2))
])

# Train
model.fit(X_train, y_train)

# Accuracy
print("\nModel Accuracy:", model.score(X_test, y_test))

# Testing
print("\nTesting model:")
print("I am happy ->", model.predict(["I am happy"]))
print("I hate this ->", model.predict(["I hate this"]))
print("It is okay ->", model.predict(["It is okay"]))

# Save model
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl ✅")