import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- STEP 1: Resources ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text))
    text = text.lower()
    words = text.split()
    cleaned = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# --- STEP 2: Load Data ---
try:
    df = pd.read_csv('sentimentdataset.csv')
    df['cleaned_text'] = df['Text'].apply(clean_text)
    df['Sentiment'] = df['Sentiment'].str.strip()

    # Feature Extraction
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- STEP 3: Model Comparison (The Research Part) ---
    models = {
        "SVM (Linear)": SVC(kernel='linear'),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    accuracies = {}

    print("\n--- Training and Comparing Models ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"{name} Accuracy: {acc*100:.2f}%")

    # --- STEP 4: Visualizations ---
    # 1. Bar Chart for Accuracy
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='magma')
    plt.title('Comparison of Model Accuracies')
    plt.ylabel('Accuracy Score')
    plt.show()

    # 2. Confusion Matrix for SVM
    best_model = models["SVM (Linear)"]
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\n--- Research Complete. Check the Graphs! ---")

except Exception as e:
    print(f"Error: {e}")