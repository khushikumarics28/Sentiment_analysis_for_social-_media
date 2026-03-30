import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- STEP 1: Resources Download ---
# Downloading necessary NLP tools as per objectives [cite: 90, 174]
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# --- STEP 2: Preprocessing Function [cite: 109-117] ---
def clean_text(text):
    # Remove URLs, mentions (@user), and special symbols [cite: 112]
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text))
    # Convert to lowercase [cite: 113]
    text = text.lower()
    words = text.split()
    # Remove stopwords and perform Stemming [cite: 116, 117]
    cleaned = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# --- STEP 3: Data Loading & Preparation ---
try:
    # Loading your specific dataset [cite: 99]
    df = pd.read_csv('sentimentdataset.csv')
    print("Dataset loaded successfully!")

    # Applying cleaning to the 'Text' column
    df['cleaned_text'] = df['Text'].apply(clean_text)
    
    # Cleaning the label column (removing leading/trailing spaces)
    df['Sentiment'] = df['Sentiment'].str.strip()

    # --- STEP 4: Feature Extraction (TF-IDF) [cite: 126, 129] ---
    # Using Unigrams and Bigrams to capture context like "not good" [cite: 129]
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['Sentiment']

    # --- STEP 5: Model Training (SVM) [cite: 132, 136] ---
    # Splitting dataset into 80% Training and 20% Testing [cite: 132]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training SVM Model... This may take a moment.")
    model = SVC(kernel='linear') # SVM is highly effective for text data [cite: 136]
    model.fit(X_train, y_train)

    # --- STEP 6: Evaluation [cite: 142-146] ---
    y_pred = model.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # --- STEP 7: Prototype Interface [cite: 186] ---
    print("\n--- Live Sentiment Predictor ---")
    while True:
        user_input = input("\nEnter text to analyze (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        processed_input = clean_text(user_input)
        vector_input = tfidf.transform([processed_input])
        result = model.predict(vector_input)
        print(f"Predicted Sentiment: {result[0]}")

except FileNotFoundError:
    print("Error: 'sentimentdataset.csv' not found. Ensure it is in the same folder.")
except Exception as e:
    print(f"An error occurred: {e}")