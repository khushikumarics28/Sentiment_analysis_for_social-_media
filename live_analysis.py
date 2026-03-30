import pandas as pd
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# --- STEP 1: Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text))
    text = text.lower()
    words = text.split()
    cleaned = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# --- STEP 2: Model Training ---
print("Preparing Model using Dataset...")
df = pd.read_csv('sentimentdataset.csv')
df['cleaned_text'] = df['Text'].apply(clean_text)
df['Sentiment'] = df['Sentiment'].str.strip()
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['Sentiment']
model = SVC(kernel='linear')
model.fit(X, y)
print("Model Ready!")

# --- STEP 3: Live Analysis with Fallback ---
def fetch_live_tweets(query):
    api_key = "bfd5c10f15msh30265154de64532p133005jsne1a6633081da"
    url = "https://twitter-aio.p.rapidapi.com/search" 
    querystring = {"query": query, "type": "Latest"} 
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"}

    print(f"\n--- Searching for: {query} ---")
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        data = response.json()
        tweets = data if isinstance(data, list) else data.get('results', [])
    except:
        tweets = []

    # --- FALLBACK MECHANISM (If API fails or is empty) ---
    if not tweets:
        print("[System Note]: API limit reached or No Live Data. Switching to Simulated Live Stream...")
        # These represent real-world tweets for your demo
        tweets = [
            {"text": f"Watching {query} right now, the energy is absolutely incredible! #excited"},
            {"text": f"I am so disappointed with the recent news about {query}. Very poor management."},
            {"text": f"Just another normal day discussing {query} with my friends at the office."},
            {"text": f"The new updates in {query} are a total game changer! Love it."},
            {"text": f"Can't believe how bad the latest version of {query} is. Total waste of time."}
        ]

    # --- Analysis Output ---
    for i, tweet in enumerate(tweets[:5]):
        raw_text = tweet.get('text', '')
        if raw_text:
            vector = tfidf.transform([clean_text(raw_text)])
            prediction = model.predict(vector)[0]
            print(f"\n[{i+1}] Tweet: {raw_text[:100]}...")
            print(f"    Detected Sentiment: {prediction}")
            print("-" * 50)

# Run
topic = input("\nEnter search topic (e.g., IPL 2024, AI, Tesla): ")
fetch_live_tweets(topic)