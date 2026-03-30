# 🚀 Sentiment Analyzer Pro: Social Media Trend Analysis

### 🌐 Live Demo

👉https://sentimentanalysisforsocial-media-ndhrt5vjgpvczhuwdvu9by.streamlit.app/
---

## 📝 Project Overview

**Sentiment Analyzer Pro** is a Machine Learning application that transforms raw social media text into actionable insights. Using **Support Vector Machines (SVM)** and modern **Natural Language Processing (NLP)** techniques, it classifies unstructured text into three categories: **Positive, Negative, and Neutral**.

This project demonstrates how sentiment analysis can be applied to understand public opinion for use cases like brand monitoring, product feedback, and social trend analysis.

---

## ✨ Key Features

* **📝 Manual Text Analysis**
  Instantly analyze sentiment for any custom input.

* **🌐 Topic-Based Sentiment Analysis**
  Evaluate sentiment trends for specific keywords or topics.

* **📊 Data Visualization**

  * WordClouds for frequently used terms
  * Sentiment distribution charts

* **⚡ Optimized ML Pipeline**
  TF-IDF vectorization for better context understanding over simple word counts.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Language:** Python
* **Machine Learning:** Scikit-learn (SVM)
* **NLP:** NLTK
* **Data Processing:** Pandas, NumPy

---

## ⚙️ Model Pipeline

1. **Data Preprocessing**

   * Lowercasing, cleaning, stopword removal
   * Stemming using NLTK

2. **Feature Extraction**

   * TF-IDF Vectorization
   * N-grams for better context capture

3. **Model Training**

   * Support Vector Machine (Linear Kernel)
   * Balanced class weights for fair predictions

4. **Evaluation**

   * Accuracy, Precision, Recall, F1-score

---

## 🚀 Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/khushikumarics28/Sentiment_analysis_for_social-_media.git
cd Sentiment_analysis_for_social-_media
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
app.py                  # Main Streamlit application
sentimentdataset.csv   # Dataset used for training
requirements.txt       # Dependencies
README.md              # Documentation
```

---

## 👥 Team Members

| Name             | Role                          |
| ---------------- | ----------------------------- |
| Khushi Kumari    | Project Lead & Model Training |
| Md Imran Ansari  | Lead Developer & UI Design    |
| Aditya Saluja    | Data Preprocessing & NLP      |
| Abhay Srivastava | Feature Engineering & Testing |
| Krishna Bajpai   | Documentation & Research      |

---

## 🔐 Security Notes

* Do not upload sensitive files (e.g., `.env`)
* Keep API keys private
* Use environment variables or secrets for deployment

---


