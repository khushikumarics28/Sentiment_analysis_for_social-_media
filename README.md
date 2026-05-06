# 🚀 Sentiment Analyzer Pro: Social Media Trend Analysis

### 🌐 Live Demo
👉 [Click here to view the Live Application](https://sentimentanalysisforsocial-media-ndhrt5vjgpvczhuwdvu9by.streamlit.app/)

---

## 📝 Project Overview
**Sentiment Analyzer Pro** is a sophisticated Machine Learning application designed to bridge the gap between raw social media noise and actionable emotional intelligence. Built using **Support Vector Machines (SVM)** and **TF-IDF Vectorization**, this tool provides a robust framework for classifying the tone of digital conversations.

Whether it’s monitoring brand reputation or analyzing shifting social trends, this application offers a high-speed, scalable solution for processing unstructured text data.

---

## ✨ Key Features

* **📝 Real-Time Prediction:** Enter any text snippet and get instant sentiment classification (Positive, Negative, or Neutral).
* **📈 Keyword Trend Analysis:** Analyze sentiments associated with specific topics or hashtags to understand public discourse.
* **🎨 Dynamic Data Visualization:**
    * **WordClouds:** Visualize dominant keywords.
    * **Sentiment Charts:** Interactive bar and pie charts showing distribution.
* **⚡ Optimized ML Pipeline:** Features a custom-built NLP pipeline that focuses on semantic relevance rather than just keyword matching.
* **📱 Responsive UI:** A clean, minimalist dashboard built with Streamlit for seamless user experience across devices.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Frontend:** Streamlit (Custom CSS for Modern UI)
* **Backend:** Flask API (Integration Layer)
* **Machine Learning:** Scikit-learn (SVM Classifier)
* **NLP:** NLTK, Regular Expressions (RegEx)
* **Data Science:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, WordCloud

---

## ⚙️ Model Pipeline

1.  **Data Preprocessing**
    * Noise reduction: Removal of URLs, user mentions (@), and hashtags.
    * Text Normalization: Lowercasing and removal of punctuation.
    * Tokenization: Converting sentences into individual word tokens.
2.  **Feature Engineering**
    * **TF-IDF Vectorization:** Assigning weights to words based on their importance across the dataset.
    * **N-gram Modeling:** Capturing context (e.g., "not good" vs "good").
3.  **Model Architecture**
    * **Algorithm:** Support Vector Machine (SVM).
    * **Accuracy:** Achieved a balanced performance optimized for social media slang.
4.  **Deployment**
    * Containerized for cloud hosting via Streamlit Cloud.

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/khushikumarics28/Sentiment_analysis_for_social-_media.git](https://github.com/khushikumarics28/Sentiment_analysis_for_social-_media.git)
```
cd Sentiment_analysis_for_social-_media
### 2. model training (any terminal)

bash
```sql
python train_model.py
```

### 3. Run backend (terminal_1)

bash
```sql
python backend.py
```
### 3. Run frontend.py (terminal_2)

bash
```sql
streamlit run frontend.py
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
