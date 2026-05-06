import streamlit as st
import requests

st.title("💬 Social Media Sentiment Analyzer")

st.write("Enter a sentence and check its sentiment")

text = st.text_area("Enter text here")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"text": text}
            )

            result = response.json()
            label = result["label"]

            st.write("Prediction:", label)

            if label == "positive":
                st.success("😊 Positive")
            elif label == "negative":
                st.error("😡 Negative")
            else:
                st.warning("😐 Neutral")

        except Exception as e:
            st.error(f"Error: {e}")