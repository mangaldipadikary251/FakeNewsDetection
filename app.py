import streamlit as st
import pickle
import re
import string

# Load the model and vectorizer
model = pickle.load(open(r"C:\dscode\project_6th_sem\models\fakenews_model.pkl", 'rb'))
vectorizer = pickle.load(open(r"C:\dscode\project_6th_sem\models\tfidf_vectorizer.pkl", 'rb'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

st.title("🛡️ AI Fake News Detector")
news_input = st.text_area("Paste news article text here:", height=200)

if st.button("Analyze News"):
    if news_input:
        cleaned_input = clean_text(news_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)
        
        if prediction[0] == 0:
            st.error("🚨 Result: This news is likely FAKE!")
        else:
            st.success("✅ Result: This news appears to be REAL.")