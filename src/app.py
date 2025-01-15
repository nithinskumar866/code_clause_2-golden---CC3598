import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
MODEL_PATH = 'data/models/personality_model.joblib'
VECTORIZER_PATH = 'data/models/tfidf_vectorizer.joblib'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Function to make predictions and convert them to scores out of 10
def predict_personality_traits(resume_text):
    resume_tfidf = vectorizer.transform([resume_text])
    predictions = model.predict(resume_tfidf)
    
    # Scale the predictions to be between 0 and 10 (for display purposes)
    # We assume that the model's predictions are between -1 and 1, and we'll scale them to 0-10.
    # For negative values, we'll consider them as 0, and positive as scaled out of 10.
    
    result = {
        "Extroversion": max(0, min(round((predictions[0][0] + 1) * 5, 2), 10)),
        "Conscientiousness": max(0, min(round((predictions[0][1] + 1) * 5, 2), 10)),
        "Openness": max(0, min(round((predictions[0][2] + 1) * 5, 2), 10)),
        "Agreeableness": max(0, min(round((predictions[0][3] + 1) * 5, 2), 10)),
        "Neuroticism": max(0, min(round((predictions[0][4] + 1) * 5, 2), 10))
    }
    
    return result

# Streamlit UI
st.title("Personality Prediction from Resume")

st.write("""
This app predicts personality traits based on a resume (CSV file). Upload your CSV file to get started!
""")

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded CSV data
    st.subheader("Uploaded Data")
    st.write(df.head())  # Display first 5 rows of the uploaded CSV
    
    # Check if 'Resume_Text' column exists
    if 'Resume_Text' in df.columns:
        # Process each resume in the 'Resume_Text' column
        for idx, resume in enumerate(df['Resume_Text']):
            st.subheader(f"Resume {idx + 1} Predictions:")
            predictions = predict_personality_traits(resume)
            for trait, value in predictions.items():
                st.write(f"{trait}: {value}/10")
    else:
        st.error("The CSV file does not contain a 'Resume_Text' column.")
