﻿# code_clause_2-golden---CC3598
# Personality Prediction System via CV Analysis

## Project ID: #CC3598
**Project Title**: Personality Prediction System via CV Analysis  
**Internship Domain**: Artificial Intelligence Intern  
**Project Level**: Golden Level  
**Assigned By**: CodeClause Internship  

## Project Overview
This project predicts an individual's personality traits by analyzing their Curriculum Vitae (CV) or resume using Natural Language Processing (NLP) and machine learning techniques. The model processes the text from resumes to extract meaningful features, such as educational background, work experience, skills, and achievements, to predict personality traits such as:

- **Extroversion**
- **Conscientiousness**
- **Openness**
- **Agreeableness**
- **Neuroticism**

## Key Features
- **Resume Text Extraction**: The system processes the resume (in CSV format) and predicts personality traits.
- **ML Model**: A `MultiOutputRegressor` with `Linear Regression` is used to predict the five personality traits.
- **Vectorization**: The `TfidfVectorizer` converts resume text into numerical data that the model can process.
- **Streamlit Web App**: Allows users to upload CV files (in CSV format) and display predicted personality traits.

## Technologies Used
- Python
- `scikit-learn` (for machine learning)
- `pandas` (for data handling)
- `Streamlit` (for the web interface)
- `joblib` (for saving models)
- `nltk` (for text preprocessing)

## Project Workflow and Execution

### Step-by-Step Procedure:
1. **Model Training (First Step)**:
   - First, you need to run the model training script (`model_training.py`) to train and save the model, as well as the vectorizer.
   - This script will create the model and the vectorizer files (`personality_model.joblib` and `tfidf_vectorizer.joblib`) which are required for predictions.
   
   **Run Command**:
   ```bash
   python src/model_training.py
