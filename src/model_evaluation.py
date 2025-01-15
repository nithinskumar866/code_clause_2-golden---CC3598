import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import joblib
import os

# Define file paths
RAW_DATA_PATH = 'data/raw/fake_resume_dataset.csv'
MODEL_PATH = 'data/models/personality_model.joblib'

# Check if the model directory exists, create it if it doesn't
model_directory = 'data/models'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Load dataset
data = pd.read_csv(RAW_DATA_PATH)

# Extract features and target variables
X = data['Resume_Text']
y = data.drop(columns=['Resume_Text'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization (ensure consistent feature space across train and test sets)
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on the training set and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vocabulary from the training set
X_test_tfidf = vectorizer.transform(X_test)

# Train MultiOutputRegressor with Linear Regression
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train_tfidf, y_train)

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
