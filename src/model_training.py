import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression  # Use Linear Regression for continuous output
import joblib
import os

# Define file paths
RAW_DATA_PATH = 'data/raw/fake_resume_dataset.csv'
MODEL_PATH = 'data/models/personality_model.joblib'
VECTORIZER_PATH = 'data/models/tfidf_vectorizer.joblib'  # Path for the vectorizer

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

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train MultiOutputRegressor with Linear Regression
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train_tfidf, y_train)

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# Save the vectorizer
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"Vectorizer saved at {VECTORIZER_PATH}")
