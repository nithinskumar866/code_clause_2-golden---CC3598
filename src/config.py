import os

# File paths
RAW_DATA_PATH = os.path.join('data', 'raw', 'resumes.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_resume_dataset.csv')
MODEL_PATH = os.path.join('data', 'models', 'personality_model.joblib')

# Hyperparameters for models
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42
