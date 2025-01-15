import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for preprocessing the text (cleaning and normalizing)
def preprocess_text(text):
    """
    Preprocess the input text by:
    - Converting to lowercase
    - Removing special characters and numbers
    - Tokenizing the text
    - Removing stopwords
    - Lemmatizing the tokens

    Args:
    text (str): The raw input text.

    Returns:
    str: The preprocessed and cleaned text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphabetic characters (including numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    return ' '.join(tokens)
