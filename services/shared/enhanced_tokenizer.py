"""
Enhanced Tokenizer Module
Provides consistent tokenization that matches the training environment
"""

from typing import List
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EnhancedTokenizer:
    """Enhanced tokenizer class that matches the one used in Colab training"""
    def __init__(self, use_spellcheck=True):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.use_spellcheck = use_spellcheck
        self.spell_checker = None  # Disable spell checking in local environment

    def __call__(self, text: str) -> List[str]:
        """Tokenization pipeline: Lemmatization THEN Stemming"""
        if not text:
            return []

        # Basic cleaning
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Replace special chars

        # Tokenize
        tokens = word_tokenize(text)
        processed_tokens = []

        for token in tokens:
            if len(token) < 2 or not token.isalnum():
                continue

            # Skip stopwords
            if token in self.stop_words:
                continue

            # Skip spell checking in local environment to avoid dependency issues
            # Lemmatization THEN Stemming
            lemmatized = self.lemmatizer.lemmatize(token)
            stemmed = self.stemmer.stem(lemmatized)
            processed_tokens.append(stemmed)

        return processed_tokens
