#!/usr/bin/env python3
"""
Model loader module to handle the QuoraTextCleaner dependency issue
when loading pickled TF-IDF vectorizer.
"""

import joblib
import logging
import sys
import os
from pathlib import Path

# Add the current directory to the module path so the classes can be found
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and setup the QuoraTextCleaner in the correct context
import pandas as pd
import re
import nltk
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class QuoraTextCleaner:
    """
    Advanced text cleaning class optimized for Quora question pairs
    with semantic preservation and question-specific optimizations.
    """

    def __init__(self):
        # Setup stopwords with exceptions for important question words
        self.stop_words = set(stopwords.words('english'))

        # Remove question words and semantic indicators that are crucial for Quora
        question_words = {
            'what', 'when', 'where', 'why', 'who', 'which', 'how',
            'can', 'could', 'would', 'should', 'will', 'shall',
            'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'not', 'no', 'never', 'none', 'nothing', 'neither',
            'more', 'most', 'less', 'least', 'very', 'quite',
            'much', 'many', 'few', 'some', 'any', 'all',
            'best', 'better', 'good', 'bad', 'right', 'wrong'
        }
        self.stop_words = self.stop_words - question_words

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Common contractions for question text
        self.contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "what's": "what is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "where's": "where is",
            "how's": "how is"
        }

        # Question patterns that should be normalized
        self.question_patterns = {
            r'\bhow do i\b': 'how to',
            r'\bhow can i\b': 'how to',
            r'\bhow should i\b': 'how to',
            r'\bwhat is the best way to\b': 'how to',
            r'\bwhat are the ways to\b': 'how to',
            r'\bwhat are some\b': 'what are',
            r'\bwhat are the\b': 'what are'
        }

    def smart_clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning optimized for Quora questions.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Normalize question patterns
        for pattern, replacement in self.question_patterns.items():
            text = re.sub(pattern, replacement, text)

        # Remove or normalize specific patterns
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)
        text = re.sub(r'<.*?>', ' ', text)

        # Handle numbers more intelligently for questions
        text = re.sub(r'\b(19|20)\d{2}\b', ' YEAR ', text)  # Years
        text = re.sub(r'\b\d+\.\d+\b', ' DECIMAL ', text)  # Decimals
        text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', ' ORDINAL ', text)  # Ordinals
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)  # Other numbers

        # Handle emphasis and punctuation
        text = re.sub(r'[!]{2,}', ' EMPHASIS ', text)
        text = re.sub(r'[?]{2,}', ' MULTIQUEST ', text)
        text = re.sub(r'[.]{3,}', ' ELLIPSIS ', text)

        # Remove special characters but preserve some important ones
        text = re.sub(r'[^a-zA-Z0-9\s\-_]', ' ', text)

        # Handle hyphenated words carefully (important for compound terms)
        text = re.sub(r'\b(\w+)-(\w+)\b', r'\1 \2 \1\2', text)  # Keep both forms

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def custom_tokenizer(self, text: str) -> List[str]:
        """
        Custom tokenizer optimized for Quora questions.

        Args:
            text (str): Input text

        Returns:
            list: List of processed tokens
        """
        # Clean the text first
        cleaned_text = self.smart_clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            # Skip very short tokens or stopwords
            if len(token) < 2 or token in self.stop_words:
                continue

            # Skip tokens that are just underscores or dashes
            if re.match(r'^[_\-]+$', token):
                continue

            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)

        return processed_tokens

def load_tfidf_models(model_dir='/Users/raafatmhanna/Downloads/quora_tfidf_models/'):
    """
    Load TF-IDF models with proper QuoraTextCleaner context.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        tuple: (vectorizer, matrix, inverted_index, doc_ids)
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Setting up QuoraTextCleaner for pickle loading...")
        
        # Create a mock __main__ module with QuoraTextCleaner
        import types
        import builtins
        
        # Save the original __main__ if it exists
        original_main = sys.modules.get('__main__')
        
        # Create a new main module with QuoraTextCleaner
        mock_main = types.ModuleType('__main__')
        mock_main.QuoraTextCleaner = QuoraTextCleaner
        sys.modules['__main__'] = mock_main
        
        logger.info("Loading TF-IDF vectorizer...")
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        
        # Restore original __main__
        if original_main is not None:
            sys.modules['__main__'] = original_main
        else:
            del sys.modules['__main__']
        
        logger.info("Loading TF-IDF matrix...")
        matrix = joblib.load(os.path.join(model_dir, 'tfidf_matrix.joblib'))
        
        logger.info("Loading inverted index...")
        inverted_index = joblib.load(os.path.join(model_dir, 'inverted_index.joblib'))
        
        logger.info("Loading document mappings...")
        doc_mappings = joblib.load(os.path.join(model_dir, 'document_mappings.joblib'))
        doc_ids = doc_mappings['doc_ids']
        
        # Validate model consistency
        if matrix.shape[0] != len(doc_ids):
            raise ValueError(f"TF-IDF matrix rows ({matrix.shape[0]}) != doc IDs count ({len(doc_ids)})")
        
        if matrix.shape[1] != len(vectorizer.get_feature_names_out()):
            raise ValueError(f"TF-IDF matrix cols ({matrix.shape[1]}) != vectorizer vocab ({len(vectorizer.get_feature_names_out())})")
        
        logger.info("Models loaded successfully:")
        logger.info(f"  - TF-IDF Matrix: {matrix.shape}")
        logger.info(f"  - Vocabulary: {len(vectorizer.get_feature_names_out())} terms")
        logger.info(f"  - Inverted Index: {len(inverted_index)} terms")
        logger.info(f"  - Documents: {len(doc_ids)} documents")
        logger.info(f"  - Matrix sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")
        
        return vectorizer, matrix, inverted_index, doc_ids
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise RuntimeError(f"Failed to load TF-IDF models: {str(e)}")

if __name__ == "__main__":
    # Test the model loading
    logging.basicConfig(level=logging.INFO)
    
    try:
        vectorizer, matrix, inverted_index, doc_ids = load_tfidf_models()
        print("✓ All models loaded successfully!")
        
        # Test a simple query
        test_query = "How to learn programming"
        query_vector = vectorizer.transform([test_query])
        print(f"✓ Test query processed: vector shape {query_vector.shape}, nnz: {query_vector.nnz}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
