#!/usr/bin/env python3
"""
Model loader module to handle the OptimizedAntiqueTextCleaner dependency issue
when loading pickled TF-IDF vectorizer for the ANTIQUE dataset.
"""

import joblib
import logging
import sys
import os
from pathlib import Path

# Add the current directory to the module path so the classes can be found
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and setup the OptimizedAntiqueTextCleaner in the correct context
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

def simple_tokenizer(text):
    """
    Simple tokenizer function that just splits on whitespace.
    Since texts are already cleaned, we just need to split them.
    This function needs to be defined at module level so it can be pickled/unpickled.
    """
    return text.split() if text else []


class OptimizedAntiqueTextCleaner:
    """
    Advanced text cleaning class for ANTIQUE medical documents
    with optimizations for MAX MAP performance.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

        # Remove key medical terms from stopwords
        important_terms = {'pain', 'cause', 'treatment'}
        self.stop_words = self.stop_words - important_terms

        self.lemmatizer = WordNetLemmatizer()

    def clean_and_tokenize(self, text: str) -> List[str]:
        """
        Enhanced text cleaning optimized for medical content.

        Args:
            text (str): Input text to clean

        Returns:
            list: Token list
        """
        if pd.isna(text) or not isinstance(text, str):
            return []

        text = text.lower()
        text = re.sub(r'[^a-z]+', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return tokens


def load_antique_tfidf_models(model_dir='/content/drive/MyDrive/tfidf-optimized/'):
    """
    Load TF-IDF models with proper OptimizedAntiqueTextCleaner context.

    Args:
        model_dir (str): Path to the model directory

    Returns:
        tuple: (vectorizer, matrix, inverted_index, doc_ids)
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info("Setting up OptimizedAntiqueTextCleaner for pickle loading...")

        # Create a mock __main__ module with OptimizedAntiqueTextCleaner
        import types
        import builtins

        original_main = sys.modules.get('__main__')

        mock_main = types.ModuleType('__main__')
        mock_main.OptimizedAntiqueTextCleaner = OptimizedAntiqueTextCleaner
        mock_main.simple_tokenizer = simple_tokenizer
        sys.modules['__main__'] = mock_main

        logger.info("Loading TF-IDF vectorizer...")
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))

        if original_main is not None:
            sys.modules['__main__'] = original_main
        else:
            del sys.modules['__main__']

        logger.info("Loading TF-IDF matrix...")
        matrix = joblib.load(os.path.join(model_dir, 'tfidf_matrix.joblib'))

        logger.info("Loading inverted index...")
        inverted_index = joblib.load(os.path.join(model_dir, 'inverted_index.joblib'))

        logger.info("Loading document mappings...")
        doc_ids = joblib.load(os.path.join(model_dir, 'doc_ids.joblib'))

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
        vectorizer, matrix, inverted_index, doc_ids = load_antique_tfidf_models()
        print("✓ All models loaded successfully!")

        test_query = "What causes knee pain?"
        query_vector = vectorizer.transform([test_query])
        print(f"✓ Test query processed: vector shape {query_vector.shape}, nnz: {query_vector.nnz}")

    except Exception as e:
        print(f"✗ Error: {e}")

