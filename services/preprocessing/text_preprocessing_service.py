#!/usr/bin/env python3
"""
Enhanced Text Preprocessing Service for Information Retrieval System
Optimized for 3-day implementation with TF-IDF, Embedding, and Hybrid support.
"""

import re
import html
from typing import List, Dict, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import string
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class TextPreprocessingService:
    """Fast and efficient text preprocessing for IR systems."""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Technical stopwords for code/technical documents
        self.technical_stopwords = {
            'code', 'function', 'method', 'class', 'variable', 'return',
            'import', 'from', 'def', 'if', 'else', 'for', 'while', 'try'
        }
        self.all_stopwords = self.stop_words.union(self.technical_stopwords)
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning for all representations."""
        if not text or not isinstance(text, str):
            return ""
        
        # HTML decoding and tag removal
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def preprocess_for_tfidf(self, text: str) -> str:
        """Optimized preprocessing for TF-IDF representation."""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if (token not in self.all_stopwords and 
                len(token) > 2 and 
                token.isalpha())
        ]
        
        return ' '.join(tokens)
    
    def preprocess_for_embedding(self, text: str) -> str:
        """Preprocessing for embedding models (preserve more structure)."""
        # Clean but preserve sentence structure
        text = self.clean_text(text)
        
        # Remove excessive punctuation but keep sentence boundaries
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_preprocess(self, texts: List[str], method: str = 'tfidf') -> List[str]:
        """Batch preprocessing for efficiency."""
        if method == 'tfidf':
            return [self.preprocess_for_tfidf(text) for text in texts]
        elif method == 'embedding':
            return [self.preprocess_for_embedding(text) for text in texts]
        else:
            return [self.clean_text(text) for text in texts]
    
    def get_statistics(self, original: str, processed: str) -> Dict:
        """Get preprocessing statistics for evaluation."""
        return {
            'original_length': len(original),
            'processed_length': len(processed),
            'original_words': len(original.split()),
            'processed_words': len(processed.split()),
            'reduction_ratio': 1 - (len(processed.split()) / max(len(original.split()), 1))
        }
