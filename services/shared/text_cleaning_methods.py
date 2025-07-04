#!/usr/bin/env python3
"""
Text Cleaning Methods Service for Shared Use
Provides various text cleaning methods for different use cases while maintaining MAP performance.
"""

import re
import html
import string
import unicodedata
from typing import List, Dict, Optional, Set, Tuple, Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import nltk
import logging
from collections import defaultdict
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

logger = logging.getLogger(__name__)

class TextCleaningMethods:
    """
    Collection of text cleaning methods for different purposes.
    Designed to work with TF-IDF while preserving MAP evaluation performance.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize text cleaning methods.
        
        Args:
            language: Language for processing
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Enhanced stopwords
        self.technical_stopwords = {
            'code', 'function', 'method', 'class', 'variable', 'return',
            'import', 'from', 'def', 'if', 'else', 'for', 'while', 'try',
            'catch', 'finally', 'throw', 'throws', 'public', 'private',
            'protected', 'static', 'final', 'abstract', 'interface'
        }
        
        self.domain_specific_stopwords = {
            'antique', 'vintage', 'old', 'item', 'piece', 'thing', 'stuff',
            'want', 'need', 'looking', 'find', 'search', 'help', 'please',
            'anyone', 'someone', 'know', 'tell', 'show', 'give', 'get',
            'would', 'could', 'should', 'might', 'maybe', 'perhaps'
        }
        
        self.all_stopwords = self.stop_words.union(
            self.technical_stopwords
        ).union(self.domain_specific_stopwords)
        
        # Common contractions
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "what's": "what is",
            "where's": "where is", "how's": "how is"
        }
        
        # Normalization patterns
        self.normalization_patterns = [
            (r'https?://[^\s<>"]{2,}', ' URL '),  # URLs
            (r'www\.[^\s<>"]{2,}', ' URL '),      # www URLs
            (r'\S+@\S+', ' EMAIL '),              # Email addresses
            (r'\$\d+(?:\.\d+)?', ' PRICE '),      # Prices
            (r'\d{4}-\d{2}-\d{2}', ' DATE '),     # Dates
            (r'\d{1,2}[:/]\d{1,2}[:/]\d{2,4}', ' DATE '),  # More dates
            (r'\b\d{4}\b', ' YEAR '),             # Years
        ]
        
        # Spell checking cache
        self.spell_check_cache = {}
        
        logger.info(f"Text cleaning methods initialized for {language}")
    
    def get_wordnet_pos(self, word: str) -> str:
        """
        Map POS tag to WordNet POS tag for lemmatization.
        
        Args:
            word: Word to get POS tag for
            
        Returns:
            WordNet POS tag
        """
        try:
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {
                'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV
            }
            return tag_dict.get(tag, wordnet.NOUN)
        except Exception:
            return wordnet.NOUN
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to ASCII equivalents.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize Unicode to decomposed form
        text = unicodedata.normalize('NFD', text)
        
        # Remove combining characters (accents)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Convert to ASCII
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    def apply_normalization_patterns(self, text: str) -> str:
        """
        Apply normalization patterns to replace specific patterns with tokens.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        for pattern, replacement in self.normalization_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def spell_check_word(self, word: str, conservative: bool = True) -> str:
        """
        Apply spell checking to a single word.
        
        Args:
            word: Word to spell check
            conservative: Whether to use conservative spell checking
            
        Returns:
            Corrected word or original if no correction needed
        """
        if len(word) < 4:
            return word
        
        # Check cache first
        cache_key = f"{word}_{conservative}"
        if cache_key in self.spell_check_cache:
            return self.spell_check_cache[cache_key]
        
        try:
            # Use TextBlob for spell checking
            blob = TextBlob(word)
            corrected = str(blob.correct())
            
            # Conservative approach: only use correction if it's significantly better
            if conservative:
                if (corrected != word and 
                    word not in self.stop_words and
                    abs(len(corrected) - len(word)) <= 2 and
                    len(set(corrected.lower()) & set(word.lower())) >= min(len(word), len(corrected)) * 0.6):
                    
                    self.spell_check_cache[cache_key] = corrected
                    return corrected
            else:
                # More aggressive spell checking
                if corrected != word and word not in self.stop_words:
                    self.spell_check_cache[cache_key] = corrected
                    return corrected
            
            self.spell_check_cache[cache_key] = word
            return word
            
        except Exception as e:
            logger.debug(f"Spell check failed for '{word}': {e}")
            self.spell_check_cache[cache_key] = word
            return word
    
    def basic_clean(self, text: str) -> str:
        """
        Basic text cleaning for MAP-preserving preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # HTML decoding and tag removal
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Apply normalization patterns
        text = self.apply_normalization_patterns(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove extra punctuation but preserve sentence boundaries
        text = re.sub(r'[^\w\s\.!?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def advanced_clean(self, text: str, 
                      enable_spell_check: bool = True,
                      enable_lemmatization: bool = True,
                      enable_stemming: bool = True,
                      conservative_spell_check: bool = True) -> str:
        """
        Advanced text cleaning with spell checking, lemmatization, and stemming.
        
        Args:
            text: Input text
            enable_spell_check: Whether to enable spell checking
            enable_lemmatization: Whether to enable lemmatization
            enable_stemming: Whether to enable stemming
            conservative_spell_check: Whether to use conservative spell checking
            
        Returns:
            Processed text
        """
        # Basic cleaning first
        text = self.basic_clean(text)
        
        if not text.strip():
            return ""
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        # Process each token
        processed_tokens = []
        for token in tokens:
            # Skip if too short, not alphabetic, or is stopword
            if len(token) <= 2 or not token.isalpha() or token in self.all_stopwords:
                continue
            
            # Apply spell checking
            if enable_spell_check:
                token = self.spell_check_word(token, conservative_spell_check)
            
            # Apply lemmatization
            if enable_lemmatization:
                pos = self.get_wordnet_pos(token)
                token = self.lemmatizer.lemmatize(token, pos)
            
            # Apply stemming
            if enable_stemming:
                token = self.stemmer.stem(token)
            
            # Keep the result if it's still meaningful
            if len(token) > 2 and token not in self.all_stopwords:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def tfidf_optimized_clean(self, text: str) -> str:
        """
        Text cleaning optimized specifically for TF-IDF vectorization.
        Conservative approach to maintain MAP performance.
        
        Args:
            text: Input text
            
        Returns:
            Processed text optimized for TF-IDF
        """
        return self.advanced_clean(
            text,
            enable_spell_check=True,
            enable_lemmatization=True,
            enable_stemming=True,
            conservative_spell_check=True  # Conservative for MAP preservation
        )
    
    def embedding_optimized_clean(self, text: str) -> str:
        """
        Text cleaning optimized for embedding models.
        Preserves more structure and context.
        
        Args:
            text: Input text
            
        Returns:
            Processed text optimized for embeddings
        """
        # Basic cleaning but preserve sentence structure
        text = self.basic_clean(text)
        
        if not text.strip():
            return ""
        
        # Light spell checking for obvious errors only
        tokens = word_tokenize(text)
        corrected_tokens = []
        
        for token in tokens:
            if len(token) > 4 and token.isalpha():
                corrected_tokens.append(self.spell_check_word(token, conservative=True))
            else:
                corrected_tokens.append(token)
        
        text = ' '.join(corrected_tokens)
        
        # Preserve punctuation for sentence boundaries
        text = re.sub(r'[^\w\s\.!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def query_optimized_clean(self, query: str) -> str:
        """
        Text cleaning optimized for search queries.
        Very conservative to preserve user intent.
        
        Args:
            query: Search query
            
        Returns:
            Processed query
        """
        # For very short queries, disable spell checking
        disable_spell_check = len(query.split()) < 3
        
        return self.advanced_clean(
            query,
            enable_spell_check=not disable_spell_check,
            enable_lemmatization=True,
            enable_stemming=True,
            conservative_spell_check=True
        )
    
    def batch_clean(self, texts: List[str], method: str = 'tfidf') -> List[str]:
        """
        Batch text cleaning using specified method.
        
        Args:
            texts: List of texts to clean
            method: Cleaning method ('basic', 'advanced', 'tfidf', 'embedding', 'query')
            
        Returns:
            List of cleaned texts
        """
        if method == 'basic':
            return [self.basic_clean(text) for text in texts]
        elif method == 'advanced':
            return [self.advanced_clean(text) for text in texts]
        elif method == 'tfidf':
            return [self.tfidf_optimized_clean(text) for text in texts]
        elif method == 'embedding':
            return [self.embedding_optimized_clean(text) for text in texts]
        elif method == 'query':
            return [self.query_optimized_clean(text) for text in texts]
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
    
    def get_cleaning_statistics(self, original: str, cleaned: str) -> Dict:
        """
        Get statistics about the cleaning process.
        
        Args:
            original: Original text
            cleaned: Cleaned text
            
        Returns:
            Dictionary with cleaning statistics
        """
        original_tokens = word_tokenize(original.lower()) if original else []
        cleaned_tokens = cleaned.split() if cleaned else []
        
        return {
            'original_length': len(original) if original else 0,
            'cleaned_length': len(cleaned) if cleaned else 0,
            'original_tokens': len(original_tokens),
            'cleaned_tokens': len(cleaned_tokens),
            'token_reduction_ratio': 1 - (len(cleaned_tokens) / max(len(original_tokens), 1)),
            'char_reduction_ratio': 1 - (len(cleaned) / max(len(original), 1)) if original else 0,
            'spell_corrections': len(self.spell_check_cache)
        }
    
    def clear_cache(self):
        """Clear spell checking cache."""
        self.spell_check_cache.clear()
        logger.info("Text cleaning cache cleared")
    
    def get_service_info(self) -> Dict:
        """
        Get service information and statistics.
        
        Returns:
            Dictionary with service information
        """
        return {
            'language': self.language,
            'total_stopwords': len(self.all_stopwords),
            'technical_stopwords': len(self.technical_stopwords),
            'domain_stopwords': len(self.domain_specific_stopwords),
            'contractions': len(self.contractions),
            'normalization_patterns': len(self.normalization_patterns),
            'spell_check_cache_size': len(self.spell_check_cache)
        }

# Factory functions
def create_text_cleaning_service(language: str = 'english') -> TextCleaningMethods:
    """
    Factory function to create text cleaning service.
    
    Args:
        language: Language for processing
        
    Returns:
        TextCleaningMethods instance
    """
    return TextCleaningMethods(language=language)

# Convenience functions for different use cases
def clean_for_tfidf(text: str, language: str = 'english') -> str:
    """
    Clean text for TF-IDF vectorization.
    
    Args:
        text: Text to clean
        language: Language for processing
        
    Returns:
        Cleaned text optimized for TF-IDF
    """
    cleaner = TextCleaningMethods(language)
    return cleaner.tfidf_optimized_clean(text)

def clean_for_embedding(text: str, language: str = 'english') -> str:
    """
    Clean text for embedding models.
    
    Args:
        text: Text to clean
        language: Language for processing
        
    Returns:
        Cleaned text optimized for embeddings
    """
    cleaner = TextCleaningMethods(language)
    return cleaner.embedding_optimized_clean(text)

def clean_query(query: str, language: str = 'english') -> str:
    """
    Clean search query.
    
    Args:
        query: Query to clean
        language: Language for processing
        
    Returns:
        Cleaned query
    """
    cleaner = TextCleaningMethods(language)
    return cleaner.query_optimized_clean(query)

# Example usage
if __name__ == "__main__":
    # Test the cleaning methods
    cleaner = TextCleaningMethods()
    
    # Test text
    test_text = "I'm looking for beautifull antique vases from the 1800s that cost around $100-200!"
    
    print("Original text:", test_text)
    print("Basic clean:", cleaner.basic_clean(test_text))
    print("TF-IDF clean:", cleaner.tfidf_optimized_clean(test_text))
    print("Embedding clean:", cleaner.embedding_optimized_clean(test_text))
    print("Query clean:", cleaner.query_optimized_clean(test_text))
    
    # Get statistics
    tfidf_cleaned = cleaner.tfidf_optimized_clean(test_text)
    stats = cleaner.get_cleaning_statistics(test_text, tfidf_cleaned)
    print("Cleaning statistics:", stats)
    
    # Service info
    info = cleaner.get_service_info()
    print("Service info:", info)
