#!/usr/bin/env python3
"""
Enhanced Text Cleaning Service for Information Retrieval System
Includes spell checking, lemmatization, stemming, and advanced normalization
while preserving MAP evaluation performance.
"""

import re
import html
import string
from typing import List, Dict, Optional, Set, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import nltk
import logging
from collections import defaultdict
from textblob import TextBlob
import unicodedata

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

class EnhancedTextCleaningService:
    """
    Advanced text cleaning service with multiple preprocessing techniques
    designed to improve retrieval performance while maintaining MAP scores.
    """
    
    def __init__(self, language: str = 'english', enable_spell_check: bool = True):
        """
        Initialize the enhanced text cleaning service.
        
        Args:
            language: Language for processing (default: 'english')
            enable_spell_check: Whether to enable spell checking (default: True)
        """
        self.language = language
        self.enable_spell_check = enable_spell_check
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Enhanced stopwords for better performance
        self.technical_stopwords = {
            'code', 'function', 'method', 'class', 'variable', 'return',
            'import', 'from', 'def', 'if', 'else', 'for', 'while', 'try',
            'catch', 'finally', 'throw', 'throws', 'public', 'private',
            'protected', 'static', 'final', 'abstract', 'interface'
        }
        
        self.domain_specific_stopwords = {
            'antique', 'vintage', 'old', 'item', 'piece', 'thing', 'stuff',
            'want', 'need', 'looking', 'find', 'search', 'help', 'please',
            'anyone', 'someone', 'know', 'tell', 'show', 'give', 'get'
        }
        
        self.all_stopwords = self.stop_words.union(
            self.technical_stopwords
        ).union(self.domain_specific_stopwords)
        
        # Common abbreviations and their expansions
        self.abbreviations = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "where's": "where is",
            "how's": "how is",
            "who's": "who is",
            "there's": "there is",
            "here's": "here is"
        }
        
        # Word frequency cache for spell checking
        self.word_freq_cache = defaultdict(int)
        self.spell_check_cache = {}
        
        # Normalization patterns
        self.normalization_patterns = [
            (r'https?://[^\s<>"]{2,}', ' URL '),  # URLs
            (r'www\.[^\s<>"]{2,}', ' URL '),      # www URLs
            (r'\S+@\S+', ' EMAIL '),              # Email addresses
            (r'\$\d+(?:\.\d+)?', ' PRICE '),      # Prices
            (r'\d{4}-\d{2}-\d{2}', ' DATE '),     # Dates
            (r'\d{1,2}[:/]\d{1,2}[:/]\d{2,4}', ' DATE '),  # More dates
            (r'\b\d{4}\b', ' YEAR '),             # Years
            (r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b', ' TIME '),  # Times
            (r'\b\d+(?:\.\d+)?%\b', ' PERCENT '), # Percentages
            (r'\b\d+(?:\.\d+)?\s*(?:lbs?|pounds?|kg|kgs?|oz|ounces?)\b', ' WEIGHT '),  # Weights
            (r'\b\d+(?:\.\d+)?\s*(?:ft|feet|inches?|in|cm|mm|meters?|m)\b', ' MEASUREMENT '),  # Measurements
        ]
        
        logger.info(f"Enhanced text cleaning service initialized with spell_check={enable_spell_check}")
    
    def get_wordnet_pos(self, word: str) -> str:
        """
        Map POS tag to first character lemmatize() accepts.
        
        Args:
            word: Word to get POS tag for
            
        Returns:
            WordNet POS tag
        """
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        for contraction, expansion in self.abbreviations.items():
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
    
    def spell_check_word(self, word: str) -> str:
        """
        Apply spell checking to a single word.
        
        Args:
            word: Word to spell check
            
        Returns:
            Corrected word or original if no correction needed
        """
        if not self.enable_spell_check or len(word) < 4:
            return word
        
        # Check cache first
        if word in self.spell_check_cache:
            return self.spell_check_cache[word]
        
        try:
            # Skip spell checking for words with numbers or special characters
            if not word.isalpha():
                self.spell_check_cache[word] = word
                return word
            
            # Use TextBlob for spell checking with better error handling
            blob = TextBlob(word)
            corrected = str(blob.correct())
            
            # Only use correction if it's significantly different
            # and the original word isn't in a standard dictionary
            if corrected != word and word not in self.stop_words:
                # Check if correction is reasonable (similar length, similar characters)
                if (abs(len(corrected) - len(word)) <= 2 and 
                    len(set(corrected.lower()) & set(word.lower())) >= min(len(word), len(corrected)) * 0.6):
                    self.spell_check_cache[word] = corrected
                    return corrected
            
            self.spell_check_cache[word] = word
            return word
            
        except Exception as e:
            # For any TextBlob errors (including zip file errors), just return original word
            logger.debug(f"Spell check failed for '{word}': {e}")
            self.spell_check_cache[word] = word
            return word
    
    def clean_text_basic(self, text: str) -> str:
        """
        Basic text cleaning without tokenization.
        
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
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def preprocess_for_tfidf(self, text: str) -> str:
        """
        Advanced preprocessing for TF-IDF with lemmatization and spell checking.
        This method is specifically designed to maintain MAP evaluation performance
        while improving text normalization.
        
        Args:
            text: Input text
            
        Returns:
            Processed text ready for TF-IDF vectorization
        """
        # Basic cleaning
        text = self.clean_text_basic(text)
        
        if not text.strip():
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Process each token with advanced techniques
        processed_tokens = []
        for token in tokens:
            # Skip if too short, not alphabetic, or is stopword
            if len(token) <= 2 or not token.isalpha() or token in self.all_stopwords:
                continue
            
            # Apply spell checking (conservative approach to preserve MAP)
            if self.enable_spell_check:
                original_token = token
                corrected_token = self.spell_check_word(token)
                # Only use correction if it significantly improves the word
                if corrected_token != original_token and len(corrected_token) >= 3:
                    token = corrected_token
            
            # Lemmatize with POS tagging for better word normalization
            pos = self.get_wordnet_pos(token)
            lemmatized = self.lemmatizer.lemmatize(token, pos)
            
            # Apply stemming to lemmatized word to reduce vocabulary
            stemmed = self.stemmer.stem(lemmatized)
            
            # Keep the result if it's still meaningful
            if len(stemmed) > 2 and stemmed not in self.all_stopwords:
                processed_tokens.append(stemmed)
        
        return ' '.join(processed_tokens)
    
    def preprocess_for_embedding(self, text: str) -> str:
        """
        Preprocessing for embedding models (preserve more structure).
        
        Args:
            text: Input text
            
        Returns:
            Processed text suitable for embedding models
        """
        # Basic cleaning but preserve sentence structure
        text = self.clean_text_basic(text)
        
        if not text.strip():
            return ""
        
        # Light spell checking for obvious errors
        if self.enable_spell_check:
            tokens = word_tokenize(text)
            corrected_tokens = []
            
            for token in tokens:
                if len(token) > 4 and token.isalpha():
                    corrected_tokens.append(self.spell_check_word(token))
                else:
                    corrected_tokens.append(token)
            
            text = ' '.join(corrected_tokens)
        
        # Preserve punctuation for sentence boundaries
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_query(self, query: str) -> str:
        """
        Special preprocessing for search queries.
        
        Args:
            query: Search query
            
        Returns:
            Processed query
        """
        # Use TF-IDF preprocessing but be more conservative with spell checking
        original_spell_check = self.enable_spell_check
        
        try:
            # Disable spell checking for very short queries
            if len(query.split()) < 3:
                self.enable_spell_check = False
            
            processed = self.preprocess_for_tfidf(query)
            return processed
            
        finally:
            self.enable_spell_check = original_spell_check
    
    def batch_preprocess(self, texts: List[str], method: str = 'tfidf') -> List[str]:
        """
        Batch preprocessing for efficiency.
        
        Args:
            texts: List of texts to process
            method: Processing method ('tfidf', 'embedding', 'query')
            
        Returns:
            List of processed texts
        """
        if method == 'tfidf':
            return [self.preprocess_for_tfidf(text) for text in texts]
        elif method == 'embedding':
            return [self.preprocess_for_embedding(text) for text in texts]
        elif method == 'query':
            return [self.preprocess_query(text) for text in texts]
        else:
            return [self.clean_text_basic(text) for text in texts]
    
    def get_preprocessing_statistics(self, original: str, processed: str) -> Dict:
        """
        Get preprocessing statistics for evaluation.
        
        Args:
            original: Original text
            processed: Processed text
            
        Returns:
            Dictionary with preprocessing statistics
        """
        original_tokens = word_tokenize(original.lower()) if original else []
        processed_tokens = processed.split() if processed else []
        
        return {
            'original_length': len(original) if original else 0,
            'processed_length': len(processed) if processed else 0,
            'original_tokens': len(original_tokens),
            'processed_tokens': len(processed_tokens),
            'token_reduction_ratio': 1 - (len(processed_tokens) / max(len(original_tokens), 1)),
            'char_reduction_ratio': 1 - (len(processed) / max(len(original), 1)) if original else 0,
            'spell_corrections': len(self.spell_check_cache) if self.enable_spell_check else 0
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self.spell_check_cache.clear()
        self.word_freq_cache.clear()
        logger.info("Text cleaning caches cleared")
    
    def get_service_info(self) -> Dict:
        """
        Get service information and statistics.
        
        Returns:
            Dictionary with service information
        """
        return {
            'language': self.language,
            'spell_check_enabled': self.enable_spell_check,
            'spell_check_cache_size': len(self.spell_check_cache),
            'total_stopwords': len(self.all_stopwords),
            'normalization_patterns': len(self.normalization_patterns),
            'abbreviations': len(self.abbreviations)
        }

# Factory function for easy service creation
def create_enhanced_text_cleaning_service(language: str = 'english', 
                                        enable_spell_check: bool = True) -> EnhancedTextCleaningService:
    """
    Factory function to create enhanced text cleaning service.
    
    Args:
        language: Language for processing
        enable_spell_check: Whether to enable spell checking
        
    Returns:
        EnhancedTextCleaningService instance
    """
    return EnhancedTextCleaningService(language=language, enable_spell_check=enable_spell_check)

# Example usage
if __name__ == "__main__":
    # Test the service
    service = EnhancedTextCleaningService(enable_spell_check=True)
    
    # Test text
    test_text = "I'm looking for an antique vase that's realy beautifull and costs around $100-200."
    
    print("Original text:", test_text)
    print("TF-IDF processed:", service.preprocess_for_tfidf(test_text))
    print("Embedding processed:", service.preprocess_for_embedding(test_text))
    print("Query processed:", service.preprocess_query(test_text))
    
    # Get statistics
    processed = service.preprocess_for_tfidf(test_text)
    stats = service.get_preprocessing_statistics(test_text, processed)
    print("Statistics:", stats)
    
    # Service info
    info = service.get_service_info()
    print("Service info:", info)
