#!/usr/bin/env python3
"""
Enhanced Tokenizer Service for TF-IDF Vectorization
Custom tokenizer with advanced features while preserving MAP evaluation performance.
"""

import re
from typing import List, Optional, Set
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import nltk
import logging
from textblob import TextBlob

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

class EnhancedTokenizer:
    """
    Enhanced tokenizer for TF-IDF vectorization with spell checking,
    lemmatization, and stemming capabilities.
    """
    
    def __init__(self, 
                 enable_spell_check: bool = True,
                 enable_lemmatization: bool = True,
                 enable_stemming: bool = True,
                 language: str = 'english',
                 min_token_length: int = 3,
                 max_token_length: int = 50):
        """
        Initialize the enhanced tokenizer.
        
        Args:
            enable_spell_check: Whether to enable spell checking
            enable_lemmatization: Whether to enable lemmatization
            enable_stemming: Whether to enable stemming
            language: Language for processing
            min_token_length: Minimum token length to keep
            max_token_length: Maximum token length to keep
        """
        self.enable_spell_check = enable_spell_check
        self.enable_lemmatization = enable_lemmatization
        self.enable_stemming = enable_stemming
        self.language = language
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Enhanced stopwords for better IR performance
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
        
        # Spell check cache for performance
        self.spell_check_cache = {}
        
        logger.info(f"Enhanced tokenizer initialized: spell_check={enable_spell_check}, "
                   f"lemmatization={enable_lemmatization}, stemming={enable_stemming}")
    
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
    
    def spell_check_token(self, token: str) -> str:
        """
        Apply spell checking to a single token.
        
        Args:
            token: Token to spell check
            
        Returns:
            Corrected token or original if no correction needed
        """
        if not self.enable_spell_check or len(token) < 4:
            return token
        
        # Check cache first
        if token in self.spell_check_cache:
            return self.spell_check_cache[token]
        
        try:
            # Use TextBlob for spell checking
            blob = TextBlob(token)
            corrected = str(blob.correct())
            
            # Only use correction if it's reasonable
            if (corrected != token and 
                token not in self.stop_words and
                abs(len(corrected) - len(token)) <= 2 and
                len(set(corrected.lower()) & set(token.lower())) >= min(len(token), len(corrected)) * 0.6):
                
                self.spell_check_cache[token] = corrected
                return corrected
            
            self.spell_check_cache[token] = token
            return token
            
        except Exception as e:
            logger.debug(f"Spell check failed for '{token}': {e}")
            self.spell_check_cache[token] = token
            return token
    
    def process_token(self, token: str) -> Optional[str]:
        """
        Process a single token through the complete pipeline.
        
        Args:
            token: Token to process
            
        Returns:
            Processed token or None if token should be filtered out
        """
        # Basic filtering
        if (len(token) < self.min_token_length or 
            len(token) > self.max_token_length or
            not token.isalpha() or
            token.lower() in self.all_stopwords):
            return None
        
        # Convert to lowercase
        token = token.lower()
        
        # Apply spell checking
        if self.enable_spell_check:
            token = self.spell_check_token(token)
        
        # Apply lemmatization
        if self.enable_lemmatization:
            pos = self.get_wordnet_pos(token)
            token = self.lemmatizer.lemmatize(token, pos)
        
        # Apply stemming
        if self.enable_stemming:
            token = self.stemmer.stem(token)
        
        # Final length check after processing
        if len(token) < self.min_token_length:
            return None
        
        return token
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the enhanced pipeline.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of processed tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Basic tokenization using NLTK
        try:
            raw_tokens = word_tokenize(text)
        except Exception as e:
            logger.debug(f"Tokenization failed, using simple split: {e}")
            raw_tokens = text.split()
        
        # Process each token
        processed_tokens = []
        for token in raw_tokens:
            processed_token = self.process_token(token)
            if processed_token:
                processed_tokens.append(processed_token)
        
        return processed_tokens
    
    def __call__(self, text: str) -> List[str]:
        """
        Make the tokenizer callable for sklearn TfidfVectorizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of processed tokens
        """
        return self.tokenize(text)
    
    def get_feature_names(self, texts: List[str]) -> Set[str]:
        """
        Get all unique features from a list of texts.
        
        Args:
            texts: List of texts to extract features from
            
        Returns:
            Set of unique features
        """
        features = set()
        for text in texts:
            tokens = self.tokenize(text)
            features.update(tokens)
        return features
    
    def clear_cache(self):
        """Clear spell check cache."""
        self.spell_check_cache.clear()
        logger.info("Tokenizer spell check cache cleared")
    
    def get_tokenizer_info(self) -> dict:
        """
        Get tokenizer configuration and statistics.
        
        Returns:
            Dictionary with tokenizer information
        """
        return {
            'spell_check_enabled': self.enable_spell_check,
            'lemmatization_enabled': self.enable_lemmatization,
            'stemming_enabled': self.enable_stemming,
            'language': self.language,
            'min_token_length': self.min_token_length,
            'max_token_length': self.max_token_length,
            'total_stopwords': len(self.all_stopwords),
            'spell_check_cache_size': len(self.spell_check_cache)
        }

def create_enhanced_tokenizer(enable_spell_check: bool = True,
                            enable_lemmatization: bool = True,
                            enable_stemming: bool = True,
                            language: str = 'english') -> EnhancedTokenizer:
    """
    Factory function to create an enhanced tokenizer.
    
    Args:
        enable_spell_check: Whether to enable spell checking
        enable_lemmatization: Whether to enable lemmatization
        enable_stemming: Whether to enable stemming
        language: Language for processing
        
    Returns:
        EnhancedTokenizer instance
    """
    return EnhancedTokenizer(
        enable_spell_check=enable_spell_check,
        enable_lemmatization=enable_lemmatization,
        enable_stemming=enable_stemming,
        language=language
    )

def create_conservative_tokenizer(language: str = 'english') -> EnhancedTokenizer:
    """
    Create a conservative tokenizer for high-precision scenarios.
    
    Args:
        language: Language for processing
        
    Returns:
        EnhancedTokenizer with conservative settings
    """
    return EnhancedTokenizer(
        enable_spell_check=False,  # Disable spell check for precision
        enable_lemmatization=True,
        enable_stemming=True,
        language=language,
        min_token_length=3,
        max_token_length=30
    )

def create_aggressive_tokenizer(language: str = 'english') -> EnhancedTokenizer:
    """
    Create an aggressive tokenizer for high-recall scenarios.
    
    Args:
        language: Language for processing
        
    Returns:
        EnhancedTokenizer with aggressive settings
    """
    return EnhancedTokenizer(
        enable_spell_check=True,   # Enable all features
        enable_lemmatization=True,
        enable_stemming=True,
        language=language,
        min_token_length=2,        # Allow shorter tokens
        max_token_length=50
    )

# Example usage
if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = EnhancedTokenizer(enable_spell_check=True)
    
    # Test text
    test_text = "I'm looking for beautifull antique vases from the 1800s that cost around $100-200."
    
    print("Original text:", test_text)
    tokens = tokenizer.tokenize(test_text)
    print("Tokens:", tokens)
    
    # Test with TfidfVectorizer compatibility
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create vectorizer with enhanced tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,  # We handle preprocessing in the tokenizer
        lowercase=False,    # We handle lowercasing in the tokenizer
        max_features=1000
    )
    
    test_docs = [
        "Beautiful antique furniture from the Victorian era",
        "Vintage collectibles and rare items for sale",
        "Old books and manuscripts from the 18th century"
    ]
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(test_docs)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Feature names sample: {vectorizer.get_feature_names_out()[:10]}")
    
    # Get tokenizer info
    info = tokenizer.get_tokenizer_info()
    print("Tokenizer info:", info)
