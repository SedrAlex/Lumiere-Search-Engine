#!/usr/bin/env python3
"""
Debug script to test loading TF-IDF models
Identifies and fixes the EnhancedTokenizer import issue
"""

import os
import sys
import joblib
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

# Define EnhancedTokenizer class to handle the pickled vectorizer
class EnhancedTokenizer:
    """Enhanced tokenizer class that matches the one used in training"""
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

def test_model_loading():
    """Test loading each model component"""
    
    model_files = [
        ("tfidf_vectorizer.joblib", "TF-IDF Vectorizer"),
        ("tfidf_matrix.joblib", "TF-IDF Matrix"),
        ("document_metadata.joblib", "Document Metadata")
    ]
    
    print("🔍 Testing model file loading...")
    print("=" * 50)
    
    for filename, description in model_files:
        print(f"\n📄 Testing {description} ({filename})")
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
            
        try:
            # Try to load the file
            data = joblib.load(filename)
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"✅ Successfully loaded - Size: {file_size:.2f} MB")
            
            # Analyze the data structure
            if filename == "tfidf_vectorizer.joblib":
                print(f"   - Type: {type(data)}")
                if hasattr(data, 'vocabulary_'):
                    print(f"   - Vocabulary size: {len(data.vocabulary_):,}")
                if hasattr(data, 'tokenizer'):
                    print(f"   - Tokenizer: {type(data.tokenizer)}")
                    
            elif filename == "tfidf_matrix.joblib":
                print(f"   - Type: {type(data)}")
                print(f"   - Shape: {data.shape}")
                print(f"   - Non-zero elements: {data.nnz:,}")
                
            elif filename == "document_metadata.joblib":
                print(f"   - Type: {type(data)}")
                if isinstance(data, list):
                    print(f"   - Number of documents: {len(data):,}")
                    if len(data) > 0:
                        print(f"   - Sample keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                        
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            print(f"   Error type: {type(e).__name__}")

def test_vectorizer_functionality():
    """Test if the vectorizer can be used for new queries"""
    
    vectorizer_file = "tfidf_vectorizer.joblib"
    
    if not os.path.exists(vectorizer_file):
        print("❌ Vectorizer file not found!")
        return
        
    try:
        print("\n🧪 Testing vectorizer functionality...")
        vectorizer = joblib.load(vectorizer_file)
        
        # Test query
        test_query = "information retrieval systems"
        print(f"Test query: '{test_query}'")
        
        # Transform query
        query_vector = vectorizer.transform([test_query])
        print(f"✅ Query vectorized successfully")
        print(f"   - Query vector shape: {query_vector.shape}")
        print(f"   - Non-zero elements: {query_vector.nnz}")
        
        # Get vocabulary sample
        vocab = vectorizer.vocabulary_
        sample_terms = list(vocab.keys())[:10]
        print(f"   - Sample vocabulary terms: {sample_terms}")
        
    except Exception as e:
        print(f"❌ Error testing vectorizer: {e}")

def check_files_in_directories():
    """Check for model files in various directories"""
    
    directories_to_check = [
        ".",
        "./model",
        "./models", 
        "/tmp",
        os.path.expanduser("~/Downloads")
    ]
    
    print("\n📂 Checking for model files in directories...")
    print("=" * 50)
    
    model_files = ["tfidf_vectorizer.joblib", "tfidf_matrix.joblib", "document_metadata.joblib"]
    
    for directory in directories_to_check:
        if os.path.exists(directory):
            print(f"\n📁 {directory}:")
            for model_file in model_files:
                file_path = os.path.join(directory, model_file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   ✅ {model_file} ({size:.2f} MB)")
                else:
                    print(f"   ❌ {model_file}")
        else:
            print(f"\n📁 {directory}: Directory not found")

def main():
    """Main debug function"""
    print("🐛 TF-IDF Model Loading Debug Script")
    print("=" * 60)
    
    # Check for files in different directories
    check_files_in_directories()
    
    # Test loading from current directory
    print("\n" + "=" * 60)
    test_model_loading()
    
    # Test vectorizer functionality
    test_vectorizer_functionality()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS:")
    print("1. If files are missing, run the corrected_tfidf_training.py in Colab")
    print("2. Download the generated .joblib files to this directory")
    print("3. Run setup_pretrained_models.py to set up the service")
    print("4. If EnhancedTokenizer errors persist, the fix is already in place")

if __name__ == "__main__":
    main()
