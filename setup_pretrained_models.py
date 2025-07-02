#!/usr/bin/env python3
"""
Setup Pre-trained ANTIQUE TF-IDF Models
Downloads and configures pre-trained models for the TF-IDF service
"""

import os
import sys
import requests
import shutil
import joblib
from pathlib import Path
import json
import re
from typing import List
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Configuration
MODEL_DIRECTORY = "/tmp"  # Where to store models
BACKUP_DIRECTORY = "./models"  # Backup location in current directory
REQUIRED_FILES = [
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.joblib",
    "document_metadata.joblib"
]

class EnhancedTokenizer:
    """Enhanced tokenizer class that matches the one used in training"""
    def __init__(self, use_spellcheck=True):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.use_spellcheck = use_spellcheck
        self.spell_checker = None

    def __call__(self, text: str) -> List[str]:
        """Tokenization pipeline: Lemmatization THEN Stemming"""
        if not text:
            return []

        # Basic cleaning
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)
        processed_tokens = []

        for token in tokens:
            if len(token) < 2 or not token.isalnum():
                continue
            if token in self.stop_words:
                continue

            lemmatized = self.lemmatizer.lemmatize(token)
            stemmed = self.stemmer.stem(lemmatized)
            processed_tokens.append(stemmed)

        return processed_tokens

def initialize_nltk():
    """Download required NLTK data"""
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

def create_directories():
    """Create necessary directories"""
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    os.makedirs(BACKUP_DIRECTORY, exist_ok=True)
    print(f"✅ Created directories: {MODEL_DIRECTORY}, {BACKUP_DIRECTORY}")

def copy_local_files(source_dir: str) -> int:
    """Copy model files from source directory"""
    success_count = 0
    for filename in REQUIRED_FILES:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(MODEL_DIRECTORY, filename)
        backup_path = os.path.join(BACKUP_DIRECTORY, filename)
        
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                shutil.copy2(source_path, backup_path)
                print(f"✅ Copied {filename} to {MODEL_DIRECTORY} and {BACKUP_DIRECTORY}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to copy {filename}: {e}")
        else:
            print(f"⚠️  File not found: {source_path}")
    return success_count

def verify_model_files() -> bool:
    """Verify that all required model files exist and are valid"""
    print("🔍 Verifying model files...")
    all_valid = True
    
    for filename in REQUIRED_FILES:
        filepath = os.path.join(MODEL_DIRECTORY, filename)
        
        if os.path.exists(filepath):
            try:
                joblib.load(filepath)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"✅ {filename}: {file_size:.2f} MB - Valid")
            except Exception as e:
                print(f"❌ {filename}: Invalid - {e}")
                all_valid = False
        else:
            print(f"❌ {filename}: Not found")
            all_valid = False
    
    return all_valid

def update_service_configuration() -> bool:
    """Update TF-IDF service configuration"""
    service_file = os.path.join("services", "representation", "tfidf_service.py")
    
    if not os.path.exists(service_file):
        print(f"⚠️  Service file not found: {service_file}")
        return False
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
        
        updated_content = content.replace(
            'USE_PRETRAINED_ANTIQUE = False',
            'USE_PRETRAINED_ANTIQUE = True'
        )
        
        path_updates = {
            'ANTIQUE_MODEL_PATH': f'"{MODEL_DIRECTORY}/tfidf_vectorizer.joblib"',
            'ANTIQUE_MATRIX_PATH': f'"{MODEL_DIRECTORY}/tfidf_matrix.joblib"',
            'ANTIQUE_METADATA_PATH': f'"{MODEL_DIRECTORY}/document_metadata.joblib"'
        }
        
        for var, new_value in path_updates.items():
            updated_content = re.sub(
                rf'{var}.*=.*".*"',
                f'{var} = {new_value}',
                updated_content
            )
        
        with open(service_file, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated service configuration in {service_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to update service configuration: {e}")
        return False

def main() -> bool:
    """Main setup function"""
    print("🚀 Setting up Pre-trained ANTIQUE TF-IDF Models")
    print("="*60)
    
    initialize_nltk()
    create_directories()
    
    # Try current directory first
    print("\n📂 Looking for model files in current directory...")
    if copy_local_files(".") == 0:
        print("\n📂 Looking for model files in Downloads directory...")
        if copy_local_files(os.path.expanduser("~/Downloads")) == 0:
            print("\n❌ No pre-trained model files found!")
            print("\n📋 TO GET PRE-TRAINED MODELS:")
            print("1. Train models using enhanced_colab_tfidf_training.py in Colab")
            print("2. Download the generated .joblib files")
            print("3. Place them in this directory and run this script again")
            return False
    
    if not verify_model_files():
        print("\n❌ Model verification failed!")
        return False
    
    if not update_service_configuration():
        print("\n⚠️  Manual configuration needed: Set USE_PRETRAINED_ANTIQUE = True")
    
    print("\n🎉 Setup completed successfully!")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)