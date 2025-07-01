# ===================================================================
# Embedding Training for Quora Dataset
# Uses all-MiniLM-L6-v2 from Hugging Face with preprocessing service
# ===================================================================

# STEP 1: Install Required Libraries
# Run this in the first cell of your Colab notebook
"""
!pip install sentence-transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install ir-datasets joblib requests scipy tqdm nltk numpy pandas scikit-learn faiss-cpu
"""

# STEP 2: Import Libraries and Setup
import joblib
import ir_datasets
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import os
import gc
from typing import List, Dict, Any, Iterator, Tuple
from google.colab import files
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import torch
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import faiss
from collections import defaultdict

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

print("✅ Libraries imported successfully!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# STEP 3: UNIFIED Text Cleaning Service
class UnifiedTextCleaningService:
    """
    Unified text cleaning service that matches your preprocessing service
    This ensures consistency between training and inference
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        print("✅ NLTK components initialized")
    
    def clean_text(self, text: str, 
                   remove_stopwords: bool = True,
                   apply_stemming: bool = True, 
                   apply_lemmatization: bool = False) -> str:
        if not text or not isinstance(text, str):
            return ""
        cleaned_text = self._basic_clean(text)
        tokens = self._tokenize(cleaned_text)
        if remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        if apply_stemming:
            tokens = self._apply_stemming(tokens)
        return " ".join(tokens)
    
    def _basic_clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        return [token.lower() for token in tokens if token.isalnum()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]
    
    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

text_cleaner = UnifiedTextCleaningService()
print("✅ Unified text cleaner initialized!")

# STEP 4: Load and Preprocess Quora Dataset
# Replace this with code to load your actual Quora data
def load_and_preprocess_quora_dataset():
    try:
        print("📚 Loading Quora dataset...")
        dataset = ir_datasets.load('quora/train')
        raw_documents = []
        queries = []
        qrels = {}
        print("Loading raw documents...")
        for doc in dataset.docs_iter():
            raw_documents.append({'doc_id': doc.doc_id, 'text': doc.text})
        print("Loading queries...")
        for query in dataset.queries_iter():
            queries.append({'query_id': query.query_id, 'text': query.text})
        print("📚 Quora dataset loaded successfully!")
        return raw_documents, queries, qrels
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None, None

# Use the rest of the code from corrected_embedding_training.py with the loaded Quora data...
