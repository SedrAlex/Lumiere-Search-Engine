#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
import joblib
import os
import re
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTextProcessor:
    """
    Unified text processor for both Quora and Antique datasets
    """
    
    def __init__(self):
        self.vectorizers = {}
        self.tfidf_matrices = {}
        self.doc_ids = {}
        self.doc_texts = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'quora': {
                'preserve_question_words': True,
                'question_words': {
                    'what', 'when', 'where', 'why', 'who', 'which', 'how',
                    'can', 'could', 'would', 'should', 'will', 'shall',
                    'do', 'does', 'did', 'is', 'are', 'was', 'were',
                    'not', 'no', 'never', 'none', 'nothing', 'neither',
                    'more', 'most', 'less', 'least', 'very', 'quite',
                    'much', 'many', 'few', 'some', 'any', 'all',
                    'best', 'better', 'good', 'bad', 'right', 'wrong'
                },
                'contractions': {
                    "don't": "do not", "won't": "will not", "can't": "cannot",
                    "n't": " not", "'re": " are", "'ve": " have",
                    "'ll": " will", "'d": " would", "'m": " am",
                    "what's": "what is", "that's": "that is"
                }
            },
            'antique': {
                'preserve_question_words': True,
                'question_words': {
                    'what', 'when', 'where', 'why', 'who', 'which', 'how',
                    'can', 'could', 'would', 'should', 'will', 'shall',
                    'do', 'does', 'did', 'is', 'are', 'was', 'were',
                    'not', 'no', 'never', 'none', 'nothing', 'neither'
                },
                'contractions': {
                    "don't": "do not", "won't": "will not", "can't": "cannot",
                    "n't": " not", "'re": " are", "'ve": " have",
                    "'ll": " will", "'d": " would", "'m": " am"
                }
            }
        }
    
    def load_tfidf_models(self, dataset_name: str, model_path: str) -> Dict[str, Any]:
        """Load TF-IDF models for a specific dataset"""
        try:
            if dataset_name not in self.dataset_configs:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Try to load pre-trained models
            vectorizer_path = os.path.join(model_path, 'tfidf_vectorizer.joblib')
            matrix_path = os.path.join(model_path, 'tfidf_matrix.joblib')
            
            if os.path.exists(vectorizer_path) and os.path.exists(matrix_path):
                logger.info(f"Loading TF-IDF models from {model_path}")
                self.vectorizers[dataset_name] = joblib.load(vectorizer_path)
                self.tfidf_matrices[dataset_name] = joblib.load(matrix_path)
                
                # Try to load document IDs
                doc_ids_path = os.path.join(model_path, 'doc_ids.joblib')
                if os.path.exists(doc_ids_path):
                    self.doc_ids[dataset_name] = joblib.load(doc_ids_path)
                
                logger.info(f"Loaded TF-IDF models for {dataset_name}")
                return {
                    'dataset': dataset_name,
                    'vectorizer_loaded': True,
                    'matrix_loaded': True,
                    'matrix_shape': self.tfidf_matrices[dataset_name].shape
                }
            else:
                # Create new vectorizer if models don't exist
                logger.warning(f"TF-IDF models not found at {model_path}, creating new vectorizer")
                self.vectorizers[dataset_name] = TfidfVectorizer(
                    tokenizer=lambda x: self._tokenize_text(x, dataset_name),
                    lowercase=False,  # Already handled by tokenizer
                    max_features=10000,
                    stop_words=None,  # Handled by tokenizer
                    ngram_range=(1, 2)
                )
                
                return {
                    'dataset': dataset_name,
                    'vectorizer_created': True,
                    'matrix_loaded': False,
                    'note': 'TF-IDF matrix will be created when documents are available'
                }
                
        except Exception as e:
            logger.error(f"Error loading TF-IDF models for {dataset_name}: {e}")
            raise e
    
    def clean_text(self, text: str, dataset: str) -> str:
        """Clean text based on dataset-specific rules"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        config = self.dataset_configs[dataset]
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        for contraction, expansion in config['contractions'].items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Handle numbers
        text = re.sub(r'\\b(19|20)\\d{2}\\b', ' ', text)  # Years
        text = re.sub(r'\\b\\d+\\.\\d+\\b', ' ', text)  # Decimals
        text = re.sub(r'\\b\\d+(?:st|nd|rd|th)\\b', ' ', text)  # Ordinals
        text = re.sub(r'\\b\\d+\\b', ' ', text)  # Other numbers
        
        # Handle punctuation
        text = re.sub(r'[!]{2,}', ' ', text)
        text = re.sub(r'[?]{2,}', ' ', text)
        text = re.sub(r'[.]{3,}', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\\s\\-_]', ' ', text)
        
        # Handle hyphenated words
        text = re.sub(r'\\b(\\w+)-(\\w+)\\b', r'\\1 \\2', text)
        
        # Normalize whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_text(self, text: str, dataset: str) -> List[str]:
        """Tokenize text based on dataset-specific rules"""
        config = self.dataset_configs[dataset]
        
        # Clean text first
        cleaned_text = self.clean_text(text, dataset)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Create dataset-specific stop words
        dataset_stop_words = self.stop_words.copy()
        if config['preserve_question_words']:
            dataset_stop_words = dataset_stop_words - config['question_words']
        
        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            # Skip very short tokens or stopwords
            if len(token) < 2 or token in dataset_stop_words:
                continue
            
            # Skip tokens that are just underscores or dashes
            if re.match(r'^[_\\-]+$', token):
                continue
            
            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def search_tfidf(self, query: str, dataset: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using TF-IDF"""
        try:
            if dataset not in self.vectorizers:
                raise ValueError(f"TF-IDF vectorizer not loaded for dataset: {dataset}")
            
            if dataset not in self.tfidf_matrices:
                raise ValueError(f"TF-IDF matrix not available for dataset: {dataset}")
            
            # Transform query
            query_vector = self.vectorizers[dataset].transform([query])
            
            if query_vector.nnz == 0:
                logger.warning("Query vector is empty, trying with cleaned query")
                cleaned_query = self.clean_text(query, dataset)
                query_vector = self.vectorizers[dataset].transform([cleaned_query])
            
            if query_vector.nnz == 0:
                logger.warning("Still empty query vector, returning empty results")
                return []
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrices[dataset]).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Format results
            results = []
            doc_ids = self.doc_ids.get(dataset, list(range(len(similarities))))
            doc_texts = self.doc_texts.get(dataset, [])
            
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:
                    result = {
                        'doc_id': doc_ids[idx] if idx < len(doc_ids) else idx,
                        'score': float(similarities[idx]),
                        'rank': i + 1
                    }
                    
                    # Add text if available
                    if idx < len(doc_texts):
                        result['text'] = doc_texts[idx]
                    
                    results.append(result)
            
            logger.info(f"TF-IDF search returned {len(results)} results for dataset {dataset}")
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search error for dataset {dataset}: {e}")
            raise e
    
    def fit_tfidf_on_documents(self, documents: List[str], dataset: str, doc_ids: List[str] = None) -> Dict[str, Any]:
        """Fit TF-IDF on a collection of documents"""
        try:
            if dataset not in self.vectorizers:
                raise ValueError(f"TF-IDF vectorizer not loaded for dataset: {dataset}")
            
            logger.info(f"Fitting TF-IDF on {len(documents)} documents for dataset {dataset}")
            
            # Fit and transform documents
            self.tfidf_matrices[dataset] = self.vectorizers[dataset].fit_transform(documents)
            self.doc_texts[dataset] = documents
            
            # Store document IDs
            if doc_ids:
                self.doc_ids[dataset] = doc_ids
            else:
                self.doc_ids[dataset] = list(range(len(documents)))
            
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrices[dataset].shape}")
            
            return {
                'dataset': dataset,
                'matrix_shape': self.tfidf_matrices[dataset].shape,
                'document_count': len(documents),
                'feature_count': self.tfidf_matrices[dataset].shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error fitting TF-IDF for dataset {dataset}: {e}")
            raise e
    
    def get_dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get information about a dataset's TF-IDF models"""
        if dataset not in self.vectorizers:
            raise ValueError(f"Dataset {dataset} not loaded")
        
        info = {
            'dataset': dataset,
            'vectorizer_loaded': True,
            'matrix_loaded': dataset in self.tfidf_matrices,
            'document_count': len(self.doc_texts.get(dataset, [])),
            'feature_count': self.tfidf_matrices[dataset].shape[1] if dataset in self.tfidf_matrices else 0
        }
        
        if dataset in self.tfidf_matrices:
            info['matrix_shape'] = self.tfidf_matrices[dataset].shape
        
        return info
    
    def get_available_datasets(self) -> List[str]:
        """Get list of datasets with loaded TF-IDF models"""
        return list(self.vectorizers.keys())
    
    def save_tfidf_models(self, dataset: str, save_path: str) -> Dict[str, Any]:
        """Save TF-IDF models for a dataset"""
        try:
            if dataset not in self.vectorizers:
                raise ValueError(f"Dataset {dataset} not loaded")
            
            os.makedirs(save_path, exist_ok=True)
            
            # Save vectorizer
            vectorizer_path = os.path.join(save_path, 'tfidf_vectorizer.joblib')
            joblib.dump(self.vectorizers[dataset], vectorizer_path)
            
            # Save matrix if available
            if dataset in self.tfidf_matrices:
                matrix_path = os.path.join(save_path, 'tfidf_matrix.joblib')
                joblib.dump(self.tfidf_matrices[dataset], matrix_path)
            
            # Save document IDs if available
            if dataset in self.doc_ids:
                doc_ids_path = os.path.join(save_path, 'doc_ids.joblib')
                joblib.dump(self.doc_ids[dataset], doc_ids_path)
            
            logger.info(f"Saved TF-IDF models for {dataset} to {save_path}")
            
            return {
                'dataset': dataset,
                'save_path': save_path,
                'files_saved': ['tfidf_vectorizer.joblib', 'tfidf_matrix.joblib', 'doc_ids.joblib']
            }
            
        except Exception as e:
            logger.error(f"Error saving TF-IDF models for {dataset}: {e}")
            raise e
