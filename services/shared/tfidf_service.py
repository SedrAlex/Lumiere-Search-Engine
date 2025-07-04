#!/usr/bin/env python3
"""
Shared TF-IDF Service with Enhanced Text Processing
Integrates enhanced text cleaning and tokenization with TF-IDF vectorization.
Designed to maintain MAP evaluation performance while improving text processing.
"""

import os
import json
import pickle
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
from collections import defaultdict

# Import our enhanced components
from services.shared.enhanced_text_cleaning_service import EnhancedTextCleaningService
from services.shared.enhanced_tokenizer import EnhancedTokenizer

logger = logging.getLogger(__name__)

class SharedTFIDFService:
    """
    Shared TF-IDF service with enhanced text processing capabilities.
    Supports both training and inference with inverted index optimization.
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 enable_spell_check: bool = True,
                 enable_lemmatization: bool = True,
                 enable_stemming: bool = True,
                 language: str = 'english'):
        """
        Initialize the shared TF-IDF service.
        
        Args:
            models_dir: Directory for model storage
            enable_spell_check: Whether to enable spell checking
            enable_lemmatization: Whether to enable lemmatization
            enable_stemming: Whether to enable stemming
            language: Language for processing
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced components
        self.text_cleaner = EnhancedTextCleaningService(
            language=language,
            enable_spell_check=enable_spell_check
        )
        
        self.enhanced_tokenizer = EnhancedTokenizer(
            enable_spell_check=enable_spell_check,
            enable_lemmatization=enable_lemmatization,
            enable_stemming=enable_stemming,
            language=language
        )
        
        # TF-IDF components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.inverted_index: Optional[Dict] = None
        self.doc_id_to_idx: Optional[Dict] = None
        self.idx_to_doc_id: Optional[Dict] = None
        self.document_metadata: Optional[Dict] = None
        self.training_stats: Optional[Dict] = None
        
        # Service state
        self.is_trained = False
        self.is_loaded = False
        self.search_cache = {}
        
        logger.info(f"Shared TF-IDF service initialized with enhanced processing")
    
    def create_enhanced_vectorizer(self, **vectorizer_params) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer with enhanced tokenizer.
        
        Args:
            **vectorizer_params: Additional parameters for TfidfVectorizer
            
        Returns:
            Configured TfidfVectorizer
        """
        # Default parameters optimized for IR performance
        default_params = {
            'max_features': 50000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 2),
            'sublinear_tf': True,
            'norm': 'l2',
            'smooth_idf': True,
            'tokenizer': self.enhanced_tokenizer,
            'preprocessor': None,  # We handle preprocessing in the tokenizer
            'lowercase': False,    # We handle lowercasing in the tokenizer
            'stop_words': None,    # We handle stop words in the tokenizer
        }
        
        # Update with user parameters
        default_params.update(vectorizer_params)
        
        return TfidfVectorizer(**default_params)
    
    def train_tfidf(self, 
                   documents: List[str], 
                   doc_ids: List[str],
                   vectorizer_params: Optional[Dict] = None,
                   build_inverted_index: bool = True) -> Dict:
        """
        Train TF-IDF model on documents with enhanced preprocessing.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            vectorizer_params: Parameters for TfidfVectorizer
            build_inverted_index: Whether to build inverted index
            
        Returns:
            Training statistics
        """
        logger.info(f"Training TF-IDF on {len(documents)} documents...")
        
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents must match number of doc_ids")
        
        # Enhanced preprocessing using our text cleaner
        logger.info("Preprocessing documents with enhanced text cleaning...")
        processed_texts = []
        valid_docs = []
        
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, documents)):
            # Use enhanced text cleaning for basic preprocessing
            cleaned_text = self.text_cleaner.clean_text_basic(doc_text)
            
            if cleaned_text.strip():  # Keep non-empty documents
                processed_texts.append(cleaned_text)
                valid_docs.append((doc_id, doc_text, cleaned_text))
        
        logger.info(f"Valid documents after preprocessing: {len(valid_docs)}")
        
        # Create enhanced vectorizer
        vectorizer_params = vectorizer_params or {}
        self.vectorizer = self.create_enhanced_vectorizer(**vectorizer_params)
        
        # Fit and transform documents
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Create document mappings
        valid_doc_ids = [item[0] for item in valid_docs]
        valid_original_texts = [item[1] for item in valid_docs]
        valid_processed_texts = [item[2] for item in valid_docs]
        
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(valid_doc_ids)}
        self.idx_to_doc_id = {idx: doc_id for doc_id, idx in self.doc_id_to_idx.items()}
        
        # Create document metadata
        self.document_metadata = {
            doc_id: {
                'original_text': valid_original_texts[idx],
                'processed_text': valid_processed_texts[idx],
                'index': idx
            }
            for doc_id, idx in self.doc_id_to_idx.items()
        }
        
        # Build inverted index if requested
        if build_inverted_index:
            logger.info("Building inverted index...")
            self.inverted_index = self._build_inverted_index()
        
        # Calculate training statistics
        self.training_stats = {
            'total_documents': len(documents),
            'valid_documents': len(valid_docs),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'matrix_shape': self.tfidf_matrix.shape,
            'non_zero_entries': int(self.tfidf_matrix.nnz),
            'sparsity': float((1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100),
            'inverted_index_terms': len(self.inverted_index) if self.inverted_index else 0,
            'vectorizer_params': self.vectorizer.get_params(),
            'text_cleaner_info': self.text_cleaner.get_service_info(),
            'tokenizer_info': self.enhanced_tokenizer.get_tokenizer_info()
        }
        
        self.is_trained = True
        logger.info("✓ TF-IDF training completed successfully!")
        
        return self.training_stats
    
    def _build_inverted_index(self) -> Dict:
        """
        Build inverted index from TF-IDF matrix.
        
        Returns:
            Inverted index dictionary
        """
        inverted_index = defaultdict(list)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Convert to COO format for efficient iteration
        coo_matrix = self.tfidf_matrix.tocoo()
        
        # Build index: term -> [(doc_id, tfidf_score), ...]
        for doc_idx, term_idx, tfidf_score in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            term = feature_names[term_idx]
            doc_id = self.idx_to_doc_id[doc_idx]
            inverted_index[term].append((doc_id, float(tfidf_score)))
        
        # Sort each posting list by TF-IDF score (descending)
        for term in inverted_index:
            inverted_index[term].sort(key=lambda x: x[1], reverse=True)
        
        return dict(inverted_index)
    
    def save_models(self, dataset_name: str) -> bool:
        """
        Save trained models and components.
        
        Args:
            dataset_name: Name for the dataset/model files
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.is_trained:
            logger.error("No trained model to save")
            return False
        
        try:
            logger.info(f"Saving TF-IDF models for {dataset_name}...")
            
            # Save TF-IDF vectorizer
            vectorizer_path = self.models_dir / f"tfidf_vectorizer_{dataset_name}.joblib"
            joblib.dump(self.vectorizer, vectorizer_path)
            
            # Save TF-IDF matrix
            matrix_path = self.models_dir / f"tfidf_matrix_{dataset_name}.joblib"
            joblib.dump(self.tfidf_matrix, matrix_path)
            
            # Save inverted index
            if self.inverted_index:
                index_path = self.models_dir / f"inverted_index_{dataset_name}.pkl"
                with open(index_path, 'wb') as f:
                    pickle.dump(self.inverted_index, f)
            
            # Save document mappings
            mappings_path = self.models_dir / f"doc_mappings_{dataset_name}.json"
            with open(mappings_path, 'w') as f:
                json.dump({
                    'doc_id_to_idx': self.doc_id_to_idx,
                    'idx_to_doc_id': self.idx_to_doc_id
                }, f)
            
            # Save document metadata
            metadata_path = self.models_dir / f"document_metadata_{dataset_name}.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.document_metadata, f)
            
            # Save training statistics
            stats_path = self.models_dir / f"training_statistics_{dataset_name}.json"
            with open(stats_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_stats = {}
                for key, value in self.training_stats.items():
                    if isinstance(value, (np.int64, np.int32)):
                        serializable_stats[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        serializable_stats[key] = float(value)
                    elif isinstance(value, tuple):
                        serializable_stats[key] = list(value)
                    else:
                        serializable_stats[key] = value
                
                json.dump(serializable_stats, f, indent=2)
            
            logger.info("✓ All models saved successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, dataset_name: str) -> bool:
        """
        Load pre-trained models and components.
        
        Args:
            dataset_name: Name of the dataset/model files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading TF-IDF models for {dataset_name}...")
            
            # Define file paths
            vectorizer_path = self.models_dir / f"tfidf_vectorizer_{dataset_name}.joblib"
            matrix_path = self.models_dir / f"tfidf_matrix_{dataset_name}.joblib"
            index_path = self.models_dir / f"inverted_index_{dataset_name}.pkl"
            mappings_path = self.models_dir / f"doc_mappings_{dataset_name}.json"
            metadata_path = self.models_dir / f"document_metadata_{dataset_name}.json"
            stats_path = self.models_dir / f"training_statistics_{dataset_name}.json"
            
            # Check required files
            required_files = [vectorizer_path, matrix_path, mappings_path]
            missing_files = [f for f in required_files if not f.exists()]
            
            if missing_files:
                logger.error(f"Missing required model files: {missing_files}")
                return False
            
            # Load components
            self.vectorizer = joblib.load(vectorizer_path)
            self.tfidf_matrix = joblib.load(matrix_path)
            
            # Load inverted index if available
            if index_path.exists():
                with open(index_path, 'rb') as f:
                    self.inverted_index = pickle.load(f)
            
            # Load document mappings
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.doc_id_to_idx = mappings['doc_id_to_idx']
                self.idx_to_doc_id = {int(k): v for k, v in mappings['idx_to_doc_id'].items()}
            
            # Load metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.document_metadata = json.load(f)
            
            # Load training statistics if available
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
            
            self.is_loaded = True
            self.is_trained = True
            
            logger.info(f"✓ TF-IDF models loaded successfully!")
            logger.info(f"  - Documents: {self.tfidf_matrix.shape[0]}")
            logger.info(f"  - Vocabulary: {len(self.vectorizer.vocabulary_)}")
            if self.inverted_index:
                logger.info(f"  - Index terms: {len(self.inverted_index)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_loaded = False
            return False
    
    def search(self, 
              query: str, 
              top_k: int = 10,
              use_inverted_index: bool = True,
              use_cache: bool = True) -> List[Dict]:
        """
        Search using the trained TF-IDF model.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            use_inverted_index: Whether to use inverted index for efficiency
            use_cache: Whether to use search cache
            
        Returns:
            List of search results with scores
        """
        if not (self.is_trained or self.is_loaded):
            raise RuntimeError("Model not trained or loaded. Train or load a model first.")
        
        # Check cache
        cache_key = f"{query}_{top_k}_{use_inverted_index}"
        if use_cache and cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Preprocess query using enhanced text cleaning
        processed_query = self.text_cleaner.clean_text_basic(query)
        
        if not processed_query.strip():
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        if use_inverted_index and self.inverted_index:
            results = self._search_with_inverted_index(query_vector, processed_query, top_k)
        else:
            results = self._search_full_matrix(query_vector, top_k)
        
        # Cache results
        if use_cache:
            self.search_cache[cache_key] = results
        
        return results
    
    def _search_with_inverted_index(self, query_vector, processed_query: str, top_k: int) -> List[Dict]:
        """Search using inverted index for efficiency."""
        query_terms = self.enhanced_tokenizer.tokenize(processed_query)
        
        # Get candidate documents from inverted index
        candidate_docs = set()
        term_scores = {}
        
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, tfidf_score in self.inverted_index[term]:
                    candidate_docs.add(doc_id)
                    if doc_id not in term_scores:
                        term_scores[doc_id] = 0
                    term_scores[doc_id] += tfidf_score
        
        if not candidate_docs:
            return []
        
        # Convert to indices and calculate similarities
        candidate_indices = [
            self.doc_id_to_idx[doc_id] 
            for doc_id in candidate_docs 
            if doc_id in self.doc_id_to_idx
        ]
        
        if not candidate_indices:
            return []
        
        candidate_matrix = self.tfidf_matrix[candidate_indices]
        similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        # Combine scores
        combined_scores = []
        for i, idx in enumerate(candidate_indices):
            doc_id = self.idx_to_doc_id[idx]
            tfidf_sim = similarities[i]
            index_score = term_scores.get(doc_id, 0)
            
            # Weighted combination
            combined_score = 0.7 * tfidf_sim + 0.3 * (index_score / len(query_terms))
            combined_scores.append(combined_score)
        
        # Get top results
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for i, candidate_idx in enumerate(top_indices):
            original_idx = candidate_indices[candidate_idx]
            doc_id = self.idx_to_doc_id[original_idx]
            
            result = {
                'doc_id': doc_id,
                'score': float(combined_scores[candidate_idx]),
                'tfidf_similarity': float(similarities[candidate_idx]),
                'index_score': float(term_scores.get(doc_id, 0)),
                'rank': i + 1
            }
            
            # Add document text if metadata is available
            if self.document_metadata and doc_id in self.document_metadata:
                result['document_text'] = self.document_metadata[doc_id]['original_text'][:200] + "..."
            
            results.append(result)
        
        return results
    
    def _search_full_matrix(self, query_vector, top_k: int) -> List[Dict]:
        """Search using full TF-IDF matrix."""
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            doc_id = self.idx_to_doc_id[idx]
            
            result = {
                'doc_id': doc_id,
                'score': float(similarities[idx]),
                'rank': i + 1
            }
            
            # Add document text if metadata is available
            if self.document_metadata and doc_id in self.document_metadata:
                result['document_text'] = self.document_metadata[doc_id]['original_text'][:200] + "..."
            
            results.append(result)
        
        return results
    
    def get_service_info(self) -> Dict:
        """Get comprehensive service information."""
        info = {
            'is_trained': self.is_trained,
            'is_loaded': self.is_loaded,
            'cache_size': len(self.search_cache),
            'text_cleaner_info': self.text_cleaner.get_service_info(),
            'tokenizer_info': self.enhanced_tokenizer.get_tokenizer_info()
        }
        
        if self.is_trained or self.is_loaded:
            info.update({
                'total_documents': self.tfidf_matrix.shape[0],
                'vocabulary_size': len(self.vectorizer.vocabulary_),
                'matrix_sparsity': (1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100,
                'has_inverted_index': self.inverted_index is not None,
                'inverted_index_terms': len(self.inverted_index) if self.inverted_index else 0
            })
            
            if self.training_stats:
                info['training_statistics'] = self.training_stats
        
        return info
    
    def clear_caches(self):
        """Clear all caches."""
        self.search_cache.clear()
        self.text_cleaner.clear_caches()
        self.enhanced_tokenizer.clear_cache()
        logger.info("All caches cleared")

# Factory functions
def create_shared_tfidf_service(models_dir: str = "models", 
                               enable_spell_check: bool = True,
                               language: str = 'english') -> SharedTFIDFService:
    """
    Factory function to create a shared TF-IDF service.
    
    Args:
        models_dir: Directory for model storage
        enable_spell_check: Whether to enable spell checking
        language: Language for processing
        
    Returns:
        SharedTFIDFService instance
    """
    return SharedTFIDFService(
        models_dir=models_dir,
        enable_spell_check=enable_spell_check,
        language=language
    )

# Example usage
if __name__ == "__main__":
    # Test the service
    service = SharedTFIDFService(enable_spell_check=True)
    
    # Sample documents
    docs = [
        "Beautiful antique furniture from the Victorian era",
        "Vintage collectibles and rare items for sale",
        "Old books and manuscripts from the 18th century",
        "Classic cars and automotive memorabilia"
    ]
    doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    
    # Train the model
    stats = service.train_tfidf(docs, doc_ids)
    print("Training stats:", stats)
    
    # Test search
    results = service.search("antique furniture", top_k=3)
    print("Search results:", results)
    
    # Get service info
    info = service.get_service_info()
    print("Service info:", info)
