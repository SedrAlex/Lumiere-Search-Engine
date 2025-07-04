#!/usr/bin/env python3
"""
Optimized TF-IDF Service with Inverted Index Support
Uses pre-trained TF-IDF models and inverted index for efficient retrieval.
"""

import os
import json
import pickle
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from pathlib import Path

# Import preprocessing service
from services.preprocessing.text_preprocessing_service import TextPreprocessingService

logger = logging.getLogger(__name__)

class OptimizedTFIDFService:
    """
    TF-IDF service using pre-trained models and inverted index for efficient retrieval.
    Designed for high performance and MAP > 0.4 evaluation.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize TF-IDF service with pre-trained models.
        
        Args:
            models_dir: Directory containing trained model files
        """
        self.models_dir = Path(models_dir)
        self.preprocessor = TextPreprocessingService()
        
        # Model components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.inverted_index: Optional[Dict] = None
        self.doc_id_to_idx: Optional[Dict] = None
        self.idx_to_doc_id: Optional[Dict] = None
        self.document_metadata: Optional[Dict] = None
        self.training_stats: Optional[Dict] = None
        
        # Performance tracking
        self.is_loaded = False
        self.search_cache = {}
        
    def load_models(self, dataset_name: str = "antique") -> bool:
        """
        Load pre-trained TF-IDF models and components.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'antique')
            
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
            
            # Check if all required files exist
            required_files = [vectorizer_path, matrix_path, index_path, mappings_path]
            missing_files = [f for f in required_files if not f.exists()]
            
            if missing_files:
                logger.error(f"Missing required model files: {missing_files}")
                return False
            
            # Load TF-IDF vectorizer
            logger.info("Loading TF-IDF vectorizer...")
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load TF-IDF matrix
            logger.info("Loading TF-IDF matrix...")
            self.tfidf_matrix = joblib.load(matrix_path)
            
            # Load inverted index
            logger.info("Loading inverted index...")
            with open(index_path, 'rb') as f:
                self.inverted_index = pickle.load(f)
            
            # Load document mappings
            logger.info("Loading document mappings...")
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.doc_id_to_idx = mappings['doc_id_to_idx']
                self.idx_to_doc_id = {int(k): v for k, v in mappings['idx_to_doc_id'].items()}
            
            # Load document metadata (optional)
            if metadata_path.exists():
                logger.info("Loading document metadata...")
                with open(metadata_path, 'r') as f:
                    self.document_metadata = json.load(f)
            
            # Load training statistics (optional)
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
            
            self.is_loaded = True
            logger.info(f"✓ TF-IDF models loaded successfully!")
            logger.info(f"  - Documents: {self.tfidf_matrix.shape[0]}")
            logger.info(f"  - Vocabulary: {len(self.vectorizer.vocabulary_)}")
            logger.info(f"  - Index terms: {len(self.inverted_index)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading TF-IDF models: {str(e)}")
            self.is_loaded = False
            return False
    
    def search_with_inverted_index(self, query: str, top_k: int = 10, 
                                  use_cache: bool = True) -> List[Dict]:
        """
        Search using inverted index for efficient retrieval.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            use_cache: Whether to use search cache
            
        Returns:
            List of search results with scores
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Check cache
        cache_key = f"{query}_{top_k}"
        if use_cache and cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess_for_tfidf(query)
        
        if not processed_query.strip():
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        query_terms = processed_query.split()
        
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
        
        # Convert candidate doc IDs to indices
        candidate_indices = [
            self.doc_id_to_idx[doc_id] 
            for doc_id in candidate_docs 
            if doc_id in self.doc_id_to_idx
        ]
        
        if not candidate_indices:
            return []
        
        # Calculate cosine similarity only for candidate documents
        candidate_matrix = self.tfidf_matrix[candidate_indices]
        similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        # Combine TF-IDF similarity with inverted index scores
        combined_scores = []
        for i, idx in enumerate(candidate_indices):
            doc_id = self.idx_to_doc_id[idx]
            tfidf_sim = similarities[i]
            index_score = term_scores.get(doc_id, 0)
            
            # Weighted combination (you can tune these weights)
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
        
        # Cache results
        if use_cache:
            self.search_cache[cache_key] = results
        
        return results
    
    def search_full_matrix(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search using full TF-IDF matrix (for comparison/evaluation).
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess_for_tfidf(query)
        
        if not processed_query.strip():
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarities with all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
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
    
    def get_document_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Get TF-IDF vector for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            TF-IDF vector or None if document not found
        """
        if not self.is_loaded or doc_id not in self.doc_id_to_idx:
            return None
        
        idx = self.doc_id_to_idx[doc_id]
        return self.tfidf_matrix[idx].toarray().flatten()
    
    def get_query_vector(self, query: str) -> Optional[np.ndarray]:
        """
        Get TF-IDF vector for a query.
        
        Args:
            query: Search query
            
        Returns:
            TF-IDF vector or None if query is empty
        """
        if not self.is_loaded:
            return None
        
        processed_query = self.preprocessor.preprocess_for_tfidf(query)
        
        if not processed_query.strip():
            return None
        
        query_vector = self.vectorizer.transform([processed_query])
        return query_vector.toarray().flatten()
    
    def get_service_statistics(self) -> Dict:
        """
        Get service statistics and model information.
        
        Returns:
            Dictionary with service statistics
        """
        stats = {
            'is_loaded': self.is_loaded,
            'cache_size': len(self.search_cache)
        }
        
        if self.is_loaded:
            stats.update({
                'total_documents': self.tfidf_matrix.shape[0],
                'vocabulary_size': len(self.vectorizer.vocabulary_),
                'matrix_sparsity': (1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100,
                'inverted_index_terms': len(self.inverted_index),
                'non_zero_entries': self.tfidf_matrix.nnz
            })
            
            if self.training_stats:
                stats['training_statistics'] = self.training_stats
        
        return stats
    
    def clear_cache(self):
        """Clear search cache."""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    def get_term_document_frequency(self, term: str) -> Optional[List[Tuple[str, float]]]:
        """
        Get document frequency for a specific term from inverted index.
        
        Args:
            term: Term to look up
            
        Returns:
            List of (doc_id, tfidf_score) tuples or None if term not found
        """
        if not self.is_loaded or term not in self.inverted_index:
            return None
        
        return self.inverted_index[term]
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query and provide detailed information.
        
        Args:
            query: Query to analyze
            
        Returns:
            Dictionary with query analysis
        """
        processed_query = self.preprocessor.preprocess_for_tfidf(query)
        query_terms = processed_query.split()
        
        analysis = {
            'original_query': query,
            'processed_query': processed_query,
            'query_terms': query_terms,
            'terms_in_vocabulary': [],
            'terms_not_in_vocabulary': [],
            'term_frequencies': {}
        }
        
        if self.is_loaded:
            for term in query_terms:
                if term in self.vectorizer.vocabulary_:
                    analysis['terms_in_vocabulary'].append(term)
                    if term in self.inverted_index:
                        analysis['term_frequencies'][term] = len(self.inverted_index[term])
                else:
                    analysis['terms_not_in_vocabulary'].append(term)
        
        return analysis

# Factory function for easy service creation
def create_tfidf_service(models_dir: str = "models", dataset_name: str = "antique") -> OptimizedTFIDFService:
    """
    Factory function to create and load TF-IDF service.
    
    Args:
        models_dir: Directory containing model files
        dataset_name: Dataset name for model files
        
    Returns:
        Loaded TF-IDF service instance
    """
    service = OptimizedTFIDFService(models_dir)
    
    if service.load_models(dataset_name):
        return service
    else:
        raise RuntimeError(f"Failed to load TF-IDF models for {dataset_name}")

# Example usage
if __name__ == "__main__":
    # Test the service
    service = OptimizedTFIDFService("models")
    
    if service.load_models("antique"):
        # Test search
        results = service.search_with_inverted_index("antique furniture restoration", top_k=5)
        print("Search Results:")
        for result in results:
            print(f"Doc {result['doc_id']}: {result['score']:.4f}")
        
        # Test query analysis
        analysis = service.analyze_query("antique furniture restoration")
        print(f"\nQuery Analysis: {analysis}")
        
        # Get statistics
        stats = service.get_service_statistics()
        print(f"\nService Statistics: {stats}")
    else:
        print("Failed to load models")
