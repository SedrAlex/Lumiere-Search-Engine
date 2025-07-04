#!/usr/bin/env python3
"""
Enhanced TF-IDF Service V2 with Advanced Text Processing
Implements spell checking, lemmatization, stemming, and normalization
while preserving MAP evaluation performance. Includes inverted index optimization.
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

class EnhancedTFIDFServiceV2:
    """
    Enhanced TF-IDF service with advanced text processing and inverted index.
    Designed to maintain MAP > 0.4 while improving text normalization.
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 enable_spell_check: bool = True,
                 enable_lemmatization: bool = True,
                 enable_stemming: bool = True,
                 language: str = 'english'):
        """
        Initialize the enhanced TF-IDF service.
        
        Args:
            models_dir: Directory for model storage
            enable_spell_check: Whether to enable spell checking
            enable_lemmatization: Whether to enable lemmatization
            enable_stemming: Whether to enable stemming
            language: Language for processing
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced components with conservative settings to preserve MAP
        self.text_cleaner = EnhancedTextCleaningService(
            language=language,
            enable_spell_check=enable_spell_check
        )
        
        self.enhanced_tokenizer = EnhancedTokenizer(
            enable_spell_check=enable_spell_check,
            enable_lemmatization=enable_lemmatization,
            enable_stemming=enable_stemming,
            language=language,
            min_token_length=3,  # Conservative to preserve MAP
            max_token_length=50
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
        
        logger.info(f"Enhanced TF-IDF V2 service initialized with advanced processing")
    
    def create_optimized_vectorizer(self, **vectorizer_params) -> TfidfVectorizer:
        """
        Create optimized TF-IDF vectorizer with enhanced tokenizer.
        Parameters are tuned for high MAP performance.
        
        Args:
            **vectorizer_params: Additional parameters for TfidfVectorizer
            
        Returns:
            Configured TfidfVectorizer optimized for MAP > 0.4
        """
        # Optimized parameters for high MAP performance
        default_params = {
            'max_features': 100000,    # Large vocabulary for better coverage
            'min_df': 2,               # Remove very rare terms
            'max_df': 0.85,            # Remove very common terms (more conservative)
            'ngram_range': (1, 3),     # Include trigrams for better phrase matching
            'sublinear_tf': True,      # Log scaling for TF
            'norm': 'l2',              # L2 normalization
            'smooth_idf': True,        # Smooth IDF weights
            'use_idf': True,           # Use IDF weighting
            'tokenizer': self.enhanced_tokenizer,  # Enhanced tokenizer with preprocessing
            'preprocessor': None,      # All preprocessing handled by tokenizer
            'lowercase': False,        # Handled by tokenizer
            'stop_words': None,        # Handled by tokenizer
            'token_pattern': None,     # Using custom tokenizer
        }
        
        # Update with user parameters
        default_params.update(vectorizer_params)
        
        logger.info(f"Creating TF-IDF vectorizer with params: {default_params}")
        return TfidfVectorizer(**default_params)
    
    def train_enhanced_tfidf(self, 
                           documents: List[str], 
                           doc_ids: List[str],
                           vectorizer_params: Optional[Dict] = None,
                           build_inverted_index: bool = True) -> Dict:
        """
        Train enhanced TF-IDF model with advanced text processing.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            vectorizer_params: Parameters for TfidfVectorizer
            build_inverted_index: Whether to build inverted index
            
        Returns:
            Training statistics
        """
        logger.info(f"Training Enhanced TF-IDF V2 on {len(documents)} documents...")
        
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents must match number of doc_ids")
        
        # Step 1: Clean text using enhanced text cleaner (basic cleaning only)
        logger.info("Step 1: Applying enhanced text cleaning...")
        cleaned_texts = []
        valid_docs = []
        
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, documents)):
            # Use only basic cleaning - tokenizer will handle advanced processing
            cleaned_text = self.text_cleaner.clean_text_basic(doc_text)
            
            if cleaned_text.strip():  # Keep non-empty documents
                cleaned_texts.append(cleaned_text)
                valid_docs.append((doc_id, doc_text, cleaned_text))
            else:
                logger.debug(f"Document {doc_id} is empty after cleaning")
        
        logger.info(f"Valid documents after cleaning: {len(valid_docs)}")
        
        # Step 2: Create enhanced vectorizer with optimized parameters
        vectorizer_params = vectorizer_params or {}
        self.vectorizer = self.create_optimized_vectorizer(**vectorizer_params)
        
        # Step 3: Fit and transform documents (tokenizer handles advanced processing)
        logger.info("Step 2: Fitting TF-IDF vectorizer with enhanced tokenization...")
        
        # Extract cleaned texts for training
        training_texts = [item[2] for item in valid_docs]
        
        # Fit and transform with enhanced tokenizer doing the heavy lifting
        self.tfidf_matrix = self.vectorizer.fit_transform(training_texts)
        
        # Step 4: Create document mappings
        valid_doc_ids = [item[0] for item in valid_docs]
        valid_original_texts = [item[1] for item in valid_docs]
        valid_cleaned_texts = [item[2] for item in valid_docs]
        
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(valid_doc_ids)}
        self.idx_to_doc_id = {idx: doc_id for doc_id, idx in self.doc_id_to_idx.items()}
        
        # Step 5: Create document metadata
        self.document_metadata = {
            doc_id: {
                'original_text': valid_original_texts[idx],
                'cleaned_text': valid_cleaned_texts[idx],
                'index': idx
            }
            for doc_id, idx in self.doc_id_to_idx.items()
        }
        
        # Step 6: Build inverted index if requested
        if build_inverted_index:
            logger.info("Step 3: Building optimized inverted index...")
            self.inverted_index = self._build_optimized_inverted_index()
        
        # Step 7: Calculate comprehensive training statistics
        self.training_stats = self._calculate_training_statistics(documents, valid_docs)
        
        self.is_trained = True
        logger.info("✓ Enhanced TF-IDF V2 training completed successfully!")
        
        return self.training_stats
    
    def _build_optimized_inverted_index(self) -> Dict:
        """
        Build optimized inverted index from TF-IDF matrix.
        Includes term frequency and document frequency statistics.
        
        Returns:
            Optimized inverted index dictionary
        """
        inverted_index = defaultdict(lambda: {
            'postings': [],
            'df': 0,
            'max_tfidf': 0.0,
            'avg_tfidf': 0.0
        })
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Convert to COO format for efficient iteration
        coo_matrix = self.tfidf_matrix.tocoo()
        
        # Build index with statistics
        term_stats = defaultdict(list)
        
        for doc_idx, term_idx, tfidf_score in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            term = feature_names[term_idx]
            doc_id = self.idx_to_doc_id[doc_idx]
            
            # Add posting
            inverted_index[term]['postings'].append((doc_id, float(tfidf_score)))
            term_stats[term].append(float(tfidf_score))
        
        # Calculate statistics and sort postings
        for term in inverted_index:
            scores = term_stats[term]
            inverted_index[term]['df'] = len(scores)
            inverted_index[term]['max_tfidf'] = max(scores)
            inverted_index[term]['avg_tfidf'] = sum(scores) / len(scores)
            
            # Sort postings by TF-IDF score (descending)
            inverted_index[term]['postings'].sort(key=lambda x: x[1], reverse=True)
        
        return dict(inverted_index)
    
    def _calculate_training_statistics(self, original_docs: List[str], valid_docs: List) -> Dict:
        """Calculate comprehensive training statistics."""
        
        # Text processing statistics
        original_lengths = [len(doc) for doc in original_docs]
        cleaned_lengths = [len(item[2]) for item in valid_docs]
        
        # Vocabulary analysis
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        
        return {
            # Document statistics
            'total_documents': len(original_docs),
            'valid_documents': len(valid_docs),
            'documents_filtered': len(original_docs) - len(valid_docs),
            'filter_rate': (len(original_docs) - len(valid_docs)) / len(original_docs) * 100,
            
            # Text processing statistics
            'avg_original_length': np.mean(original_lengths),
            'avg_cleaned_length': np.mean(cleaned_lengths),
            'text_reduction_ratio': 1 - (np.mean(cleaned_lengths) / np.mean(original_lengths)),
            
            # TF-IDF matrix statistics
            'matrix_shape': self.tfidf_matrix.shape,
            'vocabulary_size': len(feature_names),
            'non_zero_entries': int(self.tfidf_matrix.nnz),
            'sparsity': float((1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100),
            'avg_doc_length': float(self.tfidf_matrix.nnz / self.tfidf_matrix.shape[0]),
            
            # Vocabulary statistics
            'min_idf': float(np.min(idf_scores)),
            'max_idf': float(np.max(idf_scores)),
            'avg_idf': float(np.mean(idf_scores)),
            
            # Inverted index statistics
            'inverted_index_terms': len(self.inverted_index) if self.inverted_index else 0,
            'avg_postings_per_term': np.mean([len(data['postings']) for data in self.inverted_index.values()]) if self.inverted_index else 0,
            
            # Configuration
            'vectorizer_params': self.vectorizer.get_params(),
            'text_cleaner_info': self.text_cleaner.get_service_info(),
            'tokenizer_info': self.enhanced_tokenizer.get_tokenizer_info(),
            
            # Processing features enabled
            'spell_check_enabled': self.enhanced_tokenizer.enable_spell_check,
            'lemmatization_enabled': self.enhanced_tokenizer.enable_lemmatization,
            'stemming_enabled': self.enhanced_tokenizer.enable_stemming,
        }
    
    def search_with_enhanced_inverted_index(self, 
                                          query: str, 
                                          top_k: int = 10,
                                          use_cache: bool = True,
                                          fusion_alpha: float = 0.7) -> List[Dict]:
        """
        Enhanced search using inverted index with TF-IDF fusion.
        Optimized for high MAP performance.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            use_cache: Whether to use search cache
            fusion_alpha: Weight for TF-IDF similarity (vs inverted index score)
            
        Returns:
            List of search results with scores
        """
        if not (self.is_trained or self.is_loaded):
            raise RuntimeError("Model not trained or loaded. Train or load a model first.")
        
        if not self.inverted_index:
            logger.warning("No inverted index available, falling back to full matrix search")
            return self.search_with_full_matrix(query, top_k, use_cache)
        
        # Check cache
        cache_key = f"enhanced_{query}_{top_k}_{fusion_alpha}"
        if use_cache and cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Clean query using basic cleaning (tokenizer will handle advanced processing)
        cleaned_query = self.text_cleaner.clean_text_basic(query)
        
        if not cleaned_query.strip():
            return []
        
        # Get query TF-IDF vector (uses enhanced tokenizer internally)
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Get query terms using enhanced tokenizer
        query_terms = self.enhanced_tokenizer.tokenize(cleaned_query)
        
        if not query_terms:
            return []
        
        # Collect candidate documents from inverted index
        candidate_docs = set()
        term_doc_scores = defaultdict(float)
        term_weights = {}
        
        # Calculate term weights based on IDF and query term frequency
        query_term_freq = defaultdict(int)
        for term in query_terms:
            query_term_freq[term] += 1
        
        # Get candidates and calculate inverted index scores
        for term in set(query_terms):
            if term in self.inverted_index:
                term_data = self.inverted_index[term]
                postings = term_data['postings']
                
                # Term weight based on IDF and query frequency
                term_weight = query_term_freq[term] * np.log(1 + 1 / max(term_data['df'], 1))
                term_weights[term] = term_weight
                
                # Add candidate documents with weighted scores
                for doc_id, tfidf_score in postings:
                    candidate_docs.add(doc_id)
                    term_doc_scores[doc_id] += term_weight * tfidf_score
        
        if not candidate_docs:
            return []
        
        # Convert to indices for TF-IDF similarity calculation
        candidate_indices = [
            self.doc_id_to_idx[doc_id] 
            for doc_id in candidate_docs 
            if doc_id in self.doc_id_to_idx
        ]
        
        if not candidate_indices:
            return []
        
        # Calculate TF-IDF similarities for candidates
        candidate_matrix = self.tfidf_matrix[candidate_indices]
        tfidf_similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        # Fuse scores: weighted combination of TF-IDF similarity and inverted index score
        fused_scores = []
        max_index_score = max(term_doc_scores.values()) if term_doc_scores else 1.0
        
        for i, idx in enumerate(candidate_indices):
            doc_id = self.idx_to_doc_id[idx]
            tfidf_sim = tfidf_similarities[i]
            index_score = term_doc_scores.get(doc_id, 0) / max_index_score  # Normalize
            
            # Weighted fusion
            fused_score = fusion_alpha * tfidf_sim + (1 - fusion_alpha) * index_score
            fused_scores.append(fused_score)
        
        # Get top results
        top_indices = np.argsort(fused_scores)[-top_k:][::-1]
        
        results = []
        for rank, candidate_idx in enumerate(top_indices):
            original_idx = candidate_indices[candidate_idx]
            doc_id = self.idx_to_doc_id[original_idx]
            
            result = {
                'doc_id': doc_id,
                'score': float(fused_scores[candidate_idx]),
                'tfidf_similarity': float(tfidf_similarities[candidate_idx]),
                'index_score': float(term_doc_scores.get(doc_id, 0) / max_index_score),
                'rank': rank + 1,
                'fusion_alpha': fusion_alpha
            }
            
            # Add document text if metadata is available
            if self.document_metadata and doc_id in self.document_metadata:
                result['document_text'] = self.document_metadata[doc_id]['original_text'][:200] + "..."
            
            results.append(result)
        
        # Cache results
        if use_cache:
            self.search_cache[cache_key] = results
        
        return results
    
    def search_with_full_matrix(self, 
                              query: str, 
                              top_k: int = 10,
                              use_cache: bool = True) -> List[Dict]:
        """
        Search using full TF-IDF matrix (fallback method).
        
        Args:
            query: Search query
            top_k: Number of top results to return
            use_cache: Whether to use search cache
            
        Returns:
            List of search results with scores
        """
        if not (self.is_trained or self.is_loaded):
            raise RuntimeError("Model not trained or loaded. Train or load a model first.")
        
        # Check cache
        cache_key = f"full_{query}_{top_k}"
        if use_cache and cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Clean query
        cleaned_query = self.text_cleaner.clean_text_basic(query)
        
        if not cleaned_query.strip():
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Calculate similarities with all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            doc_id = self.idx_to_doc_id[idx]
            
            result = {
                'doc_id': doc_id,
                'score': float(similarities[idx]),
                'rank': rank + 1,
                'method': 'full_matrix'
            }
            
            # Add document text if metadata is available
            if self.document_metadata and doc_id in self.document_metadata:
                result['document_text'] = self.document_metadata[doc_id]['original_text'][:200] + "..."
            
            results.append(result)
        
        # Cache results
        if use_cache:
            self.search_cache[cache_key] = results
        
        return results
    
    def save_enhanced_models(self, dataset_name: str) -> bool:
        """
        Save all trained models and components.
        
        Args:
            dataset_name: Name for the dataset/model files
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.is_trained:
            logger.error("No trained model to save")
            return False
        
        try:
            logger.info(f"Saving Enhanced TF-IDF V2 models for {dataset_name}...")
            
            # Save TF-IDF vectorizer
            vectorizer_path = self.models_dir / f"enhanced_tfidf_vectorizer_{dataset_name}.joblib"
            joblib.dump(self.vectorizer, vectorizer_path)
            
            # Save TF-IDF matrix
            matrix_path = self.models_dir / f"enhanced_tfidf_matrix_{dataset_name}.joblib"
            joblib.dump(self.tfidf_matrix, matrix_path)
            
            # Save optimized inverted index
            if self.inverted_index:
                index_path = self.models_dir / f"enhanced_inverted_index_{dataset_name}.pkl"
                with open(index_path, 'wb') as f:
                    pickle.dump(self.inverted_index, f)
            
            # Save document mappings
            mappings_path = self.models_dir / f"enhanced_doc_mappings_{dataset_name}.json"
            with open(mappings_path, 'w') as f:
                json.dump({
                    'doc_id_to_idx': self.doc_id_to_idx,
                    'idx_to_doc_id': self.idx_to_doc_id
                }, f)
            
            # Save document metadata
            metadata_path = self.models_dir / f"enhanced_document_metadata_{dataset_name}.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.document_metadata, f)
            
            # Save training statistics
            stats_path = self.models_dir / f"enhanced_training_statistics_{dataset_name}.json"
            with open(stats_path, 'w') as f:
                # Convert numpy types for JSON serialization
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
            
            logger.info("✓ All Enhanced TF-IDF V2 models saved successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error saving enhanced models: {str(e)}")
            return False
    
    def load_enhanced_models(self, dataset_name: str) -> bool:
        """
        Load pre-trained enhanced models and components.
        
        Args:
            dataset_name: Name of the dataset/model files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading Enhanced TF-IDF V2 models for {dataset_name}...")
            
            # Define file paths
            vectorizer_path = self.models_dir / f"enhanced_tfidf_vectorizer_{dataset_name}.joblib"
            matrix_path = self.models_dir / f"enhanced_tfidf_matrix_{dataset_name}.joblib"
            index_path = self.models_dir / f"enhanced_inverted_index_{dataset_name}.pkl"
            mappings_path = self.models_dir / f"enhanced_doc_mappings_{dataset_name}.json"
            metadata_path = self.models_dir / f"enhanced_document_metadata_{dataset_name}.json"
            stats_path = self.models_dir / f"enhanced_training_statistics_{dataset_name}.json"
            
            # Check required files
            required_files = [vectorizer_path, matrix_path, mappings_path]
            missing_files = [f for f in required_files if not f.exists()]
            
            if missing_files:
                logger.error(f"Missing required model files: {missing_files}")
                return False
            
            # Load components
            self.vectorizer = joblib.load(vectorizer_path)
            self.tfidf_matrix = joblib.load(matrix_path)
            
            # Load optimized inverted index if available
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
            
            logger.info(f"✓ Enhanced TF-IDF V2 models loaded successfully!")
            logger.info(f"  - Documents: {self.tfidf_matrix.shape[0]}")
            logger.info(f"  - Vocabulary: {len(self.vectorizer.vocabulary_)}")
            if self.inverted_index:
                logger.info(f"  - Index terms: {len(self.inverted_index)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading enhanced models: {str(e)}")
            self.is_loaded = False
            return False
    
    def get_enhanced_service_info(self) -> Dict:
        """Get comprehensive enhanced service information."""
        info = {
            'service_version': 'Enhanced TF-IDF V2',
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
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.search_cache.clear()
        self.text_cleaner.clear_caches()
        self.enhanced_tokenizer.clear_cache()
        logger.info("All enhanced service caches cleared")

# Factory functions
def create_enhanced_tfidf_service_v2(models_dir: str = "models", 
                                   enable_spell_check: bool = True,
                                   enable_lemmatization: bool = True,
                                   enable_stemming: bool = True,
                                   language: str = 'english') -> EnhancedTFIDFServiceV2:
    """
    Factory function to create enhanced TF-IDF service V2.
    
    Args:
        models_dir: Directory for model storage
        enable_spell_check: Whether to enable spell checking
        enable_lemmatization: Whether to enable lemmatization
        enable_stemming: Whether to enable stemming
        language: Language for processing
        
    Returns:
        EnhancedTFIDFServiceV2 instance
    """
    return EnhancedTFIDFServiceV2(
        models_dir=models_dir,
        enable_spell_check=enable_spell_check,
        enable_lemmatization=enable_lemmatization,
        enable_stemming=enable_stemming,
        language=language
    )

def create_conservative_tfidf_service(models_dir: str = "models", 
                                    language: str = 'english') -> EnhancedTFIDFServiceV2:
    """
    Create conservative TF-IDF service for maximum MAP preservation.
    
    Args:
        models_dir: Directory for model storage
        language: Language for processing
        
    Returns:
        EnhancedTFIDFServiceV2 with conservative settings
    """
    return EnhancedTFIDFServiceV2(
        models_dir=models_dir,
        enable_spell_check=False,   # Disable for maximum precision
        enable_lemmatization=True,  # Keep for normalization
        enable_stemming=True,       # Keep for vocabulary reduction
        language=language
    )

# Example usage
if __name__ == "__main__":
    # Test the enhanced service
    service = create_enhanced_tfidf_service_v2(enable_spell_check=True)
    
    # Sample documents
    docs = [
        "Beautiful antique furniture from the Victorian era with intricate woodwork",
        "Vintage collectibles and rare items for sale including old coins",
        "Classic automobiles and automotive memorabilia from the 1950s",
        "Old books manuscripts and historical documents from 18th century"
    ]
    doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    
    # Train the enhanced model
    stats = service.train_enhanced_tfidf(docs, doc_ids)
    print("Enhanced Training Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test enhanced search
    results = service.search_with_enhanced_inverted_index("antique furniture", top_k=3)
    print("\nEnhanced Search Results:")
    for result in results:
        print(f"  Rank {result['rank']}: {result['doc_id']} (Score: {result['score']:.4f})")
    
    # Get service info
    info = service.get_enhanced_service_info()
    print(f"\nEnhanced Service Info: {info}")
