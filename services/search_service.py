#!/usr/bin/env python3

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import time

from services.embedding_service import EmbeddingService
from services.database.db_service import DatabaseService
from services.text_preprocessing.unified_text_processor import UnifiedTextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchService:
    """
    Unified search service that handles both Quora and Antique datasets
    with support for TF-IDF, Embedding, and Hybrid search methods
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.db_service = DatabaseService()
        self.text_processor = UnifiedTextProcessor()
        self.loaded_datasets = set()
        self.dataset_configs = {
            'quora': {
                'default_method': 'hybrid-quora',
                'embedding_model_path': '/Users/raafatmhanna/Downloads/quora_Embeddings/sentence-transformers_all-MiniLM-L6-v2',
                'tfidf_model_path': '/Users/raafatmhanna/Downloads/quora_tfidf_models/',
                'embeddings_path': '/Users/raafatmhanna/Downloads/quora_Embeddings/doc_embeddings.joblib',
                'documents_path': '/Users/raafatmhanna/Downloads/quora_Embeddings/documents_final.joblib',
                'db_path': '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_documents.db'
            },
            'antique': {
                'default_method': 'hybrid-antique',
                'embedding_model_path': '/content/drive/MyDrive/Antique_Embeddings/sentence-transformers_all-MiniLM-L6-v2',
                'tfidf_model_path': '/content/drive/MyDrive/Antique_TF-IDF/',
                'embeddings_path': '/content/drive/MyDrive/Antique_Embeddings/doc_embeddings.joblib',
                'documents_path': '/content/drive/MyDrive/Antique_Embeddings/documents_final.joblib',
                'db_path': '/content/drive/MyDrive/downloads/documents.tsv'
            }
        }
        
    def load_dataset(self, dataset_name: str, use_faiss: bool = False) -> Dict[str, Any]:
        """Load a specific dataset with its models and indices"""
        try:
            if dataset_name not in self.dataset_configs:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            config = self.dataset_configs[dataset_name]
            
            # Load embedding service for this dataset
            self.embedding_service.load_dataset(
                dataset_name=dataset_name,
                embedding_model_path=config['embedding_model_path'],
                embeddings_path=config['embeddings_path'],
                documents_path=config['documents_path'],
                use_faiss=use_faiss
            )
            
            # Load TF-IDF models
            self.text_processor.load_tfidf_models(
                dataset_name=dataset_name,
                model_path=config['tfidf_model_path']
            )
            
            # Load database
            self.db_service.load_dataset(dataset_name, config['db_path'])
            
            self.loaded_datasets.add(dataset_name)
            
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return {
                'dataset': dataset_name,
                'status': 'loaded',
                'use_faiss': use_faiss,
                'available_methods': ['tfidf', 'embedding', 'hybrid']
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load dataset {dataset_name}: {str(e)}")
    
    def search(self, 
               query: str, 
               dataset: str = 'quora', 
               method: str = 'hybrid-quora',
               top_k: int = 10,
               use_faiss: bool = False) -> Dict[str, Any]:
        """
        Perform search across datasets with different methods
        
        Args:
            query: Search query
            dataset: Dataset name ('quora' or 'antique')
            method: Search method ('tfidf', 'embedding', 'hybrid-quora', 'hybrid-antique')
            top_k: Number of results to return
            use_faiss: Whether to use FAISS for embedding search
        """
        try:
            # Validate dataset
            if dataset not in self.dataset_configs:
                raise ValueError(f"Unknown dataset: {dataset}")
                
            # Auto-load dataset if not loaded
            if dataset not in self.loaded_datasets:
                self.load_dataset(dataset, use_faiss=use_faiss)
            
            # Set default method if not specified
            if method == 'auto':
                method = self.dataset_configs[dataset]['default_method']
            
            # Process query based on method
            if method == 'tfidf':
                return self._search_tfidf(query, dataset, top_k)
            elif method == 'embedding':
                return self._search_embedding(query, dataset, top_k, use_faiss)
            elif method.startswith('hybrid'):
                return self._search_hybrid(query, dataset, top_k, use_faiss)
            else:
                raise ValueError(f"Unknown search method: {method}")
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    def _search_tfidf(self, query: str, dataset: str, top_k: int) -> Dict[str, Any]:
        """Perform TF-IDF search"""
        try:
            # Clean and process query
            cleaned_query = self.text_processor.clean_text(query, dataset)
            
            # Get TF-IDF results
            tfidf_results = self.text_processor.search_tfidf(
                query=cleaned_query,
                dataset=dataset,
                top_k=top_k * 2  # Get more results for better ranking
            )
            
            # Format results
            results = []
            for i, result in enumerate(tfidf_results[:top_k]):
                results.append({
                    'rank': i + 1,
                    'doc_id': str(result['doc_id']),
                    'document': result['text'],
                    'score': float(result['score']),
                    'method': 'tfidf'
                })
            
            return {
                'query': query,
                'cleaned_query': cleaned_query,
                'dataset': dataset,
                'method': 'tfidf',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            logger.error(f"TF-IDF search error: {e}")
            raise e
    
    def _search_embedding(self, query: str, dataset: str, top_k: int, use_faiss: bool) -> Dict[str, Any]:
        """Perform embedding search"""
        try:
            # Get embedding results
            embedding_results = self.embedding_service.search(
                query=query,
                dataset=dataset,
                top_k=top_k * 2,
                use_faiss=use_faiss
            )
            
            # Format results
            results = []
            for i, result in enumerate(embedding_results[:top_k]):
                results.append({
                    'rank': i + 1,
                    'doc_id': str(result['doc_id']),
                    'document': result['text'],
                    'score': float(result['similarity']),
                    'method': 'embedding'
                })
            
            return {
                'query': query,
                'dataset': dataset,
                'method': 'embedding',
                'use_faiss': use_faiss,
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            logger.error(f"Embedding search error: {e}")
            raise e
    
    def _search_hybrid(self, query: str, dataset: str, top_k: int, use_faiss: bool) -> Dict[str, Any]:
        """Perform hybrid search using both TF-IDF and embeddings"""
        try:
            # Run both searches in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks
                tfidf_future = executor.submit(self._search_tfidf, query, dataset, top_k * 2)
                embedding_future = executor.submit(self._search_embedding, query, dataset, top_k * 2, use_faiss)
                
                # Get results
                tfidf_response = tfidf_future.result()
                embedding_response = embedding_future.result()
            
            tfidf_results = tfidf_response['results']
            embedding_results = embedding_response['results']
            
            # Perform fusion
            fused_results = self._fusion_search_results(
                tfidf_results=tfidf_results,
                embedding_results=embedding_results,
                top_k=top_k,
                method=f'hybrid-{dataset}'
            )
            
            return {
                'query': query,
                'cleaned_query': tfidf_response.get('cleaned_query', query),
                'dataset': dataset,
                'method': f'hybrid-{dataset}',
                'use_faiss': use_faiss,
                'results': fused_results,
                'total_results': len(fused_results),
                'fusion_method': 'reciprocal_rank_fusion',
                'tfidf_results_count': len(tfidf_results),
                'embedding_results_count': len(embedding_results)
            }
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            raise e
    
    def _fusion_search_results(self, 
                              tfidf_results: List[Dict], 
                              embedding_results: List[Dict],
                              top_k: int,
                              method: str) -> List[Dict]:
        """Fuse TF-IDF and embedding results using Reciprocal Rank Fusion"""
        try:
            # Create document score maps
            tfidf_doc_scores = {result['doc_id']: result['score'] for result in tfidf_results}
            embedding_doc_scores = {result['doc_id']: result['score'] for result in embedding_results}
            
            # Create document info maps
            tfidf_doc_info = {result['doc_id']: result for result in tfidf_results}
            embedding_doc_info = {result['doc_id']: result for result in embedding_results}
            
            # Get all unique document IDs
            all_doc_ids = set(tfidf_doc_scores.keys()) | set(embedding_doc_scores.keys())
            
            # Create rank-based dictionaries for RRF
            tfidf_ranks = {}
            embedding_ranks = {}
            
            # Create TF-IDF ranks
            tfidf_sorted = sorted(tfidf_doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(tfidf_sorted, 1):
                tfidf_ranks[doc_id] = rank
                
            # Create Embedding ranks  
            embedding_sorted = sorted(embedding_doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(embedding_sorted, 1):
                embedding_ranks[doc_id] = rank
            
            # Weighted RRF parameters (adjust based on dataset)
            k = 60  # RRF constant
            if method == 'hybrid-quora':
                embedding_weight = 0.75  # Quora benefits more from semantic similarity
                tfidf_weight = 0.25
            else:  # hybrid-antique
                embedding_weight = 0.65  # Antique has more balanced needs
                tfidf_weight = 0.35
            
            # Calculate Weighted RRF scores
            fusion_scores = {}
            for doc_id in all_doc_ids:
                tfidf_rank = tfidf_ranks.get(doc_id, len(tfidf_doc_scores) + 1)
                embedding_rank = embedding_ranks.get(doc_id, len(embedding_doc_scores) + 1)
                
                # Weighted RRF formula
                weighted_rrf_score = (embedding_weight / (k + embedding_rank)) + (tfidf_weight / (k + tfidf_rank))
                
                # Boost documents that appear in both methods
                boost_factor = 1.0
                if doc_id in tfidf_doc_scores and doc_id in embedding_doc_scores:
                    boost_factor = 1.3  # 30% boost for documents found by both methods
                
                fusion_scores[doc_id] = weighted_rrf_score * boost_factor
            
            # Sort documents by fusion score
            sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Build final results
            results = []
            for i, (doc_id, fusion_score) in enumerate(sorted_docs):
                # Get document info from either service
                doc_info = tfidf_doc_info.get(doc_id) or embedding_doc_info.get(doc_id)
                
                if doc_info:
                    # Get original scores
                    original_tfidf_score = tfidf_doc_scores.get(doc_id, 0.0)
                    original_embedding_score = embedding_doc_scores.get(doc_id, 0.0)
                    
                    results.append({
                        'rank': i + 1,
                        'doc_id': doc_id,
                        'document': doc_info['document'],
                        'hybrid_score': float(fusion_score),
                        'tfidf_score': float(original_tfidf_score),
                        'embedding_score': float(original_embedding_score),
                        'method': method
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Fusion error: {e}")
            raise e
    
    def get_available_datasets(self) -> Dict[str, Any]:
        """Get information about available datasets"""
        return {
            'available_datasets': list(self.dataset_configs.keys()),
            'loaded_datasets': list(self.loaded_datasets),
            'dataset_configs': {
                name: {
                    'default_method': config['default_method'],
                    'available_methods': ['tfidf', 'embedding', 'hybrid']
                } for name, config in self.dataset_configs.items()
            }
        }
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        
        info = {
            'name': dataset_name,
            'loaded': dataset_name in self.loaded_datasets,
            'default_method': config['default_method'],
            'available_methods': ['tfidf', 'embedding', 'hybrid']
        }
        
        if dataset_name in self.loaded_datasets:
            # Get additional info from loaded dataset
            embedding_info = self.embedding_service.get_dataset_info(dataset_name)
            tfidf_info = self.text_processor.get_dataset_info(dataset_name)
            
            info.update({
                'document_count': embedding_info.get('document_count', 0),
                'embedding_model': embedding_info.get('model_name', 'N/A'),
                'tfidf_features': tfidf_info.get('feature_count', 0),
                'faiss_enabled': embedding_info.get('faiss_enabled', False)
            })
        
        return info
