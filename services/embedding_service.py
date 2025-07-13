#!/usr/bin/env python3

import logging
import numpy as np
import joblib
import os
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for handling embedding operations with optional FAISS support
    """
    
    def __init__(self):
        self.models = {}
        self.embeddings = {}
        self.documents = {}
        self.faiss_indices = {}
        self.faiss_available = self._check_faiss_availability()
        
    def _check_faiss_availability(self) -> bool:
        """Check if FAISS is available"""
        try:
            import faiss
            logger.info("FAISS is available")
            return True
        except ImportError:
            logger.warning("FAISS is not available. Install with: pip install faiss-cpu")
            return False
    
    def load_dataset(self, 
                    dataset_name: str,
                    embedding_model_path: str,
                    embeddings_path: str,
                    documents_path: str,
                    use_faiss: bool = False) -> Dict[str, Any]:
        """Load embedding model and precomputed embeddings for a dataset"""
        try:
            # Load the embedding model
            if os.path.exists(embedding_model_path):
                logger.info(f"Loading embedding model from {embedding_model_path}")
                self.models[dataset_name] = SentenceTransformer(embedding_model_path)
            else:
                logger.warning(f"Model path {embedding_model_path} not found, using default model")
                self.models[dataset_name] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load precomputed embeddings
            if os.path.exists(embeddings_path):
                logger.info(f"Loading precomputed embeddings from {embeddings_path}")
                self.embeddings[dataset_name] = joblib.load(embeddings_path)
                logger.info(f"Loaded embeddings shape: {self.embeddings[dataset_name].shape}")
            else:
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
            # Load documents
            if os.path.exists(documents_path):
                logger.info(f"Loading documents from {documents_path}")
                docs_data = joblib.load(documents_path)
                self.documents[dataset_name] = {
                    'doc_ids': docs_data['doc_ids'],
                    'texts': docs_data['texts']
                }
                logger.info(f"Loaded {len(docs_data['doc_ids'])} documents")
            else:
                raise FileNotFoundError(f"Documents file not found: {documents_path}")
            
            # Create FAISS index if requested and available
            if use_faiss and self.faiss_available:
                self._create_faiss_index(dataset_name)
            
            return {
                'dataset': dataset_name,
                'model_loaded': True,
                'embeddings_loaded': True,
                'documents_loaded': True,
                'faiss_enabled': use_faiss and self.faiss_available,
                'document_count': len(self.documents[dataset_name]['doc_ids'])
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise e
    
    def _create_faiss_index(self, dataset_name: str):
        """Create FAISS index for fast similarity search"""
        try:
            import faiss
            
            embeddings = self.embeddings[dataset_name]
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            self.faiss_indices[dataset_name] = index
            logger.info(f"Created FAISS index for {dataset_name} with {index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index for {dataset_name}: {e}")
            raise e
    
    def search(self, 
               query: str, 
               dataset: str, 
               top_k: int = 10,
               use_faiss: bool = False) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embeddings
        
        Args:
            query: Search query
            dataset: Dataset name
            top_k: Number of results to return
            use_faiss: Whether to use FAISS for search
        """
        try:
            if dataset not in self.models:
                raise ValueError(f"Dataset {dataset} not loaded")
            
            # Encode query
            query_embedding = self.models[dataset].encode([query], normalize_embeddings=True)
            
            # Search using FAISS or sklearn
            if use_faiss and dataset in self.faiss_indices:
                return self._search_with_faiss(query_embedding, dataset, top_k)
            else:
                return self._search_with_sklearn(query_embedding, dataset, top_k)
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise e
    
    def _search_with_faiss(self, 
                          query_embedding: np.ndarray, 
                          dataset: str, 
                          top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS index"""
        try:
            import faiss
            
            index = self.faiss_indices[dataset]
            
            # Normalize query embedding
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = index.search(query_embedding, top_k)
            
            # Format results
            results = []
            doc_ids = self.documents[dataset]['doc_ids']
            doc_texts = self.documents[dataset]['texts']
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1:  # Valid index
                    results.append({
                        'doc_id': doc_ids[idx],
                        'text': doc_texts[idx],
                        'similarity': float(similarity),
                        'rank': i + 1
                    })
            
            logger.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            raise e
    
    def _search_with_sklearn(self, 
                           query_embedding: np.ndarray, 
                           dataset: str, 
                           top_k: int) -> List[Dict[str, Any]]:
        """Search using sklearn cosine similarity"""
        try:
            embeddings = self.embeddings[dataset]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Format results
            results = []
            doc_ids = self.documents[dataset]['doc_ids']
            doc_texts = self.documents[dataset]['texts']
            
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:  # Only include positive similarities
                    results.append({
                        'doc_id': doc_ids[idx],
                        'text': doc_texts[idx],
                        'similarity': float(similarities[idx]),
                        'rank': i + 1
                    })
            
            logger.info(f"Sklearn search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Sklearn search error: {e}")
            raise e
    
    def encode_query(self, query: str, dataset: str) -> np.ndarray:
        """Encode a query using the dataset's embedding model"""
        if dataset not in self.models:
            raise ValueError(f"Dataset {dataset} not loaded")
        
        return self.models[dataset].encode([query], normalize_embeddings=True)
    
    def get_dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get information about a loaded dataset"""
        if dataset not in self.models:
            raise ValueError(f"Dataset {dataset} not loaded")
        
        return {
            'dataset': dataset,
            'model_name': self.models[dataset].get_sentence_embedding_dimension(),
            'document_count': len(self.documents[dataset]['doc_ids']) if dataset in self.documents else 0,
            'embedding_dimension': self.embeddings[dataset].shape[1] if dataset in self.embeddings else 0,
            'faiss_enabled': dataset in self.faiss_indices,
            'faiss_available': self.faiss_available
        }
    
    def get_available_datasets(self) -> List[str]:
        """Get list of loaded datasets"""
        return list(self.models.keys())
    
    def reload_dataset(self, dataset: str, use_faiss: bool = False) -> Dict[str, Any]:
        """Reload a dataset (useful for switching FAISS on/off)"""
        if dataset not in self.models:
            raise ValueError(f"Dataset {dataset} not loaded")
        
        # Remove existing FAISS index if present
        if dataset in self.faiss_indices:
            del self.faiss_indices[dataset]
        
        # Create new FAISS index if requested
        if use_faiss and self.faiss_available:
            self._create_faiss_index(dataset)
        
        return {
            'dataset': dataset,
            'reloaded': True,
            'faiss_enabled': use_faiss and self.faiss_available
        }
    
    def install_faiss_instructions(self) -> Dict[str, str]:
        """Get instructions for installing FAISS"""
        return {
            'faiss_available': self.faiss_available,
            'install_cpu': 'pip install faiss-cpu',
            'install_gpu': 'pip install faiss-gpu',
            'note': 'FAISS provides significant speed improvements for large document collections'
        }
