#!/usr/bin/env python3
"""
FAISS Index Service
Provides optimized FAISS indexing and similarity search functionality
"""

import numpy as np
import faiss
import joblib
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSIndexService:
    """Advanced FAISS index service with multiple index types and optimization strategies"""
    
    def __init__(self, dimension: int = 384, index_type: str = "auto"):
        """
        Initialize FAISS index service
        
        Args:
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
            index_type: Type of index to use ("auto", "flat", "hnsw", "ivf", "pq")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.is_trained = False
        self.num_vectors = 0
        self.metadata = {}
        
        # Index parameters
        self.index_params = {
            "hnsw_m": 32,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 50,
            "ivf_nlist": None,  # Will be calculated based on data size
            "pq_m": 8,  # Number of sub-quantizers
            "pq_nbits": 8  # Bits per sub-quantizer
        }
        
        logger.info(f"FAISS Index Service initialized with dimension={dimension}, type={index_type}")
    
    def _choose_optimal_index_type(self, num_vectors: int) -> str:
        """Choose optimal index type based on dataset size"""
        if self.index_type != "auto":
            return self.index_type
        
        if num_vectors < 1000:
            return "flat"
        elif num_vectors < 100000:
            return "hnsw"
        elif num_vectors < 1000000:
            return "ivf"
        else:
            return "pq"  # For very large datasets
    
    def _create_flat_index(self) -> faiss.Index:
        """Create a flat (brute force) index for exact search"""
        logger.info("Creating FAISS Flat index for exact search")
        return faiss.IndexFlatIP(self.dimension)
    
    def _create_hnsw_index(self) -> faiss.Index:
        """Create HNSW index for fast approximate search"""
        logger.info(f"Creating FAISS HNSW index (M={self.index_params['hnsw_m']})")
        index = faiss.IndexHNSWFlat(self.dimension, self.index_params["hnsw_m"])
        index.hnsw.efConstruction = self.index_params["hnsw_ef_construction"]
        index.hnsw.efSearch = self.index_params["hnsw_ef_search"]
        return index
    
    def _create_ivf_index(self, num_vectors: int) -> faiss.Index:
        """Create IVF index for large datasets"""
        nlist = self.index_params["ivf_nlist"] or min(int(np.sqrt(num_vectors)), 1000)
        logger.info(f"Creating FAISS IVF index with {nlist} clusters")
        
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        return index
    
    def _create_pq_index(self, num_vectors: int) -> faiss.Index:
        """Create Product Quantization index for very large datasets"""
        nlist = self.index_params["ivf_nlist"] or min(int(np.sqrt(num_vectors)), 1000)
        m = self.index_params["pq_m"]
        nbits = self.index_params["pq_nbits"]
        
        logger.info(f"Creating FAISS PQ index (nlist={nlist}, m={m}, bits={nbits})")
        
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
        return index
    
    def create_index(self, embeddings: np.ndarray, force_rebuild: bool = False) -> bool:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings (shape: [num_vectors, dimension])
            force_rebuild: Whether to force rebuilding if index already exists
            
        Returns:
            bool: Success status
        """
        try:
            if self.index is not None and not force_rebuild:
                logger.info("Index already exists. Use force_rebuild=True to recreate.")
                return True
            
            start_time = time.time()
            
            # Validate embeddings
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
            
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension mismatch. Expected: {self.dimension}, got: {embeddings.shape[1]}")
            
            num_vectors = embeddings.shape[0]
            self.num_vectors = num_vectors
            
            # Ensure float32 format for FAISS
            embeddings_float32 = embeddings.astype(np.float32)
            
            # Choose optimal index type
            chosen_type = self._choose_optimal_index_type(num_vectors)
            logger.info(f"Chosen index type: {chosen_type} for {num_vectors:,} vectors")
            
            # Create index based on type
            if chosen_type == "flat":
                self.index = self._create_flat_index()
            elif chosen_type == "hnsw":
                self.index = self._create_hnsw_index()
            elif chosen_type == "ivf":
                self.index = self._create_ivf_index(num_vectors)
            elif chosen_type == "pq":
                self.index = self._create_pq_index(num_vectors)
            else:
                raise ValueError(f"Unknown index type: {chosen_type}")
            
            # Train index if needed
            if not self.index.is_trained:
                logger.info("Training index...")
                self.index.train(embeddings_float32)
            
            # Add vectors to index
            logger.info("Adding vectors to index...")
            self.index.add(embeddings_float32)
            
            self.is_trained = True
            
            build_time = time.time() - start_time
            logger.info(f"✅ FAISS index created successfully!")
            logger.info(f"   - Index type: {chosen_type}")
            logger.info(f"   - Vectors: {self.index.ntotal:,}")
            logger.info(f"   - Dimension: {self.dimension}")
            logger.info(f"   - Build time: {build_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating FAISS index: {e}")
            return False
    
    def search(self, query_embeddings: np.ndarray, k: int = 10, 
               return_distances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Args:
            query_embeddings: Query embeddings (shape: [num_queries, dimension])
            k: Number of nearest neighbors to return
            return_distances: Whether to return distances/scores
            
        Returns:
            Tuple of (scores, indices) if return_distances=True, else just indices
        """
        if not self.is_trained:
            raise ValueError("Index not trained. Call create_index() first.")
        
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Ensure float32 format
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Perform search
        scores, indices = self.index.search(query_embeddings, k)
        
        if return_distances:
            return scores, indices
        else:
            return indices
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10, 
                    batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform batch search for large number of queries
        
        Args:
            query_embeddings: Query embeddings (shape: [num_queries, dimension])
            k: Number of nearest neighbors to return
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (scores, indices)
        """
        if not self.is_trained:
            raise ValueError("Index not trained. Call create_index() first.")
        
        num_queries = query_embeddings.shape[0]
        all_scores = []
        all_indices = []
        
        logger.info(f"Performing batch search for {num_queries:,} queries")
        
        for i in range(0, num_queries, batch_size):
            batch_queries = query_embeddings[i:i + batch_size]
            scores, indices = self.search(batch_queries, k)
            all_scores.append(scores)
            all_indices.append(indices)
        
        return np.vstack(all_scores), np.vstack(all_indices)
    
    def add_vectors(self, embeddings: np.ndarray) -> bool:
        """
        Add new vectors to existing index
        
        Args:
            embeddings: New embeddings to add
            
        Returns:
            bool: Success status
        """
        try:
            if not self.is_trained:
                raise ValueError("Index not trained. Call create_index() first.")
            
            embeddings_float32 = embeddings.astype(np.float32)
            
            # For some index types, we need to retrain
            if hasattr(self.index, 'ntotal') and isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
                logger.info("Adding vectors to IVF index...")
            
            self.index.add(embeddings_float32)
            self.num_vectors += embeddings.shape[0]
            
            logger.info(f"Added {embeddings.shape[0]:,} vectors. Total: {self.index.ntotal:,}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return False
    
    def remove_vectors(self, ids: List[int]) -> bool:
        """
        Remove vectors by ID (if supported by index type)
        
        Args:
            ids: List of vector IDs to remove
            
        Returns:
            bool: Success status
        """
        try:
            if hasattr(self.index, 'remove_ids'):
                ids_array = np.array(ids, dtype=np.int64)
                removed_count = self.index.remove_ids(ids_array)
                logger.info(f"Removed {removed_count} vectors")
                return True
            else:
                logger.warning("Index type does not support vector removal")
                return False
                
        except Exception as e:
            logger.error(f"Error removing vectors: {e}")
            return False
    
    def save_index(self, filepath: str, save_metadata: bool = True) -> bool:
        """
        Save FAISS index to disk
        
        Args:
            filepath: Path to save index
            save_metadata: Whether to save metadata alongside index
            
        Returns:
            bool: Success status
        """
        try:
            if not self.is_trained:
                raise ValueError("No trained index to save")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, filepath)
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = filepath.replace('.faiss', '_metadata.joblib')
                metadata = {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'num_vectors': self.num_vectors,
                    'index_params': self.index_params,
                    'is_trained': self.is_trained
                }
                joblib.dump(metadata, metadata_path)
                logger.info(f"Metadata saved to: {metadata_path}")
            
            logger.info(f"✅ FAISS index saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, filepath: str, load_metadata: bool = True) -> bool:
        """
        Load FAISS index from disk
        
        Args:
            filepath: Path to load index from
            load_metadata: Whether to load metadata alongside index
            
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Index file not found: {filepath}")
            
            # Load FAISS index
            self.index = faiss.read_index(filepath)
            self.is_trained = True
            self.num_vectors = self.index.ntotal
            
            # Load metadata if available
            if load_metadata:
                metadata_path = filepath.replace('.faiss', '_metadata.joblib')
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    self.dimension = metadata.get('dimension', self.dimension)
                    self.index_type = metadata.get('index_type', self.index_type)
                    self.index_params = metadata.get('index_params', self.index_params)
                    logger.info(f"Metadata loaded from: {metadata_path}")
            
            logger.info(f"✅ FAISS index loaded from: {filepath}")
            logger.info(f"   - Vectors: {self.num_vectors:,}")
            logger.info(f"   - Dimension: {self.dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading index: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_vectors": self.num_vectors,
            "is_trained": self.is_trained,
            "index_params": self.index_params
        }
        
        # Add index-specific information
        if hasattr(self.index, 'ntotal'):
            info["vectors_in_index"] = self.index.ntotal
        
        if isinstance(self.index, faiss.IndexHNSWFlat):
            info["hnsw_info"] = {
                "M": self.index.hnsw.M,
                "efConstruction": self.index.hnsw.efConstruction,
                "efSearch": self.index.hnsw.efSearch
            }
        
        return info
    
    def benchmark_search(self, query_embeddings: np.ndarray, k_values: List[int] = [1, 5, 10, 50]) -> Dict[str, Any]:
        """
        Benchmark search performance
        
        Args:
            query_embeddings: Query embeddings for benchmarking
            k_values: Different k values to test
            
        Returns:
            Dict with benchmark results
        """
        if not self.is_trained:
            raise ValueError("Index not trained")
        
        results = {}
        
        for k in k_values:
            start_time = time.time()
            scores, indices = self.search(query_embeddings, k)
            search_time = time.time() - start_time
            
            results[f"k_{k}"] = {
                "search_time": search_time,
                "queries_per_second": query_embeddings.shape[0] / search_time,
                "results_shape": (scores.shape, indices.shape)
            }
        
        logger.info(f"Benchmark completed for {query_embeddings.shape[0]} queries")
        return results

# Convenience functions for common use cases
def create_faiss_index_from_embeddings(embeddings: np.ndarray, 
                                     dimension: int = None,
                                     index_type: str = "auto") -> FAISSIndexService:
    """
    Create and train FAISS index from embeddings in one step
    
    Args:
        embeddings: Numpy array of embeddings
        dimension: Embedding dimension (inferred if None)
        index_type: Type of index to create
        
    Returns:
        Trained FAISSIndexService instance
    """
    if dimension is None:
        dimension = embeddings.shape[1]
    
    service = FAISSIndexService(dimension=dimension, index_type=index_type)
    service.create_index(embeddings)
    
    return service

def load_faiss_index_from_file(filepath: str) -> FAISSIndexService:
    """
    Load FAISS index from file
    
    Args:
        filepath: Path to FAISS index file
        
    Returns:
        Loaded FAISSIndexService instance
    """
    service = FAISSIndexService()  # Dimension will be loaded from metadata
    service.load_index(filepath)
    
    return service

if __name__ == "__main__":
    # Example usage
    logger.info("FAISS Index Service - Example Usage")
    
    # Create sample embeddings
    dimension = 384
    num_vectors = 1000
    embeddings = np.random.normal(0, 1, (num_vectors, dimension)).astype(np.float32)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create and train index
    service = FAISSIndexService(dimension=dimension, index_type="auto")
    service.create_index(embeddings)
    
    # Test search
    query = embeddings[:5]  # Use first 5 vectors as queries
    scores, indices = service.search(query, k=10)
    
    logger.info(f"Search results shape: scores={scores.shape}, indices={indices.shape}")
    logger.info(f"Sample scores: {scores[0][:5]}")
    logger.info(f"Sample indices: {indices[0][:5]}")
    
    # Save index
    service.save_index("test_index.faiss")
    
    # Benchmark
    benchmark_results = service.benchmark_search(query)
    logger.info(f"Benchmark results: {benchmark_results}")
