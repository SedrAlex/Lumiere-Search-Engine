#!/usr/bin/env python3
"""
Quora Query Processing Service
Online service that processes user queries and performs similarity search using 
SQLite database and cosine similarity. Based on ANTIQUE service.
"""

import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
import time
import joblib
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

class QuoraQueryProcessor:
    """
    Query processing service that uses SQLite database and cosine similarity
    to perform similarity search with top-10 ranked results.
    """

    def __init__(self, text_processing_service_url="http://localhost:5003", models_dir="../models", use_faiss=False):
        """
        Initialize the query processor.
        
        Args:
            text_processing_service_url (str): URL of the text processing service
            models_dir (str): Directory containing the models (not used anymore)
            use_faiss (bool): Whether to use FAISS for similarity search
        """
        self.text_processing_url = text_processing_service_url
        self.sqlite_db_path = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_documents.db'
        self.tsv_file_path = '/Users/raafatmhanna/Downloads/quora/docs.tsv'
        
        # Initialize components
        self.model = None
        
        # FAISS configuration
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_indices_dir = '/Users/raafatmhanna/Downloads/faiss_indices'
        self.faiss_indices = {}
        self.current_faiss_index = None
        
        if self.use_faiss and not FAISS_AVAILABLE:
            logger.warning("FAISS requested but not available. Falling back to cosine similarity.")
            self.use_faiss = False
        
        # Caching and performance improvements
        self.embeddings_cache_path = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_embeddings_cache.pkl'
        self.doc_embeddings = None
        self.doc_ids_cache = None
        self.doc_texts_cache = None
        self.cache_timestamp = None
        
        # Advanced caching for ultra-fast search
        self.query_embedding_cache = {}
        self.similarity_cache = {}
        self.max_cache_size = 10000
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Load the model and setup documents
        self.load_embeddings_from_joblib()
        
        # Load FAISS indices if enabled
        if self.use_faiss:
            self.load_faiss_indices()
        
    def test_text_processing_service(self):
        """Test if the text processing service is available."""
        try:
            response = requests.get(f"{self.text_processing_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Text processing service not available: {e}")
            return False
            
    def save_embeddings_cache(self, doc_ids, doc_texts, doc_embeddings):
        """Save embeddings to cache file."""
        try:
            cache_data = {
                'doc_ids': doc_ids,
                'doc_texts': doc_texts,
                'doc_embeddings': doc_embeddings,
                'timestamp': time.time()
            }
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Embeddings cache saved to {self.embeddings_cache_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {e}")
            
    def load_embeddings_cache(self):
        """Load embeddings from cache file."""
        try:
            if os.path.exists(self.embeddings_cache_path):
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.doc_ids_cache = cache_data['doc_ids']
                self.doc_texts_cache = cache_data['doc_texts']
                self.doc_embeddings = cache_data['doc_embeddings']
                self.cache_timestamp = cache_data['timestamp']
                
                logger.info(f"Loaded embeddings cache with {len(self.doc_ids_cache)} documents")
                return True
            else:
                logger.info("No embeddings cache found")
                return False
        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            return False
            
    def is_cache_valid(self):
        """Check if the cache is still valid by comparing with database."""
        try:
            if not all([self.doc_ids_cache, self.doc_texts_cache, self.doc_embeddings is not None]):
                return False
                
            # Check if SQLite database has been modified
            if os.path.exists(self.sqlite_db_path):
                db_mtime = os.path.getmtime(self.sqlite_db_path)
                if self.cache_timestamp is None or db_mtime > self.cache_timestamp:
                    logger.info("Database has been modified, cache is invalid")
                    return False
                    
            # Check if document count matches
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents WHERE processed_text IS NOT NULL')
            db_count = cursor.fetchone()[0]
            conn.close()
            
            if db_count != len(self.doc_ids_cache):
                logger.info(f"Document count mismatch: cache={len(self.doc_ids_cache)}, db={db_count}")
                return False
                
            logger.info("Cache is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
            
    def precompute_embeddings(self):
        """Precompute and cache all document embeddings."""
        try:
            logger.info("Precomputing document embeddings...")
            # Load documents from SQLite
            doc_ids, doc_texts = self.load_documents_from_sqlite()
            
            if not doc_ids:
                raise ValueError("No documents found in database.")
            
            # Encode all documents in batches for better performance
            batch_size = 100
            all_embeddings = []
            
            logger.info(f"Processing {len(doc_texts)} documents in batches of {batch_size}")
            
            for i in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=True)
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(doc_texts) + batch_size - 1)//batch_size}")
            
            # Convert to numpy array
            doc_embeddings = np.array(all_embeddings)
            
            # Cache the results
            self.doc_ids_cache = doc_ids
            self.doc_texts_cache = doc_texts
            self.doc_embeddings = doc_embeddings
            self.cache_timestamp = time.time()
            
            # Save to file
            self.save_embeddings_cache(doc_ids, doc_texts, doc_embeddings)
            
            logger.info(f"Successfully precomputed embeddings for {len(doc_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error precomputing embeddings: {e}")
            raise
            
    def load_and_store_documents_to_sqlite(self):
        """
        Load documents from TSV, clean them using the text service, and store in SQLite.
        """
        logger.info("Loading documents from TSV and storing in SQLite...")
        
        try:
            # Read the TSV file
            logger.info(f"Reading documents from: {self.tsv_file_path}")
            docs_df = pd.read_csv(self.tsv_file_path, sep='\t')
            
            # Check if required columns exist
            if 'doc_id' not in docs_df.columns or 'text' not in docs_df.columns:
                raise ValueError("TSV file must contain 'doc_id' and 'text' columns")
            
            logger.info(f"Found {len(docs_df)} documents in TSV file")
            
            # Connect to SQLite database
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            
            # Create table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    original_text TEXT,
                    processed_text TEXT
                )
            ''')
            
            # Process and insert documents in batches
            batch_size = 100
            total_processed = 0
            
            for i in range(0, len(docs_df), batch_size):
                batch = docs_df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    doc_id = str(row['doc_id'])
                    original_text = str(row['text']) if pd.notna(row['text']) else ""
                    
                    # Process text using the text processing service
                    processed_text = self.call_text_processing_service(original_text)
                    
                    # Insert into database
                    cursor.execute('''
                        INSERT OR REPLACE INTO documents (doc_id, original_text, processed_text) 
                        VALUES (?, ?, ?)
                    ''', (doc_id, original_text, processed_text))
                    
                    total_processed += 1
                
                # Commit batch
                conn.commit()
                logger.info(f"Processed {total_processed}/{len(docs_df)} documents...")
            
            # Final commit and close
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully stored {total_processed} documents in SQLite database")
            
        except Exception as e:
            logger.error(f"Error loading and storing documents: {e}")
            raise

    def load_embeddings_from_joblib(self):
        """Load precomputed embeddings from joblib files."""
        try:
            logger.info("Loading embeddings from joblib files...")

            # Load doc embeddings
            embeddings_path = '/Users/raafatmhanna/Downloads/quora_Embeddings/doc_embeddings.joblib'
            self.doc_embeddings = joblib.load(embeddings_path)
            logger.info(f"Loaded doc_embeddings successfully. Shape: {self.doc_embeddings.shape}")

            # Load doc IDs
            documents_path = '/Users/raafatmhanna/Downloads/quora_Embeddings/documents_final.joblib'
            docs_data = joblib.load(documents_path)
            self.doc_ids_cache = docs_data['doc_ids']
            self.doc_texts_cache = docs_data['texts']
            logger.info(f"Loaded document IDs and texts successfully. Count: {len(self.doc_ids_cache)}")
            
            # Load the SentenceTransformer model for encoding queries
            logger.info("Loading SentenceTransformer model for query encoding...")
            model_path = '/Users/raafatmhanna/Downloads/quora_Embeddings/sentence-transformers_all-MiniLM-L6-v2'
            self.model = SentenceTransformer(model_path)
            logger.info("SentenceTransformer model loaded successfully from local path.")

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
            
    def load_documents_from_sqlite(self):
        """Load documents from SQLite database."""
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT doc_id, processed_text FROM documents WHERE processed_text IS NOT NULL')
            rows = cursor.fetchall()
            
            conn.close()
            
            doc_ids = [row[0] for row in rows]
            doc_texts = [row[1] for row in rows]
            
            return doc_ids, doc_texts
            
        except Exception as e:
            logger.error(f"Error loading documents from SQLite: {e}")
            raise
        
    @lru_cache(maxsize=1000)
    def call_text_processing_service(self, text: str, endpoint: str = "process") -> str:
        """
        Call the text processing service to clean text with LRU caching.
        
        Args:
            text (str): Text to process
            endpoint (str): Service endpoint to call
            
        Returns:
            str: Processed text
        """
        try:
            if endpoint == "query":
                url = f"{self.text_processing_url}/process/query"
                payload = {"query": text}
                response_key = "processed_query"
            else:
                url = f"{self.text_processing_url}/process"
                payload = {"text": text}
                response_key = "processed_text"
                
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get(response_key, text)
            else:
                logger.warning(f"Text processing service error: {response.status_code}")
                return text  # Return original text as fallback
                
        except Exception as e:
            logger.warning(f"Error calling text processing service: {e}")
            return text  # Return original text as fallback
            
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector with caching.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding
        """
        # Generate cache key
        cache_key = self._get_cache_key(query)
        
        # Check if embedding is cached
        if cache_key in self.query_embedding_cache:
            self.cache_hit_count += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.query_embedding_cache[cache_key]
        
        # Cache miss - compute embedding
        self.cache_miss_count += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        
        # Process the query using the text processing service
        processed_query = self.call_text_processing_service(query, "query")
        
        # Generate embedding
        embedding = self.model.encode([processed_query], normalize_embeddings=True)[0]
        
        # Cache the result
        self.query_embedding_cache[cache_key] = embedding
        
        # Manage cache size
        self._manage_cache_size()
        
        return embedding
        
    def search_similar_documents(self, query: str, top_k: int = 10, use_faiss: bool = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS (if enabled) or cosine similarity with advanced caching.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            use_faiss (bool): Whether to use FAISS index. If None, uses default setting
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with metadata
        """
        try:
            # Determine which search method to use
            if use_faiss is None:
                use_faiss = self.use_faiss
            
            if use_faiss and self.faiss_indices:
                return self.search_with_faiss(query, top_k)
            else:
                return self._search_with_cosine_similarity(query, top_k)
        
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def _search_with_cosine_similarity(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using traditional cosine similarity.
        """
        # Generate cache key for this query and top_k combination
        cache_key = self._get_cache_key(query) + f"_k{top_k}_cosine"
        
        # Check if results are cached
        if cache_key in self.similarity_cache:
            logger.info(f"Cache hit for query: {query[:50]}... (top_k={top_k})")
            return self.similarity_cache[cache_key]
        
        # Cache miss - compute similarity
        logger.info(f"Cache miss for query: {query[:50]}... (top_k={top_k})")
        logger.info(f"Using precomputed embeddings for {len(self.doc_ids_cache)} documents")
        
        doc_ids = self.doc_ids_cache
        doc_texts = self.doc_texts_cache
        doc_embeddings = self.doc_embeddings
        
        # Encode the query (this will use query embedding cache)
        start_time = time.time()
        query_embedding = self.encode_query(query)
        query_time = time.time() - start_time
        
        # Calculate cosine similarity
        start_time = time.time()
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        similarity_time = time.time() - start_time
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for i, doc_idx in enumerate(top_indices):
            doc_id = str(doc_ids[doc_idx])  # Ensure doc_id is string
            doc_text = doc_texts[doc_idx]
            similarity_score = similarities[doc_idx]
            
            results.append({
                'rank': i + 1,
                'doc_id': doc_id,
                'document': doc_text,
                'similarity_score': float(similarity_score),
                'doc_index': int(doc_idx),
                'search_method': 'cosine_similarity'
            })
        
        # Cache the results
        self.similarity_cache[cache_key] = results
        
        # Manage cache size
        self._manage_cache_size()
        
        logger.info(f"Cosine search completed in {query_time + similarity_time:.3f}s (query: {query_time:.3f}s, similarity: {similarity_time:.3f}s)")
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by its ID.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Optional[Dict[str, Any]]: Document data or None if not found
        """
        try:
            # Find document in cache
            if self.doc_ids_cache:
                for i, cached_id in enumerate(self.doc_ids_cache):
                    if str(cached_id) == str(doc_id):
                        return {
                            'doc_id': doc_id,
                            'document': self.doc_texts_cache[i],
                            'doc_index': i
                        }
            
            # If not in cache, try SQLite database
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT doc_id, original_text, processed_text FROM documents WHERE doc_id = ?', (doc_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'doc_id': row[0],
                    'original_document': row[1],
                    'processed_document': row[2],
                    'doc_index': 0  # Not really meaningful in SQLite context
                }
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
            
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        # Determine current search method
        if self.use_faiss and self.faiss_indices:
            search_method = f"faiss_{self.current_faiss_index}" if self.current_faiss_index else "faiss"
        else:
            search_method = "cosine_similarity"
        
        stats = {
            'service': 'Quora Query Processing',
            'search_method': search_method,
            'cache_stats': self.get_cache_stats(),
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'data_source': self.tsv_file_path,
            'faiss_available': FAISS_AVAILABLE,
            'use_faiss': self.use_faiss
        }
        
        # Get document count from SQLite
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            doc_count = cursor.fetchone()[0]
            conn.close()
            
            stats['num_documents'] = doc_count
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            stats['num_documents'] = 0
            
        return stats
        
    def clear_query_cache(self):
        """Clear all caches for maximum performance reset."""
        self.call_text_processing_service.cache_clear()
        self.query_embedding_cache.clear()
        self.similarity_cache.clear()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        logger.info("All caches cleared: LRU cache, query embeddings, and similarity results")
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory bloat."""
        if len(self.query_embedding_cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self.query_embedding_cache) // 5
            keys_to_remove = list(self.query_embedding_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.query_embedding_cache[key]
            logger.info(f"Removed {items_to_remove} entries from query embedding cache")
            
        if len(self.similarity_cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self.similarity_cache) // 5
            keys_to_remove = list(self.similarity_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.similarity_cache[key]
            logger.info(f"Removed {items_to_remove} entries from similarity cache")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        # Use hash of processed query for consistent caching
        processed_query = self.call_text_processing_service(query, "query")
        return str(hash(processed_query))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = (self.cache_hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "hit_rate_percentage": round(hit_rate, 2),
            "query_embedding_cache_size": len(self.query_embedding_cache),
            "similarity_cache_size": len(self.similarity_cache),
            "max_cache_size": self.max_cache_size
        }
    
    def load_faiss_indices(self):
        """Load FAISS indices from the faiss_indices directory."""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available, cannot load indices")
            return
            
        try:
            logger.info(f"Loading FAISS indices from {self.faiss_indices_dir}")
            
            if not os.path.exists(self.faiss_indices_dir):
                logger.warning(f"FAISS indices directory not found: {self.faiss_indices_dir}")
                return
            
            # Load available indices
            index_files = {
                'flat': 'flat_index.faiss',
                'ivf': 'ivf_index.faiss',
                'hnsw': 'hnsw_index.faiss'
            }
            
            loaded_indices = 0
            for index_name, filename in index_files.items():
                index_path = os.path.join(self.faiss_indices_dir, filename)
                if os.path.exists(index_path):
                    try:
                        index = faiss.read_index(index_path)
                        self.faiss_indices[index_name] = index
                        logger.info(f"Loaded {index_name} index with {index.ntotal:,} vectors from {filename}")
                        loaded_indices += 1
                        
                        # Set the first loaded index as current
                        if self.current_faiss_index is None:
                            self.current_faiss_index = index_name
                            
                    except Exception as e:
                        logger.error(f"Error loading {index_name} index from {filename}: {e}")
                else:
                    logger.debug(f"Index file not found: {filename}")
            
            if loaded_indices > 0:
                logger.info(f"Successfully loaded {loaded_indices} FAISS indices")
                logger.info(f"Current active index: {self.current_faiss_index}")
            else:
                logger.warning("No FAISS indices found, falling back to cosine similarity")
                self.use_faiss = False
                
        except Exception as e:
            logger.error(f"Error loading FAISS indices: {e}")
            self.use_faiss = False
    
    def search_with_faiss(self, query: str, top_k: int = 10, index_type: str = None) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        if not self.use_faiss or not self.faiss_indices:
            raise ValueError("FAISS not available or no indices loaded")
        
        # Select index
        if index_type and index_type in self.faiss_indices:
            index = self.faiss_indices[index_type]
            active_index = index_type
        elif self.current_faiss_index and self.current_faiss_index in self.faiss_indices:
            index = self.faiss_indices[self.current_faiss_index]
            active_index = self.current_faiss_index
        else:
            # Use first available index
            active_index = list(self.faiss_indices.keys())[0]
            index = self.faiss_indices[active_index]
        
        try:
            # Encode query
            start_time = time.time()
            query_embedding = self.encode_query(query)
            query_time = time.time() - start_time
            
            # Perform FAISS search
            start_time = time.time()
            query_embedding_f32 = query_embedding.astype(np.float32).reshape(1, -1)
            scores, indices = index.search(query_embedding_f32, top_k)
            search_time = time.time() - start_time
            
            # Prepare results
            results = []
            for i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
                if doc_idx >= 0:  # Valid index
                    doc_id = str(self.doc_ids_cache[doc_idx])
                    doc_text = self.doc_texts_cache[doc_idx]
                    
                    results.append({
                        'rank': i + 1,
                        'doc_id': doc_id,
                        'document': doc_text,
                        'similarity_score': float(score),
                        'doc_index': int(doc_idx),
                        'search_method': f'faiss_{active_index}'
                    })
            
            logger.info(f"FAISS search completed in {query_time + search_time:.3f}s (query: {query_time:.3f}s, search: {search_time:.3f}s) using {active_index} index")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            raise
    
    def set_faiss_index(self, index_type: str) -> bool:
        """Set the active FAISS index type."""
        if not self.use_faiss:
            return False
            
        if index_type in self.faiss_indices:
            self.current_faiss_index = index_type
            logger.info(f"Set active FAISS index to: {index_type}")
            return True
        else:
            logger.warning(f"FAISS index type '{index_type}' not available. Available: {list(self.faiss_indices.keys())}")
            return False
    
    def get_faiss_info(self) -> Dict[str, Any]:
        """Get information about loaded FAISS indices."""
        info = {
            'faiss_available': FAISS_AVAILABLE,
            'use_faiss': self.use_faiss,
            'current_index': self.current_faiss_index,
            'available_indices': {},
            'indices_directory': self.faiss_indices_dir
        }
        
        for name, index in self.faiss_indices.items():
            info['available_indices'][name] = {
                'total_vectors': index.ntotal,
                'dimension': index.d,
                'is_trained': index.is_trained
            }
        
        return info

# Initialize FastAPI application
app = FastAPI(
    title="Quora Query Processing Service",
    description="Online query processing with cosine similarity search using SQLite database",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for 404 errors
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with helpful information."""
    path = request.url.path
    method = request.method
    
    # Check if it's a malformed search URL
    if 'search' in path.lower():
        return JSONResponse(
            status_code=404,
            content={
                "error": "Endpoint not found",
                "message": f"The endpoint '{path}' was not found.",
                "hint": "Did you mean to use POST /search? Check for trailing spaces or slashes.",
                "correct_endpoint": "POST /search",
                "available_endpoints": [
                    "GET /",
                    "GET /health",
                    "POST /search",
                    "GET /document/{doc_id}",
                    "GET /stats",
                    "GET /info",
                    "POST /cache/refresh",
                    "GET /cache/status",
                    "GET /cache/stats",
                    "POST /cache/clear"
                ]
            }
        )
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The {method} endpoint '{path}' was not found.",
            "available_endpoints": [
                "GET /",
                "GET /health",
                "POST /search",
                "GET /document/{doc_id}",
                "GET /stats",
                "GET /info",
                "POST /cache/refresh",
                "GET /cache/status",
                "POST /cache/clear"
            ],
            "docs": "http://localhost:5004/docs"
        }
    )

# Initialize the query processor
query_processor = QuoraQueryProcessor(use_faiss=True)  # Enable FAISS by default if available

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_faiss: Optional[bool] = True  # Default to True if FAISS is available

class SearchResponse(BaseModel):
    query: str
    processed_query: str
    results: List[Dict[str, Any]]
    total_results: int

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "quora-query-processing", "version": "1.0.0"}

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for similar documents using FAISS index or cosine similarity.
    
    Returns top-k most similar documents based on the query.
    The user can choose between FAISS index (if available) or cosine similarity via the use_faiss parameter.
    """
    try:
        # Limit top_k to prevent abuse
        top_k = min(request.top_k, 50)
        
        # Process query and get results with user's choice of search method
        processed_query = query_processor.call_text_processing_service(request.query, "query")
        results = query_processor.search_similar_documents(request.query, top_k, use_faiss=request.use_faiss)
        
        return {
            "query": request.query,
            "processed_query": processed_query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Handle malformed search URLs (common issue with trailing spaces)
@app.post("/search/")
@app.post("/search ")
async def search_documents_with_trailing_slash_or_space(request: SearchRequest):
    """Handle malformed search URLs with trailing slash or space."""
    return await search_documents(request)

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    try:
        document = query_processor.get_document_by_id(doc_id)
        if document:
            return document
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    try:
        stats = query_processor.get_service_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/info")
async def get_info():
    """Get service information including FAISS indices."""
    try:
        info = {
            "service": "Quora Query Processing",
            "version": "1.0.0",
            "faiss_info": query_processor.get_faiss_info() if hasattr(query_processor, 'get_faiss_info') else None
        }
        return info
    except Exception as e:
        logger.error(f"Error getting info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get info: {str(e)}")

@app.post("/cache/refresh")
async def refresh_cache():
    """Refresh the embeddings cache."""
    try:
        if hasattr(query_processor, 'precompute_embeddings'):
            query_processor.precompute_embeddings()
            return {"message": "Cache refreshed successfully"}
        else:
            return {"message": "Cache refresh not available for this configuration"}
    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

@app.get("/cache/status")
async def get_cache_status():
    """Get cache status."""
    try:
        status = {
            "cache_available": hasattr(query_processor, 'doc_embeddings') and query_processor.doc_embeddings is not None,
            "document_count": len(query_processor.doc_ids_cache) if hasattr(query_processor, 'doc_ids_cache') and query_processor.doc_ids_cache else 0,
            "embedding_shape": query_processor.doc_embeddings.shape if hasattr(query_processor, 'doc_embeddings') and query_processor.doc_embeddings is not None else None
        }
        return status
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        stats = {
            "cache_hit_count": getattr(query_processor, 'cache_hit_count', 0),
            "cache_miss_count": getattr(query_processor, 'cache_miss_count', 0),
            "query_cache_size": len(getattr(query_processor, 'query_embedding_cache', {})),
            "similarity_cache_size": len(getattr(query_processor, 'similarity_cache', {}))
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    try:
        if hasattr(query_processor, 'query_embedding_cache'):
            query_processor.query_embedding_cache.clear()
        if hasattr(query_processor, 'similarity_cache'):
            query_processor.similarity_cache.clear()
        return {"message": "Caches cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Quora Query Processing Service",
        "version": "1.0.0",
        "description": "Online query processing with embedding-based similarity search",
        "endpoints": {
            "health": "GET /health",
            "search": "POST /search",
            "document": "GET /document/{doc_id}",
            "stats": "GET /stats",
            "info": "GET /info",
            "cache_refresh": "POST /cache/refresh",
            "cache_status": "GET /cache/status",
            "cache_stats": "GET /cache/stats",
            "cache_clear": "POST /cache/clear"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    # Run the service on port 5004
    uvicorn.run(app, host="0.0.0.0", port=5004)
