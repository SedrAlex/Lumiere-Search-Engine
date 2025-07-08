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
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraQueryProcessor:
    """
    Query processing service that uses SQLite database and cosine similarity
    to perform similarity search with top-10 ranked results.
    """
    
    def __init__(self, text_processing_service_url="http://localhost:5003", models_dir="../models"):
        """
        Initialize the query processor.
        
        Args:
            text_processing_service_url (str): URL of the text processing service
            models_dir (str): Directory containing the models (not used anymore)
        """
        self.text_processing_url = text_processing_service_url
        self.sqlite_db_path = 'quora_documents.db'
        self.tsv_file_path = '/Users/raafatmhanna/Downloads/quora/docs.tsv'
        
        # Initialize components
        self.model = None
        
        # Caching and performance improvements
        self.embeddings_cache_path = 'quora_embeddings_cache.pkl'
        self.doc_embeddings = None
        self.doc_ids_cache = None
        self.doc_texts_cache = None
        self.cache_timestamp = None
        
        # Load the model and setup documents
        self.load_model_and_setup_documents()
        
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

    def load_model_and_setup_documents(self):
        """Load the SentenceTransformer model and setup documents in SQLite."""
        try:
            logger.info("Loading Quora model and setting up documents...")
            
            # Check if text processing service is available
            if not self.test_text_processing_service():
                logger.warning("Text processing service not available. Some features may not work.")
            
            # Load the SentenceTransformer model
            logger.info("Loading model from HuggingFace: sentence-transformers/all-MiniLM-L6-v2")
            
            try:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("Model loaded successfully")
            except Exception as model_error:
                logger.error(f"Failed to load model: {model_error}")
                raise
            
            # Check if SQLite database exists, if not create it
            if not os.path.exists(self.sqlite_db_path):
                logger.info("SQLite database not found. Creating new database with processed documents...")
                self.load_and_store_documents_to_sqlite()
            else:
                logger.info("SQLite database found. Using existing processed documents.")
            
            # Load embeddings cache
            logger.info("Checking for embeddings cache...")
            if self.load_embeddings_cache() and self.is_cache_valid():
                logger.info("Using valid embeddings cache")
            else:
                logger.info("Cache invalid or missing. Precomputing embeddings...")
                self.precompute_embeddings()
                
        except Exception as e:
            logger.error(f"Error loading model and setting up documents: {e}")
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
        Encode a query into an embedding vector.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding
        """
        # Process the query using the text processing service
        processed_query = self.call_text_processing_service(query)
        
        # Generate embedding
        embedding = self.model.encode([processed_query], normalize_embeddings=True)
        return embedding[0]
        
    def search_similar_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity with cached embeddings.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of similar documents with scores
        """
        try:
            # Check if we have cached embeddings
            if self.doc_embeddings is None or self.doc_ids_cache is None:
                logger.warning("No cached embeddings found. Falling back to real-time encoding...")
                # Fallback to loading documents from SQLite
                doc_ids, doc_texts = self.load_documents_from_sqlite()
                
                if not doc_ids:
                    raise ValueError("No documents found in database.")
                
                # Encode all documents (this will be slow)
                logger.info(f"Encoding {len(doc_texts)} documents for similarity search...")
                doc_embeddings = self.model.encode(doc_texts, normalize_embeddings=True)
            else:
                # Use cached embeddings for ultra-fast search
                logger.info(f"Using cached embeddings for {len(self.doc_ids_cache)} documents")
                doc_ids = self.doc_ids_cache
                doc_texts = self.doc_texts_cache
                doc_embeddings = self.doc_embeddings
            
            # Encode the query
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
                doc_id = doc_ids[doc_idx]
                doc_text = doc_texts[doc_idx]
                similarity_score = similarities[doc_idx]
                
                results.append({
                    'rank': i + 1,
                    'doc_id': doc_id,
                    'document': doc_text,
                    'similarity_score': float(similarity_score),
                    'doc_index': int(doc_idx)
                })
            
            logger.info(f"Search completed in {query_time + similarity_time:.3f}s (query: {query_time:.3f}s, similarity: {similarity_time:.3f}s)")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
            
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Get a specific document by its ID.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Dict: Document data
        """
        try:
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
        stats = {
            'model_loaded': self.model is not None,
            'database_loaded': os.path.exists(self.sqlite_db_path),
            'text_processing_service_available': self.test_text_processing_service(),
            'search_method': 'cosine_similarity',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'data_source': self.tsv_file_path
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
        """Clear the LRU cache for text processing."""
        self.call_text_processing_service.cache_clear()
        logger.info("Query processing cache cleared")

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

# Initialize the query processor
query_processor = QuoraQueryProcessor()

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class SearchResult(BaseModel):
    rank: int
    doc_id: str
    document: str
    similarity_score: float
    doc_index: int

class SearchResponse(BaseModel):
    query: str
    processed_query: str
    results: List[SearchResult]
    total_results: int

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "quora-query-processing", "version": "1.0.0"}

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for similar documents using cosine similarity.
    
    Returns top-k most similar documents based on the query.
    """
    try:
        # Limit top_k to prevent abuse
        top_k = min(request.top_k, 50)
        
        # Process query and get results
        processed_query = query_processor.call_text_processing_service(request.query)
        results = query_processor.search_similar_documents(request.query, top_k)
        
        return {
            "query": request.query,
            "processed_query": processed_query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    try:
        document = query_processor.get_document_by_id(doc_id)
        
        if document:
            return document
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
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

if __name__ == '__main__':
    print("🚀 Starting Quora Query Processing Service v1.0 with FastAPI...")
    print("🔍 This service performs similarity search using SQLite + cosine similarity")
    print("📄 Data source: /Users/raafatmhanna/Downloads/quora/docs.tsv")
    print("🔗 Service will be available at: http://localhost:5004")
    print(f"📡 Text processing service: {query_processor.text_processing_url}")
    print("📖 API docs available at: http://localhost:5004/docs")
    
    # Run the service on port 5004
    uvicorn.run(app, host="0.0.0.0", port=5004)
