#!/usr/bin/env python3
"""
ANTIQUE Query Processing Service
Online service that processes user queries and performs similarity search using 
the same model and embeddings from the ANTIQUE notebook.
Built with FastAPI for better performance and automatic API documentation.
"""

import os
import sys
import logging
import requests
import numpy as np
import joblib
import json
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiqueQueryProcessor:
    """
    Query processing service that uses the ANTIQUE embeddings and model
    to perform similarity search with top-10 ranked results.
    """
    
    def __init__(self, text_processing_service_url="http://localhost:5001", models_dir="../models"):
        """
        Initialize the query processor.
        
        Args:
            text_processing_service_url (str): URL of the text processing service
            models_dir (str): Directory containing the models and embeddings
        """
        self.text_processing_url = text_processing_service_url
        self.models_dir = os.path.abspath(models_dir)
        
        # Initialize components
        self.model = None
        self.doc_embeddings = None
        self.query_embeddings = None
        self.doc_data = None
        self.query_data = None
        self.metadata = None
        self.faiss_index = None
        
        # Load the model and embeddings
        self.load_model_and_embeddings()
        
    def test_text_processing_service(self):
        """Test if the text processing service is available."""
        try:
            response = requests.get(f"{self.text_processing_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Text processing service not available: {e}")
            return False
            
    def load_model_and_embeddings(self):
        """Load the SentenceTransformer model and pre-computed embeddings."""
        try:
            logger.info("Loading ANTIQUE model and embeddings...")
            
            # Check if text processing service is available
            if not self.test_text_processing_service():
                logger.warning("Text processing service not available. Some features may not work.")
            
            # Expected file paths (look for files in models directory or common locations)
            possible_paths = [
                self.models_dir,
                os.path.join(self.models_dir, "antique"),
                os.path.join(self.models_dir, "embeddings"),
                os.path.join(os.path.dirname(self.models_dir), "temp"),
                os.path.join(os.path.dirname(self.models_dir), "data", "embeddings")
            ]
            
            # Look for the embeddings files
            embedding_files = [
                "doc_embeddings.joblib",
                "query_embeddings.joblib", 
                "embedding_metadata.joblib",
                "documents_final.joblib",
                "queries_final.joblib"
            ]
            
            model_paths = [
                "sentence-transformers_all-MiniLM-L6-v2",
                "all-MiniLM-L6-v2"
            ]
            
            # Find the correct paths
            embeddings_path = None
            model_path = None
            
            for base_path in possible_paths:
                if os.path.exists(base_path):
                    # Check for embeddings
                    if all(os.path.exists(os.path.join(base_path, f)) for f in embedding_files):
                        embeddings_path = base_path
                        logger.info(f"Found embeddings at: {embeddings_path}")
                        break
                        
                    # Check for model
                    for model_name in model_paths:
                        model_candidate = os.path.join(base_path, model_name)
                        if os.path.exists(model_candidate):
                            model_path = model_candidate
                            logger.info(f"Found model at: {model_path}")
            
            # Load the SentenceTransformer model
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from: {model_path}")
                self.model = SentenceTransformer(model_path)
            else:
                logger.info("Loading model from HuggingFace: sentence-transformers/all-MiniLM-L6-v2")
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load embeddings if available
            if embeddings_path:
                self.load_embeddings_from_path(embeddings_path)
            else:
                logger.warning("Pre-computed embeddings not found. Will compute embeddings on-the-fly.")
                self.setup_fallback_mode()
                
        except Exception as e:
            logger.error(f"Error loading model and embeddings: {e}")
            self.setup_fallback_mode()
            
    def load_embeddings_from_path(self, embeddings_path):
        """Load embeddings from the specified path."""
        try:
            # Load document embeddings
            doc_emb_path = os.path.join(embeddings_path, "doc_embeddings.joblib")
            self.doc_embeddings = joblib.load(doc_emb_path)
            logger.info(f"Loaded document embeddings: {self.doc_embeddings.shape}")
            
            # Load query embeddings
            query_emb_path = os.path.join(embeddings_path, "query_embeddings.joblib")
            self.query_embeddings = joblib.load(query_emb_path)
            logger.info(f"Loaded query embeddings: {self.query_embeddings.shape}")
            
            # Load metadata
            metadata_path = os.path.join(embeddings_path, "embedding_metadata.joblib")
            self.metadata = joblib.load(metadata_path)
            logger.info(f"Loaded metadata: {self.metadata['num_docs']} docs, {self.metadata['num_queries']} queries")
            
            # Load document data
            doc_data_path = os.path.join(embeddings_path, "documents_final.joblib")
            self.doc_data = joblib.load(doc_data_path)
            logger.info(f"Loaded document data: {len(self.doc_data['doc_ids'])} documents")
            
            # Load query data
            query_data_path = os.path.join(embeddings_path, "queries_final.joblib")
            self.query_data = joblib.load(query_data_path)
            logger.info(f"Loaded query data: {len(self.query_data['query_ids'])} queries")
            
            # Create FAISS index for fast similarity search
            self.setup_faiss_index()
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
            
    def setup_faiss_index(self):
        """Setup FAISS index for fast similarity search."""
        try:
            # Create FAISS index
            dimension = self.doc_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            
            # Add embeddings to index
            self.faiss_index.add(self.doc_embeddings.astype(np.float32))
            
            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error setting up FAISS index: {e}")
            raise
            
    def setup_fallback_mode(self):
        """Setup fallback mode without pre-computed embeddings."""
        logger.info("Setting up fallback mode...")
        self.doc_embeddings = None
        self.query_embeddings = None
        self.metadata = None
        self.doc_data = None
        self.query_data = None
        self.faiss_index = None
        
    def call_text_processing_service(self, text: str, endpoint: str = "process") -> str:
        """
        Call the text processing service to clean text.
        
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
        Encode a query into an embedding vector.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding
        """
        # Process the query using the text processing service
        processed_query = self.call_text_processing_service(query, "query")
        
        # Generate embedding
        embedding = self.model.encode([processed_query], normalize_embeddings=True)
        return embedding[0]
        
    def search_similar_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of similar documents with scores
        """
        if not self.faiss_index or not self.doc_data:
            raise ValueError("Embeddings not loaded. Cannot perform search.")
            
        try:
            # Encode the query
            query_embedding = self.encode_query(query)
            
            # Search using FAISS
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                top_k
            )
            
            # Prepare results
            results = []
            for i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
                if doc_idx < len(self.doc_data['doc_ids']):
                    doc_id = self.doc_data['doc_ids'][doc_idx]
                    doc_text = self.doc_data['texts'][doc_idx]
                    
                    results.append({
                        'rank': i + 1,
                        'doc_id': doc_id,
                        'document': doc_text,
                        'similarity_score': float(score),
                        'doc_index': int(doc_idx)
                    })
                    
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
        if not self.doc_data:
            raise ValueError("Document data not loaded.")
            
        try:
            # Find document index
            doc_index = self.doc_data['doc_ids'].index(doc_id)
            doc_text = self.doc_data['texts'][doc_index]
            
            return {
                'doc_id': doc_id,
                'document': doc_text,
                'doc_index': doc_index
            }
            
        except ValueError:
            return None
            
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            'model_loaded': self.model is not None,
            'embeddings_loaded': self.doc_embeddings is not None,
            'text_processing_service_available': self.test_text_processing_service()
        }
        
        if self.metadata:
            stats.update({
                'num_documents': self.metadata.get('num_docs', 0),
                'num_queries': self.metadata.get('num_queries', 0),
                'embedding_dimension': self.metadata.get('embedding_dim', 0),
                'model_name': self.metadata.get('model_name', 'unknown')
            })
            
        if self.faiss_index:
            stats['faiss_index_size'] = self.faiss_index.ntotal
            
        return stats

# Initialize FastAPI application
app = FastAPI(
    title="ANTIQUE Query Processing Service",
    description="Online query processing with cosine similarity search using ANTIQUE embeddings",
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
query_processor = AntiqueQueryProcessor()

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
    return {"status": "healthy", "service": "antique-query-processing"}

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
        processed_query = query_processor.call_text_processing_service(request.query, "query")
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

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "ANTIQUE Query Processing Service",
        "version": "1.0.0",
        "description": "Online query processing with cosine similarity search using ANTIQUE embeddings",
        "dependencies": {
            "text_processing_service": query_processor.text_processing_url
        }
    }

if __name__ == '__main__':
    print("🚀 Starting ANTIQUE Query Processing Service with FastAPI...")
    print("🔍 This service performs similarity search using ANTIQUE embeddings")
    print("🔗 Service will be available at: http://localhost:5002")
    print(f"📡 Text processing service: {query_processor.text_processing_url}")
    print("📖 API docs available at: http://localhost:5002/docs")
    print("⚡ Ready to process queries!")
    
    # Run the service on port 5002
    uvicorn.run(app, host="0.0.0.0", port=5002)
