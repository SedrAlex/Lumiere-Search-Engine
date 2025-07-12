#!/usr/bin/env python3
"""
QUORA TF-IDF Representation Service
A standalone FastAPI service that serves the pre-trained QUORA TF-IDF model.
This service loads the saved models from the models folder and provides search functionality.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import logging
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraTFIDFService:
    """
    QUORA TF-IDF service that loads pre-trained models and provides search functionality.
    """
    
    def __init__(self, models_path="models", text_service_url="http://localhost:5003"):
        """
        Initialize the service.
        
        Args:
            models_path (str): Path to the directory containing model files
            text_service_url (str): URL of the text processing service
        """
        self.models_path = models_path
        self.text_service_url = text_service_url
        self.vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index = None
        self.doc_id_to_index = None
        self.index_to_doc_id = None
        self.feature_names = None
        self.processed_documents = None
        self.is_loaded = False
        
        # Load models on initialization
        self.load_models()
        
    def load_models(self):
        """Load all pre-trained models and indices."""
        logger.info("Loading QUORA TF-IDF models...")
        
        try:
            # Load TF-IDF vectorizer
            vectorizer_path = os.path.join(self.models_path, "quora_tfidf_vectorizer.joblib")
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("✅ TF-IDF vectorizer loaded")
            else:
                logger.error(f"Vectorizer not found at {vectorizer_path}")
                return False
            
            # Load TF-IDF matrix
            matrix_path = os.path.join(self.models_path, "quora_tfidf_matrix.joblib")
            if os.path.exists(matrix_path):
                self.tfidf_matrix = joblib.load(matrix_path)
                logger.info(f"✅ TF-IDF matrix loaded: {self.tfidf_matrix.shape}")
            else:
                logger.error(f"TF-IDF matrix not found at {matrix_path}")
                return False
            
            # Load inverted index
            index_path = os.path.join(self.models_path, "quora_inverted_index.joblib")
            if os.path.exists(index_path):
                self.inverted_index = joblib.load(index_path)
                logger.info(f"✅ Inverted index loaded with {len(self.inverted_index)} terms")
            else:
                logger.warning("Inverted index not found")
            
            # Load document mappings
            doc_id_path = os.path.join(self.models_path, "quora_doc_id_to_index.joblib")
            index_doc_path = os.path.join(self.models_path, "quora_index_to_doc_id.joblib")
            
            if os.path.exists(doc_id_path) and os.path.exists(index_doc_path):
                self.doc_id_to_index = joblib.load(doc_id_path)
                self.index_to_doc_id = joblib.load(index_doc_path)
                logger.info(f"✅ Document mappings loaded: {len(self.doc_id_to_index)} documents")
            else:
                logger.error("Document mappings not found")
                return False
            
            # Load feature names
            features_path = os.path.join(self.models_path, "quora_feature_names.joblib")
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                logger.info(f"✅ Feature names loaded: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names not found")
            
            # Load processed documents (optional)
            docs_path = os.path.join(self.models_path, "quora_processed_documents.joblib")
            if os.path.exists(docs_path):
                self.processed_documents = joblib.load(docs_path)
                logger.info(f"✅ Processed documents loaded: {len(self.processed_documents)} documents")
            else:
                logger.warning("Processed documents not found")
                
            self.is_loaded = True
            logger.info("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            return False
    
    def process_query_text(self, query_text):
        """
        Process query text using the text processing service.
        
        Args:
            query_text (str): Raw query text
            
        Returns:
            str: Processed query text
        """
        try:
            response = requests.post(
                f"{self.text_service_url}/process/query",
                json={"query": query_text},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["processed_query"]
            else:
                logger.warning(f"Text processing service error: {response.status_code}")
                # Fallback to basic processing
                return query_text.lower().strip()
                
        except Exception as e:
            logger.warning(f"Could not connect to text processing service: {e}")
            # Fallback to basic processing
            return query_text.lower().strip()
    
    def search_documents(self, query, top_k=10, use_inverted_index=False):
        """
        Search for documents using TF-IDF similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            use_inverted_index (bool): Whether to use inverted index for search
            
        Returns:
            dict: Search results with document IDs and scores
        """
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Process query text
        processed_query = self.process_query_text(query)
        
        if use_inverted_index and self.inverted_index:
            return self._search_with_inverted_index(processed_query, top_k)
        else:
            return self._search_with_cosine_similarity(processed_query, top_k)
    
    def _search_with_cosine_similarity(self, processed_query, top_k):
        """Search using cosine similarity with the full TF-IDF matrix."""
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include results with positive similarity
                doc_id = self.index_to_doc_id.get(idx, f"doc_{idx}")
                results.append({
                    "doc_id": doc_id,
                    "score": float(similarities[idx]),
                    "rank": len(results) + 1
                })
        
        return {
            "query": processed_query,
            "total_results": len(results),
            "results": results
        }
    
    def _search_with_inverted_index(self, processed_query, top_k):
        """Search using the inverted index for faster term-based retrieval."""
        # Tokenize the processed query
        query_terms = processed_query.split()
        
        # Score documents based on term matches
        doc_scores = {}
        
        for term in query_terms:
            if term in self.inverted_index:
                # Get documents containing this term
                for doc_id, tfidf_score in self.inverted_index[term]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                    doc_scores[doc_id] += tfidf_score
        
        # Sort by score and get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank
            })
        
        return {
            "query": processed_query,
            "total_results": len(results),
            "results": results
        }
    
    def get_document_vector(self, doc_id):
        """
        Get the TF-IDF vector for a specific document.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            dict: Document vector information
        """
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        if doc_id not in self.doc_id_to_index:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_index = self.doc_id_to_index[doc_id]
        doc_vector = self.tfidf_matrix[doc_index]
        
        # Get non-zero terms and their scores
        non_zero_indices = doc_vector.nonzero()[1]
        terms_and_scores = []
        
        for idx in non_zero_indices:
            term = self.feature_names[idx] if self.feature_names else f"term_{idx}"
            score = doc_vector[0, idx]
            terms_and_scores.append({
                "term": term,
                "tfidf_score": float(score)
            })
        
        # Sort by TF-IDF score
        terms_and_scores.sort(key=lambda x: x["tfidf_score"], reverse=True)
        
        return {
            "doc_id": doc_id,
            "doc_index": doc_index,
            "num_terms": len(terms_and_scores),
            "top_terms": terms_and_scores[:20]  # Top 20 terms
        }
    
    def get_service_stats(self):
        """Get service statistics."""
        if not self.is_loaded:
            return {"status": "not_loaded", "error": "Models not loaded"}
        
        return {
            "status": "ready",
            "model_info": {
                "vocabulary_size": len(self.feature_names) if self.feature_names else 0,
                "num_documents": len(self.index_to_doc_id),
                "matrix_shape": list(self.tfidf_matrix.shape),
                "matrix_density": float(self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])),
                "has_inverted_index": self.inverted_index is not None,
                "has_processed_documents": self.processed_documents is not None
            },
            "service_info": {
                "text_service_url": self.text_service_url,
                "models_path": self.models_path
            }
        }

# Initialize FastAPI application
app = FastAPI(
    title="QUORA TF-IDF Representation Service",
    description="Service for QUORA TF-IDF document representation and search",
    version="1.0.0"
)

# Initialize the service
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
tfidf_service = QuoraTFIDFService(models_path=models_path)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results")
    use_inverted_index: bool = Field(default=False, description="Use inverted index for search")

class DocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID")

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "quora-tfidf-representation",
        "models_loaded": tfidf_service.is_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search documents using TF-IDF similarity.
    """
    try:
        results = tfidf_service.search_documents(
            query=request.query,
            top_k=request.top_k,
            use_inverted_index=request.use_inverted_index
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document/vector")
async def get_document_vector(request: DocumentRequest):
    """
    Get TF-IDF vector for a specific document.
    """
    try:
        vector_info = tfidf_service.get_document_vector(request.doc_id)
        return vector_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document vector error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_service_stats():
    """Get service statistics and model information."""
    return tfidf_service.get_service_stats()

@app.get("/vocabulary")
async def get_vocabulary(limit: int = 100):
    """
    Get vocabulary terms (limited for performance).
    """
    if not tfidf_service.is_loaded or not tfidf_service.feature_names:
        raise HTTPException(status_code=503, detail="Models not loaded or vocabulary not available")
    
    vocab_size = len(tfidf_service.feature_names)
    terms = list(tfidf_service.feature_names[:limit])
    
    return {
        "vocabulary_size": vocab_size,
        "returned_terms": len(terms),
        "terms": terms
    }

@app.get("/inverted_index/term/{term}")
async def get_term_documents(term: str, limit: int = 20):
    """
    Get documents containing a specific term from the inverted index.
    """
    if not tfidf_service.is_loaded or not tfidf_service.inverted_index:
        raise HTTPException(status_code=503, detail="Inverted index not available")
    
    if term not in tfidf_service.inverted_index:
        return {
            "term": term,
            "found": False,
            "documents": []
        }
    
    docs = tfidf_service.inverted_index[term][:limit]
    
    return {
        "term": term,
        "found": True,
        "total_documents": len(tfidf_service.inverted_index[term]),
        "returned_documents": len(docs),
        "documents": [{"doc_id": doc_id, "tfidf_score": float(score)} for doc_id, score in docs]
    }

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "QUORA TF-IDF Representation Service",
        "docs": "/docs",
        "health": "/health",
        "models_loaded": tfidf_service.is_loaded
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("🚀 QUORA TF-IDF Representation Service starting up...")
    if tfidf_service.is_loaded:
        logger.info("✅ Service ready to serve requests")
    else:
        logger.error("❌ Service started but models are not loaded")

if __name__ == "__main__":
    print("🚀 Starting QUORA TF-IDF Representation Service...")
    print(f"📁 Models path: {models_path}")
    print("🔗 Make sure the text processing service is running on port 5003")
    uvicorn.run(app, host="0.0.0.0", port=5004)
