"""
TF-IDF Quora Representation Service
Provides TF-IDF document representation and search functionality for the Quora dataset
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TFIDF_QUORA_MODEL_PATH = "/tmp/tfidf_quora_model.joblib"
TFIDF_QUORA_VECTORS_PATH = "/tmp/tfidf_quora_vectors.joblib"

# Request/Response Models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class IndexDocumentsRequest(BaseModel):
    documents: List[Document]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    vocabulary_size: int
    processing_time: float

class TFIDFQuoraService:
    """TF-IDF document representation and search service for the Quora dataset"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.tfidf_matrix = None
        self.documents = {}
        self.document_order = []
        self.is_trained = False
        
    async def index_documents(self, documents: List[Document]) -> IndexResponse:
        start_time = asyncio.get_event_loop().time()
        
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        # Clean and prepare texts
        texts = [doc.text for doc in documents]
        self.documents = {doc.id: doc for doc in documents}
        self.document_order = [doc.id for doc in documents]
        
        # Fit and transform the TF-IDF model
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        processing_time = asyncio.get_event_loop().time() - start_time
        vocabulary_size = len(self.vectorizer.vocabulary_)
        
        logger.info(f"Indexed {len(documents)} documents in {processing_time:.2f}s")
        logger.info(f"Vocabulary size: {vocabulary_size}")
        
        return IndexResponse(
            message="Documents indexed successfully",
            documents_indexed=len(documents),
            vocabulary_size=vocabulary_size,
            processing_time=processing_time
        )
    
    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        start_time = asyncio.get_event_loop().time()
        
        if not self.is_trained:
            raise ValueError("TF-IDF model not trained. Please index documents first.")
        
        # Transform the query
        query_vector = self.vectorizer.transform([query])
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc_id = self.document_order[idx]
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    document_id=doc_id,
                    score=float(similarities[idx]),
                    text=doc.text,
                    metadata=doc.metadata or {}
                ))
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Search completed in {processing_time:.2f}s, found {len(results)} results")
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=processing_time
        )
    
    def _save_model(self):
        """Save TF-IDF model and vectors to disk using joblib"""
        try:
            # Save model components
            joblib.dump({
                'vectorizer': self.vectorizer,
                'documents': self.documents,
                'document_order': self.document_order
            }, TFIDF_QUORA_MODEL_PATH)
            
            # Save TF-IDF matrix
            joblib.dump(self.tfidf_matrix, TFIDF_QUORA_VECTORS_PATH)
            
            logger.info("TF-IDF Quora model saved successfully with joblib")
        except Exception as e:
            logger.error(f"Error saving TF-IDF model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "is_trained": self.is_trained,
            "documents_count": len(self.documents),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }

# FastAPI app for the TF-IDF Quora service
app = FastAPI(
    title="TF-IDF Quora Representation Service",
    description="Document representation and search using TF-IDF for the Quora dataset",
    version="1.0.0"
)

# Global service instance
tfidf_quora_service = TFIDFQuoraService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return tfidf_quora_service.get_status()

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexDocumentsRequest):
    """Index documents for TF-IDF representation"""
    try:
        result = await tfidf_quora_service.index_documents(request.documents)
        return result
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using TF-IDF similarity"""
    try:
        result = await tfidf_quora_service.search(request.query, request.top_k)
        return result
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    # This service runs on a different port (e.g., 8006)
    uvicorn.run(app, host="0.0.0.0", port=8006)

