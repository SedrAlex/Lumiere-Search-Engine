"""
TF-IDF Query Processing Service
Dedicated service for processing user queries and retrieving relevant documents
Uses the same enhanced cleaning pipeline as TF-IDF training for consistency
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ENHANCED_CLEANING_SERVICE_URL = "http://localhost:8003"
MODEL_BASE_PATH = "/Users/raafatmhanna/Desktop/custom-search-engine/backend/models"

# Pre-trained model paths
TFIDF_VECTORIZER_PATH = f"{MODEL_BASE_PATH}/antique_corrected_tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = f"{MODEL_BASE_PATH}/antique_corrected_tfidf_matrix.joblib"
DOCUMENT_METADATA_PATH = f"{MODEL_BASE_PATH}/antique_corrected_document_metadata.joblib"

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_enhanced_cleaning: bool = True
    # similarity_threshold removed - will be auto-calculated

class DocumentResult(BaseModel):
    doc_id: str
    score: float
    text: str
    rank: int
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    query: str
    cleaned_query: str
    results: List[DocumentResult]
    total_results: int
    processing_time_ms: float
    similarity_stats: Dict[str, float]

class StatusResponse(BaseModel):
    service: str
    model_loaded: bool
    documents_count: int
    vocabulary_size: int
    model_info: Dict[str, Any]
    cleaning_service_status: str

class TFIDFQueryProcessor:
    """TF-IDF Query Processing Service with enhanced cleaning and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.document_order = []
        self.model_loaded = False
        
        # HTTP client for enhanced cleaning service
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load pre-trained models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained TF-IDF models"""
        try:
            logger.info("Loading TF-IDF models...")
            
            # Check if all required files exist
            required_files = [TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, DOCUMENT_METADATA_PATH]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                logger.error(f"Missing model files: {missing_files}")
                logger.error("Please ensure TF-IDF models are trained and saved in the models directory")
                return
            
            # Load vectorizer
            logger.info("Loading TF-IDF vectorizer...")
            self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            
            # Load TF-IDF matrix
            logger.info("Loading TF-IDF matrix...")
            self.tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
            
            # Load document metadata
            logger.info("Loading document metadata...")
            metadata = joblib.load(DOCUMENT_METADATA_PATH)
            
            # Extract documents and order
            if isinstance(metadata, dict):
                self.documents = metadata.get('documents', [])
                self.document_order = metadata.get('document_order', [])
            elif isinstance(metadata, list):
                # If metadata is just a list of documents
                self.documents = metadata
                self.document_order = [doc['doc_id'] for doc in metadata]
            else:
                logger.error(f"Unexpected metadata format: {type(metadata)}")
                return
            
            self.model_loaded = True
            
            logger.info(f"✅ Models loaded successfully!")
            logger.info(f"   - Documents: {len(self.documents):,}")
            logger.info(f"   - Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
            logger.info(f"   - Matrix shape: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.model_loaded = False
    
    async def clean_query(self, query: str, use_enhanced: bool = True) -> str:
        """Clean query using enhanced cleaning service"""
        if not use_enhanced:
            return self._basic_clean(query)
        
        try:
            response = await self.http_client.post(
                f"{ENHANCED_CLEANING_SERVICE_URL}/clean",
                json={
                    "text": query,
                    "use_lemmatization": True,
                    "use_stemming": True,
                    "use_spellcheck": False,  # Can be enabled if spell check dictionary is available
                    "remove_stopwords": True,
                    "min_token_length": 2
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["cleaned_text"]
            
        except httpx.RequestError as e:
            logger.warning(f"Enhanced cleaning service unavailable: {e}")
            logger.info("Falling back to basic cleaning")
            return self._basic_clean(query)
        except Exception as e:
            logger.warning(f"Error in enhanced cleaning: {e}")
            return self._basic_clean(query)
    
    def _basic_clean(self, text: str) -> str:
        """Basic fallback cleaning if enhanced service is unavailable"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query and return ranked results using cosine similarity"""
        start_time = time.time()
        
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="TF-IDF models not loaded")
        
        # Clean the query
        cleaned_query = await self.clean_query(request.query, request.use_enhanced_cleaning)
        
        if not cleaned_query.strip():
            return QueryResponse(
                query=request.query,
                cleaned_query=cleaned_query,
                results=[],
                total_results=0,
                processing_time_ms=0,
                similarity_stats={"min": 0, "max": 0, "mean": 0, "std": 0}
            )
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Auto-calculate similarity threshold as mean of similarities
        calculated_threshold = np.mean(similarities)
        valid_indices = np.where(similarities > calculated_threshold)[0]
        valid_similarities = similarities[valid_indices]
        
        if len(valid_similarities) == 0:
            return QueryResponse(
                query=request.query,
                cleaned_query=cleaned_query,
                results=[],
                total_results=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                similarity_stats={"min": 0, "max": 0, "mean": 0, "std": 0}
            )
        
        # Get top-k results
        top_k = min(request.top_k, len(valid_similarities))
        top_indices = valid_indices[np.argsort(valid_similarities)[::-1][:top_k]]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx]
            doc_id = self.document_order[idx]
            
            # Find document by ID
            doc_data = None
            for doc in self.documents:
                if doc['doc_id'] == doc_id:
                    doc_data = doc
                    break
            
            if doc_data:
                results.append(DocumentResult(
                    doc_id=doc_id,
                    score=float(similarity_score),
                    text=doc_data.get('text', doc_data.get('raw_text', '')),
                    rank=rank,
                    metadata={
                        "length": doc_data.get('length', 0),
                        "original_length": len(doc_data.get('raw_text', ''))
                    }
                ))
        
        # Calculate similarity statistics
        similarity_stats = {
            "min": float(np.min(valid_similarities)),
            "max": float(np.max(valid_similarities)),
            "mean": float(np.mean(valid_similarities)),
            "std": float(np.std(valid_similarities))
        }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query processed in {processing_time_ms:.2f}ms, returned {len(results)} results")
        
        return QueryResponse(
            query=request.query,
            cleaned_query=cleaned_query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time_ms,
            similarity_stats=similarity_stats
        )
    
    async def get_status(self) -> StatusResponse:
        """Get service status and model information"""
        # Check enhanced cleaning service status
        cleaning_status = "unavailable"
        try:
            response = await self.http_client.get(f"{ENHANCED_CLEANING_SERVICE_URL}/health")
            if response.status_code == 200:
                cleaning_status = "available"
        except:
            pass
        
        model_info = {}
        if self.model_loaded:
            model_info = {
                "vectorizer_features": len(self.vectorizer.vocabulary_),
                "matrix_shape": list(self.tfidf_matrix.shape),
                "ngram_range": getattr(self.vectorizer, 'ngram_range', None),
                "max_features": getattr(self.vectorizer, 'max_features', None),
                "min_df": getattr(self.vectorizer, 'min_df', None),
                "max_df": getattr(self.vectorizer, 'max_df', None)
            }
        
        return StatusResponse(
            service="TF-IDF Query Processor",
            model_loaded=self.model_loaded,
            documents_count=len(self.documents),
            vocabulary_size=len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            model_info=model_info,
            cleaning_service_status=cleaning_status
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app
app = FastAPI(
    title="TF-IDF Query Processing Service",
    description="Dedicated query processing service for TF-IDF with enhanced cleaning and cosine similarity",
    version="1.0.0"
)

# Global service instance
query_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global query_processor
    query_processor = TFIDFQueryProcessor()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if query_processor:
        await query_processor.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TF-IDF Query Processing Service",
        "version": "1.0.0",
        "description": "Dedicated query processing with enhanced cleaning and cosine similarity",
        "features": [
            "Enhanced text cleaning (lemmatization + stemming)",
            "Cosine similarity ranking",
            "Configurable top-k retrieval",
            "Similarity threshold filtering",
            "Pre-trained ANTIQUE dataset models"
        ],
        "endpoints": {
            "POST /search": "Process query and return ranked results",
            "GET /status": "Get service status and model info",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if query_processor.model_loaded else "degraded",
        "service": "tfidf_query_processor",
        "model_loaded": query_processor.model_loaded,
        "documents_available": len(query_processor.documents) > 0
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed service status"""
    return await query_processor.get_status()

@app.post("/search", response_model=QueryResponse)
async def search_documents(request: QueryRequest):
    """Process query and return ranked documents"""
    try:
        result = await query_processor.process_query(request)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.post("/search/batch")
async def search_batch(queries: List[str], top_k: int = 10):
    """Process multiple queries in batch"""
    try:
        results = []
        for query in queries:
            request = QueryRequest(query=query, top_k=top_k)
            result = await query_processor.process_query(request)
            results.append(result)
        
        return {
            "results": results,
            "query_count": len(queries),
            "total_processing_time_ms": sum(r.processing_time_ms for r in results)
        }
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

if __name__ == "__main__":
    # TF-IDF Query Processor runs on port 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)
