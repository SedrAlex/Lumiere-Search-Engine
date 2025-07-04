#!/usr/bin/env python3
"""
TF-IDF Vectorizer Microservice
Provides TF-IDF vectorization services via HTTP API on port 8002
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import uvicorn
import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import asynccontextmanager

# Import enhanced tokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.shared.enhanced_tokenizer import EnhancedTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vectorizer instance
tfidf_vectorizer: Optional[TfidfVectorizer] = None

# Port for the text cleaning microservice
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize TF-IDF service on startup"""
    global tfidf_vectorizer
    logger.info("Initializing TF-IDF Vectorization Service...")
    
    # Create enhanced tokenizer
    tokenizer = EnhancedTokenizer(
        enable_spell_check=True,
        enable_lemmatization=True,
        enable_stemming=True
    )
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 3),
        sublinear_tf=True,
        norm='l2',
        smooth_idf=True,
        use_idf=True,
        tokenizer=tokenizer,
        preprocessor=None,
        lowercase=False,
        stop_words=None,
        token_pattern=None
    )
    
    logger.info("✓ TF-IDF Vectorization Service ready on port 8002")
    yield
    
    # Cleanup
    logger.info("Shutting down TF-IDF Vectorization Service...")

# Create FastAPI app
app = FastAPI(
    title="TF-IDF Vectorization Microservice",
    description="TF-IDF vectorization with integrated enhanced tokenization",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class Document(BaseModel):
    text: str
    id: Optional[str] = None

class DocumentsRequest(BaseModel):
    documents: List[Document]

class VectorizeRequest(BaseModel):
    texts: List[str]

class VectorizeResponse(BaseModel):
    document_vectors: Dict[str, Any]
    statistics: Dict[str, Any]
    vectorizer_info: Dict[str, Any]

class TrainRequest(BaseModel):
    documents: List[str]
    doc_ids: List[str]
    build_inverted_index: Optional[bool] = True

class ServiceInfoResponse(BaseModel):
    service_name: str
    version: str
    port: int
    vectorizer_params: Dict[str, Any]


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TF-IDF Vectorization Microservice",
        "version": "2.0.0",
        "port": 8002,
        "status": "running",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /info": "Detailed service info",
            "POST /train": "Train TF-IDF model",
            "POST /vectorize": "Vectorize documents"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global tfidf_vectorizer
    
    if tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="TF-IDF Vectorization service not initialized")
    
    return {
        "status": "healthy",
        "service": "tfidf_vectorization",
        "port": 8002,
        "ready": True
    }

@app.get("/info", response_model=ServiceInfoResponse)
async def get_service_info():
    """Get detailed service information"""
    global tfidf_vectorizer
    
    if tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="TF-IDF service not initialized")

    return ServiceInfoResponse(
        service_name="TF-IDF Vectorization Microservice",
        version="2.0.0",
        port=8002,
        vectorizer_params=tfidf_vectorizer.get_params()
    )

@app.post("/train")
async def train_model(request: TrainRequest):
    """Train TF-IDF model on given documents"""
    global tfidf_vectorizer
    
    if tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="TF-IDF Vectorization service not initialized")
    
    try:
        # Fetch cleaned text using the Text Cleaning Service
        cleaned_texts = []
        for document in request.documents:
            response = requests.post(f"{TEXT_CLEANING_SERVICE_URL}/clean/tfidf", json={"text": document})
            if response.status_code == 200:
                cleaned = response.json().get("cleaned_text", "")
                cleaned_texts.append(cleaned)
            else:
                logger.error(f"Text cleaning service error: {response.content}")
                raise HTTPException(status_code=500, detail=f"Text cleaning error for document")
        
        # Fit TF-IDF with cleaned texts
        tfidf_vectorizer.fit(cleaned_texts)
        
        return {
            "message": "TF-IDF model trained successfully",
            "total_documents": len(request.documents),
            "vectorizer_info": tfidf_vectorizer.get_params()
        }
    except Exception as e:
        logger.error(f"Error training TF-IDF model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest):
    """Vectorize texts using trained TF-IDF vectorizer"""
    global tfidf_vectorizer
    
    if tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="TF-IDF Vectorization service not initialized")
    
    try:
        # Transform texts to vectors
        vectorized_matrix = tfidf_vectorizer.transform(request.texts)
        
        # Convert matrix to dense format
        dense_matrix = vectorized_matrix.todense()
        
        # Prepare response
        document_vectors = {text: dense_matrix[i].tolist() for i, text in enumerate(request.texts)}
        statistics = {
            "total_texts_vectorized": len(request.texts),
            "vector_shape": list(vectorized_matrix.shape),
            "non_zero_features": int(vectorized_matrix.nnz),
            "average_features_per_text": vectorized_matrix.nnz / len(request.texts) if request.texts else 0
        }
        
        return VectorizeResponse(
            document_vectors=document_vectors,
            statistics=statistics,
            vectorizer_info=tfidf_vectorizer.get_params()
        )
    except Exception as e:
        logger.error(f"Error vectorizing texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vectorization error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
