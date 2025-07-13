#!/usr/bin/env python3
"""
FastAPI Web Application for Search Engine with SOA Architecture
Main endpoint service that handles all search requests
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import logging
import numpy as np
from contextlib import asynccontextmanager

from services.search_service import SearchService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Search Service instance
search_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize search service on startup"""
    global search_service
    logger.info("Initializing Search Service...")
    
    # Initialize search service
    search_service = SearchService()
    
    logger.info("Search Service ready - datasets can be loaded on demand")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Search Service...")

# Create FastAPI app
app = FastAPI(
    title="Search Engine API with SOA Architecture",
    description="API for search engine with SOA architecture supporting Quora and Antique datasets",
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
class SearchRequest(BaseModel):
    query: str
    dataset: str = "quora"  # quora or antique
    method: str = "hybrid-quora"  # tfidf, embedding, hybrid-quora, hybrid-antique
    top_k: int = 10
    use_faiss: bool = False

class DatasetLoadRequest(BaseModel):
    dataset_name: str = "quora"  # quora or antique
    use_faiss: bool = False

class PreprocessingRequest(BaseModel):
    text: str
    dataset: str = "quora"  # quora or antique

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Search Engine API with SOA Architecture",
        "version": "2.0.0",
        "datasets": ["quora", "antique"],
        "methods": ["tfidf", "embedding", "hybrid-quora", "hybrid-antique"],
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /datasets": "Dataset information",
            "GET /datasets/{dataset_name}": "Specific dataset info",
            "POST /datasets/load": "Load dataset",
            "POST /search": "Search documents",
            "POST /preprocess": "Test text preprocessing",
            "GET /faiss/status": "FAISS availability status"
        },
        "architecture": "Service-Oriented Architecture (SOA)",
        "services": {
            "search_service": "Main search coordination",
            "embedding_service": "Vector operations with FAISS support",
            "text_processor": "Text preprocessing and TF-IDF",
            "database_service": "Data management"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    return {
        "status": "healthy",
        "loaded_datasets": list(search_service.loaded_datasets),
        "message": "Search Service is running"
    }

@app.get("/datasets")
async def get_datasets_info():
    """Get information about all available datasets"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    return search_service.get_available_datasets()

@app.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    try:
        return search_service.get_dataset_info(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/datasets/load")
async def load_dataset(request: DatasetLoadRequest):
    """Load dataset with optional FAISS support"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    try:
        result = search_service.load_dataset(request.dataset_name, request.use_faiss)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search documents using SOA services"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    try:
        result = search_service.search(
            query=request.query,
            dataset=request.dataset,
            method=request.method,
            top_k=request.top_k,
            use_faiss=request.use_faiss
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_text(request: PreprocessingRequest):
    """Test text preprocessing pipeline"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    try:
        # Clean text using the text processor
        cleaned_text = search_service.text_processor.clean_text(request.text, request.dataset)
        
        # Tokenize text
        tokens = search_service.text_processor._tokenize_text(request.text, request.dataset)
        
        return {
            "original_text": request.text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "dataset": request.dataset,
            "preprocessing_steps": {
                "text_cleaning": "Dataset-specific text cleaning applied",
                "tokenization": "Dataset-specific tokenization applied",
                "stopword_removal": "Dataset-specific stopword removal applied",
                "lemmatization": "Applied"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faiss/status")
async def get_faiss_status():
    """Get FAISS availability status"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    return search_service.embedding_service.install_faiss_instructions()

@app.get("/services/status")
async def get_services_status():
    """Get status of all SOA services"""
    global search_service
    
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search Service not initialized")
    
    return {
        "search_service": "running",
        "embedding_service": {
            "status": "running",
            "faiss_available": search_service.embedding_service.faiss_available,
            "loaded_datasets": search_service.embedding_service.get_available_datasets()
        },
        "text_processor": {
            "status": "running",
            "loaded_datasets": search_service.text_processor.get_available_datasets()
        },
        "database_service": {
            "status": "running"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
