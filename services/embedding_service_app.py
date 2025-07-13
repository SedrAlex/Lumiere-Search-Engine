#!/usr/bin/env python3
"""
Standalone Embedding Service API for testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from embedding_service import EmbeddingService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Embedding Service API",
    description="Standalone embedding service with FAISS support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding service
embedding_service = EmbeddingService()

# Request models
class LoadDatasetRequest(BaseModel):
    dataset_name: str
    embedding_model_path: str
    embeddings_path: str
    documents_path: str
    use_faiss: bool = False

class SearchRequest(BaseModel):
    query: str
    dataset: str
    top_k: int = 10
    use_faiss: bool = False

class ReloadRequest(BaseModel):
    dataset: str
    use_faiss: bool = False

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Embedding Service",
        "version": "1.0.0",
        "faiss_available": embedding_service.faiss_available,
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /datasets": "List loaded datasets",
            "GET /datasets/{dataset}": "Dataset information",
            "POST /datasets/load": "Load dataset",
            "POST /search": "Search embeddings",
            "POST /datasets/reload": "Reload dataset with FAISS",
            "GET /faiss/status": "FAISS installation status"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "embedding_service",
        "faiss_available": embedding_service.faiss_available,
        "loaded_datasets": embedding_service.get_available_datasets()
    }

@app.get("/datasets")
async def list_datasets():
    return {
        "loaded_datasets": embedding_service.get_available_datasets(),
        "faiss_available": embedding_service.faiss_available
    }

@app.get("/datasets/{dataset}")
async def get_dataset_info(dataset: str):
    try:
        return embedding_service.get_dataset_info(dataset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/datasets/load")
async def load_dataset(request: LoadDatasetRequest):
    try:
        result = embedding_service.load_dataset(
            dataset_name=request.dataset_name,
            embedding_model_path=request.embedding_model_path,
            embeddings_path=request.embeddings_path,
            documents_path=request.documents_path,
            use_faiss=request.use_faiss
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_embeddings(request: SearchRequest):
    try:
        results = embedding_service.search(
            query=request.query,
            dataset=request.dataset,
            top_k=request.top_k,
            use_faiss=request.use_faiss
        )
        return {
            "query": request.query,
            "dataset": request.dataset,
            "use_faiss": request.use_faiss,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/reload")
async def reload_dataset(request: ReloadRequest):
    try:
        result = embedding_service.reload_dataset(
            dataset=request.dataset,
            use_faiss=request.use_faiss
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faiss/status")
async def get_faiss_status():
    return embedding_service.install_faiss_instructions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
