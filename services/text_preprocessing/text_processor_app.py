#!/usr/bin/env python3
"""
Standalone Text Processing Service API for testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from unified_text_processor import UnifiedTextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Text Processing Service API",
    description="Standalone text processing service with TF-IDF support",
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

# Initialize text processor
text_processor = UnifiedTextProcessor()

# Request models
class LoadTFIDFRequest(BaseModel):
    dataset_name: str
    model_path: str

class CleanTextRequest(BaseModel):
    text: str
    dataset: str

class TokenizeRequest(BaseModel):
    text: str
    dataset: str

class SearchTFIDFRequest(BaseModel):
    query: str
    dataset: str
    top_k: int = 10

class FitTFIDFRequest(BaseModel):
    documents: List[str]
    dataset: str
    doc_ids: Optional[List[str]] = None

class SaveTFIDFRequest(BaseModel):
    dataset: str
    save_path: str

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Text Processing Service",
        "version": "1.0.0",
        "supported_datasets": ["quora", "antique"],
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /datasets": "List loaded datasets",
            "GET /datasets/{dataset}": "Dataset information",
            "POST /tfidf/load": "Load TF-IDF models",
            "POST /text/clean": "Clean text",
            "POST /text/tokenize": "Tokenize text",
            "POST /tfidf/search": "Search using TF-IDF",
            "POST /tfidf/fit": "Fit TF-IDF on documents",
            "POST /tfidf/save": "Save TF-IDF models"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "text_processing_service",
        "loaded_datasets": text_processor.get_available_datasets()
    }

@app.get("/datasets")
async def list_datasets():
    return {
        "loaded_datasets": text_processor.get_available_datasets(),
        "supported_datasets": ["quora", "antique"]
    }

@app.get("/datasets/{dataset}")
async def get_dataset_info(dataset: str):
    try:
        return text_processor.get_dataset_info(dataset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/tfidf/load")
async def load_tfidf_models(request: LoadTFIDFRequest):
    try:
        result = text_processor.load_tfidf_models(
            dataset_name=request.dataset_name,
            model_path=request.model_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text/clean")
async def clean_text(request: CleanTextRequest):
    try:
        cleaned_text = text_processor.clean_text(request.text, request.dataset)
        return {
            "original_text": request.text,
            "cleaned_text": cleaned_text,
            "dataset": request.dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text/tokenize")
async def tokenize_text(request: TokenizeRequest):
    try:
        tokens = text_processor._tokenize_text(request.text, request.dataset)
        return {
            "original_text": request.text,
            "tokens": tokens,
            "dataset": request.dataset,
            "token_count": len(tokens)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tfidf/search")
async def search_tfidf(request: SearchTFIDFRequest):
    try:
        results = text_processor.search_tfidf(
            query=request.query,
            dataset=request.dataset,
            top_k=request.top_k
        )
        return {
            "query": request.query,
            "dataset": request.dataset,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tfidf/fit")
async def fit_tfidf(request: FitTFIDFRequest):
    try:
        result = text_processor.fit_tfidf_on_documents(
            documents=request.documents,
            dataset=request.dataset,
            doc_ids=request.doc_ids
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tfidf/save")
async def save_tfidf_models(request: SaveTFIDFRequest):
    try:
        result = text_processor.save_tfidf_models(
            dataset=request.dataset,
            save_path=request.save_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
