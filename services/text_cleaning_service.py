#!/usr/bin/env python3
"""
Text Cleaning Microservice
Provides advanced text cleaning capabilities via HTTP API on port 8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import uvicorn
from contextlib import asynccontextmanager

# Import our text cleaning components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.shared.text_cleaning_methods import TextCleaningMethods, create_text_cleaning_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
text_cleaner = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize text cleaning service on startup"""
    global text_cleaner
    logger.info("Initializing Text Cleaning Service...")
    
    # Initialize the text cleaning service
    text_cleaner = create_text_cleaning_service(language='english')
    
    logger.info("✓ Text Cleaning Service ready on port 8001")
    yield
    
    # Cleanup
    logger.info("Shutting down Text Cleaning Service...")

# Create FastAPI app
app = FastAPI(
    title="Text Cleaning Microservice",
    description="Advanced text cleaning service with spell checking, lemmatization, and stemming",
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
class TextCleanRequest(BaseModel):
    text: str
    method: str = "tfidf"  # basic, advanced, tfidf, embedding, query
    enable_spell_check: Optional[bool] = True
    enable_lemmatization: Optional[bool] = True
    enable_stemming: Optional[bool] = True
    conservative_spell_check: Optional[bool] = True

class BatchTextCleanRequest(BaseModel):
    texts: List[str]
    method: str = "tfidf"

class TextCleanResponse(BaseModel):
    original_text: str
    cleaned_text: str
    method_used: str
    statistics: Dict
    processing_info: Dict

class BatchTextCleanResponse(BaseModel):
    processed_texts: List[str]
    method_used: str
    total_processed: int
    statistics_summary: Dict

class ServiceInfoResponse(BaseModel):
    service_name: str
    version: str
    port: int
    language: str
    available_methods: List[str]
    service_statistics: Dict

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Text Cleaning Microservice",
        "version": "2.0.0",
        "port": 8001,
        "status": "running",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /info": "Detailed service info",
            "POST /clean": "Clean single text",
            "POST /clean/batch": "Clean multiple texts",
            "POST /clean/tfidf": "TF-IDF optimized cleaning",
            "POST /clean/embedding": "Embedding optimized cleaning",
            "POST /clean/query": "Query optimized cleaning",
            "GET /statistics": "Service usage statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    return {
        "status": "healthy",
        "service": "text_cleaning",
        "port": 8001,
        "ready": True
    }

@app.get("/info", response_model=ServiceInfoResponse)
async def get_service_info():
    """Get detailed service information"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    service_stats = text_cleaner.get_service_info()
    
    return ServiceInfoResponse(
        service_name="Text Cleaning Microservice",
        version="2.0.0",
        port=8001,
        language=text_cleaner.language,
        available_methods=["basic", "advanced", "tfidf", "embedding", "query"],
        service_statistics=service_stats
    )

@app.post("/clean", response_model=TextCleanResponse)
async def clean_text(request: TextCleanRequest):
    """Clean text using specified method and parameters"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    try:
        # Choose cleaning method
        if request.method == "basic":
            cleaned_text = text_cleaner.basic_clean(request.text)
        elif request.method == "advanced":
            cleaned_text = text_cleaner.advanced_clean(
                request.text,
                enable_spell_check=request.enable_spell_check,
                enable_lemmatization=request.enable_lemmatization,
                enable_stemming=request.enable_stemming,
                conservative_spell_check=request.conservative_spell_check
            )
        elif request.method == "tfidf":
            cleaned_text = text_cleaner.tfidf_optimized_clean(request.text)
        elif request.method == "embedding":
            cleaned_text = text_cleaner.embedding_optimized_clean(request.text)
        elif request.method == "query":
            cleaned_text = text_cleaner.query_optimized_clean(request.text)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown cleaning method: {request.method}")
        
        # Get processing statistics
        statistics = text_cleaner.get_cleaning_statistics(request.text, cleaned_text)
        
        # Processing info
        processing_info = {
            "method": request.method,
            "spell_check_enabled": request.enable_spell_check if request.method == "advanced" else "method_default",
            "lemmatization_enabled": request.enable_lemmatization if request.method == "advanced" else "method_default",
            "stemming_enabled": request.enable_stemming if request.method == "advanced" else "method_default",
            "conservative_spell_check": request.conservative_spell_check if request.method == "advanced" else "method_default"
        }
        
        return TextCleanResponse(
            original_text=request.text,
            cleaned_text=cleaned_text,
            method_used=request.method,
            statistics=statistics,
            processing_info=processing_info
        )
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning error: {str(e)}")

@app.post("/clean/batch", response_model=BatchTextCleanResponse)
async def clean_texts_batch(request: BatchTextCleanRequest):
    """Clean multiple texts using batch processing"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    try:
        # Clean texts in batch
        cleaned_texts = text_cleaner.batch_clean(request.texts, method=request.method)
        
        # Calculate summary statistics
        total_original_chars = sum(len(text) for text in request.texts)
        total_cleaned_chars = sum(len(text) for text in cleaned_texts)
        
        statistics_summary = {
            "total_texts_processed": len(request.texts),
            "total_original_chars": total_original_chars,
            "total_cleaned_chars": total_cleaned_chars,
            "average_reduction_ratio": 1 - (total_cleaned_chars / max(total_original_chars, 1)),
            "method_used": request.method
        }
        
        return BatchTextCleanResponse(
            processed_texts=cleaned_texts,
            method_used=request.method,
            total_processed=len(cleaned_texts),
            statistics_summary=statistics_summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch text cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch text cleaning error: {str(e)}")

@app.post("/clean/tfidf")
async def clean_for_tfidf(request: dict):
    """Clean text specifically for TF-IDF vectorization"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        cleaned_text = text_cleaner.tfidf_optimized_clean(text)
        statistics = text_cleaner.get_cleaning_statistics(text, cleaned_text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "method": "tfidf_optimized",
            "statistics": statistics,
            "features": {
                "spell_check": "conservative",
                "lemmatization": "enabled",
                "stemming": "enabled",
                "stopwords": "enhanced_set",
                "normalization": "advanced"
            }
        }
    except Exception as e:
        logger.error(f"Error in TF-IDF text cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TF-IDF text cleaning error: {str(e)}")

@app.post("/clean/embedding")
async def clean_for_embedding(request: dict):
    """Clean text specifically for embedding models"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        cleaned_text = text_cleaner.embedding_optimized_clean(text)
        statistics = text_cleaner.get_cleaning_statistics(text, cleaned_text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "method": "embedding_optimized",
            "statistics": statistics,
            "features": {
                "spell_check": "light",
                "structure_preservation": "enabled",
                "sentence_boundaries": "preserved",
                "normalization": "basic"
            }
        }
    except Exception as e:
        logger.error(f"Error in embedding text cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding text cleaning error: {str(e)}")

@app.post("/clean/query")
async def clean_query(request: dict):
    """Clean text specifically for search queries"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        cleaned_query = text_cleaner.query_optimized_clean(query)
        statistics = text_cleaner.get_cleaning_statistics(query, cleaned_query)
        
        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "method": "query_optimized",
            "statistics": statistics,
            "features": {
                "spell_check": "conservative_or_disabled_for_short_queries",
                "lemmatization": "enabled",
                "stemming": "enabled",
                "user_intent": "preserved"
            }
        }
    except Exception as e:
        logger.error(f"Error in query cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query cleaning error: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get service usage statistics"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    try:
        service_info = text_cleaner.get_service_info()
        
        return {
            "service_statistics": service_info,
            "runtime_info": {
                "service_name": "Text Cleaning Microservice",
                "version": "2.0.0",
                "port": 8001,
                "status": "running"
            }
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

@app.post("/cache/clear")
async def clear_cache():
    """Clear internal caches"""
    global text_cleaner
    
    if text_cleaner is None:
        raise HTTPException(status_code=503, detail="Text cleaning service not initialized")
    
    try:
        text_cleaner.clear_cache()
        
        return {
            "message": "Cache cleared successfully",
            "service": "text_cleaning",
            "timestamp": "cache_cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

# Health check for other services to verify this service is running
@app.get("/ping")
async def ping():
    """Simple ping endpoint for service discovery"""
    return {"service": "text_cleaning", "status": "pong", "port": 8001}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
