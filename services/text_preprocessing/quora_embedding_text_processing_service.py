#!/usr/bin/env python3
"""
Quora Text Processing Service
This service provides text cleaning/processing functionality for Quora embeddings.
"""

import os
import sys
import re
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraTextProcessor:
    """
    Text processing service for Quora dataset.
    """
    
    def __init__(self):
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
            'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out',
            'many', 'then', 'them', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time',
            'two', 'more', 'very', 'when', 'come', 'may', 'say', 'get', 'go', 'no', 'way', 'could', 'my',
            'than', 'first', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'long', 'down', 'day',
            'did', 'get', 'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take', 'only', 'little',
            'work', 'know', 'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very', 'after', 'thing',
            'our', 'just', 'name', 'good', 'sentence', 'man', 'think', 'say', 'great', 'where', 'help',
            'through', 'much', 'before', 'line', 'right', 'too', 'mean', 'old', 'any', 'same', 'tell',
            'boy', 'follow', 'came', 'want', 'show', 'also', 'around', 'form', 'three', 'small', 'set',
            'put', 'end', 'why', 'again', 'turn', 'here', 'off', 'went', 'old', 'number', 'great', 'tell',
            'men', 'say', 'small', 'every', 'found', 'still', 'between', 'mightn', 'being', 'where', 'much',
            'your', 'well', 'without', 'should', 'never', 'does', 'must', 'can', 'cannot', 'might', 'shall',
            'am', 'i', 'you', 'we', 'us', 'about', 'all', 'were', 'one', 'other', 'use', 'or', 'can', 'not'
        }
    def safe_clean_text(self, text):
        """
        Ultra-safe cleaning that preserves Quora question format (matching notebook logic)
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to string to be safe
        text = str(text)

        # Minimal cleaning - preserve most information
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_clean_text(self, text, remove_stopwords=True):
        """
        Advanced text cleaning with optional stopword removal.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to string to be safe
        text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra digits (numbers longer than 4 digits)
        text = re.sub(r'\b\d{5,}\b', '', text)
        
        # Remove single characters (except 'a' and 'i')
        text = re.sub(r'\b(?![ai]\b)[a-z]\b', '', text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def process_batch(self, texts):
        """
        Process a batch of texts.
        """
        return [self.safe_clean_text(text) for text in texts]
        
    def process_batch_advanced(self, texts, remove_stopwords=True):
        """
        Process a batch of texts with advanced cleaning.
        """
        return [self.advanced_clean_text(text, remove_stopwords) for text in texts]

# Initialize FastAPI application
app = FastAPI(
    title="Quora Text Processing Service",
    description="Text cleaning and preprocessing service for Quora embeddings",
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

# Initialize text processor
text_processor = QuoraTextProcessor()

# Pydantic models for request/response
class TextProcessingRequest(BaseModel):
    text: str

class BatchProcessingRequest(BaseModel):
    texts: List[str]

class TextProcessingResponse(BaseModel):
    original_text: str
    processed_text: str

class BatchProcessingResponse(BaseModel):
    processed_texts: List[str]

class QueryProcessingRequest(BaseModel):
    query: str
    remove_stopwords: Optional[bool] = False

class QueryProcessingResponse(BaseModel):
    original_query: str
    processed_query: str

class AdvancedTextProcessingRequest(BaseModel):
    text: str
    remove_stopwords: Optional[bool] = True

class AdvancedTextProcessingResponse(BaseModel):
    original_text: str
    processed_text: str

class AdvancedBatchProcessingRequest(BaseModel):
    texts: List[str]
    remove_stopwords: Optional[bool] = True

class AdvancedBatchProcessingResponse(BaseModel):
    processed_texts: List[str]

@app.post("/process/query", response_model=QueryProcessingResponse)
async def process_query(request: QueryProcessingRequest):
    """
    Process a single query with optional advanced cleaning.
    """
    try:
        if request.remove_stopwords:
            processed_query = text_processor.advanced_clean_text(request.query, remove_stopwords=True)
        else:
            processed_query = text_processor.safe_clean_text(request.query)

        return {
            "original_query": request.query,
            "processed_query": processed_query
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/process", response_model=TextProcessingResponse)
async def process_text(request: TextProcessingRequest):
    """
    Process a single text.
    """
    try:
        processed_text = text_processor.safe_clean_text(request.text)

        return {
            "original_text": request.text,
            "processed_text": processed_text
        }

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@app.post("/process/batch", response_model=BatchProcessingResponse)
async def process_batch(request: BatchProcessingRequest):
    """
    Process multiple texts in a single request.
    """
    try:
        processed_texts = text_processor.process_batch(request.texts)

        return {
            "processed_texts": processed_texts
        }

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/process/advanced", response_model=AdvancedTextProcessingResponse)
async def process_text_advanced(request: AdvancedTextProcessingRequest):
    """
    Process a single text with advanced cleaning options.
    """
    try:
        processed_text = text_processor.advanced_clean_text(request.text, request.remove_stopwords)

        return {
            "original_text": request.text,
            "processed_text": processed_text
        }

    except Exception as e:
        logger.error(f"Error processing text (advanced): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced text processing failed: {str(e)}")

@app.post("/process/batch/advanced", response_model=AdvancedBatchProcessingResponse)
async def process_batch_advanced(request: AdvancedBatchProcessingRequest):
    """
    Process multiple texts with advanced cleaning options.
    """
    try:
        processed_texts = text_processor.process_batch_advanced(request.texts, request.remove_stopwords)

        return {
            "processed_texts": processed_texts
        }

    except Exception as e:
        logger.error(f"Error processing batch (advanced): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced batch processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "quora-text-processing"}

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "Quora Text Processing Service",
        "version": "1.0.0",
        "description": "Text cleaning and preprocessing for Quora embeddings",
        "endpoints": {
            "/process": "Process single text (basic cleaning)",
            "/process/query": "Process single query (basic cleaning, optional stopword removal)",
            "/process/advanced": "Process single text (advanced cleaning with stopword removal)",
            "/process/batch": "Process multiple texts (basic cleaning)",
            "/process/batch/advanced": "Process multiple texts (advanced cleaning with stopword removal)"
        }
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quora Text Processing Service')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the service on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    print("🚀 Starting Quora Text Processing Service...")
    print(f"🔗 Service will be available at: http://localhost:{args.port}")

    # Run the service on the specified port
    uvicorn.run(app, host=args.host, port=args.port)

