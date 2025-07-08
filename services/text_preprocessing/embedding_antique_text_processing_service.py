#!/usr/bin/env python3
"""
ANTIQUE Text Processing Service (Simplified)

This service ONLY provides text cleaning/processing functionality.
It does NOT handle embeddings or similarity search.

The embedding generation and similarity search is handled by:
- backend/services/query_processing/antiqua/embedding_antique_query_processing.py
"""

import os
import sys
import re
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiqueTextProcessor:
    """
    Text processing service using the exact methods from the ANTIQUE notebook.
    This ONLY cleans and processes text - no embeddings or similarity search.
    """
    
    def __init__(self):
        """Initialize the text processor with NLTK resources."""
        self.download_nltk_resources()
        self.setup_preprocessing_tools()
        
    def download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            
    def setup_preprocessing_tools(self):
        """Setup preprocessing tools exactly as in the notebook."""
        # Setup stopwords with exceptions for important semantic words
        self.stop_words = set(stopwords.words('english'))
        self.stop_words = self.stop_words - {
            'not', 'no', 'nor', 'against', 'up', 'down', 
            'over', 'under', 'more', 'most', 'very'
        }
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
    def smart_clean_text(self, text):
        """
        Smart text cleaning function from the ANTIQUE notebook.
        Preserves semantics while normalizing text for embeddings.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs with placeholder
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Normalize specific patterns
        text = re.sub(r'\b\d{4}\b', ' YEAR ', text)  # Years
        text = re.sub(r'\b\d+\.\d+\b', ' DECIMAL ', text)  # Decimals
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)  # Numbers
        
        # Normalize emphasis patterns
        text = re.sub(r'[!]{2,}', ' EMPHASIS ', text)
        text = re.sub(r'[?]{2,}', ' QUESTION ', text)
        
        # Keep important characters and remove isolated special characters
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\;\'\"\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def process_batch(self, texts):
        """
        Process a batch of texts.
        
        Args:
            texts (list): List of texts to process
            
        Returns:
            list: List of processed texts
        """
        return [self.smart_clean_text(text) for text in texts]

# Initialize FastAPI application
app = FastAPI(
    title="ANTIQUE Text Processing Service",
    description="Text cleaning and preprocessing service for ANTIQUE embeddings",
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
text_processor = AntiqueTextProcessor()

# Pydantic models for request/response
class TextProcessingRequest(BaseModel):
    text: str

class QueryProcessingRequest(BaseModel):
    query: str

class BatchProcessingRequest(BaseModel):
    texts: List[str]

class TextProcessingResponse(BaseModel):
    original_text: str
    processed_text: str

class QueryProcessingResponse(BaseModel):
    original_query: str
    processed_query: str

class BatchProcessingResponse(BaseModel):
    processed_texts: List[str]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "antique-text-processing"}

@app.post("/process", response_model=TextProcessingResponse)
async def process_text(request: TextProcessingRequest):
    """
    Process a single text using ANTIQUE cleaning methods.
    """
    try:
        processed_text = text_processor.smart_clean_text(request.text)
        
        return {
            "original_text": request.text,
            "processed_text": processed_text
        }
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@app.post("/process/query", response_model=QueryProcessingResponse)
async def process_query(request: QueryProcessingRequest):
    """
    Process a query text using ANTIQUE cleaning methods.
    This is the same as process_text but with different field names.
    """
    try:
        processed_query = text_processor.smart_clean_text(request.query)
        
        return {
            "original_query": request.query,
            "processed_query": processed_query
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

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

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "ANTIQUE Text Processing Service",
        "version": "1.0.0",
        "description": "Text cleaning and preprocessing for ANTIQUE embeddings",
        "endpoints": {
            "/process": "Process single text",
            "/process/query": "Process query text",
            "/process/batch": "Process multiple texts"
        }
    }

# Add a catch-all route to handle incorrect requests

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def handle_api_routes(path: str):
    """Handle any other API routes that don't match our endpoints."""
    raise HTTPException(
        status_code=404,
        detail=f"Endpoint '/api/{path}' not found. This is a text processing service. Available endpoints: /process, /process/query, /process/batch, /health, /info"
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ANTIQUE Text Processing Service')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the service on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    print("🚀 Starting ANTIQUE Text Processing Service...")
    print("📝 This service provides text cleaning and preprocessing")
    print(f"🔗 Service will be available at: http://localhost:{args.port}")
    print(f"📖 API docs available at: http://localhost:{args.port}/docs")
    print("\n📋 Available endpoints:")
    print(f"  • POST /process - Process single text")
    print(f"  • POST /process/query - Process query text")
    print(f"  • POST /process/batch - Process multiple texts")
    print(f"  • GET /health - Health check")
    print(f"  • GET /info - Service information")
    
    # Run the service on the specified port
    uvicorn.run(app, host=args.host, port=args.port)
