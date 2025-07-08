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
from typing import List
import uvicorn
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraTextProcessor:
    """
    Text processing service for Quora dataset.
    """
    def safe_clean_text(self, text):
        """
        Ultra-safe cleaning that preserves Quora question format.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to string to be safe
        text = str(text)

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def process_batch(self, texts):
        """
        Process a batch of texts.
        """
        return [self.safe_clean_text(text) for text in texts]

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
            "/process": "Process single text",
            "/process/batch": "Process multiple texts"
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

