#!/usr/bin/env python3
"""
Quora TF-IDF Text Cleaning Service
Provides text cleaning functionality optimized for Quora questions
Based on the notebook implementation
"""

import os
import ssl
import uvicorn
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

class QuoraTextCleaner:
    """
    Advanced text cleaning class optimized for Quora question pairs
    with semantic preservation and question-specific optimizations.
    """

    def __init__(self):
        # Setup stopwords with exceptions for important question words
        self.stop_words = set(stopwords.words('english'))

        # Remove question words and semantic indicators that are crucial for Quora
        question_words = {
            'what', 'when', 'where', 'why', 'who', 'which', 'how',
            'can', 'could', 'would', 'should', 'will', 'shall',
            'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'not', 'no', 'never', 'none', 'nothing', 'neither',
            'more', 'most', 'less', 'least', 'very', 'quite',
            'much', 'many', 'few', 'some', 'any', 'all',
            'best', 'better', 'good', 'bad', 'right', 'wrong'
        }
        self.stop_words = self.stop_words - question_words

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Common contractions for question text
        self.contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "what's": "what is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "where's": "where is",
            "how's": "how is"
        }

        # Question patterns that should be normalized
        self.question_patterns = {
            r'\bhow do i\b': 'how to',
            r'\bhow can i\b': 'how to',
            r'\bhow should i\b': 'how to',
            r'\bwhat is the best way to\b': 'how to',
            r'\bwhat are the ways to\b': 'how to',
            r'\bwhat are some\b': 'what are',
            r'\bwhat are the\b': 'what are'
        }

    def smart_clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning optimized for Quora questions.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Normalize question patterns
        for pattern, replacement in self.question_patterns.items():
            text = re.sub(pattern, replacement, text)

        # Remove or normalize specific patterns
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)
        text = re.sub(r'<.*?>', ' ', text)

        # Handle numbers more intelligently for questions
        text = re.sub(r'\b(19|20)\d{2}\b', ' YEAR ', text)  # Years
        text = re.sub(r'\b\d+\.\d+\b', ' DECIMAL ', text)  # Decimals
        text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', ' ORDINAL ', text)  # Ordinals
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)  # Other numbers

        # Handle emphasis and punctuation
        text = re.sub(r'[!]{2,}', ' EMPHASIS ', text)
        text = re.sub(r'[?]{2,}', ' MULTIQUEST ', text)
        text = re.sub(r'[.]{3,}', ' ELLIPSIS ', text)

        # Remove special characters but preserve some important ones
        text = re.sub(r'[^a-zA-Z0-9\s\-\'_]', ' ', text)

        # Handle hyphenated words carefully (important for compound terms)
        text = re.sub(r'\b(\w+)-(\w+)\b', r'\1 \2 \1\2', text)  # Keep both forms

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def custom_tokenizer(self, text: str) -> List[str]:
        """
        Custom tokenizer optimized for Quora questions.

        Args:
            text (str): Input text

        Returns:
            list: List of processed tokens
        """
        # Clean the text first
        cleaned_text = self.smart_clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            # Skip very short tokens or stopwords
            if len(token) < 2 or token in self.stop_words:
                continue

            # Skip tokens that are just underscores or dashes
            if re.match(r'^[_\-]+$', token):
                continue

            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)

        return processed_tokens

# Initialize the text cleaner
text_cleaner = QuoraTextCleaner()

# FastAPI app
app = FastAPI(
    title="Quora TF-IDF Text Cleaning Service",
    description="Text cleaning service optimized for Quora questions with TF-IDF processing",
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

# Request models
class TextCleanRequest(BaseModel):
    text: str

class TextCleanResponse(BaseModel):
    original_text: str
    cleaned_text: str
    tokens: List[str]

class BatchTextCleanRequest(BaseModel):
    texts: List[str]

class BatchTextCleanResponse(BaseModel):
    results: List[TextCleanResponse]

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Quora TF-IDF Text Cleaning Service",
        "version": "1.0.0",
        "description": "Text cleaning service optimized for Quora questions",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "POST /clean": "Clean single text",
            "POST /clean/batch": "Clean multiple texts",
            "POST /tokenize": "Tokenize text",
            "POST /test": "Test cleaning with sample text"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Quora TF-IDF Text Cleaning Service",
        "nltk_data_available": True
    }

@app.post("/clean", response_model=TextCleanResponse)
async def clean_text(request: TextCleanRequest):
    """
    Clean text using Quora-optimized cleaning pipeline
    
    This endpoint applies the same text cleaning logic used in the notebook:
    1. Lowercase conversion
    2. Contraction expansion
    3. Question pattern normalization
    4. URL/email normalization
    5. Number handling
    6. Special character removal
    7. Hyphenated word handling
    8. Whitespace normalization
    """
    try:
        cleaned_text = text_cleaner.smart_clean_text(request.text)
        tokens = text_cleaner.custom_tokenizer(request.text)
        
        return TextCleanResponse(
            original_text=request.text,
            cleaned_text=cleaned_text,
            tokens=tokens
        )
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning text: {str(e)}")

@app.post("/clean/batch", response_model=BatchTextCleanResponse)
async def clean_batch_texts(request: BatchTextCleanRequest):
    """Clean multiple texts in batch"""
    try:
        results = []
        for text in request.texts:
            cleaned_text = text_cleaner.smart_clean_text(text)
            tokens = text_cleaner.custom_tokenizer(text)
            
            results.append(TextCleanResponse(
                original_text=text,
                cleaned_text=cleaned_text,
                tokens=tokens
            ))
        
        return BatchTextCleanResponse(results=results)
    except Exception as e:
        logger.error(f"Error cleaning batch texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning batch texts: {str(e)}")

@app.post("/tokenize")
async def tokenize_text(request: TextCleanRequest):
    """Tokenize text using custom tokenizer"""
    try:
        tokens = text_cleaner.custom_tokenizer(request.text)
        cleaned_text = text_cleaner.smart_clean_text(request.text)
        
        return {
            "original_text": request.text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "token_count": len(tokens)
        }
    except Exception as e:
        logger.error(f"Error tokenizing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error tokenizing text: {str(e)}")

@app.post("/test")
async def test_cleaning():
    """Test the cleaning pipeline with sample Quora questions"""
    sample_texts = [
        "What's the best way to learn machine learning? How can I improve my programming skills?",
        "How do I refuse to chose between different things to do in my life?",
        "Did Ben Affleck shine more than Christian Bale as Batman?",
        "What are the effects of demonitization of 500 and 1000 rupees notes on real estate sector?",
        "Why creativity is important?"
    ]
    
    try:
        results = []
        for text in sample_texts:
            cleaned_text = text_cleaner.smart_clean_text(text)
            tokens = text_cleaner.custom_tokenizer(text)
            
            results.append({
                "original": text,
                "cleaned": cleaned_text,
                "tokens": tokens,
                "token_count": len(tokens)
            })
        
        return {
            "message": "Sample cleaning results",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in test cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in test cleaning: {str(e)}")

if __name__ == "__main__":
    # Create SSL context for HTTPS
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Run with HTTPS on port 8001
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        ssl_keyfile=None,  # For development, we'll use HTTP
        ssl_certfile=None,
        log_level="info"
    )
