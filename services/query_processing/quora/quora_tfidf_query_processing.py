#!/usr/bin/env python3
"""
Query Processing Service using TF-IDF for Quora
- Uses pre-trained TF-IDF models
- Integrates with text cleaning service
- Performs cosine similarity search
- Retrieves documents from database
- Supports both TF-IDF matrix and inverted index search
"""

import ssl
import uvicorn
import joblib
import numpy as np
import requests
import sqlite3
import os
import sys
import pandas as pd
import re
import nltk
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# QuoraTextCleaner is now imported from model_loader module


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained models using the model loader
from model_loader import load_tfidf_models, QuoraTextCleaner

# Initialize text cleaner
text_cleaner = QuoraTextCleaner()
logger.info("Text cleaner initialized")

# Load models using the specialized loader
MODEL_DIR = '/Users/raafatmhanna/Downloads/quora_tfidf_models/'
try:
    TfidfVectorizer, TfidfMatrix, InvertedIndex, DocIDs = load_tfidf_models(MODEL_DIR)
    logger.info("All TF-IDF models loaded successfully via model loader")
except Exception as e:
    logger.error(f"Failed to load models via model loader: {str(e)}")
    raise RuntimeError(f"Failed to load TF-IDF models: {str(e)}")

# Database path
DB_PATH = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_documents.db'

# Text cleaning service configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001/clean"

# FastAPI app
app = FastAPI(
    title="Quora TF-IDF Query Processing Service",
    description="Query processing service using TF-IDF and cosine similarity",
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
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    search_method: str = "tfidf"  # "tfidf" or "inverted_index"
    include_documents: bool = True

class DocumentResult(BaseModel):
    doc_id: str
    score: float
    original_text: Optional[str] = None
    processed_text: Optional[str] = None
    tfidf_processing: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    cleaned_query: str
    search_method: str
    results: List[DocumentResult]
    processing_time: float
    total_results: int

# Database helper functions
def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a document from the database by ID"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tfidf_processing column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'tfidf_processing' in columns:
            cursor.execute(
                "SELECT doc_id, original_text, processed_text, tfidf_processing FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            result = cursor.fetchone()
            if result:
                return {
                    "doc_id": result[0],
                    "original_text": result[1],
                    "processed_text": result[2],
                    "tfidf_processing": result[3]
                }
        else:
            cursor.execute(
                "SELECT doc_id, original_text, processed_text FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            result = cursor.fetchone()
            if result:
                return {
                    "doc_id": result[0],
                    "original_text": result[1],
                    "processed_text": result[2]
                }
        
        conn.close()
        return None
    except Exception as e:
        logger.error(f"Error retrieving document {doc_id}: {str(e)}")
        return None

def get_documents_by_ids(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retrieve multiple documents from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Use IN clause for batch retrieval
        placeholders = ','.join(['?' for _ in doc_ids])
        cursor.execute(
            f"SELECT doc_id, original_text, processed_text FROM documents WHERE doc_id IN ({placeholders})",
            doc_ids
        )
        
        results = cursor.fetchall()
        conn.close()
        
        # Create a dictionary for fast lookup
        documents = {}
        for result in results:
            documents[result[0]] = {
                "doc_id": result[0],
                "original_text": result[1],
                "processed_text": result[2]
            }
        
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {}

def update_document_tfidf_processing(doc_id: str, tfidf_processing: str):
    """Update document with TF-IDF processing information"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # First, check if the column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'tfidf_processing' not in columns:
            # Add the column if it doesn't exist
            cursor.execute("ALTER TABLE documents ADD COLUMN tfidf_processing TEXT")
            logger.info("Added tfidf_processing column to documents table")
        
        # Update the document
        cursor.execute(
            "UPDATE documents SET tfidf_processing = ? WHERE doc_id = ?",
            (tfidf_processing, doc_id)
        )
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error updating document {doc_id}: {str(e)}")

def search_documents_tfidf(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search documents using TF-IDF matrix and cosine similarity"""
    try:
        # Transform query using the fitted vectorizer
        query_vector = TfidfVectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, TfidfMatrix).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append({
                    "doc_id": DocIDs[idx],
                    "score": float(similarities[idx])
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in TF-IDF search: {str(e)}")
        return []

def search_documents_inverted_index(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search documents using inverted index exactly as implemented in training notebook"""
    try:
        # Get query terms using the same tokenizer as training
        query_terms = TfidfVectorizer.build_analyzer()(query)
        
        # Collect candidate documents using the exact approach from training
        candidate_docs = defaultdict(float)
        
        # This matches the training implementation (lines 892-895 in notebook)
        for term in query_terms:
            if term in InvertedIndex:
                for doc_id, score in InvertedIndex[term].items():
                    candidate_docs[doc_id] += score
        
        # Sort by score and return top-k (matches training implementation)
        sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if score > 0:  # Only include positive scores
                results.append({
                    "doc_id": doc_id,
                    "score": float(score)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in inverted index search: {str(e)}")
        return []

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Quora TF-IDF Query Processing Service",
        "version": "1.0.0",
        "description": "Processes queries using pre-trained Quora TF-IDF models with inverted index support",
        "features": [
            "TF-IDF matrix search with cosine similarity",
            "Inverted index search for faster retrieval",
            "Custom Quora-optimized text cleaning",
            "Database integration for document storage",
            "Model validation and diagnostics"
        ],
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /stats": "Get system statistics",
            "GET /validate": "Validate model consistency",
            "POST /query": "Process a query using TF-IDF (supports both 'tfidf' and 'inverted_index' methods)"
        },
        "search_methods": {
            "tfidf": "Full TF-IDF matrix search with cosine similarity (more accurate)",
            "inverted_index": "Fast inverted index search (faster but less accurate)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Quora TF-IDF Query Processing Service",
        "models_loaded": True,
        "database_connected": os.path.exists(DB_PATH),
        "tfidf_matrix_shape": TfidfMatrix.shape,
        "vocabulary_size": len(TfidfVectorizer.get_feature_names_out()),
        "inverted_index_terms": len(InvertedIndex),
        "total_documents": len(DocIDs)
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get database stats
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        db_document_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "database_documents": db_document_count,
            "tfidf_documents": len(DocIDs),
            "tfidf_matrix_shape": TfidfMatrix.shape,
            "vocabulary_size": len(TfidfVectorizer.get_feature_names_out()),
            "inverted_index_terms": len(InvertedIndex),
            "matrix_sparsity": f"{(1 - TfidfMatrix.nnz / (TfidfMatrix.shape[0] * TfidfMatrix.shape[1])) * 100:.2f}%"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"error": str(e)}

@app.get("/validate")
async def validate_models():
    """Validate model consistency and correctness"""
    try:
        validation_results = {
            "consistency_checks": {},
            "sample_tests": {},
            "issues": []
        }
        
        # Check model consistency
        validation_results["consistency_checks"]["tfidf_matrix_docs"] = TfidfMatrix.shape[0]
        validation_results["consistency_checks"]["doc_ids_count"] = len(DocIDs)
        validation_results["consistency_checks"]["vocab_size_vectorizer"] = len(TfidfVectorizer.get_feature_names_out())
        validation_results["consistency_checks"]["vocab_size_matrix"] = TfidfMatrix.shape[1]
        validation_results["consistency_checks"]["inverted_index_terms"] = len(InvertedIndex)
        
        # Check for inconsistencies
        if TfidfMatrix.shape[0] != len(DocIDs):
            validation_results["issues"].append(f"TF-IDF matrix rows ({TfidfMatrix.shape[0]}) != doc IDs count ({len(DocIDs)})")
        
        if TfidfMatrix.shape[1] != len(TfidfVectorizer.get_feature_names_out()):
            validation_results["issues"].append(f"TF-IDF matrix cols ({TfidfMatrix.shape[1]}) != vectorizer vocab ({len(TfidfVectorizer.get_feature_names_out())})")
        
        # Test sample query
        test_query = "How to learn programming"
        try:
            # Test TF-IDF search
            tfidf_results = search_documents_tfidf(test_query, 5)
            validation_results["sample_tests"]["tfidf_search"] = {
                "query": test_query,
                "results_count": len(tfidf_results),
                "top_score": tfidf_results[0]["score"] if tfidf_results else 0
            }
            
            # Test inverted index search
            index_results = search_documents_inverted_index(test_query, 5)
            validation_results["sample_tests"]["inverted_index_search"] = {
                "query": test_query,
                "results_count": len(index_results),
                "top_score": index_results[0]["score"] if index_results else 0
            }
            
            # Check query processing
            query_vector = TfidfVectorizer.transform([test_query])
            validation_results["sample_tests"]["query_processing"] = {
                "query_vector_shape": query_vector.shape,
                "query_vector_nnz": query_vector.nnz,
                "analyzer_terms": TfidfVectorizer.build_analyzer()(test_query)
            }
            
        except Exception as test_error:
            validation_results["issues"].append(f"Sample query test failed: {str(test_error)}")
        
        # Database validation
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            db_count = cursor.fetchone()[0]
            validation_results["consistency_checks"]["database_docs"] = db_count
            
            # Check if doc IDs exist in database
            sample_doc_ids = DocIDs[:5]
            placeholders = ','.join(['?' for _ in sample_doc_ids])
            cursor.execute(f"SELECT COUNT(*) FROM documents WHERE doc_id IN ({placeholders})", sample_doc_ids)
            found_count = cursor.fetchone()[0]
            validation_results["sample_tests"]["doc_id_validation"] = {
                "sample_size": len(sample_doc_ids),
                "found_in_db": found_count
            }
            
            if found_count != len(sample_doc_ids):
                validation_results["issues"].append(f"Some doc IDs not found in database: {found_count}/{len(sample_doc_ids)}")
            
            conn.close()
        except Exception as db_error:
            validation_results["issues"].append(f"Database validation failed: {str(db_error)}")
        
        validation_results["status"] = "valid" if not validation_results["issues"] else "issues_found"
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in model validation: {str(e)}")
        return {"error": str(e), "status": "error"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query using TF-IDF models
    - Cleans the query using the text cleaning service
    - Transforms the query using the TF-IDF vectorizer
    - Computes cosine similarity between query and document vectors
    - Retrieves documents from database
    """
    import time
    start_time = time.time()
    
    try:
        # Call the text cleaning service
        response = requests.post(TEXT_CLEANING_SERVICE_URL, json={"text": request.query})
        response.raise_for_status()
        cleaned_query = response.json().get('cleaned_text')
        
        # Search using specified method
        if request.search_method == "inverted_index":
            search_results = search_documents_inverted_index(cleaned_query, request.top_k)
        else:
            search_results = search_documents_tfidf(cleaned_query, request.top_k)
        
        # Get document IDs for database lookup
        doc_ids = [result["doc_id"] for result in search_results]
        
        # Retrieve documents from database if requested
        documents = {}
        if request.include_documents and doc_ids:
            documents = get_documents_by_ids(doc_ids)
        
        # Build response
        results = []
        for result in search_results:
            doc_id = result["doc_id"]
            doc_result = DocumentResult(
                doc_id=doc_id,
                score=result["score"]
            )
            
            # Add document content if available
            if doc_id in documents:
                doc_result.original_text = documents[doc_id]["original_text"]
                doc_result.processed_text = documents[doc_id]["processed_text"]
                
                # Store TF-IDF processing information
                tfidf_info = {
                    "search_method": request.search_method,
                    "cleaned_query": cleaned_query,
                    "score": result["score"]
                }
                doc_result.tfidf_processing = str(tfidf_info)
                
                # Update database with TF-IDF processing info
                update_document_tfidf_processing(doc_id, str(tfidf_info))
            
            results.append(doc_result)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            cleaned_query=cleaned_query,
            search_method=request.search_method,
            results=results,
            processing_time=processing_time,
            total_results=len(results)
        )
        
    except requests.HTTPError as e:
        logger.error(f"HTTP error calling text cleaning service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning service error: {str(e)}")
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

if __name__ == "__main__":
    # Check if database exists and create tfidf_processing column if needed
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(documents)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'tfidf_processing' not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN tfidf_processing TEXT")
                conn.commit()
                logger.info("Added tfidf_processing column to documents table")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    # Run with HTTP on port 8002
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
