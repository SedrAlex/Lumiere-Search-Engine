#!/usr/bin/env python3
"""
TF-IDF ANTIQUE Query Processing Service
- Uses pre-trained TF-IDF models for ANTIQUE dataset
- Integrates with text cleaning service
- Performs cosine similarity search
- Similar to Quora TF-IDF query processing
"""

import ssl
import uvicorn
import joblib
import numpy as np
import requests
import sqlite3
import os
import time
import logging
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from nltk.corpus import wordnet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models using the model loader
from model_loader import load_antique_tfidf_models

# Initialize text cleaner
TEXT_CLEANING_SERVICE_URL = "http://localhost:8008/clean"

# Database path
DB_PATH = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/antiqua/antique_documents.db'

# Load pre-trained models
MODEL_DIR = '/Users/raafatmhanna/Downloads/tfidf-optimized/'
try:
    TfidfVectorizer, TfidfMatrix, InvertedIndex, DocIDs = load_antique_tfidf_models(MODEL_DIR)
    logger.info("All TF-IDF models loaded successfully via model loader")
except Exception as e:
    logger.error(f"Failed to load models via model loader: {str(e)}")
    raise RuntimeError(f"Failed to load TF-IDF models: {str(e)}")

# Pre-load existing document IDs for faster lookup
DB_DOCUMENT_IDS = set()
def load_existing_document_ids():
    """Load all existing document IDs from database into a set for fast lookup"""
    global DB_DOCUMENT_IDS
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id FROM documents")
        DB_DOCUMENT_IDS = set(row[0] for row in cursor.fetchall())
        conn.close()
        logger.info(f"Loaded {len(DB_DOCUMENT_IDS)} existing document IDs from database")
    except Exception as e:
        logger.error(f"Error loading document IDs: {str(e)}")
        DB_DOCUMENT_IDS = set()

# Load existing document IDs at startup
if os.path.exists(DB_PATH):
    load_existing_document_ids()
else:
    logger.warning(f"Database file not found at {DB_PATH}")

# FastAPI app
app = FastAPI(
    title="TF-IDF ANTIQUE Query Processing Service",
    description="Query processing service using TF-IDF for ANTIQUE dataset",
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
        
        # Check if tfidf_processing column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Use IN clause for batch retrieval
        placeholders = ','.join(['?' for _ in doc_ids])
        
        if 'tfidf_processing' in columns:
            cursor.execute(
                f"SELECT doc_id, original_text, processed_text, tfidf_processing FROM documents WHERE doc_id IN ({placeholders})",
                doc_ids
            )
        else:
            cursor.execute(
                f"SELECT doc_id, original_text, processed_text FROM documents WHERE doc_id IN ({placeholders})",
                doc_ids
            )
        
        results = cursor.fetchall()
        conn.close()
        
        # Create a dictionary for fast lookup
        documents = {}
        for result in results:
            if len(result) == 4:  # With tfidf_processing column
                documents[result[0]] = {
                    "doc_id": result[0],
                    "original_text": result[1],
                    "processed_text": result[2],
                    "tfidf_processing": result[3]
                }
            else:  # Without tfidf_processing column
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

def document_exists_in_db(doc_id: str) -> bool:
    """Check if a document exists in the database using pre-loaded set"""
    return doc_id in DB_DOCUMENT_IDS

# Search functions
def expand_query_smart(query, top_n=5):
    """Smart query expansion using WordNet synonyms"""
    words = query.split()
    expanded_terms = set(words)  # Start with original words

    for word in words:
        # Get synonyms from WordNet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemma_names():
                if '_' not in lemma and len(lemma) > 2:
                    synonyms.add(lemma.lower())

        # Add top synonyms (shorter ones first for better precision)
        sorted_synonyms = sorted(synonyms, key=len)[:top_n]
        expanded_terms.update(sorted_synonyms)

    # Remove very short or redundant terms
    filtered_terms = [term for term in expanded_terms if len(term) > 2]
    return ' '.join(filtered_terms)

def search_documents_tfidf(query: str, top_k: int = 10) -> List[Dict[str, any]]:
    """Search documents using TF-IDF matrix and cosine similarity"""
    try:
        # Transform query using the fitted vectorizer
        query_vector = TfidfVectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, TfidfMatrix).flatten()
        
        # Find all documents in the database that have non-zero similarity
        db_matches = []
        for i, doc_id in enumerate(DocIDs):
            if doc_id in DB_DOCUMENT_IDS and similarities[i] > 0:
                db_matches.append({
                    "doc_id": doc_id,
                    "score": float(similarities[i])
                })
        
        # Sort by score and return top-k
        db_matches.sort(key=lambda x: x["score"], reverse=True)
        
        return db_matches[:top_k]
        
    except Exception as e:
        logger.error(f"Error in TF-IDF search: {str(e)}")
        return []

def search_documents_inverted_index(query: str, top_k: int = 10) -> List[Dict[str, any]]:
    """Search documents using inverted index"""
    try:
        # Get query terms using the same tokenizer as training
        query_terms = TfidfVectorizer.build_analyzer()(query)
        
        # Collect candidate documents
        candidate_docs = defaultdict(float)
        
        # This matches the training implementation
        for term in query_terms:
            if term in InvertedIndex:
                for doc_id_item in InvertedIndex[term]:
                    if isinstance(doc_id_item, tuple) and len(doc_id_item) == 2:
                        doc_id, score = doc_id_item
                        candidate_docs[doc_id] += score
                    else:
                        # Handle different inverted index format
                        candidate_docs[doc_id_item] += 1.0
        
        # Sort by score and return top-k
        sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs:
            if score > 0 and document_exists_in_db(str(doc_id)):  # Only include positive scores and existing documents
                results.append({
                    "doc_id": str(doc_id),
                    "score": float(score)
                })
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
        
        return results
        
    except Exception as e:
        logger.error(f"Error in inverted index search: {str(e)}")
        return []

def search_documents_enhanced(query_text, top_k=10, use_expansion=True, use_feedback=True):
    """Enhanced search with query expansion and pseudo-relevance feedback"""
    if not query_text or not query_text.strip():
        return []

    original_query = query_text.strip()

    # Initial TF-IDF search
    query_vector = TfidfVectorizer.transform([original_query])
    scores = cosine_similarity(query_vector, TfidfMatrix).flatten()

    # Query expansion with synonyms
    if use_expansion and np.max(scores) > 0:
        # Expand with WordNet synonyms
        expanded_query = expand_query_smart(original_query, top_n=3)
        if expanded_query != original_query:
            expanded_vector = TfidfVectorizer.transform([expanded_query])
            expanded_scores = cosine_similarity(expanded_vector, TfidfMatrix).flatten()
            # Combine original and expanded scores
            scores = 0.7 * scores + 0.3 * expanded_scores

    # Find all documents in the database that have non-zero similarity
    db_matches = []
    for i, doc_id in enumerate(DocIDs):
        if doc_id in DB_DOCUMENT_IDS and scores[i] > 0:
            db_matches.append((doc_id, scores[i]))
    
    # Sort by score and return top-k
    db_matches.sort(key=lambda x: x[1], reverse=True)
    
    return db_matches[:top_k]

# Request models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    search_method: str = "tfidf"  # "tfidf" or "inverted_index" or "enhanced"
    use_expansion: bool = True
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

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TF-IDF ANTIQUE Query Processing Service",
        "version": "1.0.0",
        "description": "Processes queries using pre-trained ANTIQUE TF-IDF models with database integration",
        "features": [
            "TF-IDF matrix search with cosine similarity",
            "Inverted index search for faster retrieval",
            "Enhanced search with query expansion",
            "Database integration for document storage",
            "Text cleaning service integration"
        ],
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "POST /query": "Process a query using TF-IDF (supports 'tfidf', 'inverted_index', and 'enhanced' methods)"
        },
        "search_methods": {
            "tfidf": "Full TF-IDF matrix search with cosine similarity (most accurate)",
            "inverted_index": "Fast inverted index search (faster but less accurate)",
            "enhanced": "Enhanced search with query expansion using WordNet synonyms"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "TF-IDF ANTIQUE Query Processing Service",
        "models_loaded": True,
        "database_connected": os.path.exists(DB_PATH),
        "tfidf_matrix_shape": TfidfMatrix.shape,
        "vocabulary_size": len(TfidfVectorizer.get_feature_names_out()),
        "inverted_index_terms": len(InvertedIndex),
        "total_documents": len(DocIDs)
    }

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
        elif request.search_method == "enhanced":
            enhanced_results = search_documents_enhanced(cleaned_query, request.top_k, request.use_expansion)
            search_results = [{
                "doc_id": doc_id,
                "score": float(score)
            } for doc_id, score in enhanced_results]
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
    else:
        logger.warning(f"Database file not found at {DB_PATH}")
    
    # Run with HTTP on port 8009
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8009,
        log_level="info"
    )

