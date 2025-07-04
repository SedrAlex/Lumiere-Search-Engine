#!/usr/bin/env python3
"""
Enhanced TF-IDF Microservice
Complete TF-IDF service with inverted index and advanced text processing via HTTP API on port 8003
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import uvicorn
import requests
import joblib
import pickle
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager

# Import enhanced components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.shared.enhanced_tokenizer import EnhancedTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service state
service_state = {
    "vectorizer": None,
    "tfidf_matrix": None,
    "inverted_index": None,
    "doc_id_to_idx": None,
    "idx_to_doc_id": None,
    "document_metadata": None,
    "is_trained": False,
    "search_cache": {},
    "training_stats": None
}

# Service URLs
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Enhanced TF-IDF service on startup"""
    logger.info("Initializing Enhanced TF-IDF Service...")
    logger.info("✓ Enhanced TF-IDF Service ready on port 8003")
    yield
    logger.info("Shutting down Enhanced TF-IDF Service...")

# Create FastAPI app
app = FastAPI(
    title="Enhanced TF-IDF Microservice",
    description="Complete TF-IDF service with inverted index and advanced text processing",
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
class TrainRequest(BaseModel):
    documents: List[str]
    doc_ids: List[str]
    vectorizer_params: Optional[Dict] = None
    build_inverted_index: Optional[bool] = True

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_cache: Optional[bool] = True
    fusion_alpha: Optional[float] = 0.7
    method: Optional[str] = "enhanced_inverted"  # enhanced_inverted, full_matrix

class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    document_text: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    method_used: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

class ServiceInfoResponse(BaseModel):
    service_name: str
    version: str
    port: int
    is_trained: bool
    training_statistics: Optional[Dict] = None

# Helper functions
async def call_text_cleaning_service(text: str, method: str = "tfidf") -> str:
    """Call the text cleaning microservice"""
    try:
        if method == "tfidf":
            response = requests.post(f"{TEXT_CLEANING_SERVICE_URL}/clean/tfidf", json={"text": text})
        elif method == "query":
            response = requests.post(f"{TEXT_CLEANING_SERVICE_URL}/clean/query", json={"query": text})
        else:
            response = requests.post(f"{TEXT_CLEANING_SERVICE_URL}/clean", json={"text": text, "method": method})
        
        if response.status_code == 200:
            if method == "query":
                return response.json().get("cleaned_query", text)
            else:
                return response.json().get("cleaned_text", text)
        else:
            logger.error(f"Text cleaning service error: {response.status_code}")
            return text  # Fallback to original text
    except Exception as e:
        logger.error(f"Error calling text cleaning service: {str(e)}")
        return text  # Fallback to original text

def create_enhanced_vectorizer(**vectorizer_params) -> TfidfVectorizer:
    """Create optimized TF-IDF vectorizer with enhanced tokenizer"""
    # Create enhanced tokenizer
    tokenizer = EnhancedTokenizer(
        enable_spell_check=True,
        enable_lemmatization=True,
        enable_stemming=True,
        language='english',
        min_token_length=3,
        max_token_length=50
    )
    
    # Default parameters optimized for high MAP performance
    default_params = {
        'max_features': 100000,
        'min_df': 2,
        'max_df': 0.85,
        'ngram_range': (1, 3),
        'sublinear_tf': True,
        'norm': 'l2',
        'smooth_idf': True,
        'use_idf': True,
        'tokenizer': tokenizer,
        'preprocessor': None,
        'lowercase': False,
        'stop_words': None,
        'token_pattern': None
    }
    
    # Update with user parameters
    default_params.update(vectorizer_params)
    
    return TfidfVectorizer(**default_params)

def build_optimized_inverted_index(tfidf_matrix, vectorizer, idx_to_doc_id) -> Dict:
    """Build optimized inverted index from TF-IDF matrix"""
    inverted_index = defaultdict(lambda: {
        'postings': [],
        'df': 0,
        'max_tfidf': 0.0,
        'avg_tfidf': 0.0
    })
    
    feature_names = vectorizer.get_feature_names_out()
    coo_matrix = tfidf_matrix.tocoo()
    term_stats = defaultdict(list)
    
    for doc_idx, term_idx, tfidf_score in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        term = feature_names[term_idx]
        doc_id = idx_to_doc_id[doc_idx]
        
        inverted_index[term]['postings'].append((doc_id, float(tfidf_score)))
        term_stats[term].append(float(tfidf_score))
    
    # Calculate statistics and sort postings
    for term in inverted_index:
        scores = term_stats[term]
        inverted_index[term]['df'] = len(scores)
        inverted_index[term]['max_tfidf'] = max(scores)
        inverted_index[term]['avg_tfidf'] = sum(scores) / len(scores)
        inverted_index[term]['postings'].sort(key=lambda x: x[1], reverse=True)
    
    return dict(inverted_index)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced TF-IDF Microservice",
        "version": "2.0.0",
        "port": 8003,
        "status": "running",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /info": "Detailed service info",
            "POST /train": "Train enhanced TF-IDF model",
            "POST /search": "Search using enhanced inverted index",
            "POST /save": "Save trained models",
            "POST /load": "Load pre-trained models",
            "GET /statistics": "Get training statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "enhanced_tfidf",
        "port": 8003,
        "is_trained": service_state["is_trained"],
        "ready": True
    }

@app.get("/info", response_model=ServiceInfoResponse)
async def get_service_info():
    """Get detailed service information"""
    return ServiceInfoResponse(
        service_name="Enhanced TF-IDF Microservice",
        version="2.0.0",
        port=8003,
        is_trained=service_state["is_trained"],
        training_statistics=service_state["training_stats"]
    )

@app.post("/train")
async def train_enhanced_tfidf(request: TrainRequest):
    """Train enhanced TF-IDF model with advanced text processing"""
    try:
        logger.info(f"Training Enhanced TF-IDF on {len(request.documents)} documents...")
        
        if len(request.documents) != len(request.doc_ids):
            raise HTTPException(status_code=400, detail="Number of documents must match number of doc_ids")
        
        # Step 1: Clean documents using text cleaning service
        logger.info("Step 1: Cleaning documents using text cleaning service...")
        cleaned_documents = []
        valid_docs = []
        
        for i, (doc_id, doc_text) in enumerate(zip(request.doc_ids, request.documents)):
            cleaned_text = await call_text_cleaning_service(doc_text, "tfidf")
            
            if cleaned_text.strip():
                cleaned_documents.append(cleaned_text)
                valid_docs.append((doc_id, doc_text, cleaned_text))
            else:
                logger.debug(f"Document {doc_id} is empty after cleaning")
        
        logger.info(f"Valid documents after cleaning: {len(valid_docs)}")
        
        if not valid_docs:
            raise HTTPException(status_code=400, detail="No valid documents after cleaning")
        
        # Step 2: Create enhanced vectorizer
        vectorizer_params = request.vectorizer_params or {}
        service_state["vectorizer"] = create_enhanced_vectorizer(**vectorizer_params)
        
        # Step 3: Fit and transform documents
        logger.info("Step 2: Fitting TF-IDF vectorizer with enhanced tokenization...")
        training_texts = [item[2] for item in valid_docs]
        service_state["tfidf_matrix"] = service_state["vectorizer"].fit_transform(training_texts)
        
        # Step 4: Create document mappings
        valid_doc_ids = [item[0] for item in valid_docs]
        valid_original_texts = [item[1] for item in valid_docs]
        valid_cleaned_texts = [item[2] for item in valid_docs]
        
        service_state["doc_id_to_idx"] = {doc_id: idx for idx, doc_id in enumerate(valid_doc_ids)}
        service_state["idx_to_doc_id"] = {idx: doc_id for doc_id, idx in service_state["doc_id_to_idx"].items()}
        
        # Step 5: Create document metadata
        service_state["document_metadata"] = {
            doc_id: {
                'original_text': valid_original_texts[idx],
                'cleaned_text': valid_cleaned_texts[idx],
                'index': idx
            }
            for doc_id, idx in service_state["doc_id_to_idx"].items()
        }
        
        # Step 6: Build inverted index if requested
        if request.build_inverted_index:
            logger.info("Step 3: Building optimized inverted index...")
            service_state["inverted_index"] = build_optimized_inverted_index(
                service_state["tfidf_matrix"],
                service_state["vectorizer"],
                service_state["idx_to_doc_id"]
            )
        
        # Step 7: Calculate training statistics
        feature_names = service_state["vectorizer"].get_feature_names_out()
        idf_scores = service_state["vectorizer"].idf_
        
        service_state["training_stats"] = {
            'total_documents': len(request.documents),
            'valid_documents': len(valid_docs),
            'documents_filtered': len(request.documents) - len(valid_docs),
            'vocabulary_size': len(feature_names),
            'matrix_shape': list(service_state["tfidf_matrix"].shape),
            'non_zero_entries': int(service_state["tfidf_matrix"].nnz),
            'sparsity': float((1 - service_state["tfidf_matrix"].nnz / (service_state["tfidf_matrix"].shape[0] * service_state["tfidf_matrix"].shape[1])) * 100),
            'inverted_index_terms': len(service_state["inverted_index"]) if service_state["inverted_index"] else 0,
            'avg_idf': float(np.mean(idf_scores)),
            'vectorizer_params': service_state["vectorizer"].get_params()
        }
        
        service_state["is_trained"] = True
        service_state["search_cache"].clear()  # Clear search cache
        
        logger.info("✓ Enhanced TF-IDF training completed successfully!")
        
        return {
            "message": "Enhanced TF-IDF model trained successfully",
            "training_statistics": service_state["training_stats"]
        }
        
    except Exception as e:
        logger.error(f"Error training enhanced TF-IDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search using enhanced inverted index with TF-IDF fusion"""
    if not service_state["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained. Train the model first.")
    
    import time
    start_time = time.time()
    
    try:
        # Check cache
        cache_key = f"{request.query}_{request.top_k}_{request.method}_{request.fusion_alpha}"
        if request.use_cache and cache_key in service_state["search_cache"]:
            cached_results = service_state["search_cache"][cache_key]
            search_time = (time.time() - start_time) * 1000
            return SearchResponse(
                query=request.query,
                method_used=f"{request.method}_cached",
                results=cached_results,
                total_results=len(cached_results),
                search_time_ms=search_time
            )
        
        # Clean query using text cleaning service
        cleaned_query = await call_text_cleaning_service(request.query, "query")
        
        if not cleaned_query.strip():
            return SearchResponse(
                query=request.query,
                method_used=request.method,
                results=[],
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get query TF-IDF vector
        query_vector = service_state["vectorizer"].transform([cleaned_query])
        
        if request.method == "enhanced_inverted" and service_state["inverted_index"]:
            results = await _search_with_enhanced_inverted_index(
                query_vector, cleaned_query, request.top_k, request.fusion_alpha
            )
        else:
            results = _search_with_full_matrix(query_vector, request.top_k)
        
        # Cache results
        if request.use_cache:
            service_state["search_cache"][cache_key] = results
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            method_used=request.method,
            results=results,
            total_results=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

async def _search_with_enhanced_inverted_index(query_vector, cleaned_query: str, top_k: int, fusion_alpha: float) -> List[SearchResult]:
    """Enhanced search using inverted index with TF-IDF fusion"""
    # Get query terms using enhanced tokenizer
    tokenizer = service_state["vectorizer"].tokenizer
    query_terms = tokenizer.tokenize(cleaned_query)
    
    if not query_terms:
        return []
    
    # Collect candidate documents from inverted index
    candidate_docs = set()
    term_doc_scores = defaultdict(float)
    
    # Calculate term weights based on query frequency
    query_term_freq = defaultdict(int)
    for term in query_terms:
        query_term_freq[term] += 1
    
    # Get candidates and calculate inverted index scores
    for term in set(query_terms):
        if term in service_state["inverted_index"]:
            term_data = service_state["inverted_index"][term]
            postings = term_data['postings']
            
            # Term weight based on query frequency and document frequency
            term_weight = query_term_freq[term] * np.log(1 + 1 / max(term_data['df'], 1))
            
            # Add candidate documents with weighted scores
            for doc_id, tfidf_score in postings:
                candidate_docs.add(doc_id)
                term_doc_scores[doc_id] += term_weight * tfidf_score
    
    if not candidate_docs:
        return []
    
    # Convert to indices for TF-IDF similarity calculation
    candidate_indices = [
        service_state["doc_id_to_idx"][doc_id] 
        for doc_id in candidate_docs 
        if doc_id in service_state["doc_id_to_idx"]
    ]
    
    if not candidate_indices:
        return []
    
    # Calculate TF-IDF similarities for candidates
    candidate_matrix = service_state["tfidf_matrix"][candidate_indices]
    tfidf_similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
    
    # Fuse scores: weighted combination of TF-IDF similarity and inverted index score
    fused_scores = []
    max_index_score = max(term_doc_scores.values()) if term_doc_scores else 1.0
    
    for i, idx in enumerate(candidate_indices):
        doc_id = service_state["idx_to_doc_id"][idx]
        tfidf_sim = tfidf_similarities[i]
        index_score = term_doc_scores.get(doc_id, 0) / max_index_score  # Normalize
        
        # Weighted fusion
        fused_score = fusion_alpha * tfidf_sim + (1 - fusion_alpha) * index_score
        fused_scores.append(fused_score)
    
    # Get top results
    top_indices = np.argsort(fused_scores)[-top_k:][::-1]
    
    results = []
    for rank, candidate_idx in enumerate(top_indices):
        original_idx = candidate_indices[candidate_idx]
        doc_id = service_state["idx_to_doc_id"][original_idx]
        
        # Get document text if metadata is available
        document_text = None
        if service_state["document_metadata"] and doc_id in service_state["document_metadata"]:
            document_text = service_state["document_metadata"][doc_id]['original_text'][:200] + "..."
        
        results.append(SearchResult(
            doc_id=doc_id,
            score=float(fused_scores[candidate_idx]),
            rank=rank + 1,
            document_text=document_text
        ))
    
    return results

def _search_with_full_matrix(query_vector, top_k: int) -> List[SearchResult]:
    """Search using full TF-IDF matrix"""
    similarities = cosine_similarity(query_vector, service_state["tfidf_matrix"]).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for rank, idx in enumerate(top_indices):
        doc_id = service_state["idx_to_doc_id"][idx]
        
        # Get document text if metadata is available
        document_text = None
        if service_state["document_metadata"] and doc_id in service_state["document_metadata"]:
            document_text = service_state["document_metadata"][doc_id]['original_text'][:200] + "..."
        
        results.append(SearchResult(
            doc_id=doc_id,
            score=float(similarities[idx]),
            rank=rank + 1,
            document_text=document_text
        ))
    
    return results

@app.get("/statistics")
async def get_training_statistics():
    """Get training statistics"""
    if not service_state["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    return {
        "training_statistics": service_state["training_stats"],
        "service_info": {
            "cache_size": len(service_state["search_cache"]),
            "has_inverted_index": service_state["inverted_index"] is not None
        }
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear search cache"""
    service_state["search_cache"].clear()
    return {"message": "Search cache cleared successfully"}

@app.get("/ping")
async def ping():
    """Simple ping endpoint for service discovery"""
    return {"service": "enhanced_tfidf", "status": "pong", "port": 8003}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
