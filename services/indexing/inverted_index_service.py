"""
Inverted Index Service
A standalone service for building and querying inverted indices for efficient document retrieval
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from collections import defaultdict, Counter
import joblib
import os
import uvicorn
import math
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8005"
INVERTED_INDEX_PATH = "/tmp/inverted_index.joblib"
DOCUMENT_STATS_PATH = "/tmp/document_stats.joblib"

# Request/Response Models
class IndexDocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]  # [{"doc_id": str, "text": str, "metadata": dict}]

class QueryRequest(BaseModel):
    terms: List[str]
    top_k: int = 100
    query_type: str = "disjunctive"  # "disjunctive" or "conjunctive"
    scoring_method: str = "tf_idf"   # "tf_idf", "bm25", "count"

class DocumentScore(BaseModel):
    doc_id: str
    score: float
    matched_terms: List[str]
    term_frequencies: Dict[str, int]

class QueryResponse(BaseModel):
    results: List[Tuple[str, float]]  # [(doc_id, score)]
    total_results: int
    processing_time_ms: float
    query_stats: Dict[str, Any]

class IndexStats(BaseModel):
    total_documents: int
    vocabulary_size: int
    total_terms: int
    average_doc_length: float
    index_size_mb: float

@dataclass
class PostingListEntry:
    """Entry in a posting list"""
    doc_id: str
    term_frequency: int
    normalized_tf: float
    positions: List[int] = None  # Optional position information

class InvertedIndexService:
    """High-performance inverted index service with multiple scoring methods"""
    
    def __init__(self):
        self.inverted_index = defaultdict(list)  # term -> List[PostingListEntry]
        self.document_lengths = {}  # doc_id -> document length
        self.document_frequencies = defaultdict(int)  # term -> number of docs containing term
        self.vocabulary = set()
        self.total_documents = 0
        self.total_terms = 0
        self.collection_stats = {}
        
        # HTTP client for text cleaning
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load existing index if available
        self._load_index()
    
    async def clean_text(self, text: str) -> str:
        """Clean text using the TF-IDF text cleaning service"""
        try:
            response = await self.http_client.post(
                f"{TEXT_CLEANING_SERVICE_URL}/clean",
                json={"text": text, "preserve_document_structure": True}
            )
            response.raise_for_status()
            result = response.json()
            return result["cleaned_text"]
        except Exception as e:
            logger.warning(f"Text cleaning service unavailable: {e}")
            return self._basic_clean(text)
    
    def _basic_clean(self, text: str) -> str:
        """Basic fallback text cleaning"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    async def build_index(self, documents: List[Dict[str, Any]]) -> IndexStats:
        """Build inverted index from documents"""
        start_time = time.time()
        logger.info(f"Building inverted index for {len(documents)} documents...")
        
        # Reset index
        self.inverted_index.clear()
        self.document_lengths.clear()
        self.document_frequencies.clear()
        self.vocabulary.clear()
        self.total_documents = len(documents)
        self.total_terms = 0
        
        # Process each document
        for doc in documents:
            doc_id = doc["doc_id"]
            text = doc["text"]
            
            # Clean text
            cleaned_text = await self.clean_text(text)
            terms = cleaned_text.split()
            
            # Calculate term frequencies
            term_counts = Counter(terms)
            doc_length = len(terms)
            self.document_lengths[doc_id] = doc_length
            self.total_terms += doc_length
            
            # Add to vocabulary
            self.vocabulary.update(terms)
            
            # Build posting lists
            for term, tf in term_counts.items():
                # Calculate normalized TF (with sublinear scaling)
                normalized_tf = 1 + math.log(tf) if tf > 0 else 0
                
                # Create posting list entry
                entry = PostingListEntry(
                    doc_id=doc_id,
                    term_frequency=tf,
                    normalized_tf=normalized_tf
                )
                
                self.inverted_index[term].append(entry)
        
        # Calculate document frequencies
        for term in self.vocabulary:
            self.document_frequencies[term] = len(self.inverted_index[term])
        
        # Calculate collection statistics
        self.collection_stats = {
            "total_documents": self.total_documents,
            "vocabulary_size": len(self.vocabulary),
            "total_terms": self.total_terms,
            "average_doc_length": self.total_terms / self.total_documents if self.total_documents > 0 else 0,
            "most_frequent_terms": self._get_most_frequent_terms(10),
            "document_length_stats": self._get_doc_length_stats()
        }
        
        # Save index
        self._save_index()
        
        processing_time = (time.time() - start_time) * 1000
        index_size = self._estimate_index_size()
        
        logger.info(f"Index built in {processing_time:.2f}ms")
        logger.info(f"Vocabulary: {len(self.vocabulary):,} terms")
        logger.info(f"Total terms: {self.total_terms:,}")
        
        return IndexStats(
            total_documents=self.total_documents,
            vocabulary_size=len(self.vocabulary),
            total_terms=self.total_terms,
            average_doc_length=self.collection_stats["average_doc_length"],
            index_size_mb=index_size
        )
    
    async def query_index(self, request: QueryRequest) -> QueryResponse:
        """Query the inverted index"""
        start_time = time.time()
        
        if not self.vocabulary:
            return QueryResponse(
                results=[],
                total_results=0,
                processing_time_ms=0,
                query_stats={"error": "Index not built"}
            )
        
        # Filter query terms to only include terms in vocabulary
        valid_terms = [term for term in request.terms if term in self.vocabulary]
        
        if not valid_terms:
            return QueryResponse(
                results=[],
                total_results=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                query_stats={
                    "original_terms": len(request.terms),
                    "valid_terms": 0,
                    "out_of_vocabulary": len(request.terms)
                }
            )
        
        # Calculate scores based on scoring method
        if request.scoring_method == "tf_idf":
            scores = self._calculate_tf_idf_scores(valid_terms, request.query_type)
        elif request.scoring_method == "bm25":
            scores = self._calculate_bm25_scores(valid_terms, request.query_type)
        else:  # count
            scores = self._calculate_count_scores(valid_terms, request.query_type)
        
        # Sort by score and return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:request.top_k]
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=top_results,
            total_results=len(top_results),
            processing_time_ms=processing_time,
            query_stats={
                "original_terms": len(request.terms),
                "valid_terms": len(valid_terms),
                "out_of_vocabulary": len(request.terms) - len(valid_terms),
                "candidate_documents": len(scores),
                "scoring_method": request.scoring_method,
                "query_type": request.query_type
            }
        )
    
    def _calculate_tf_idf_scores(self, terms: List[str], query_type: str) -> Dict[str, float]:
        """Calculate TF-IDF scores for documents"""
        scores = defaultdict(float)
        
        for term in terms:
            if term not in self.inverted_index:
                continue
            
            # Calculate IDF
            df = self.document_frequencies[term]
            idf = math.log(self.total_documents / df) if df > 0 else 0
            
            # Process posting list
            for entry in self.inverted_index[term]:
                tf_idf_score = entry.normalized_tf * idf
                
                if query_type == "disjunctive":
                    scores[entry.doc_id] += tf_idf_score
                else:  # conjunctive
                    if entry.doc_id not in scores:
                        scores[entry.doc_id] = tf_idf_score
                    else:
                        scores[entry.doc_id] += tf_idf_score
        
        # For conjunctive queries, only keep documents that contain ALL terms
        if query_type == "conjunctive":
            required_docs = None
            for term in terms:
                if term in self.inverted_index:
                    term_docs = {entry.doc_id for entry in self.inverted_index[term]}
                    if required_docs is None:
                        required_docs = term_docs
                    else:
                        required_docs &= term_docs
            
            if required_docs:
                scores = {doc_id: score for doc_id, score in scores.items() if doc_id in required_docs}
            else:
                scores = {}
        
        return scores
    
    def _calculate_bm25_scores(self, terms: List[str], query_type: str, k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        """Calculate BM25 scores for documents"""
        scores = defaultdict(float)
        avg_doc_length = self.collection_stats.get("average_doc_length", 0)
        
        for term in terms:
            if term not in self.inverted_index:
                continue
            
            # Calculate IDF component
            df = self.document_frequencies[term]
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5)) if df > 0 else 0
            
            # Process posting list
            for entry in self.inverted_index[term]:
                doc_length = self.document_lengths.get(entry.doc_id, avg_doc_length)
                
                # BM25 formula
                tf = entry.term_frequency
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                bm25_score = idf * (numerator / denominator)
                
                if query_type == "disjunctive":
                    scores[entry.doc_id] += bm25_score
                else:  # conjunctive
                    if entry.doc_id not in scores:
                        scores[entry.doc_id] = bm25_score
                    else:
                        scores[entry.doc_id] += bm25_score
        
        # Handle conjunctive queries
        if query_type == "conjunctive":
            required_docs = None
            for term in terms:
                if term in self.inverted_index:
                    term_docs = {entry.doc_id for entry in self.inverted_index[term]}
                    if required_docs is None:
                        required_docs = term_docs
                    else:
                        required_docs &= term_docs
            
            if required_docs:
                scores = {doc_id: score for doc_id, score in scores.items() if doc_id in required_docs}
            else:
                scores = {}
        
        return scores
    
    def _calculate_count_scores(self, terms: List[str], query_type: str) -> Dict[str, float]:
        """Calculate simple count-based scores"""
        scores = defaultdict(float)
        
        for term in terms:
            if term not in self.inverted_index:
                continue
            
            for entry in self.inverted_index[term]:
                if query_type == "disjunctive":
                    scores[entry.doc_id] += 1.0
                else:  # conjunctive - will be filtered later
                    scores[entry.doc_id] += 1.0
        
        # For conjunctive queries, only keep documents that contain ALL terms
        if query_type == "conjunctive":
            required_count = len([term for term in terms if term in self.inverted_index])
            scores = {doc_id: score for doc_id, score in scores.items() if score >= required_count}
        
        return scores
    
    def _get_most_frequent_terms(self, top_k: int) -> List[Tuple[str, int]]:
        """Get most frequent terms by document frequency"""
        term_freqs = [(term, df) for term, df in self.document_frequencies.items()]
        return sorted(term_freqs, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _get_doc_length_stats(self) -> Dict[str, float]:
        """Get document length statistics"""
        if not self.document_lengths:
            return {}
        
        lengths = list(self.document_lengths.values())
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths)
        }
    
    def _estimate_index_size(self) -> float:
        """Estimate index size in MB"""
        # Rough estimation based on terms and posting lists
        size_bytes = 0
        for term, postings in self.inverted_index.items():
            size_bytes += len(term.encode('utf-8'))  # Term size
            size_bytes += len(postings) * 64  # Approximate posting entry size
        
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def _save_index(self):
        """Save inverted index to disk"""
        try:
            index_data = {
                "inverted_index": dict(self.inverted_index),
                "document_lengths": self.document_lengths,
                "document_frequencies": dict(self.document_frequencies),
                "vocabulary": list(self.vocabulary),
                "total_documents": self.total_documents,
                "total_terms": self.total_terms,
                "collection_stats": self.collection_stats
            }
            
            joblib.dump(index_data, INVERTED_INDEX_PATH)
            logger.info("Inverted index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving inverted index: {e}")
    
    def _load_index(self):
        """Load inverted index from disk"""
        try:
            if os.path.exists(INVERTED_INDEX_PATH):
                index_data = joblib.load(INVERTED_INDEX_PATH)
                
                # Reconstruct posting lists with PostingListEntry objects
                for term, postings in index_data["inverted_index"].items():
                    reconstructed_postings = []
                    for posting in postings:
                        if isinstance(posting, dict):
                            entry = PostingListEntry(
                                doc_id=posting["doc_id"],
                                term_frequency=posting["term_frequency"],
                                normalized_tf=posting["normalized_tf"]
                            )
                        else:
                            entry = posting
                        reconstructed_postings.append(entry)
                    self.inverted_index[term] = reconstructed_postings
                
                self.document_lengths = index_data["document_lengths"]
                self.document_frequencies = defaultdict(int, index_data["document_frequencies"])
                self.vocabulary = set(index_data["vocabulary"])
                self.total_documents = index_data["total_documents"]
                self.total_terms = index_data["total_terms"]
                self.collection_stats = index_data["collection_stats"]
                
                logger.info(f"Inverted index loaded: {len(self.vocabulary):,} terms, {self.total_documents:,} documents")
                
        except Exception as e:
            logger.error(f"Error loading inverted index: {e}")
    
    def get_stats(self) -> IndexStats:
        """Get index statistics"""
        return IndexStats(
            total_documents=self.total_documents,
            vocabulary_size=len(self.vocabulary),
            total_terms=self.total_terms,
            average_doc_length=self.collection_stats.get("average_doc_length", 0),
            index_size_mb=self._estimate_index_size()
        )
    
    def get_term_info(self, term: str) -> Dict[str, Any]:
        """Get information about a specific term"""
        if term not in self.vocabulary:
            return {"error": f"Term '{term}' not in vocabulary"}
        
        postings = self.inverted_index[term]
        return {
            "term": term,
            "document_frequency": self.document_frequencies[term],
            "total_occurrences": sum(entry.term_frequency for entry in postings),
            "average_tf": np.mean([entry.term_frequency for entry in postings]),
            "idf": math.log(self.total_documents / self.document_frequencies[term]) if self.document_frequencies[term] > 0 else 0,
            "documents": [(entry.doc_id, entry.term_frequency) for entry in postings[:10]]  # First 10 docs
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app
app = FastAPI(
    title="Inverted Index Service",
    description="High-performance inverted index for document retrieval with multiple scoring methods",
    version="1.0.0"
)

# Global service instance
index_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global index_service
    index_service = InvertedIndexService()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if index_service:
        await index_service.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Inverted Index Service",
        "version": "1.0.0",
        "description": "High-performance inverted index with TF-IDF, BM25, and count-based scoring",
        "features": [
            "Multiple scoring methods (TF-IDF, BM25, Count)",
            "Disjunctive and conjunctive query types",
            "Persistent index storage",
            "Term and collection statistics",
            "Optimized posting lists"
        ],
        "endpoints": {
            "POST /build_index": "Build inverted index from documents",
            "POST /query_index": "Query the inverted index",
            "GET /stats": "Get index statistics",
            "GET /term/{term}": "Get term information",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "inverted_index_service",
        "index_ready": len(index_service.vocabulary) > 0,
        "documents_indexed": index_service.total_documents
    }

@app.get("/stats", response_model=IndexStats)
async def get_stats():
    """Get index statistics"""
    return index_service.get_stats()

@app.get("/term/{term}")
async def get_term_info(term: str):
    """Get information about a specific term"""
    return index_service.get_term_info(term)

@app.post("/build_index", response_model=IndexStats)
async def build_index(request: IndexDocumentRequest):
    """Build inverted index from documents"""
    try:
        stats = await index_service.build_index(request.documents)
        return stats
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index building error: {str(e)}")

@app.post("/query_index", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """Query the inverted index"""
    try:
        result = await index_service.query_index(request)
        return result
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/collection_stats")
async def get_collection_stats():
    """Get detailed collection statistics"""
    return {
        "collection_stats": index_service.collection_stats,
        "vocabulary_sample": list(index_service.vocabulary)[:20],
        "most_frequent_terms": index_service._get_most_frequent_terms(20),
        "document_length_stats": index_service._get_doc_length_stats()
    }

if __name__ == "__main__":
    # Inverted Index Service runs on port 8006
    uvicorn.run(app, host="0.0.0.0", port=8006)
