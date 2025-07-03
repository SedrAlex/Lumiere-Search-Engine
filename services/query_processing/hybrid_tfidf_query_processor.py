"""
Hybrid TF-IDF Query Processing Service
Optimized approach using inverted index for fast candidate retrieval + TF-IDF for accurate scoring
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8005"
INVERTED_INDEX_SERVICE_URL = "http://localhost:8006"
MODEL_BASE_PATH = "/Users/raafatmhanna/Desktop/custom-search-engine/backend/models"

# Pre-trained model paths
TFIDF_VECTORIZER_PATH = f"{MODEL_BASE_PATH}/tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = f"{MODEL_BASE_PATH}/tfidf_matrix.joblib"
DOCUMENT_METADATA_PATH = f"{MODEL_BASE_PATH}/document_metadata.joblib"

# Request/Response Models
class HybridQueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_inverted_index: bool = True
    candidate_multiplier: int = 3  # Get 3x candidates from inverted index
    similarity_threshold: float = 0.0
    scoring_method: str = "tf_idf"  # "tf_idf", "bm25", "count"
    query_type: str = "disjunctive"  # "disjunctive", "conjunctive"

class DocumentResult(BaseModel):
    doc_id: str
    score: float
    text: str
    rank: int
    explanation: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class HybridQueryResponse(BaseModel):
    query: str
    cleaned_query: str
    results: List[DocumentResult]
    total_results: int
    processing_time_ms: float
    search_stats: Dict[str, Any]

class HybridStatusResponse(BaseModel):
    service: str
    model_loaded: bool
    documents_count: int
    vocabulary_size: int
    inverted_index_available: bool
    text_cleaning_available: bool
    optimization_mode: str

class HybridTFIDFQueryProcessor:
    """Hybrid TF-IDF Query Processor with Inverted Index optimization"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.document_order = []
        self.doc_id_to_index = {}  # Fast lookup: doc_id -> matrix index
        self.model_loaded = False
        
        # HTTP clients
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Service availability flags
        self.inverted_index_available = False
        self.text_cleaning_available = False
        
        # Load models and check services
        self._load_models()
        asyncio.create_task(self._check_services())
    
    def _load_models(self):
        """Load pre-trained TF-IDF models"""
        try:
            logger.info("Loading hybrid TF-IDF models...")
            
            # Check required files
            required_files = [TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, DOCUMENT_METADATA_PATH]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                logger.error(f"Missing model files: {missing_files}")
                return
            
            # Load vectorizer
            self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            
            # Load TF-IDF matrix
            self.tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
            
            # Load document metadata
            metadata = joblib.load(DOCUMENT_METADATA_PATH)
            
            # Process metadata
            if isinstance(metadata, list):
                self.documents = metadata
                self.document_order = [doc['doc_id'] for doc in metadata]
            elif isinstance(metadata, dict):
                self.documents = metadata.get('documents', [])
                self.document_order = metadata.get('document_order', [])
            
            # Build doc_id to index mapping for fast lookup
            self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.document_order)}
            
            self.model_loaded = True
            
            logger.info(f"✅ Hybrid models loaded successfully!")
            logger.info(f"   - Documents: {len(self.documents):,}")
            logger.info(f"   - Vocabulary: {len(self.vectorizer.vocabulary_):,}")
            logger.info(f"   - Matrix shape: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.model_loaded = False
    
    async def _check_services(self):
        """Check availability of external services"""
        # Check inverted index service
        try:
            response = await self.http_client.get(f"{INVERTED_INDEX_SERVICE_URL}/health", timeout=5.0)
            self.inverted_index_available = response.status_code == 200
            logger.info(f"Inverted Index Service: {'✅ Available' if self.inverted_index_available else '❌ Unavailable'}")
        except:
            self.inverted_index_available = False
            logger.warning("Inverted Index Service unavailable - falling back to full TF-IDF search")
        
        # Check text cleaning service
        try:
            response = await self.http_client.get(f"{TEXT_CLEANING_SERVICE_URL}/health", timeout=5.0)
            self.text_cleaning_available = response.status_code == 200
            logger.info(f"Text Cleaning Service: {'✅ Available' if self.text_cleaning_available else '❌ Unavailable'}")
        except:
            self.text_cleaning_available = False
            logger.warning("Text Cleaning Service unavailable - using fallback cleaning")
    
    async def clean_query(self, query: str) -> str:
        """Clean query using text cleaning service or fallback"""
        if self.text_cleaning_available:
            try:
                response = await self.http_client.post(
                    f"{TEXT_CLEANING_SERVICE_URL}/clean",
                    json={"text": query, "preserve_document_structure": True}
                )
                response.raise_for_status()
                result = response.json()
                return result["cleaned_text"]
            except Exception as e:
                logger.warning(f"Text cleaning failed: {e}")
        
        # Fallback cleaning
        return self._basic_clean(query)
    
    def _basic_clean(self, text: str) -> str:
        """Basic fallback text cleaning"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    async def _get_candidates_from_inverted_index(self, query_terms: List[str], top_k: int, 
                                                 scoring_method: str, query_type: str) -> List[Tuple[str, float]]:
        """Get candidate documents from inverted index service"""
        try:
            response = await self.http_client.post(
                f"{INVERTED_INDEX_SERVICE_URL}/query_index",
                json={
                    "terms": query_terms,
                    "top_k": top_k,
                    "query_type": query_type,
                    "scoring_method": scoring_method
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["results"]  # List of (doc_id, score) tuples
        except Exception as e:
            logger.warning(f"Inverted index query failed: {e}")
            return []
    
    async def process_hybrid_query(self, request: HybridQueryRequest) -> HybridQueryResponse:
        """Process query using hybrid approach: Inverted Index + TF-IDF"""
        start_time = time.time()
        
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="TF-IDF models not loaded")
        
        # Step 1: Clean the query
        cleaned_query = await self.clean_query(request.query)
        query_terms = cleaned_query.split()
        
        if not query_terms:
            return HybridQueryResponse(
                query=request.query,
                cleaned_query=cleaned_query,
                results=[],
                total_results=0,
                processing_time_ms=0,
                search_stats={"method": "empty_query"}
            )
        
        search_stats = {
            "original_query": request.query,
            "cleaned_query": cleaned_query,
            "query_terms": len(query_terms),
            "optimization_used": "none"
        }
        
        # Step 2: Choose search strategy
        if request.use_inverted_index and self.inverted_index_available:
            # HYBRID APPROACH: Inverted Index + TF-IDF
            results = await self._hybrid_search(request, query_terms, search_stats)
        else:
            # FALLBACK: Full TF-IDF search
            results = await self._full_tfidf_search(request, cleaned_query, search_stats)
        
        processing_time_ms = (time.time() - start_time) * 1000
        search_stats["processing_time_ms"] = processing_time_ms
        
        logger.info(f"Hybrid query processed in {processing_time_ms:.2f}ms using {search_stats['optimization_used']}")
        
        return HybridQueryResponse(
            query=request.query,
            cleaned_query=cleaned_query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time_ms,
            search_stats=search_stats
        )
    
    async def _hybrid_search(self, request: HybridQueryRequest, query_terms: List[str], 
                           search_stats: Dict[str, Any]) -> List[DocumentResult]:
        """Optimized hybrid search: Inverted Index + TF-IDF"""
        
        # Step 1: Get candidates from inverted index (fast!)
        candidate_count = request.top_k * request.candidate_multiplier
        candidates = await self._get_candidates_from_inverted_index(
            query_terms, candidate_count, request.scoring_method, request.query_type
        )
        
        if not candidates:
            logger.warning("No candidates from inverted index, falling back to full search")
            return await self._full_tfidf_search(request, " ".join(query_terms), search_stats)
        
        search_stats.update({
            "optimization_used": "hybrid_inverted_index",
            "candidates_retrieved": len(candidates),
            "candidate_multiplier": request.candidate_multiplier
        })
        
        # Step 2: Get document indices for candidates
        candidate_indices = []
        candidate_doc_ids = []
        inverted_scores = {}
        
        for doc_id, inv_score in candidates:
            if doc_id in self.doc_id_to_index:
                idx = self.doc_id_to_index[doc_id]
                candidate_indices.append(idx)
                candidate_doc_ids.append(doc_id)
                inverted_scores[doc_id] = inv_score
        
        if not candidate_indices:
            logger.warning("No valid candidates found in TF-IDF matrix")
            return []
        
        # Step 3: Calculate TF-IDF scores for candidates only (fast!)
        query_vector = self.vectorizer.transform([" ".join(query_terms)])
        candidate_matrix = self.tfidf_matrix[candidate_indices]
        tfidf_similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        
        # Step 4: Combine inverted index scores with TF-IDF scores
        combined_scores = []
        for i, doc_id in enumerate(candidate_doc_ids):
            # Weighted combination: 70% TF-IDF + 30% Inverted Index
            tfidf_score = tfidf_similarities[i]
            inv_score = inverted_scores[doc_id]
            
            # Normalize inverted index score to [0,1] range
            normalized_inv_score = min(inv_score / 10.0, 1.0) if inv_score > 0 else 0
            
            combined_score = 0.7 * tfidf_score + 0.3 * normalized_inv_score
            combined_scores.append((doc_id, combined_score, tfidf_score, inv_score))
        
        # Step 5: Sort by combined score and return top-k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = combined_scores[:request.top_k]
        
        # Step 6: Build response
        results = []
        for rank, (doc_id, combined_score, tfidf_score, inv_score) in enumerate(top_results, 1):
            if combined_score >= request.similarity_threshold:
                # Find document data
                doc_data = None
                for doc in self.documents:
                    if doc['doc_id'] == doc_id:
                        doc_data = doc
                        break
                
                if doc_data:
                    results.append(DocumentResult(
                        doc_id=doc_id,
                        score=float(combined_score),
                        text=doc_data.get('text', doc_data.get('raw_text', '')),
                        rank=rank,
                        explanation={
                            "combined_score": float(combined_score),
                            "tfidf_score": float(tfidf_score),
                            "inverted_index_score": float(inv_score),
                            "score_combination": "70% TF-IDF + 30% Inverted Index",
                            "method": "hybrid_search"
                        },
                        metadata={
                            "length": doc_data.get('length', 0),
                            "original_length": len(doc_data.get('raw_text', ''))
                        }
                    ))
        
        search_stats.update({
            "final_results": len(results),
            "candidates_used": len(candidate_indices),
            "scoring_combination": "70% TF-IDF + 30% Inverted Index"
        })
        
        return results
    
    async def _full_tfidf_search(self, request: HybridQueryRequest, cleaned_query: str, 
                               search_stats: Dict[str, Any]) -> List[DocumentResult]:
        """Fallback: Full TF-IDF search (when inverted index unavailable)"""
        
        search_stats["optimization_used"] = "full_tfidf_fallback"
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Calculate cosine similarities with ALL documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:request.top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx]
            
            if similarity_score >= request.similarity_threshold:
                doc_id = self.document_order[idx]
                
                # Find document data
                doc_data = None
                for doc in self.documents:
                    if doc['doc_id'] == doc_id:
                        doc_data = doc
                        break
                
                if doc_data:
                    results.append(DocumentResult(
                        doc_id=doc_id,
                        score=float(similarity_score),
                        text=doc_data.get('text', doc_data.get('raw_text', '')),
                        rank=rank,
                        explanation={
                            "tfidf_score": float(similarity_score),
                            "method": "full_tfidf_search"
                        },
                        metadata={
                            "length": doc_data.get('length', 0),
                            "original_length": len(doc_data.get('raw_text', ''))
                        }
                    ))
        
        search_stats.update({
            "documents_scored": len(similarities),
            "final_results": len(results)
        })
        
        return results
    
    async def get_status(self) -> HybridStatusResponse:
        """Get hybrid service status"""
        # Re-check service availability
        await self._check_services()
        
        optimization_mode = "hybrid" if self.inverted_index_available else "full_tfidf"
        
        return HybridStatusResponse(
            service="Hybrid TF-IDF Query Processor",
            model_loaded=self.model_loaded,
            documents_count=len(self.documents),
            vocabulary_size=len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            inverted_index_available=self.inverted_index_available,
            text_cleaning_available=self.text_cleaning_available,
            optimization_mode=optimization_mode
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app
app = FastAPI(
    title="Hybrid TF-IDF Query Processing Service",
    description="Optimized query processing using Inverted Index + TF-IDF for fast and accurate search",
    version="2.0.0"
)

# Global service instance
hybrid_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global hybrid_processor
    hybrid_processor = HybridTFIDFQueryProcessor()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hybrid_processor:
        await hybrid_processor.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Hybrid TF-IDF Query Processing Service",
        "version": "2.0.0",
        "description": "Optimized search using Inverted Index for candidate retrieval + TF-IDF for accurate scoring",
        "optimization_strategy": {
            "step_1": "Clean query and extract terms",
            "step_2": "Get candidates from Inverted Index (fast)",
            "step_3": "Calculate TF-IDF scores for candidates only", 
            "step_4": "Combine scores (70% TF-IDF + 30% Inverted Index)",
            "step_5": "Return top-k results",
            "fallback": "Full TF-IDF search if Inverted Index unavailable"
        },
        "performance": {
            "complexity": "O(C) where C = candidates vs O(N) where N = all documents",
            "speedup": "~5-10x faster for large collections",
            "accuracy": "Maintains high accuracy through score combination"
        },
        "endpoints": {
            "POST /search": "Hybrid search with automatic optimization",
            "GET /status": "Service status and optimization mode",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if hybrid_processor.model_loaded else "degraded",
        "service": "hybrid_tfidf_query_processor",
        "model_loaded": hybrid_processor.model_loaded,
        "optimization_available": hybrid_processor.inverted_index_available
    }

@app.get("/status", response_model=HybridStatusResponse)
async def get_status():
    """Get detailed service status"""
    return await hybrid_processor.get_status()

@app.post("/search", response_model=HybridQueryResponse)
async def search_documents(request: HybridQueryRequest):
    """Hybrid search with automatic optimization"""
    try:
        result = await hybrid_processor.process_hybrid_query(request)
        return result
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")

@app.post("/search/benchmark")
async def benchmark_search(query: str, top_k: int = 10, runs: int = 5):
    """Benchmark hybrid vs full TF-IDF search"""
    try:
        results = {
            "query": query,
            "top_k": top_k,
            "runs": runs,
            "hybrid_times": [],
            "full_tfidf_times": [],
            "speedup": 0
        }
        
        # Benchmark hybrid search
        for _ in range(runs):
            request = HybridQueryRequest(query=query, top_k=top_k, use_inverted_index=True)
            result = await hybrid_processor.process_hybrid_query(request)
            results["hybrid_times"].append(result.processing_time_ms)
        
        # Benchmark full TF-IDF search
        for _ in range(runs):
            request = HybridQueryRequest(query=query, top_k=top_k, use_inverted_index=False)
            result = await hybrid_processor.process_hybrid_query(request)
            results["full_tfidf_times"].append(result.processing_time_ms)
        
        # Calculate averages and speedup
        avg_hybrid = sum(results["hybrid_times"]) / runs
        avg_full = sum(results["full_tfidf_times"]) / runs
        results["avg_hybrid_ms"] = avg_hybrid
        results["avg_full_tfidf_ms"] = avg_full
        results["speedup"] = avg_full / avg_hybrid if avg_hybrid > 0 else 0
        
        return results
        
    except Exception as e:
        logger.error(f"Error in benchmark: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")

if __name__ == "__main__":
    # Hybrid TF-IDF Query Processor runs on port 8009
    uvicorn.run(app, host="0.0.0.0", port=8009)
