"""
Enhanced TF-IDF Quora Representation Service
Optimized for higher MAP scores with improved vectorization parameters,
query expansion, and advanced scoring techniques specifically for Quora dataset
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib
import os
import uvicorn
from collections import defaultdict, Counter
import math
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8003"
INVERTED_INDEX_SERVICE_URL = "http://localhost:8006"

# Enhanced model paths for Quora
ENHANCED_QUORA_MODEL_PATH = "/tmp/enhanced_tfidf_quora_model.joblib"
ENHANCED_QUORA_VECTORS_PATH = "/tmp/enhanced_tfidf_quora_vectors.joblib"
QUERY_EXPANSION_QUORA_PATH = "/tmp/query_expansion_quora_data.joblib"

# Pre-trained Quora model paths
QUORA_MODEL_PATH = "/tmp/tfidf_quora_vectorizer.joblib"
QUORA_MATRIX_PATH = "/tmp/tfidf_quora_matrix.joblib"
QUORA_METADATA_PATH = "/tmp/quora_document_metadata.joblib"

# Request/Response Models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class IndexDocumentsRequest(BaseModel):
    documents: List[Document]
    use_enhanced_parameters: bool = True
    enable_query_expansion: bool = True

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_query_expansion: bool = True
    enable_reranking: bool = True
    similarity_threshold: float = 0.0
    boost_recent: bool = False

class SearchResult(BaseModel):
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    expanded_query: Optional[str] = None
    results: List[SearchResult]
    total_results: int
    processing_time: float
    search_stats: Dict[str, Any]

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    vocabulary_size: int
    processing_time: float
    model_info: Dict[str, Any]

class EnhancedTFIDFQuoraService:
    """Enhanced TF-IDF service optimized for higher MAP scores on Quora dataset"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.lsa_model = None  # For semantic similarity
        self.lsa_vectors = None
        self.documents = {}
        self.document_order = []
        self.is_trained = False
        self.using_pretrained = False
        
        # Query expansion data
        self.query_expansion_enabled = False
        self.term_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.term_similarities = {}
        
        # Enhanced features
        self.document_lengths = {}
        self.collection_stats = {}
        self.idf_values = {}
        
        # HTTP clients
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Try to load pre-trained model
        self._load_pretrained_quora_model()
        if not self.is_trained:
            self._load_model()
    
    def _get_enhanced_vectorizer_params(self) -> Dict[str, Any]:
        """Get optimized TF-IDF parameters for higher MAP scores on Quora"""
        return {
            'max_features': 100000,  # Increased from 10k to 100k
            'ngram_range': (1, 3),   # Include trigrams for better phrase matching
            'min_df': 2,             # Keep minimum document frequency low
            'max_df': 0.85,          # Slightly more restrictive for common terms
            'sublinear_tf': True,    # Apply log normalization to TF
            'norm': 'l2',            # L2 normalization
            'use_idf': True,         # Use IDF weighting
            'smooth_idf': True,      # Smooth IDF weights
            'token_pattern': r'(?u)\b\w\w+\b',  # Default pattern for word boundaries
            'strip_accents': 'unicode',  # Remove accents for better matching
        }
    
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
        """Enhanced fallback text cleaning"""
        # Preserve more structure while cleaning
        text = text.lower()
        # Keep alphanumeric, spaces, and some punctuation
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        # Handle multiple spaces and dashes
        text = re.sub(r'[-\s]+', ' ', text)
        return text.strip()
    
    async def index_documents(self, request: IndexDocumentsRequest) -> IndexResponse:
        """Index documents with enhanced TF-IDF parameters for Quora"""
        start_time = time.time()
        
        if not request.documents:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"Starting enhanced Quora indexing of {len(request.documents)} documents")
        
        # Clean all document texts
        cleaned_texts = []
        for doc in request.documents:
            cleaned_text = await self.clean_text(doc.text)
            cleaned_texts.append(cleaned_text)
            self.documents[doc.id] = doc
            self.document_lengths[doc.id] = len(doc.text.split())
        
        self.document_order = [doc.id for doc in request.documents]
        
        # Create enhanced TF-IDF vectorizer
        if request.use_enhanced_parameters:
            vectorizer_params = self._get_enhanced_vectorizer_params()
        else:
            vectorizer_params = {
                'max_features': 50000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.8,
                'sublinear_tf': True
            }
        
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        
        # Fit and transform documents
        logger.info("Training enhanced TF-IDF model for Quora...")
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        
        # Store IDF values for later use
        feature_names = self.vectorizer.get_feature_names_out()
        self.idf_values = dict(zip(feature_names, self.vectorizer.idf_))
        
        # Build LSA model for semantic similarity (optional enhancement)
        logger.info("Building LSA model for semantic similarity...")
        self.lsa_model = TruncatedSVD(n_components=min(300, self.tfidf_matrix.shape[1] - 1))
        self.lsa_vectors = self.lsa_model.fit_transform(self.tfidf_matrix)
        self.lsa_vectors = normalize(self.lsa_vectors, norm='l2')
        
        # Build query expansion data if enabled
        if request.enable_query_expansion:
            self._build_query_expansion_data(cleaned_texts)
        
        # Calculate collection statistics
        self._calculate_collection_stats()
        
        # Save enhanced model
        self._save_enhanced_model()
        
        # Index documents in the inverted index service
        await self._index_in_inverted_service(request.documents)
        
        self.is_trained = True
        processing_time = time.time() - start_time
        vocabulary_size = len(self.vectorizer.vocabulary_)
        
        model_info = {
            "vectorizer_params": vectorizer_params,
            "vocabulary_size": vocabulary_size,
            "matrix_shape": list(self.tfidf_matrix.shape),
            "lsa_components": self.lsa_model.n_components,
            "query_expansion_enabled": request.enable_query_expansion,
            "average_doc_length": np.mean(list(self.document_lengths.values())),
            "dataset": "Quora"
        }
        
        logger.info(f"Enhanced Quora indexing completed in {processing_time:.2f}s")
        logger.info(f"Vocabulary size: {vocabulary_size:,}")
        logger.info(f"Matrix shape: {self.tfidf_matrix.shape}")
        
        return IndexResponse(
            message="Quora documents indexed successfully with enhanced parameters",
            documents_indexed=len(request.documents),
            vocabulary_size=vocabulary_size,
            processing_time=processing_time,
            model_info=model_info
        )
    
    async def _index_in_inverted_service(self, documents: List[Document]):
        """Index documents in the separate inverted index service"""
        try:
            # Prepare documents for inverted index
            index_docs = [
                {"doc_id": doc.id, "text": doc.text, "metadata": doc.metadata or {}}
                for doc in documents
            ]
            
            response = await self.http_client.post(
                f"{INVERTED_INDEX_SERVICE_URL}/build_index",
                json={"documents": index_docs}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Inverted index built: {result}")
            
        except Exception as e:
            logger.warning(f"Could not build inverted index: {e}")
    
    def _build_query_expansion_data(self, cleaned_texts: List[str]):
        """Build query expansion data using term co-occurrence"""
        logger.info("Building query expansion data for Quora...")
        
        # Build term co-occurrence matrix
        for text in cleaned_texts:
            terms = text.split()
            # Calculate co-occurrence within a window
            window_size = 5
            for i, term1 in enumerate(terms):
                for j in range(max(0, i - window_size), min(len(terms), i + window_size + 1)):
                    if i != j:
                        term2 = terms[j]
                        self.term_cooccurrence[term1][term2] += 1
        
        # Calculate term similarities based on co-occurrence
        for term1, cooccur_dict in self.term_cooccurrence.items():
            similarities = []
            for term2, count in cooccur_dict.items():
                if count >= 2:  # Minimum co-occurrence threshold
                    # Simple Jaccard-like similarity
                    term1_total = sum(self.term_cooccurrence[term1].values())
                    term2_total = sum(self.term_cooccurrence[term2].values())
                    similarity = count / (term1_total + term2_total - count + 1)
                    similarities.append((term2, similarity))
            
            # Keep top 10 similar terms
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.term_similarities[term1] = similarities[:10]
        
        self.query_expansion_enabled = True
        logger.info(f"Query expansion data built for {len(self.term_similarities)} terms")
    
    def _expand_query(self, query_terms: List[str], max_expansions: int = 3) -> List[str]:
        """Expand query with similar terms"""
        if not self.query_expansion_enabled:
            return query_terms
        
        expanded_terms = list(query_terms)
        
        for term in query_terms:
            if term in self.term_similarities:
                # Add top similar terms
                similar_terms = self.term_similarities[term][:max_expansions]
                for similar_term, similarity in similar_terms:
                    if similarity > 0.1 and similar_term not in expanded_terms:
                        expanded_terms.append(similar_term)
        
        return expanded_terms
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Enhanced search with query expansion and reranking for Quora"""
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Enhanced TF-IDF Quora model not trained. Please index documents first.")
        
        original_query = request.query
        
        # Clean and prepare query
        cleaned_query = await self.clean_text(request.query)
        query_terms = cleaned_query.split()
        
        # Expand query if enabled
        expanded_terms = query_terms
        expanded_query_str = cleaned_query
        
        if request.use_query_expansion and self.query_expansion_enabled:
            expanded_terms = self._expand_query(query_terms, max_expansions=2)
            expanded_query_str = " ".join(expanded_terms)
            logger.debug(f"Quora query expanded from '{cleaned_query}' to '{expanded_query_str}'")
        
        # Get candidates from inverted index service (if available)
        candidate_results = await self._get_inverted_index_candidates(expanded_terms, request.top_k * 3)
        
        # Calculate TF-IDF similarity scores
        query_vector = self.vectorizer.transform([expanded_query_str])
        
        if candidate_results:
            # Score only candidate documents
            candidate_indices = []
            for doc_id, _ in candidate_results:
                if doc_id in self.document_order:
                    candidate_indices.append(self.document_order.index(doc_id))
            
            if candidate_indices:
                candidate_matrix = self.tfidf_matrix[candidate_indices]
                similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
                
                # Combine with inverted index scores
                combined_scores = []
                for i, (doc_id, inv_score) in enumerate(candidate_results):
                    if i < len(similarities):
                        # Weighted combination: 70% cosine similarity + 30% inverted index score
                        combined_score = 0.7 * similarities[i] + 0.3 * min(inv_score, 1.0)
                        combined_scores.append((self.document_order.index(doc_id), combined_score))
                
                # Sort by combined score
                combined_scores.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in combined_scores[:request.top_k * 2]]
                top_similarities = [score for _, score in combined_scores[:request.top_k * 2]]
            else:
                # Fallback to full search
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                top_indices = np.argsort(similarities)[::-1][:request.top_k * 2]
                top_similarities = similarities[top_indices]
        else:
            # Full TF-IDF search
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:request.top_k * 2]
            top_similarities = similarities[top_indices]
        
        # Apply semantic reranking with LSA if enabled
        if request.enable_reranking and self.lsa_vectors is not None:
            top_indices, top_similarities = self._apply_semantic_reranking(
                query_vector, top_indices, top_similarities, expanded_query_str
            )
        
        # Build results
        results = []
        for i, idx in enumerate(top_indices[:request.top_k]):
            similarity_score = top_similarities[i]
            
            if similarity_score >= request.similarity_threshold:
                doc_id = self.document_order[idx]
                doc = self.documents[doc_id]
                
                # Add explanation for transparency
                explanation = {
                    "tfidf_score": float(similarity_score),
                    "rank": i + 1,
                    "matched_terms": self._get_matched_terms(cleaned_query, doc.text),
                    "query_expanded": request.use_query_expansion and len(expanded_terms) > len(query_terms),
                    "dataset": "Quora"
                }
                
                results.append(SearchResult(
                    document_id=doc_id,
                    score=float(similarity_score),
                    text=doc.text,
                    metadata=doc.metadata or {},
                    explanation=explanation
                ))
        
        processing_time = time.time() - start_time
        
        search_stats = {
            "original_query_terms": len(query_terms),
            "expanded_query_terms": len(expanded_terms),
            "candidates_from_index": len(candidate_results) if candidate_results else 0,
            "semantic_reranking_applied": request.enable_reranking,
            "similarity_threshold": request.similarity_threshold,
            "processing_method": "hybrid" if candidate_results else "full_tfidf",
            "dataset": "Quora"
        }
        
        logger.info(f"Enhanced Quora search completed in {processing_time:.3f}s, found {len(results)} results")
        
        return SearchResponse(
            query=original_query,
            expanded_query=expanded_query_str if expanded_query_str != cleaned_query else None,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            search_stats=search_stats
        )
    
    async def _get_inverted_index_candidates(self, terms: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Get candidate documents from inverted index service"""
        try:
            response = await self.http_client.post(
                f"{INVERTED_INDEX_SERVICE_URL}/query_index",
                json={
                    "terms": terms,
                    "top_k": top_k,
                    "query_type": "disjunctive",
                    "scoring_method": "tf_idf"
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["results"]
        except Exception as e:
            logger.debug(f"Inverted index service unavailable: {e}")
            return []
    
    def _apply_semantic_reranking(self, query_vector, top_indices, top_similarities, expanded_query):
        """Apply semantic reranking using LSA"""
        try:
            # Transform query to LSA space
            query_lsa = self.lsa_model.transform(query_vector)
            query_lsa = normalize(query_lsa, norm='l2')
            
            # Get LSA vectors for top documents
            doc_lsa_vectors = self.lsa_vectors[top_indices]
            
            # Calculate semantic similarities
            semantic_similarities = np.dot(doc_lsa_vectors, query_lsa.T).flatten()
            
            # Combine TF-IDF and semantic scores (60% TF-IDF + 40% semantic)
            combined_scores = 0.6 * np.array(top_similarities) + 0.4 * semantic_similarities
            
            # Re-sort by combined scores
            rerank_order = np.argsort(combined_scores)[::-1]
            reranked_indices = [top_indices[i] for i in rerank_order]
            reranked_scores = [combined_scores[i] for i in rerank_order]
            
            return reranked_indices, reranked_scores
            
        except Exception as e:
            logger.warning(f"Semantic reranking failed: {e}")
            return top_indices, top_similarities
    
    def _get_matched_terms(self, query: str, document: str) -> List[str]:
        """Get terms that match between query and document"""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        return list(query_terms.intersection(doc_terms))
    
    def _calculate_collection_stats(self):
        """Calculate enhanced collection statistics"""
        doc_lengths = list(self.document_lengths.values())
        vocab_size = len(self.vectorizer.vocabulary_)
        
        self.collection_stats = {
            "total_documents": len(self.documents),
            "vocabulary_size": vocab_size,
            "average_doc_length": np.mean(doc_lengths),
            "median_doc_length": np.median(doc_lengths),
            "std_doc_length": np.std(doc_lengths),
            "min_doc_length": np.min(doc_lengths),
            "max_doc_length": np.max(doc_lengths),
            "tfidf_matrix_density": self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]),
            "lsa_explained_variance": np.sum(self.lsa_model.explained_variance_ratio_) if self.lsa_model else 0,
            "dataset": "Quora"
        }
    
    def _save_enhanced_model(self):
        """Save enhanced model components"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'lsa_model': self.lsa_model,
                'documents': self.documents,
                'document_order': self.document_order,
                'document_lengths': self.document_lengths,
                'collection_stats': self.collection_stats,
                'idf_values': self.idf_values,
                'term_similarities': self.term_similarities,
                'query_expansion_enabled': self.query_expansion_enabled,
                'dataset': 'Quora'
            }
            
            joblib.dump(model_data, ENHANCED_QUORA_MODEL_PATH)
            joblib.dump({
                'tfidf_matrix': self.tfidf_matrix,
                'lsa_vectors': self.lsa_vectors
            }, ENHANCED_QUORA_VECTORS_PATH)
            
            logger.info("Enhanced TF-IDF Quora model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving enhanced Quora model: {e}")
    
    def _load_model(self):
        """Load enhanced model from disk"""
        try:
            if os.path.exists(ENHANCED_QUORA_MODEL_PATH) and os.path.exists(ENHANCED_QUORA_VECTORS_PATH):
                # Load model components
                model_data = joblib.load(ENHANCED_QUORA_MODEL_PATH)
                vector_data = joblib.load(ENHANCED_QUORA_VECTORS_PATH)
                
                self.vectorizer = model_data['vectorizer']
                self.lsa_model = model_data.get('lsa_model')
                self.documents = model_data['documents']
                self.document_order = model_data['document_order']
                self.document_lengths = model_data.get('document_lengths', {})
                self.collection_stats = model_data.get('collection_stats', {})
                self.idf_values = model_data.get('idf_values', {})
                self.term_similarities = model_data.get('term_similarities', {})
                self.query_expansion_enabled = model_data.get('query_expansion_enabled', False)
                
                self.tfidf_matrix = vector_data['tfidf_matrix']
                self.lsa_vectors = vector_data.get('lsa_vectors')
                
                self.is_trained = True
                logger.info("Enhanced TF-IDF Quora model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading enhanced Quora model: {e}")
    
    def _load_pretrained_quora_model(self):
        """Load pre-trained Quora model and enhance it"""
        try:
            if os.path.exists(QUORA_MODEL_PATH):
                logger.info("Loading and enhancing pre-trained Quora model...")
                
                # Load basic components
                self.vectorizer = joblib.load(QUORA_MODEL_PATH)
                self.tfidf_matrix = joblib.load(QUORA_MATRIX_PATH)
                metadata = joblib.load(QUORA_METADATA_PATH)
                
                # Process metadata
                if isinstance(metadata, list):
                    self.documents = {doc['doc_id']: Document(
                        id=doc['doc_id'],
                        text=doc['raw_text'],
                        metadata={'length': doc.get('length', 0)}
                    ) for doc in metadata}
                    self.document_order = [doc['doc_id'] for doc in metadata]
                    self.document_lengths = {doc['doc_id']: len(doc['raw_text'].split()) for doc in metadata}
                
                # Build LSA model for pre-trained data
                logger.info("Building LSA model for pre-trained Quora data...")
                self.lsa_model = TruncatedSVD(n_components=min(300, self.tfidf_matrix.shape[1] - 1))
                self.lsa_vectors = self.lsa_model.fit_transform(self.tfidf_matrix)
                self.lsa_vectors = normalize(self.lsa_vectors, norm='l2')
                
                # Calculate collection stats
                self._calculate_collection_stats()
                
                self.is_trained = True
                self.using_pretrained = True
                
                logger.info(f"Enhanced pre-trained Quora model loaded!")
                logger.info(f"Documents: {len(self.documents):,}")
                logger.info(f"Vocabulary: {len(self.vectorizer.vocabulary_):,}")
                logger.info(f"LSA components: {self.lsa_model.n_components}")
                
        except Exception as e:
            logger.error(f"Error loading pre-trained Quora model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced service status"""
        return {
            "is_trained": self.is_trained,
            "using_pretrained": self.using_pretrained,
            "model_type": "enhanced_pretrained_quora" if self.using_pretrained else "enhanced_custom_quora",
            "documents_count": len(self.documents),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            "lsa_enabled": self.lsa_model is not None,
            "query_expansion_enabled": self.query_expansion_enabled,
            "collection_stats": self.collection_stats,
            "matrix_shape": list(self.tfidf_matrix.shape) if self.tfidf_matrix is not None else None,
            "dataset": "Quora",
            "services": {
                "text_cleaning": TEXT_CLEANING_SERVICE_URL,
                "inverted_index": INVERTED_INDEX_SERVICE_URL
            }
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app
app = FastAPI(
    title="Enhanced TF-IDF Quora Representation Service",
    description="Advanced TF-IDF service optimized for higher MAP scores on Quora dataset",
    version="2.0.0"
)

# Global service instance
enhanced_tfidf_quora_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global enhanced_tfidf_quora_service
    enhanced_tfidf_quora_service = EnhancedTFIDFQuoraService()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if enhanced_tfidf_quora_service:
        await enhanced_tfidf_quora_service.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced TF-IDF Quora Representation Service",
        "version": "2.0.0",
        "description": "Advanced TF-IDF service optimized for higher MAP scores on Quora dataset",
        "dataset": "Quora",
        "enhancements": [
            "Increased vocabulary size (100k features)",
            "Trigram support for better phrase matching",
            "Query expansion using term co-occurrence",
            "Semantic reranking with LSA",
            "Hybrid search with inverted index",
            "Advanced text cleaning pipeline",
            "Detailed scoring explanations",
            "Optimized for Quora question-answer pairs"
        ],
        "endpoints": {
            "POST /index": "Index documents with enhanced parameters",
            "POST /search": "Enhanced search with query expansion",
            "GET /status": "Get service status and statistics",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "enhanced_tfidf_quora_service",
        "dataset": "Quora",
        "is_trained": enhanced_tfidf_quora_service.is_trained,
        "model_type": "enhanced_pretrained_quora" if enhanced_tfidf_quora_service.using_pretrained else "enhanced_custom_quora",
        "features_enabled": {
            "lsa": enhanced_tfidf_quora_service.lsa_model is not None,
            "query_expansion": enhanced_tfidf_quora_service.query_expansion_enabled
        }
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return enhanced_tfidf_quora_service.get_status()

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexDocumentsRequest):
    """Index documents with enhanced TF-IDF parameters for Quora"""
    try:
        result = await enhanced_tfidf_quora_service.index_documents(request)
        return result
    except Exception as e:
        logger.error(f"Error indexing Quora documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced Quora indexing error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Enhanced search with query expansion and reranking for Quora"""
    try:
        result = await enhanced_tfidf_quora_service.search(request)
        return result
    except Exception as e:
        logger.error(f"Error in enhanced Quora search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced Quora search error: {str(e)}")

@app.get("/collection_stats")
async def get_collection_stats():
    """Get detailed collection statistics for Quora"""
    if not enhanced_tfidf_quora_service.is_trained:
        raise HTTPException(status_code=503, detail="Quora model not trained")
    
    return {
        "collection_stats": enhanced_tfidf_quora_service.collection_stats,
        "vocabulary_sample": list(enhanced_tfidf_quora_service.vectorizer.vocabulary_.keys())[:20],
        "top_idf_terms": sorted(enhanced_tfidf_quora_service.idf_values.items(), 
                              key=lambda x: x[1], reverse=True)[:20],
        "query_expansion_terms": len(enhanced_tfidf_quora_service.term_similarities),
        "dataset": "Quora",
        "model_info": {
            "tfidf_params": enhanced_tfidf_quora_service._get_enhanced_vectorizer_params(),
            "lsa_components": enhanced_tfidf_quora_service.lsa_model.n_components if enhanced_tfidf_quora_service.lsa_model else 0,
            "matrix_density": enhanced_tfidf_quora_service.collection_stats.get("tfidf_matrix_density", 0)
        }
    }

if __name__ == "__main__":
    # Enhanced TF-IDF Quora Service runs on port 8008
    uvicorn.run(app, host="0.0.0.0", port=8008)
