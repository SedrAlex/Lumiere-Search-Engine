"""
TF-IDF Representation Service
Provides TF-IDF document representation and search functionality
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001"
TFIDF_MODEL_PATH = "/tmp/tfidf_model.joblib"
TFIDF_VECTORS_PATH = "/tmp/tfidf_vectors.joblib"

# Pre-trained Antique model paths (update these paths after training)
ANTIQUE_MODEL_PATH = "/tmp/antique_enhanced_tfidf_vectorizer.joblib"
ANTIQUE_MATRIX_PATH = "/tmp/antique_enhanced_tfidf_matrix.joblib"
ANTIQUE_METADATA_PATH = "/tmp/antique_enhanced_document_metadata.joblib"
USE_PRETRAINED_ANTIQUE = True  # Set to True to use pre-trained Antique model

# Request/Response Models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class IndexDocumentsRequest(BaseModel):
    documents: List[Document]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    vocabulary_size: int
    processing_time: float

class TFIDFService:
    """TF-IDF document representation and search service"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = {}  # document_id -> Document
        self.document_order = []  # To maintain order for matrix indexing
        self.is_trained = False
        self.using_pretrained = False
        
        # HTTP client for calling text cleaning service
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Try to load pre-trained Antique model first, then fallback to regular model
        if USE_PRETRAINED_ANTIQUE:
            self._load_pretrained_antique_model()
        if not self.is_trained:
            self._load_model()
    
    async def clean_text(self, text: str) -> str:
        """Clean text using the shared text cleaning service"""
        try:
            response = await self.http_client.post(
                f"{TEXT_CLEANING_SERVICE_URL}/clean",
                json={
                    "text": text,
                    "remove_stopwords": True,
                    "apply_stemming": True,
                    "apply_lemmatization": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["cleaned_text"]
        except httpx.RequestError as e:
            logger.error(f"Error connecting to text cleaning service: {e}")
            # Fallback to basic cleaning
            return self._basic_fallback_clean(text)
        except Exception as e:
            logger.error(f"Error in text cleaning: {e}")
            return self._basic_fallback_clean(text)
    
    def _basic_fallback_clean(self, text: str) -> str:
        """Basic fallback text cleaning if service is unavailable"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    async def index_documents(self, documents: List[Document]) -> IndexResponse:
        """Index documents using TF-IDF"""
        start_time = asyncio.get_event_loop().time()
        
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"Starting to index {len(documents)} documents")
        
        # Clean all document texts
        cleaned_texts = []
        for doc in documents:
            cleaned_text = await self.clean_text(doc.text)
            cleaned_texts.append(cleaned_text)
            self.documents[doc.id] = doc
        
        # Build document order list
        self.document_order = [doc.id for doc in documents]
        
        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            sublinear_tf=True  # Apply sublinear TF scaling
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        processing_time = asyncio.get_event_loop().time() - start_time
        vocabulary_size = len(self.vectorizer.vocabulary_)
        
        logger.info(f"Indexed {len(documents)} documents in {processing_time:.2f}s")
        logger.info(f"Vocabulary size: {vocabulary_size}")
        
        return IndexResponse(
            message="Documents indexed successfully",
            documents_indexed=len(documents),
            vocabulary_size=vocabulary_size,
            processing_time=processing_time
        )
    
    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Search documents using TF-IDF similarity"""
        start_time = asyncio.get_event_loop().time()
        
        if not self.is_trained:
            raise ValueError("TF-IDF model not trained. Please index documents first.")
        
        # Clean query using text cleaning service
        cleaned_query = await self.clean_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include results with positive similarity
                doc_id = self.document_order[idx]
                doc = self.documents[doc_id]
                
                results.append(SearchResult(
                    document_id=doc_id,
                    score=float(similarities[idx]),
                    text=doc.text,
                    metadata=doc.metadata or {}
                ))
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Search completed in {processing_time:.2f}s, found {len(results)} results")
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=processing_time
        )
    
    def _save_model(self):
        """Save TF-IDF model and vectors to disk using joblib"""
        try:
            # Save model components
            joblib.dump({
                'vectorizer': self.vectorizer,
                'documents': self.documents,
                'document_order': self.document_order
            }, TFIDF_MODEL_PATH)
            
            # Save TF-IDF matrix
            joblib.dump(self.tfidf_matrix, TFIDF_VECTORS_PATH)
            
            logger.info("TF-IDF model saved successfully with joblib")
        except Exception as e:
            logger.error(f"Error saving TF-IDF model: {e}")
    
    def _load_pretrained_antique_model(self):
        """Load pre-trained Antique TF-IDF model"""
        try:
            if os.path.exists(ANTIQUE_MODEL_PATH) and os.path.exists(ANTIQUE_MATRIX_PATH) and os.path.exists(ANTIQUE_METADATA_PATH):
                logger.info("Loading pre-trained Antique TF-IDF model...")
                
                # Load vectorizer
                self.vectorizer = joblib.load(ANTIQUE_MODEL_PATH)
                
                # Load TF-IDF matrix
                self.tfidf_matrix = joblib.load(ANTIQUE_MATRIX_PATH)
                
                # Load document metadata
                metadata = joblib.load(ANTIQUE_METADATA_PATH)
                
                # Convert documents list to dict for compatibility
                self.documents = {doc['doc_id']: Document(
                    id=doc['doc_id'],
                    text=doc['text'],
                    metadata={}
                ) for doc in metadata['documents']}
                
                self.document_order = metadata['document_order']
                self.is_trained = True
                self.using_pretrained = True
                
                logger.info(f"Pre-trained Antique model loaded successfully!")
                logger.info(f"Documents: {len(self.documents):,}")
                logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
                logger.info(f"Matrix shape: {self.tfidf_matrix.shape}")
                
        except Exception as e:
            logger.error(f"Error loading pre-trained Antique model: {e}")
            logger.info("Falling back to regular model loading...")
    
    def _load_model(self):
        """Load TF-IDF model and vectors from disk using joblib"""
        try:
            if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(TFIDF_VECTORS_PATH):
                # Load model components
                data = joblib.load(TFIDF_MODEL_PATH)
                self.vectorizer = data['vectorizer']
                self.documents = data['documents']
                self.document_order = data['document_order']
                
                # Load TF-IDF matrix
                self.tfidf_matrix = joblib.load(TFIDF_VECTORS_PATH)
                
                self.is_trained = True
                logger.info("TF-IDF model loaded successfully with joblib")
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "is_trained": self.is_trained,
            "using_pretrained": self.using_pretrained,
            "model_type": "pre-trained Antique" if self.using_pretrained else "custom",
            "documents_count": len(self.documents),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            "text_cleaning_service": TEXT_CLEANING_SERVICE_URL,
            "matrix_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app for the TF-IDF service
app = FastAPI(
    title="TF-IDF Representation Service",
    description="Document representation and search using TF-IDF",
    version="1.0.0"
)

# Global service instance
tfidf_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global tfidf_service
    tfidf_service = TFIDFService()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await tfidf_service.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TF-IDF Representation Service",
        "version": "1.0.0",
        "description": "Document representation and search using TF-IDF",
        "endpoints": {
            "POST /index": "Index documents",
            "POST /search": "Search documents",
            "GET /status": "Get service status",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tfidf_service",
        "is_trained": tfidf_service.is_trained,
        "documents_count": len(tfidf_service.documents)
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return tfidf_service.get_status()

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexDocumentsRequest):
    """Index documents for TF-IDF representation"""
    try:
        result = await tfidf_service.index_documents(request.documents)
        return result
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using TF-IDF similarity"""
    try:
        result = await tfidf_service.search(request.query, request.top_k)
        return result
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)
