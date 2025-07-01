"""
Embedding Representation Service
Provides semantic document representation using all-MiniLM-L6-v2 with inverted index
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import faiss
import os
import uvicorn
import torch
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001"
DATABASE_SERVICE_URL = "http://localhost:8004"  # Database service for cleaned documents
EMBEDDING_MODEL_PATH = "/tmp/embedding_model"
EMBEDDING_VECTORS_PATH = "/tmp/embedding_vectors.joblib"

# Pre-trained Antique model paths (update these paths after training)
ANTIQUE_MODEL_PATH = "/tmp/antique_embedding_model/"
ANTIQUE_EMBEDDINGS_PATH = "/tmp/antique_embeddings_matrix.joblib"
ANTIQUE_FAISS_PATH = "/tmp/antique_faiss_index.faiss"
ANTIQUE_INVERTED_INDEX_PATH = "/tmp/antique_inverted_index.joblib"
ANTIQUE_METADATA_PATH = "/tmp/antique_embedding_document_metadata.joblib"
USE_PRETRAINED_ANTIQUE = True  # Set to True to use pre-trained Antique model
USE_DATABASE_CLEANED_DOCS = True  # Set to True to load cleaned docs from database

# Request/Response Models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class IndexDocumentsRequest(BaseModel):
    documents: List[Document]
    dataset_name: Optional[str] = "custom"  # For database storage
    use_cleaned_from_db: Optional[bool] = False  # Load cleaned docs from database

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_inverted_index: bool = True  # Use inverted index for term filtering
    semantic_weight: float = 0.7     # Weight for semantic similarity
    term_weight: float = 0.3         # Weight for term matching

class SearchResult(BaseModel):
    document_id: str
    score: float
    semantic_score: float
    term_score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    search_method: str

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int
    embedding_dimension: int
    processing_time: float

class EmbeddingService:
    """Embedding-based document representation and search service"""
    
    def __init__(self):
        self.model = None
        self.embeddings_matrix = None
        self.faiss_index = None
        self.inverted_index = {}
        self.documents = {}  # document_id -> Document
        self.document_order = []  # To maintain order for matrix indexing
        self.is_trained = False
        self.using_pretrained = False
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # HTTP client for calling text cleaning service
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Try to load pre-trained Antique model first, then fallback to regular model
        if USE_PRETRAINED_ANTIQUE:
            self._load_pretrained_antique_model()
        if not self.is_trained:
            self._load_model()
    
    async def clean_text(self, text: str, for_query: bool = False) -> str:
        """Clean text using the shared text cleaning service"""
        try:
            response = await self.http_client.post(
                f"{TEXT_CLEANING_SERVICE_URL}/clean",
                json={
                    "text": text,
                    "remove_stopwords": not for_query,  # Keep stopwords for queries
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
    
    async def load_cleaned_documents_from_database(self, dataset_name: str) -> Tuple[List[Dict], List[str]]:
        """Load cleaned documents from database service"""
        try:
            logger.info(f"Loading cleaned documents from database for dataset: {dataset_name}")
            response = await self.http_client.get(
                f"{DATABASE_SERVICE_URL}/documents/{dataset_name}"
            )
            response.raise_for_status()
            result = response.json()
            
            documents = []
            cleaned_texts = []
            
            for doc_data in result["documents"]:
                documents.append({
                    'doc_id': doc_data['doc_id'],
                    'text': doc_data['text']  # Original text for metadata
                })
                # Use cleaned text if available, otherwise clean on-the-fly
                if 'cleaned_text' in doc_data and doc_data['cleaned_text'].strip():
                    cleaned_texts.append(doc_data['cleaned_text'])
                else:
                    # Fallback to cleaning if not available
                    cleaned_text = await self.clean_text(doc_data['text'], for_query=False)
                    cleaned_texts.append(cleaned_text)
            
            logger.info(f"Loaded {len(documents)} cleaned documents from database")
            return documents, cleaned_texts
            
        except httpx.RequestError as e:
            logger.error(f"Error connecting to database service: {e}")
            raise ValueError(f"Cannot load cleaned documents from database: {e}")
        except Exception as e:
            logger.error(f"Error loading cleaned documents: {e}")
            raise ValueError(f"Error processing cleaned documents: {e}")
    
    def _basic_fallback_clean(self, text: str) -> str:
        """Basic fallback text cleaning if service is unavailable"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    async def index_documents(self, documents: List[Document], dataset_name: str = "custom", 
                             use_cleaned_from_db: bool = False) -> IndexResponse:
        """Index documents using embeddings"""
        start_time = asyncio.get_event_loop().time()
        
        if not documents and not use_cleaned_from_db:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"Starting to index documents for dataset: {dataset_name}")
        logger.info(f"Use cleaned from database: {use_cleaned_from_db}")
        
        # Initialize model if not loaded
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Load documents and cleaned texts
        if use_cleaned_from_db:
            # Load cleaned documents from database
            logger.info("Loading cleaned documents from database...")
            doc_list, cleaned_texts = await self.load_cleaned_documents_from_database(dataset_name)
            
            # Convert to Document objects and build mappings
            documents = [Document(id=doc['doc_id'], text=doc['text']) for doc in doc_list]
            for doc in documents:
                self.documents[doc.id] = doc
            self.document_order = [doc.id for doc in documents]
            
            logger.info(f"Loaded {len(documents)} cleaned documents from database")
        else:
            # Clean documents on-the-fly
            logger.info("Cleaning documents on-the-fly...")
            cleaned_texts = []
            for doc in documents:
                cleaned_text = await self.clean_text(doc.text, for_query=False)
                cleaned_texts.append(cleaned_text)
                self.documents[doc.id] = doc
            
            # Build document order list
            self.document_order = [doc.id for doc in documents]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        self.embeddings_matrix = np.vstack(all_embeddings)
        
        # Create FAISS index
        self._create_faiss_index()
        
        # Build inverted index
        self._build_inverted_index(cleaned_texts)
        
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Indexed {len(documents)} documents in {processing_time:.2f}s")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
        return IndexResponse(
            message="Documents indexed successfully",
            documents_indexed=len(documents),
            embedding_dimension=self.embedding_dimension,
            processing_time=processing_time
        )
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search"""
        if self.embeddings_matrix is not None:
            # Ensure embeddings are float32
            embeddings_float32 = self.embeddings_matrix.astype(np.float32)
            
            # Create FAISS index for cosine similarity (using IP since embeddings are normalized)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            self.faiss_index.add(embeddings_float32)
            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
    
    def _build_inverted_index(self, cleaned_texts: List[str]):
        """Build inverted index for term-based filtering"""
        self.inverted_index = defaultdict(list)
        
        for doc_idx, cleaned_text in enumerate(cleaned_texts):
            tokens = set(cleaned_text.split())  # Use set to avoid duplicates
            for token in tokens:
                if token.strip():
                    self.inverted_index[token].append(doc_idx)
        
        # Convert to regular dict and sort document lists
        self.inverted_index = {term: sorted(doc_list) for term, doc_list in self.inverted_index.items()}
        logger.info(f"Inverted index built with {len(self.inverted_index)} terms")
    
    def _get_term_filtered_candidates(self, query_tokens: List[str], max_candidates: int = 1000) -> Set[int]:
        """Get candidate documents using inverted index"""
        if not self.inverted_index or not query_tokens:
            return set(range(len(self.document_order)))  # Return all if no filtering
        
        # Get documents that contain any query terms
        candidates = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidates.update(self.inverted_index[token])
        
        # If too few candidates, add more documents
        if len(candidates) < max_candidates // 2:
            all_indices = set(range(len(self.document_order)))
            remaining = all_indices - candidates
            candidates.update(list(remaining)[:max_candidates - len(candidates)])
        
        return candidates
    
    async def search(self, query: str, top_k: int = 10, use_inverted_index: bool = True,
                    semantic_weight: float = 0.7, term_weight: float = 0.3) -> SearchResponse:
        """Search documents using hybrid semantic + term matching"""
        start_time = asyncio.get_event_loop().time()
        
        if not self.is_trained:
            raise ValueError("Embedding model not trained. Please index documents first.")
        
        # Clean query
        cleaned_query = await self.clean_text(query, for_query=True)
        query_tokens = cleaned_query.split()
        
        # Generate query embedding
        with torch.no_grad():
            query_embedding = self.model.encode(
                [cleaned_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
        
        # Get candidate documents using inverted index
        if use_inverted_index:
            candidates = self._get_term_filtered_candidates(query_tokens)
            search_method = "hybrid_semantic_term"
        else:
            candidates = set(range(len(self.document_order)))
            search_method = "semantic_only"
        
        # Calculate semantic similarities for candidates
        if candidates:
            candidate_list = sorted(list(candidates))
            candidate_embeddings = self.embeddings_matrix[candidate_list]
            
            # Use FAISS for fast similarity computation
            candidate_embeddings_float32 = candidate_embeddings.astype(np.float32)
            temp_index = faiss.IndexFlatIP(self.embedding_dimension)
            temp_index.add(candidate_embeddings_float32)
            
            semantic_scores, _ = temp_index.search(query_embedding, len(candidate_list))
            semantic_scores = semantic_scores.flatten()
        else:
            candidate_list = []
            semantic_scores = np.array([])
        
        # Calculate term matching scores for candidates
        term_scores = []
        if use_inverted_index and query_tokens:
            for idx in candidate_list:
                doc_id = self.document_order[idx]
                doc = self.documents[doc_id]
                doc_cleaned = await self.clean_text(doc.text, for_query=False)
                doc_tokens = set(doc_cleaned.split())
                
                # Calculate term overlap
                query_token_set = set(query_tokens)
                overlap = len(query_token_set.intersection(doc_tokens))
                term_score = overlap / len(query_token_set) if query_token_set else 0
                term_scores.append(term_score)
        else:
            term_scores = [0.0] * len(candidate_list)
        
        # Combine scores
        combined_scores = []
        for i, idx in enumerate(candidate_list):
            semantic_score = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0
            term_score = term_scores[i] if i < len(term_scores) else 0.0
            
            # Normalize semantic score to [0, 1] range
            semantic_score = max(0, semantic_score)
            
            # Combined score
            combined_score = semantic_weight * semantic_score + term_weight * term_score
            
            combined_scores.append({
                'idx': idx,
                'semantic_score': semantic_score,
                'term_score': term_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score and get top-k
        combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = combined_scores[:top_k]
        
        # Build results
        results = []
        for result in top_results:
            idx = result['idx']
            doc_id = self.document_order[idx]
            doc = self.documents[doc_id]
            
            results.append(SearchResult(
                document_id=doc_id,
                score=result['combined_score'],
                semantic_score=result['semantic_score'],
                term_score=result['term_score'],
                text=doc.text,
                metadata=doc.metadata or {}
            ))
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Search completed in {processing_time:.2f}s, found {len(results)} results")
        logger.info(f"Candidates filtered: {len(candidates)} out of {len(self.document_order)}")
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            search_method=search_method
        )
    
    def _save_model(self):
        """Save embedding model and data to disk"""
        try:
            # Save model components
            joblib.dump({
                'embeddings_matrix': self.embeddings_matrix,
                'inverted_index': self.inverted_index,
                'documents': self.documents,
                'document_order': self.document_order,
                'embedding_dimension': self.embedding_dimension
            }, EMBEDDING_VECTORS_PATH)
            
            # Save SentenceTransformer model
            if self.model:
                self.model.save(EMBEDDING_MODEL_PATH)
            
            # Save FAISS index
            if self.faiss_index:
                faiss.write_index(self.faiss_index, f"{EMBEDDING_MODEL_PATH}.faiss")
            
            logger.info("Embedding model saved successfully")
        except Exception as e:
            logger.error(f"Error saving embedding model: {e}")
    
    def _load_pretrained_antique_model(self):
        """Load pre-trained Antique embedding model"""
        try:
            if (os.path.exists(ANTIQUE_MODEL_PATH) and 
                os.path.exists(ANTIQUE_EMBEDDINGS_PATH) and 
                os.path.exists(ANTIQUE_FAISS_PATH) and
                os.path.exists(ANTIQUE_INVERTED_INDEX_PATH) and
                os.path.exists(ANTIQUE_METADATA_PATH)):
                
                logger.info("Loading pre-trained Antique embedding model...")
                
                # Load SentenceTransformer model
                self.model = SentenceTransformer(ANTIQUE_MODEL_PATH)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = self.model.to(device)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                
                # Load embeddings matrix
                self.embeddings_matrix = joblib.load(ANTIQUE_EMBEDDINGS_PATH)
                
                # Load FAISS index
                self.faiss_index = faiss.read_index(ANTIQUE_FAISS_PATH)
                
                # Load inverted index
                self.inverted_index = joblib.load(ANTIQUE_INVERTED_INDEX_PATH)
                
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
                logger.info(f"Embedding dimension: {self.embedding_dimension}")
                logger.info(f"FAISS index size: {self.faiss_index.ntotal:,}")
                logger.info(f"Inverted index terms: {len(self.inverted_index):,}")
                
        except Exception as e:
            logger.error(f"Error loading pre-trained Antique model: {e}")
            logger.info("Falling back to regular model loading...")
    
    def _load_model(self):
        """Load embedding model from disk"""
        try:
            if os.path.exists(EMBEDDING_VECTORS_PATH):
                # Load model data
                data = joblib.load(EMBEDDING_VECTORS_PATH)
                self.embeddings_matrix = data['embeddings_matrix']
                self.inverted_index = data['inverted_index']
                self.documents = data['documents']
                self.document_order = data['document_order']
                self.embedding_dimension = data['embedding_dimension']
                
                # Load SentenceTransformer model
                if os.path.exists(EMBEDDING_MODEL_PATH):
                    self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.model = self.model.to(device)
                
                # Load FAISS index
                faiss_path = f"{EMBEDDING_MODEL_PATH}.faiss"
                if os.path.exists(faiss_path):
                    self.faiss_index = faiss.read_index(faiss_path)
                
                self.is_trained = True
                logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "is_trained": self.is_trained,
            "using_pretrained": self.using_pretrained,
            "model_type": "pre-trained Antique" if self.using_pretrained else "custom",
            "documents_count": len(self.documents),
            "embedding_dimension": self.embedding_dimension,
            "inverted_index_terms": len(self.inverted_index),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "text_cleaning_service": TEXT_CLEANING_SERVICE_URL,
            "embeddings_shape": self.embeddings_matrix.shape if self.embeddings_matrix is not None else None
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app for the embedding service
app = FastAPI(
    title="Embedding Representation Service",
    description="Semantic document representation using all-MiniLM-L6-v2 with inverted index",
    version="1.0.0"
)

# Global service instance
embedding_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global embedding_service
    embedding_service = EmbeddingService()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await embedding_service.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Embedding Representation Service",
        "version": "1.0.0",
        "description": "Semantic document representation using all-MiniLM-L6-v2",
        "model": "all-MiniLM-L6-v2",
        "features": [
            "Semantic embeddings",
            "Inverted index filtering",
            "Hybrid search (semantic + term)",
            "FAISS fast search",
            "Preprocessing service integration"
        ],
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
        "service": "embedding_service",
        "is_trained": embedding_service.is_trained,
        "documents_count": len(embedding_service.documents),
        "model_loaded": embedding_service.model is not None
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return embedding_service.get_status()

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexDocumentsRequest):
    """Index documents for embedding representation"""
    try:
        result = await embedding_service.index_documents(
            request.documents, 
            request.dataset_name, 
            request.use_cleaned_from_db
        )
        return result
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using hybrid semantic + term matching"""
    try:
        result = await embedding_service.search(
            request.query, 
            request.top_k,
            request.use_inverted_index,
            request.semantic_weight,
            request.term_weight
        )
        return result
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8003
    uvicorn.run(app, host="0.0.0.0", port=8003)
