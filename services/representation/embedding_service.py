"""
Embedding Representation Service
Provides pure semantic document representation using all-MiniLM-L6-v2
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

# Model base path
MODEL_BASE_PATH = "/Users/raafatmhanna/Desktop/custom-search-engine/models"

# Pre-trained Antique model paths
ANTIQUE_MODEL_PATH = f"{MODEL_BASE_PATH}/antique_embedding_model/"
ANTIQUE_EMBEDDINGS_PATH = f"{MODEL_BASE_PATH}/antique_embeddings_matrix.joblib"
ANTIQUE_FAISS_PATH = f"{MODEL_BASE_PATH}/antique_faiss_index.faiss"
# Removed inverted index - not needed for pure embeddings
ANTIQUE_METADATA_PATH = f"{MODEL_BASE_PATH}/antique_embedding_document_metadata.joblib"
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
        
        # No inverted index needed for pure embeddings
        
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
        """Create optimized FAISS index for fast similarity search"""
        if self.embeddings_matrix is not None:
            # Ensure embeddings are float32
            embeddings_float32 = self.embeddings_matrix.astype(np.float32)
            
            # Choose index type based on dataset size
            num_vectors = embeddings_float32.shape[0]
            
            if num_vectors < 1000:
                # For small datasets, use flat index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
                logger.info(f"Using FAISS flat index for {num_vectors} vectors")
            elif num_vectors < 100000:
                # For medium datasets, use HNSW for better speed/accuracy tradeoff
                self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 50
                logger.info(f"Using FAISS HNSW index for {num_vectors} vectors")
            else:
                # For large datasets, use IVF index
                nlist = min(int(np.sqrt(num_vectors)), 1000)
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
                # Train the index
                self.faiss_index.train(embeddings_float32)
                logger.info(f"Using FAISS IVF index with {nlist} clusters for {num_vectors} vectors")
            
            # Add vectors to index
            self.faiss_index.add(embeddings_float32)
            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
    
    # Inverted index removed - pure embeddings don't need term-based filtering
    
    # Term filtering removed - pure embeddings use semantic similarity for all documents
    
    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Search documents using pure semantic similarity"""
        start_time = asyncio.get_event_loop().time()
        
        # Check if model and embeddings are available
        if self.model is None:
            raise ValueError("Embedding model not loaded. Please load a model first.")
        
        if self.embeddings_matrix is None or len(self.documents) == 0:
            raise ValueError("No embeddings or documents available. Please index documents or load a pre-trained model.")
        
        # Clean query
        cleaned_query = await self.clean_text(query, for_query=True)
        
        # Generate query embedding
        with torch.no_grad():
            query_embedding = self.model.encode(
                [cleaned_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
        
        # Use FAISS index if available, otherwise fall back to numpy cosine similarity
        if self.faiss_index is not None:
            # Use FAISS index for fast similarity search
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.document_order)))
            similarities = similarities.flatten()
            indices = indices.flatten()
        else:
            # Fall back to numpy-based cosine similarity search
            logger.info("FAISS index not available, using numpy cosine similarity")
            
            # Compute cosine similarities with all document embeddings
            query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
            embeddings_normalized = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            
            similarities = np.dot(embeddings_normalized, query_embedding_normalized.T).flatten()
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            similarities = similarities[top_k_indices]
            indices = top_k_indices
        
        # Build results
        results = []
        for i, idx in enumerate(indices):
            if similarities[i] > 0:  # Only include results with positive similarity
                doc_id = self.document_order[idx]
                doc = self.documents[doc_id]
                
                results.append(SearchResult(
                    document_id=doc_id,
                    score=float(similarities[i]),
                    text=doc.text,
                    metadata=doc.metadata or {}
                ))
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        search_method = "faiss_index" if self.faiss_index is not None else "numpy_cosine"
        logger.info(f"Search completed in {processing_time:.2f}s using {search_method}, found {len(results)} results")
        
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
            # Check minimum required files (embeddings and metadata)
            required_files = [ANTIQUE_EMBEDDINGS_PATH, ANTIQUE_METADATA_PATH]
            if not all(os.path.exists(f) for f in required_files):
                logger.info("Pre-trained Antique embeddings/metadata files not found, skipping...")
                return
                
            logger.info("Loading pre-trained Antique embedding model...")
            
            # IMPORTANT: Use the same model that was used to generate the embeddings matrix
            # According to training info, embeddings were generated with 'all-MiniLM-L6-v2'
            logger.info("Loading original SentenceTransformer model that matches embeddings matrix...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model: sentence-transformers/all-MiniLM-L6-v2 (matches embeddings matrix)")
            
            # Load embeddings matrix
            self.embeddings_matrix = joblib.load(ANTIQUE_EMBEDDINGS_PATH)
            logger.info(f"Loaded embeddings matrix with shape: {self.embeddings_matrix.shape}")
            
            # Try to load FAISS index (optional)
            try:
                if os.path.exists(ANTIQUE_FAISS_PATH):
                    self.faiss_index = faiss.read_index(ANTIQUE_FAISS_PATH)
                    logger.info(f"FAISS index loaded with {self.faiss_index.ntotal:,} vectors")
                else:
                    logger.info("FAISS index not found, will use numpy similarity search")
                    self.faiss_index = None
            except Exception as faiss_e:
                logger.warning(f"Could not load FAISS index: {faiss_e}")
                logger.info("Will use numpy similarity search instead")
                self.faiss_index = None
            
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
            
            search_method = "FAISS index" if self.faiss_index else "numpy similarity"
            logger.info(f"Pre-trained Antique model loaded successfully using {search_method}!")
            logger.info(f"Documents: {len(self.documents):,}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            if self.faiss_index:
                logger.info(f"FAISS index size: {self.faiss_index.ntotal:,}")
            else:
                logger.info(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")
                
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
            "uses_faiss_index": self.faiss_index is not None,
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
    description="Pure semantic document representation using all-MiniLM-L6-v2",
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
            "Pure semantic embeddings",
            "FAISS fast similarity search",
            "Neural text representation",
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
    """Search documents using pure semantic similarity"""
    try:
        result = await embedding_service.search(
            request.query, 
            request.top_k
        )
        return result
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8008
    uvicorn.run(app, host="0.0.0.0", port=8008)
