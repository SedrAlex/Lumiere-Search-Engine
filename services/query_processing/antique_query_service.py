#!/usr/bin/env python3
"""
Antique Query Processing Service
Processes queries for ANTIQUE using embedding models and FAISS index
"""

import asyncio
import logging
import os
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import faiss
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_BASE_PATH = "/Users/raafatmhanna/Desktop/custom-search-engine/models"
# Configuration
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001"
# Use the original model name directly since local path has DTensor issues
ANTIQUE_MODEL_NAME = f"{MODEL_BASE_PATH}/antique_embedding_model/"  # Use original model directly
ANTIQUE_FAISS_PATH = f"{MODEL_BASE_PATH}/antique_faiss_index.faiss"
ANTIQUE_METADATA_PATH = f"{MODEL_BASE_PATH}/antique_embedding_document_metadata.joblib"

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class QueryResult(BaseModel):
    document_id: str
    score: float
    text: str

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int
    processing_time: float

class AntiqueQueryService:
    """Service for querying ANTIQUE embeddings"""

    def __init__(self):
        self.model = None
        self.faiss_index = None
        self.documents = {}
        self.embeddings_matrix = None
        self.document_order = []
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.embedding_dimension = 384
        self._load_antique_model()

    async def clean_text(self, text: str) -> str:
        """Clean query text using the text cleaning service"""
        try:
            response = await self.http_client.post(
                f"{TEXT_CLEANING_SERVICE_URL}/clean",
                json={"text": text, "remove_stopwords": False, "apply_stemming": True, "apply_lemmatization": False}
            )
            response.raise_for_status()
            result = response.json()
            return result["cleaned_text"]
        except httpx.RequestError as e:
            logger.error(f"Error connecting to text cleaning service: {e}")
            return self._basic_clean(text)
        except Exception as e:
            logger.error(f"Error in text cleaning: {e}")
            return self._basic_clean(text)

    def _basic_clean(self, text: str) -> str:
        """Fallback basic cleaning if service is unavailable"""
        return text.lower()

    def _load_antique_model(self):
        """Load ANTIQUE model and optionally FAISS index"""
        try:
            logger.info("Loading ANTIQUE model...")
            
            # Load SentenceTransformer model using original model name
            self.model = SentenceTransformer(ANTIQUE_MODEL_NAME, device='cpu')  # Force CPU to avoid issues
            logger.info(f"Loaded model: {ANTIQUE_MODEL_NAME}")
            
            # Try to load FAISS index (optional)
            try:
                self.faiss_index = faiss.read_index(ANTIQUE_FAISS_PATH)
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as faiss_e:
                logger.warning(f"Could not load FAISS index: {faiss_e}")
                logger.info("Will use numpy-based similarity search instead")
                self.faiss_index = None
            
            # Load metadata and embeddings matrix
            metadata = joblib.load(ANTIQUE_METADATA_PATH)
            self.documents = {doc['doc_id']: doc for doc in metadata['documents']}
            self.document_order = metadata['document_order']
            
            # Try to load embeddings matrix for fallback search
            try:
                # Look for embeddings matrix file
                embeddings_path = f"{MODEL_BASE_PATH}/antique_embeddings_matrix.joblib"
                if os.path.exists(embeddings_path):
                    self.embeddings_matrix = joblib.load(embeddings_path)
                    logger.info(f"Loaded embeddings matrix with shape: {self.embeddings_matrix.shape}")
                else:
                    logger.warning("Embeddings matrix not found, will generate embeddings on-demand")
                    self.embeddings_matrix = None
            except Exception as emb_e:
                logger.warning(f"Could not load embeddings matrix: {emb_e}")
                self.embeddings_matrix = None
            
            logger.info(f"Loaded {len(self.documents)} documents")
            search_method = "FAISS index" if self.faiss_index else "numpy similarity"
            logger.info(f"Model loaded successfully using {search_method}!")
            
        except Exception as e:
            logger.error(f"Error loading ANTIQUE model: {e}")
            raise

    async def query(self, query_text: str, top_k: int) -> QueryResponse:
        """Process query and return ranked results"""
        start_time = asyncio.get_event_loop().time()
        
        if not self.model:
            raise ValueError("ANTIQUE model not loaded")
        
        if not self.documents:
            raise ValueError("No documents available for search")
        
        logger.info(f"Processing query: {query_text}")
        
        # Clean query text using the same service as embedding service
        cleaned_query = await self.clean_text(query_text)
        logger.info(f"Cleaned query: {cleaned_query}")
        
        # Generate query embedding
        with torch.no_grad():
            query_embedding = self.model.encode(
                [cleaned_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
        
        # Use FAISS index if available, otherwise fall back to numpy similarity
        if self.faiss_index is not None:
            # Search using FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.document_order)))
            similarities = similarities.flatten()
            indices = indices.flatten()
            search_method = "FAISS index"
        else:
            # Fall back to numpy-based similarity search
            logger.info("Using numpy-based similarity search")
            
            if self.embeddings_matrix is not None:
                # Use pre-computed embeddings matrix
                query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
                embeddings_normalized = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
                
                similarities = np.dot(embeddings_normalized, query_embedding_normalized.T).flatten()
                
                # Get top-k indices
                top_k_indices = np.argsort(similarities)[::-1][:top_k]
                similarities = similarities[top_k_indices]
                indices = top_k_indices
                search_method = "numpy similarity (pre-computed embeddings)"
            else:
                # Generate embeddings on-demand (slower but works)
                logger.warning("No pre-computed embeddings available, generating on-demand")
                
                doc_texts = [self.documents[doc_id]['text'] for doc_id in self.document_order]
                
                # Generate embeddings for all documents (this is slow but necessary)
                with torch.no_grad():
                    doc_embeddings = self.model.encode(
                        doc_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=32,
                        show_progress_bar=False
                    ).astype(np.float32)
                
                # Compute similarities
                query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
                doc_embeddings_normalized = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
                
                similarities = np.dot(doc_embeddings_normalized, query_embedding_normalized.T).flatten()
                
                # Get top-k indices
                top_k_indices = np.argsort(similarities)[::-1][:top_k]
                similarities = similarities[top_k_indices]
                indices = top_k_indices
                search_method = "numpy similarity (on-demand embeddings)"
        
        # Build results using document order to map indices to document IDs
        results = []
        for i, idx in enumerate(indices):
            if similarities[i] > 0:  # Only include results with positive similarity
                doc_id = self.document_order[idx]
                doc = self.documents[doc_id]
                
                results.append(QueryResult(
                    document_id=doc_id,
                    score=float(similarities[i]),
                    text=doc['text']
                ))
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Query processed in {processing_time:.2f}s using {search_method}, found {len(results)} results")
        
        return QueryResponse(
            query=query_text,
            results=results,
            total_results=len(results),
            processing_time=processing_time
        )

app = FastAPI()
service = AntiqueQueryService()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    return await service.query(request.query, request.top_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("antique_query_service:app", host="0.0.0.0", port=8005)
