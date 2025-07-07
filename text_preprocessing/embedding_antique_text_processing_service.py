#!/usr/bin/env python3
"""
ANTIQUE Text Processing Service with Embedding and Database Retrieval
A standalone microservice for processing text using the exact methods from the ANTIQUE notebook,
with added embedding generation for queries and similarity-based document retrieval from database.
This service follows SOA architecture and runs on a separate port.
Built with FastAPI for better performance and automatic API documentation.
"""

import os
import sys
import re
import pandas as pd
import logging
import sqlite3
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import aiosqlite

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiqueEmbeddingProcessor:
    """
    Enhanced text processing service using the exact methods from the ANTIQUE notebook.
    This preserves semantic information while cleaning text for embedding generation,
    and includes embedding model for query processing and database retrieval.
    """
    
    def __init__(self, db_path="data/database/documents.db"):
        """Initialize the text processor with NLTK resources and embedding model."""
        self.db_path = db_path
        self.download_nltk_resources()
        self.setup_preprocessing_tools()
        self.load_embedding_model()
        self.ensure_database_exists()
        
    def download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            
    def setup_preprocessing_tools(self):
        """Setup preprocessing tools exactly as in the notebook."""
        # Setup stopwords with exceptions for important semantic words
        self.stop_words = set(stopwords.words('english'))
        self.stop_words = self.stop_words - {
            'not', 'no', 'nor', 'against', 'up', 'down', 
            'over', 'under', 'more', 'most', 'very'
        }
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
    def load_embedding_model(self):
        """Load the SentenceTransformer model for embeddings."""
        try:
            logger.info("Loading SentenceTransformer model...")
            # Use the same model as in the ANTIQUE training
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
            
    def ensure_database_exists(self):
        """Ensure the database and required tables exist."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Create database and tables if they don't exist
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create documents table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        dataset_name TEXT NOT NULL,
                        title TEXT,
                        text TEXT,
                        processed_text TEXT,
                        tokens TEXT,
                        stemmed_tokens TEXT,
                        lemmatized_tokens TEXT,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("✅ Database setup completed")
                
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
        
    def smart_clean_text(self, text):
        """
        Smart text cleaning function from the ANTIQUE notebook.
        Preserves semantics while normalizing text for embeddings.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs with placeholder
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Normalize specific patterns
        text = re.sub(r'\\b\\d{4}\\b', ' YEAR ', text)  # Years
        text = re.sub(r'\\b\\d+\\.\\d+\\b', ' DECIMAL ', text)  # Decimals
        text = re.sub(r'\\b\\d+\\b', ' NUMBER ', text)  # Numbers
        
        # Normalize emphasis patterns
        text = re.sub(r'[!]{2,}', ' EMPHASIS ', text)
        text = re.sub(r'[?]{2,}', ' QUESTION ', text)
        
        # Keep important characters and remove isolated special characters
        text = re.sub(r'[^a-zA-Z0-9\\s\\.\\,\\;\\'\\\"\\-\\!\\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Note: We don't do tokenization/lemmatization here as the SentenceTransformer
        # model's tokenizer will handle this internally
        
        return text
        
    def process_batch(self, texts):
        """
        Process a batch of texts.
        
        Args:
            texts (list): List of texts to process
            
        Returns:
            list: List of processed texts
        """
        return [self.smart_clean_text(text) for text in texts]
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Text embedding vector
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
            
        # First clean the text
        cleaned_text = self.smart_clean_text(text)
        
        # Generate embedding
        embedding = self.embedding_model.encode([cleaned_text], normalize_embeddings=True)
        return embedding[0]
        
    async def get_documents_from_db(self, dataset_name: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents from the database.
        
        Args:
            dataset_name (str): Filter by dataset name
            limit (int): Limit number of results
            
        Returns:
            List[Dict]: List of documents
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                if dataset_name:
                    query = "SELECT doc_id, title, text, processed_text, embedding FROM documents WHERE dataset_name = ?"
                    params = [dataset_name]
                else:
                    query = "SELECT doc_id, title, text, processed_text, embedding FROM documents"
                    params = []
                    
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                    
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                
                documents = []
                for row in rows:
                    doc_id, title, text, processed_text, embedding_blob = row
                    
                    # Deserialize embedding if it exists
                    embedding = None
                    if embedding_blob:
                        try:
                            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        except:
                            embedding = None
                    
                    documents.append({
                        'doc_id': doc_id,
                        'title': title or '',
                        'text': text or '',
                        'processed_text': processed_text or '',
                        'embedding': embedding
                    })
                    
                return documents
                
        except Exception as e:
            logger.error(f"Error retrieving documents from database: {e}")
            return []
            
    async def search_similar_documents(self, query: str, top_k: int = 10, dataset_name: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        This method:
        1. Generates an embedding for the query using SentenceTransformer
        2. Retrieves all documents from the database
        3. Calculates cosine similarity between query and each document
        4. Sorts by similarity score and returns top-k results
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return (default: 10)
            dataset_name (str): Filter by dataset name
            
        Returns:
            List[Dict]: List of similar documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Get documents from database
            documents = await self.get_documents_from_db(dataset_name)
            
            if not documents:
                return []
            
            # Calculate similarities
            similarities = []
            valid_documents = []
            
            for doc in documents:
                if doc['embedding'] is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        doc['embedding'].reshape(1, -1)
                    )[0][0]
                    
                    similarities.append(similarity)
                    valid_documents.append(doc)
            
            if not similarities:
                # If no embeddings, generate them on the fly
                logger.info("No pre-computed embeddings found, computing on-the-fly...")
                for doc in documents:
                    if doc['text']:
                        doc_embedding = self.generate_embedding(doc['text'])
                        similarity = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            doc_embedding.reshape(1, -1)
                        )[0][0]
                        
                        similarities.append(similarity)
                        valid_documents.append(doc)
            
            # Sort by similarity and get top-k
            if similarities:
                sorted_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for i, idx in enumerate(sorted_indices):
                    doc = valid_documents[idx]
                    score = similarities[idx]
                    
                    results.append({
                        'rank': i + 1,
                        'doc_id': doc['doc_id'],
                        'title': doc['title'],
                        'text': doc['text'][:500] + '...' if len(doc['text']) > 500 else doc['text'],  # Truncate for display
                        'similarity_score': float(score),
                        'processed_text': doc['processed_text'][:200] + '...' if len(doc['processed_text']) > 200 else doc['processed_text']
                    })
                    
                return results
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
            
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict: Database statistics
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                # Get total documents
                await cursor.execute("SELECT COUNT(*) FROM documents")
                total_docs = (await cursor.fetchone())[0]
                
                # Get documents by dataset
                await cursor.execute("""
                    SELECT dataset_name, COUNT(*) as count 
                    FROM documents 
                    GROUP BY dataset_name
                """)
                dataset_counts = await cursor.fetchall()
                
                # Get documents with embeddings
                await cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
                docs_with_embeddings = (await cursor.fetchone())[0]
                
                return {
                    'total_documents': total_docs,
                    'documents_with_embeddings': docs_with_embeddings,
                    'datasets': dict(dataset_counts),
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}

# Define FastAPI application
app = FastAPI(
    title="ANTIQUE Text Processing Service with Embeddings",
    description="Service using ANTIQUE methods for text processing with embedding generation and database retrieval",
    version="2.0.0"
)

# Initialize the enhanced text processor
processor = AntiqueEmbeddingProcessor()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class TextData(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    dataset_name: Optional[str] = None

class EmbeddingRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "antique-text-processing-with-embeddings",
        "embedding_model_loaded": processor.embedding_model is not None
    }

@app.post("/process")
async def process_text(data: TextData):
    """
    Process text using ANTIQUE text processing methods.
    """
    if data.text:
        processed = processor.smart_clean_text(data.text)
        return {"processed_text": processed}
    elif data.texts:
        if not isinstance(data.texts, list):
            raise HTTPException(status_code=400, detail="texts must be a list")
        processed = processor.process_batch(data.texts)
        return {"processed_texts": processed}
    raise HTTPException(status_code=400, detail="Either 'text' or 'texts' field required")
class QueryData(BaseModel):
    query: str

@app.post("/process/query")
async def process_query(data: QueryData):
    """
    Process a search query specifically.
    """
    processed = processor.smart_clean_text(data.query)
    return {"processed_query": processed}

class DocumentData(BaseModel):
    document: str
    doc_id: Optional[str] = None

@app.post("/process/document")
async def process_document(data: DocumentData):
    """
    Process a document specifically.
    """
    processed = processor.smart_clean_text(data.document)
    return {"processed_document": processed, "doc_id": data.doc_id}

@app.post("/generate_embedding")
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate embedding for a given text.
    """
    try:
        embedding = processor.generate_embedding(request.text)
        return {
            "text": request.text,
            "embedding": embedding.tolist(),
            "embedding_dimension": len(embedding)
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation error: {str(e)}")

@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search for similar documents using query embeddings and cosine similarity.
    Returns top-k most similar documents from the database.
    """
    try:
        # Limit top_k to prevent abuse
        top_k = min(request.top_k, 50)
        
        # Process query and get results
        processed_query = processor.smart_clean_text(request.query)
        results = await processor.search_similar_documents(
            query=request.query,
            top_k=top_k,
            dataset_name=request.dataset_name
        )
        
        return {
            "query": request.query,
            "processed_query": processed_query,
            "results": results,
            "total_results": len(results),
            "dataset_filter": request.dataset_name
        }
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/database/stats")
async def get_database_stats():
    """
    Get database statistics including document counts and embedding status.
    """
    try:
        stats = await processor.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database stats error: {str(e)}")

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "ANTIQUE Text Processing Service with Embeddings",
        "version": "2.0.0",
        "description": "Text preprocessing service using methods from ANTIQUE notebook with embedding generation and database retrieval",
        "features": [
            "Text preprocessing using ANTIQUE methods",
            "Embedding generation with SentenceTransformers",
            "Database document retrieval",
            "Cosine similarity search",
            "Top-k ranked results"
        ],
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "database_path": processor.db_path
    }

if __name__ == '__main__':
    print("🚀 Starting ANTIQUE Text Processing Service with Embeddings and Database Retrieval...")
    print("📝 This service uses ANTIQUE text processing methods")
    print("🤖 Includes SentenceTransformer embedding generation")
    print("🗄️  Retrieves documents from SQLite database")
    print("🔍 Performs cosine similarity search with top-10 ranking")
    print("📊 Calculates similarity scores directly from database embeddings")
    print("🚫 No FAISS indexing - uses direct cosine similarity calculation")
    print("📖 API docs available at: http://localhost:5001/docs")
    print("⚡ Ready to process queries and generate embeddings!")
    uvicorn.run(app, host="0.0.0.0", port=5001)
