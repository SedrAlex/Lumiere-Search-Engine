#!/usr/bin/env python3

import os
import logging
import numpy as np
import joblib
import sqlite3
import pandas as pd
import re
import nltk
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class QuoraTextCleaner:
    """
    Advanced text cleaning class optimized for Quora question pairs
    with semantic preservation and question-specific optimizations.
    """

    def __init__(self):
        # Setup stopwords with exceptions for important question words
        self.stop_words = set(stopwords.words('english'))

        # Remove question words and semantic indicators that are crucial for Quora
        question_words = {
            'what', 'when', 'where', 'why', 'who', 'which', 'how',
            'can', 'could', 'would', 'should', 'will', 'shall',
            'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'not', 'no', 'never', 'none', 'nothing', 'neither',
            'more', 'most', 'less', 'least', 'very', 'quite',
            'much', 'many', 'few', 'some', 'any', 'all',
            'best', 'better', 'good', 'bad', 'right', 'wrong'
        }
        self.stop_words = self.stop_words - question_words

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Common contractions for question text
        self.contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "what's": "what is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "where's": "where is",
            "how's": "how is"
        }

        # Question patterns that should be normalized
        self.question_patterns = {
            r'\bhow do i\b': 'how to',
            r'\bhow can i\b': 'how to',
            r'\bhow should i\b': 'how to',
            r'\bwhat is the best way to\b': 'how to',
            r'\bwhat are the ways to\b': 'how to',
            r'\bwhat are some\b': 'what are',
            r'\bwhat are the\b': 'what are'
        }

    def smart_clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning optimized for Quora questions.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Normalize question patterns
        for pattern, replacement in self.question_patterns.items():
            text = re.sub(pattern, replacement, text)

        # Remove or normalize specific patterns
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)
        text = re.sub(r'<.*?>', ' ', text)

        # Handle numbers more intelligently for questions
        text = re.sub(r'\b(19|20)\d{2}\b', ' YEAR ', text)  # Years
        text = re.sub(r'\b\d+\.\d+\b', ' DECIMAL ', text)  # Decimals
        text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', ' ORDINAL ', text)  # Ordinals
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)  # Other numbers

        # Handle emphasis and punctuation
        text = re.sub(r'[!]{2,}', ' EMPHASIS ', text)
        text = re.sub(r'[?]{2,}', ' MULTIQUEST ', text)
        text = re.sub(r'[.]{3,}', ' ELLIPSIS ', text)

        # Remove special characters but preserve some important ones
        text = re.sub(r'[^a-zA-Z0-9\s\-_]', ' ', text)

        # Handle hyphenated words carefully (important for compound terms)
        text = re.sub(r'\b(\w+)-(\w+)\b', r'\1 \2 \1\2', text)  # Keep both forms

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def custom_tokenizer(self, text: str) -> List[str]:
        """
        Custom tokenizer optimized for Quora questions.

        Args:
            text (str): Input text

        Returns:
            list: List of processed tokens
        """
        # Clean the text first
        cleaned_text = self.smart_clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            # Skip very short tokens or stopwords
            if len(token) < 2 or token in self.stop_words:
                continue

            # Skip tokens that are just underscores or dashes
            if re.match(r'^[_\-]+$', token):
                continue

            # Lemmatize
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)

        return processed_tokens

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths and URLs
DB_PATH = '/content/drive/MyDrive/downloads/docs.tsv'
TEXT_PROCESSING_SERVICE_URL = "http://localhost:8001"
TFIDF_VECTORIZER_PATH = '/content/drive/MyDrive/quora_tfidf_models/tfidf_vectorizer.joblib'
TFIDF_MATRIX_PATH = '/content/drive/MyDrive/quora_tfidf_models/tfidf_matrix.joblib'

# Initialize text cleaner first
text_cleaner = QuoraTextCleaner()
logger.info("Text cleaner initialized.")

# Load models with proper error handling using model_loader
logger.info("Loading TF-IDF models using model_loader...")
try:
    from model_loader import load_tfidf_models
    MODEL_DIR = '/content/drive/MyDrive/quora_tfidf_models/'
    tfidf_vectorizer, tfidf_matrix, inverted_index, tfidf_doc_ids = load_tfidf_models(MODEL_DIR)
    logger.info(f"TF-IDF models loaded successfully via model_loader")
    logger.info(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    logger.info(f"TF-IDF Documents: {len(tfidf_doc_ids)}")
except Exception as e:
    logger.error(f"Error loading TF-IDF models: {e}")
    logger.info("Falling back to creating new TF-IDF vectorizer...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Create a simple fallback vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=text_cleaner.custom_tokenizer,
        lowercase=False,  # Already handled by custom tokenizer
        max_features=10000,
        stop_words=None,  # Handled by custom tokenizer
        ngram_range=(1, 2)
    )
    tfidf_matrix = None  # Will be None if we can't load the pre-trained matrix
    tfidf_doc_ids = None
    inverted_index = None
    logger.info("Created fallback TF-IDF vectorizer.")

# Initialize Sentence Transformer model for embeddings
logger.info("Loading SentenceTransformer model...")
sentence_model = SentenceTransformer('/content/drive/MyDrive/Quora_Embeddings/sentence-transformers_all-MiniLM-L6-v2')
logger.info("SentenceTransformer model loaded successfully.")

# Load document embeddings and IDs from database (not from documents_final.joblib)
logger.info("Loading document embeddings and IDs from database...")
doc_embeddings = joblib.load('/content/drive/MyDrive/Quora_Embeddings/doc_embeddings.joblib')
logger.info(f"Loaded document embeddings with shape: {doc_embeddings.shape}")

# Load documents and embeddings with proper alignment
# Use the same documents that were used to create embeddings
try:
    documents_path = '/content/drive/MyDrive/Quora_Embeddings/documents_final.joblib'
    docs_data = joblib.load(documents_path)
    embedding_doc_ids = docs_data['doc_ids']
    embedding_doc_texts = docs_data['texts']
    logger.info(f"Loaded {len(embedding_doc_ids)} documents from embeddings joblib file")
except Exception as e:
    logger.error(f"Error loading documents from embeddings joblib: {e}")
    # Fallback to database loading
    def load_documents_from_sqlite():
        """Load documents from SQLite database as fallback."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('SELECT doc_id, processed_text FROM documents WHERE processed_text IS NOT NULL ORDER BY doc_id')
            rows = cursor.fetchall()
            
            conn.close()
            
            embedding_doc_ids = [row[0] for row in rows]
            embedding_doc_texts = [row[1] for row in rows]
            
            return embedding_doc_ids, embedding_doc_texts
            
        except Exception as e:
            logger.error(f"Error loading documents from SQLite: {e}")
            raise
    
    embedding_doc_ids, embedding_doc_texts = load_documents_from_sqlite()
    logger.info(f"Loaded {len(embedding_doc_ids)} documents from database as fallback")

# Create enhanced document alignment and mapping
logger.info("Creating enhanced document alignment and mapping...")

# Create document ID to index mappings for both TF-IDF and embeddings
if tfidf_matrix is not None and tfidf_doc_ids is not None:
    # Create mapping from doc_id to TF-IDF matrix index
    tfidf_id_to_index = {doc_id: idx for idx, doc_id in enumerate(tfidf_doc_ids)}
    logger.info(f"Created TF-IDF ID to index mapping for {len(tfidf_id_to_index)} documents")
else:
    tfidf_id_to_index = {}
    logger.warning("TF-IDF matrix not available - will use embedding-only search")

# Create mapping from doc_id to embedding index
embedding_id_to_index = {doc_id: idx for idx, doc_id in enumerate(embedding_doc_ids)}
logger.info(f"Created embedding ID to index mapping for {len(embedding_id_to_index)} documents")

# Find common documents between TF-IDF and embeddings
# Handle type mismatch: TF-IDF uses strings, embeddings use integers
logger.info("Handling document ID type alignment...")

# Convert TF-IDF doc IDs to integers for comparison
tfidf_id_to_index_int = {}
for doc_id_str, idx in tfidf_id_to_index.items():
    try:
        doc_id_int = int(doc_id_str)
        tfidf_id_to_index_int[doc_id_int] = idx
    except ValueError:
        continue

# Convert embedding doc IDs to integers (they should already be integers)
embedding_id_to_index_int = {}
for doc_id, idx in embedding_id_to_index.items():
    try:
        doc_id_int = int(doc_id)
        embedding_id_to_index_int[doc_id_int] = idx
    except (ValueError, TypeError):
        continue

logger.info(f"TF-IDF integer doc IDs: {len(tfidf_id_to_index_int)}")
logger.info(f"Embedding integer doc IDs: {len(embedding_id_to_index_int)}")

common_doc_ids = set(tfidf_id_to_index_int.keys()) & set(embedding_id_to_index_int.keys())
logger.info(f"Found {len(common_doc_ids)} common documents between TF-IDF and embeddings")

# Create unified document mapping for hybrid search
if common_doc_ids:
    # Use common documents for best hybrid performance
    unified_doc_ids = list(common_doc_ids)
    unified_doc_texts = [embedding_doc_texts[embedding_id_to_index_int[doc_id]] for doc_id in unified_doc_ids]
    logger.info(f"Using {len(unified_doc_ids)} common documents for hybrid search")
else:
    # Fallback to embedding documents if no common documents found
    unified_doc_ids = embedding_doc_ids
    unified_doc_texts = embedding_doc_texts
    logger.warning("No common documents found - using embedding documents only")

# Create final mappings
final_doc_ids = unified_doc_ids
final_doc_texts = unified_doc_texts

# Create unified ID to index mappings for search functions
final_id_to_index = {doc_id: idx for idx, doc_id in enumerate(final_doc_ids)}
final_id_to_tfidf_index = {doc_id: tfidf_id_to_index_int.get(doc_id) for doc_id in final_doc_ids if doc_id in tfidf_id_to_index_int}
final_id_to_embedding_index = {doc_id: embedding_id_to_index_int.get(doc_id) for doc_id in final_doc_ids if doc_id in embedding_id_to_index_int}

logger.info(f"Final document count: {len(final_doc_ids)}")
logger.info(f"TF-IDF availability: {tfidf_matrix is not None}")
logger.info(f"Embeddings availability: {doc_embeddings is not None}")

# FastAPI app
app = FastAPI(
    title="Hybrid Quora Query Processing Service",
    description="Query processing using TF-IDF and Embeddings with Fusion",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/query")
async def process_hybrid_query(request: QueryRequest):
    try:
        query = request.query
        top_k = request.top_k * 10  # Fetch larger initial result set
        
        # Clean the query at the beginning for use in all paths
        cleaned_query = text_cleaner.smart_clean_text(query)
        
        # Process query using TF-IDF and embedding models directly
        tfidf_results = []
        embedding_results = []
        
        # Get TF-IDF results using loaded models directly
        def get_tfidf_results():
            try:
                if tfidf_matrix is None or not final_id_to_tfidf_index:
                    logger.warning("TF-IDF matrix not available")
                    return []
                
                # Use the pre-cleaned query
                logger.info(f"TF-IDF query cleaning: '{query}' -> '{cleaned_query}'")
                
                # Transform query using TF-IDF vectorizer
                tfidf_query_vector = tfidf_vectorizer.transform([cleaned_query])
                logger.info(f"TF-IDF query vector: shape={tfidf_query_vector.shape}, nnz={tfidf_query_vector.nnz}")
                
                if tfidf_query_vector.nnz == 0:
                    logger.warning("TF-IDF query vector is empty, trying original query")
                    tfidf_query_vector = tfidf_vectorizer.transform([query])
                    logger.info(f"Original query vector: nnz={tfidf_query_vector.nnz}")

                if tfidf_query_vector.nnz == 0:
                    logger.warning("Enhanced tokenization for non-zero vector generation")
                    enhanced_tokens = [word for word in cleaned_query.split() if len(word) > 2]
                    tfidf_query_vector = tfidf_vectorizer.transform([' '.join(enhanced_tokens)])
                    logger.info(f"Enhanced query vector: nnz={tfidf_query_vector.nnz}")
                
                if tfidf_query_vector.nnz == 0:
                    logger.warning("Fallback to single character tokens")
                    fallback_tokens = [word for word in query.lower().split() if len(word) >= 1]
                    tfidf_query_vector = tfidf_vectorizer.transform([' '.join(fallback_tokens)])
                    logger.info(f"Fallback query vector: nnz={tfidf_query_vector.nnz}")
                
                if tfidf_query_vector.nnz == 0:
                    logger.warning("Creating synthetic scores for embedding-only documents")
                    # Return minimal scores for all documents to ensure hybrid fusion works
                    minimal_scores = np.full(len(tfidf_doc_ids), 0.001)
                    results = []
                    for idx, doc_id_str in enumerate(tfidf_doc_ids[:top_k*2]):
                        try:
                            doc_id_int = int(doc_id_str)
                            if doc_id_int in final_id_to_index:
                                results.append({
                                    "doc_id": doc_id_int,
                                    "score": float(minimal_scores[idx]),
                                    "processed_text": final_doc_texts[final_id_to_index[doc_id_int]],
                                    "doc_index": final_id_to_index[doc_id_int]
                                })
                        except (ValueError, KeyError):
                            continue
                    return results
                
                # Calculate cosine similarity with TF-IDF matrix
                cosine_similarities = cosine_similarity(tfidf_query_vector, tfidf_matrix).flatten()
                logger.info(f"TF-IDF similarities: max={np.max(cosine_similarities):.6f}, non-zero={np.sum(cosine_similarities > 0)}")
                
                # Get indices of non-zero similarities
                non_zero_indices = np.where(cosine_similarities > 0)[0]
                if len(non_zero_indices) == 0:
                    logger.info("No TF-IDF matches found")
                    return []
                
                # Sort by similarity and get top results
                sorted_indices = non_zero_indices[np.argsort(cosine_similarities[non_zero_indices])[::-1]]
                top_indices = sorted_indices[:top_k*2]
                
                results = []
                for tfidf_idx in top_indices:
                    # Convert TF-IDF matrix index to document ID
                    doc_id_str = tfidf_doc_ids[tfidf_idx]
                    try:
                        doc_id_int = int(doc_id_str)
                        if doc_id_int in final_id_to_index:  # Only include documents that are in our unified set
                            results.append({
                                "doc_id": doc_id_int,
                                "score": float(cosine_similarities[tfidf_idx]),
                                "processed_text": final_doc_texts[final_id_to_index[doc_id_int]],
                                "doc_index": final_id_to_index[doc_id_int]
                            })
                    except (ValueError, KeyError):
                        continue
                
                logger.info(f"TF-IDF calculated {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"TF-IDF processing error: {e}")
                return []

        def get_embedding_results():
            try:
                # Use the original query for embeddings (they handle their own preprocessing)
                query_embedding = sentence_model.encode([query], normalize_embeddings=True)
                logger.info(f"Embedding query vector shape: {query_embedding.shape}")
                
                # Calculate cosine similarity with document embeddings
                cosine_similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
                logger.info(f"Embedding similarities: max={np.max(cosine_similarities):.6f}, non-zero={np.sum(cosine_similarities > 0)}")
                
                # Validate that we have meaningful similarity scores
                if np.max(cosine_similarities) == 0:
                    logger.warning("All embedding similarities are zero - using minimal scores")
                    # Create minimal similarity scores to ensure hybrid fusion works
                    cosine_similarities = np.full_like(cosine_similarities, 0.001)
                    logger.info("Created minimal similarity scores for hybrid fusion")
                
                # Get indices of highest similarities
                sorted_indices = np.argsort(cosine_similarities)[::-1]
                top_indices = sorted_indices[:top_k*2]
                
                results = []
                for embedding_idx in top_indices:
                    if cosine_similarities[embedding_idx] > 0:
                        # Convert embedding matrix index to document ID
                        doc_id = embedding_doc_ids[embedding_idx]
                        doc_id_int = int(doc_id) if not isinstance(doc_id, int) else doc_id
                        
                        if doc_id_int in final_id_to_index:  # Only include documents that are in our unified set
                            results.append({
                                "doc_id": doc_id_int,
                                "similarity_score": float(cosine_similarities[embedding_idx]),
                                "document": final_doc_texts[final_id_to_index[doc_id_int]],
                                "doc_index": final_id_to_index[doc_id_int]
                            })
                
                logger.info(f"Embeddings calculated {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"Embedding processing error: {e}")
                return []

        logger.info("Starting parallel execution of TF-IDF and embedding search")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tfidf_future = executor.submit(get_tfidf_results)
            embedding_future = executor.submit(get_embedding_results)

            tfidf_results = tfidf_future.result()
            embedding_results = embedding_future.result()
        logger.info(f"Parallel execution completed. TF-IDF: {len(tfidf_results)} results, Embedding: {len(embedding_results)} results")
        # If both services failed, return error
        if not tfidf_results and not embedding_results:
            raise HTTPException(status_code=500, detail="Both TF-IDF and Embedding services are unavailable")
        
        # If only one service is available, return those results
        if not tfidf_results:
            logger.info("Using embedding-only results")
            results = []
            for i, result in enumerate(embedding_results[:top_k]):
                embedding_score = float(result["similarity_score"])
                if embedding_score > 0.1:  # Filter out low-relevance results
                    results.append({
                        "rank": i + 1,
                        "doc_id": str(result["doc_id"]),
                        "document": result["document"],
                        "hybrid_score": embedding_score,
                        "tfidf_score": 0.0,
                        "embedding_score": embedding_score,
                        "doc_index": int(result.get("doc_index", 0))
                    })
            cleaned_query = text_cleaner.smart_clean_text(query)
            return {
                "query": query,
                "cleaned_query": cleaned_query,
                "results": results,
                "total_results": len(results),
                "fusion_method": "embedding_only"
            }
        
        if not embedding_results:
            logger.info("Using TF-IDF-only results")
            results = []
            for i, result in enumerate(tfidf_results[:top_k]):
                tfidf_score = float(result["score"])
                if tfidf_score > 0.01:  # Filter out low-relevance results
                    results.append({
                        "rank": i + 1,
                        "doc_id": str(result["doc_id"]),
                        "document": result.get("processed_text", ""),
                        "hybrid_score": tfidf_score,
                        "tfidf_score": tfidf_score,
                        "embedding_score": 0.0,
                        "doc_index": int(result.get("doc_index", 0))
                    })
            cleaned_query = text_cleaner.smart_clean_text(query)
            return {
                "query": query,
                "cleaned_query": cleaned_query,
                "results": results,
                "total_results": len(results),
                "fusion_method": "tfidf_only"
            }
        
        # Enhanced Hybrid fusion using Weighted Reciprocal Rank Fusion (RRF)
        logger.info("Performing Weighted Reciprocal Rank Fusion (RRF) of TF-IDF and Embedding results")
        
        # Create document score maps
        tfidf_doc_scores = {result["doc_id"]: result["score"] for result in tfidf_results}
        embedding_doc_scores = {result["doc_id"]: result["similarity_score"] for result in embedding_results}
        
        # Create document info maps
        tfidf_doc_info = {result["doc_id"]: result for result in tfidf_results}
        embedding_doc_info = {result["doc_id"]: result for result in embedding_results}
        
        # Get all unique document IDs
        all_doc_ids = set(tfidf_doc_scores.keys()) | set(embedding_doc_scores.keys())
        
        # Normalize scores to [0, 1] range
        if tfidf_doc_scores:
            tfidf_max = max(tfidf_doc_scores.values())
            tfidf_min = min(tfidf_doc_scores.values())
            tfidf_range = tfidf_max - tfidf_min if tfidf_max != tfidf_min else 1
            normalized_tfidf = {doc_id: (score - tfidf_min) / tfidf_range for doc_id, score in tfidf_doc_scores.items()}
        else:
            normalized_tfidf = {}
            
        if embedding_doc_scores:
            embedding_max = max(embedding_doc_scores.values())
            embedding_min = min(embedding_doc_scores.values())
            embedding_range = embedding_max - embedding_min if embedding_max != embedding_min else 1
            normalized_embedding = {doc_id: (score - embedding_min) / embedding_range for doc_id, score in embedding_doc_scores.items()}
        else:
            normalized_embedding = {}
        
        # Create rank-based dictionaries for RRF
        tfidf_ranks = {}
        embedding_ranks = {}
        
        # Create TF-IDF ranks
        tfidf_sorted = sorted(tfidf_doc_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(tfidf_sorted, 1):
            tfidf_ranks[doc_id] = rank
            
        # Create Embedding ranks  
        embedding_sorted = sorted(embedding_doc_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(embedding_sorted, 1):
            embedding_ranks[doc_id] = rank
            
        # Weighted RRF parameters
        k = 60  # RRF constant, commonly used value
        embedding_weight = 0.75  # 75% weight to embeddings (higher semantic relevance)
        tfidf_weight = 0.25      # 25% weight to TF-IDF (keyword matching)
        
        logger.info(f"Using weighted RRF: embedding_weight={embedding_weight}, tfidf_weight={tfidf_weight}")
        
        # Calculate Weighted RRF scores
        fusion_scores = {}
        for doc_id in all_doc_ids:
            tfidf_rank = tfidf_ranks.get(doc_id, len(tfidf_doc_scores) + 1)
            embedding_rank = embedding_ranks.get(doc_id, len(embedding_doc_scores) + 1)
            
            # Weighted RRF formula: weighted sum of 1/(k + rank) for each ranking system
            weighted_rrf_score = (embedding_weight / (k + embedding_rank)) + (tfidf_weight / (k + tfidf_rank))
            
            # Boost documents that appear in both methods
            boost_factor = 1.0
            if doc_id in tfidf_doc_scores and doc_id in embedding_doc_scores:
                boost_factor = 1.3  # 30% boost for documents found by both methods
            
            fusion_scores[doc_id] = weighted_rrf_score * boost_factor
            
        logger.info(f"RRF fusion calculated for {len(fusion_scores)} documents")
        
        # Sort documents by fusion score
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build final results
        results = []
        for i, (doc_id, fusion_score) in enumerate(sorted_docs):
            # Get document info from either service
            doc_info = tfidf_doc_info.get(doc_id) or embedding_doc_info.get(doc_id)
            
            if doc_info:
                # Get the document text
                if doc_id in tfidf_doc_info:
                    document_text = tfidf_doc_info[doc_id].get("processed_text", "")
                else:
                    document_text = embedding_doc_info[doc_id].get("document", "")
                
                # Get the original scores (not normalized)
                original_tfidf_score = tfidf_doc_scores.get(doc_id, 0.0)
                original_embedding_score = embedding_doc_scores.get(doc_id, 0.0)
                
                results.append({
                    "rank": i + 1,
                    "doc_id": str(doc_id),
                    "document": document_text,
                    "hybrid_score": float(fusion_score),
                    "tfidf_score": float(original_tfidf_score),
                    "embedding_score": float(original_embedding_score),
                    "doc_index": int(doc_info.get("doc_index", 0))
                })
        
        return {
            "query": query,
            "cleaned_query": cleaned_query,
            "results": results,
            "total_results": len(results),
            "fusion_method": "reciprocal_rank_fusion",
            "tfidf_results_count": len(tfidf_results),
            "embedding_results_count": len(embedding_results)
        }

    except Exception as e:
        logger.error(f"Error processing hybrid query: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid query processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

