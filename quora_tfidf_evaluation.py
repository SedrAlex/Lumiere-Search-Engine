#!/usr/bin/env python3
"""
Quora TF-IDF Evaluation FastAPI Service
A microservice for evaluating TF-IDF models with proper MAP calculation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
import re
import unicodedata
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
import inflect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quora TF-IDF Evaluation Service",
    description="Microservice for evaluating TF-IDF models with MAP calculation",
    version="1.0.0"
)

# Configuration
MODELS_PATH = "/Users/raafatmhanna/Desktop/custom-search-engine/backend/tfidf_output"
QUERIES_PATH = "/Users/raafatmhanna/Downloads/quora/queries.tsv"
QRELS_PATH = "/Users/raafatmhanna/Downloads/quora/qrels.tsv"

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text processing functions
class TextProcessor:
    def __init__(self):
        self.tokenizer = word_tokenize
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.inflect_engine = inflect.engine()

    def process(self, text):
        text = self.clean_text(text)
        text = self.number_to_words(text)
        text = self.remove_html_tags(text)
        text = self.expand_contractions(text)
        text = self.normalize_unicode(text)
        text = self.handle_negations(text)
        text = self.remove_urls(text)
        return text

    def clean_text(self, text):
        return re.sub(r'\W', ' ', text)

    def number_to_words(self, text):
        words = self.tokenizer(text)
        converted_words = []
        for word in words:
            if word.replace('.', '', 1).isdigit():
                converted_words.append(word)
            elif word.isdigit():
                converted_words.append(self.inflect_engine.number_to_words(word))
            else:
                converted_words.append(word)
        return ' '.join(converted_words)

    def remove_html_tags(self, text):
        return BeautifulSoup(text, 'lxml').get_text()

    def normalize_unicode(self, text):
        return unicodedata.normalize('NFKD', text)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def handle_negations(self, text):
        words = self.tokenizer(text)
        negated_text = []
        negate = False
        for word in words:
            if word.lower() in ['not', "n't"]:
                negate = True
            elif negate:
                negated_text.append(f'NOT_{word}')
                negate = False
            else:
                negated_text.append(word)
        return ' '.join(negated_text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


# Initialize text processor
text_processor = TextProcessor()

# Define custom tokenizer function to match the one used in training
def custom_tokenizer(text):
    """Custom tokenizer that applies our text cleaning process"""
    if not text or pd.isna(text):
        return []
    
    # Apply the same processing pipeline as in the notebook
    processed = text_processor.process(str(text))
    return processed.split()

# Global variables to store loaded data
vectorizer = None
tfidf_matrix = None
index_to_doc_id = None
doc_id_to_index = None
queries_df = None
qrels_df = None

# FastAPI Models
class QueryRequest(BaseModel):
    query_id: str

class QueryTextRequest(BaseModel):
    query_text: str
    k: int = 10

class MapRequest(BaseModel):
    k: int = 1000

class EvaluationResponse(BaseModel):
    query_id: str
    results: List[Dict[str, Any]]
    map_score: float = None
    precision_at_k: Dict[int, float] = None
    recall_at_k: Dict[int, float] = None

def load_models():
    """Load TF-IDF models and data files."""
    global vectorizer, tfidf_matrix, index_to_doc_id, doc_id_to_index, queries_df, qrels_df
    
    try:
        logger.info("Loading TF-IDF models...")
        
        # Check if model files exist
        if not os.path.exists(f"{MODELS_PATH}/tfidf_vectorizer.joblib"):
            raise FileNotFoundError(f"Vectorizer not found at {MODELS_PATH}/tfidf_vectorizer.joblib")
        
        # Load models with error handling for custom tokenizer
        try:
            vectorizer = joblib.load(f"{MODELS_PATH}/tfidf_vectorizer.joblib")
        except AttributeError as e:
            logger.error(f"Error loading vectorizer due to custom tokenizer: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed due to custom tokenizer. Please retrain the model.")
            
        tfidf_matrix = joblib.load(f"{MODELS_PATH}/tfidf_matrix.joblib")
        index_to_doc_id = joblib.load(f"{MODELS_PATH}/doc_mapping.joblib")
        
        # Create reverse mapping
        doc_id_to_index = {doc_id: idx for idx, doc_id in index_to_doc_id.items()}
        
        # Load evaluation data
        queries_df = pd.read_csv(QUERIES_PATH, sep='\t', header=None, names=['query_id', 'text'])
        qrels_df = pd.read_csv(QRELS_PATH, sep='\t', header=None, names=['query_id', 'Q0', 'doc_id', 'relevance'])
        
        logger.info(f"Models loaded successfully. Matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Queries: {len(queries_df)}, QRels: {len(qrels_df)}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def process_text(query: str) -> str:
    """Process text using the internal text processor."""
    if not query or pd.isna(query):
        return ""
    
    try:
        # Use the internal text processor
        processed = text_processor.process(str(query))
        return processed.lower().strip()
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        # Fallback: return original text with basic cleaning
        return query.lower().strip()

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "quora-tfidf-evaluation",
        "models_loaded": vectorizer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/evaluate/query", response_model=EvaluationResponse)
async def evaluate_query(request: QueryRequest):
    """Evaluate a specific query by ID."""
    if vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    query_id = request.query_id
    query_row = queries_df[queries_df['query_id'] == query_id]
    
    if query_row.empty:
        raise HTTPException(status_code=404, detail="Query ID not found")
    
    query_text = query_row['text'].values[0]
    
    # Process query using the text processing service
    processed_query = process_text(query_text)
    
    # Vectorize query
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get relevance judgments
    query_qrels = qrels_df[qrels_df['query_id'] == query_id]
    relevant_docs = set(query_qrels[query_qrels['relevance'] > 0]['doc_id'].values)
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:10]
    results = []
    
    for i in top_indices:
        if i in index_to_doc_id:
            doc_id = index_to_doc_id[i]
            results.append({
                "doc_id": doc_id,
                "score": float(similarities[i]),
                "relevant": doc_id in relevant_docs
            })
    
    return EvaluationResponse(
        query_id=query_id,
        results=results
    )

@app.post("/evaluate/text")
async def evaluate_text(request: QueryTextRequest):
    """Evaluate arbitrary query text."""
    if vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Process query using the text processing service
    processed_query = process_text(request.query_text)
    
    # Vectorize query
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:request.k]
    results = []
    
    for i in top_indices:
        if i in index_to_doc_id:
            results.append({
                "doc_id": index_to_doc_id[i],
                "score": float(similarities[i])
            })
    
    return {
        "query_text": request.query_text,
        "processed_query": processed_query,
        "results": results
    }

@app.post("/evaluate/map")
async def calculate_map(request: MapRequest = MapRequest()):
    """Calculate Mean Average Precision (MAP) for all queries with optimized processing."""
    if vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    logger.info(f"Starting MAP@{request.k} calculation for {len(queries_df)} queries...")
    start_time = datetime.now()
    
    # Process queries in batches to manage memory and provide progress updates
    batch_size = 100
    total_queries = len(queries_df)
    average_precisions = []
    queries_with_relevance = 0
    
    for batch_start in range(0, total_queries, batch_size):
        batch_end = min(batch_start + batch_size, total_queries)
        batch_queries = queries_df.iloc[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_queries-1)//batch_size + 1} (queries {batch_start+1}-{batch_end})")
        
        # Process batch queries
        processed_queries = []
        for query_text in batch_queries['text']:
            processed_query = process_text(str(query_text))
            processed_queries.append(processed_query)
        
        # Transform batch queries
        batch_query_vectors = vectorizer.transform(processed_queries)
        
        # Calculate similarities for batch
        batch_similarities = cosine_similarity(batch_query_vectors, tfidf_matrix)
        
        # Process each query in the batch
        for i, (_, query_row) in enumerate(batch_queries.iterrows()):
            query_id = query_row['query_id']
            
            # Get relevance judgments for this query
            query_qrels = qrels_df[qrels_df['query_id'] == query_id]
            
            if len(query_qrels) == 0:
                continue
            
            # Get similarity scores for this query
            similarities = batch_similarities[i]
            
            # Sort documents by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1][:request.k]
            
            # Get relevant documents
            relevant_docs = set(query_qrels[query_qrels['relevance'] > 0]['doc_id'].values)
            
            if len(relevant_docs) == 0:
                continue
            
            queries_with_relevance += 1
            
            # Calculate precision at each relevant document
            precisions = []
            num_relevant_found = 0
            
            for rank, doc_index in enumerate(sorted_indices, 1):
                if doc_index in index_to_doc_id:
                    doc_id = index_to_doc_id[doc_index]
                    
                    if doc_id in relevant_docs:
                        num_relevant_found += 1
                        precision = num_relevant_found / rank
                        precisions.append(precision)
            
            if precisions:
                average_precision = np.mean(precisions)
                average_precisions.append(average_precision)
        
        # Log progress
        elapsed = datetime.now() - start_time
        if average_precisions:
            current_map = np.mean(average_precisions)
            logger.info(f"Batch completed. Current MAP: {current_map:.4f}, Queries with relevance: {queries_with_relevance}, Elapsed: {elapsed}")
    
    map_score = np.mean(average_precisions) if average_precisions else 0.0
    total_time = datetime.now() - start_time
    
    logger.info(f"MAP@{request.k} calculation completed: {map_score:.4f} (from {queries_with_relevance} queries) in {total_time}")
    
    return {
        "map_score": float(map_score),
        "k": request.k,
        "queries_evaluated": queries_with_relevance,
        "total_queries": len(queries_df),
        "calculation_time_seconds": total_time.total_seconds()
    }

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "Quora TF-IDF Evaluation Service",
        "version": "1.0.0",
        "description": "FastAPI service for evaluating TF-IDF models with MAP calculation",
        "models_loaded": vectorizer is not None,
        "models_path": MODELS_PATH,
        "queries_path": QUERIES_PATH,
        "qrels_path": QRELS_PATH,
        "text_processing": "internal"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Quora TF-IDF Evaluation Service",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "evaluate_query": "/evaluate/query",
            "evaluate_text": "/evaluate/text",
            "calculate_map": "/evaluate/map",
            "service_info": "/info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Quora TF-IDF Evaluation Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
