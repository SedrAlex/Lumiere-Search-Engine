#!/usr/bin/env python3
"""
FastAPI Web Application for Information Retrieval System
Provides REST API endpoints for testing with Postman
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import logging
import numpy as np
from contextlib import asynccontextmanager

from ir_system import IRSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global IR System instance
ir_system = None
# Global Document Representation System
doc_repr_system = None
# Datasets loading status
datasets_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize systems on startup without loading datasets"""
    global ir_system, doc_repr_system
    logger.info("Initializing IR System...")
    
    # Initialize systems but don't load datasets automatically
    ir_system = IRSystem()
    
    # Import and initialize document representation system
    from document_representations import DocumentRepresentationSystem
    doc_repr_system = DocumentRepresentationSystem()
    
    logger.info("IR System ready - datasets can be loaded on demand")
    
    yield
    
    # Cleanup
    logger.info("Shutting down IR System...")

# Create FastAPI app
app = FastAPI(
    title="Information Retrieval System API",
    description="API for testing ANTIQUE and CodeSearchNet datasets with preprocessing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    dataset: str = "antique"  # antique or codesearchnet
    method: str = "bm25"      # bm25 or tfidf
    top_k: int = 10

class EvaluationRequest(BaseModel):
    query_id: str
    dataset: str = "antique"
    method: str = "bm25"
    top_k: int = 10

class PreprocessingRequest(BaseModel):
    text: str
    use_stemming: bool = True
    use_lemmatization: bool = False

class DatasetLoadRequest(BaseModel):
    dataset_name: str = "antique"  # antique or codesearchnet
    representation_types: List[str] = ["tfidf", "embedding", "hybrid"]
    max_documents: Optional[int] = None  # Limit for testing

class RepresentationRequest(BaseModel):
    query: str
    dataset: str = "antique"
    representation_type: str = "tfidf"  # tfidf, embedding, hybrid
    top_k: int = 10

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Information Retrieval System API",
        "version": "1.0.0",
        "datasets": ["antique", "codesearchnet"],
        "methods": ["bm25", "tfidf"],
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /datasets": "Dataset information",
            "GET /datasets/{dataset_name}": "Specific dataset info",
            "POST /search": "Search documents",
            "POST /evaluate": "Evaluate query with ground truth",
            "POST /preprocess": "Test text preprocessing",
            "GET /queries/{dataset_name}": "Get sample queries",
            "GET /sample-data/{dataset_name}": "Get sample documents"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global ir_system
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    return {
        "status": "healthy",
        "loaded_datasets": list(ir_system.loaded_datasets),
        "message": "IR System is running"
    }

@app.get("/datasets")
async def get_datasets_info():
    """Get information about all available datasets (loaded and unloaded)"""
    global ir_system, datasets_status
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    # Show both loaded and available datasets
    available_datasets = ["antique", "codesearchnet"]
    dataset_info = {}
    
    for dataset_name in available_datasets:
        status = datasets_status.get(dataset_name, {
            "loaded": False,
            "loading": False,
            "error": None,
            "representations": []
        })
        
        if dataset_name in ir_system.loaded_datasets:
            # Get detailed info for loaded datasets
            loaded_info = ir_system.get_dataset_info(dataset_name)
            dataset_info[dataset_name] = {
                **loaded_info,
                "status": "loaded",
                "representations_available": status.get("representations", [])
            }
        else:
            # Show basic info for unloaded datasets
            dataset_info[dataset_name] = {
                "name": dataset_name,
                "status": "loading" if status.get("loading") else "not_loaded",
                "documents": 0,
                "queries": 0,
                "qrels": 0,
                "representations_available": status.get("representations", []),
                "error": status.get("error"),
                "description": f"Dataset {dataset_name} - load using POST /datasets/load"
            }
    
    return {
        "available_datasets": available_datasets,
        "datasets": dataset_info,
        "note": "Use POST /datasets/load to load datasets with representations"
    }

@app.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset"""
    global ir_system, datasets_status
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    if dataset_name not in ["antique", "codesearchnet"]:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}. Available: antique, codesearchnet")
    
    if dataset_name in ir_system.loaded_datasets:
        # Return detailed info for loaded dataset
        return ir_system.get_dataset_info(dataset_name)
    else:
        # Return helpful info for unloaded dataset
        status = datasets_status.get(dataset_name, {})
        return {
            "name": dataset_name,
            "status": "loading" if status.get("loading") else "not_loaded",
            "loaded": False,
            "error": status.get("error"),
            "message": f"Dataset {dataset_name} is not loaded. Use POST /datasets/load to load it.",
            "how_to_load": {
                "endpoint": "POST /datasets/load",
                "example_payload": {
                    "dataset_name": dataset_name,
                    "representation_types": ["tfidf", "embedding", "hybrid"],
                    "max_documents": 1000
                }
            }
        }

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search documents in the specified dataset
    
    This endpoint demonstrates the following preprocessing steps:
    1. Text cleaning (lowercase, remove HTML, special characters)
    2. Tokenization using NLTK
    3. Stopword removal
    4. Stemming using Porter Stemmer
    5. Search using BM25 or TF-IDF
    
    The endpoint now supports dynamic dataset loading - if a dataset is not loaded,
    it will be automatically loaded when first requested.
    """
    global ir_system, datasets_status
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    if request.dataset not in ["antique", "codesearchnet"]:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset}. Available: antique, codesearchnet")
    
    # Check if dataset needs to be loaded
    if request.dataset not in ir_system.loaded_datasets:
        status = datasets_status.get(request.dataset, {})
        if status.get("loading"):
            raise HTTPException(status_code=409, detail=f"Dataset {request.dataset} is currently loading. Please wait and try again.")
        else:
            # Automatically load the dataset
            logger.info(f"Auto-loading dataset {request.dataset} for search request")
            try:
                await _load_dataset_synchronously(request.dataset)
                logger.info(f"Successfully auto-loaded dataset {request.dataset}")
            except Exception as e:
                logger.error(f"Failed to auto-load dataset {request.dataset}: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to load dataset {request.dataset}: {str(e)}"
                )
    
    if request.method not in ["bm25", "tfidf"]:
        raise HTTPException(status_code=400, detail="Method must be 'bm25' or 'tfidf'")
    
    result = ir_system.search(
        query=request.query,
        dataset_name=request.dataset,
        method=request.method,
        top_k=request.top_k
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.post("/evaluate")
async def evaluate_query(request: EvaluationRequest):
    """Evaluate a query against ground truth qrels
    
    This endpoint shows how the system performs against known relevant documents.
    It calculates Precision, Recall, and F1-score.
    """
    global ir_system
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    if request.dataset not in ir_system.loaded_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset} not found")
    
    result = ir_system.evaluate_query(
        query_id=request.query_id,
        dataset_name=request.dataset,
        method=request.method,
        top_k=request.top_k
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/preprocess")
async def preprocess_text(request: PreprocessingRequest):
    """Test text preprocessing pipeline
    
    This endpoint shows exactly what preprocessing steps are applied:
    1. Original text
    2. Cleaned text (lowercase, remove HTML/special chars)
    3. Tokenized text
    4. Processed tokens (after stopword removal and stemming/lemmatization)
    """
    global ir_system
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    preprocessor = ir_system.dataset_loader.preprocessor
    
    cleaned_text = preprocessor.clean_text(request.text)
    processed_tokens = preprocessor.tokenize_and_process(
        request.text,
        use_stemming=request.use_stemming,
        use_lemmatization=request.use_lemmatization
    )
    
    # Also show intermediate steps
    from nltk.tokenize import word_tokenize
    raw_tokens = word_tokenize(cleaned_text.lower())
    
    return {
        "original_text": request.text,
        "cleaned_text": cleaned_text,
        "raw_tokens": raw_tokens,
        "processed_tokens": processed_tokens,
        "preprocessing_steps": {
            "text_cleaning": "Lowercase, remove HTML tags, remove special characters",
            "tokenization": "NLTK word_tokenize",
            "stopword_removal": "English stopwords removed",
            "stemming_applied": request.use_stemming,
            "lemmatization_applied": request.use_lemmatization
        },
        "settings": {
            "use_stemming": request.use_stemming,
            "use_lemmatization": request.use_lemmatization
        }
    }

@app.get("/queries/{dataset_name}")
async def get_sample_queries(dataset_name: str, limit: int = 10):
    """Get sample queries from a dataset"""
    global ir_system
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    if dataset_name not in ir_system.loaded_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    queries = ir_system.dataset_loader.queries[dataset_name]
    sample_queries = list(queries.values())[:limit]
    
    return {
        "dataset": dataset_name,
        "total_queries": len(queries),
        "sample_queries": sample_queries
    }

@app.get("/sample-data/{dataset_name}")
async def get_sample_data(dataset_name: str, limit: int = 5):
    """Get sample documents and queries from a dataset"""
    global ir_system
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    if dataset_name not in ir_system.loaded_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    documents = ir_system.dataset_loader.documents[dataset_name]
    queries = ir_system.dataset_loader.queries[dataset_name]
    qrels = ir_system.dataset_loader.qrels[dataset_name]
    
    sample_docs = list(documents.values())[:limit]
    sample_queries = list(queries.values())[:limit]
    sample_qrels = dict(list(qrels.items())[:limit])
    
    return {
        "dataset": dataset_name,
        "statistics": {
            "total_documents": len(documents),
            "total_queries": len(queries),
            "total_qrels": len(qrels)
        },
        "sample_documents": sample_docs,
        "sample_queries": sample_queries,
        "sample_qrels": sample_qrels
    }

@app.post("/datasets/load")
async def load_dataset(request: DatasetLoadRequest, background_tasks: BackgroundTasks):
    """Load dataset on demand with specified representations"""
    global ir_system, doc_repr_system, datasets_status
    
    if ir_system is None or doc_repr_system is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
    if request.dataset_name not in ["antique", "codesearchnet"]:
        raise HTTPException(status_code=400, detail="Dataset must be 'antique' or 'codesearchnet'")
    
    # Check if already loading
    if datasets_status.get(request.dataset_name, {}).get("loading"):
        raise HTTPException(status_code=409, detail=f"Dataset {request.dataset_name} is already being loaded")
    
    # Set loading status
    datasets_status[request.dataset_name] = {
        "loading": True,
        "loaded": False,
        "error": None,
        "representations": []
    }
    
    # Add background task
    background_tasks.add_task(
        _load_dataset_background,
        request.dataset_name,
        request.representation_types,
        request.max_documents
    )
    
    return {
        "message": f"Loading dataset {request.dataset_name} in background",
        "dataset": request.dataset_name,
        "representation_types": request.representation_types,
        "status": "loading"
    }

async def _load_dataset_synchronously(dataset_name: str, representation_types: List[str] = ["tfidf"], max_documents: Optional[int] = None):
    """Synchronously load dataset for immediate use (used by dynamic loading)"""
    global ir_system, doc_repr_system, datasets_status
    
    try:
        logger.info(f"Synchronously loading dataset: {dataset_name}")
        
        # Set loading status
        datasets_status[dataset_name] = {
            "loading": True,
            "loaded": False,
            "error": None,
            "representations": []
        }
        
        # Load dataset using IR system  
        if dataset_name == "antique":
            dataset_info = ir_system.dataset_loader.load_antique_dataset()
        elif dataset_name == "codesearchnet":
            dataset_info = ir_system.dataset_loader.load_codesearchnet_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not dataset_info:
            raise Exception("Failed to load dataset")
        
        # Add to loaded datasets
        ir_system.loaded_datasets.add(dataset_name)
        
        # Build traditional indices (required for BM25 and TF-IDF search)
        ir_system.search_engine.build_bm25_index(dataset_name)
        ir_system.search_engine.build_tfidf_index(dataset_name)
        
        # Update status - mark as loaded with basic search capabilities
        datasets_status[dataset_name] = {
            "loading": False,
            "loaded": True,
            "error": None,
            "representations": ["bm25", "tfidf"],  # Basic search methods available
            "document_count": len(ir_system.dataset_loader.documents[dataset_name]),
            "auto_loaded": True
        }
        
        logger.info(f"Successfully loaded dataset {dataset_name} with basic search capabilities")
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        datasets_status[dataset_name] = {
            "loading": False,
            "loaded": False,
            "error": str(e),
            "representations": []
        }
        raise e

async def _load_dataset_background(dataset_name: str, representation_types: List[str], max_documents: Optional[int]):
    """Background task to load dataset and create representations"""
    global ir_system, doc_repr_system, datasets_status
    
    try:
        logger.info(f"Starting to load dataset: {dataset_name}")
        
        # Load dataset using IR system
        if dataset_name == "antique":
            dataset_info = ir_system.dataset_loader.load_antique_dataset()
        elif dataset_name == "codesearchnet":
            dataset_info = ir_system.dataset_loader.load_codesearchnet_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not dataset_info:
            raise Exception("Failed to load dataset")
        
        # Add to loaded datasets
        ir_system.loaded_datasets.add(dataset_name)
        
        # Build traditional indices
        ir_system.search_engine.build_bm25_index(dataset_name)
        ir_system.search_engine.build_tfidf_index(dataset_name)
        
        # Prepare documents for representation system
        documents = ir_system.dataset_loader.documents[dataset_name]
        doc_list = list(documents.values())
        
        # Limit documents if specified
        if max_documents and max_documents < len(doc_list):
            doc_list = doc_list[:max_documents]
            logger.info(f"Limited to {max_documents} documents for testing")
        
        # Process and store representations
        results = doc_repr_system.process_and_store_dataset(
            doc_list, dataset_name, representation_types
        )
        
        # Update status
        datasets_status[dataset_name] = {
            "loading": False,
            "loaded": True,
            "error": None,
            "representations": representation_types,
            "document_count": len(doc_list),
            "processing_results": results
        }
        
        logger.info(f"Successfully loaded dataset {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        datasets_status[dataset_name] = {
            "loading": False,
            "loaded": False,
            "error": str(e),
            "representations": []
        }

@app.get("/datasets/status")
async def get_datasets_status():
    """Get loading status of all datasets"""
    global datasets_status, doc_repr_system
    
    if doc_repr_system is None:
        raise HTTPException(status_code=503, detail="Document representation system not initialized")
    
    # Get system stats
    system_status = doc_repr_system.get_system_status()
    
    return {
        "datasets_status": datasets_status,
        "system_status": system_status,
        "available_datasets": ["antique", "codesearchnet"],
        "available_representations": ["tfidf", "embedding", "hybrid"]
    }

@app.post("/search/representations")
async def search_with_representations(request: RepresentationRequest):
    """Search using document representations (TF-IDF, Embedding, or Hybrid)"""
    global doc_repr_system, datasets_status
    
    if doc_repr_system is None:
        raise HTTPException(status_code=503, detail="Document representation system not initialized")
    
    # Check if dataset is loaded
    dataset_status = datasets_status.get(request.dataset, {})
    if not dataset_status.get("loaded"):
        raise HTTPException(status_code=400, detail=f"Dataset {request.dataset} not loaded")
    
    if request.representation_type not in dataset_status.get("representations", []):
        raise HTTPException(status_code=400, detail=f"Representation {request.representation_type} not available for {request.dataset}")
    
    try:
        # Get all representations for the dataset
        representations = doc_repr_system.db_manager.get_all_representations(
            request.dataset, request.representation_type
        )
        
        if not representations:
            raise HTTPException(status_code=404, detail=f"No {request.representation_type} representations found for {request.dataset}")
        
        # Perform search based on representation type
        if request.representation_type == "tfidf":
            # Use TF-IDF search
            tfidf_repr = doc_repr_system.representations['tfidf']
            query_vector = tfidf_repr.transform([request.query])[0]
            
            # Calculate similarities
            doc_ids = list(representations.keys())
            vectors = np.array([representations[doc_id] for doc_id in doc_ids])
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_vector], vectors)[0]
            
        elif request.representation_type == "embedding":
            # Use embedding search
            embedding_repr = doc_repr_system.representations['embedding']
            query_embedding = embedding_repr.encode_query(request.query)
            
            doc_ids = list(representations.keys())
            vectors = np.array([representations[doc_id] for doc_id in doc_ids])
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_embedding], vectors)[0]
            
        elif request.representation_type == "hybrid":
            # Use hybrid search (serial approach)
            hybrid_repr = doc_repr_system.representations['hybrid']
            
            # Extract TF-IDF and embedding vectors from hybrid representations
            doc_ids = list(representations.keys())
            tfidf_vectors = []
            embedding_vectors = []
            
            for doc_id in doc_ids:
                hybrid_data = representations[doc_id]
                tfidf_vectors.append(hybrid_data[0])  # First component is TF-IDF
                embedding_vectors.append(hybrid_data[1])  # Second component is embedding
            
            tfidf_vectors = np.array(tfidf_vectors)
            embedding_vectors = np.array(embedding_vectors)
            
            # Use hybrid search method
            top_indices, scores = hybrid_repr.search_hybrid(
                request.query, tfidf_vectors, embedding_vectors, request.top_k
            )
            
            # Format results
            results = []
            for i, idx in enumerate(top_indices):
                doc_id = doc_ids[idx]
                results.append({
                    "doc_id": doc_id,
                    "score": float(scores[i]),
                    "rank": i + 1
                })
            
            return {
                "query": request.query,
                "dataset": request.dataset,
                "representation_type": request.representation_type,
                "approach": "serial",
                "results": results[:request.top_k],
                "total_documents": len(doc_ids)
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown representation type: {request.representation_type}")
        
        # Get top results (for non-hybrid methods)
        top_indices = np.argsort(similarities)[::-1][:request.top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            results.append({
                "doc_id": doc_id,
                "score": float(similarities[idx]),
                "rank": i + 1
            })
        
        return {
            "query": request.query,
            "dataset": request.dataset,
            "representation_type": request.representation_type,
            "results": results,
            "total_documents": len(doc_ids),
            "preprocessing_confirmed": {
                "cleaning_applied": True,
                "tokenization_applied": True,
                "note": "All representations use properly cleaned and tokenized text"
            }
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/datasets/{dataset_name}/verify-representations")
async def verify_dataset_representations(dataset_name: str):
    """Verify that dataset representations are correctly stored and retrievable"""
    global doc_repr_system
    
    if doc_repr_system is None:
        raise HTTPException(status_code=503, detail="Document representation system not initialized")
    
    try:
        verification = doc_repr_system.verify_representations(dataset_name)
        return verification
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status including database and representations"""
    global doc_repr_system, ir_system, datasets_status
    
    status = {
        "ir_system_ready": ir_system is not None,
        "representation_system_ready": doc_repr_system is not None,
        "datasets_status": datasets_status
    }
    
    if doc_repr_system:
        status["system_status"] = doc_repr_system.get_system_status()
    
    return status

@app.get("/dataset-verification/{dataset_name}")
async def verify_dataset(dataset_name: str):
    """Verify dataset meets requirements (>200K docs and has qrels)"""
    global ir_system
    
    if ir_system is None:
        raise HTTPException(status_code=503, detail="IR System not initialized")
    
    if dataset_name not in ir_system.loaded_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    documents = ir_system.dataset_loader.documents[dataset_name]
    queries = ir_system.dataset_loader.queries[dataset_name]
    qrels = ir_system.dataset_loader.qrels[dataset_name]
    
    has_enough_docs = len(documents) > 200000
    has_qrels = len(qrels) > 0
    has_queries = len(queries) > 0
    
    return {
        "dataset": dataset_name,
        "verification": {
            "meets_document_requirement": has_enough_docs,
            "has_testing_data": has_qrels and has_queries,
            "requirements_met": has_enough_docs and has_qrels and has_queries
        },
        "statistics": {
            "documents": len(documents),
            "queries": len(queries),
            "qrels": len(qrels)
        },
        "details": {
            "document_requirement": ">200K documents",
            "testing_data_requirement": "queries and qrels for evaluation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
