#!/usr/bin/env python3
"""
MAP Evaluation Microservice
Provides MAP (Mean Average Precision) evaluation services via HTTP API on port 8004
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import uvicorn
import requests
import ir_datasets
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global evaluation data
evaluation_data = {
    "antique": {
        "queries": {},
        "qrels": {},
        "loaded": False
    }
}

# Service URLs
ENHANCED_TFIDF_SERVICE_URL = "http://localhost:8003"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MAP Evaluation service on startup"""
    logger.info("Initializing MAP Evaluation Service...")
    
    # Load ANTIQUE evaluation data
    try:
        logger.info("Loading ANTIQUE evaluation data...")
        dataset = ir_datasets.load("antique")
        
        # Load queries
        queries = {}
        for query in dataset.queries_iter():
            queries[query.query_id] = query.text
        
        # Load qrels (relevance judgments)
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        evaluation_data["antique"]["queries"] = queries
        evaluation_data["antique"]["qrels"] = dict(qrels)
        evaluation_data["antique"]["loaded"] = True
        
        logger.info(f"✓ Loaded {len(queries)} queries and {len(qrels)} qrels for ANTIQUE")
        
    except Exception as e:
        logger.error(f"Failed to load ANTIQUE evaluation data: {str(e)}")
    
    logger.info("✓ MAP Evaluation Service ready on port 8004")
    yield
    logger.info("Shutting down MAP Evaluation Service...")

# Create FastAPI app
app = FastAPI(
    title="MAP Evaluation Microservice",
    description="MAP and IR metrics evaluation service",
    version="2.0.0",
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
class EvaluationRequest(BaseModel):
    dataset_name: str = "antique"
    max_queries: Optional[int] = None
    k_eval: Optional[int] = 10

class SearchResultsRequest(BaseModel):
    search_results: Dict[str, List[str]]  # query_id -> [doc_ids]
    dataset_name: str = "antique"
    k: Optional[int] = None

class MAPResult(BaseModel):
    MAP: float
    num_queries: int
    individual_ap_scores: List[float]
    query_metrics: Dict[str, Dict]
    cutoff_k: Optional[int]

class PrecisionRecallResult(BaseModel):
    precision_at_k: Dict[str, float]
    recall_at_k: Dict[str, float]

class EvaluationResponse(BaseModel):
    dataset: str
    evaluation_method: str
    num_queries_evaluated: int
    cutoff_k: int
    MAP: float
    precision_recall: Dict[str, float]
    query_performance: Dict[str, Any]
    recommendations: List[str]
    evaluation_time_ms: float

class ServiceInfoResponse(BaseModel):
    service_name: str
    version: str
    port: int
    datasets_loaded: Dict[str, bool]
    available_datasets: List[str]

# Helper functions
def calculate_average_precision(retrieved_docs: List[str], 
                               relevant_docs: Dict[str, int],
                               k: Optional[int] = None) -> float:
    """Calculate Average Precision (AP) for a single query"""
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    # Use only top-k documents if specified
    if k is not None:
        retrieved_docs = retrieved_docs[:k]
    
    # Calculate precision at each relevant document position
    num_relevant = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
            num_relevant += 1
            precision_at_i = num_relevant / i
            precision_sum += precision_at_i
    
    # Calculate average precision
    total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
    
    if total_relevant == 0:
        return 0.0
    
    return precision_sum / total_relevant

def calculate_map(search_results: Dict[str, List[str]], 
                 dataset_name: str = 'antique',
                 k: Optional[int] = None) -> Dict:
    """Calculate Mean Average Precision (MAP) across all queries"""
    if dataset_name not in evaluation_data or not evaluation_data[dataset_name]["loaded"]:
        raise ValueError(f"No evaluation data loaded for dataset: {dataset_name}")
    
    qrels = evaluation_data[dataset_name]["qrels"]
    ap_scores = []
    query_metrics = {}
    
    for query_id, retrieved_docs in search_results.items():
        if query_id in qrels:
            relevant_docs = qrels[query_id]
            ap_score = calculate_average_precision(retrieved_docs, relevant_docs, k)
            ap_scores.append(ap_score)
            
            # Calculate additional metrics for this query
            query_metrics[query_id] = {
                'average_precision': ap_score,
                'relevant_docs_count': sum(1 for rel in relevant_docs.values() if rel > 0),
                'retrieved_docs_count': len(retrieved_docs),
                'relevant_retrieved': len([doc for doc in retrieved_docs[:k] if doc in relevant_docs and relevant_docs[doc] > 0]) if k else len([doc for doc in retrieved_docs if doc in relevant_docs and relevant_docs[doc] > 0])
            }
    
    # Calculate MAP
    map_score = np.mean(ap_scores) if ap_scores else 0.0
    
    return {
        'MAP': map_score,
        'num_queries': len(ap_scores),
        'individual_ap_scores': ap_scores,
        'query_metrics': query_metrics,
        'cutoff_k': k
    }

def calculate_precision_recall_at_k(search_results: Dict[str, List[str]], 
                                   dataset_name: str = 'antique',
                                   k_values: List[int] = [1, 5, 10, 20]) -> Dict:
    """Calculate Precision@K and Recall@K for multiple K values"""
    if dataset_name not in evaluation_data or not evaluation_data[dataset_name]["loaded"]:
        raise ValueError(f"No evaluation data loaded for dataset: {dataset_name}")
    
    qrels = evaluation_data[dataset_name]["qrels"]
    metrics = {f'P@{k}': [] for k in k_values}
    metrics.update({f'R@{k}': [] for k in k_values})
    
    for query_id, retrieved_docs in search_results.items():
        if query_id in qrels:
            relevant_docs = qrels[query_id]
            total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
            
            for k in k_values:
                # Get top-k retrieved documents
                top_k_docs = retrieved_docs[:k]
                
                # Count relevant documents in top-k
                relevant_in_k = sum(1 for doc in top_k_docs if doc in relevant_docs and relevant_docs[doc] > 0)
                
                # Calculate Precision@K
                precision_k = relevant_in_k / k if k > 0 else 0
                metrics[f'P@{k}'].append(precision_k)
                
                # Calculate Recall@K
                recall_k = relevant_in_k / total_relevant if total_relevant > 0 else 0
                metrics[f'R@{k}'].append(recall_k)
    
    # Calculate averages
    result = {}
    for metric, values in metrics.items():
        result[metric] = np.mean(values) if values else 0.0
    
    return result

def analyze_query_performance(query_metrics: Dict) -> Dict:
    """Analyze query performance patterns"""
    ap_scores = [metrics['average_precision'] for metrics in query_metrics.values()]
    
    return {
        'mean_ap': np.mean(ap_scores),
        'median_ap': np.median(ap_scores),
        'std_ap': np.std(ap_scores),
        'min_ap': np.min(ap_scores),
        'max_ap': np.max(ap_scores),
        'queries_above_0_4': sum(1 for ap in ap_scores if ap > 0.4),
        'percentage_above_0_4': (sum(1 for ap in ap_scores if ap > 0.4) / len(ap_scores)) * 100 if ap_scores else 0
    }

def generate_recommendations(map_score: float, precision_recall: Dict) -> List[str]:
    """Generate recommendations for improving performance"""
    recommendations = []
    
    if map_score < 0.4:
        recommendations.append("MAP is below 0.4 target. Consider:")
        recommendations.append("- Tuning TF-IDF parameters (min_df, max_df, ngram_range)")
        recommendations.append("- Improving text preprocessing (stemming, stopwords)")
        recommendations.append("- Adjusting inverted index + TF-IDF fusion weights")
    
    if precision_recall.get('P@1', 0) < 0.3:
        recommendations.append("Low P@1 suggests poor ranking. Consider query expansion or term weighting.")
    
    if precision_recall.get('R@10', 0) < 0.5:
        recommendations.append("Low R@10 suggests missing relevant docs. Consider relaxing filtering.")
    
    if not recommendations:
        recommendations.append("Performance looks good! Consider testing on larger query sets.")
    
    return recommendations

async def call_enhanced_tfidf_service(endpoint: str, data: Dict = None) -> Dict:
    """Call the Enhanced TF-IDF microservice"""
    try:
        if data:
            response = requests.post(f"{ENHANCED_TFIDF_SERVICE_URL}{endpoint}", json=data)
        else:
            response = requests.get(f"{ENHANCED_TFIDF_SERVICE_URL}{endpoint}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Enhanced TF-IDF service error: {response.status_code}")
            return {"error": f"Service error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error calling Enhanced TF-IDF service: {str(e)}")
        return {"error": str(e)}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "MAP Evaluation Microservice",
        "version": "2.0.0",
        "port": 8004,
        "status": "running",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /info": "Detailed service info",
            "POST /evaluate": "Evaluate TF-IDF service with MAP",
            "POST /calculate_map": "Calculate MAP from search results",
            "POST /calculate_precision_recall": "Calculate P@K and R@K",
            "GET /datasets": "List available datasets"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "map_evaluation",
        "port": 8004,
        "antique_loaded": evaluation_data["antique"]["loaded"],
        "ready": True
    }

@app.get("/info", response_model=ServiceInfoResponse)
async def get_service_info():
    """Get detailed service information"""
    datasets_loaded = {
        dataset: data["loaded"] 
        for dataset, data in evaluation_data.items()
    }
    
    return ServiceInfoResponse(
        service_name="MAP Evaluation Microservice",
        version="2.0.0",
        port=8004,
        datasets_loaded=datasets_loaded,
        available_datasets=list(evaluation_data.keys())
    )

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_tfidf_service(request: EvaluationRequest):
    """Comprehensive evaluation of Enhanced TF-IDF service"""
    import time
    start_time = time.time()
    
    if request.dataset_name not in evaluation_data or not evaluation_data[request.dataset_name]["loaded"]:
        raise HTTPException(status_code=400, detail=f"Dataset {request.dataset_name} not loaded")
    
    try:
        # Check if Enhanced TF-IDF service is trained
        service_info = await call_enhanced_tfidf_service("/info")
        if "error" in service_info:
            raise HTTPException(status_code=503, detail="Enhanced TF-IDF service not available")
        
        if not service_info.get("is_trained", False):
            raise HTTPException(status_code=400, detail="Enhanced TF-IDF service not trained")
        
        # Get queries for evaluation
        queries = evaluation_data[request.dataset_name]["queries"]
        query_items = list(queries.items())
        
        if request.max_queries:
            query_items = query_items[:request.max_queries]
        
        logger.info(f"Evaluating {len(query_items)} queries...")
        
        # Run searches for all queries
        search_results = {}
        for query_id, query_text in query_items:
            try:
                # Search using Enhanced TF-IDF service
                search_request = {
                    "query": query_text,
                    "top_k": request.k_eval,
                    "method": "enhanced_inverted"
                }
                
                search_response = await call_enhanced_tfidf_service("/search", search_request)
                
                if "error" not in search_response:
                    search_results[query_id] = [result["doc_id"] for result in search_response.get("results", [])]
                else:
                    logger.warning(f"Search error for query {query_id}: {search_response['error']}")
                    search_results[query_id] = []
                
            except Exception as e:
                logger.warning(f"Error searching query {query_id}: {str(e)}")
                search_results[query_id] = []
        
        # Calculate evaluation metrics
        map_results = calculate_map(search_results, request.dataset_name, request.k_eval)
        precision_recall = calculate_precision_recall_at_k(
            search_results, request.dataset_name, [1, 5, 10, 20]
        )
        
        # Analyze query performance
        query_performance = analyze_query_performance(map_results['query_metrics'])
        
        # Generate recommendations
        recommendations = generate_recommendations(map_results['MAP'], precision_recall)
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return EvaluationResponse(
            dataset=request.dataset_name,
            evaluation_method="enhanced_inverted_index",
            num_queries_evaluated=len(query_items),
            cutoff_k=request.k_eval,
            MAP=map_results['MAP'],
            precision_recall=precision_recall,
            query_performance=query_performance,
            recommendations=recommendations,
            evaluation_time_ms=evaluation_time
        )
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/calculate_map", response_model=MAPResult)
async def calculate_map_endpoint(request: SearchResultsRequest):
    """Calculate MAP from provided search results"""
    try:
        map_results = calculate_map(
            request.search_results, 
            request.dataset_name, 
            request.k
        )
        
        return MAPResult(**map_results)
        
    except Exception as e:
        logger.error(f"Error calculating MAP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MAP calculation error: {str(e)}")

@app.post("/calculate_precision_recall", response_model=PrecisionRecallResult)
async def calculate_precision_recall_endpoint(request: SearchResultsRequest):
    """Calculate Precision@K and Recall@K from provided search results"""
    try:
        k_values = [1, 5, 10, 20]
        if request.k:
            k_values = [1, 5, 10, 20, request.k] if request.k not in [1, 5, 10, 20] else [1, 5, 10, 20]
        
        precision_recall = calculate_precision_recall_at_k(
            request.search_results,
            request.dataset_name,
            k_values
        )
        
        # Split into precision and recall
        precision_at_k = {k: v for k, v in precision_recall.items() if k.startswith('P@')}
        recall_at_k = {k: v for k, v in precision_recall.items() if k.startswith('R@')}
        
        return PrecisionRecallResult(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k
        )
        
    except Exception as e:
        logger.error(f"Error calculating precision/recall: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Precision/Recall calculation error: {str(e)}")

@app.get("/datasets")
async def get_datasets():
    """Get information about available datasets"""
    datasets_info = {}
    
    for dataset_name, data in evaluation_data.items():
        datasets_info[dataset_name] = {
            "loaded": data["loaded"],
            "num_queries": len(data["queries"]) if data["loaded"] else 0,
            "num_qrels": len(data["qrels"]) if data["loaded"] else 0
        }
    
    return {
        "available_datasets": datasets_info,
        "note": "Only loaded datasets can be used for evaluation"
    }

@app.get("/ping")
async def ping():
    """Simple ping endpoint for service discovery"""
    return {"service": "map_evaluation", "status": "pong", "port": 8004}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
