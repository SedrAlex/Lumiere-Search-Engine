#!/usr/bin/env python3
"""
TF-IDF Evaluation Script with Service Query Processing
Calculates MAP, Recall, Precision@10, and MRR metrics for TF-IDF system
"""

import asyncio
import logging
import time
import ir_datasets
import numpy as np
import joblib
import httpx
from typing import List, Dict, Any
from tqdm import tqdm

# Import evaluation engine
from evaluation_engine import IRMetrics, SearchEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TFIDF_QUERY_SERVICE_URL = "http://localhost:8004"
MODEL_BASE_PATH = "/Users/raafatmhanna/Desktop/custom-search-engine/backend/models"

class TFIDFEvaluator:
    """Evaluator for TF-IDF system using service query processing"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.evaluator = SearchEvaluator()
        self.metrics = IRMetrics()
        
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    async def query_tfidf_service(self, query: str, top_k: int = 1000) -> List[str]:
        """Query the TF-IDF service and return ranked document IDs"""
        try:
            response = await self.http_client.post(
                f"{TFIDF_QUERY_SERVICE_URL}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "similarity_threshold": 0.0,
                    "use_enhanced_cleaning": True
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract document IDs in rank order
            return [doc["doc_id"] for doc in result["results"]]
            
        except httpx.RequestError as e:
            logger.error(f"Service request error for query '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return []
    
    async def check_service_status(self) -> bool:
        """Check if TF-IDF service is available"""
        try:
            response = await self.http_client.get(f"{TFIDF_QUERY_SERVICE_URL}/health")
            return response.status_code == 200
        except:
            return False
    
    def load_dataset_queries_and_qrels(self):
        """Load queries and relevance judgments from ANTIQUE dataset"""
        logger.info("Loading ANTIQUE dataset queries and qrels...")
        
        dataset = ir_datasets.load('antique/train')
        
        # Load queries
        queries = []
        for query in dataset.queries_iter():
            queries.append({
                'query_id': query.query_id,
                'text': query.text
            })
        
        # Load qrels (relevance judgments)
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        logger.info(f"Loaded {len(queries)} queries and qrels for {len(qrels)} queries")
        return queries, qrels
    
    async def evaluate_single_query(self, query: Dict[str, str], query_qrels: Dict[str, float]) -> Dict[str, float]:
        """Evaluate a single query and return metrics"""
        query_id = query['query_id']
        query_text = query['text']
        
        # Get search results from TF-IDF service
        retrieved_docs = await self.query_tfidf_service(query_text, top_k=1000)
        
        if not retrieved_docs:
            logger.warning(f"No results for query {query_id}")
            return {
                'precision_at_10': 0.0,
                'recall_at_10': 0.0,
                'average_precision': 0.0,
                'reciprocal_rank': 0.0
            }
        
        # Get relevant documents (relevance > 0)
        relevant_docs = [doc_id for doc_id, rel in query_qrels.items() if rel > 0]
        
        if not relevant_docs:
            logger.warning(f"No relevant documents for query {query_id}")
            return {
                'precision_at_10': 0.0,
                'recall_at_10': 0.0,
                'average_precision': 0.0,
                'reciprocal_rank': 0.0
            }
        
        # Calculate metrics
        precision_at_10 = self.metrics.precision_at_k(retrieved_docs, relevant_docs, 10)
        recall_at_10 = self.metrics.recall_at_k(retrieved_docs, relevant_docs, 10)
        average_precision = self.metrics.average_precision(retrieved_docs, relevant_docs)
        reciprocal_rank = self.metrics.reciprocal_rank(retrieved_docs, relevant_docs)
        
        return {
            'precision_at_10': precision_at_10,
            'recall_at_10': recall_at_10,
            'average_precision': average_precision,
            'reciprocal_rank': reciprocal_rank
        }
    
    async def evaluate_system(self, queries: List[Dict], qrels: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Evaluate the entire TF-IDF system"""
        logger.info("Starting TF-IDF system evaluation...")
        
        # Check service availability
        if not await self.check_service_status():
            raise RuntimeError("TF-IDF service is not available. Please start the service first.")
        
        query_results = []
        evaluated_queries = 0
        
        # Process each query
        for query in tqdm(queries, desc="Evaluating queries"):
            query_id = query['query_id']
            
            # Skip queries without relevance judgments
            if query_id not in qrels or not qrels[query_id]:
                continue
            
            try:
                query_metrics = await self.evaluate_single_query(query, qrels[query_id])
                query_results.append(query_metrics)
                evaluated_queries += 1
                
                # Log progress every 50 queries
                if evaluated_queries % 50 == 0:
                    logger.info(f"Evaluated {evaluated_queries} queries...")
                    
            except Exception as e:
                logger.error(f"Error evaluating query {query_id}: {e}")
                continue
        
        if not query_results:
            logger.error("No queries were successfully evaluated")
            return {}
        
        # Aggregate results
        aggregated = {}
        for metric in query_results[0].keys():
            values = [result[metric] for result in query_results if metric in result]
            aggregated[metric] = sum(values) / len(values) if values else 0.0
        
        # Calculate MAP and MRR
        aggregated['map'] = aggregated.get('average_precision', 0.0)  # MAP = mean AP
        aggregated['mrr'] = aggregated.get('reciprocal_rank', 0.0)    # MRR = mean RR
        aggregated['num_queries_evaluated'] = evaluated_queries
        
        return aggregated
    
    def print_detailed_results(self, results: Dict[str, float]):
        """Print detailed evaluation results"""
        print("\n" + "="*60)
        print("TF-IDF EVALUATION RESULTS")
        print("="*60)
        
        if not results:
            print("❌ No results available")
            return
        
        print(f"📊 Evaluated on {results.get('num_queries_evaluated', 0)} queries")
        print("\n📈 CORE METRICS:")
        print(f"   Mean Average Precision (MAP): {results.get('map', 0.0):.4f}")
        print(f"   Mean Reciprocal Rank (MRR):  {results.get('mrr', 0.0):.4f}")
        print(f"   Precision@10:                {results.get('precision_at_10', 0.0):.4f}")
        print(f"   Recall@10:                   {results.get('recall_at_10', 0.0):.4f}")
        
        print("\n📋 INTERPRETATION:")
        map_score = results.get('map', 0.0)
        if map_score >= 0.3:
            print("   MAP: Excellent performance ✅")
        elif map_score >= 0.2:
            print("   MAP: Good performance 👍")
        elif map_score >= 0.1:
            print("   MAP: Fair performance ⚠️")
        else:
            print("   MAP: Poor performance ❌")
        
        mrr_score = results.get('mrr', 0.0)
        if mrr_score >= 0.5:
            print("   MRR: Excellent first relevant result ranking ✅")
        elif mrr_score >= 0.3:
            print("   MRR: Good first relevant result ranking 👍")
        elif mrr_score >= 0.2:
            print("   MRR: Fair first relevant result ranking ⚠️")
        else:
            print("   MRR: Poor first relevant result ranking ❌")
        
        print("="*60)

async def main():
    """Main evaluation function"""
    evaluator = TFIDFEvaluator()
    
    try:
        # Load dataset
        queries, qrels = evaluator.load_dataset_queries_and_qrels()
        
        # Evaluate system
        start_time = time.time()
        results = await evaluator.evaluate_system(queries, qrels)
        evaluation_time = time.time() - start_time
        
        # Print results
        evaluator.print_detailed_results(results)
        print(f"\n⏱️  Total evaluation time: {evaluation_time:.2f} seconds")
        
        # Save results
        import json
        output_file = "tfidf_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        await evaluator.close()

if __name__ == "__main__":
    print("🚀 Starting TF-IDF Evaluation with Service Query Processing")
    print("📋 Metrics to calculate: MAP, MRR, Precision@10, Recall@10")
    print("🔧 Make sure TF-IDF query service is running on port 8004")
    print("")
    
    # Run evaluation
    asyncio.run(main())
