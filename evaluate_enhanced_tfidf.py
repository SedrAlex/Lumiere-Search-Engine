#!/usr/bin/env python3
"""
Enhanced TF-IDF Evaluation Script
Evaluates the enhanced TF-IDF service with optimized parameters to achieve MAP ≥ 0.4
"""

import asyncio
import logging
import time
import ir_datasets
import numpy as np
import pandas as pd
import httpx
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation engine
from evaluation_engine import IRMetrics, SearchEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ENHANCED_TFIDF_SERVICE_URL = "http://localhost:8007"
INVERTED_INDEX_SERVICE_URL = "http://localhost:8006"
RESULTS_DIR = Path("enhanced_evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

class EnhancedTFIDFEvaluator:
    """Enhanced TF-IDF system evaluator with optimization tracking"""
    
    def __init__(self, service_url: str = ENHANCED_TFIDF_SERVICE_URL):
        self.service_url = service_url
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self.search_evaluator = SearchEvaluator()
        self.metrics = IRMetrics()
        self.evaluation_results = {}
        
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    async def check_services_status(self) -> Dict[str, Any]:
        """Check if enhanced services are running and ready"""
        services_status = {}
        
        # Check Enhanced TF-IDF Service
        try:
            response = await self.http_client.get(f"{self.service_url}/health")
            services_status["enhanced_tfidf"] = {
                "available": response.status_code == 200,
                "details": response.json() if response.status_code == 200 else None
            }
            if response.status_code == 200:
                service_info = response.json()
                logger.info(f"✅ Enhanced TF-IDF Service: {service_info}")
            else:
                logger.error(f"❌ Enhanced TF-IDF Service not healthy: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Enhanced TF-IDF Service unavailable: {e}")
            services_status["enhanced_tfidf"] = {"available": False, "error": str(e)}
        
        # Check Inverted Index Service
        try:
            response = await self.http_client.get(f"{INVERTED_INDEX_SERVICE_URL}/health")
            services_status["inverted_index"] = {
                "available": response.status_code == 200,
                "details": response.json() if response.status_code == 200 else None
            }
            if response.status_code == 200:
                logger.info(f"✅ Inverted Index Service available")
        except Exception as e:
            logger.warning(f"⚠️ Inverted Index Service unavailable: {e}")
            services_status["inverted_index"] = {"available": False, "error": str(e)}
        
        return services_status
    
    def load_antique_dataset(self) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
        """Load ANTIQUE test queries and qrels"""
        logger.info("📚 Loading ANTIQUE dataset...")
        
        # Load test dataset
        dataset = ir_datasets.load('antique/test')
        
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
        
        logger.info(f"📊 Loaded {len(queries)} queries")
        logger.info(f"📊 Loaded qrels for {len(qrels)} queries")
        
        # Filter queries that have relevance judgments
        filtered_queries = [q for q in queries if q['query_id'] in qrels]
        logger.info(f"📊 {len(filtered_queries)} queries have relevance judgments")
        
        return filtered_queries, qrels
    
    async def query_enhanced_system(self, query_text: str, 
                                    use_query_expansion: bool = True,
                                    enable_reranking: bool = True,
                                    top_k: int = 100) -> Tuple[List[str], Dict]:
        """Query the enhanced TF-IDF system"""
        try:
            request_data = {
                "query": query_text,
                "top_k": top_k,
                "use_query_expansion": use_query_expansion,
                "enable_reranking": enable_reranking,
                "similarity_threshold": 0.0
            }
            
            response = await self.http_client.post(
                f"{self.service_url}/search",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract document IDs in rank order
            doc_ids = [doc["document_id"] for doc in result["results"]]
            
            # Extract metadata for analysis
            metadata = {
                "original_query": result.get("query", ""),
                "expanded_query": result.get("expanded_query"),
                "total_results": result.get("total_results", 0),
                "processing_time": result.get("processing_time", 0),
                "search_stats": result.get("search_stats", {}),
                "scores": [doc["score"] for doc in result["results"]],
                "explanations": [doc.get("explanation", {}) for doc in result["results"]]
            }
            
            return doc_ids, metadata
            
        except httpx.RequestError as e:
            logger.error(f"❌ Service request error for query '{query_text}': {e}")
            return [], {}
        except Exception as e:
            logger.error(f"❌ Error processing query '{query_text}': {e}")
            return [], {}
    
    async def evaluate_configuration(self, config: Dict[str, Any], 
                                     queries: List[Dict], qrels: Dict) -> Dict[str, Any]:
        """Evaluate a specific configuration"""
        config_name = config.get("name", "default")
        logger.info(f"🔬 Evaluating configuration: {config_name}")
        
        query_results = []
        overall_metrics = {
            'map': 0.0,
            'mrr': 0.0,
            'ndcg_at_10': 0.0,
            'precision_at_1': 0.0,
            'precision_at_5': 0.0,
            'precision_at_10': 0.0,
            'recall_at_10': 0.0,
            'evaluated_queries': 0
        }
        
        # Evaluation settings
        use_query_expansion = config.get("use_query_expansion", True)
        enable_reranking = config.get("enable_reranking", True)
        
        for query in tqdm(queries[:50], desc=f"Evaluating {config_name}"):  # Limit to 50 queries for faster testing
            query_id = query['query_id']
            query_text = query['text']
            query_qrels = qrels.get(query_id, {})
            
            if not query_qrels:
                continue
            
            # Get search results
            retrieved_docs, query_metadata = await self.query_enhanced_system(
                query_text, 
                use_query_expansion=use_query_expansion,
                enable_reranking=enable_reranking,
                top_k=100
            )
            
            if not retrieved_docs:
                continue
            
            # Calculate metrics for this query
            query_metrics = self.search_evaluator.evaluate_single_query(
                retrieved_docs, query_qrels, k_values=[1, 5, 10, 20]
            )
            
            # Add query metadata
            query_metrics.update({
                'query_id': query_id,
                'query_text': query_text,
                'query_metadata': query_metadata,
                'config': config_name
            })
            
            query_results.append(query_metrics)
            
            # Update overall metrics
            overall_metrics['map'] += query_metrics.get('average_precision', 0)
            overall_metrics['mrr'] += query_metrics.get('reciprocal_rank', 0)
            overall_metrics['ndcg_at_10'] += query_metrics.get('ndcg_at_10', 0)
            overall_metrics['precision_at_1'] += query_metrics.get('precision_at_1', 0)
            overall_metrics['precision_at_5'] += query_metrics.get('precision_at_5', 0)
            overall_metrics['precision_at_10'] += query_metrics.get('precision_at_10', 0)
            overall_metrics['recall_at_10'] += query_metrics.get('recall_at_10', 0)
            overall_metrics['evaluated_queries'] += 1
        
        # Calculate averages
        if overall_metrics['evaluated_queries'] > 0:
            for metric in ['map', 'mrr', 'ndcg_at_10', 'precision_at_1', 
                          'precision_at_5', 'precision_at_10', 'recall_at_10']:
                overall_metrics[metric] /= overall_metrics['evaluated_queries']
        
        return {
            'config': config,
            'overall_metrics': overall_metrics,
            'query_results': query_results
        }
    
    async def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with different configurations"""
        logger.info("🚀 Starting comprehensive enhanced TF-IDF evaluation")
        
        # Check services
        services_status = await self.check_services_status()
        if not services_status.get("enhanced_tfidf", {}).get("available", False):
            logger.error("❌ Enhanced TF-IDF service not available. Please start it first.")
            return
        
        # Load dataset
        queries, qrels = self.load_antique_dataset()
        
        # Define configurations to test
        configurations = [
            {
                "name": "baseline",
                "description": "Basic enhanced TF-IDF without query expansion or reranking",
                "use_query_expansion": False,
                "enable_reranking": False
            },
            {
                "name": "query_expansion_only",
                "description": "Enhanced TF-IDF with query expansion",
                "use_query_expansion": True,
                "enable_reranking": False
            },
            {
                "name": "reranking_only", 
                "description": "Enhanced TF-IDF with semantic reranking",
                "use_query_expansion": False,
                "enable_reranking": True
            },
            {
                "name": "full_enhanced",
                "description": "Full enhanced TF-IDF with all optimizations",
                "use_query_expansion": True,
                "enable_reranking": True
            }
        ]
        
        all_results = {}
        
        # Evaluate each configuration
        for config in configurations:
            try:
                result = await self.evaluate_configuration(config, queries, qrels)
                all_results[config["name"]] = result
                
                # Log results
                metrics = result['overall_metrics']
                logger.info(f"📊 Results for {config['name']}:")
                logger.info(f"   MAP: {metrics['map']:.4f}")
                logger.info(f"   MRR: {metrics['mrr']:.4f}")
                logger.info(f"   NDCG@10: {metrics['ndcg_at_10']:.4f}")
                logger.info(f"   P@10: {metrics['precision_at_10']:.4f}")
                logger.info(f"   R@10: {metrics['recall_at_10']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating {config['name']}: {e}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"enhanced_tfidf_evaluation_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for config_name, result in all_results.items():
            json_results[config_name] = {
                'config': result['config'],
                'overall_metrics': result['overall_metrics'],
                'query_count': len(result['query_results'])
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_file}")
        
        # Generate comparison report
        self._generate_comparison_report(all_results, timestamp)
        
        # Find best configuration
        best_config = max(all_results.items(), key=lambda x: x[1]['overall_metrics']['map'])
        best_map = best_config[1]['overall_metrics']['map']
        
        logger.info(f"🏆 Best configuration: {best_config[0]} with MAP = {best_map:.4f}")
        
        if best_map >= 0.4:
            logger.info(f"🎉 SUCCESS! Achieved MAP ≥ 0.4 with {best_config[0]} configuration")
        else:
            logger.info(f"📈 Current best MAP: {best_map:.4f}. Target: 0.4")
            logger.info("💡 Suggestions for further improvement:")
            logger.info("   • Increase vocabulary size further (150k-200k)")
            logger.info("   • Tune LSA components (300 → 500)")
            logger.info("   • Implement pseudo-relevance feedback")
            logger.info("   • Add document-specific boosting")
            logger.info("   • Optimize query expansion parameters")
        
        return all_results
    
    def _generate_comparison_report(self, results: Dict[str, Any], timestamp: str):
        """Generate a detailed comparison report"""
        report_file = RESULTS_DIR / f"enhanced_comparison_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Enhanced TF-IDF Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration Comparison:\n")
            f.write("-" * 30 + "\n")
            
            # Create comparison table
            configs = list(results.keys())
            metrics = ['map', 'mrr', 'ndcg_at_10', 'precision_at_10', 'recall_at_10']
            
            # Header
            f.write(f"{'Config':<20}")
            for metric in metrics:
                f.write(f"{metric.upper():<12}")
            f.write("\n")
            f.write("-" * (20 + 12 * len(metrics)) + "\n")
            
            # Data rows
            for config in configs:
                f.write(f"{config:<20}")
                for metric in metrics:
                    value = results[config]['overall_metrics'][metric]
                    f.write(f"{value:<12.4f}")
                f.write("\n")
            
            f.write("\n\nDetailed Analysis:\n")
            f.write("-" * 20 + "\n")
            
            for config_name, result in results.items():
                f.write(f"\n{config_name.upper()}:\n")
                f.write(f"Description: {result['config']['description']}\n")
                metrics = result['overall_metrics']
                
                f.write(f"MAP: {metrics['map']:.4f}\n")
                f.write(f"MRR: {metrics['mrr']:.4f}\n")
                f.write(f"NDCG@10: {metrics['ndcg_at_10']:.4f}\n")
                f.write(f"Precision@10: {metrics['precision_at_10']:.4f}\n")
                f.write(f"Recall@10: {metrics['recall_at_10']:.4f}\n")
                f.write(f"Evaluated queries: {metrics['evaluated_queries']}\n")
        
        logger.info(f"📋 Comparison report saved to {report_file}")

async def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Enhanced TF-IDF Evaluation")
    parser.add_argument("--service-url", default=ENHANCED_TFIDF_SERVICE_URL,
                       help="Enhanced TF-IDF service URL")
    parser.add_argument("--queries-limit", type=int, default=50,
                       help="Limit number of queries to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    evaluator = EnhancedTFIDFEvaluator(service_url=args.service_url)
    
    try:
        results = await evaluator.run_comprehensive_evaluation()
        return results
    finally:
        await evaluator.close()

if __name__ == "__main__":
    results = asyncio.run(main())
