#!/usr/bin/env python3
"""
Complete End-to-End TF-IDF Evaluation System
=============================================

This script evaluates the COMPLETE TF-IDF pipeline including:
1. Query processing service (with text cleaning)
2. TF-IDF vectorization 
3. Cosine similarity computation
4. Ranking and result retrieval

Evaluation Metrics:
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank) 
- Precision@K (K=1,5,10,20)
- Recall@K (K=1,5,10,20)
- NDCG@K (K=1,5,10,20)

Uses ANTIQUE dataset queries and qrels for ground truth.
"""

import asyncio
import logging
import time
import ir_datasets
import numpy as np
import pandas as pd
import httpx
import json
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
TFIDF_QUERY_SERVICE_URL = "http://localhost:8004"
RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

class CompleteTFIDFEvaluator:
    """Complete end-to-end TF-IDF system evaluator"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.search_evaluator = SearchEvaluator()
        self.metrics = IRMetrics()
        self.evaluation_results = {}
        
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    async def check_service_availability(self) -> Dict[str, bool]:
        """Check if all required services are running"""
        services_status = {}
        
        # Check TF-IDF Query Service
        try:
            response = await self.http_client.get(f"{TFIDF_QUERY_SERVICE_URL}/health")
            services_status["tfidf_query_service"] = response.status_code == 200
            if response.status_code == 200:
                service_info = response.json()
                logger.info(f"✅ TF-IDF Query Service: {service_info}")
            else:
                logger.error(f"❌ TF-IDF Query Service not healthy: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ TF-IDF Query Service unavailable: {e}")
            services_status["tfidf_query_service"] = False
            
        return services_status
    
    def load_antique_dataset(self) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
        """Load ANTIQUE test queries and qrels"""
        logger.info("📚 Loading ANTIQUE dataset...")
        
        # Load test dataset (for evaluation)
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
    
    async def query_tfidf_system(self, query_text: str, top_k: int = 100) -> Tuple[List[str], Dict]:
        """Query the complete TF-IDF system and return results with metadata"""
        try:
            response = await self.http_client.post(
                f"{TFIDF_QUERY_SERVICE_URL}/search",
                json={
                    "query": query_text,
                    "top_k": top_k,
                    "use_enhanced_cleaning": True
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract document IDs in rank order
            doc_ids = [doc["doc_id"] for doc in result["results"]]
            
            # Extract metadata for analysis
            metadata = {
                "original_query": result.get("query", ""),
                "cleaned_query": result.get("cleaned_query", ""),
                "total_results": result.get("total_results", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "similarity_stats": result.get("similarity_stats", {}),
                "scores": [doc["score"] for doc in result["results"]]
            }
            
            return doc_ids, metadata
            
        except httpx.RequestError as e:
            logger.error(f"❌ Service request error for query '{query_text}': {e}")
            return [], {}
        except Exception as e:
            logger.error(f"❌ Error processing query '{query_text}': {e}")
            return [], {}
    
    async def evaluate_single_query(self, query: Dict[str, str], query_qrels: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a single query and return detailed metrics"""
        query_id = query['query_id']
        query_text = query['text']
        
        logger.debug(f"🔍 Evaluating query {query_id}: '{query_text}'")
        
        # Get search results from TF-IDF system
        retrieved_docs, query_metadata = await self.query_tfidf_system(query_text, top_k=100)
        
        if not retrieved_docs:
            logger.warning(f"⚠️ No results for query {query_id}")
            return self._get_zero_metrics(query_id, query_text, query_metadata)
        
        # Get relevant documents (relevance > 0)
        relevant_docs = [doc_id for doc_id, rel in query_qrels.items() if rel > 0]
        
        if not relevant_docs:
            logger.warning(f"⚠️ No relevant documents for query {query_id}")
            return self._get_zero_metrics(query_id, query_text, query_metadata)
        
        # Calculate all metrics using our evaluation engine
        metrics = self.search_evaluator.evaluate_single_query(
            retrieved_docs, query_qrels, k_values=[1, 5, 10, 20]
        )
        
        # Add query-specific information
        metrics.update({
            'query_id': query_id,
            'query_text': query_text,
            'num_retrieved': len(retrieved_docs),
            'num_relevant': len(relevant_docs),
            'num_relevant_retrieved': len(set(retrieved_docs) & set(relevant_docs))
        })
        
        # Add system metadata
        metrics.update(query_metadata)
        
        return metrics
    
    def _get_zero_metrics(self, query_id: str, query_text: str, metadata: Dict) -> Dict[str, Any]:
        """Return zero metrics for failed queries"""
        zero_metrics = {
            'query_id': query_id,
            'query_text': query_text,
            'num_retrieved': 0,
            'num_relevant': 0,
            'num_relevant_retrieved': 0,
            'precision_at_1': 0.0,
            'precision_at_5': 0.0,
            'precision_at_10': 0.0,
            'precision_at_20': 0.0,
            'recall_at_1': 0.0,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
            'recall_at_20': 0.0,
            'ndcg_at_1': 0.0,
            'ndcg_at_5': 0.0,
            'ndcg_at_10': 0.0,
            'ndcg_at_20': 0.0,
            'average_precision': 0.0,
            'reciprocal_rank': 0.0
        }
        zero_metrics.update(metadata)
        return zero_metrics
    
    async def run_complete_evaluation(self, max_queries: Optional[int] = None) -> Dict[str, Any]:
        """Run complete evaluation of TF-IDF system"""
        logger.info("🚀 Starting Complete TF-IDF System Evaluation")
        
        # Check service availability
        services_status = await self.check_service_availability()
        if not services_status.get("tfidf_query_service", False):
            raise RuntimeError("❌ TF-IDF Query Service is not available. Please start it first.")
        
        # Load dataset
        queries, qrels = self.load_antique_dataset()
        
        # Limit queries if specified
        if max_queries:
            queries = queries[:max_queries]
            logger.info(f"📊 Limited evaluation to {len(queries)} queries")
        
        # Run evaluation
        start_time = time.time()
        query_results = []
        failed_queries = []
        
        logger.info(f"🔍 Evaluating {len(queries)} queries...")
        
        # Process queries with progress bar
        for query in tqdm(queries, desc="Evaluating queries"):
            query_id = query['query_id']
            
            try:
                # Skip queries without relevance judgments
                if query_id not in qrels or not qrels[query_id]:
                    continue
                
                query_metrics = await self.evaluate_single_query(query, qrels[query_id])
                query_results.append(query_metrics)
                
            except Exception as e:
                logger.error(f"❌ Error evaluating query {query_id}: {e}")
                failed_queries.append({
                    'query_id': query_id,
                    'query_text': query['text'],
                    'error': str(e)
                })
                continue
        
        evaluation_time = time.time() - start_time
        
        if not query_results:
            logger.error("❌ No queries were successfully evaluated")
            return {}
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(query_results)
        
        # Add evaluation metadata
        evaluation_summary = {
            'total_queries_attempted': len(queries),
            'successful_evaluations': len(query_results),
            'failed_evaluations': len(failed_queries),
            'evaluation_time_seconds': evaluation_time,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'aggregated_metrics': aggregated_metrics,
            'query_results': query_results,
            'failed_queries': failed_queries,
            'services_status': services_status
        }
        
        # Store results
        self.evaluation_results = evaluation_summary
        
        return evaluation_summary
    
    def _calculate_aggregated_metrics(self, query_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics across all queries"""
        if not query_results:
            return {}
        
        # Get all metric names
        metric_names = [key for key in query_results[0].keys() 
                       if key.startswith(('precision_', 'recall_', 'ndcg_', 'average_precision', 'reciprocal_rank'))]
        
        # Calculate means
        aggregated = {}
        for metric in metric_names:
            values = [result.get(metric, 0.0) for result in query_results]
            aggregated[metric] = np.mean(values) if values else 0.0
        
        # Calculate special aggregated metrics
        aggregated['map'] = aggregated.get('average_precision', 0.0)  # MAP = mean AP
        aggregated['mrr'] = aggregated.get('reciprocal_rank', 0.0)    # MRR = mean RR
        
        # Calculate additional statistics
        processing_times = [r.get('processing_time_ms', 0) for r in query_results if 'processing_time_ms' in r]
        if processing_times:
            aggregated['avg_processing_time_ms'] = np.mean(processing_times)
            aggregated['median_processing_time_ms'] = np.median(processing_times)
            aggregated['max_processing_time_ms'] = np.max(processing_times)
        
        return aggregated
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("🎯 COMPLETE TF-IDF SYSTEM EVALUATION RESULTS")
        print("="*80)
        
        if not results:
            print("❌ No evaluation results available")
            return
        
        # Basic statistics
        print(f"📊 EVALUATION OVERVIEW:")
        print(f"   • Total queries attempted: {results['total_queries_attempted']}")
        print(f"   • Successful evaluations: {results['successful_evaluations']}")
        print(f"   • Failed evaluations: {results['failed_evaluations']}")
        print(f"   • Evaluation time: {results['evaluation_time_seconds']:.2f} seconds")
        print(f"   • Timestamp: {results['evaluation_timestamp']}")
        
        # Core IR metrics
        metrics = results['aggregated_metrics']
        print(f"\n📈 CORE IR METRICS:")
        print(f"   • Mean Average Precision (MAP): {metrics.get('map', 0.0):.4f}")
        print(f"   • Mean Reciprocal Rank (MRR):  {metrics.get('mrr', 0.0):.4f}")
        
        # Precision at K
        print(f"\n🎯 PRECISION@K:")
        for k in [1, 5, 10, 20]:
            p_k = metrics.get(f'precision_at_{k}', 0.0)
            print(f"   • P@{k:2d}: {p_k:.4f}")
        
        # Recall at K
        print(f"\n📊 RECALL@K:")
        for k in [1, 5, 10, 20]:
            r_k = metrics.get(f'recall_at_{k}', 0.0)
            print(f"   • R@{k:2d}: {r_k:.4f}")
        
        # NDCG at K
        print(f"\n⭐ NDCG@K:")
        for k in [1, 5, 10, 20]:
            ndcg_k = metrics.get(f'ndcg_at_{k}', 0.0)
            print(f"   • NDCG@{k:2d}: {ndcg_k:.4f}")
        
        # Performance metrics
        if 'avg_processing_time_ms' in metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   • Average query time: {metrics['avg_processing_time_ms']:.2f}ms")
            print(f"   • Median query time:  {metrics['median_processing_time_ms']:.2f}ms")
            print(f"   • Max query time:     {metrics['max_processing_time_ms']:.2f}ms")
        
        # Quality assessment
        print(f"\n📋 QUALITY ASSESSMENT:")
        map_score = metrics.get('map', 0.0)
        if map_score >= 0.3:
            print("   • MAP: Excellent performance ✅")
        elif map_score >= 0.2:
            print("   • MAP: Good performance 👍")
        elif map_score >= 0.1:
            print("   • MAP: Fair performance ⚠️")
        else:
            print("   • MAP: Poor performance ❌")
        
        mrr_score = metrics.get('mrr', 0.0)
        if mrr_score >= 0.5:
            print("   • MRR: Excellent first relevant result ranking ✅")
        elif mrr_score >= 0.3:
            print("   • MRR: Good first relevant result ranking 👍")
        elif mrr_score >= 0.2:
            print("   • MRR: Fair first relevant result ranking ⚠️")
        else:
            print("   • MRR: Poor first relevant result ranking ❌")
        
        print("="*80)
    
    def save_detailed_results(self, results: Dict[str, Any], output_file: str = None):
        """Save detailed evaluation results to files"""
        if not output_file:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = f"tfidf_evaluation_{timestamp}"
        
        # Save summary JSON
        summary_file = RESULTS_DIR / f"{output_file}_summary.json"
        with open(summary_file, 'w') as f:
            # Remove query_results for summary (too large)
            summary = {k: v for k, v in results.items() if k != 'query_results'}
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"💾 Summary saved to {summary_file}")
        
        # Save detailed query results as CSV
        if 'query_results' in results and results['query_results']:
            query_df = pd.DataFrame(results['query_results'])
            csv_file = RESULTS_DIR / f"{output_file}_query_results.csv"
            query_df.to_csv(csv_file, index=False)
            logger.info(f"💾 Query results saved to {csv_file}")
        
        # Save complete results as JSON
        complete_file = RESULTS_DIR / f"{output_file}_complete.json"
        with open(complete_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"💾 Complete results saved to {complete_file}")
        
        return {
            'summary_file': summary_file,
            'csv_file': csv_file if 'query_results' in results else None,
            'complete_file': complete_file
        }
    
    def create_evaluation_plots(self, results: Dict[str, Any], output_prefix: str = None):
        """Create evaluation plots and charts"""
        if not results.get('query_results'):
            logger.warning("⚠️ No query results available for plotting")
            return
        
        if not output_prefix:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_prefix = f"tfidf_eval_{timestamp}"
        
        query_df = pd.DataFrame(results['query_results'])
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TF-IDF System Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Precision@K distribution
        precision_cols = [col for col in query_df.columns if col.startswith('precision_at_')]
        if precision_cols:
            precision_data = query_df[precision_cols].melt(var_name='Metric', value_name='Score')
            sns.boxplot(data=precision_data, x='Metric', y='Score', ax=axes[0,0])
            axes[0,0].set_title('Precision@K Distribution')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Recall@K distribution  
        recall_cols = [col for col in query_df.columns if col.startswith('recall_at_')]
        if recall_cols:
            recall_data = query_df[recall_cols].melt(var_name='Metric', value_name='Score')
            sns.boxplot(data=recall_data, x='Metric', y='Score', ax=axes[0,1])
            axes[0,1].set_title('Recall@K Distribution')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. NDCG@K distribution
        ndcg_cols = [col for col in query_df.columns if col.startswith('ndcg_at_')]
        if ndcg_cols:
            ndcg_data = query_df[ndcg_cols].melt(var_name='Metric', value_name='Score')
            sns.boxplot(data=ndcg_data, x='Metric', y='Score', ax=axes[0,2])
            axes[0,2].set_title('NDCG@K Distribution')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Average Precision histogram
        if 'average_precision' in query_df.columns:
            query_df['average_precision'].hist(bins=30, ax=axes[1,0], alpha=0.7)
            axes[1,0].axvline(query_df['average_precision'].mean(), color='red', linestyle='--', 
                            label=f'Mean: {query_df["average_precision"].mean():.3f}')
            axes[1,0].set_title('Average Precision Distribution')
            axes[1,0].set_xlabel('Average Precision')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].legend()
        
        # 5. Reciprocal Rank histogram
        if 'reciprocal_rank' in query_df.columns:
            query_df['reciprocal_rank'].hist(bins=30, ax=axes[1,1], alpha=0.7)
            axes[1,1].axvline(query_df['reciprocal_rank'].mean(), color='red', linestyle='--',
                            label=f'Mean: {query_df["reciprocal_rank"].mean():.3f}')
            axes[1,1].set_title('Reciprocal Rank Distribution')
            axes[1,1].set_xlabel('Reciprocal Rank')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
        
        # 6. Processing time vs performance
        if 'processing_time_ms' in query_df.columns and 'average_precision' in query_df.columns:
            scatter = axes[1,2].scatter(query_df['processing_time_ms'], query_df['average_precision'], 
                                      alpha=0.6)
            axes[1,2].set_title('Processing Time vs Average Precision')
            axes[1,2].set_xlabel('Processing Time (ms)')
            axes[1,2].set_ylabel('Average Precision')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = RESULTS_DIR / f"{output_prefix}_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Evaluation plots saved to {plot_file}")
        plt.close()
        
        return plot_file

async def main():
    """Main evaluation function"""
    evaluator = CompleteTFIDFEvaluator()
    
    try:
        print("🚀 Complete TF-IDF System Evaluation")
        print("=" * 50)
        print("📋 This will evaluate the COMPLETE pipeline:")
        print("   • Query processing & text cleaning")
        print("   • TF-IDF vectorization")
        print("   • Cosine similarity computation")
        print("   • Ranking & result retrieval")
        print("📊 Metrics: MAP, MRR, P@K, R@K, NDCG@K")
        print("🔧 Make sure TF-IDF query service is running on port 8004")
        print("")
        
        # Run evaluation (limit to 50 queries for testing, remove limit for full evaluation)
        results = await evaluator.run_complete_evaluation(max_queries=None)  # Set to None for full evaluation
        
        if results:
            # Print summary
            evaluator.print_evaluation_summary(results)
            
            # Save results
            saved_files = evaluator.save_detailed_results(results)
            print(f"\n💾 Results saved:")
            for file_type, file_path in saved_files.items():
                if file_path:
                    print(f"   • {file_type}: {file_path}")
            
            # Create plots
            plot_file = evaluator.create_evaluation_plots(results)
            print(f"   • plots: {plot_file}")
            
            print(f"\n✅ Evaluation completed successfully!")
            
        else:
            print("❌ Evaluation failed - no results generated")
            
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    finally:
        await evaluator.close()

if __name__ == "__main__":
    print("🎯 Complete TF-IDF System Evaluation")
    print("📊 Evaluating end-to-end query processing pipeline")
    print("🔧 Ensure services are running:")
    print("   • TF-IDF Query Processor (port 8004)")
    print("   • TF-IDF Text Cleaning Service (port 8005)")
    print("")
    
    # Run evaluation
    asyncio.run(main())
