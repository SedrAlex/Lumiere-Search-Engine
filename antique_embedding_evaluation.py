#!/usr/bin/env python3
"""
ANTIQUE Embedding Model Evaluation
=================================

This script provides a comprehensive evaluation of the embedding system using:
1. ANTIQUE dataset queries (with proper cleaning as in embedding service)
2. ANTIQUE qrels file for relevance judgments  
3. MAP, MRR, NDCG, and additional IR metrics
4. Ensures antique_query_service and embedding_service use same text cleaning

Features:
- Query cleaning exactly as used in embedding services
- Multiple evaluation metrics
- Dataset verification
- Detailed logging and results
- Consistent text preprocessing across services
"""

import asyncio
import logging
import time
import ir_datasets
import numpy as np
import pandas as pd
import httpx
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ANTIQUE_QUERY_SERVICE_URL = "http://localhost:8005"  # ANTIQUE Query Service
EMBEDDING_SERVICE_URL = "http://localhost:8003"      # Embedding Service  
TEXT_CLEANING_SERVICE_URL = "http://localhost:8001"  # Shared Text Cleaning Service
RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

class AntiqueDatasetVerifier:
    """Verify we're using the correct ANTIQUE dataset files"""
    
    @staticmethod
    def verify_antique_dataset():
        """Verify ANTIQUE dataset is available and show dataset info"""
        logger.info("🔍 Verifying ANTIQUE dataset availability...")
        
        try:
            # Test dataset
            test_dataset = ir_datasets.load('antique/test')
            
            # Count queries
            test_queries = list(test_dataset.queries_iter())
            logger.info(f"✅ ANTIQUE test queries: {len(test_queries)}")
            
            # Count qrels
            test_qrels = list(test_dataset.qrels_iter())
            logger.info(f"✅ ANTIQUE test qrels: {len(test_qrels)}")
            
            # Count docs (sample)
            docs_sample = []
            for i, doc in enumerate(test_dataset.docs_iter()):
                docs_sample.append(doc)
                if i >= 100:  # Just sample first 100
                    break
            logger.info(f"✅ ANTIQUE documents (sample): {len(docs_sample)}")
            
            # Show example query
            example_query = test_queries[0]
            logger.info(f"📝 Example query: ID={example_query.query_id}, Text='{example_query.text[:100]}...'")
            
            # Show example qrel
            example_qrel = test_qrels[0]
            logger.info(f"📝 Example qrel: Query={example_qrel.query_id}, Doc={example_qrel.doc_id}, Rel={example_qrel.relevance}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying ANTIQUE dataset: {e}")
            return False

class QueryCleaner:
    """Query cleaning using shared text cleaning service (exactly as used in embedding services)"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.http_client.aclose()
    
    async def clean_query_shared(self, query: str) -> str:
        """Clean query using shared text cleaning service (exactly as in embedding services)"""
        try:
            response = await self.http_client.post(
                f"{TEXT_CLEANING_SERVICE_URL}/clean",
                json={
                    "text": query,
                    "remove_stopwords": False,  # Keep stopwords for queries (as in embedding service)
                    "apply_stemming": True,
                    "apply_lemmatization": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["cleaned_text"]
            
        except Exception as e:
            logger.warning(f"Shared cleaning failed for '{query}': {e}")
            return self._basic_clean(query)
    
    def _basic_clean(self, text: str) -> str:
        """Basic fallback cleaning (exactly as in embedding services)"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class IRMetrics:
    """Information Retrieval metrics calculator"""
    
    @staticmethod
    def average_precision(ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Average Precision (AP)"""
        if not relevant_docs or not ranked_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc in enumerate(ranked_docs):
            if doc in relevant_set:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs)
    
    @staticmethod
    def precision_at_k(ranked_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k <= 0 or not ranked_docs:
            return 0.0
        
        top_k = ranked_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_set)
        
        return relevant_in_top_k / min(k, len(top_k))
    
    @staticmethod
    def recall_at_k(ranked_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs or not ranked_docs:
            return 0.0
        
        top_k = ranked_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_set)
        
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def reciprocal_rank(ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Reciprocal Rank"""
        if not relevant_docs or not ranked_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(ranked_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(ranked_docs: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        """Calculate NDCG@K"""
        if k <= 0 or not ranked_docs:
            return 0.0
        
        # Get relevance scores for top-k documents
        top_k = ranked_docs[:k]
        y_true = []
        y_score = []
        
        for i, doc in enumerate(top_k):
            relevance = relevance_scores.get(doc, 0)
            y_true.append(relevance)
            y_score.append(len(top_k) - i)  # Higher rank = higher score
        
        if not any(y_true):
            return 0.0
        
        # Reshape for sklearn
        y_true = np.array([y_true])
        y_score = np.array([y_score])
        
        try:
            return ndcg_score(y_true, y_score, k=k)
        except:
            return 0.0

class AntiqueEmbeddingEvaluator:
    """Complete ANTIQUE embedding evaluation system"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.query_cleaner = QueryCleaner()
        self.metrics = IRMetrics()
        self.dataset_verified = False
    
    async def close(self):
        """Close HTTP clients"""
        await self.http_client.aclose()
        await self.query_cleaner.close()
    
    async def verify_services(self) -> Dict[str, bool]:
        """Verify all required services are running"""
        services = {
            "antique_query_service": False,
            "embedding_service": False,
            "text_cleaning_service": False
        }
        
        # Check ANTIQUE Query Service
        try:
            response = await self.http_client.get(f"{ANTIQUE_QUERY_SERVICE_URL}/docs")
            services["antique_query_service"] = response.status_code == 200
        except:
            pass
        
        # Check Embedding Service
        try:
            response = await self.http_client.get(f"{EMBEDDING_SERVICE_URL}/health")
            services["embedding_service"] = response.status_code == 200
        except:
            pass
        
        # Check Text Cleaning Service
        try:
            response = await self.http_client.get(f"{TEXT_CLEANING_SERVICE_URL}/health")
            services["text_cleaning_service"] = response.status_code == 200
        except:
            pass
        
        logger.info(f"🔧 Service Status:")
        logger.info(f"   - ANTIQUE Query Service (port 8005): {'✅' if services['antique_query_service'] else '❌'}")
        logger.info(f"   - Embedding Service (port 8003): {'✅' if services['embedding_service'] else '❌'}")
        logger.info(f"   - Text Cleaning Service (port 8001): {'✅' if services['text_cleaning_service'] else '❌'}")
        
        return services
    
    def load_antique_data(self) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
        """Load ANTIQUE test queries and qrels with verification"""
        logger.info("📚 Loading ANTIQUE test dataset with verification...")
        
        # Verify dataset first
        if not AntiqueDatasetVerifier.verify_antique_dataset():
            raise RuntimeError("ANTIQUE dataset verification failed")
        
        # Load test dataset
        dataset = ir_datasets.load('antique/test')
        
        # Load queries
        queries = []
        for query in dataset.queries_iter():
            queries.append({
                'query_id': query.query_id,
                'text': query.text
            })
        
        # Load qrels
        qrels = {}
        total_judgments = 0
        relevance_distribution = {}
        
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            total_judgments += 1
            
            # Track relevance distribution
            rel = qrel.relevance
            relevance_distribution[rel] = relevance_distribution.get(rel, 0) + 1
        
        logger.info(f"📊 Dataset loaded:")
        logger.info(f"   - Queries: {len(queries)}")
        logger.info(f"   - Qrels: {total_judgments} judgments for {len(qrels)} queries")
        logger.info(f"   - Relevance distribution: {relevance_distribution}")
        
        # Show sample data to verify
        if queries:
            sample_query = queries[0]
            logger.info(f"📝 Sample query: {sample_query['query_id']} = '{sample_query['text'][:50]}...'")
        
        if qrels:
            sample_query_id = list(qrels.keys())[0]
            sample_qrels = qrels[sample_query_id]
            logger.info(f"📝 Sample qrels for {sample_query_id}: {dict(list(sample_qrels.items())[:3])}")
        
        self.dataset_verified = True
        return queries, qrels
    
    async def query_antique_system(self, query_text: str, top_k: int = 1000) -> List[str]:
        """Query ANTIQUE embedding system (via antique_query_service)"""
        try:
            response = await self.http_client.post(
                f"{ANTIQUE_QUERY_SERVICE_URL}/query",
                json={
                    "query": query_text,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract document IDs in rank order
            return [doc["document_id"] for doc in result["results"]]
            
        except Exception as e:
            logger.error(f"❌ Error querying ANTIQUE system: {e}")
            return []
    
    async def query_embedding_system(self, query_text: str, top_k: int = 1000) -> List[str]:
        """Query generic embedding system (via embedding_service)"""
        try:
            response = await self.http_client.post(
                f"{EMBEDDING_SERVICE_URL}/search",
                json={
                    "query": query_text,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract document IDs in rank order
            return [doc["document_id"] for doc in result["results"]]
            
        except Exception as e:
            logger.error(f"❌ Error querying embedding system: {e}")
            return []
    
    async def evaluate_single_query(self, query: Dict[str, str], 
                                  query_qrels: Dict[str, int],
                                  use_antique_service: bool = True) -> Dict[str, Any]:
        """Evaluate a single query with comprehensive metrics"""
        query_id = query['query_id']
        query_text = query['text']
        
        # Clean query for logging (using shared service)
        cleaned_query = await self.query_cleaner.clean_query_shared(query_text)
        
        # Get search results
        if use_antique_service:
            retrieved_docs = await self.query_antique_system(query_text)
            system_name = "antique_query_service"
        else:
            retrieved_docs = await self.query_embedding_system(query_text)
            system_name = "embedding_service"
        
        if not retrieved_docs:
            logger.warning(f"⚠️ No results for query {query_id} using {system_name}")
            return self._empty_metrics(query_id, query_text, cleaned_query, system_name)
        
        # Get relevant documents
        relevant_docs = [doc_id for doc_id, rel in query_qrels.items() if rel > 0]
        
        if not relevant_docs:
            logger.warning(f"⚠️ No relevant documents for query {query_id}")
            return self._empty_metrics(query_id, query_text, cleaned_query, system_name)
        
        # Calculate all metrics
        metrics = {
            'query_id': query_id,
            'query_text': query_text,
            'cleaned_query': cleaned_query,
            'system_used': system_name,
            'num_retrieved': len(retrieved_docs),
            'num_relevant': len(relevant_docs),
            'num_relevant_retrieved': len(set(retrieved_docs) & set(relevant_docs))
        }
        
        # Core metrics
        metrics['average_precision'] = self.metrics.average_precision(retrieved_docs, relevant_docs)
        metrics['reciprocal_rank'] = self.metrics.reciprocal_rank(retrieved_docs, relevant_docs)
        
        # Precision and Recall at different K values
        for k in [1, 5, 10, 20, 50, 100]:
            metrics[f'precision_at_{k}'] = self.metrics.precision_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'recall_at_{k}'] = self.metrics.recall_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'ndcg_at_{k}'] = self.metrics.ndcg_at_k(retrieved_docs, query_qrels, k)
        
        return metrics
    
    def _empty_metrics(self, query_id: str, query_text: str, cleaned_query: str, system_name: str) -> Dict[str, Any]:
        """Return empty metrics for failed queries"""
        metrics = {
            'query_id': query_id,
            'query_text': query_text,
            'cleaned_query': cleaned_query,
            'system_used': system_name,
            'num_retrieved': 0,
            'num_relevant': 0,
            'num_relevant_retrieved': 0,
            'average_precision': 0.0,
            'reciprocal_rank': 0.0
        }
        
        for k in [1, 5, 10, 20, 50, 100]:
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        
        return metrics
    
    async def run_evaluation(self, max_queries: Optional[int] = None, 
                           use_antique_service: bool = True) -> Dict[str, Any]:
        """Run complete evaluation"""
        service_name = "ANTIQUE Query Service" if use_antique_service else "Embedding Service"
        logger.info(f"🚀 Starting ANTIQUE Embedding Evaluation using {service_name}...")
        
        # Verify services
        services = await self.verify_services()
        
        if use_antique_service and not services["antique_query_service"]:
            raise RuntimeError("❌ ANTIQUE Query Service not available. Start with: python start_antique_services.py")
        elif not use_antique_service and not services["embedding_service"]:
            raise RuntimeError("❌ Embedding Service not available. Start with: python backend/services/representation/embedding_service.py")
        
        if not services["text_cleaning_service"]:
            raise RuntimeError("❌ Text Cleaning Service not available. Start with: python backend/services/shared/text_cleaning_service.py")
        
        # Load dataset
        queries, qrels = self.load_antique_data()
        
        # Filter queries with qrels
        queries_with_qrels = [q for q in queries if q['query_id'] in qrels and qrels[q['query_id']]]
        logger.info(f"📊 {len(queries_with_qrels)} queries have relevance judgments")
        
        if max_queries:
            queries_with_qrels = queries_with_qrels[:max_queries]
            logger.info(f"📊 Limited to {max_queries} queries for testing")
        
        # Evaluate queries
        query_results = []
        failed_queries = []
        
        start_time = time.time()
        
        for query in tqdm(queries_with_qrels, desc="Evaluating queries"):
            query_id = query['query_id']
            
            try:
                metrics = await self.evaluate_single_query(query, qrels[query_id], use_antique_service)
                query_results.append(metrics)
                
                # Log progress
                if len(query_results) % 20 == 0:
                    logger.info(f"✅ Evaluated {len(query_results)} queries...")
                    
            except Exception as e:
                logger.error(f"❌ Failed to evaluate query {query_id}: {e}")
                failed_queries.append({'query_id': query_id, 'error': str(e)})
        
        evaluation_time = time.time() - start_time
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(query_results)
        
        return {
            'evaluation_info': {
                'dataset': 'antique/test',
                'system_used': service_name,
                'total_queries': len(queries),
                'queries_with_qrels': len(queries_with_qrels),
                'evaluated_queries': len(query_results),
                'failed_queries': len(failed_queries),
                'evaluation_time_seconds': evaluation_time,
                'dataset_verified': self.dataset_verified,
                'services_status': services,
                'text_cleaning_shared': services["text_cleaning_service"]
            },
            'aggregated_metrics': aggregated_metrics,
            'query_results': query_results,
            'failed_queries': failed_queries
        }
    
    def _calculate_aggregated_metrics(self, query_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics across all queries"""
        if not query_results:
            return {}
        
        aggregated = {
            'num_queries': len(query_results)
        }
        
        # Calculate means for all metrics
        metric_keys = [key for key in query_results[0].keys() 
                      if key not in ['query_id', 'query_text', 'cleaned_query', 'system_used'] and 
                      isinstance(query_results[0][key], (int, float))]
        
        for metric in metric_keys:
            values = [result[metric] for result in query_results]
            aggregated[metric] = np.mean(values)
        
        # Special names for key metrics
        aggregated['MAP'] = aggregated.get('average_precision', 0.0)
        aggregated['MRR'] = aggregated.get('reciprocal_rank', 0.0)
        
        return aggregated
    
    def print_results(self, results: Dict[str, Any]):
        """Print comprehensive evaluation results"""
        print("\n" + "="*80)
        print("🎯 ANTIQUE EMBEDDING EVALUATION RESULTS")
        print("="*80)
        
        info = results['evaluation_info']
        metrics = results['aggregated_metrics']
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   • Dataset: {info['dataset']}")
        print(f"   • System used: {info['system_used']}")
        print(f"   • Dataset verified: {'✅' if info['dataset_verified'] else '❌'}")
        print(f"   • Text cleaning shared: {'✅' if info['text_cleaning_shared'] else '❌'}")
        print(f"   • Total queries in dataset: {info['total_queries']}")
        print(f"   • Queries with qrels: {info['queries_with_qrels']}")
        print(f"   • Successfully evaluated: {info['evaluated_queries']}")
        print(f"   • Failed queries: {info['failed_queries']}")
        print(f"   • Evaluation time: {info['evaluation_time_seconds']:.2f} seconds")
        
        if metrics:
            print(f"\n📈 CORE METRICS:")
            print(f"   • MAP (Mean Average Precision): {metrics['MAP']:.4f}")
            print(f"   • MRR (Mean Reciprocal Rank):   {metrics['MRR']:.4f}")
            
            print(f"\n📊 PRECISION @ K:")
            for k in [1, 5, 10, 20]:
                if f'precision_at_{k}' in metrics:
                    print(f"   • P@{k:2d}: {metrics[f'precision_at_{k}']:.4f}")
            
            print(f"\n📊 RECALL @ K:")
            for k in [1, 5, 10, 20]:
                if f'recall_at_{k}' in metrics:
                    print(f"   • R@{k:2d}: {metrics[f'recall_at_{k}']:.4f}")
            
            print(f"\n📊 NDCG @ K:")
            for k in [1, 5, 10, 20]:
                if f'ndcg_at_{k}' in metrics:
                    print(f"   • NDCG@{k:2d}: {metrics[f'ndcg_at_{k}']:.4f}")
            
            # Performance interpretation
            print(f"\n🎯 PERFORMANCE INTERPRETATION:")
            map_score = metrics['MAP']
            if map_score >= 0.3:
                print("   MAP: 🏆 Excellent performance")
            elif map_score >= 0.2:
                print("   MAP: 👍 Good performance")
            elif map_score >= 0.1:
                print("   MAP: ⚠️  Fair performance")
            else:
                print("   MAP: ❌ Poor performance")
        
        print("="*80)
    
    def save_results(self, results: Dict[str, Any], filename_prefix: str = None) -> Path:
        """Save evaluation results"""
        if not filename_prefix:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            system_name = results['evaluation_info']['system_used'].lower().replace(' ', '_')
            filename_prefix = f"antique_embedding_eval_{system_name}_{timestamp}"
        
        # Save complete results
        json_file = RESULTS_DIR / f"{filename_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save query results as CSV
        if results['query_results']:
            df = pd.DataFrame(results['query_results'])
            csv_file = RESULTS_DIR / f"{filename_prefix}_queries.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"💾 Query results saved to {csv_file}")
        
        logger.info(f"💾 Complete results saved to {json_file}")
        return json_file

async def main():
    """Main evaluation function"""
    evaluator = AntiqueEmbeddingEvaluator()
    
    try:
        print("🎯 ANTIQUE Embedding Evaluation")
        print("=" * 50)
        print("📋 This evaluation:")
        print("   • Uses ANTIQUE test dataset (verified)")
        print("   • Applies shared text cleaning service")
        print("   • Tests both ANTIQUE Query Service and Embedding Service")
        print("   • Calculates MAP, MRR, P@K, R@K, NDCG@K")
        print("   • Ensures consistent text preprocessing")
        print("")
        
        # Choose which service to test
        print("🔧 Testing both services:")
        
        # Test 1: ANTIQUE Query Service
        print("\n1️⃣ Testing ANTIQUE Query Service...")
        results_antique = await evaluator.run_evaluation(max_queries=50, use_antique_service=True)
        evaluator.print_results(results_antique)
        output_file_antique = evaluator.save_results(results_antique)
        print(f"💾 ANTIQUE service results saved to: {output_file_antique}")
        
        # Test 2: Embedding Service (if available)
        print("\n2️⃣ Testing Embedding Service...")
        try:
            results_embedding = await evaluator.run_evaluation(max_queries=50, use_antique_service=False)
            evaluator.print_results(results_embedding)
            output_file_embedding = evaluator.save_results(results_embedding)
            print(f"💾 Embedding service results saved to: {output_file_embedding}")
            
            # Compare results
            print("\n🔄 COMPARISON:")
            antique_map = results_antique['aggregated_metrics']['MAP']
            embedding_map = results_embedding['aggregated_metrics']['MAP']
            print(f"   • ANTIQUE Query Service MAP: {antique_map:.4f}")
            print(f"   • Embedding Service MAP: {embedding_map:.4f}")
            
            if antique_map > embedding_map:
                print("   🏆 ANTIQUE Query Service performs better")
            elif embedding_map > antique_map:
                print("   🏆 Embedding Service performs better")
            else:
                print("   🤝 Both services perform equally")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not test Embedding Service: {e}")
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        print(f"\n❌ Error: {e}")
        raise
    finally:
        await evaluator.close()

if __name__ == "__main__":
    print("🎯 ANTIQUE Embedding Evaluation with Shared Text Cleaning")
    print("🔧 Make sure services are running:")
    print("   - ANTIQUE Query Service: python start_antique_services.py")
    print("   - Embedding Service (optional): python backend/services/representation/embedding_service.py")
    print("   - Text Cleaning Service: python backend/services/shared/text_cleaning_service.py")
    print("")
    
    asyncio.run(main())
