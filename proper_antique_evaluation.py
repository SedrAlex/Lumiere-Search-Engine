#!/usr/bin/env python3
"""
Proper ANTIQUE Dataset Evaluation
=================================

This script correctly evaluates TF-IDF system using:
1. ANTIQUE test queries (raw, not pre-cleaned)
2. ANTIQUE qrels file for relevance judgments
3. Proper MAP calculation
4. Standard IR evaluation metrics

Key questions addressed:
- Should queries be cleaned before evaluation? (We'll test both)
- Are we using the correct qrels format?
- Is MAP calculation correct?
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TFIDF_QUERY_SERVICE_URL = "http://localhost:8004"
RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

class ProperAntiqueEvaluator:
    """Proper ANTIQUE evaluation with correct qrels handling"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=120.0)
        
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    def load_antique_test_data(self) -> Tuple[List[Dict], Dict[str, Dict[str, int]], List[Dict]]:
        """
        Load ANTIQUE test data correctly
        Returns: (queries, qrels, documents_info)
        """
        logger.info("📚 Loading ANTIQUE test dataset...")
        
        # Load the test split of ANTIQUE
        dataset = ir_datasets.load('antique/test')
        
        # Load queries (should NOT be pre-cleaned for evaluation)
        queries = []
        for query in dataset.queries_iter():
            queries.append({
                'query_id': query.query_id,
                'text': query.text  # Original query text
            })
        
        # Load qrels (relevance judgments)
        qrels = {}
        total_judgments = 0
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            total_judgments += 1
        
        # Get some document info for analysis
        documents_info = []
        for doc in dataset.docs_iter():
            documents_info.append({
                'doc_id': doc.doc_id,
                'text_length': len(doc.text),
                'title_length': len(doc.title) if hasattr(doc, 'title') else 0
            })
            # Only get first 1000 for analysis
            if len(documents_info) >= 1000:
                break
        
        logger.info(f"📊 Loaded {len(queries)} test queries")
        logger.info(f"📊 Loaded {total_judgments} relevance judgments for {len(qrels)} queries")
        logger.info(f"📊 Analyzed {len(documents_info)} documents")
        
        # Show qrels statistics
        self._analyze_qrels(qrels)
        
        return queries, qrels, documents_info
    
    def _analyze_qrels(self, qrels: Dict[str, Dict[str, int]]):
        """Analyze qrels structure to ensure we understand it correctly"""
        logger.info("🔍 Analyzing qrels structure...")
        
        relevance_levels = {}
        queries_with_relevant = 0
        total_relevant_docs = 0
        
        for query_id, judgments in qrels.items():
            has_relevant = False
            for doc_id, relevance in judgments.items():
                if relevance not in relevance_levels:
                    relevance_levels[relevance] = 0
                relevance_levels[relevance] += 1
                
                if relevance > 0:
                    has_relevant = True
                    total_relevant_docs += 1
            
            if has_relevant:
                queries_with_relevant += 1
        
        logger.info(f"📈 Relevance level distribution:")
        for level, count in sorted(relevance_levels.items()):
            logger.info(f"   Level {level}: {count} judgments")
        
        logger.info(f"📊 {queries_with_relevant} queries have relevant documents")
        logger.info(f"📊 {total_relevant_docs} total relevant documents")
        
        # Show some example qrels
        example_query = list(qrels.keys())[0]
        logger.info(f"📝 Example qrels for query {example_query}:")
        for doc_id, rel in list(qrels[example_query].items())[:5]:
            logger.info(f"   {doc_id}: {rel}")
    
    async def query_tfidf_system(self, query_text: str, use_cleaning: bool = True, top_k: int = 1000) -> List[str]:
        """Query the TF-IDF system and return ranked document IDs"""
        try:
            response = await self.http_client.post(
                f"{TFIDF_QUERY_SERVICE_URL}/search",
                json={
                    "query": query_text,
                    "top_k": top_k,
                    "use_enhanced_cleaning": use_cleaning
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract document IDs in rank order
            doc_ids = [doc["doc_id"] for doc in result["results"]]
            return doc_ids
            
        except Exception as e:
            logger.error(f"❌ Error querying system: {e}")
            return []
    
    def calculate_average_precision(self, ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Average Precision (AP) correctly
        AP = (1/R) * Σ(P(k) * rel(k)) for k=1 to n
        where R is total number of relevant documents
        """
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
        
        # Average Precision = sum of precisions / total relevant documents
        return precision_sum / len(relevant_docs)
    
    def calculate_precision_at_k(self, ranked_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k <= 0 or not ranked_docs:
            return 0.0
        
        top_k = ranked_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_set)
        
        return relevant_in_top_k / min(k, len(top_k))
    
    def calculate_recall_at_k(self, ranked_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs or not ranked_docs:
            return 0.0
        
        top_k = ranked_docs[:k]
        relevant_set = set(relevant_docs)
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_set)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_reciprocal_rank(self, ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Reciprocal Rank (first relevant document position)"""
        if not relevant_docs or not ranked_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(ranked_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    async def evaluate_with_and_without_cleaning(self, queries: List[Dict], qrels: Dict[str, Dict[str, int]], 
                                               max_queries: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate system both with and without query cleaning to see the difference
        """
        logger.info("🧪 Evaluating with AND without query cleaning...")
        
        if max_queries:
            queries = queries[:max_queries]
            logger.info(f"📊 Limited to {max_queries} queries for testing")
        
        # Filter to queries with relevance judgments
        queries_with_qrels = [q for q in queries if q['query_id'] in qrels and qrels[q['query_id']]]
        logger.info(f"📊 {len(queries_with_qrels)} queries have relevance judgments")
        
        results = {
            'with_cleaning': {'query_results': [], 'failed': []},
            'without_cleaning': {'query_results': [], 'failed': []}
        }
        
        for query in tqdm(queries_with_qrels, desc="Evaluating queries"):
            query_id = query['query_id']
            query_text = query['text']
            query_qrels = qrels[query_id]
            
            # Get relevant documents (relevance > 0)
            relevant_docs = [doc_id for doc_id, rel in query_qrels.items() if rel > 0]
            
            if not relevant_docs:
                continue
            
            # Test WITH cleaning
            try:
                ranked_docs_clean = await self.query_tfidf_system(query_text, use_cleaning=True)
                if ranked_docs_clean:
                    metrics_clean = self._calculate_query_metrics(ranked_docs_clean, relevant_docs, query_id, query_text, "with_cleaning")
                    results['with_cleaning']['query_results'].append(metrics_clean)
                else:
                    results['with_cleaning']['failed'].append({'query_id': query_id, 'reason': 'no_results'})
            except Exception as e:
                results['with_cleaning']['failed'].append({'query_id': query_id, 'reason': str(e)})
            
            # Test WITHOUT cleaning
            try:
                ranked_docs_raw = await self.query_tfidf_system(query_text, use_cleaning=False)
                if ranked_docs_raw:
                    metrics_raw = self._calculate_query_metrics(ranked_docs_raw, relevant_docs, query_id, query_text, "without_cleaning")
                    results['without_cleaning']['query_results'].append(metrics_raw)
                else:
                    results['without_cleaning']['failed'].append({'query_id': query_id, 'reason': 'no_results'})
            except Exception as e:
                results['without_cleaning']['failed'].append({'query_id': query_id, 'reason': str(e)})
        
        # Calculate aggregated metrics
        for condition in ['with_cleaning', 'without_cleaning']:
            query_results = results[condition]['query_results']
            if query_results:
                results[condition]['aggregated'] = self._calculate_aggregated_metrics(query_results)
        
        return results
    
    def _calculate_query_metrics(self, ranked_docs: List[str], relevant_docs: List[str], 
                                query_id: str, query_text: str, condition: str) -> Dict[str, Any]:
        """Calculate all metrics for a single query"""
        metrics = {
            'query_id': query_id,
            'query_text': query_text,
            'condition': condition,
            'num_retrieved': len(ranked_docs),
            'num_relevant': len(relevant_docs),
            'num_relevant_retrieved': len(set(ranked_docs) & set(relevant_docs))
        }
        
        # Calculate AP
        metrics['average_precision'] = self.calculate_average_precision(ranked_docs, relevant_docs)
        
        # Calculate RR
        metrics['reciprocal_rank'] = self.calculate_reciprocal_rank(ranked_docs, relevant_docs)
        
        # Calculate Precision@K and Recall@K
        for k in [1, 5, 10, 20]:
            metrics[f'precision_at_{k}'] = self.calculate_precision_at_k(ranked_docs, relevant_docs, k)
            metrics[f'recall_at_{k}'] = self.calculate_recall_at_k(ranked_docs, relevant_docs, k)
        
        return metrics
    
    def _calculate_aggregated_metrics(self, query_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics across all queries"""
        if not query_results:
            return {}
        
        # Calculate MAP (Mean Average Precision)
        aps = [r['average_precision'] for r in query_results]
        map_score = np.mean(aps)
        
        # Calculate MRR (Mean Reciprocal Rank)
        rrs = [r['reciprocal_rank'] for r in query_results]
        mrr_score = np.mean(rrs)
        
        # Calculate mean metrics for P@K and R@K
        aggregated = {
            'MAP': map_score,
            'MRR': mrr_score,
            'num_queries': len(query_results)
        }
        
        for k in [1, 5, 10, 20]:
            p_values = [r[f'precision_at_{k}'] for r in query_results]
            r_values = [r[f'recall_at_{k}'] for r in query_results]
            aggregated[f'P@{k}'] = np.mean(p_values)
            aggregated[f'R@{k}'] = np.mean(r_values)
        
        return aggregated
    
    def print_comparison_results(self, results: Dict[str, Any]):
        """Print detailed comparison results"""
        print("\n" + "="*80)
        print("🎯 PROPER ANTIQUE EVALUATION RESULTS")
        print("="*80)
        
        for condition in ['with_cleaning', 'without_cleaning']:
            if 'aggregated' in results[condition]:
                metrics = results[condition]['aggregated']
                failed_count = len(results[condition]['failed'])
                
                print(f"\n📊 RESULTS {condition.upper().replace('_', ' ')}:")
                print(f"   • Queries evaluated: {metrics['num_queries']}")
                print(f"   • Failed queries: {failed_count}")
                print(f"   • MAP (Mean Average Precision): {metrics['MAP']:.4f}")
                print(f"   • MRR (Mean Reciprocal Rank): {metrics['MRR']:.4f}")
                
                print(f"   • Precision@1:  {metrics['P@1']:.4f}")
                print(f"   • Precision@5:  {metrics['P@5']:.4f}")
                print(f"   • Precision@10: {metrics['P@10']:.4f}")
                print(f"   • Precision@20: {metrics['P@20']:.4f}")
                
                print(f"   • Recall@1:  {metrics['R@1']:.4f}")
                print(f"   • Recall@5:  {metrics['R@5']:.4f}")
                print(f"   • Recall@10: {metrics['R@10']:.4f}")
                print(f"   • Recall@20: {metrics['R@20']:.4f}")
        
        # Compare the two approaches
        if ('aggregated' in results['with_cleaning'] and 
            'aggregated' in results['without_cleaning']):
            
            clean_map = results['with_cleaning']['aggregated']['MAP']
            raw_map = results['without_cleaning']['aggregated']['MAP']
            
            print(f"\n🔍 COMPARISON:")
            print(f"   • MAP with cleaning:    {clean_map:.4f}")
            print(f"   • MAP without cleaning: {raw_map:.4f}")
            print(f"   • Difference:           {clean_map - raw_map:+.4f}")
            
            if clean_map > raw_map:
                print(f"   ✅ Query cleaning IMPROVES performance")
            elif raw_map > clean_map:
                print(f"   ⚠️  Raw queries perform BETTER")
            else:
                print(f"   ➡️  No significant difference")
        
        print("="*80)
    
    def save_evaluation_results(self, results: Dict[str, Any], output_prefix: str = None):
        """Save detailed evaluation results"""
        if not output_prefix:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_prefix = f"proper_antique_eval_{timestamp}"
        
        # Save complete results
        output_file = RESULTS_DIR / f"{output_prefix}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_file}")
        
        # Save CSV for analysis
        all_query_results = []
        for condition in ['with_cleaning', 'without_cleaning']:
            for result in results[condition]['query_results']:
                all_query_results.append(result)
        
        if all_query_results:
            df = pd.DataFrame(all_query_results)
            csv_file = RESULTS_DIR / f"{output_prefix}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"💾 Query results saved to {csv_file}")
        
        return output_file

async def main():
    """Main evaluation function"""
    evaluator = ProperAntiqueEvaluator()
    
    try:
        print("🎯 Proper ANTIQUE Dataset Evaluation")
        print("=" * 50)
        print("📋 This evaluation:")
        print("   • Uses original ANTIQUE test queries (not pre-cleaned)")
        print("   • Uses correct ANTIQUE qrels file")
        print("   • Calculates MAP properly")
        print("   • Tests both cleaned and raw query processing")
        print("🔧 Make sure TF-IDF query service is running on port 8004")
        print("")
        
        # Load ANTIQUE data
        queries, qrels, docs_info = evaluator.load_antique_test_data()
        
        # Run evaluation with limited queries for testing (remove limit for full evaluation)
        results = await evaluator.evaluate_with_and_without_cleaning(
            queries, qrels, max_queries=10  # Set to None for full evaluation
        )
        
        # Print results
        evaluator.print_comparison_results(results)
        
        # Save results
        output_file = evaluator.save_evaluation_results(results)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        print(f"\n✅ Proper evaluation completed!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    finally:
        await evaluator.close()

if __name__ == "__main__":
    print("🎯 Proper ANTIQUE Evaluation")
    print("📊 Testing query cleaning vs raw queries")
    print("🔧 Ensure TF-IDF Query Processor is running on port 8004")
    print("")
    
    asyncio.run(main())
