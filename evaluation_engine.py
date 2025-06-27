#!/usr/bin/env python3
"""
Information Retrieval Evaluation Engine
Calculates proper IR metrics: Precision@K, Recall, MAP, MRR, NDCG
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class IRMetrics:
    """Information Retrieval evaluation metrics calculator"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: List of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Precision@K score
        """
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: List of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Recall@K score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        return relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: List of relevant document IDs
            
        Returns:
            Average Precision score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        precision_scores = []
        relevant_found = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_scores.append(precision_at_i)
        
        if precision_scores:
            return sum(precision_scores) / len(relevant_docs)
        return 0.0
    
    @staticmethod
    def reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Reciprocal Rank (RR)
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevant_docs: List of relevant document IDs
            
        Returns:
            Reciprocal Rank score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def dcg_at_k(retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Cut-off rank
            
        Returns:
            DCG@K score
        """
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = relevance_scores.get(doc, 0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i + 1)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked)
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Cut-off rank
            
        Returns:
            NDCG@K score
        """
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        # Calculate DCG@K
        dcg = IRMetrics.dcg_at_k(retrieved_docs, relevance_scores, k)
        
        # Calculate IDCG@K (perfect ranking)
        sorted_docs = sorted(relevance_scores.keys(), 
                           key=lambda x: relevance_scores[x], 
                           reverse=True)
        idcg = IRMetrics.dcg_at_k(sorted_docs, relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

class SearchEvaluator:
    """Main evaluation engine for search results"""
    
    def __init__(self):
        self.metrics = IRMetrics()
    
    def evaluate_single_query(self, 
                            retrieved_docs: List[str], 
                            query_qrels: Dict[str, float],
                            k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate a single query's results
        
        Args:
            retrieved_docs: List of retrieved document IDs (ranked by score)
            query_qrels: Dictionary mapping doc_id to relevance score for this query
            k_values: List of K values for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get relevant documents (relevance > 0)
        relevant_docs = [doc_id for doc_id, rel in query_qrels.items() if rel > 0]
        
        results = {}
        
        # Calculate precision and recall at different K values
        for k in k_values:
            results[f'precision_at_{k}'] = self.metrics.precision_at_k(retrieved_docs, relevant_docs, k)
            results[f'recall_at_{k}'] = self.metrics.recall_at_k(retrieved_docs, relevant_docs, k)
            results[f'ndcg_at_{k}'] = self.metrics.ndcg_at_k(retrieved_docs, query_qrels, k)
        
        # Calculate other metrics
        results['average_precision'] = self.metrics.average_precision(retrieved_docs, relevant_docs)
        results['reciprocal_rank'] = self.metrics.reciprocal_rank(retrieved_docs, relevant_docs)
        
        return results
    
    def evaluate_system(self, 
                       search_results: Dict[str, List[str]], 
                       qrels: Dict[str, Dict[str, float]],
                       k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate the entire search system across all queries
        
        Args:
            search_results: Dictionary mapping query_id to list of retrieved doc_ids
            qrels: Dictionary mapping query_id to dict of doc_id -> relevance_score
            k_values: List of K values for evaluation
            
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        query_results = []
        
        for query_id in search_results:
            if query_id in qrels:
                retrieved = search_results[query_id]
                query_qrels = qrels[query_id]
                
                query_metrics = self.evaluate_single_query(retrieved, query_qrels, k_values)
                query_results.append(query_metrics)
        
        if not query_results:
            logger.warning("No queries with relevance judgments found")
            return {}
        
        # Aggregate results
        aggregated = {}
        for metric in query_results[0].keys():
            values = [result[metric] for result in query_results if metric in result]
            aggregated[metric] = sum(values) / len(values) if values else 0.0
        
        # Calculate additional aggregated metrics
        aggregated['map'] = aggregated.get('average_precision', 0.0)  # MAP = mean AP
        aggregated['mrr'] = aggregated.get('reciprocal_rank', 0.0)    # MRR = mean RR
        aggregated['num_queries'] = len(query_results)
        
        return aggregated

class ComparisonEvaluator:
    """Compare different search representations"""
    
    def __init__(self):
        self.evaluator = SearchEvaluator()
    
    def compare_representations(self, 
                              results_by_method: Dict[str, Dict[str, List[str]]], 
                              qrels: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple search methods
        
        Args:
            results_by_method: Dict mapping method_name -> query_id -> retrieved_docs
            qrels: Relevance judgments
            
        Returns:
            Dict mapping method_name -> evaluation_metrics
        """
        comparison = {}
        
        for method_name, method_results in results_by_method.items():
            logger.info(f"Evaluating {method_name}...")
            method_metrics = self.evaluator.evaluate_system(method_results, qrels)
            comparison[method_name] = method_metrics
        
        return comparison
    
    def statistical_significance_test(self, 
                                    method1_results: Dict[str, List[str]], 
                                    method2_results: Dict[str, List[str]], 
                                    qrels: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Perform statistical significance testing between two methods
        
        Args:
            method1_results: Results from first method
            method2_results: Results from second method
            qrels: Relevance judgments
            
        Returns:
            Dictionary with statistical test results
        """
        # Calculate per-query metrics for both methods
        method1_aps = []
        method2_aps = []
        
        common_queries = set(method1_results.keys()) & set(method2_results.keys())
        
        for query_id in common_queries:
            if query_id in qrels:
                # Calculate AP for method 1
                retrieved1 = method1_results[query_id]
                query_qrels = qrels[query_id]
                relevant_docs = [doc for doc, rel in query_qrels.items() if rel > 0]
                ap1 = IRMetrics.average_precision(retrieved1, relevant_docs)
                method1_aps.append(ap1)
                
                # Calculate AP for method 2
                retrieved2 = method2_results[query_id]
                ap2 = IRMetrics.average_precision(retrieved2, relevant_docs)
                method2_aps.append(ap2)
        
        if len(method1_aps) < 2:
            return {"error": "Not enough queries for statistical testing"}
        
        # Paired t-test (simplified)
        differences = [ap1 - ap2 for ap1, ap2 in zip(method1_aps, method2_aps)]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0:
            t_stat = 0
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(len(differences)))
        
        return {
            "mean_difference": mean_diff,
            "t_statistic": t_stat,
            "num_queries": len(differences),
            "method1_mean_ap": np.mean(method1_aps),
            "method2_mean_ap": np.mean(method2_aps)
        }

# Example usage
async def evaluate_search_engine_example():
    """Example of how to use the evaluation engine"""
    
    # Mock search results
    search_results = {
        "query_1": ["doc_1", "doc_3", "doc_5", "doc_2", "doc_4"],
        "query_2": ["doc_6", "doc_7", "doc_8", "doc_9", "doc_10"]
    }
    
    # Mock relevance judgments
    qrels = {
        "query_1": {"doc_1": 2, "doc_2": 1, "doc_3": 0, "doc_4": 1, "doc_5": 0},
        "query_2": {"doc_6": 1, "doc_7": 0, "doc_8": 2, "doc_9": 1, "doc_10": 0}
    }
    
    # Evaluate
    evaluator = SearchEvaluator()
    metrics = evaluator.evaluate_system(search_results, qrels)
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_search_engine_example())
