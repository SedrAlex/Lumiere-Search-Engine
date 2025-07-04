#!/usr/bin/env python3
"""
MAP (Mean Average Precision) Evaluation Service
Evaluates TF-IDF retrieval performance and optimizes for MAP > 0.4
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import ir_datasets
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MAPEvaluationService:
    """
    Service for evaluating retrieval performance using MAP and other IR metrics.
    Designed to achieve and measure MAP > 0.4 on ANTIQUE dataset.
    """
    
    def __init__(self):
        """Initialize the evaluation service."""
        self.dataset_qrels = {}
        self.dataset_queries = {}
        self.evaluation_cache = {}
        
    def load_antique_evaluation_data(self) -> bool:
        """
        Load ANTIQUE dataset queries and qrels for evaluation.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info("Loading ANTIQUE evaluation data...")
            
            # Load ANTIQUE dataset
            dataset = ir_datasets.load("antique")
            
            # Load queries
            queries = {}
            for query in dataset.queries_iter():
                queries[query.query_id] = query.text
            
            # Load qrels (relevance judgments)
            qrels = defaultdict(dict)
            for qrel in dataset.qrels_iter():
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            self.dataset_queries['antique'] = queries
            self.dataset_qrels['antique'] = dict(qrels)
            
            logger.info(f"Loaded {len(queries)} queries and {len(qrels)} qrels for ANTIQUE")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ANTIQUE evaluation data: {str(e)}")
            return False
    
    def calculate_average_precision(self, retrieved_docs: List[str], 
                                   relevant_docs: Dict[str, int],
                                   k: Optional[int] = None) -> float:
        """
        Calculate Average Precision (AP) for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Dictionary of relevant doc_id -> relevance_score
            k: Cut-off rank (if None, use all retrieved docs)
            
        Returns:
            Average Precision score
        """
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
    
    def calculate_map(self, search_results: Dict[str, List[str]], 
                     dataset_name: str = 'antique',
                     k: Optional[int] = None) -> Dict:
        """
        Calculate Mean Average Precision (MAP) across all queries.
        
        Args:
            search_results: Dictionary of query_id -> list of retrieved doc_ids
            dataset_name: Name of the dataset for qrels
            k: Cut-off rank (if None, use all retrieved docs)
            
        Returns:
            Dictionary with MAP and detailed metrics
        """
        if dataset_name not in self.dataset_qrels:
            raise ValueError(f"No qrels loaded for dataset: {dataset_name}")
        
        qrels = self.dataset_qrels[dataset_name]
        ap_scores = []
        query_metrics = {}
        
        for query_id, retrieved_docs in search_results.items():
            if query_id in qrels:
                relevant_docs = qrels[query_id]
                ap_score = self.calculate_average_precision(retrieved_docs, relevant_docs, k)
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
    
    def calculate_precision_recall_at_k(self, search_results: Dict[str, List[str]], 
                                       dataset_name: str = 'antique',
                                       k_values: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        Calculate Precision@K and Recall@K for multiple K values.
        
        Args:
            search_results: Dictionary of query_id -> list of retrieved doc_ids
            dataset_name: Name of the dataset for qrels
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with P@K and R@K metrics
        """
        if dataset_name not in self.dataset_qrels:
            raise ValueError(f"No qrels loaded for dataset: {dataset_name}")
        
        qrels = self.dataset_qrels[dataset_name]
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
    
    def evaluate_tfidf_service(self, tfidf_service, 
                              dataset_name: str = 'antique',
                              max_queries: Optional[int] = None,
                              k_eval: int = 10) -> Dict:
        """
        Comprehensive evaluation of TF-IDF service.
        
        Args:
            tfidf_service: TF-IDF service instance
            dataset_name: Dataset name for evaluation
            max_queries: Maximum number of queries to evaluate (for testing)
            k_eval: Cut-off rank for evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        if dataset_name not in self.dataset_queries:
            raise ValueError(f"No queries loaded for dataset: {dataset_name}")
        
        queries = self.dataset_queries[dataset_name]
        search_results = {}
        
        # Limit queries for testing if specified
        query_items = list(queries.items())
        if max_queries:
            query_items = query_items[:max_queries]
        
        logger.info(f"Evaluating {len(query_items)} queries...")
        
        # Run searches for all queries
        for query_id, query_text in query_items:
            try:
                # Try enhanced inverted index search first (if available)
                if hasattr(tfidf_service, 'search_with_enhanced_inverted_index'):
                    results = tfidf_service.search_with_enhanced_inverted_index(query_text, top_k=k_eval)
                # Fall back to regular inverted index search
                elif hasattr(tfidf_service, 'search_with_inverted_index'):
                    results = tfidf_service.search_with_inverted_index(query_text, top_k=k_eval)
                # Fall back to full matrix search
                elif hasattr(tfidf_service, 'search_with_full_matrix'):
                    results = tfidf_service.search_with_full_matrix(query_text, top_k=k_eval)
                # Generic search method
                else:
                    results = tfidf_service.search(query_text, top_k=k_eval)
                
                search_results[query_id] = [result['doc_id'] for result in results]
                
            except Exception as e:
                logger.warning(f"Error searching query {query_id}: {str(e)}")
                search_results[query_id] = []
        
        # Calculate evaluation metrics
        map_results = self.calculate_map(search_results, dataset_name, k_eval)
        precision_recall = self.calculate_precision_recall_at_k(
            search_results, dataset_name, [1, 5, 10, 20]
        )
        
        # Compile comprehensive results
        evaluation_results = {
            'dataset': dataset_name,
            'evaluation_method': 'inverted_index',
            'num_queries_evaluated': len(query_items),
            'cutoff_k': k_eval,
            'MAP': map_results['MAP'],
            'precision_recall': precision_recall,
            'detailed_map_results': map_results,
            'query_performance': self._analyze_query_performance(map_results['query_metrics']),
            'recommendations': self._generate_recommendations(map_results['MAP'], precision_recall)
        }
        
        # Cache results
        cache_key = f"{dataset_name}_{max_queries}_{k_eval}"
        self.evaluation_cache[cache_key] = evaluation_results
        
        return evaluation_results
    
    def compare_search_methods(self, tfidf_service,
                             dataset_name: str = 'antique',
                             max_queries: int = 50) -> Dict:
        """
        Compare inverted index vs full matrix search methods.
        
        Args:
            tfidf_service: TF-IDF service instance
            dataset_name: Dataset name for evaluation
            max_queries: Number of queries to test
            
        Returns:
            Comparison results
        """
        queries = self.dataset_queries[dataset_name]
        query_items = list(queries.items())[:max_queries]
        
        inverted_results = {}
        full_matrix_results = {}
        
        logger.info(f"Comparing search methods on {len(query_items)} queries...")
        
        for query_id, query_text in query_items:
            try:
                # Inverted index method
                inv_results = tfidf_service.search_with_inverted_index(query_text, top_k=10)
                inverted_results[query_id] = [result['doc_id'] for result in inv_results]
                
                # Full matrix method
                full_results = tfidf_service.search_full_matrix(query_text, top_k=10)
                full_matrix_results[query_id] = [result['doc_id'] for result in full_results]
                
            except Exception as e:
                logger.warning(f"Error in comparison for query {query_id}: {str(e)}")
                inverted_results[query_id] = []
                full_matrix_results[query_id] = []
        
        # Evaluate both methods
        inv_map = self.calculate_map(inverted_results, dataset_name, 10)
        full_map = self.calculate_map(full_matrix_results, dataset_name, 10)
        
        return {
            'inverted_index': {
                'MAP': inv_map['MAP'],
                'method': 'inverted_index_with_tfidf_fusion'
            },
            'full_matrix': {
                'MAP': full_map['MAP'],
                'method': 'cosine_similarity_full_matrix'
            },
            'improvement': inv_map['MAP'] - full_map['MAP'],
            'queries_tested': len(query_items)
        }
    
    def _analyze_query_performance(self, query_metrics: Dict) -> Dict:
        """Analyze query performance patterns."""
        ap_scores = [metrics['average_precision'] for metrics in query_metrics.values()]
        
        return {
            'mean_ap': np.mean(ap_scores),
            'median_ap': np.median(ap_scores),
            'std_ap': np.std(ap_scores),
            'min_ap': np.min(ap_scores),
            'max_ap': np.max(ap_scores),
            'queries_above_0_4': sum(1 for ap in ap_scores if ap > 0.4),
            'percentage_above_0_4': (sum(1 for ap in ap_scores if ap > 0.4) / len(ap_scores)) * 100
        }
    
    def _generate_recommendations(self, map_score: float, precision_recall: Dict) -> List[str]:
        """Generate recommendations for improving performance."""
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
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate a formatted evaluation report."""
        report = f"""
=== TF-IDF Retrieval Evaluation Report ===

Dataset: {results['dataset']}
Queries Evaluated: {results['num_queries_evaluated']}
Evaluation Method: {results['evaluation_method']}

=== Main Results ===
MAP@{results['cutoff_k']}: {results['MAP']:.4f}
{'✓ TARGET ACHIEVED' if results['MAP'] >= 0.4 else '✗ Below Target (0.4)'}

=== Precision and Recall ===
"""
        
        for metric, value in results['precision_recall'].items():
            report += f"{metric}: {value:.4f}\n"
        
        report += f"""
=== Query Performance Analysis ===
Mean AP: {results['query_performance']['mean_ap']:.4f}
Median AP: {results['query_performance']['median_ap']:.4f}
Std AP: {results['query_performance']['std_ap']:.4f}
Queries above 0.4 AP: {results['query_performance']['queries_above_0_4']} ({results['query_performance']['percentage_above_0_4']:.1f}%)

=== Recommendations ===
"""
        
        for rec in results['recommendations']:
            report += f"- {rec}\n"
        
        return report

# Factory function
def create_map_evaluator() -> MAPEvaluationService:
    """Create and initialize MAP evaluation service."""
    evaluator = MAPEvaluationService()
    evaluator.load_antique_evaluation_data()
    return evaluator

# Example usage
if __name__ == "__main__":
    # Test the evaluation service
    evaluator = create_map_evaluator()
    
    # Mock search results for testing
    mock_results = {
        "1": ["doc1", "doc2", "doc3"],
        "2": ["doc4", "doc5", "doc6"]
    }
    
    # Calculate MAP
    map_results = evaluator.calculate_map(mock_results, 'antique')
    print(f"MAP: {map_results['MAP']:.4f}")
