#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import time
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiqueEvaluator:
    """
    Comprehensive evaluation class for ANTIQUE hybrid search system.
    Calculates MAP, MRR, and Precision@K metrics.
    """
    
    def __init__(self, 
                 queries_file: str = "/content/drive/MyDrive/downloads/queries.tsv",
                 qrels_file: str = "/content/drive/MyDrive/downloads/qrels.tsv",
                 service_url: str = "http://localhost:8006"):
        """
        Initialize the evaluator.
        
        Args:
            queries_file: Path to queries TSV file
            qrels_file: Path to qrels TSV file 
            service_url: URL of the hybrid search service
        """
        self.queries_file = queries_file
        self.qrels_file = qrels_file
        self.service_url = service_url
        
        # Load queries and qrels
        self.queries = self._load_queries()
        self.qrels = self._load_qrels()
        
        logger.info(f"Loaded {len(self.queries)} queries and {len(self.qrels)} query-document pairs")
    
    def _load_queries(self) -> Dict[int, str]:
        """Load queries from TSV file."""
        queries = {}
        try:
            df = pd.read_csv(self.queries_file, sep='\t')
            for _, row in df.iterrows():
                queries[int(row['query_id'])] = row['text']
            logger.info(f"Loaded {len(queries)} queries")
            return queries
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            return {}
    
    def _load_qrels(self) -> Dict[int, Set[int]]:
        """Load query relevance judgments from TSV file."""
        qrels = defaultdict(set)
        try:
            df = pd.read_csv(self.qrels_file, sep='\t')
            for _, row in df.iterrows():
                query_id = int(row['query_id'])
                doc_id = int(row['doc_id'])
                relevance = int(row['relevance'])
                
                if relevance > 0:  # Only consider relevant documents
                    qrels[query_id].add(doc_id)
            
            logger.info(f"Loaded qrels for {len(qrels)} queries")
            return dict(qrels)
        except Exception as e:
            logger.error(f"Error loading qrels: {e}")
            return {}
    
    def _query_search_service(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Query the hybrid search service.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of search results
        """
        try:
            response = requests.post(
                f"{self.service_url}/query",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying search service: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    
    def calculate_precision_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved document IDs (in rank order)
            relevant_docs: Set of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Precision@K value
        """
        if k == 0 or len(retrieved_docs) == 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_in_top_k = len([doc for doc in top_k_docs if doc in relevant_docs])
        
        return relevant_in_top_k / k
    
    def calculate_average_precision(self, retrieved_docs: List[int], relevant_docs: Set[int]) -> float:
        """
        Calculate Average Precision (AP) for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs (in rank order)
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Average Precision value
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(relevant_docs)
    
    def calculate_reciprocal_rank(self, retrieved_docs: List[int], relevant_docs: Set[int]) -> float:
        """
        Calculate Reciprocal Rank (RR) for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs (in rank order)
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Reciprocal Rank value
        """
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
        """
        Calculate NDCG@K for binary relevance.
        
        Args:
            retrieved_docs: List of retrieved document IDs (in rank order)
            relevant_docs: Set of relevant document IDs
            k: Cut-off rank
            
        Returns:
            NDCG@K value
        """
        if k == 0 or len(retrieved_docs) == 0 or len(relevant_docs) == 0:
            return 0.0
        
        # Calculate DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG@K (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_docs))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_single_query(self, query_id: int, top_k: int = 100) -> Dict:
        """
        Evaluate a single query.
        
        Args:
            query_id: Query ID
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation metrics
        """
        if query_id not in self.queries:
            logger.warning(f"Query {query_id} not found in queries file")
            return {}
        
        if query_id not in self.qrels:
            logger.warning(f"Query {query_id} not found in qrels file")
            return {}
        
        query_text = self.queries[query_id]
        relevant_docs = self.qrels[query_id]
        
        # Query the search service
        results = self._query_search_service(query_text, top_k)
        
        if not results:
            logger.warning(f"No results returned for query {query_id}")
            return {
                'query_id': query_id,
                'query_text': query_text,
                'num_relevant': len(relevant_docs),
                'num_retrieved': 0,
                'ap': 0.0,
                'rr': 0.0,
                'p_at_1': 0.0,
                'p_at_5': 0.0,
                'p_at_10': 0.0,
                'p_at_20': 0.0,
                'ndcg_at_1': 0.0,
                'ndcg_at_5': 0.0,
                'ndcg_at_10': 0.0,
                'ndcg_at_20': 0.0
            }
        
        # Extract document IDs from results
        retrieved_docs = [int(result['doc_id']) for result in results]
        
        # Calculate metrics
        ap = self.calculate_average_precision(retrieved_docs, relevant_docs)
        rr = self.calculate_reciprocal_rank(retrieved_docs, relevant_docs)
        p_at_1 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 1)
        p_at_5 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 5)
        p_at_10 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 10)
        p_at_20 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 20)
        ndcg_at_1 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 1)
        ndcg_at_5 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 5)
        ndcg_at_10 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 10)
        ndcg_at_20 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 20)
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'num_relevant': len(relevant_docs),
            'num_retrieved': len(retrieved_docs),
            'ap': ap,
            'rr': rr,
            'p_at_1': p_at_1,
            'p_at_5': p_at_5,
            'p_at_10': p_at_10,
            'p_at_20': p_at_20,
            'ndcg_at_1': ndcg_at_1,
            'ndcg_at_5': ndcg_at_5,
            'ndcg_at_10': ndcg_at_10,
            'ndcg_at_20': ndcg_at_20
        }
    
    def evaluate_all_queries(self, max_queries: int = None, top_k: int = 100, 
                           parallel: bool = True, max_workers: int = 4) -> Dict:
        """
        Evaluate all queries and calculate aggregate metrics.
        
        Args:
            max_queries: Maximum number of queries to evaluate (None for all)
            top_k: Number of results to retrieve per query
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary with aggregate evaluation metrics
        """
        # Get query IDs to evaluate
        query_ids = list(self.queries.keys())
        if max_queries:
            query_ids = query_ids[:max_queries]
        
        # Filter to only queries that have qrels
        query_ids = [qid for qid in query_ids if qid in self.qrels]
        
        logger.info(f"Evaluating {len(query_ids)} queries")
        
        # Evaluate queries
        if parallel and len(query_ids) > 1:
            logger.info(f"Using parallel evaluation with {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_query = {
                    executor.submit(self.evaluate_single_query, qid, top_k): qid 
                    for qid in query_ids
                }
                
                results = []
                for future in tqdm(concurrent.futures.as_completed(future_to_query), 
                                 total=len(query_ids), desc="Evaluating queries"):
                    query_id = future_to_query[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating query {query_id}: {e}")
        else:
            logger.info("Using sequential evaluation")
            results = []
            for query_id in tqdm(query_ids, desc="Evaluating queries"):
                result = self.evaluate_single_query(query_id, top_k)
                if result:
                    results.append(result)
        
        # Calculate aggregate metrics
        if not results:
            logger.error("No evaluation results obtained")
            return {}
        
        # Calculate means
        map_score = np.mean([r['ap'] for r in results])
        mrr_score = np.mean([r['rr'] for r in results])
        mean_p_at_1 = np.mean([r['p_at_1'] for r in results])
        mean_p_at_5 = np.mean([r['p_at_5'] for r in results])
        mean_p_at_10 = np.mean([r['p_at_10'] for r in results])
        mean_p_at_20 = np.mean([r['p_at_20'] for r in results])
        mean_ndcg_at_1 = np.mean([r['ndcg_at_1'] for r in results])
        mean_ndcg_at_5 = np.mean([r['ndcg_at_5'] for r in results])
        mean_ndcg_at_10 = np.mean([r['ndcg_at_10'] for r in results])
        mean_ndcg_at_20 = np.mean([r['ndcg_at_20'] for r in results])
        
        # Calculate additional statistics
        total_relevant = sum([r['num_relevant'] for r in results])
        total_retrieved = sum([r['num_retrieved'] for r in results])
        queries_with_results = len([r for r in results if r['num_retrieved'] > 0])
        
        aggregate_results = {
            'evaluation_summary': {
                'total_queries_evaluated': len(results),
                'queries_with_results': queries_with_results,
                'total_relevant_docs': total_relevant,
                'total_retrieved_docs': total_retrieved,
                'avg_relevant_per_query': total_relevant / len(results),
                'avg_retrieved_per_query': total_retrieved / len(results)
            },
            'aggregate_metrics': {
                'MAP': map_score,
                'MRR': mrr_score,
                'Mean_P@1': mean_p_at_1,
                'Mean_P@5': mean_p_at_5,
                'Mean_P@10': mean_p_at_10,
                'Mean_P@20': mean_p_at_20,
                'Mean_NDCG@1': mean_ndcg_at_1,
                'Mean_NDCG@5': mean_ndcg_at_5,
                'Mean_NDCG@10': mean_ndcg_at_10,
                'Mean_NDCG@20': mean_ndcg_at_20
            },
            'per_query_results': results
        }
        
        return aggregate_results
    
    def save_results(self, results: Dict, output_file: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results: Dict):
        """
        Print a summary of evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        if not results:
            print("No results to display")
            return
        
        summary = results['evaluation_summary']
        metrics = results['aggregate_metrics']
        
        print("\n" + "="*60)
        print("ANTIQUE HYBRID SEARCH EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nEvaluation Summary:")
        print(f"  Total Queries Evaluated: {summary['total_queries_evaluated']}")
        print(f"  Queries with Results: {summary['queries_with_results']}")
        print(f"  Total Relevant Documents: {summary['total_relevant_docs']}")
        print(f"  Total Retrieved Documents: {summary['total_retrieved_docs']}")
        print(f"  Avg Relevant per Query: {summary['avg_relevant_per_query']:.2f}")
        print(f"  Avg Retrieved per Query: {summary['avg_retrieved_per_query']:.2f}")
        
        print(f"\nAggregate Metrics:")
        print(f"  MAP (Mean Average Precision): {metrics['MAP']:.4f}")
        print(f"  MRR (Mean Reciprocal Rank): {metrics['MRR']:.4f}")
        print(f"  Mean P@1: {metrics['Mean_P@1']:.4f}")
        print(f"  Mean P@5: {metrics['Mean_P@5']:.4f}")
        print(f"  Mean P@10: {metrics['Mean_P@10']:.4f}")
        print(f"  Mean P@20: {metrics['Mean_P@20']:.4f}")
        print(f"  Mean NDCG@1: {metrics['Mean_NDCG@1']:.4f}")
        print(f"  Mean NDCG@5: {metrics['Mean_NDCG@5']:.4f}")
        print(f"  Mean NDCG@10: {metrics['Mean_NDCG@10']:.4f}")
        print(f"  Mean NDCG@20: {metrics['Mean_NDCG@20']:.4f}")
        
        print("\n" + "="*60)

def main():
    """Main evaluation function."""
    
    # Initialize evaluator
    evaluator = AntiqueEvaluator()
    
    # Check if service is running
    try:
        response = requests.get(f"{evaluator.service_url}/docs")
        if response.status_code != 200:
            logger.error(f"Search service not available at {evaluator.service_url}")
            return
    except:
        logger.error(f"Search service not available at {evaluator.service_url}")
        return
    
    # Run evaluation
    print("Starting evaluation...")
    start_time = time.time()
    
    # Evaluate first 50 queries for testing (remove max_queries for full evaluation)
    results = evaluator.evaluate_all_queries(
        max_queries=50,  # Remove this line for full evaluation
        top_k=100,
        parallel=True,
        max_workers=4
    )
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    if results:
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        output_file = f"antique_evaluation_results_{int(time.time())}.json"
        evaluator.save_results(results, output_file)
        
        print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
        print(f"Results saved to: {output_file}")
        
        # Print top 10 best and worst performing queries
        per_query = results['per_query_results']
        if per_query:
            print(f"\nTop 10 Best Performing Queries (by AP):")
            best_queries = sorted(per_query, key=lambda x: x['ap'], reverse=True)[:10]
            for i, q in enumerate(best_queries, 1):
                print(f"  {i}. Query {q['query_id']}: AP={q['ap']:.4f}, RR={q['rr']:.4f}")
                print(f"     \"{q['query_text'][:80]}{'...' if len(q['query_text']) > 80 else ''}\"")
            
            print(f"\nTop 10 Worst Performing Queries (by AP):")
            worst_queries = sorted(per_query, key=lambda x: x['ap'])[:10]
            for i, q in enumerate(worst_queries, 1):
                print(f"  {i}. Query {q['query_id']}: AP={q['ap']:.4f}, RR={q['rr']:.4f}")
                print(f"     \"{q['query_text'][:80]}{'...' if len(q['query_text']) > 80 else ''}\"")
    else:
        print("Evaluation failed - no results obtained")

if __name__ == "__main__":
    main()
