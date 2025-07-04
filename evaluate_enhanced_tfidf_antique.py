#!/usr/bin/env python3
"""
Enhanced TF-IDF Evaluation Script for ANTIQUE Dataset
Tests the enhanced TF-IDF service with advanced text processing and evaluates MAP performance.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our services
from services.shared.enhanced_tfidf_service_v2 import create_enhanced_tfidf_service_v2, create_conservative_tfidf_service
from services.evaluation.map_evaluation_service import create_map_evaluator
from services.data.antique_loader_service import AntiqueLoaderService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_antique_data():
    """Load ANTIQUE dataset."""
    logger.info("Loading ANTIQUE dataset...")
    
    loader = AntiqueLoaderService()
    dataset = loader.load_antique_data()
    
    if not dataset:
        raise RuntimeError("Failed to load ANTIQUE dataset")
    
    docs = dataset['documents']
    queries = dataset['queries']
    qrels = dataset['qrels']
    
    # Convert to lists for training
    doc_texts = [doc['text'] for doc in docs.values()]
    doc_ids = list(docs.keys())
    
    logger.info(f"Loaded {len(docs)} documents, {len(queries)} queries, {len(qrels)} qrels")
    
    return {
        'doc_texts': doc_texts,
        'doc_ids': doc_ids,
        'docs': docs,
        'queries': queries,
        'qrels': qrels
    }

def train_enhanced_tfidf_service(doc_texts, doc_ids, service_type='enhanced'):
    """Train enhanced TF-IDF service with different configurations."""
    
    if service_type == 'enhanced':
        logger.info("Training Enhanced TF-IDF Service with all features...")
        service = create_enhanced_tfidf_service_v2(
            enable_spell_check=True,
            enable_lemmatization=True,
            enable_stemming=True
        )
    elif service_type == 'conservative':
        logger.info("Training Conservative TF-IDF Service...")
        service = create_conservative_tfidf_service()
    else:
        raise ValueError(f"Unknown service type: {service_type}")
    
    # Train the service
    start_time = time.time()
    training_stats = service.train_enhanced_tfidf(
        documents=doc_texts,
        doc_ids=doc_ids,
        build_inverted_index=True
    )
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Print training statistics
    print(f"\n=== {service_type.title()} Training Statistics ===")
    for key, value in training_stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    return service, training_stats

def evaluate_service_performance(service, service_name, max_queries=None):
    """Evaluate service performance using MAP."""
    
    logger.info(f"Evaluating {service_name} performance...")
    
    # Create MAP evaluator
    evaluator = create_map_evaluator()
    
    # Run comprehensive evaluation
    start_time = time.time()
    evaluation_results = evaluator.evaluate_tfidf_service(
        tfidf_service=service,
        dataset_name='antique',
        max_queries=max_queries,
        k_eval=10
    )
    evaluation_time = time.time() - start_time
    
    evaluation_results['evaluation_time'] = evaluation_time
    evaluation_results['service_name'] = service_name
    
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    return evaluation_results

def compare_search_methods(service):
    """Compare different search methods within the service."""
    
    logger.info("Comparing search methods...")
    
    # Load a few test queries
    loader = AntiqueLoaderService()
    dataset = loader.load_antique_data()
    queries = dataset['queries']
    
    # Test queries
    test_queries = list(queries.items())[:10]
    
    results_comparison = {}
    
    for query_id, query_text in test_queries:
        logger.info(f"Testing query: {query_text[:50]}...")
        
        # Enhanced inverted index search
        enhanced_results = service.search_with_enhanced_inverted_index(query_text, top_k=5)
        
        # Full matrix search
        full_matrix_results = service.search_with_full_matrix(query_text, top_k=5)
        
        results_comparison[query_id] = {
            'query': query_text,
            'enhanced_inverted': [r['doc_id'] for r in enhanced_results],
            'full_matrix': [r['doc_id'] for r in full_matrix_results],
            'enhanced_scores': [r['score'] for r in enhanced_results],
            'full_matrix_scores': [r['score'] for r in full_matrix_results]
        }
    
    return results_comparison

def save_models_and_results(service, training_stats, evaluation_results, service_name):
    """Save trained models and evaluation results."""
    
    # Save models
    models_saved = service.save_enhanced_models(f"{service_name}_antique")
    
    if models_saved:
        logger.info(f"✓ {service_name} models saved successfully")
    else:
        logger.error(f"✗ Failed to save {service_name} models")
    
    # Save evaluation results
    results_file = f"evaluation_results_{service_name}_antique.json"
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'training_stats': training_stats,
                'evaluation_results': evaluation_results,
                'service_info': service.get_enhanced_service_info()
            }, f, indent=2, default=str)
        
        logger.info(f"✓ Evaluation results saved to {results_file}")
    except Exception as e:
        logger.error(f"✗ Failed to save evaluation results: {e}")

def print_evaluation_report(evaluation_results):
    """Print a formatted evaluation report."""
    
    print(f"\n{'='*80}")
    print(f"ENHANCED TF-IDF EVALUATION REPORT")
    print(f"{'='*80}")
    
    print(f"\nService: {evaluation_results['service_name']}")
    print(f"Dataset: {evaluation_results['dataset']}")
    print(f"Queries Evaluated: {evaluation_results['num_queries_evaluated']}")
    print(f"Evaluation Time: {evaluation_results.get('evaluation_time', 'N/A'):.2f}s")
    
    print(f"\n{'='*40}")
    print(f"MAIN RESULTS")
    print(f"{'='*40}")
    
    map_score = evaluation_results['MAP']
    print(f"MAP@{evaluation_results['cutoff_k']}: {map_score:.4f}")
    
    if map_score >= 0.4:
        print("✓ TARGET ACHIEVED (MAP ≥ 0.4)")
    else:
        print("✗ BELOW TARGET (MAP < 0.4)")
        print(f"  Gap to target: {0.4 - map_score:.4f}")
    
    print(f"\n{'='*40}")
    print(f"PRECISION AND RECALL")
    print(f"{'='*40}")
    
    pr_metrics = evaluation_results['precision_recall']
    for metric, value in pr_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\n{'='*40}")
    print(f"QUERY PERFORMANCE ANALYSIS")
    print(f"{'='*40}")
    
    query_perf = evaluation_results['query_performance']
    print(f"Mean AP: {query_perf['mean_ap']:.4f}")
    print(f"Median AP: {query_perf['median_ap']:.4f}")
    print(f"Std AP: {query_perf['std_ap']:.4f}")
    print(f"Queries above 0.4 AP: {query_perf['queries_above_0_4']} ({query_perf['percentage_above_0_4']:.1f}%)")
    
    print(f"\n{'='*40}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*40}")
    
    for rec in evaluation_results['recommendations']:
        print(f"• {rec}")
    
    print(f"\n{'='*80}")

def main():
    """Main evaluation function."""
    
    print("Enhanced TF-IDF Evaluation for ANTIQUE Dataset")
    print("=" * 60)
    
    try:
        # Load ANTIQUE data
        antique_data = load_antique_data()
        
        # Train and evaluate enhanced service
        print("\n1. Training Enhanced TF-IDF Service...")
        enhanced_service, enhanced_stats = train_enhanced_tfidf_service(
            antique_data['doc_texts'], 
            antique_data['doc_ids'], 
            'enhanced'
        )
        
        print("\n2. Evaluating Enhanced Service...")
        enhanced_results = evaluate_service_performance(
            enhanced_service, 
            "Enhanced TF-IDF",
            max_queries=100  # Limit for testing
        )
        
        # Print evaluation report
        print_evaluation_report(enhanced_results)
        
        # Save models and results
        print("\n3. Saving Models and Results...")
        save_models_and_results(
            enhanced_service, 
            enhanced_stats, 
            enhanced_results, 
            "enhanced"
        )
        
        # Compare search methods
        print("\n4. Comparing Search Methods...")
        search_comparison = compare_search_methods(enhanced_service)
        
        print("\nSample Search Method Comparison:")
        for i, (query_id, comparison) in enumerate(list(search_comparison.items())[:3]):
            print(f"\nQuery {i+1}: {comparison['query'][:60]}...")
            print(f"Enhanced Inverted: {comparison['enhanced_inverted'][:3]}")
            print(f"Full Matrix:       {comparison['full_matrix'][:3]}")
        
        # Optional: Train and compare with conservative service
        if input("\nTrain conservative service for comparison? (y/n): ").lower() == 'y':
            print("\n5. Training Conservative TF-IDF Service...")
            conservative_service, conservative_stats = train_enhanced_tfidf_service(
                antique_data['doc_texts'], 
                antique_data['doc_ids'], 
                'conservative'
            )
            
            print("\n6. Evaluating Conservative Service...")
            conservative_results = evaluate_service_performance(
                conservative_service, 
                "Conservative TF-IDF",
                max_queries=100
            )
            
            print_evaluation_report(conservative_results)
            
            # Compare results
            print(f"\n{'='*40}")
            print(f"SERVICE COMPARISON")
            print(f"{'='*40}")
            print(f"Enhanced MAP:     {enhanced_results['MAP']:.4f}")
            print(f"Conservative MAP: {conservative_results['MAP']:.4f}")
            print(f"Improvement:      {enhanced_results['MAP'] - conservative_results['MAP']:.4f}")
            
            save_models_and_results(
                conservative_service, 
                conservative_stats, 
                conservative_results, 
                "conservative"
            )
        
        print(f"\n{'='*60}")
        print("✓ Enhanced TF-IDF evaluation completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
