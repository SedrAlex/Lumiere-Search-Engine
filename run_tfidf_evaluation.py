#!/usr/bin/env python3
"""
Complete TF-IDF Evaluation Script - Day 1 Implementation
Integrates trained models, TF-IDF service, and MAP evaluation for achieving MAP > 0.4
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

# Add services to path
sys.path.append(str(Path(__file__).parent))

# Import our services
from services.preprocessing.text_preprocessing_service import TextPreprocessingService
from services.representation.optimized_tfidf_service import OptimizedTFIDFService
from services.evaluation.map_evaluation_service import MAPEvaluationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_models_directory():
    """Ensure models directory exists."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

def check_required_model_files(models_dir: Path, dataset_name: str = "antique") -> bool:
    """Check if all required model files are present."""
    required_files = [
        f"tfidf_vectorizer_{dataset_name}.joblib",
        f"tfidf_matrix_{dataset_name}.joblib", 
        f"inverted_index_{dataset_name}.pkl",
        f"doc_mappings_{dataset_name}.json"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = models_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Missing required model files: {missing_files}")
        logger.error("Please run the Colab training notebook first to generate these files.")
        return False
    
    return True

def run_comprehensive_evaluation(max_queries: int = 100) -> dict:
    """
    Run comprehensive TF-IDF evaluation for MAP > 0.4 target.
    
    Args:
        max_queries: Maximum number of queries to evaluate (for testing)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=== Starting Comprehensive TF-IDF Evaluation ===")
    
    # Setup
    models_dir = setup_models_directory()
    
    # Check if model files exist
    if not check_required_model_files(models_dir):
        return {"error": "Required model files not found"}
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        
        # TF-IDF Service
        logger.info("Loading TF-IDF service...")
        tfidf_service = OptimizedTFIDFService(str(models_dir))
        
        if not tfidf_service.load_models("antique"):
            raise RuntimeError("Failed to load TF-IDF models")
        
        # MAP Evaluation Service
        logger.info("Initializing MAP evaluation service...")
        evaluator = MAPEvaluationService()
        
        if not evaluator.load_antique_evaluation_data():
            raise RuntimeError("Failed to load ANTIQUE evaluation data")
        
        # Get service statistics
        service_stats = tfidf_service.get_service_statistics()
        logger.info(f"Service loaded - Documents: {service_stats['total_documents']}, "
                   f"Vocabulary: {service_stats['vocabulary_size']}")
        
        # Run evaluation
        logger.info(f"Running evaluation on up to {max_queries} queries...")
        evaluation_results = evaluator.evaluate_tfidf_service(
            tfidf_service, 
            dataset_name='antique',
            max_queries=max_queries,
            k_eval=10
        )
        
        # Compare search methods
        logger.info("Comparing search methods...")
        comparison_results = evaluator.compare_search_methods(
            tfidf_service,
            dataset_name='antique', 
            max_queries=min(50, max_queries)
        )
        
        # Test sample queries
        logger.info("Testing sample queries...")
        sample_queries = [
            "antique furniture restoration techniques",
            "vintage jewelry appraisal value",
            "old coins identification guide",
            "ancient pottery dating methods",
            "collectible stamps price catalog"
        ]
        
        sample_results = {}
        for query in sample_queries:
            results = tfidf_service.search_with_inverted_index(query, top_k=5)
            sample_results[query] = {
                'num_results': len(results),
                'top_scores': [r['score'] for r in results[:3]],
                'top_docs': [r['doc_id'] for r in results[:3]]
            }
        
        # Compile comprehensive results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_summary': {
                'MAP': evaluation_results['MAP'],
                'target_achieved': evaluation_results['MAP'] >= 0.4,
                'queries_evaluated': evaluation_results['num_queries_evaluated'],
                'precision_recall': evaluation_results['precision_recall'],
                'query_performance': evaluation_results['query_performance']
            },
            'method_comparison': comparison_results,
            'sample_query_tests': sample_results,
            'service_statistics': service_stats,
            'detailed_evaluation': evaluation_results,
            'recommendations': evaluation_results['recommendations']
        }
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(evaluation_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"tfidf_evaluation_results_{timestamp}.json"
        report_file = f"tfidf_evaluation_report_{timestamp}.txt"
        
        # Save JSON results
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Save text report
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("TF-IDF EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"MAP@10: {final_results['evaluation_summary']['MAP']:.4f}")
        print(f"Target (0.4): {'✓ ACHIEVED' if final_results['evaluation_summary']['target_achieved'] else '✗ NOT ACHIEVED'}")
        print(f"Queries evaluated: {final_results['evaluation_summary']['queries_evaluated']}")
        print(f"P@1: {final_results['evaluation_summary']['precision_recall']['P@1']:.4f}")
        print(f"P@5: {final_results['evaluation_summary']['precision_recall']['P@5']:.4f}")
        print(f"P@10: {final_results['evaluation_summary']['precision_recall']['P@10']:.4f}")
        print(f"R@10: {final_results['evaluation_summary']['precision_recall']['R@10']:.4f}")
        print("\nMethod Comparison:")
        print(f"  Inverted Index: {comparison_results['inverted_index']['MAP']:.4f}")
        print(f"  Full Matrix: {comparison_results['full_matrix']['MAP']:.4f}")
        print(f"  Improvement: {comparison_results['improvement']:.4f}")
        print(f"\nQueries above 0.4 AP: {final_results['evaluation_summary']['query_performance']['queries_above_0_4']} "
              f"({final_results['evaluation_summary']['query_performance']['percentage_above_0_4']:.1f}%)")
        print("="*60)
        
        # Print recommendations if MAP < 0.4
        if not final_results['evaluation_summary']['target_achieved']:
            print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
            for rec in final_results['recommendations']:
                print(f"  {rec}")
            print("\nConsider retraining with different parameters in Colab.")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {"error": str(e)}

def test_individual_components():
    """Test individual components for debugging."""
    logger.info("=== Testing Individual Components ===")
    
    try:
        # Test preprocessing
        logger.info("Testing preprocessing service...")
        preprocessor = TextPreprocessingService()
        test_text = "This is a test of antique furniture restoration techniques."
        processed = preprocessor.preprocess_for_tfidf(test_text)
        logger.info(f"Preprocessing test: '{test_text}' -> '{processed}'")
        
        # Test TF-IDF service loading
        logger.info("Testing TF-IDF service loading...")
        models_dir = setup_models_directory()
        
        if check_required_model_files(models_dir):
            tfidf_service = OptimizedTFIDFService(str(models_dir))
            loaded = tfidf_service.load_models("antique")
            logger.info(f"TF-IDF service loading: {'✓ SUCCESS' if loaded else '✗ FAILED'}")
            
            if loaded:
                # Test search
                results = tfidf_service.search_with_inverted_index("antique furniture", top_k=3)
                logger.info(f"Search test returned {len(results)} results")
        
        # Test evaluation service
        logger.info("Testing evaluation service...")
        evaluator = MAPEvaluationService()
        loaded = evaluator.load_antique_evaluation_data()
        logger.info(f"Evaluation data loading: {'✓ SUCCESS' if loaded else '✗ FAILED'}")
        
        logger.info("✓ Component testing complete")
        return True
        
    except Exception as e:
        logger.error(f"Component testing failed: {str(e)}")
        return False

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TF-IDF Evaluation Script")
    parser.add_argument("--test-components", action="store_true", 
                       help="Test individual components only")
    parser.add_argument("--max-queries", type=int, default=100,
                       help="Maximum number of queries to evaluate")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with 20 queries")
    
    args = parser.parse_args()
    
    if args.test_components:
        success = test_individual_components()
        sys.exit(0 if success else 1)
    
    max_queries = 20 if args.quick else args.max_queries
    
    logger.info(f"Starting evaluation with max_queries={max_queries}")
    results = run_comprehensive_evaluation(max_queries=max_queries)
    
    if "error" in results:
        logger.error(f"Evaluation failed: {results['error']}")
        sys.exit(1)
    
    # Check if target achieved
    if results['evaluation_summary']['target_achieved']:
        logger.info("🎯 TARGET ACHIEVED: MAP >= 0.4!")
        sys.exit(0)
    else:
        logger.warning(f"⚠️  Target not achieved: MAP = {results['evaluation_summary']['MAP']:.4f} < 0.4")
        sys.exit(1)

if __name__ == "__main__":
    main()
