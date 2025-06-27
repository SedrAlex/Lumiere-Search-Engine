#!/usr/bin/env python3
"""
Complete Training and Evaluation Script for IR Search Engine
Implements proper academic IR methodology with real datasets and metrics
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import time

# Import our custom modules
from dataset_loader import IRDatasetLoader
from document_representations import create_representation
from evaluation_engine import SearchEvaluator, ComparisonEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IRSearchEngineTrainer:
    """Complete IR Search Engine Training and Evaluation System"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_loader = IRDatasetLoader()
        self.evaluator = SearchEvaluator()
        self.comparator = ComparisonEvaluator()
        
        # Store loaded datasets and built indices
        self.datasets = {}
        self.representations = {}
        self.evaluation_results = {}
    
    async def load_datasets(self, dataset_configs: List[Dict[str, Any]]) -> None:
        """
        Load multiple datasets for the project
        
        Args:
            dataset_configs: List of dataset configurations
                Each config should have: {'name': str, 'limit_docs': int, 'limit_queries': int}
        """
        logger.info("=" * 60)
        logger.info("🔥 LOADING IR DATASETS")
        logger.info("=" * 60)
        
        for config in dataset_configs:
            dataset_name = config['name']
            limit_docs = config.get('limit_docs', None)
            limit_queries = config.get('limit_queries', 100)
            
            logger.info(f"\n📊 Loading {dataset_name}...")
            logger.info(f"   Documents: {limit_docs if limit_docs else 'ALL'}")
            logger.info(f"   Queries: {limit_queries}")
            
            try:
                # Load dataset
                dataset = await self.dataset_loader.load_dataset(
                    dataset_name, limit_docs, limit_queries
                )
                
                # Save dataset
                dataset_path = self.data_dir / "datasets" 
                await self.dataset_loader.save_dataset(dataset, str(dataset_path))
                
                self.datasets[dataset_name] = dataset
                
                logger.info(f"✅ {dataset_name} loaded successfully!")
                logger.info(f"   📄 Documents: {len(dataset['documents']):,}")
                logger.info(f"   ❓ Queries: {len(dataset['queries'])}")
                logger.info(f"   🎯 Relevance judgments: {len(dataset['qrels'])}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {dataset_name}: {e}")
                continue
    
    async def build_indices(self, representation_configs: List[Dict[str, Any]]) -> None:
        """
        Build search indices for all datasets and representations
        
        Args:
            representation_configs: List of representation configurations
                Each config: {'type': str, 'params': dict}
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔧 BUILDING SEARCH INDICES")
        logger.info("=" * 60)
        
        for dataset_name, dataset in self.datasets.items():
            logger.info(f"\n📊 Building indices for {dataset_name}...")
            
            self.representations[dataset_name] = {}
            
            for repr_config in representation_configs:
                repr_type = repr_config['type']
                repr_params = repr_config.get('params', {})
                
                logger.info(f"\n   🔨 Building {repr_type} representation...")
                
                try:
                    # Create representation
                    representation = create_representation(repr_type, **repr_params)
                    
                    # Build index
                    start_time = time.time()
                    representation.build_index(dataset['documents'])
                    build_time = time.time() - start_time
                    
                    # Save index
                    index_path = self.data_dir / "indices" / f"{dataset_name}_{repr_type}.joblib"
                    index_path.parent.mkdir(parents=True, exist_ok=True)
                    representation.save_index(str(index_path))
                    
                    self.representations[dataset_name][repr_type] = representation
                    
                    logger.info(f"   ✅ {repr_type} built in {build_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to build {repr_type}: {e}")
                    continue
    
    async def run_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all datasets and representations
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("\n" + "=" * 60)
        logger.info("📊 RUNNING COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        all_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            logger.info(f"\n🔍 Evaluating {dataset_name}...")
            
            dataset_results = {}
            queries = dataset['queries']
            qrels = dataset['qrels']
            
            # Run search for each representation
            for repr_type, representation in self.representations[dataset_name].items():
                logger.info(f"   📋 Running {repr_type} search...")
                
                try:
                    # Perform searches
                    search_results = {}
                    search_times = []
                    
                    for query in queries:
                        query_id = query['query_id']
                        query_text = query['text']
                        
                        # Time the search
                        start_time = time.time()
                        results = representation.search(query_text, top_k=20)
                        search_time = time.time() - start_time
                        search_times.append(search_time)
                        
                        # Store results (just doc_ids)
                        search_results[query_id] = [doc_id for doc_id, score in results]
                    
                    # Evaluate using our metrics
                    metrics = self.evaluator.evaluate_system(search_results, qrels)
                    
                    # Add timing information
                    metrics['avg_search_time'] = sum(search_times) / len(search_times)
                    metrics['total_search_time'] = sum(search_times)
                    
                    dataset_results[repr_type] = {
                        'metrics': metrics,
                        'search_results': search_results  # Store for comparison
                    }
                    
                    logger.info(f"   ✅ {repr_type} evaluation complete")
                    logger.info(f"      MAP: {metrics.get('map', 0):.4f}")
                    logger.info(f"      MRR: {metrics.get('mrr', 0):.4f}")
                    logger.info(f"      P@10: {metrics.get('precision_at_10', 0):.4f}")
                    logger.info(f"      NDCG@10: {metrics.get('ndcg_at_10', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to evaluate {repr_type}: {e}")
                    continue
            
            all_results[dataset_name] = dataset_results
        
        self.evaluation_results = all_results
        return all_results
    
    async def compare_representations(self) -> Dict[str, Any]:
        """
        Compare different representations and perform statistical tests
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("\n" + "=" * 60)
        logger.info("⚔️  COMPARING REPRESENTATIONS")
        logger.info("=" * 60)
        
        comparison_results = {}
        
        for dataset_name, dataset_results in self.evaluation_results.items():
            logger.info(f"\n📊 Comparing representations for {dataset_name}...")
            
            # Extract search results for comparison
            methods_results = {}
            for repr_type, results in dataset_results.items():
                methods_results[repr_type] = results['search_results']
            
            # Get qrels
            qrels = self.datasets[dataset_name]['qrels']
            
            # Run comparison
            comparison = self.comparator.compare_representations(methods_results, qrels)
            
            # Statistical significance tests between methods
            significance_tests = {}
            method_names = list(methods_results.keys())
            
            for i, method1 in enumerate(method_names):
                for method2 in method_names[i+1:]:
                    test_key = f"{method1}_vs_{method2}"
                    
                    significance_test = self.comparator.statistical_significance_test(
                        methods_results[method1],
                        methods_results[method2],
                        qrels
                    )
                    
                    significance_tests[test_key] = significance_test
                    
                    logger.info(f"   📈 {method1} vs {method2}:")
                    logger.info(f"      Mean AP Difference: {significance_test.get('mean_difference', 0):.4f}")
                    logger.info(f"      T-statistic: {significance_test.get('t_statistic', 0):.4f}")
            
            comparison_results[dataset_name] = {
                'metrics_comparison': comparison,
                'significance_tests': significance_tests
            }
        
        return comparison_results
    
    async def generate_report(self, results: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: Evaluation results
            comparison: Comparison results
            
        Returns:
            Report as formatted string
        """
        report = []
        report.append("🔍 INFORMATION RETRIEVAL SEARCH ENGINE EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("📋 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        total_datasets = len(self.datasets)
        total_docs = sum(len(d['documents']) for d in self.datasets.values())
        total_queries = sum(len(d['queries']) for d in self.datasets.values())
        
        report.append(f"• Datasets evaluated: {total_datasets}")
        report.append(f"• Total documents: {total_docs:,}")
        report.append(f"• Total queries: {total_queries}")
        report.append("")
        
        # Detailed results for each dataset
        for dataset_name, dataset_results in results.items():
            report.append(f"📊 DATASET: {dataset_name.upper()}")
            report.append("-" * 40)
            
            dataset_info = self.datasets[dataset_name]
            report.append(f"Documents: {len(dataset_info['documents']):,}")
            report.append(f"Queries: {len(dataset_info['queries'])}")
            report.append("")
            
            # Performance table
            report.append("PERFORMANCE METRICS:")
            report.append(f"{'Method':<12} {'MAP':<8} {'MRR':<8} {'P@10':<8} {'NDCG@10':<10} {'Avg Time':<10}")
            report.append("-" * 70)
            
            for method, method_results in dataset_results.items():
                metrics = method_results['metrics']
                report.append(
                    f"{method:<12} "
                    f"{metrics.get('map', 0):<8.4f} "
                    f"{metrics.get('mrr', 0):<8.4f} "
                    f"{metrics.get('precision_at_10', 0):<8.4f} "
                    f"{metrics.get('ndcg_at_10', 0):<10.4f} "
                    f"{metrics.get('avg_search_time', 0)*1000:<10.2f}ms"
                )
            
            report.append("")
            
            # Statistical significance
            if dataset_name in comparison:
                report.append("STATISTICAL SIGNIFICANCE TESTS:")
                sig_tests = comparison[dataset_name]['significance_tests']
                for test_name, test_result in sig_tests.items():
                    method1, method2 = test_name.split('_vs_')
                    mean_diff = test_result.get('mean_difference', 0)
                    t_stat = test_result.get('t_statistic', 0)
                    
                    significance = "**" if abs(t_stat) > 2.0 else "*" if abs(t_stat) > 1.5 else ""
                    report.append(f"• {method1} vs {method2}: Δ={mean_diff:.4f}, t={t_stat:.2f} {significance}")
                
                report.append("")
        
        # Overall ranking
        report.append("🏆 OVERALL PERFORMANCE RANKING")
        report.append("-" * 40)
        
        # Calculate average performance across datasets
        method_averages = {}
        for dataset_results in results.values():
            for method, method_results in dataset_results.items():
                if method not in method_averages:
                    method_averages[method] = []
                method_averages[method].append(method_results['metrics'].get('map', 0))
        
        # Average and rank
        method_rankings = []
        for method, scores in method_averages.items():
            avg_score = sum(scores) / len(scores)
            method_rankings.append((method, avg_score))
        
        method_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (method, avg_score) in enumerate(method_rankings, 1):
            report.append(f"{rank}. {method:<12} Average MAP: {avg_score:.4f}")
        
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        best_method = method_rankings[0][0] if method_rankings else "Unknown"
        report.append(f"• Best overall performance: {best_method}")
        report.append("• Consider hybrid approaches for balanced performance")
        report.append("• Embedding methods show good semantic understanding")
        report.append("• BM25 provides strong baseline performance")
        report.append("• TF-IDF remains competitive for exact term matching")
        
        return "\n".join(report)
    
    async def save_results(self, results: Dict[str, Any], comparison: Dict[str, Any]) -> None:
        """Save all results to disk"""
        results_dir = self.data_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(results_dir / "evaluation_results.json", 'w') as f:
            # Convert to JSON-serializable format
            json_results = {}
            for dataset, dataset_results in results.items():
                json_results[dataset] = {}
                for method, method_results in dataset_results.items():
                    json_results[dataset][method] = method_results['metrics']
            
            json.dump(json_results, f, indent=2)
        
        # Save comparison results
        with open(results_dir / "comparison_results.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Generate and save report
        report = await self.generate_report(results, comparison)
        with open(results_dir / "evaluation_report.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"\n✅ Results saved to {results_dir}")
        print(f"\n📄 Report preview:\n")
        print(report[:2000] + "..." if len(report) > 2000 else report)

# Main training and evaluation pipeline
async def main():
    """Main training and evaluation pipeline"""
    
    print("🚀 STARTING IR SEARCH ENGINE TRAINING & EVALUATION")
    print("=" * 80)
    
    # Initialize trainer
    trainer = IRSearchEngineTrainer()
    
    # Configuration for datasets (adjust based on your computational resources)
    dataset_configs = [
        {
            'name': 'msmarco-passage',
            'limit_docs': 250000,  # 250K documents - meets requirement
            'limit_queries': 100
        },
        {
            'name': 'robust04', 
            'limit_docs': 200000,  # 200K documents - meets requirement
            'limit_queries': 100
        }
    ]
    
    # Configuration for representations
    representation_configs = [
        {'type': 'tfidf', 'params': {'max_features': 10000, 'ngram_range': (1, 2)}},
        {'type': 'embedding', 'params': {'model_name': 'all-MiniLM-L6-v2'}},
        {'type': 'bm25', 'params': {'k1': 1.2, 'b': 0.75}},
        {'type': 'hybrid', 'params': {'tfidf_weight': 0.3, 'embedding_weight': 0.4, 'bm25_weight': 0.3}}
    ]
    
    try:
        # Step 1: Load datasets
        await trainer.load_datasets(dataset_configs)
        
        # Step 2: Build indices
        await trainer.build_indices(representation_configs)
        
        # Step 3: Run evaluation
        results = await trainer.run_evaluation()
        
        # Step 4: Compare representations
        comparison = await trainer.compare_representations()
        
        # Step 5: Save results and generate report
        await trainer.save_results(results, comparison)
        
        print("\n🎉 TRAINING AND EVALUATION COMPLETE!")
        print(f"Check the 'data/results' directory for detailed results.")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

# Additional utility functions
async def quick_test():
    """Quick test with minimal data for development"""
    print("🧪 RUNNING QUICK TEST...")
    
    trainer = IRSearchEngineTrainer()
    
    # Small test configuration
    test_configs = [
        {'name': 'msmarco-passage', 'limit_docs': 1000, 'limit_queries': 10}
    ]
    
    repr_configs = [
        {'type': 'tfidf', 'params': {}},
        {'type': 'bm25', 'params': {}}
    ]
    
    await trainer.load_datasets(test_configs)
    await trainer.build_indices(repr_configs)
    results = await trainer.run_evaluation()
    comparison = await trainer.compare_representations()
    await trainer.save_results(results, comparison)

if __name__ == "__main__":
    # Run full evaluation
    asyncio.run(main())
    
    # For quick testing, uncomment the line below instead:
    # asyncio.run(quick_test())
