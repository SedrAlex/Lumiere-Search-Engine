#!/usr/bin/env python3
"""
Enhanced ANTIQUE TF-IDF Evaluation with Full Dataset Verification
===============================================================

Key Improvements:
1. Uses exact same cleaning pipeline as training
2. Proper vocabulary coverage verification
3. Evaluates all queries with relevance judgments (~2.4k)
4. Added detailed vocabulary analysis
5. Better error handling and logging
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
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('antique_tfidf_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TFIDF_QUERY_SERVICE_URL = "http://localhost:8007"
RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize NLTK components
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TFIDFTextCleaner:
    """EXACT replica of the cleaning pipeline used in training"""
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Identical cleaning to training pipeline"""
        if not text:
            return ""
            
        # Step 1: Basic cleaning
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 2: Tokenization and filtering
        tokens = word_tokenize(text)
        filtered_tokens = [
            token for token in tokens 
            if len(token) >= 2 and token.isalnum() and token not in self.stop_words
        ]
        
        # Step 3: Lemmatization then stemming
        final_tokens = []
        for token in filtered_tokens:
            lemmatized = self.lemmatizer.lemmatize(token)
            stemmed = self.stemmer.stem(lemmatized)
            final_tokens.append(stemmed)
            
        return " ".join(final_tokens)

class AntiqueDatasetVerifier:
    """Enhanced dataset verification with vocabulary analysis"""
    
    @staticmethod
    def verify_datasets():
        """Verify both train and test sets and compare vocabularies"""
        logger.info("🔍 Verifying ANTIQUE datasets...")
        
        try:
            # Load both datasets
            train_dataset = ir_datasets.load('antique/train')
            test_dataset = ir_datasets.load('antique/test')
            
            # Basic stats
            train_queries = list(train_dataset.queries_iter())
            test_queries = list(test_dataset.queries_iter())
            
            logger.info(f"✅ ANTIQUE train queries: {len(train_queries)}")
            logger.info(f"✅ ANTIQUE test queries: {len(test_queries)}")
            logger.info(f"✅ ANTIQUE train docs: {len(list(train_dataset.docs_iter()))}")
            logger.info(f"✅ ANTIQUE test docs: {len(list(test_dataset.docs_iter()))}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying datasets: {e}")
            return False

class IRMetrics:
    """Enhanced IR metrics with vocabulary coverage tracking"""
    
    @staticmethod
    def average_precision(ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Average Precision (AP) with edge case handling"""
        if not relevant_docs:
            return 0.0  # Query has no relevant docs
            
        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc in enumerate(ranked_docs, 1):
            if doc in relevant_set:
                relevant_found += 1
                precision_at_i = relevant_found / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs)
    
    @staticmethod
    def reciprocal_rank(ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Reciprocal Rank (RR)"""
        if not relevant_docs:
            return 0.0
            
        relevant_set = set(relevant_docs)
        for i, doc in enumerate(ranked_docs, 1):
            if doc in relevant_set:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def precision_at_k(ranked_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if not relevant_docs or k <= 0:
            return 0.0
            
        relevant_set = set(relevant_docs)
        top_k_docs = ranked_docs[:k]
        return len([doc for doc in top_k_docs if doc in relevant_set]) / min(k, len(ranked_docs))
    
    @staticmethod
    def recall_at_k(ranked_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs or k <= 0:
            return 0.0
            
        relevant_set = set(relevant_docs)
        top_k_docs = ranked_docs[:k]
        return len([doc for doc in top_k_docs if doc in relevant_set]) / len(relevant_docs)
    
    @staticmethod
    def ndcg_at_k(ranked_docs: List[str], qrels: Dict[str, int], k: int) -> float:
        """Calculate NDCG@K using sklearn implementation"""
        if not qrels or k <= 0:
            return 0.0
            
        # Get relevance scores for ranked docs
        y_true = []
        y_score = []
        
        for i, doc in enumerate(ranked_docs[:k]):
            relevance = qrels.get(doc, 0)
            y_true.append(relevance)
            # Use inverse rank as score (higher rank = higher score)
            y_score.append(1.0 / (i + 1))
        
        if not y_true or max(y_true) == 0:
            return 0.0
            
        try:
            return ndcg_score([y_true], [y_score], k=k)
        except:
            return 0.0

class AntiqueTFIDFEvaluator:
    """Complete evaluation system with vocabulary verification"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.text_cleaner = TFIDFTextCleaner()
        self.metrics = IRMetrics()
        self.vocab_coverage_stats = []
    
    async def close(self):
        await self.http_client.aclose()
    
    async def verify_services(self) -> bool:
        """Verify TF-IDF service is running"""
        try:
            response = await self.http_client.get(f"{TFIDF_QUERY_SERVICE_URL}/health")
            if response.status_code == 200:
                logger.info("✅ TF-IDF service is running")
                return True
        except Exception as e:
            logger.error(f"❌ TF-IDF service not available: {e}")
        return False
    
    def load_antique_data(self) -> Tuple[List[Dict], Dict[str, Dict[str, int]], set]:
        """Load train data with vocabulary analysis"""
        logger.info("📚 Loading ANTIQUE train dataset...")
        
        if not AntiqueDatasetVerifier.verify_datasets():
            raise RuntimeError("Dataset verification failed")
        
        dataset = ir_datasets.load('antique/train')
        
        # Load queries and track vocabulary
        queries = []
        train_terms = set()
        
        for query in dataset.queries_iter():
            cleaned = self.text_cleaner.clean_text(query.text)
            queries.append({
                'query_id': query.query_id,
                'text': query.text,
                'cleaned_text': cleaned
            })
            train_terms.update(cleaned.split())
        
        # Load qrels
        qrels = {}
        relevance_counts = Counter()
        
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            relevance_counts[qrel.relevance] += 1
        
        logger.info("📊 Relevance distribution:")
        for rel, count in sorted(relevance_counts.items()):
            logger.info(f"  Relevance {rel}: {count} judgments")
        
        return queries, qrels, train_terms
    
    async def check_vocabulary_coverage(self, train_terms: set):
        """Verify how many train terms exist in the service vocabulary"""
        try:
            # Get vocabulary from service
            response = await self.http_client.get(f"{TFIDF_QUERY_SERVICE_URL}/vocabulary")
            if response.status_code == 200:
                vocab = set(response.json().get('vocabulary', []))
                covered = train_terms & vocab
                coverage = len(covered) / len(train_terms) if train_terms else 0
                
                logger.info(f"📊 Vocabulary Coverage: {coverage:.2%}")
                logger.info(f"  Train terms: {len(train_terms)}")
                logger.info(f"  Covered terms: {len(covered)}")
                logger.info(f"  Missing terms: {len(train_terms - vocab)}")
                
                # Sample some missing terms
                missing = list(train_terms - vocab)[:10]
                if missing:
                    logger.info(f"  Sample missing terms: {missing}")
                
                return coverage
        except Exception as e:
            logger.warning(f"Couldn't check vocabulary coverage: {e}")
        return 0
    
    async def query_tfidf_system(self, query_text: str, top_k: int = 1000) -> List[str]:
        """Query TF-IDF system with proper cleaning"""
        cleaned_query = self.text_cleaner.clean_text(query_text)
        
        try:
            response = await self.http_client.post(
                f"{TFIDF_QUERY_SERVICE_URL}/search",
                json={
                    "query": cleaned_query,
                    "top_k": top_k,
                    "similarity_threshold": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            return [doc["doc_id"] for doc in result["results"]]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    async def evaluate_single_query(self, query: Dict[str, str], 
                                  query_qrels: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Enhanced evaluation with vocabulary tracking"""
        relevant_docs = [doc_id for doc_id, rel in query_qrels.items() if rel > 0]
        if not relevant_docs:
            return None  # Skip queries with no relevant docs
            
        retrieved_docs = await self.query_tfidf_system(query['text'])
        if not retrieved_docs:
            return self._empty_metrics(query['query_id'], query['text'], query['cleaned_text'])
        
        # Track vocabulary coverage for this query
        query_terms = set(query['cleaned_text'].split())
        vocab_coverage = 0  # Will be updated after checking with service
        
        metrics = {
            'query_id': query['query_id'],
            'query_text': query['text'],
            'cleaned_text': query['cleaned_text'],
            'num_retrieved': len(retrieved_docs),
            'num_relevant': len(relevant_docs),
            'num_relevant_retrieved': len(set(retrieved_docs) & set(relevant_docs)),
            'vocab_coverage': vocab_coverage,
            'average_precision': self.metrics.average_precision(retrieved_docs, relevant_docs),
            'reciprocal_rank': self.metrics.reciprocal_rank(retrieved_docs, relevant_docs)
        }
        
        # Add precision/recall/ndcg at various K values
        for k in [1, 5, 10, 20, 50, 100]:
            metrics[f'precision_at_{k}'] = self.metrics.precision_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'recall_at_{k}'] = self.metrics.recall_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'ndcg_at_{k}'] = self.metrics.ndcg_at_k(retrieved_docs, query_qrels, k)
        
        return metrics
    
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation on all queries"""
        logger.info("🚀 Starting comprehensive evaluation...")
        
        if not await self.verify_services():
            raise RuntimeError("Required services not available")
        
        # Load data and verify vocabulary
        queries, qrels, train_terms = self.load_antique_data()
        vocab_coverage = await self.check_vocabulary_coverage(train_terms)
        
        # Filter to queries with relevance judgments
        queries_with_qrels = [q for q in queries if q['query_id'] in qrels and qrels[q['query_id']]]
        logger.info(f"📊 Evaluating {len(queries_with_qrels)} queries with relevance judgments")
        
        # Evaluate all queries (not just a sample)
        query_results = []
        start_time = time.time()
        
        for query in tqdm(queries_with_qrels, desc="Evaluating queries"):
            try:
                metrics = await self.evaluate_single_query(query, qrels[query['query_id']])
                if metrics:  # Skip queries with no relevant docs
                    query_results.append(metrics)
                    
                    # Periodic logging
                    if len(query_results) % 100 == 0:
                        avg_map = np.mean([r['average_precision'] for r in query_results])
                        logger.info(f"✅ Evaluated {len(query_results)} queries | Current MAP: {avg_map:.4f}")
                        
            except Exception as e:
                logger.error(f"Failed on query {query['query_id']}: {e}")
        
        # Calculate final metrics
        evaluation_time = time.time() - start_time
        metrics = self._calculate_aggregated_metrics(query_results)
        
        return {
            'evaluation_info': {
                'dataset': 'antique/train',
                'total_queries': len(queries),
                'evaluated_queries': len(query_results),
                'evaluation_time': evaluation_time,
                'vocab_coverage': vocab_coverage,
                'avg_query_time': evaluation_time / len(query_results) if query_results else 0
            },
            'metrics': metrics,
            'query_results': query_results
        }
    
    def _calculate_aggregated_metrics(self, query_results: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive metrics across all queries"""
        if not query_results:
            return {}
            
        metrics = {
            'num_queries': len(query_results),
            'MAP': np.mean([r['average_precision'] for r in query_results]),
            'MRR': np.mean([r['reciprocal_rank'] for r in query_results]),
            'avg_vocab_coverage': np.mean([r['vocab_coverage'] for r in query_results])
        }
        
        # Add all precision/recall/ndcg metrics
        for k in [1, 5, 10, 20, 50, 100]:
            metrics[f'P@{k}'] = np.mean([r[f'precision_at_{k}'] for r in query_results])
            metrics[f'R@{k}'] = np.mean([r[f'recall_at_{k}'] for r in query_results])
            metrics[f'NDCG@{k}'] = np.mean([r[f'ndcg_at_{k}'] for r in query_results])
        
        return metrics
    
    def _empty_metrics(self, query_id: str, query_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Return empty metrics for queries with no results"""
        metrics = {
            'query_id': query_id,
            'query_text': query_text,
            'cleaned_text': cleaned_text,
            'num_retrieved': 0,
            'num_relevant': 0,
            'num_relevant_retrieved': 0,
            'vocab_coverage': 0,
            'average_precision': 0.0,
            'reciprocal_rank': 0.0
        }
        
        # Add zeros for all precision/recall/ndcg metrics
        for k in [1, 5, 10, 20, 50, 100]:
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        
        return metrics
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save comprehensive results with vocabulary analysis"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"antique_tfidf_eval_{timestamp}"
        
        # Save JSON
        json_path = RESULTS_DIR / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(results['query_results'])
        csv_path = RESULTS_DIR / f"{filename}_queries.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"💾 Results saved to {json_path} and {csv_path}")
        return json_path

async def main():
    """Run full evaluation"""
    evaluator = AntiqueTFIDFEvaluator()
    
    try:
        logger.info("="*80)
        logger.info("ANTIQUE TF-IDF COMPREHENSIVE EVALUATION")
        logger.info("="*80)
        
        results = await evaluator.run_full_evaluation()
        
        # Print summary
        logger.info("\n📊 FINAL RESULTS:")
        logger.info(f"  MAP: {results['metrics']['MAP']:.4f}")
        logger.info(f"  MRR: {results['metrics']['MRR']:.4f}")
        logger.info(f"  Vocabulary Coverage: {results['evaluation_info']['vocab_coverage']:.2%}")
        logger.info(f"  Evaluated {results['evaluation_info']['evaluated_queries']} queries")
        
        # Save results
        output_file = evaluator.save_results(results)
        logger.info(f"\n💾 Complete results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        await evaluator.close()

if __name__ == "__main__":
    asyncio.run(main())