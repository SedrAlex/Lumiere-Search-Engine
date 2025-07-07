#!/usr/bin/env python3
"""
QUORA TF-IDF Evaluation Script
Standalone evaluation for the QUORA TF-IDF offline service.
Calculates MAP, Precision@K, Recall@K, and other IR metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraTFIDFEvaluator:
    """
    Evaluator for QUORA TF-IDF system using offline models and data.
    """
    
    def __init__(self, models_path="models"):
        """
        Initialize the evaluator.
        
        Args:
            models_path (str): Path to the directory containing model files
        """
        self.models_path = models_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index = None
        self.doc_id_to_index = None
        self.index_to_doc_id = None
        self.feature_names = None
        self.queries_df = None
        self.qrels_df = None
        
    def load_models(self):
        """Load all pre-trained models and indices."""
        logger.info("Loading QUORA TF-IDF models...")
        
        try:
            # Load TF-IDF vectorizer
            vectorizer_path = os.path.join(self.models_path, "quora_tfidf_vectorizer.joblib")
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("✅ TF-IDF vectorizer loaded")
            else:
                raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
            
            # Load TF-IDF matrix
            matrix_path = os.path.join(self.models_path, "quora_tfidf_matrix.joblib")
            if os.path.exists(matrix_path):
                self.tfidf_matrix = joblib.load(matrix_path)
                logger.info(f"✅ TF-IDF matrix loaded: {self.tfidf_matrix.shape}")
            else:
                raise FileNotFoundError(f"TF-IDF matrix not found at {matrix_path}")
            
            # Load inverted index
            index_path = os.path.join(self.models_path, "quora_inverted_index.joblib")
            if os.path.exists(index_path):
                self.inverted_index = joblib.load(index_path)
                logger.info(f"✅ Inverted index loaded with {len(self.inverted_index)} terms")
            else:
                logger.warning("Inverted index not found, will create during evaluation")
            
            # Load document mappings
            doc_id_path = os.path.join(self.models_path, "quora_doc_id_to_index.joblib")
            index_doc_path = os.path.join(self.models_path, "quora_index_to_doc_id.joblib")
            
            if os.path.exists(doc_id_path) and os.path.exists(index_doc_path):
                self.doc_id_to_index = joblib.load(doc_id_path)
                self.index_to_doc_id = joblib.load(index_doc_path)
                logger.info(f"✅ Document mappings loaded: {len(self.doc_id_to_index)} documents")
            else:
                raise FileNotFoundError("Document mappings not found")
            
            # Load feature names
            features_path = os.path.join(self.models_path, "quora_feature_names.joblib")
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                logger.info(f"✅ Feature names loaded: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names not found")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_evaluation_data(self, queries_path, qrels_path):
        """
        Load queries and relevance judgments for evaluation.
        
        Args:
            queries_path (str): Path to queries TSV file
            qrels_path (str): Path to qrels TSV file
        """
        logger.info("Loading evaluation data...")
        
        try:
            # Load queries
            self.queries_df = pd.read_csv(queries_path, sep='\t', header=None, names=['query_id', 'text'])
            logger.info(f"✅ Queries loaded: {len(self.queries_df)}")
            
            # Load relevance judgments
            self.qrels_df = pd.read_csv(qrels_path, sep='\t', header=None, names=['query_id', 'Q0', 'doc_id', 'relevance'])
            logger.info(f"✅ Relevance judgments loaded: {len(self.qrels_df)}")
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            raise
    
    def preprocess_text(self, text):
        """
        Preprocess text using the same method as in training.
        This should match the custom_tokenizer used in the notebook.
        """
        if not text or pd.isna(text):
            return ""
        
        # Basic preprocessing - you may need to adjust this based on your exact processing
        import re
        
        # Remove special characters except letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def calculate_map(self, k=1000):
        """
        Calculate Mean Average Precision (MAP) score.
        
        Args:
            k (int): Number of top documents to consider
            
        Returns:
            float: MAP score
        """
        logger.info(f"Calculating MAP@{k}...")
        
        if self.queries_df is None or self.qrels_df is None:
            raise ValueError("Queries and qrels must be loaded first")
        
        # Process queries
        processed_queries = []
        for query_text in self.queries_df['text']:
            processed_query = self.preprocess_text(str(query_text))
            processed_queries.append(processed_query)
        
        # Transform queries using the fitted vectorizer
        query_vectors = self.vectorizer.transform(processed_queries)
        
        # Calculate similarities
        query_doc_similarities = cosine_similarity(query_vectors, self.tfidf_matrix)
        
        average_precisions = []
        queries_with_relevance = 0
        
        for i, query_id in enumerate(self.queries_df['query_id']):
            # Get relevance judgments for this query
            query_qrels = self.qrels_df[self.qrels_df['query_id'] == query_id]
            
            if len(query_qrels) == 0:
                continue
            
            # Get similarity scores for this query
            similarities = query_doc_similarities[i]
            
            # Sort documents by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1][:k]
            
            # Get relevant documents
            relevant_docs = set(query_qrels[query_qrels['relevance'] > 0]['doc_id'].values)
            
            if len(relevant_docs) == 0:
                continue
            
            queries_with_relevance += 1
            
            # Calculate precision at each relevant document
            precisions = []
            num_relevant_found = 0
            
            for rank, doc_index in enumerate(sorted_indices, 1):
                if doc_index in self.index_to_doc_id:
                    doc_id = self.index_to_doc_id[doc_index]
                    
                    if doc_id in relevant_docs:
                        num_relevant_found += 1
                        precision = num_relevant_found / rank
                        precisions.append(precision)
            
            if precisions:
                average_precision = np.mean(precisions)
                average_precisions.append(average_precision)
        
        map_score = np.mean(average_precisions) if average_precisions else 0.0
        logger.info(f"MAP@{k}: {map_score:.4f} (calculated from {queries_with_relevance} queries)")
        
        return map_score
    
    def calculate_precision_at_k(self, k_values=[1, 5, 10, 20, 50, 100]):
        """
        Calculate Precision@K for different K values.
        
        Args:
            k_values (list): List of K values to calculate
            
        Returns:
            dict: Dictionary with K values and their precision scores
        """
        logger.info(f"Calculating Precision@K for K={k_values}...")
        
        # Process queries
        processed_queries = []
        for query_text in self.queries_df['text']:
            processed_query = self.preprocess_text(str(query_text))
            processed_queries.append(processed_query)
        
        query_vectors = self.vectorizer.transform(processed_queries)
        query_doc_similarities = cosine_similarity(query_vectors, self.tfidf_matrix)
        
        precision_at_k = {k: [] for k in k_values}
        
        for i, query_id in enumerate(self.queries_df['query_id']):
            query_qrels = self.qrels_df[self.qrels_df['query_id'] == query_id]
            
            if len(query_qrels) == 0:
                continue
            
            similarities = query_doc_similarities[i]
            sorted_indices = np.argsort(similarities)[::-1]
            
            relevant_docs = set(query_qrels[query_qrels['relevance'] > 0]['doc_id'].values)
            
            if len(relevant_docs) == 0:
                continue
            
            for k in k_values:
                top_k_indices = sorted_indices[:k]
                top_k_doc_ids = []
                
                for idx in top_k_indices:
                    if idx in self.index_to_doc_id:
                        top_k_doc_ids.append(self.index_to_doc_id[idx])
                
                relevant_in_top_k = len([doc_id for doc_id in top_k_doc_ids if doc_id in relevant_docs])
                precision_k = relevant_in_top_k / k if k > 0 else 0
                precision_at_k[k].append(precision_k)
        
        # Calculate mean precision for each K
        mean_precision_at_k = {}
        for k, precisions in precision_at_k.items():
            mean_precision_at_k[k] = np.mean(precisions) if precisions else 0.0
            logger.info(f"P@{k}: {mean_precision_at_k[k]:.4f}")
        
        return mean_precision_at_k
    
    def calculate_recall_at_k(self, k_values=[1, 5, 10, 20, 50, 100]):
        """
        Calculate Recall@K for different K values.
        
        Args:
            k_values (list): List of K values to calculate
            
        Returns:
            dict: Dictionary with K values and their recall scores
        """
        logger.info(f"Calculating Recall@K for K={k_values}...")
        
        # Process queries
        processed_queries = []
        for query_text in self.queries_df['text']:
            processed_query = self.preprocess_text(str(query_text))
            processed_queries.append(processed_query)
        
        query_vectors = self.vectorizer.transform(processed_queries)
        query_doc_similarities = cosine_similarity(query_vectors, self.tfidf_matrix)
        
        recall_at_k = {k: [] for k in k_values}
        
        for i, query_id in enumerate(self.queries_df['query_id']):
            query_qrels = self.qrels_df[self.qrels_df['query_id'] == query_id]
            
            if len(query_qrels) == 0:
                continue
            
            similarities = query_doc_similarities[i]
            sorted_indices = np.argsort(similarities)[::-1]
            
            relevant_docs = set(query_qrels[query_qrels['relevance'] > 0]['doc_id'].values)
            
            if len(relevant_docs) == 0:
                continue
            
            for k in k_values:
                top_k_indices = sorted_indices[:k]
                top_k_doc_ids = []
                
                for idx in top_k_indices:
                    if idx in self.index_to_doc_id:
                        top_k_doc_ids.append(self.index_to_doc_id[idx])
                
                relevant_in_top_k = len([doc_id for doc_id in top_k_doc_ids if doc_id in relevant_docs])
                recall_k = relevant_in_top_k / len(relevant_docs) if len(relevant_docs) > 0 else 0
                recall_at_k[k].append(recall_k)
        
        # Calculate mean recall for each K
        mean_recall_at_k = {}
        for k, recalls in recall_at_k.items():
            mean_recall_at_k[k] = np.mean(recalls) if recalls else 0.0
            logger.info(f"R@{k}: {mean_recall_at_k[k]:.4f}")
        
        return mean_recall_at_k
    
    def run_full_evaluation(self, queries_path, qrels_path, output_dir="evaluation_results"):
        """
        Run complete evaluation and save results.
        
        Args:
            queries_path (str): Path to queries file
            qrels_path (str): Path to qrels file
            output_dir (str): Directory to save results
        """
        logger.info("🚀 Starting QUORA TF-IDF evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models and data
        self.load_models()
        self.load_evaluation_data(queries_path, qrels_path)
        
        # Calculate metrics
        map_score = self.calculate_map()
        precision_at_k = self.calculate_precision_at_k()
        recall_at_k = self.calculate_recall_at_k()
        
        # Prepare results
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset": "QUORA",
            "model": "TF-IDF",
            "metrics": {
                "map": map_score,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k
            },
            "model_info": {
                "vocabulary_size": len(self.feature_names) if self.feature_names else 0,
                "matrix_shape": list(self.tfidf_matrix.shape),
                "num_documents": len(self.index_to_doc_id),
                "num_queries": len(self.queries_df),
                "num_qrels": len(self.qrels_df)
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"quora_tfidf_evaluation_{timestamp}.json"
        csv_filename = f"quora_tfidf_evaluation_{timestamp}.csv"
        
        # Save JSON
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Results saved to {json_path}")
        
        # Save CSV
        csv_data = []
        csv_data.append(["Metric", "Value"])
        csv_data.append(["MAP", map_score])
        
        for k, precision in precision_at_k.items():
            csv_data.append([f"P@{k}", precision])
        
        for k, recall in recall_at_k.items():
            csv_data.append([f"R@{k}", recall])
        
        csv_path = os.path.join(output_dir, csv_filename)
        pd.DataFrame(csv_data[1:], columns=csv_data[0]).to_csv(csv_path, index=False)
        logger.info(f"✅ Results saved to {csv_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("🎯 QUORA TF-IDF EVALUATION RESULTS")
        print("="*50)
        print(f"MAP: {map_score:.4f}")
        print("\nPrecision@K:")
        for k, precision in precision_at_k.items():
            print(f"  P@{k}: {precision:.4f}")
        print("\nRecall@K:")
        for k, recall in recall_at_k.items():
            print(f"  R@{k}: {recall:.4f}")
        print("="*50)
        
        return results

def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QUORA TF-IDF Evaluation")
    parser.add_argument("--models-path", default="models", help="Path to models directory")
    parser.add_argument("--queries-path", required=True, help="Path to queries TSV file")
    parser.add_argument("--qrels-path", required=True, help="Path to qrels TSV file")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = QuoraTFIDFEvaluator(args.models_path)
    evaluator.run_full_evaluation(args.queries_path, args.qrels_path, args.output_dir)

if __name__ == "__main__":
    main()
