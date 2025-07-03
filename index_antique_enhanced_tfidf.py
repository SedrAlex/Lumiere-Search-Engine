#!/usr/bin/env python3
"""
ANTIQUE Enhanced TF-IDF Indexing Script with Batch Processing
============================================================

This script indexes the ANTIQUE dataset using enhanced TF-IDF parameters
with efficient batch processing for large datasets (404k documents).

Features:
- Batch processing with configurable batch size (default: 10,000)
- Memory-efficient document loading and processing
- Enhanced TF-IDF parameters for better MAP scores
- Progress tracking and time estimation
- Incremental model building and saving
- LSA semantic similarity enhancement
- Query expansion preparation
"""

import os
import sys
import time
import gc
import math
import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
from collections import defaultdict, Counter
from tqdm import tqdm
import psutil

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import scipy.sparse as sp

# Dataset and text processing
import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('antique_enhanced_tfidf_indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTFIDFTextCleaner:
    """Enhanced text cleaning service for TF-IDF indexing"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
            logger.info(f"✅ Loaded {len(self.stop_words)} English stopwords")
        except Exception as e:
            logger.warning(f"⚠️ Could not load stopwords: {e}")
            self.stop_words = set()
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """Clean text with enhanced parameters for TF-IDF"""
        if not text or not text.strip():
            return ""
        
        # Step 1: Basic cleaning
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Keep only alphanumeric
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        # Step 2: Tokenization
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Step 3: Token filtering
        filtered_tokens = []
        for token in tokens:
            if len(token) >= 2 and token.isalnum() and token not in self.stop_words:
                # Lemmatize then stem
                lemmatized = self.lemmatizer.lemmatize(token)
                stemmed = self.stemmer.stem(lemmatized)
                filtered_tokens.append(stemmed)
        
        return " ".join(filtered_tokens)

class BatchedAntiqueIndexer:
    """Efficient batch processing indexer for ANTIQUE dataset"""
    
    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size
        self.text_cleaner = EnhancedTFIDFTextCleaner()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.lsa_model = None
        self.lsa_vectors = None
        self.documents = {}
        self.document_order = []
        self.total_docs = 0
        
        # Enhanced vectorizer parameters for better MAP scores
        self.vectorizer_params = {
            'max_features': 100000,      # Increased vocabulary size
            'ngram_range': (1, 3),       # Include trigrams
            'min_df': 2,                 # Minimum document frequency
            'max_df': 0.85,              # Maximum document frequency
            'sublinear_tf': True,        # Log normalization
            'norm': 'l2',                # L2 normalization
            'use_idf': True,             # IDF weighting
            'smooth_idf': True,          # Smooth IDF
            'token_pattern': r'(?u)\b\w\w+\b',
            'strip_accents': 'unicode'
        }
        
        # Query expansion data
        self.term_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Output paths
        self.output_dir = Path("models/antique_enhanced_tfidf")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    
    def load_antique_dataset(self) -> Iterator[Tuple[str, str]]:
        """Load ANTIQUE dataset with memory-efficient iteration"""
        logger.info("📚 Loading ANTIQUE dataset...")
        
        try:
            dataset = ir_datasets.load('antique/train')
            
            # Count total documents first
            logger.info("🔢 Counting total documents...")
            doc_count = 0
            for _ in dataset.docs_iter():
                doc_count += 1
                if doc_count % 50000 == 0:
                    logger.info(f"  Counted {doc_count:,} documents...")
            
            self.total_docs = doc_count
            logger.info(f"✅ Total documents in dataset: {self.total_docs:,}")
            
            # Yield documents one by one
            for doc in dataset.docs_iter():
                yield doc.doc_id, doc.text
                
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def process_batch(self, batch_docs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """Process a batch of documents"""
        doc_ids = []
        cleaned_texts = []
        
        logger.info(f"🧹 Cleaning batch of {len(batch_docs)} documents...")
        
        for doc_id, text in tqdm(batch_docs, desc="Cleaning"):
            cleaned_text = self.text_cleaner.clean_text(text)
            if cleaned_text.strip():  # Only include non-empty documents
                doc_ids.append(doc_id)
                cleaned_texts.append(cleaned_text)
                self.documents[doc_id] = {
                    'text': text,
                    'cleaned_text': cleaned_text,
                    'length': len(text.split())
                }
        
        return doc_ids, cleaned_texts
    
    def build_term_cooccurrence(self, texts: List[str]) -> None:
        """Build term co-occurrence matrix for query expansion"""
        logger.info("🔗 Building term co-occurrence data...")
        
        for text in tqdm(texts, desc="Co-occurrence"):
            terms = text.split()
            for i, term1 in enumerate(terms):
                for j, term2 in enumerate(terms[max(0, i-5):i+6]):  # 5-word window
                    if i != j:
                        self.term_cooccurrence[term1][term2] += 1
    
    def fit_vectorizer_incremental(self, all_texts: List[str]) -> None:
        """Fit vectorizer on all texts"""
        logger.info("🔧 Fitting TF-IDF vectorizer...")
        
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        
        # Fit on all texts at once (memory permitting)
        self.vectorizer.fit(all_texts)
        
        logger.info(f"✅ Vectorizer fitted with vocabulary size: {len(self.vectorizer.vocabulary_):,}")
    
    def transform_batch(self, texts: List[str]) -> sp.csr_matrix:
        """Transform a batch of texts to TF-IDF vectors"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet!")
        
        return self.vectorizer.transform(texts)
    
    def estimate_processing_time(self, sample_batch_time: float) -> Dict[str, Any]:
        """Estimate total processing time based on sample batch"""
        total_batches = math.ceil(self.total_docs / self.batch_size)
        estimated_total_time = sample_batch_time * total_batches
        
        return {
            'total_batches': total_batches,
            'estimated_batch_time': sample_batch_time,
            'estimated_total_time_seconds': estimated_total_time,
            'estimated_total_time_minutes': estimated_total_time / 60,
            'estimated_total_time_hours': estimated_total_time / 3600,
        }
    
    def save_incremental_progress(self, batch_num: int, batch_matrix: sp.csr_matrix) -> None:
        """Save incremental progress"""
        batch_file = self.output_dir / f"batch_{batch_num:04d}_matrix.npz"
        sp.save_npz(batch_file, batch_matrix)
        
        # Save metadata
        metadata = {
            'batch_num': batch_num,
            'batch_size': batch_matrix.shape[0],
            'features': batch_matrix.shape[1],
            'timestamp': time.time()
        }
        
        metadata_file = self.output_dir / f"batch_{batch_num:04d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def combine_batch_matrices(self, num_batches: int) -> sp.csr_matrix:
        """Combine all batch matrices into final matrix"""
        logger.info("🔗 Combining batch matrices...")
        
        matrices = []
        for batch_num in range(num_batches):
            batch_file = self.output_dir / f"batch_{batch_num:04d}_matrix.npz"
            if batch_file.exists():
                batch_matrix = sp.load_npz(batch_file)
                matrices.append(batch_matrix)
                logger.info(f"  Loaded batch {batch_num}: {batch_matrix.shape}")
        
        if not matrices:
            raise ValueError("No batch matrices found!")
        
        # Combine matrices vertically
        combined_matrix = sp.vstack(matrices, format='csr')
        logger.info(f"✅ Combined matrix shape: {combined_matrix.shape}")
        
        # Clean up batch files
        for batch_num in range(num_batches):
            batch_file = self.output_dir / f"batch_{batch_num:04d}_matrix.npz"
            metadata_file = self.output_dir / f"batch_{batch_num:04d}_metadata.json"
            if batch_file.exists():
                batch_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
        
        return combined_matrix
    
    def build_lsa_model(self, tfidf_matrix: sp.csr_matrix) -> None:
        """Build LSA model for semantic similarity"""
        logger.info("🧠 Building LSA model...")
        
        n_components = min(300, tfidf_matrix.shape[1] - 1)
        self.lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        logger.info(f"  Fitting LSA with {n_components} components...")
        self.lsa_vectors = self.lsa_model.fit_transform(tfidf_matrix)
        self.lsa_vectors = normalize(self.lsa_vectors, norm='l2')
        
        logger.info(f"✅ LSA model built with explained variance: {self.lsa_model.explained_variance_ratio_.sum():.3f}")
    
    def save_models(self) -> None:
        """Save all models and data"""
        logger.info("💾 Saving models and data...")
        
        # Save vectorizer
        vectorizer_path = self.output_dir / "tfidf_vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"✅ Saved vectorizer: {vectorizer_path}")
        
        # Save TF-IDF matrix
        matrix_path = self.output_dir / "tfidf_matrix.npz"
        sp.save_npz(matrix_path, self.tfidf_matrix)
        logger.info(f"✅ Saved TF-IDF matrix: {matrix_path}")
        
        # Save LSA model and vectors
        if self.lsa_model is not None:
            lsa_model_path = self.output_dir / "lsa_model.joblib"
            lsa_vectors_path = self.output_dir / "lsa_vectors.joblib"
            
            joblib.dump(self.lsa_model, lsa_model_path)
            joblib.dump(self.lsa_vectors, lsa_vectors_path)
            
            logger.info(f"✅ Saved LSA model: {lsa_model_path}")
            logger.info(f"✅ Saved LSA vectors: {lsa_vectors_path}")
        
        # Save document metadata
        metadata_path = self.output_dir / "document_metadata.joblib"
        document_metadata = {
            'documents': self.documents,
            'document_order': self.document_order,
            'total_documents': len(self.document_order)
        }
        joblib.dump(document_metadata, metadata_path)
        logger.info(f"✅ Saved document metadata: {metadata_path}")
        
        # Save query expansion data
        if self.term_cooccurrence:
            cooccurrence_path = self.output_dir / "term_cooccurrence.joblib"
            joblib.dump(dict(self.term_cooccurrence), cooccurrence_path)
            logger.info(f"✅ Saved term co-occurrence: {cooccurrence_path}")
        
        # Save training info
        training_info = {
            'dataset': 'antique/train',
            'total_documents': len(self.document_order),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'vectorizer_params': self.vectorizer_params,
            'matrix_shape': list(self.tfidf_matrix.shape),
            'batch_size': self.batch_size,
            'features': [
                'enhanced_tfidf_parameters',
                'batch_processing',
                'lsa_semantic_similarity',
                'query_expansion_preparation',
                'memory_efficient_processing'
            ],
            'timestamp': time.time()
        }
        
        if self.lsa_model is not None:
            training_info['lsa_components'] = self.lsa_model.n_components
            training_info['lsa_explained_variance'] = float(self.lsa_model.explained_variance_ratio_.sum())
        
        info_path = self.output_dir / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        logger.info(f"✅ Saved training info: {info_path}")
    
    def run_indexing(self) -> None:
        """Run the complete indexing process"""
        start_time = time.time()
        logger.info("🚀 Starting ANTIQUE Enhanced TF-IDF Indexing")
        logger.info(f"📊 Batch size: {self.batch_size:,}")
        logger.info(f"💾 Initial memory usage: {self.get_memory_usage():.2f} GB")
        
        # Collect all texts first for vectorizer fitting
        all_texts = []
        all_doc_ids = []
        batch_count = 0
        
        # Process in batches
        batch_docs = []
        sample_batch_time = None
        
        for doc_id, text in self.load_antique_dataset():
            batch_docs.append((doc_id, text))
            
            if len(batch_docs) >= self.batch_size:
                batch_start_time = time.time()
                
                # Process batch
                batch_doc_ids, batch_texts = self.process_batch(batch_docs)
                all_doc_ids.extend(batch_doc_ids)
                all_texts.extend(batch_texts)
                
                batch_time = time.time() - batch_start_time
                if sample_batch_time is None:
                    sample_batch_time = batch_time
                    time_estimate = self.estimate_processing_time(sample_batch_time)
                    logger.info(f"⏱️  Estimated total time: {time_estimate['estimated_total_time_hours']:.1f} hours")
                
                logger.info(f"✅ Processed batch {batch_count + 1}, docs so far: {len(all_doc_ids):,}")
                logger.info(f"💾 Memory usage: {self.get_memory_usage():.2f} GB")
                
                batch_count += 1
                batch_docs = []
                
                # Force garbage collection
                gc.collect()
        
        # Process remaining documents
        if batch_docs:
            batch_doc_ids, batch_texts = self.process_batch(batch_docs)
            all_doc_ids.extend(batch_doc_ids)
            all_texts.extend(batch_texts)
            batch_count += 1
        
        self.document_order = all_doc_ids
        logger.info(f"📊 Total processed documents: {len(all_doc_ids):,}")
        
        # Fit vectorizer on all texts
        logger.info("🔧 Fitting vectorizer on all texts...")
        self.fit_vectorizer_incremental(all_texts)
        
        # Transform in batches to save memory
        logger.info("🔄 Transforming documents in batches...")
        batch_matrices = []
        
        for i in range(0, len(all_texts), self.batch_size):
            batch_texts = all_texts[i:i + self.batch_size]
            batch_matrix = self.transform_batch(batch_texts)
            
            # Save batch matrix temporarily
            batch_num = i // self.batch_size
            self.save_incremental_progress(batch_num, batch_matrix)
            
            logger.info(f"  Transformed batch {batch_num + 1}/{math.ceil(len(all_texts) / self.batch_size)}")
            
            # Clear memory
            del batch_matrix
            gc.collect()
        
        # Combine all batch matrices
        total_batches = math.ceil(len(all_texts) / self.batch_size)
        self.tfidf_matrix = self.combine_batch_matrices(total_batches)
        
        # Build LSA model
        self.build_lsa_model(self.tfidf_matrix)
        
        # Build term co-occurrence (sample for efficiency)
        sample_size = min(50000, len(all_texts))
        sample_texts = all_texts[:sample_size]
        self.build_term_cooccurrence(sample_texts)
        
        # Save all models
        self.save_models()
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Indexing completed in {total_time / 3600:.2f} hours!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"  - Documents indexed: {len(self.document_order):,}")
        logger.info(f"  - Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        logger.info(f"  - Matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"  - Memory usage: {self.get_memory_usage():.2f} GB")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index ANTIQUE dataset with Enhanced TF-IDF")
    parser.add_argument('--batch-size', type=int, default=10000, 
                       help='Batch size for processing (default: 10000)')
    parser.add_argument('--output-dir', type=str, default='models/antique_enhanced_tfidf',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = BatchedAntiqueIndexer(batch_size=args.batch_size)
    if args.output_dir:
        indexer.output_dir = Path(args.output_dir)
        indexer.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        indexer.run_indexing()
        logger.info("✅ Indexing completed successfully!")
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")
        raise

if __name__ == "__main__":
    main()
