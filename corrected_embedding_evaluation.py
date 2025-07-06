# ===================================================================
# CORRECTED Embedding Evaluation for Antique Dataset
# Fixes preprocessing consistency issues for better MAP scores
# ===================================================================

import ir_datasets
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
import ir_measures
from ir_measures import *
import torch
import gc
import os
from typing import List, Dict, Any
import json
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("✅ Libraries imported successfully!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# STEP 1: Load EXACT Same Text Cleaning Service as Training
class UnifiedTextCleaningService:
    """EXACT COPY from training notebook to ensure perfect consistency"""
    
    def __init__(self):
        # Initialize NLTK components exactly like training
        self.stemmer = None
        self.stop_words = set()
        
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            print("✅ NLTK components initialized (matching training)")
        except Exception as e:
            print(f"❌ Error initializing NLTK: {e}")
    
    def clean_text(self, text: str, 
                   remove_stopwords: bool = True,
                   apply_stemming: bool = True, 
                   apply_lemmatization: bool = False) -> str:
        """EXACT COPY from training to ensure consistency"""
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Basic cleaning (same as training)
        cleaned_text = self._basic_clean(text)
        
        # Step 2: Tokenization (same as training)
        tokens = self._tokenize(cleaned_text)
        
        # Step 3: Remove stopwords (same as training)
        if remove_stopwords and self.stop_words:
            tokens = self._remove_stopwords(tokens)
        
        # Step 4: Stemming (same as training)
        if apply_stemming and self.stemmer:
            tokens = self._apply_stemming(tokens)
        
        return " ".join(tokens)
    
    def _basic_clean(self, text: str) -> str:
        """EXACT COPY from training"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)
        text = text.strip()
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """EXACT COPY from training - uses NLTK word_tokenize when available"""
        if not text:
            return []
        
        if self.stemmer:  # NLTK available (same logic as training)
            try:
                tokens = word_tokenize(text)
                return [token.lower() for token in tokens if token.isalnum()]
            except:
                pass
        
        # Fallback tokenization (same as training)
        return [word.strip() for word in text.lower().split() if word.strip()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """EXACT COPY from training"""
        if not self.stop_words:
            return tokens
        return [token for token in tokens if token not in self.stop_words]
    
    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        """EXACT COPY from training"""
        if not self.stemmer:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]

# Initialize text cleaner (same as training)
text_cleaner = UnifiedTextCleaningService()
print("✅ Unified text cleaner initialized (matching training)!")

# STEP 2: Load Model and Embeddings
def load_embedding_model_and_data(embeddings_path: str, metadata_path: str):
    """Load all components needed for evaluation"""
    print("📂 Loading embedding model and data...")
    
    # Load SentenceTransformer model
    print("🚀 Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Load embeddings matrix
    print("📊 Loading embeddings matrix...")
    embeddings_matrix = joblib.load(embeddings_path)
    print(f"✅ Embeddings matrix loaded with shape {embeddings_matrix.shape}")
    
    # Load document metadata
    print("📄 Loading document metadata...")
    metadata = joblib.load(metadata_path)
    documents = metadata['documents']
    document_order = metadata['document_order']
    cleaned_texts = metadata['cleaned_texts']
    
    print(f"✅ Metadata loaded with {len(documents):,} documents")
    print(f"✅ Document order aligned: {len(document_order):,}")
    
    # Verify alignment
    assert len(documents) == len(document_order) == len(cleaned_texts) == embeddings_matrix.shape[0]
    print("✅ Perfect alignment verified!")
    
    # Create docid to index mapping
    docid_to_index = {doc_id: idx for idx, doc_id in enumerate(document_order)}
    
    return model, embeddings_matrix, documents, document_order, docid_to_index

# Load the model and data (update these paths)
EMBEDDINGS_PATH = "antique_embeddings_matrix.joblib"
METADATA_PATH = "antique_embedding_document_metadata.joblib"

model, embeddings_matrix, documents, document_order, docid_to_index = load_embedding_model_and_data(
    EMBEDDINGS_PATH, METADATA_PATH
)

# STEP 3: Load Antique Test Set
def load_antique_test_set():
    """Load test queries and qrels from Antique dataset"""
    print("📚 Loading Antique test set...")
    
    dataset = ir_datasets.load('antique/test')
    
    test_queries = []
    test_qrels = {}
    
    print("Loading test queries...")
    for query in dataset.queries_iter():
        test_queries.append({
            'query_id': query.query_id,
            'text': query.text
        })
    
    print("Loading test qrels...")
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in test_qrels:
            test_qrels[qrel.query_id] = {}
        test_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    
    print(f"✅ Test set loaded successfully!")
    print(f"   🔍 Test queries: {len(test_queries):,}")
    print(f"   🎯 Qrels: {len(test_qrels):,}")
    
    return test_queries, test_qrels

test_queries, test_qrels = load_antique_test_set()

# STEP 4: CORRECTED Evaluation with Consistent Preprocessing
def evaluate_with_consistent_preprocessing(model, embeddings_matrix, documents, document_order,
                                         docid_to_index, test_queries, test_qrels, top_k=100):
    """
    CORRECTED evaluation using EXACT same preprocessing as training
    """
    
    print("🧪 Running CORRECTED evaluation with consistent preprocessing...")
    print("🔧 Using SAME preprocessing parameters as training:")
    print("   📄 Documents: remove_stopwords=True, apply_stemming=True")
    print("   🔍 Queries: remove_stopwords=True, apply_stemming=True (SAME AS DOCUMENTS)")
    
    # Process each query and collect results
    run_results = []
    query_times = []
    
    for query in tqdm(test_queries, desc="Processing queries"):
        query_id = query['query_id']
        query_text = query['text']
        
        # Skip queries without relevant documents
        if query_id not in test_qrels:
            continue
        
        start_time = time.time()
        
        # CORRECTED: Use EXACT same preprocessing as documents in training
        cleaned_query = text_cleaner.clean_text(
            query_text,
            remove_stopwords=True,    # FIXED: Same as documents (was False)
            apply_stemming=True,      # Same as documents
            apply_lemmatization=False # Same as documents
        )
        
        # Generate query embedding
        with torch.no_grad():
            query_embedding = model.encode(
                [cleaned_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Convert to run format for ir_measures
        for rank, idx in enumerate(top_indices, 1):
            doc_id = document_order[idx]
            score = similarities[idx]
            run_results.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'score': float(score),
                'rank': rank
            })
        
        query_times.append(time.time() - start_time)
    
    # Convert to pandas DataFrame for ir_measures
    run_df = pd.DataFrame(run_results)
    
    # Convert qrels to proper format
    qrel_list = []
    for query_id, docs in test_qrels.items():
        for doc_id, rel in docs.items():
            qrel_list.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'relevance': rel
            })
    qrels_df = pd.DataFrame(qrel_list)
    
    # Calculate metrics
    print("\n📊 CORRECTED Evaluation Results:")
    metrics = ir_measures.calc_aggregate(
        [AP@100, P@10, P@100, R@100, nDCG@100],
        qrels_df,
        run_df
    )
    
    avg_query_time = np.mean(query_times) * 1000  # in milliseconds
    
    # Print results
    print(f"🔍 Queries evaluated: {len([q for q in test_queries if q['query_id'] in test_qrels]):,}")
    print(f"⏱️  Average query time: {avg_query_time:.2f} ms")
    print(f"📊 MAP@100: {metrics[AP@100]:.4f}")
    print(f"📊 P@10: {metrics[P@10]:.4f}")
    print(f"📊 P@100: {metrics[P@100]:.4f}")
    print(f"📊 R@100: {metrics[R@100]:.4f}")
    print(f"📊 nDCG@100: {metrics[nDCG@100]:.4f}")
    
    return metrics

# Run CORRECTED evaluation
print("\n" + "="*70)
print("🔧 RUNNING CORRECTED EVALUATION")
print("="*70)

corrected_metrics = evaluate_with_consistent_preprocessing(
    model, embeddings_matrix, documents, document_order,
    docid_to_index, test_queries, test_qrels
)

# STEP 5: Compare with Training Data Evaluation
def evaluate_on_training_data(model, embeddings_matrix, documents, document_order, top_k=100):
    """Evaluate on training data to check for overfitting"""
    print("\n🔬 Evaluating on TRAINING data for comparison...")
    
    # Load training queries and qrels
    train_dataset = ir_datasets.load('antique/train')
    
    train_queries = []
    train_qrels = {}
    
    print("Loading training queries...")
    for query in train_dataset.queries_iter():
        train_queries.append({
            'query_id': query.query_id,
            'text': query.text
        })
    
    print("Loading training qrels...")
    for qrel in train_dataset.qrels_iter():
        if qrel.query_id not in train_qrels:
            train_qrels[qrel.query_id] = {}
        train_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    
    # Sample subset for evaluation (training set is large)
    sample_queries = train_queries[:50]  # Evaluate on first 50 training queries
    
    run_results = []
    
    for query in tqdm(sample_queries, desc="Evaluating training queries"):
        query_id = query['query_id']
        query_text = query['text']
        
        if query_id not in train_qrels:
            continue
        
        # Use same preprocessing
        cleaned_query = text_cleaner.clean_text(
            query_text,
            remove_stopwords=True,
            apply_stemming=True,
            apply_lemmatization=False
        )
        
        # Generate query embedding
        with torch.no_grad():
            query_embedding = model.encode(
                [cleaned_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for rank, idx in enumerate(top_indices, 1):
            doc_id = document_order[idx]
            score = similarities[idx]
            run_results.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'score': float(score),
                'rank': rank
            })
    
    # Calculate metrics
    run_df = pd.DataFrame(run_results)
    
    qrel_list = []
    for query_id in [q['query_id'] for q in sample_queries]:
        if query_id in train_qrels:
            for doc_id, rel in train_qrels[query_id].items():
                qrel_list.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance': rel
                })
    qrels_df = pd.DataFrame(qrel_list)
    
    if len(run_df) > 0 and len(qrels_df) > 0:
        train_metrics = ir_measures.calc_aggregate(
            [AP@100, P@10, P@100, R@100, nDCG@100],
            qrels_df,
            run_df
        )
        
        print(f"📊 TRAINING MAP@100: {train_metrics[AP@100]:.4f}")
        print(f"📊 TRAINING P@10: {train_metrics[P@10]:.4f}")
        print(f"📊 TRAINING nDCG@100: {train_metrics[nDCG@100]:.4f}")
        
        return train_metrics
    else:
        print("❌ Could not evaluate on training data")
        return None

# Evaluate on training data
train_metrics = evaluate_on_training_data(model, embeddings_matrix, documents, document_order)

# STEP 6: Save Results and Analysis
def save_corrected_results(test_metrics, train_metrics=None):
    """Save corrected evaluation results"""
    results = {
        'model': 'all-MiniLM-L6-v2',
        'evaluation_type': 'corrected_preprocessing',
        'test_metrics': {
            'MAP@100': float(test_metrics[AP@100]),
            'P@10': float(test_metrics[P@10]),
            'P@100': float(test_metrics[P@100]),
            'R@100': float(test_metrics[R@100]),
            'nDCG@100': float(test_metrics[nDCG@100])
        },
        'preprocessing_corrections': {
            'issue_fixed': 'Query preprocessing now matches document preprocessing',
            'before': 'remove_stopwords=False for queries',
            'after': 'remove_stopwords=True for queries (same as documents)',
            'tokenization': 'Now uses NLTK word_tokenize consistently'
        }
    }
    
    if train_metrics:
        results['train_metrics'] = {
            'MAP@100': float(train_metrics[AP@100]),
            'P@10': float(train_metrics[P@10]),
            'P@100': float(train_metrics[P@100]),
            'R@100': float(train_metrics[R@100]),
            'nDCG@100': float(train_metrics[nDCG@100])
        }
    
    with open('corrected_embedding_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Corrected results saved to corrected_embedding_evaluation.json")

save_corrected_results(corrected_metrics, train_metrics)

# STEP 7: Analysis and Recommendations
print("\n" + "="*70)
print("📊 EVALUATION ANALYSIS")
print("="*70)

test_map = float(corrected_metrics[AP@100])
print(f"🎯 CORRECTED Test MAP@100: {test_map:.4f}")

if train_metrics:
    train_map = float(train_metrics[AP@100])
    print(f"🎯 Training MAP@100: {train_map:.4f}")
    print(f"📈 Performance gap: {train_map - test_map:.4f}")

print(f"\n🔍 Expected MAP ranges for Antique:")
print(f"   📊 TF-IDF baseline: ~0.17 (your current system)")
print(f"   📊 Basic embeddings: 0.15-0.25")
print(f"   📊 Tuned embeddings: 0.25-0.35")
print(f"   📊 SOTA methods: 0.35-0.45")

if test_map < 0.2:
    print(f"\n⚠️  MAP is still below expected range. Consider:")
    print(f"   🔧 Fine-tuning the SentenceTransformer model")
    print(f"   🔧 Using different embedding models (e.g., all-mpnet-base-v2)")
    print(f"   🔧 Query expansion techniques")
    print(f"   🔧 Hybrid retrieval (TF-IDF + embeddings)")
elif test_map >= 0.2:
    print(f"\n✅ MAP is in acceptable range for basic embeddings!")
    print(f"   🎯 Consider fine-tuning for further improvements")

print(f"\n✅ Corrected evaluation completed!")
print(f"🔧 Preprocessing consistency issues have been fixed!")
