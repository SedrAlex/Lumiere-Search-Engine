# ===================================================================
# COMPREHENSIVE Embedding Diagnostic Script
# Identifies why MAP is low and provides specific fixes
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
from collections import defaultdict
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("🔍 EMBEDDING DIAGNOSTIC SCRIPT")
print("="*50)
print("This script will identify why your embedding MAP is low")

# STEP 1: Load the exact same text cleaner from training
class UnifiedTextCleaningService:
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            print("✅ Text cleaner initialized")
        except Exception as e:
            print(f"❌ Error initializing NLTK: {e}")
    
    def clean_text(self, text: str, 
                   remove_stopwords: bool = True,
                   apply_stemming: bool = True, 
                   apply_lemmatization: bool = False) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        cleaned_text = self._basic_clean(text)
        tokens = self._tokenize(cleaned_text)
        
        if remove_stopwords and self.stop_words:
            tokens = self._remove_stopwords(tokens)
        
        if apply_stemming and self.stemmer:
            tokens = self._apply_stemming(tokens)
        
        return " ".join(tokens)
    
    def _basic_clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)
        text = text.strip()
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        if self.stemmer:
            try:
                tokens = word_tokenize(text)
                return [token.lower() for token in tokens if token.isalnum()]
            except:
                pass
        
        return [word.strip() for word in text.lower().split() if word.strip()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        if not self.stop_words:
            return tokens
        return [token for token in tokens if token not in self.stop_words]
    
    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        if not self.stemmer:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]

text_cleaner = UnifiedTextCleaningService()

# STEP 2: Diagnostic Functions
def diagnose_embedding_files():
    """Check if embedding files exist and are valid"""
    print("\n🔍 DIAGNOSTIC 1: Checking embedding files...")
    
    required_files = [
        "antique_embeddings_matrix.joblib",
        "antique_embedding_document_metadata.joblib"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            try:
                if "matrix" in file_path:
                    matrix = joblib.load(file_path)
                    print(f"✅ {file_path}: {matrix.shape}, dtype: {matrix.dtype}")
                else:
                    metadata = joblib.load(file_path)
                    print(f"✅ {file_path}: {len(metadata.get('documents', []))} documents")
            except Exception as e:
                print(f"❌ {file_path}: Error loading - {e}")
        else:
            print(f"❌ {file_path}: File not found")
            return False
    
    return True

def diagnose_preprocessing_impact():
    """Test different preprocessing combinations"""
    print("\n🔍 DIAGNOSTIC 2: Testing preprocessing impact...")
    
    # Load test data
    dataset = ir_datasets.load('antique/test')
    sample_queries = []
    for i, query in enumerate(dataset.queries_iter()):
        sample_queries.append(query.text)
        if i >= 5:  # Just test 5 queries
            break
    
    print("Testing preprocessing combinations:")
    combinations = [
        {"remove_stopwords": True, "apply_stemming": True, "name": "Training (remove_stop=True, stem=True)"},
        {"remove_stopwords": False, "apply_stemming": True, "name": "Original Eval (remove_stop=False, stem=True)"},
        {"remove_stopwords": False, "apply_stemming": False, "name": "No preprocessing"},
        {"remove_stopwords": True, "apply_stemming": False, "name": "Only remove stopwords"},
    ]
    
    for combo in combinations:
        print(f"\n📝 {combo['name']}:")
        for query in sample_queries[:2]:  # Show 2 examples
            cleaned = text_cleaner.clean_text(
                query, 
                remove_stopwords=combo["remove_stopwords"],
                apply_stemming=combo["apply_stemming"]
            )
            print(f"   Original: '{query[:60]}...'")
            print(f"   Cleaned:  '{cleaned[:60]}...'")

def diagnose_embeddings_quality(model, embeddings_matrix, documents, document_order):
    """Check embedding quality with similarity analysis"""
    print("\n🔍 DIAGNOSTIC 3: Analyzing embedding quality...")
    
    # Test semantic similarity
    test_queries = [
        "how to lose weight",
        "what is gravity", 
        "programming languages",
        "medical treatment",
        "cooking recipe"
    ]
    
    print("Testing semantic similarity:")
    for query in test_queries:
        cleaned_query = text_cleaner.clean_text(query, remove_stopwords=True, apply_stemming=True)
        
        with torch.no_grad():
            query_embedding = model.encode([cleaned_query], convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\n🔍 Query: '{query}' -> '{cleaned_query}'")
        for i, idx in enumerate(top_indices, 1):
            doc_id = document_order[idx]
            score = similarities[idx]
            doc_text = documents[idx]['text'][:100]
            print(f"   {i}. {doc_id} (score={score:.4f}): {doc_text}...")

def diagnose_train_test_distribution():
    """Compare training and test data distributions"""
    print("\n🔍 DIAGNOSTIC 4: Comparing train/test distributions...")
    
    # Load both datasets
    train_dataset = ir_datasets.load('antique/train')
    test_dataset = ir_datasets.load('antique/test')
    
    # Sample queries from both
    train_queries = []
    test_queries = []
    
    for i, query in enumerate(train_dataset.queries_iter()):
        train_queries.append(query.text)
        if i >= 100:
            break
    
    for i, query in enumerate(test_dataset.queries_iter()):
        test_queries.append(query.text)
        if i >= 100:
            break
    
    # Analyze query lengths
    train_lengths = [len(q.split()) for q in train_queries]
    test_lengths = [len(q.split()) for q in test_queries]
    
    print(f"📊 Train query length: avg={np.mean(train_lengths):.1f}, std={np.std(train_lengths):.1f}")
    print(f"📊 Test query length: avg={np.mean(test_lengths):.1f}, std={np.std(test_lengths):.1f}")
    
    # Analyze vocabulary overlap
    train_vocab = set()
    test_vocab = set()
    
    for query in train_queries:
        cleaned = text_cleaner.clean_text(query, remove_stopwords=True, apply_stemming=True)
        train_vocab.update(cleaned.split())
    
    for query in test_queries:
        cleaned = text_cleaner.clean_text(query, remove_stopwords=True, apply_stemming=True)
        test_vocab.update(cleaned.split())
    
    overlap = len(train_vocab & test_vocab)
    total_unique = len(train_vocab | test_vocab)
    
    print(f"📊 Vocabulary overlap: {overlap}/{total_unique} ({overlap/total_unique*100:.1f}%)")

def diagnose_baseline_comparison():
    """Compare with simple TF-IDF baseline"""
    print("\n🔍 DIAGNOSTIC 5: Comparing with TF-IDF baseline...")
    
    try:
        # Try to load your TF-IDF results for comparison
        tfidf_results_file = "evaluation_results/proper_antique_eval_20250630_213405.json"
        if os.path.exists(tfidf_results_file):
            with open(tfidf_results_file, 'r') as f:
                tfidf_results = json.load(f)
            
            tfidf_map = tfidf_results.get('with_cleaning', {}).get('aggregated', {}).get('MAP', 0)
            print(f"📊 Your TF-IDF MAP: {tfidf_map:.4f}")
            print(f"📊 Current Embedding MAP: ~0.14")
            print(f"📊 Performance gap: {tfidf_map - 0.14:.4f}")
            
            if tfidf_map > 0.14:
                print("⚠️  TF-IDF outperforms embeddings - this suggests an issue!")
        else:
            print("📊 TF-IDF results not found for comparison")
    except Exception as e:
        print(f"❌ Error loading TF-IDF results: {e}")

def test_different_embedding_models():
    """Test if the issue is with the specific embedding model"""
    print("\n🔍 DIAGNOSTIC 6: Testing different embedding models...")
    
    models_to_test = [
        "all-MiniLM-L6-v2",  # Current model
        "all-mpnet-base-v2", # Usually better
        "sentence-transformers/all-MiniLM-L12-v2"  # Larger version
    ]
    
    sample_query = "how to lose weight fast"
    sample_docs = [
        "weight loss tips and diet advice",
        "fast food restaurants near me", 
        "computer programming tutorial",
        "healthy eating and exercise plans"
    ]
    
    print("Testing semantic understanding across models:")
    for model_name in models_to_test:
        try:
            print(f"\n🤖 Testing {model_name}:")
            model = SentenceTransformer(model_name)
            
            query_emb = model.encode([sample_query], normalize_embeddings=True)
            doc_embs = model.encode(sample_docs, normalize_embeddings=True)
            
            similarities = cosine_similarity(query_emb, doc_embs)[0]
            
            for i, (doc, sim) in enumerate(zip(sample_docs, similarities)):
                relevance = "✅" if i in [0, 3] else "❌"  # Docs 0,3 are relevant
                print(f"   {relevance} {sim:.4f}: {doc}")
                
        except Exception as e:
            print(f"   ❌ Error with {model_name}: {e}")

def run_comprehensive_diagnostic():
    """Run all diagnostics"""
    print("🚀 Starting comprehensive embedding diagnostic...")
    
    # Check files
    if not diagnose_embedding_files():
        print("❌ Cannot proceed - embedding files missing or corrupted")
        return
    
    # Load model and data for testing
    try:
        print("\n📂 Loading model and data for testing...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_matrix = joblib.load("antique_embeddings_matrix.joblib")
        metadata = joblib.load("antique_embedding_document_metadata.joblib")
        documents = metadata['documents']
        document_order = metadata['document_order']
        
        print(f"✅ Loaded: Model, {embeddings_matrix.shape[0]} embeddings, {len(documents)} documents")
        
        # Run all diagnostics
        diagnose_preprocessing_impact()
        diagnose_embeddings_quality(model, embeddings_matrix, documents, document_order)
        diagnose_train_test_distribution()
        diagnose_baseline_comparison()
        test_different_embedding_models()
        
        # Final recommendations
        print("\n" + "="*70)
        print("🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
        print("="*70)
        
        print("""
Based on the diagnostics, here are the most likely causes of low MAP:

1. 🔧 MODEL CHOICE ISSUE:
   - all-MiniLM-L6-v2 might not be optimal for Antique dataset
   - Try all-mpnet-base-v2 or all-MiniLM-L12-v2
   
2. 🔧 PREPROCESSING MISMATCH:
   - Even small differences can hurt performance significantly
   - Verify exact same cleaning between training and evaluation
   
3. 🔧 DOMAIN MISMATCH:
   - Pre-trained embeddings might not capture Antique's domain well
   - Consider fine-tuning on Antique data
   
4. 🔧 EVALUATION METHODOLOGY:
   - Check if document retrieval covers all relevant docs
   - Verify qrels alignment with your document IDs

IMMEDIATE ACTIONS TO TRY:
✅ 1. Use all-mpnet-base-v2 instead of all-MiniLM-L6-v2
✅ 2. Try hybrid approach: TF-IDF + embeddings
✅ 3. Fine-tune the model on Antique training data
✅ 4. Check for document ID mismatches in evaluation
        """)
        
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")
        print("Cannot run embedding quality tests")

# Run the diagnostic
if __name__ == "__main__":
    run_comprehensive_diagnostic()
