# ===================================================================
# QUICK FIX: Better Embedding Model Test
# Tests all-mpnet-base-v2 which usually performs significantly better
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

print("🚀 QUICK FIX: Testing Better Embedding Model")
print("="*50)
print("Testing all-mpnet-base-v2 (usually 20-30% better than all-MiniLM-L6-v2)")

# STEP 1: Same text cleaner as training
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

# STEP 2: Load your existing cleaned documents
def load_cleaned_documents():
    """Load the cleaned documents from your training"""
    print("📂 Loading your existing cleaned documents...")
    
    try:
        metadata = joblib.load("antique_embedding_document_metadata.joblib")
        documents = metadata['documents']
        document_order = metadata['document_order']
        cleaned_texts = metadata['cleaned_texts']
        
        print(f"✅ Loaded {len(documents)} documents with cleaned texts")
        return documents, document_order, cleaned_texts
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return None, None, None

# STEP 3: Generate embeddings with BETTER model
def generate_embeddings_with_better_model(cleaned_texts, model_name="all-mpnet-base-v2"):
    """Generate embeddings using a better model"""
    print(f"🤖 Generating embeddings with {model_name}...")
    print("⚠️  This will take some time but should give much better results")
    
    # Load the better model
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"✅ Model loaded on {device}")
    print(f"📏 Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Generate embeddings in batches
    batch_size = 32
    total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    
    all_embeddings = []
    
    print(f"📊 Processing {len(cleaned_texts)} documents in {total_batches} batches...")
    
    for batch_idx in tqdm(range(0, len(cleaned_texts), batch_size), 
                          desc=f"Generating {model_name} embeddings"):
        
        end_idx = min(batch_idx + batch_size, len(cleaned_texts))
        batch_texts = cleaned_texts[batch_idx:end_idx]
        
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        
        all_embeddings.append(batch_embeddings)
        
        # Clear cache periodically
        if batch_idx % (20 * batch_size) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings)
    
    print(f"✅ Generated {embeddings_matrix.shape[0]} embeddings")
    print(f"📊 Matrix shape: {embeddings_matrix.shape}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, embeddings_matrix

# STEP 4: Quick evaluation with better model
def quick_evaluation_better_model(model, embeddings_matrix, documents, document_order):
    """Quick evaluation with the better model"""
    print("🧪 Running quick evaluation with better model...")
    
    # Load test queries and qrels
    dataset = ir_datasets.load('antique/test')
    
    test_queries = []
    test_qrels = {}
    
    print("Loading test data...")
    for query in dataset.queries_iter():
        test_queries.append({
            'query_id': query.query_id,
            'text': query.text
        })
    
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in test_qrels:
            test_qrels[qrel.query_id] = {}
        test_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    
    # Sample a subset for quick testing (first 50 queries)
    sample_queries = test_queries[:50]
    
    print(f"🧪 Testing on {len(sample_queries)} queries (subset for speed)...")
    
    run_results = []
    
    for query in tqdm(sample_queries, desc="Processing sample queries"):
        query_id = query['query_id']
        query_text = query['text']
        
        if query_id not in test_qrels:
            continue
        
        # Clean query with same method as documents
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
        top_indices = np.argsort(similarities)[::-1][:100]
        
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
        if query_id in test_qrels:
            for doc_id, rel in test_qrels[query_id].items():
                qrel_list.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance': rel
                })
    qrels_df = pd.DataFrame(qrel_list)
    
    if len(run_df) > 0 and len(qrels_df) > 0:
        metrics = ir_measures.calc_aggregate(
            [AP@100, P@10, P@100, R@100, nDCG@100],
            qrels_df,
            run_df
        )
        
        print(f"\n📊 QUICK EVALUATION RESULTS (subset of {len(sample_queries)} queries):")
        print(f"📊 MAP@100: {metrics[AP@100]:.4f}")
        print(f"📊 P@10: {metrics[P@10]:.4f}")
        print(f"📊 P@100: {metrics[P@100]:.4f}")
        print(f"📊 R@100: {metrics[R@100]:.4f}")
        print(f"📊 nDCG@100: {metrics[nDCG@100]:.4f}")
        
        return metrics
    else:
        print("❌ Could not calculate metrics")
        return None

# STEP 5: Run the quick fix test
def run_quick_fix():
    """Run the quick fix test with better model"""
    print("🚀 Starting quick fix test...")
    
    # Load existing cleaned documents
    documents, document_order, cleaned_texts = load_cleaned_documents()
    
    if documents is None:
        print("❌ Cannot load documents - make sure you have the metadata file")
        return
    
    # Test with better model
    print(f"\n🤖 Testing with all-mpnet-base-v2 (better model)...")
    model, embeddings_matrix = generate_embeddings_with_better_model(cleaned_texts)
    
    # Quick evaluation
    metrics = quick_evaluation_better_model(model, embeddings_matrix, documents, document_order)
    
    if metrics:
        map_score = float(metrics[AP@100])
        
        print(f"\n" + "="*60)
        print(f"🎯 QUICK FIX RESULTS")
        print(f"="*60)
        print(f"📊 Original model (all-MiniLM-L6-v2): MAP ≈ 0.14")
        print(f"📊 Better model (all-mpnet-base-v2): MAP = {map_score:.4f}")
        print(f"📈 Improvement: {map_score - 0.14:.4f} ({((map_score/0.14-1)*100):+.1f}%)")
        
        if map_score > 0.18:
            print(f"\n✅ SIGNIFICANT IMPROVEMENT! The better model works much better.")
            print(f"🎯 Recommendation: Use all-mpnet-base-v2 for your final system")
            
            # Save the better embeddings
            print(f"\n💾 Saving improved embeddings...")
            joblib.dump(embeddings_matrix, 'antique_embeddings_matrix_mpnet.joblib')
            model.save('antique_embedding_model_mpnet')
            print(f"✅ Saved improved model and embeddings")
            
        elif map_score > 0.16:
            print(f"\n✅ MODERATE IMPROVEMENT. The better model helps.")
            print(f"🎯 Consider using all-mpnet-base-v2 or trying fine-tuning")
        else:
            print(f"\n⚠️  Only small improvement. The issue might be deeper.")
            print(f"🎯 Consider fine-tuning or hybrid approaches")
    
    print(f"\n🔧 Next steps if this doesn't work:")
    print(f"   1. Try fine-tuning the model on Antique training data")
    print(f"   2. Use hybrid retrieval (TF-IDF + embeddings)")
    print(f"   3. Check for document ID alignment issues")
    print(f"   4. Try different preprocessing approaches")

if __name__ == "__main__":
    run_quick_fix()
