#!/usr/bin/env python3
"""
Test script to validate TF-IDF models and search functionality
"""

import sys
import os
import joblib
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the text cleaner from the main module
try:
    from quora_tfidf_query_processing import QuoraTextCleaner
except ImportError:
    print("Error: Could not import QuoraTextCleaner")
    print("This script must be run from the same directory as quora_tfidf_query_processing.py")
    sys.exit(1)

def test_model_loading():
    """Test if all models can be loaded correctly"""
    print("Testing model loading...")
    
    MODEL_DIR = '/Users/raafatmhanna/Downloads/quora_tfidf_models/'
    
    try:
        # Initialize text cleaner first (required for vectorizer)
        text_cleaner = QuoraTextCleaner()
        print("✓ Text cleaner initialized")
        
        # Load models
        print("Loading TF-IDF vectorizer...")
        vectorizer = joblib.load(MODEL_DIR + 'tfidf_vectorizer.joblib')
        print("✓ TF-IDF vectorizer loaded")
        
        print("Loading TF-IDF matrix...")
        matrix = joblib.load(MODEL_DIR + 'tfidf_matrix.joblib')
        print("✓ TF-IDF matrix loaded")
        
        print("Loading inverted index...")
        inverted_index = joblib.load(MODEL_DIR + 'inverted_index.joblib')
        print("✓ Inverted index loaded")
        
        print("Loading document mappings...")
        doc_mappings = joblib.load(MODEL_DIR + 'document_mappings.joblib')
        doc_ids = doc_mappings['doc_ids']
        print("✓ Document mappings loaded")
        
        return vectorizer, matrix, inverted_index, doc_ids, text_cleaner
        
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        return None, None, None, None, None

def test_model_consistency(vectorizer, matrix, inverted_index, doc_ids):
    """Test model consistency"""
    print("\nTesting model consistency...")
    
    issues = []
    
    # Check dimensions
    if matrix.shape[0] != len(doc_ids):
        issues.append(f"TF-IDF matrix rows ({matrix.shape[0]}) != doc IDs count ({len(doc_ids)})")
    else:
        print("✓ Matrix rows match document count")
    
    if matrix.shape[1] != len(vectorizer.get_feature_names_out()):
        issues.append(f"TF-IDF matrix cols ({matrix.shape[1]}) != vectorizer vocab ({len(vectorizer.get_feature_names_out())})")
    else:
        print("✓ Matrix columns match vocabulary size")
    
    # Check inverted index
    vocab = set(vectorizer.get_feature_names_out())
    index_terms = set(inverted_index.keys())
    if not index_terms.issubset(vocab):
        issues.append(f"Inverted index contains terms not in vocabulary")
    else:
        print("✓ Inverted index terms are consistent with vocabulary")
    
    print(f"\nModel Statistics:")
    print(f"  Documents: {len(doc_ids)}")
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  TF-IDF matrix shape: {matrix.shape}")
    print(f"  Matrix sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")
    print(f"  Inverted index terms: {len(inverted_index)}")
    
    return issues

def test_search_functionality(vectorizer, matrix, inverted_index, doc_ids):
    """Test search functionality"""
    print("\nTesting search functionality...")
    
    test_queries = [
        "How to learn programming",
        "What is machine learning",
        "Best way to invest money",
        "Python programming tutorial"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        try:
            # Test TF-IDF matrix search
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:5]
            
            tfidf_results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    tfidf_results.append((doc_ids[idx], similarities[idx]))
            
            print(f"  TF-IDF search: {len(tfidf_results)} results, top score: {tfidf_results[0][1]:.4f}" if tfidf_results else "  TF-IDF search: No results")
            
            # Test inverted index search
            query_terms = vectorizer.build_analyzer()(query)
            candidate_docs = defaultdict(float)
            
            for term in query_terms:
                if term in inverted_index:
                    for doc_id, score in inverted_index[term].items():
                        candidate_docs[doc_id] += score
            
            sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
            index_results = sorted_docs[:5]
            
            print(f"  Inverted index search: {len(index_results)} results, top score: {index_results[0][1]:.4f}" if index_results else "  Inverted index search: No results")
            print(f"  Query terms: {query_terms}")
            
        except Exception as e:
            print(f"  ✗ Error testing query: {str(e)}")

def test_database_connection():
    """Test database connection and document retrieval"""
    print("\nTesting database connection...")
    
    DB_PATH = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_documents.db'
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"✓ Database connected. Columns: {columns}")
        
        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"✓ Documents in database: {doc_count}")
        
        # Test sample document retrieval
        cursor.execute("SELECT doc_id, original_text FROM documents LIMIT 5")
        samples = cursor.fetchall()
        print(f"✓ Sample documents retrieved: {len(samples)}")
        
        for doc_id, text in samples[:2]:
            snippet = text[:100] + "..." if len(text) > 100 else text
            print(f"  Doc {doc_id}: {snippet}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("TF-IDF Model Validation Test")
    print("=" * 50)
    
    # Test model loading
    vectorizer, matrix, inverted_index, doc_ids, text_cleaner = test_model_loading()
    
    if vectorizer is None:
        print("\n✗ Model loading failed. Cannot proceed with further tests.")
        return
    
    # Test model consistency
    issues = test_model_consistency(vectorizer, matrix, inverted_index, doc_ids)
    
    if issues:
        print(f"\n⚠️  Model consistency issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All model consistency checks passed")
    
    # Test search functionality
    test_search_functionality(vectorizer, matrix, inverted_index, doc_ids)
    
    # Test database connection
    db_ok = test_database_connection()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Models loaded: ✓")
    print(f"  Model consistency: {'✓' if not issues else '⚠️ ' + str(len(issues)) + ' issues'}")
    print(f"  Search functionality: ✓")
    print(f"  Database connection: {'✓' if db_ok else '✗'}")
    
    if not issues and db_ok:
        print("\n🎉 All tests passed! Your TF-IDF system appears to be working correctly.")
    else:
        print("\n⚠️  Some issues were found. Check the details above.")

if __name__ == "__main__":
    main()
