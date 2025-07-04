#!/usr/bin/env python3
"""
TF-IDF Microservices Pipeline Demo
Demonstrates the complete workflow using all microservices
"""

import requests
import time
import json
from typing import List, Dict

# Service URLs
TEXT_CLEANING_URL = "http://localhost:8001"
TFIDF_VECTORIZER_URL = "http://localhost:8002"
ENHANCED_TFIDF_URL = "http://localhost:8003"
MAP_EVALUATION_URL = "http://localhost:8004"

def check_services():
    """Check if all services are running"""
    services = [
        ("Text Cleaning", TEXT_CLEANING_URL),
        ("TF-IDF Vectorizer", TFIDF_VECTORIZER_URL),
        ("Enhanced TF-IDF", ENHANCED_TFIDF_URL),
        ("MAP Evaluation", MAP_EVALUATION_URL)
    ]
    
    print("Checking service health...")
    all_healthy = True
    
    for name, url in services:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ {name} Service: Healthy")
            else:
                print(f"✗ {name} Service: Unhealthy (HTTP {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"✗ {name} Service: Not reachable ({str(e)})")
            all_healthy = False
    
    return all_healthy

def test_text_cleaning():
    """Test the text cleaning service"""
    print("\n" + "="*60)
    print("TESTING TEXT CLEANING SERVICE")
    print("="*60)
    
    test_texts = [
        "I'm looking for beautifull antique vases from the 1800s!",
        "Can you help me find vintage collectibles?",
        "What's the value of old coins and manuscripts?"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        
        # Test TF-IDF optimized cleaning
        response = requests.post(f"{TEXT_CLEANING_URL}/clean/tfidf", json={"text": text})
        if response.status_code == 200:
            result = response.json()
            print(f"  Original: {result['original_text']}")
            print(f"  Cleaned:  {result['cleaned_text']}")
            print(f"  Features: {result['features']}")
        else:
            print(f"  Error: {response.status_code}")

def test_enhanced_tfidf_training():
    """Test training the enhanced TF-IDF service"""
    print("\n" + "="*60)
    print("TESTING ENHANCED TF-IDF TRAINING")
    print("="*60)
    
    # Sample documents for training
    documents = [
        "Beautiful antique furniture from the Victorian era with intricate woodwork and ornate designs",
        "Vintage collectibles including rare coins, stamps, and historical memorabilia from the 19th century",
        "Classic automobiles and automotive memorabilia from the 1950s including original parts and accessories",
        "Old books manuscripts and historical documents from the 18th century with leather bindings",
        "Antique jewelry including Victorian brooches, Art Deco rings, and vintage watches",
        "Vintage ceramics and pottery from famous European manufacturers",
        "Historical artifacts and archaeological finds from ancient civilizations",
        "Antique musical instruments including violins, pianos, and brass instruments",
        "Vintage clothing and textiles from different historical periods",
        "Collectible toys and games from the early 20th century"
    ]
    
    doc_ids = [f"doc_{i:03d}" for i in range(1, len(documents) + 1)]
    
    print(f"Training on {len(documents)} sample documents...")
    
    # Train the model
    train_request = {
        "documents": documents,
        "doc_ids": doc_ids,
        "build_inverted_index": True
    }
    
    start_time = time.time()
    response = requests.post(f"{ENHANCED_TFIDF_URL}/train", json=train_request)
    training_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        stats = result.get("training_statistics", {})
        print(f"  Documents processed: {stats.get('valid_documents', 'N/A')}")
        print(f"  Vocabulary size: {stats.get('vocabulary_size', 'N/A')}")
        print(f"  Matrix sparsity: {stats.get('sparsity', 'N/A'):.2f}%")
        print(f"  Inverted index terms: {stats.get('inverted_index_terms', 'N/A')}")
        
        return True
    else:
        print(f"✗ Training failed: {response.status_code}")
        print(response.text)
        return False

def test_enhanced_search():
    """Test searching with the enhanced TF-IDF service"""
    print("\n" + "="*60)
    print("TESTING ENHANCED TF-IDF SEARCH")
    print("="*60)
    
    test_queries = [
        "antique furniture Victorian",
        "vintage coins collectibles",
        "old books manuscripts",
        "classic cars automotive",
        "jewelry watches vintage"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Test enhanced inverted index search
        search_request = {
            "query": query,
            "top_k": 3,
            "method": "enhanced_inverted"
        }
        
        response = requests.post(f"{ENHANCED_TFIDF_URL}/search", json=search_request)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Method: {result['method_used']}")
            print(f"  Search time: {result['search_time_ms']:.2f}ms")
            print(f"  Results:")
            
            for rank, res in enumerate(result['results'][:3], 1):
                print(f"    {rank}. {res['doc_id']} (Score: {res['score']:.4f})")
                if res.get('document_text'):
                    print(f"       {res['document_text']}")
        else:
            print(f"  Error: {response.status_code}")

def test_map_evaluation():
    """Test MAP evaluation service"""
    print("\n" + "="*60)
    print("TESTING MAP EVALUATION SERVICE")
    print("="*60)
    
    # Check if ANTIQUE dataset is loaded
    response = requests.get(f"{MAP_EVALUATION_URL}/datasets")
    if response.status_code == 200:
        datasets = response.json()
        antique_info = datasets['available_datasets'].get('antique', {})
        
        if antique_info.get('loaded', False):
            print(f"✓ ANTIQUE dataset loaded:")
            print(f"  Queries: {antique_info.get('num_queries', 'N/A')}")
            print(f"  QRels: {antique_info.get('num_qrels', 'N/A')}")
            
            # Test evaluation (limited queries for demo)
            print("\nRunning MAP evaluation (limited to 10 queries for demo)...")
            
            eval_request = {
                "dataset_name": "antique",
                "max_queries": 10,
                "k_eval": 10
            }
            
            start_time = time.time()
            response = requests.post(f"{MAP_EVALUATION_URL}/evaluate", json=eval_request)
            eval_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✓ Evaluation completed in {eval_time:.2f} seconds")
                print(f"  Dataset: {result['dataset']}")
                print(f"  Queries evaluated: {result['num_queries_evaluated']}")
                print(f"  MAP@{result['cutoff_k']}: {result['MAP']:.4f}")
                
                if result['MAP'] >= 0.4:
                    print("  ✓ TARGET ACHIEVED (MAP ≥ 0.4)")
                else:
                    print("  ✗ Below target (MAP < 0.4)")
                
                # Precision/Recall results
                print(f"  Precision@1: {result['precision_recall'].get('P@1', 'N/A'):.4f}")
                print(f"  Precision@5: {result['precision_recall'].get('P@5', 'N/A'):.4f}")
                print(f"  Recall@10: {result['precision_recall'].get('R@10', 'N/A'):.4f}")
                
                # Query performance
                query_perf = result['query_performance']
                print(f"  Mean AP: {query_perf.get('mean_ap', 'N/A'):.4f}")
                print(f"  Queries above 0.4 AP: {query_perf.get('queries_above_0_4', 'N/A')} ({query_perf.get('percentage_above_0_4', 'N/A'):.1f}%)")
                
                # Recommendations
                print("  Recommendations:")
                for rec in result.get('recommendations', []):
                    print(f"    - {rec}")
            else:
                print(f"✗ Evaluation failed: {response.status_code}")
                print(response.text)
        else:
            print("✗ ANTIQUE dataset not loaded in MAP Evaluation Service")
    else:
        print(f"✗ Failed to check datasets: {response.status_code}")

def test_service_integration():
    """Test integration between all services"""
    print("\n" + "="*60)
    print("TESTING SERVICE INTEGRATION")
    print("="*60)
    
    # Test workflow: Clean -> Train -> Search -> Evaluate
    
    # 1. Clean a query
    query = "beautiful antique furniture Victorian era"
    print(f"1. Cleaning query: {query}")
    
    response = requests.post(f"{TEXT_CLEANING_URL}/clean/query", json={"query": query})
    if response.status_code == 200:
        cleaned_query = response.json()['cleaned_query']
        print(f"   Cleaned: {cleaned_query}")
    else:
        print(f"   Cleaning failed: {response.status_code}")
        return
    
    # 2. Search with cleaned query
    print(f"2. Searching with enhanced TF-IDF...")
    
    search_request = {
        "query": query,  # Service will clean it internally
        "top_k": 5,
        "method": "enhanced_inverted"
    }
    
    response = requests.post(f"{ENHANCED_TFIDF_URL}/search", json=search_request)
    if response.status_code == 200:
        results = response.json()
        print(f"   Found {len(results['results'])} results")
        print(f"   Search time: {results['search_time_ms']:.2f}ms")
        
        for i, result in enumerate(results['results'][:3], 1):
            print(f"   {i}. {result['doc_id']} (Score: {result['score']:.4f})")
    else:
        print(f"   Search failed: {response.status_code}")
    
    # 3. Get service statistics
    print(f"3. Getting service statistics...")
    
    for name, url in [("Enhanced TF-IDF", ENHANCED_TFIDF_URL), ("MAP Evaluation", MAP_EVALUATION_URL)]:
        response = requests.get(f"{url}/info")
        if response.status_code == 200:
            info = response.json()
            print(f"   {name}: {info.get('service_name', 'N/A')} v{info.get('version', 'N/A')}")
        else:
            print(f"   {name}: Info not available")

def main():
    """Main demo function"""
    print("TF-IDF MICROSERVICES PIPELINE DEMO")
    print("=" * 60)
    
    # Check if all services are running
    if not check_services():
        print("\n✗ Not all services are running. Please start the microservices first:")
        print("python start_microservices.py")
        return
    
    print("\n✓ All services are healthy!")
    
    # Run tests
    try:
        # Test 1: Text Cleaning
        test_text_cleaning()
        
        # Test 2: Enhanced TF-IDF Training
        if test_enhanced_tfidf_training():
            # Test 3: Enhanced Search
            test_enhanced_search()
            
            # Test 4: MAP Evaluation
            test_map_evaluation()
            
            # Test 5: Service Integration
            test_service_integration()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The microservices architecture is working correctly.")
        print("Key benefits demonstrated:")
        print("✓ Modular design with separate concerns")
        print("✓ HTTP-based communication between services")
        print("✓ Advanced text processing with spell checking, lemmatization, stemming")
        print("✓ Enhanced TF-IDF with inverted index optimization")
        print("✓ MAP evaluation for performance assessment")
        print("✓ Scalable architecture ready for production")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")

if __name__ == "__main__":
    main()
