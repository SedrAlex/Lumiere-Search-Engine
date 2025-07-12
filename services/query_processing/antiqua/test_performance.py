#!/usr/bin/env python3
"""
Performance test script for the improved ANTIQUE query processing service.
Tests search speed with cached embeddings vs without caching.
"""

import requests
import time
import json

def test_query_performance():
    """Test the performance improvements of the query processing service."""
    base_url = "http://localhost:5002"
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How to improve website performance?",
        "Best practices for software development",
        "Python programming tutorials",
        "dd"
    ]
    
    print("🚀 Testing ANTIQUE Query Processing Service Performance...")
    print(f"📍 Service URL: {base_url}")
    
    try:
        # Check service health
        print("\n1. Checking service health...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Check cache status
        print("\n2. Checking cache status...")
        response = requests.get(f"{base_url}/cache/status", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            cache_info = response.json()
            print(f"   Cache exists: {cache_info['cache_exists']}")
            print(f"   Cache valid: {cache_info['cache_valid']}")
            print(f"   Cached documents: {cache_info['cached_documents']}")
        
        # Get service stats
        print("\n3. Getting service statistics...")
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   Model loaded: {stats['model_loaded']}")
            print(f"   Database loaded: {stats['database_loaded']}")
            print(f"   Documents: {stats['num_documents']}")
            print(f"   Search method: {stats['search_method']}")
        
        # Performance test: Run multiple queries and measure time
        print("\n4. Performance testing with multiple queries...")
        total_time = 0
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            start_time = time.time()
            try:
                search_payload = {"query": query, "top_k": 5}
                response = requests.post(f"{base_url}/search", json=search_payload, timeout=30)
                end_time = time.time()
                
                query_time = end_time - start_time
                total_time += query_time
                
                if response.status_code == 200:
                    result = response.json()
                    successful_queries += 1
                    
                    print(f"   ✅ Success in {query_time:.3f}s")
                    print(f"   📄 Found {result['total_results']} results")
                    print(f"   🔄 Processed query: '{result['processed_query']}'")
                    
                    # Show top result
                    if result['results']:
                        top_result = result['results'][0]
                        print(f"   🥇 Top result: doc_id={top_result['doc_id']}, score={top_result['similarity_score']:.3f}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"   ⏰ Timeout after 30 seconds")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   Successful queries: {successful_queries}/{len(test_queries)}")
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"   Average query time: {avg_time:.3f}s")
            print(f"   Total time: {total_time:.3f}s")
            
            if avg_time < 1.0:
                print("   🚀 EXCELLENT: Sub-second query performance!")
            elif avg_time < 3.0:
                print("   ✅ GOOD: Fast query performance")
            elif avg_time < 10.0:
                print("   ⚠️  ACCEPTABLE: Moderate performance")
            else:
                print("   ❌ SLOW: Performance needs improvement")
        
        # Test cache refresh if needed
        print("\n5. Testing cache operations...")
        cache_response = requests.get(f"{base_url}/cache/status", timeout=5)
        if cache_response.status_code == 200:
            cache_info = cache_response.json()
            if not cache_info['cache_exists'] or not cache_info['cache_valid']:
                print("   🔄 Cache invalid/missing. Testing cache refresh...")
                refresh_start = time.time()
                refresh_response = requests.post(f"{base_url}/cache/refresh", timeout=300)  # 5 min timeout
                refresh_time = time.time() - refresh_start
                
                if refresh_response.status_code == 200:
                    refresh_result = refresh_response.json()
                    print(f"   ✅ Cache refreshed in {refresh_time:.1f}s")
                    print(f"   📄 Processed {refresh_result['documents_processed']} documents")
                else:
                    print(f"   ❌ Cache refresh failed: {refresh_response.text}")
            else:
                print("   ✅ Cache is valid and ready")
        
        print("\n🎉 Performance test completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Could not connect to service at {base_url}")
        print("   Make sure the query processing service is running on port 5002")
        print("   Run: python embedding_antique_query_processing.py")
    except Exception as e:
        print(f"\n❌ Error during performance test: {e}")

if __name__ == "__main__":
    test_query_performance()
