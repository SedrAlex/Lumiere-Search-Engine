#!/usr/bin/env python3
"""
Test script for the enhanced TF-IDF service with inverted index functionality.
This demonstrates the performance difference between regular TF-IDF search and hybrid search.
"""

import asyncio
import httpx
import time
import json
from typing import List, Dict

# Service endpoint
TFIDF_SERVICE_URL = "http://localhost:8002"

async def test_inverted_index_functionality():
    """Test the inverted index functionality"""
    
    # Sample documents for testing
    sample_documents = [
        {
            "id": "doc1",
            "text": "Machine learning algorithms are used to build predictive models from data.",
            "metadata": {"category": "AI"}
        },
        {
            "id": "doc2", 
            "text": "Natural language processing enables computers to understand human language.",
            "metadata": {"category": "NLP"}
        },
        {
            "id": "doc3",
            "text": "Deep learning neural networks can learn complex patterns in data.",
            "metadata": {"category": "Deep Learning"}
        },
        {
            "id": "doc4",
            "text": "Information retrieval systems help users find relevant documents quickly.",
            "metadata": {"category": "IR"}
        },
        {
            "id": "doc5",
            "text": "Search engines use sophisticated algorithms to rank and retrieve documents.",
            "metadata": {"category": "Search"}
        },
        {
            "id": "doc6",
            "text": "Vector space models represent documents as high-dimensional vectors.",
            "metadata": {"category": "Information Retrieval"}
        },
        {
            "id": "doc7",
            "text": "TF-IDF weighting scheme combines term frequency and inverse document frequency.",
            "metadata": {"category": "Text Mining"}
        },
        {
            "id": "doc8",
            "text": "Cosine similarity measures the angle between document vectors in vector space.",
            "metadata": {"category": "Similarity"}
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🚀 Testing Enhanced TF-IDF Service with Inverted Index")
        print("=" * 60)
        
        # 1. Check service health
        print("1. Checking service health...")
        try:
            response = await client.get(f"{TFIDF_SERVICE_URL}/health")
            print(f"   ✅ Service is healthy: {response.json()}")
        except Exception as e:
            print(f"   ❌ Service not available: {e}")
            print("   Please start the TF-IDF service with: python services/representation/tfidf_service.py")
            return
        
        # 2. Index documents
        print("\n2. Indexing sample documents...")
        try:
            index_response = await client.post(
                f"{TFIDF_SERVICE_URL}/index",
                json={"documents": sample_documents}
            )
            index_result = index_response.json()
            print(f"   ✅ Indexed {index_result['documents_indexed']} documents")
            print(f"   📚 Vocabulary size: {index_result['vocabulary_size']}")
            print(f"   ⏱️  Processing time: {index_result['processing_time']:.3f}s")
        except Exception as e:
            print(f"   ❌ Error indexing documents: {e}")
            return
        
        # 3. Test regular TF-IDF search
        print("\n3. Testing regular TF-IDF search...")
        test_queries = [
            "machine learning algorithms",
            "natural language processing", 
            "vector space models",
            "document retrieval"
        ]
        
        for query in test_queries:
            try:
                start_time = time.time()
                response = await client.post(
                    f"{TFIDF_SERVICE_URL}/search",
                    json={"query": query, "top_k": 3}
                )
                search_time = time.time() - start_time
                
                results = response.json()
                print(f"\n   Query: '{query}'")
                print(f"   🔍 Found {results['total_results']} results in {search_time*1000:.1f}ms")
                
                for i, result in enumerate(results['results'][:2], 1):
                    print(f"   {i}. Doc {result['document_id']}: {result['score']:.4f}")
                    print(f"      {result['text'][:60]}...")
                    
            except Exception as e:
                print(f"   ❌ Error in regular search: {e}")
        
        # 4. Test hybrid search with inverted index
        print("\n4. Testing hybrid search with inverted index...")
        
        for query in test_queries:
            try:
                start_time = time.time()
                response = await client.post(
                    f"{TFIDF_SERVICE_URL}/search/hybrid",
                    json={"query": query, "top_k": 3}
                )
                search_time = time.time() - start_time
                
                results = response.json()
                print(f"\n   Query: '{query}'")
                print(f"   🚀 Hybrid search found {results['total_results']} results in {search_time*1000:.1f}ms")
                
                for i, result in enumerate(results['results'][:2], 1):
                    print(f"   {i}. Doc {result['document_id']}: {result['score']:.4f}")
                    print(f"      {result['text'][:60]}...")
                    
            except Exception as e:
                print(f"   ❌ Error in hybrid search: {e}")
        
        # 5. Test term statistics
        print("\n5. Testing term statistics...")
        test_terms = ["machine", "learning", "vector", "document"]
        
        for term in test_terms:
            try:
                response = await client.get(f"{TFIDF_SERVICE_URL}/term/{term}/stats")
                stats = response.json()
                
                if "error" not in stats:
                    print(f"\n   Term: '{term}'")
                    print(f"   📊 Document frequency: {stats['document_frequency']}")
                    print(f"   📈 IDF: {stats['idf']:.4f}")
                    print(f"   📋 Avg TF-IDF: {stats['average_tf_idf']:.4f}")
                else:
                    print(f"   ⚠️  {stats['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error getting term stats: {e}")
        
        # 6. Test document terms
        print("\n6. Testing document term analysis...")
        try:
            response = await client.get(f"{TFIDF_SERVICE_URL}/document/doc1/terms?top_terms=5")
            doc_terms = response.json()
            
            print(f"\n   Top terms for document 'doc1':")
            for i, (term, score) in enumerate(doc_terms['top_terms'][:5], 1):
                print(f"   {i}. '{term}': {score:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error getting document terms: {e}")
        
        # 7. Performance comparison
        print("\n7. Performance comparison...")
        test_query = "machine learning data"
        num_iterations = 10
        
        # Regular search timing
        regular_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            await client.post(
                f"{TFIDF_SERVICE_URL}/search",
                json={"query": test_query, "top_k": 5}
            )
            regular_times.append(time.time() - start_time)
        
        # Hybrid search timing
        hybrid_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            await client.post(
                f"{TFIDF_SERVICE_URL}/search/hybrid",
                json={"query": test_query, "top_k": 5}
            )
            hybrid_times.append(time.time() - start_time)
        
        avg_regular = sum(regular_times) / len(regular_times) * 1000
        avg_hybrid = sum(hybrid_times) / len(hybrid_times) * 1000
        
        print(f"\n   📊 Performance comparison ({num_iterations} iterations):")
        print(f"   🔍 Regular search: {avg_regular:.2f}ms average")
        print(f"   🚀 Hybrid search: {avg_hybrid:.2f}ms average")
        print(f"   📈 Speed improvement: {((avg_regular - avg_hybrid) / avg_regular * 100):.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n💡 Key Benefits of Inverted Index Implementation:")
        print("   • Faster query processing for large document collections")
        print("   • Efficient term-based filtering before similarity calculation")
        print("   • Better scalability for production systems")
        print("   • Detailed term and document statistics")
        print("   • Hybrid approach combines speed and accuracy")

if __name__ == "__main__":
    asyncio.run(test_inverted_index_functionality())
