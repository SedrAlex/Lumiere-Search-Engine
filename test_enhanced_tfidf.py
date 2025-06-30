#!/usr/bin/env python3
"""
Enhanced TF-IDF Testing Script
Tests the new query processing service with enhanced cleaning and cosine similarity
"""

import asyncio
import json
import time
import requests
from typing import List, Dict, Any

# Service URLs
ENHANCED_CLEANING_URL = "http://localhost:8003"
TFIDF_QUERY_PROCESSOR_URL = "http://localhost:8004"

class TFIDFTester:
    def __init__(self):
        self.base_cleaning_url = ENHANCED_CLEANING_URL
        self.query_processor_url = TFIDF_QUERY_PROCESSOR_URL
    
    def check_services(self) -> Dict[str, bool]:
        """Check if all services are running"""
        print("🔍 Checking service availability...")
        
        services = {
            "Enhanced Cleaning Service": self.base_cleaning_url,
            "TF-IDF Query Processor": self.query_processor_url
        }
        
        status = {}
        for name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                status[name] = response.status_code == 200
                print(f"{'✅' if status[name] else '❌'} {name}: {url}")
            except Exception as e:
                status[name] = False
                print(f"❌ {name}: {url} - {e}")
        
        return status
    
    def test_cleaning_pipeline(self):
        """Test the enhanced cleaning pipeline"""
        print("\n🧹 Testing Enhanced Cleaning Pipeline")
        print("=" * 50)
        
        test_texts = [
            "Information retrieval systems using machine learning",
            "Document ranking and search algorithms", 
            "Natural language processing and text mining",
            "Database systems and query optimization"
        ]
        
        for text in test_texts:
            try:
                response = requests.post(
                    f"{self.base_cleaning_url}/clean",
                    json={
                        "text": text,
                        "use_lemmatization": True,
                        "use_stemming": True,
                        "use_spellcheck": False,
                        "remove_stopwords": True,
                        "min_token_length": 2
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Original: {text}")
                    print(f"Cleaned:  {result['cleaned_text']}")
                    print(f"Tokens:   {result['tokens']}")
                    print(f"Stats:    {result['processing_stats']['steps_applied']}")
                    print("-" * 50)
                else:
                    print(f"❌ Error cleaning text: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error in cleaning test: {e}")
    
    def test_query_processing(self):
        """Test query processing with different scenarios"""
        print("\n🔍 Testing Query Processing with Cosine Similarity")
        print("=" * 50)
        
        test_queries = [
            {
                "query": "information retrieval systems",
                "description": "Standard IR query"
            },
            {
                "query": "machine learning algorithms",
                "description": "ML-focused query"
            },
            {
                "query": "database management systems",
                "description": "Database query"
            },
            {
                "query": "natural language processing",
                "description": "NLP query"
            },
            {
                "query": "web search engines",
                "description": "Web search query"
            }
        ]
        
        for test_case in test_queries:
            print(f"\n📝 Testing: {test_case['description']}")
            print(f"Query: '{test_case['query']}'")
            
            try:
                response = requests.post(
                    f"{self.query_processor_url}/search",
                    json={
                        "query": test_case["query"],
                        "top_k": 5,
                        "similarity_threshold": 0.0,
                        "use_enhanced_cleaning": True
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"Original Query: {result['query']}")
                    print(f"Cleaned Query:  {result['cleaned_query']}")
                    print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
                    print(f"Total Results: {result['total_results']}")
                    
                    # Show similarity statistics
                    stats = result['similarity_stats']
                    print(f"Similarity Stats - Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}")
                    
                    # Show top results
                    print("\n📊 Top Results:")
                    for i, doc in enumerate(result['results'][:3], 1):
                        text_preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                        print(f"  {i}. Score: {doc['score']:.4f} | Doc: {doc['doc_id']}")
                        print(f"     Text: {text_preview}")
                    
                else:
                    print(f"❌ Error processing query: {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ Error in query test: {e}")
            
            print("-" * 50)
    
    def test_performance_comparison(self):
        """Test performance with different cleaning options"""
        print("\n⚡ Testing Performance with Different Cleaning Options")
        print("=" * 50)
        
        test_query = "information retrieval machine learning"
        
        # Test with enhanced cleaning
        print("🔬 Testing with Enhanced Cleaning (Lemmatization + Stemming):")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.query_processor_url}/search",
                json={
                    "query": test_query,
                    "top_k": 10,
                    "use_enhanced_cleaning": True
                },
                timeout=30
            )
            
            enhanced_time = time.time() - start_time
            if response.status_code == 200:
                enhanced_result = response.json()
                print(f"  ✅ Processing Time: {enhanced_result['processing_time_ms']:.2f}ms")
                print(f"  ✅ Total Time: {enhanced_time*1000:.2f}ms")
                print(f"  ✅ Results: {enhanced_result['total_results']}")
                print(f"  ✅ Cleaned Query: '{enhanced_result['cleaned_query']}'")
                
                if enhanced_result['results']:
                    avg_score = sum(r['score'] for r in enhanced_result['results']) / len(enhanced_result['results'])
                    print(f"  ✅ Average Score: {avg_score:.4f}")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Test with basic cleaning
        print("\n🔬 Testing with Basic Cleaning (Fallback):")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.query_processor_url}/search",
                json={
                    "query": test_query,
                    "top_k": 10,
                    "use_enhanced_cleaning": False
                },
                timeout=30
            )
            
            basic_time = time.time() - start_time
            if response.status_code == 200:
                basic_result = response.json()
                print(f"  ✅ Processing Time: {basic_result['processing_time_ms']:.2f}ms")
                print(f"  ✅ Total Time: {basic_time*1000:.2f}ms") 
                print(f"  ✅ Results: {basic_result['total_results']}")
                print(f"  ✅ Cleaned Query: '{basic_result['cleaned_query']}'")
                
                if basic_result['results']:
                    avg_score = sum(r['score'] for r in basic_result['results']) / len(basic_result['results'])
                    print(f"  ✅ Average Score: {avg_score:.4f}")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    def test_batch_processing(self):
        """Test batch query processing"""
        print("\n📦 Testing Batch Query Processing")
        print("=" * 50)
        
        batch_queries = [
            "information retrieval",
            "machine learning", 
            "database systems",
            "web search"
        ]
        
        try:
            response = requests.post(
                f"{self.query_processor_url}/search/batch",
                json=batch_queries,
                params={"top_k": 3},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Processed {result['query_count']} queries")
                print(f"✅ Total Processing Time: {result['total_processing_time_ms']:.2f}ms")
                print(f"✅ Average Time per Query: {result['total_processing_time_ms']/result['query_count']:.2f}ms")
                
                # Show summary of results
                for i, query_result in enumerate(result['results']):
                    print(f"  Query {i+1}: '{query_result['query']}' -> {query_result['total_results']} results")
                    
            else:
                print(f"❌ Error in batch processing: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error in batch test: {e}")
    
    def get_service_status(self):
        """Get detailed service status"""
        print("\n📊 Service Status and Configuration")
        print("=" * 50)
        
        # Enhanced cleaning service stats
        try:
            response = requests.get(f"{self.base_cleaning_url}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                print("🧹 Enhanced Cleaning Service:")
                print(f"  ✅ NLTK Available: {stats['nltk_available']}")
                print(f"  ✅ Spellcheck Available: {stats['spellcheck_available']}")
                print(f"  ✅ Stopwords Loaded: {stats['stopwords_loaded']}")
        except Exception as e:
            print(f"❌ Error getting cleaning stats: {e}")
        
        # Query processor status
        try:
            response = requests.get(f"{self.query_processor_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print("\n🔍 TF-IDF Query Processor:")
                print(f"  ✅ Model Loaded: {status['model_loaded']}")
                print(f"  ✅ Documents Count: {status['documents_count']:,}")
                print(f"  ✅ Vocabulary Size: {status['vocabulary_size']:,}")
                print(f"  ✅ Cleaning Service: {status['cleaning_service_status']}")
                
                if status['model_info']:
                    info = status['model_info']
                    print(f"  ✅ Matrix Shape: {info['matrix_shape']}")
                    print(f"  ✅ N-gram Range: {info['ngram_range']}")
                    print(f"  ✅ Max Features: {info['max_features']}")
                    
        except Exception as e:
            print(f"❌ Error getting processor status: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🎯 Enhanced TF-IDF Query Processing Tests")
        print("=" * 60)
        
        # Check services first
        service_status = self.check_services()
        
        if not all(service_status.values()):
            print("\n❌ Not all services are running. Please start services first:")
            print("   python start_tfidf_services.py")
            return False
        
        # Run tests
        self.get_service_status()
        self.test_cleaning_pipeline()
        self.test_query_processing()
        self.test_performance_comparison()
        self.test_batch_processing()
        
        print("\n🎉 All tests completed!")
        print("=" * 60)
        
        return True

def main():
    """Main testing function"""
    tester = TFIDFTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
