#!/usr/bin/env python3
"""
Test script for the Quora TF-IDF Evaluation Service
"""

import requests
import json
import pandas as pd
import time

# Service configuration
SERVICE_URL = "http://localhost:8000"
QUERIES_PATH = "/Users/raafatmhanna/Downloads/quora/queries.tsv"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to service: {e}")
        return False

def test_service_info():
    """Test the service info endpoint"""
    print("\n🔍 Testing service info endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/info")
        if response.status_code == 200:
            print("✅ Service info retrieved")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Service info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not get service info: {e}")
        return False

def test_query_evaluation():
    """Test query evaluation with a specific query ID"""
    print("\n🔍 Testing query evaluation...")
    
    # Load queries to get a sample query ID
    try:
        queries_df = pd.read_csv(QUERIES_PATH, sep='\t', header=None, names=['query_id', 'text'])
        sample_query_id = queries_df['query_id'].iloc[0]  # Get first query ID
        sample_query_text = queries_df['text'].iloc[0]
        
        print(f"📝 Testing with Query ID: {sample_query_id}")
        print(f"📝 Query Text: {sample_query_text}")
        
        # Test query evaluation
        payload = {"query_id": str(sample_query_id)}
        response = requests.post(f"{SERVICE_URL}/evaluate/query", json=payload)
        
        if response.status_code == 200:
            print("✅ Query evaluation successful")
            result = response.json()
            print(f"📊 Found {len(result['results'])} results")
            
            # Show top 3 results
            for i, res in enumerate(result['results'][:3]):
                print(f"  {i+1}. Doc ID: {res['doc_id']}, Score: {res['score']:.4f}, Relevant: {res['relevant']}")
            return True
        else:
            print(f"❌ Query evaluation failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Query evaluation error: {e}")
        return False

def test_text_evaluation():
    """Test text evaluation with arbitrary text"""
    print("\n🔍 Testing text evaluation...")
    
    test_query = "What is machine learning?"
    payload = {"query_text": test_query, "k": 5}
    
    try:
        response = requests.post(f"{SERVICE_URL}/evaluate/text", json=payload)
        
        if response.status_code == 200:
            print("✅ Text evaluation successful")
            result = response.json()
            print(f"📝 Query: {result['query_text']}")
            print(f"🔧 Processed: {result['processed_query']}")
            print(f"📊 Found {len(result['results'])} results")
            
            # Show top 3 results
            for i, res in enumerate(result['results'][:3]):
                print(f"  {i+1}. Doc ID: {res['doc_id']}, Score: {res['score']:.4f}")
            return True
        else:
            print(f"❌ Text evaluation failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Text evaluation error: {e}")
        return False

def test_map_calculation():
    """Test MAP calculation"""
    print("\n🔍 Testing MAP calculation...")
    
    payload = {"k": 1000}  # Use full k for comprehensive MAP calculation
    
    try:
        print("⏳ Calculating MAP (this may take a while)...")
        response = requests.post(f"{SERVICE_URL}/evaluate/map", json=payload)  # Removing timeout to allow full processing
        
        if response.status_code == 200:
            print("✅ MAP calculation successful")
            result = response.json()
            print(f"📊 MAP@{result['k']}: {result['map_score']:.4f}")
            print(f"📈 Queries evaluated: {result['queries_evaluated']}/{result['total_queries']}")
            return True
        else:
            print(f"❌ MAP calculation failed: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ MAP calculation timed out (this is normal for large datasets)")
        return False
    except Exception as e:
        print(f"❌ MAP calculation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Quora TF-IDF Evaluation Service Tests")
    print("=" * 60)
    
    # Wait a moment for service to be ready
    print("⏳ Waiting for service to be ready...")
    time.sleep(2)
    
    tests = [
        test_health,
        test_service_info,
        test_query_evaluation,
        test_text_evaluation,
        test_map_calculation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the service and try again.")

if __name__ == "__main__":
    main()
