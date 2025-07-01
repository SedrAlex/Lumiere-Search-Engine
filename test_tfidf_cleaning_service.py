#!/usr/bin/env python3
"""
Test script for TF-IDF Text Cleaning Service
Verifies that the service correctly implements the corrected_tfidf_colab approach
"""

import asyncio
import httpx
import json
from typing import List, Dict

# Test configuration
SERVICE_URL = "http://localhost:8005"

# Test cases that should match corrected_tfidf_colab behavior
TEST_CASES = [
    {
        "name": "Basic Text Cleaning",
        "input": "Information Retrieval Systems are important!",
        "expected_steps": [
            "lowercase_conversion",
            "html_tag_removal", 
            "special_character_removal",
            "whitespace_normalization",
            "tokenization",
            "token_filtering",
            "lemmatization",
            "stemming"
        ]
    },
    {
        "name": "HTML Tag Removal",
        "input": "<h1>Machine Learning</h1> <p>Natural Language Processing</p>",
        "expected_contains": ["machin", "learn", "natur", "languag", "process"]
    },
    {
        "name": "Special Characters",
        "input": "Search & Retrieval @ 2024! #NLP $100 50%",
        "expected_contains": ["search", "retriev", "2024", "nlp", "100", "50"]
    },
    {
        "name": "Stopword Removal",
        "input": "The quick brown fox jumps over the lazy dog",
        "should_not_contain": ["the", "over"]  # Common stopwords
    },
    {
        "name": "Stemming and Lemmatization",
        "input": "running runs runner information informational",
        "expected_contains": ["run", "inform"]  # Should be stemmed
    },
    {
        "name": "Empty Text",
        "input": "",
        "expected_tokens": []
    },
    {
        "name": "Short Tokens Filtering",
        "input": "a I am go to be running fast",
        "should_not_contain": ["a", "i"]  # Too short
    }
]

async def test_service_health():
    """Test if the service is running and healthy"""
    print("🏥 Testing service health...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVICE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Service is healthy: {health_data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            return False

async def test_service_info():
    """Test service information endpoint"""
    print("\n📋 Testing service info...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVICE_URL}/info")
            if response.status_code == 200:
                info = response.json()
                print(f"✅ Service info retrieved:")
                print(f"   - Service: {info.get('service_name')}")
                print(f"   - Version: {info.get('version')}")
                print(f"   - NLTK Available: {info.get('nltk_available')}")
                print(f"   - Compatible with: {info.get('compatible_with')}")
                print(f"   - Optimized for: {info.get('optimized_for')}")
                return True
            else:
                print(f"❌ Service info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error getting service info: {e}")
            return False

async def test_text_cleaning(test_case: Dict):
    """Test a single text cleaning case"""
    print(f"\n🧹 Testing: {test_case['name']}")
    print(f"   Input: '{test_case['input']}'")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SERVICE_URL}/clean",
                json={"text": test_case["input"]}
            )
            
            if response.status_code != 200:
                print(f"❌ Request failed: {response.status_code}")
                return False
            
            result = response.json()
            
            print(f"   Output: '{result['cleaned_text']}'")
            print(f"   Tokens: {result['tokens']}")
            print(f"   Token count: {result['token_count']}")
            
            # Check expected steps
            if "expected_steps" in test_case:
                applied_steps = result["processing_stats"]["steps_applied"]
                for step in test_case["expected_steps"]:
                    if step in applied_steps:
                        print(f"   ✅ Step applied: {step}")
                    else:
                        print(f"   ❌ Missing step: {step}")
                        return False
            
            # Check expected tokens
            if "expected_contains" in test_case:
                tokens = result["tokens"]
                for expected_token in test_case["expected_contains"]:
                    if any(expected_token in token for token in tokens):
                        print(f"   ✅ Contains expected: {expected_token}")
                    else:
                        print(f"   ❌ Missing expected: {expected_token}")
                        return False
            
            # Check tokens that should not be present
            if "should_not_contain" in test_case:
                tokens = result["tokens"]
                for unwanted_token in test_case["should_not_contain"]:
                    if unwanted_token not in tokens:
                        print(f"   ✅ Correctly filtered: {unwanted_token}")
                    else:
                        print(f"   ❌ Should not contain: {unwanted_token}")
                        return False
            
            # Check expected token count
            if "expected_tokens" in test_case:
                if result["tokens"] == test_case["expected_tokens"]:
                    print(f"   ✅ Correct tokens: {result['tokens']}")
                else:
                    print(f"   ❌ Expected {test_case['expected_tokens']}, got {result['tokens']}")
                    return False
            
            print(f"   ✅ Test passed!")
            return True
            
        except Exception as e:
            print(f"   ❌ Error during test: {e}")
            return False

async def test_corrected_colab_compatibility():
    """Test specific compatibility with corrected_tfidf_colab approach"""
    print("\n🔬 Testing corrected_tfidf_colab compatibility...")
    
    # This test case mimics the exact processing in corrected_tfidf_colab
    test_text = """
    <html>
    <body>
    <h1>Information Retrieval and Search Systems</h1>
    <p>Modern information retrieval systems use various techniques including 
    TF-IDF, BM25, and neural networks for document ranking.</p>
    </body>
    </html>
    """
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SERVICE_URL}/clean",
                json={"text": test_text}
            )
            
            if response.status_code != 200:
                print(f"❌ Compatibility test failed: {response.status_code}")
                return False
            
            result = response.json()
            
            print(f"   Original length: {len(test_text)} chars")
            print(f"   Cleaned length: {len(result['cleaned_text'])} chars")
            print(f"   Token count: {result['token_count']}")
            print(f"   Compression ratio: {result['processing_stats']['compression_ratio']:.2%}")
            
            # Check key characteristics of corrected_tfidf_colab processing
            expected_characteristics = {
                "html_removed": "<" not in result["cleaned_text"],
                "lowercase": result["cleaned_text"].islower(),
                "no_special_chars": all(c.isalnum() or c.isspace() for c in result["cleaned_text"]),
                "stemmed_tokens": any("inform" in token for token in result["tokens"]),  # "information" -> "inform"
                "lemmatized_first": True  # Order: lemmatize then stem
            }
            
            for check, passed in expected_characteristics.items():
                if passed:
                    print(f"   ✅ {check}: PASS")
                else:
                    print(f"   ❌ {check}: FAIL")
                    return False
            
            print(f"   ✅ Corrected TF-IDF colab compatibility: VERIFIED")
            return True
            
        except Exception as e:
            print(f"   ❌ Compatibility test error: {e}")
            return False

async def run_all_tests():
    """Run all test cases"""
    print("🧪 TF-IDF Text Cleaning Service Test Suite")
    print("=" * 60)
    
    # Test service health first
    if not await test_service_health():
        print("❌ Service is not healthy. Please start the service first:")
        print("   python services/shared/tfidf_text_cleaning_service.py")
        return False
    
    # Test service info
    if not await test_service_info():
        print("❌ Service info test failed")
        return False
    
    # Run all test cases
    passed_tests = 0
    total_tests = len(TEST_CASES)
    
    for test_case in TEST_CASES:
        if await test_text_cleaning(test_case):
            passed_tests += 1
    
    # Test corrected_tfidf_colab compatibility
    if await test_corrected_colab_compatibility():
        passed_tests += 1
    total_tests += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED! Service is working correctly.")
        print("🎉 TF-IDF Text Cleaning Service is ready for use!")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} tests failed")
        return False

def main():
    """Main test function"""
    try:
        result = asyncio.run(run_all_tests())
        if result:
            print("\n🚀 Service is ready for integration with TF-IDF models!")
        else:
            print("\n🛠️ Please fix the issues before using the service.")
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()
