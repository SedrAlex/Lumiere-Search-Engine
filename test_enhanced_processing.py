#!/usr/bin/env python3
"""
Test Enhanced Text Processing Components
Verify that enhanced text cleaning and tokenization work correctly.
"""

import sys
sys.path.append('.')

from services.shared.enhanced_text_cleaning_service import EnhancedTextCleaningService
from services.shared.enhanced_tokenizer import EnhancedTokenizer
from services.shared.tfidf_service import SharedTFIDFService

def test_enhanced_text_cleaning():
    """Test enhanced text cleaning service."""
    print("=== Testing Enhanced Text Cleaning ===")
    
    # Initialize with spell checking
    cleaner = EnhancedTextCleaningService(enable_spell_check=True)
    
    # Test texts with various issues
    test_texts = [
        "I'm looking for beautifull antique vases that're realy old.",
        "Can't find any informatoin about this vintag piece?",
        "What's the pric of thes antique furnatur items?",
        "Looking for antique books from the 1800s that cost $100-200.",
        "Beautiful <b>antique</b> furniture from https://example.com"
    ]
    
    print("\nOriginal -> TF-IDF Processed:")
    for text in test_texts:
        processed = cleaner.preprocess_for_tfidf(text)
        print(f"'{text}' -> '{processed}'")
    
    # Get service info
    info = cleaner.get_service_info()
    print(f"\nService info: {info}")
    
    print("✓ Enhanced text cleaning test completed")

def test_enhanced_tokenizer():
    """Test enhanced tokenizer."""
    print("\n=== Testing Enhanced Tokenizer ===")
    
    # Initialize tokenizer
    tokenizer = EnhancedTokenizer(
        enable_spell_check=True,
        enable_lemmatization=True,
        enable_stemming=True
    )
    
    # Test texts
    test_texts = [
        "I'm looking for beautiful antique vases",
        "These items are really expensive",
        "The books were written in the 1800s"
    ]
    
    print("\nText -> Tokens:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"'{text}' -> {tokens}")
    
    # Get tokenizer info
    info = tokenizer.get_tokenizer_info()
    print(f"\nTokenizer info: {info}")
    
    print("✓ Enhanced tokenizer test completed")

def test_shared_tfidf_service():
    """Test shared TF-IDF service with enhanced processing."""
    print("\n=== Testing Shared TF-IDF Service ===")
    
    # Initialize service
    service = SharedTFIDFService(
        models_dir="test_models",
        enable_spell_check=True,
        enable_lemmatization=True,
        enable_stemming=True
    )
    
    # Sample documents
    documents = [
        "Beautiful antique furniture from the Victorian era including chairs and tables",
        "Vintage collectibles and rare items for antique collectors",
        "Old books and manuscripts from the 18th and 19th centuries",
        "Classic cars and automotive memorabilia from the early 1900s",
        "Antique jewelry including rings, necklaces, and bracelets"
    ]
    
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
    
    # Train the model
    print("Training TF-IDF model...")
    stats = service.train_tfidf(documents, doc_ids)
    
    print(f"Training completed. Stats:")
    print(f"  - Total documents: {stats['total_documents']}")
    print(f"  - Valid documents: {stats['valid_documents']}")
    print(f"  - Vocabulary size: {stats['vocabulary_size']}")
    print(f"  - Matrix sparsity: {stats['sparsity']:.2f}%")
    
    # Test search
    test_queries = [
        "antique furniture",
        "vintage books",
        "classic cars"
    ]
    
    print("\nTesting search functionality:")
    for query in test_queries:
        results = service.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Doc {result['doc_id']} (Score: {result['score']:.4f})")
    
    # Save models
    print("\nSaving models...")
    success = service.save_models("test_enhanced")
    print(f"Models saved: {success}")
    
    # Test loading
    print("Testing model loading...")
    new_service = SharedTFIDFService(models_dir="test_models")
    load_success = new_service.load_models("test_enhanced")
    print(f"Models loaded: {load_success}")
    
    if load_success:
        # Test search with loaded model
        results = new_service.search("antique furniture", top_k=2)
        print(f"Search with loaded model returned {len(results)} results")
    
    print("✓ Shared TF-IDF service test completed")

def test_comparison_with_original():
    """Compare enhanced processing with simpler processing."""
    print("\n=== Comparison Test ===")
    
    # Test text with spelling errors
    test_text = "I'm looking for beautifull antique vases that're realy expensiv"
    
    # Enhanced processing
    enhanced_cleaner = EnhancedTextCleaningService(enable_spell_check=True)
    enhanced_result = enhanced_cleaner.preprocess_for_tfidf(test_text)
    
    # Simple processing (without spell check)
    simple_cleaner = EnhancedTextCleaningService(enable_spell_check=False)
    simple_result = simple_cleaner.preprocess_for_tfidf(test_text)
    
    print(f"Original: '{test_text}'")
    print(f"Enhanced: '{enhanced_result}'")
    print(f"Simple:   '{simple_result}'")
    
    print("✓ Comparison test completed")

if __name__ == "__main__":
    try:
        test_enhanced_text_cleaning()
        test_enhanced_tokenizer()
        test_shared_tfidf_service()
        test_comparison_with_original()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
