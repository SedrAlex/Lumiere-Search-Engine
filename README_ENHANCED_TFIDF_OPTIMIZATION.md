# Enhanced TF-IDF Optimization for Higher MAP Scores

## Overview

This document outlines the implementation of enhanced TF-IDF services designed to increase Mean Average Precision (MAP) from 0.17 to at least 0.4. The optimization includes multiple improvements to the TF-IDF pipeline, query processing, and document retrieval.

## Current Issue Analysis

Your current TF-IDF implementation achieved a MAP score of 0.17, which is relatively low. The main issues identified were:

1. **Limited Vocabulary Size**: Only 10,000 features vs. potential 50,000-100,000
2. **Basic N-gram Range**: Only unigrams and bigrams (1,2) vs. trigrams (1,3)
3. **No Query Expansion**: Missing query term expansion for better recall
4. **No Semantic Understanding**: Lack of semantic similarity beyond exact term matching
5. **Simple Scoring**: Basic cosine similarity without advanced ranking techniques
6. **Text Cleaning Inconsistency**: Different preprocessing between training and inference

## Enhanced Architecture

### 1. Inverted Index Service (Port 8006)
A standalone high-performance inverted index service that provides:

**Features:**
- Multiple scoring methods: TF-IDF, BM25, Count-based
- Disjunctive and conjunctive query types
- Persistent index storage with joblib
- Term and collection statistics
- Optimized posting lists with normalized TF scores

**Key Improvements:**
- Sublinear TF scaling: `1 + log(tf)`
- BM25 scoring with tunable parameters (k1=1.5, b=0.75)
- Efficient candidate document filtering
- Memory-optimized data structures

### 2. Enhanced TF-IDF Service (Port 8007)
An advanced TF-IDF service with multiple optimization layers:

**Core Enhancements:**
- **Increased Vocabulary**: 100,000 features (10x increase)
- **Extended N-grams**: (1,3) range for better phrase matching
- **Query Expansion**: Term co-occurrence based expansion
- **Semantic Reranking**: LSA with 300 components
- **Hybrid Search**: Combines inverted index + TF-IDF cosine similarity

**Advanced Features:**
- Term co-occurrence analysis for query expansion
- Weighted score combination (70% cosine + 30% inverted index)
- Semantic similarity with Truncated SVD (LSA)
- Detailed scoring explanations for transparency
- Enhanced text preprocessing pipeline

## Implementation Details

### TF-IDF Vectorizer Parameters

**Original Configuration:**
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    sublinear_tf=True
)
```

**Enhanced Configuration:**
```python
TfidfVectorizer(
    max_features=100000,      # 10x increase
    ngram_range=(1, 3),       # Include trigrams
    min_df=2,                 # Keep low for better coverage
    max_df=0.85,              # Slightly more restrictive
    sublinear_tf=True,        # Apply log normalization
    norm='l2',                # L2 normalization
    use_idf=True,             # Use IDF weighting
    smooth_idf=True,          # Smooth IDF weights
    strip_accents='unicode'   # Better text normalization
)
```

### Query Expansion Algorithm

The system builds term co-occurrence matrices and calculates similarities:

```python
def _expand_query(self, query_terms, max_expansions=3):
    expanded_terms = list(query_terms)
    
    for term in query_terms:
        if term in self.term_similarities:
            similar_terms = self.term_similarities[term][:max_expansions]
            for similar_term, similarity in similar_terms:
                if similarity > 0.1 and similar_term not in expanded_terms:
                    expanded_terms.append(similar_term)
    
    return expanded_terms
```

### Semantic Reranking

LSA-based semantic reranking combines TF-IDF and semantic scores:

```python
# 60% TF-IDF + 40% semantic similarity
combined_scores = 0.6 * tfidf_scores + 0.4 * semantic_similarities
```

### Hybrid Search Strategy

1. **Candidate Retrieval**: Use inverted index for fast candidate selection
2. **TF-IDF Scoring**: Apply cosine similarity on candidates
3. **Score Combination**: Weighted combination of both scores
4. **Semantic Reranking**: Final reranking with LSA if enabled

## Getting Started

### 1. Start Enhanced Services

```bash
# Start all enhanced services
python start_enhanced_services.py
```

This starts three services:
- TF-IDF Text Cleaning Service (Port 8005)
- Inverted Index Service (Port 8006)  
- Enhanced TF-IDF Service (Port 8007)

### 2. Verify Services

```bash
# Check service health
curl -X GET http://localhost:8005/health
curl -X GET http://localhost:8006/health
curl -X GET http://localhost:8007/health
```

### 3. Run Evaluation

```bash
# Comprehensive evaluation with different configurations
python evaluate_enhanced_tfidf.py

# Custom service URL
python evaluate_enhanced_tfidf.py --service-url http://localhost:8007
```

## Evaluation Configurations

The evaluation tests four configurations:

1. **Baseline**: Enhanced TF-IDF without query expansion or reranking
2. **Query Expansion Only**: With term expansion but no semantic reranking
3. **Reranking Only**: With LSA reranking but no query expansion
4. **Full Enhanced**: All optimizations enabled

## Expected MAP Score Improvements

Based on the enhancements, expected MAP improvements:

| Configuration | Expected MAP | Improvement |
|---------------|--------------|-------------|
| Original | 0.17 | Baseline |
| Enhanced Baseline | 0.25-0.30 | +47-76% |
| + Query Expansion | 0.30-0.35 | +76-106% |
| + Semantic Reranking | 0.35-0.42 | +106-147% |
| Full Enhanced | 0.40-0.45 | +135-165% |

## Key Performance Optimizations

### 1. Vocabulary Size Impact
- **10k features**: Limited term coverage, higher OOV rate
- **100k features**: Better term coverage, lower OOV rate, improved precision

### 2. N-gram Enhancement
- **Unigrams + Bigrams**: Basic phrase matching
- **Unigrams + Bigrams + Trigrams**: Better phrase and concept matching

### 3. Query Expansion Benefits
- **Term Co-occurrence**: Finds related terms based on corpus statistics
- **Similarity Threshold**: 0.1 minimum to avoid noise
- **Expansion Limit**: Max 3 terms per original term to control query drift

### 4. Semantic Reranking Advantages
- **LSA Components**: 300 components capture semantic relationships
- **Score Weighting**: 60% lexical + 40% semantic for balanced ranking
- **L2 Normalization**: Ensures proper vector space operations

## API Usage Examples

### Index Documents (Enhanced)

```python
import httpx

async def index_documents():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8007/index",
            json={
                "documents": [
                    {
                        "id": "doc1",
                        "text": "Machine learning algorithms for data analysis",
                        "metadata": {"category": "AI"}
                    }
                ],
                "use_enhanced_parameters": True,
                "enable_query_expansion": True
            }
        )
        return response.json()
```

### Enhanced Search

```python
async def enhanced_search():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8007/search",
            json={
                "query": "machine learning data",
                "top_k": 10,
                "use_query_expansion": True,
                "enable_reranking": True,
                "similarity_threshold": 0.0
            }
        )
        return response.json()
```

### Query Inverted Index

```python
async def query_inverted_index():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8006/query_index",
            json={
                "terms": ["machine", "learning", "data"],
                "top_k": 50,
                "query_type": "disjunctive",
                "scoring_method": "bm25"
            }
        )
        return response.json()
```

## Monitoring and Debugging

### Service Status

```bash
# Detailed service status
curl -X GET http://localhost:8007/status
```

### Collection Statistics

```bash
# Get collection and model statistics
curl -X GET http://localhost:8007/collection_stats
```

### Term Information

```bash
# Get information about specific terms
curl -X GET http://localhost:8006/term/machine
```

## Further Optimization Suggestions

If MAP scores don't reach 0.4, consider these additional improvements:

### 1. Advanced Query Expansion
- **Pseudo-Relevance Feedback**: Use top-ranked documents to expand queries
- **Word Embeddings**: Use pre-trained embeddings for semantic expansion
- **Synonym Expansion**: Integrate WordNet or domain-specific thesauri

### 2. Document Boosting
- **Recency Boost**: Boost newer documents based on timestamps
- **Quality Signals**: Use document length, readability scores
- **Authority Boost**: Boost documents from authoritative sources

### 3. Advanced Ranking
- **Learning to Rank**: Train ML models on query-document features
- **Neural Reranking**: Use BERT or similar models for reranking
- **Ensemble Methods**: Combine multiple scoring methods

### 4. Parameter Tuning
- **Grid Search**: Systematically tune all parameters
- **Bayesian Optimization**: Efficient parameter optimization
- **Cross-Validation**: Robust evaluation across different query sets

### 5. Text Processing
- **Named Entity Recognition**: Special handling for entities
- **Concept Extraction**: Extract and index key concepts
- **Multilingual Support**: Handle multiple languages if applicable

## Troubleshooting

### Common Issues

1. **Services Won't Start**
   - Check port availability
   - Verify Python dependencies
   - Check log files for errors

2. **Low MAP Scores**
   - Verify text cleaning consistency
   - Check vocabulary coverage
   - Validate query expansion quality

3. **Performance Issues**
   - Monitor memory usage with large vocabularies
   - Check disk space for model storage
   - Consider batching for large document sets

### Debug Commands

```bash
# Check service logs
tail -f /var/log/enhanced_tfidf.log

# Monitor resource usage
htop

# Test individual components
python -m pytest tests/test_enhanced_tfidf.py
```

## Conclusion

The enhanced TF-IDF implementation provides multiple layers of optimization designed to significantly improve MAP scores. The modular architecture allows for incremental improvements and easy experimentation with different configurations.

The key to achieving MAP ≥ 0.4 lies in the combination of:
1. **Increased vocabulary size** for better term coverage
2. **Query expansion** for improved recall
3. **Semantic reranking** for better precision
4. **Hybrid search** for efficiency and accuracy

Start with the enhanced services and run the evaluation to measure improvements. Based on the results, further optimizations can be applied incrementally.
