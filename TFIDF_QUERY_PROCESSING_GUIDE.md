# TF-IDF Query Processing Guide

## Complete Step-by-Step Pipeline

### Overview
The TF-IDF query processing pipeline follows these main steps:
1. **Query Reception** (API Endpoint)
2. **Query Cleaning** (Preprocessing)
3. **Vectorization** (Transform to TF-IDF space)
4. **Similarity Computation** (Cosine similarity)
5. **Ranking & Filtering** (Sort by relevance)
6. **Response Generation** (Return results)

---

## 📋 Step-by-Step Process

### Step 1: Query Reception
**What happens:** The system receives a raw user query through the REST API endpoint.

**API Endpoint:** `POST http://localhost:8004/search`

**Input JSON:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 10,
  "similarity_threshold": 0.0,
  "use_enhanced_cleaning": true
}
```

### Step 2: Query Cleaning & Preprocessing
**What happens:** The raw query is processed through several cleaning steps:

1. **Enhanced Text Cleaning Service** (if enabled):
   - Lowercasing
   - HTML tag removal
   - Special character removal
   - Tokenization
   - Stop word removal
   - Lemmatization (finding root word forms)
   - Stemming (reducing words to stems)
   - Spell checking (optional)

2. **Fallback Basic Cleaning** (if service unavailable):
   - Lowercasing
   - Remove special characters
   - Normalize whitespace

**Example Transformation:**
- Original: `"Machine Learning Algorithms for Data Science"`
- Cleaned: `"machin learn algorithm data scienc"`

### Step 3: Vectorization (TF-IDF Transformation)
**What happens:** The cleaned query is converted into a TF-IDF vector.

**The Process:**
1. **Tokenization:** Split into individual terms
2. **Term Frequency (TF):** Count occurrences of each term
3. **Document Frequency (DF):** Use pre-computed document frequencies from training
4. **IDF Calculation:** Calculate Inverse Document Frequency
   ```
   IDF(term) = log(N / DF(term))
   where N = total documents, DF = documents containing term
   ```
5. **TF-IDF Score:** Multiply TF × IDF for each term
6. **L2 Normalization:** Normalize the vector to unit length

**Example:**
- Query: "machine learning"
- TF-IDF Vector: [0.0, 0.7, 0.0, 0.6, 0.0, ...] (sparse vector)

### Step 4: Inverted Index & Document Retrieval
**What happens:** The system uses the inverted index structure for efficient retrieval.

**Inverted Index Structure:**
```python
{
  "machine": [(doc1, 0.3), (doc5, 0.2), (doc12, 0.4)],
  "learning": [(doc1, 0.5), (doc3, 0.3), (doc8, 0.6)],
  "algorithm": [(doc2, 0.4), (doc5, 0.3), (doc9, 0.5)]
}
```

**Process:**
1. For each term in query, lookup documents in inverted index
2. Collect all candidate documents
3. Use pre-computed TF-IDF matrix for exact similarity calculation

### Step 5: Cosine Similarity Calculation
**What happens:** Calculate similarity between query vector and all document vectors.

**Formula:**
```
cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)
```

**Implementation:**
```python
# Query vector: shape (1, vocabulary_size)
query_vector = vectorizer.transform([cleaned_query])

# Document matrix: shape (num_docs, vocabulary_size)
# Pre-computed during offline training

# Cosine similarities: shape (num_docs,)
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
```

**Why Cosine Similarity?**
- Measures angle between vectors (not magnitude)
- Perfect for text similarity (focuses on term patterns, not document length)
- Range: [0, 1] where 1 = identical, 0 = no similarity

### Step 6: Ranking & Filtering
**What happens:** Results are ranked and filtered based on relevance.

**Process:**
1. **Filter by threshold:** Remove documents below similarity threshold
2. **Sort by similarity:** Rank documents in descending order of similarity
3. **Apply top-k:** Return only the top K results
4. **Metadata enrichment:** Add document metadata and ranking information

### Step 7: Response Generation
**What happens:** Generate the final JSON response with ranked results.

**Output JSON:**
```json
{
  "query": "machine learning algorithms",
  "cleaned_query": "machin learn algorithm",
  "results": [
    {
      "doc_id": "doc_123",
      "score": 0.8547,
      "text": "Machine learning algorithms are computational methods...",
      "rank": 1,
      "metadata": {
        "length": 1250,
        "original_length": 1280
      }
    }
  ],
  "total_results": 5,
  "processing_time_ms": 45.2,
  "similarity_stats": {
    "min": 0.1234,
    "max": 0.8547,
    "mean": 0.4521,
    "std": 0.2156
  }
}
```

---

## 🔧 Testing with Postman

### Prerequisites
1. Start the TF-IDF query service:
   ```bash
   cd /Users/raafatmhanna/Desktop/custom-search-engine/backend
   python services/query_processing/tfidf_query_processor.py
   ```

2. Start the enhanced cleaning service:
   ```bash
   python services/shared/enhanced_text_cleaning_service.py
   ```

### Test Cases for Postman

#### Test 1: Basic Query
**URL:** `POST http://localhost:8004/search`
**Headers:** `Content-Type: application/json`
**Body:**
```json
{
  "query": "artificial intelligence",
  "top_k": 5,
  "similarity_threshold": 0.0,
  "use_enhanced_cleaning": true
}
```

#### Test 2: Complex Query with Higher Threshold
**Body:**
```json
{
  "query": "deep learning neural networks for computer vision",
  "top_k": 10,
  "similarity_threshold": 0.2,
  "use_enhanced_cleaning": true
}
```

#### Test 3: Query with Basic Cleaning
**Body:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 3,
  "similarity_threshold": 0.1,
  "use_enhanced_cleaning": false
}
```

#### Test 4: Empty/Invalid Query
**Body:**
```json
{
  "query": "",
  "top_k": 5,
  "similarity_threshold": 0.0,
  "use_enhanced_cleaning": true
}
```

#### Test 5: Health Check
**URL:** `GET http://localhost:8004/health`
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "TF-IDF Query Processor",
  "model_loaded": true,
  "documents_count": 400000,
  "vocabulary_size": 50000
}
```

#### Test 6: Service Status
**URL:** `GET http://localhost:8004/status`
**Expected Response:**
```json
{
  "service": "TF-IDF Query Processor",
  "model_loaded": true,
  "documents_count": 400000,
  "vocabulary_size": 50000,
  "model_info": {
    "vectorizer_features": 50000,
    "matrix_shape": [400000, 50000],
    "ngram_range": [1, 2],
    "max_features": 50000,
    "min_df": 2,
    "max_df": 0.9
  },
  "cleaning_service_status": "available"
}
```

### What to Observe in Responses

1. **Processing Time:** How long each step takes
2. **Similarity Scores:** Range and distribution of scores
3. **Query Transformation:** Original vs cleaned query
4. **Result Quality:** Relevance of returned documents
5. **Metadata:** Document lengths and other stats

---

## 🏗️ Online vs Offline Services in IR Systems

### Offline Services (Training/Indexing Phase)
**When:** Happens periodically, not during user queries
**Purpose:** Prepare the system for fast online retrieval

**TF-IDF Offline Operations:**
1. **Document Preprocessing:**
   - Clean all documents in the collection
   - Apply stemming, lemmatization, stop word removal
   - Tokenization and normalization

2. **Vocabulary Building:**
   - Extract all unique terms across the collection
   - Apply frequency filtering (min_df, max_df)
   - Build term-to-index mapping

3. **TF-IDF Matrix Construction:**
   - Calculate term frequencies for each document
   - Compute document frequencies for each term
   - Build sparse TF-IDF matrix (documents × vocabulary)
   - Apply L2 normalization

4. **Inverted Index Creation:**
   - Build term → document list mappings
   - Store TF-IDF scores for efficient lookup
   - Create auxiliary data structures

5. **Model Serialization:**
   - Save vectorizer (vocabulary + parameters)
   - Save TF-IDF matrix (sparse format)
   - Save document metadata
   - Store on disk for online loading

**Characteristics:**
- ⏰ **Time:** Can take hours/days for large collections
- 🔄 **Frequency:** Run periodically (daily/weekly/monthly)
- 💾 **Storage:** Heavy disk I/O, large memory usage
- 🔧 **Resources:** Can use distributed computing
- 📊 **Output:** Pre-computed models and indices

### Online Services (Query Processing Phase)
**When:** Happens during user queries in real-time
**Purpose:** Provide fast, relevant search results

**TF-IDF Online Operations:**
1. **Query Reception:**
   - Receive user query via API
   - Parse request parameters

2. **Query Preprocessing:**
   - Apply same cleaning as training data
   - Use lightweight, fast operations only

3. **Query Vectorization:**
   - Transform query using pre-trained vectorizer
   - Generate sparse TF-IDF query vector

4. **Similarity Computation:**
   - Calculate cosine similarity with pre-computed matrix
   - Use optimized linear algebra operations

5. **Ranking & Filtering:**
   - Sort results by similarity score
   - Apply top-k and threshold filtering
   - Enrich with metadata

6. **Response Generation:**
   - Format results as JSON
   - Include timing and statistics

**Characteristics:**
- ⚡ **Speed:** Must respond in milliseconds
- 🔄 **Frequency:** Continuous, thousands per second
- 💾 **Memory:** Uses pre-loaded models in RAM
- 🎯 **Focus:** Optimize for latency and throughput
- 📊 **Output:** Real-time search results

### Key Differences

| Aspect | Offline (Training) | Online (Serving) |
|--------|------------------|------------------|
| **Latency** | Hours to days | Milliseconds |
| **Data Size** | Entire collection | Single query |
| **Complexity** | Heavy ML operations | Simple lookups |
| **Resources** | High CPU/Memory/Disk | Optimized RAM usage |
| **Scalability** | Batch processing | Real-time serving |
| **Updates** | Periodic rebuilds | No model changes |

### Architecture Example

```
Offline Pipeline:
Raw Documents → Cleaning → TF-IDF Training → Model Storage
     ↓
[Scheduled Jobs, ETL, Batch Processing]

Online Pipeline:
User Query → API → Query Processing → Model Lookup → Results
     ↓
[Real-time Services, Low Latency, High Throughput]
```

### Why This Separation?

1. **Performance:** Offline pre-computation enables fast online serving
2. **Scalability:** Can handle large document collections offline
3. **Reliability:** Online services are simpler and more stable
4. **Resource Efficiency:** Heavy computation done once, not per query
5. **Flexibility:** Can retrain models without affecting online service

This architecture pattern is fundamental to most production IR systems, including search engines, recommendation systems, and question-answering systems.
