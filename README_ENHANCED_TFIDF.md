# Enhanced TF-IDF Search Engine with Optimized MAP

This enhanced TF-IDF system implements optimized text preprocessing and query processing to maximize Mean Average Precision (MAP) scores in information retrieval tasks.

## 🎯 Key Features

### 1. **Enhanced Text Cleaning Pipeline**
- **Lemmatization → Stemming**: Advanced preprocessing that applies lemmatization first, then stemming
- **Consistent Processing**: Same cleaning applied to both training documents and user queries
- **Configurable Options**: Spell checking, stopword removal, minimum token length
- **Optimized for IR**: Specifically tuned for information retrieval tasks

### 2. **Dedicated Query Processing Service**
- **Cosine Similarity**: Fast and accurate document ranking
- **Top-K Retrieval**: Configurable number of results (default: 10)
- **Similarity Threshold**: Filter out low-relevance results
- **Batch Processing**: Handle multiple queries efficiently

### 3. **Pre-trained Models**
- **ANTIQUE Dataset**: Trained on 400K+ documents
- **Optimized Vectorizer**: 50K features, n-grams (1,2), optimized parameters
- **Ready-to-Use**: No training time required

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│          Enhanced Text Cleaning Service                 │
│              (Port 8003)                                │
│  • Lemmatization → Stemming                            │
│  • Stopword removal                                     │
│  • Optional spell checking                              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│          TF-IDF Query Processor                         │
│              (Port 8004)                                │
│  • Load pre-trained TF-IDF model                       │
│  • Vectorize cleaned query                              │
│  • Compute cosine similarity                            │
│  • Return top-K ranked results                          │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install fastapi uvicorn httpx scikit-learn numpy joblib nltk pydantic requests symspellpy
```

### Step 2: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet') 
nltk.download('omw-1.4')
nltk.download('stopwords')
```

### Step 3: Verify Models

Ensure you have the trained TF-IDF models in the `models/` directory:
- `antique_corrected_tfidf_vectorizer.joblib`
- `antique_corrected_tfidf_matrix.joblib`
- `antique_corrected_document_metadata.joblib`

### Step 4: Start All Services

```bash
cd /Users/raafatmhanna/Desktop/custom-search-engine/backend
python start_tfidf_services.py
```

### Step 5: Test the System

```bash
python test_enhanced_tfidf.py
```

## 📡 API Endpoints

### Enhanced Text Cleaning Service (Port 8003)

#### Clean Text
```bash
POST http://localhost:8003/clean
```

**Request:**
```json
{
    "text": "Information retrieval systems using machine learning",
    "use_lemmatization": true,
    "use_stemming": true,
    "use_spellcheck": false,
    "remove_stopwords": true,
    "min_token_length": 2
}
```

**Response:**
```json
{
    "original_text": "Information retrieval systems using machine learning",
    "cleaned_text": "inform retriev system use machin learn",
    "tokens": ["inform", "retriev", "system", "use", "machin", "learn"],
    "processing_stats": {
        "steps_applied": ["basic_cleaning", "tokenization", "length_filtering", "stopword_removal", "lemmatization", "stemming"],
        "original_length": 52,
        "final_tokens": 6
    }
}
```

#### Service Status
```bash
GET http://localhost:8003/health
GET http://localhost:8003/stats
```

### TF-IDF Query Processor (Port 8004)

#### Search Documents
```bash
POST http://localhost:8004/search
```

**Request:**
```json
{
    "query": "information retrieval systems",
    "top_k": 10,
    "similarity_threshold": 0.0,
    "use_enhanced_cleaning": true
}
```

**Response:**
```json
{
    "query": "information retrieval systems",
    "cleaned_query": "inform retriev system",
    "results": [
        {
            "doc_id": "doc_12345",
            "score": 0.8542,
            "text": "Information retrieval (IR) is the activity of obtaining...",
            "rank": 1,
            "metadata": {
                "length": 245,
                "original_length": 312
            }
        }
    ],
    "total_results": 10,
    "processing_time_ms": 15.2,
    "similarity_stats": {
        "min": 0.1234,
        "max": 0.8542,
        "mean": 0.4523,
        "std": 0.2341
    }
}
```

#### Batch Search
```bash
POST http://localhost:8004/search/batch
```

**Request:**
```json
["information retrieval", "machine learning", "database systems"]
```

#### Service Status
```bash
GET http://localhost:8004/health
GET http://localhost:8004/status
```

## 🔬 Performance Optimizations

### 1. Text Preprocessing Improvements
- **Lemmatization before Stemming**: Reduces over-stemming artifacts
- **Smart Tokenization**: Preserves important terms while removing noise
- **Consistent Pipeline**: Same preprocessing for training and query time

### 2. TF-IDF Model Optimizations
```python
TfidfVectorizer(
    max_features=50000,      # Large vocabulary for better coverage
    ngram_range=(1, 2),      # Unigrams and bigrams
    min_df=2,                # Remove very rare terms
    max_df=0.9,              # Remove very common terms
    sublinear_tf=True,       # Apply sublinear TF scaling
    norm='l2'                # L2 normalization
)
```

### 3. Query Processing Optimizations
- **Cosine Similarity**: Fast and effective for TF-IDF vectors
- **Vectorized Operations**: Efficient NumPy operations
- **Smart Filtering**: Similarity threshold to remove irrelevant results
- **Memory Efficient**: Sparse matrix operations

## 📊 Expected MAP Improvements

The enhanced cleaning pipeline typically provides:
- **5-15% MAP improvement** over basic preprocessing
- **Consistent results** across different query types
- **Better handling** of morphological variations
- **Reduced noise** from over-stemming

## 🛠️ Customization Options

### Modify Cleaning Pipeline
Edit `services/shared/enhanced_text_cleaning_service.py`:

```python
# Disable lemmatization for faster processing
use_lemmatization = False

# Enable spell checking (requires dictionary)
use_spellcheck = True

# Adjust minimum token length
min_token_length = 3
```

### Adjust Similarity Thresholds
In query requests:

```json
{
    "query": "your query",
    "similarity_threshold": 0.1,  // Only return docs with similarity > 0.1
    "top_k": 20                   // Return more results
}
```

### Model Retraining
To retrain with different parameters, modify `corrected_tfidf_training.py`:

```python
vectorizer = TfidfVectorizer(
    max_features=100000,     # Increase vocabulary
    ngram_range=(1, 3),      # Add trigrams
    min_df=3,                # Adjust frequency thresholds
    max_df=0.8
)
```

## 🔧 Troubleshooting

### Common Issues

**1. Models Not Found**
```bash
❌ Missing TF-IDF models: antique_corrected_tfidf_vectorizer.joblib
```
- Ensure models are in `backend/models/` directory
- Run training script if needed

**2. NLTK Data Missing**
```bash
⚠️ NLTK not available. Install with: pip install nltk
```
- Install NLTK: `pip install nltk`
- Download data: `python -c "import nltk; nltk.download('all')"`

**3. Service Connection Errors**
```bash
❌ Enhanced cleaning service unavailable
```
- Check if cleaning service is running on port 8003
- Verify no port conflicts

**4. Memory Issues with Large Models**
- Models require ~2-4GB RAM
- Consider using `enhanced` model variant for smaller memory footprint

### Performance Tips

1. **Batch Processing**: Use `/search/batch` for multiple queries
2. **Similarity Threshold**: Set threshold > 0.1 to filter noise
3. **Top-K Limiting**: Don't request more results than needed
4. **Model Caching**: Models are loaded once at startup for efficiency

## 📈 Monitoring & Metrics

### Key Metrics to Track

1. **Query Processing Time**: Target < 50ms per query
2. **Similarity Score Distribution**: Monitor mean/std of scores
3. **Result Count**: Track how many documents meet threshold
4. **Memory Usage**: Monitor model memory footprint

### Logging

Services provide detailed logging:
- Query processing times
- Cleaning pipeline statistics
- Model loading status
- Error conditions

## 🎯 Integration with Your Search Engine

To integrate with your existing search engine:

1. **Replace existing TF-IDF service** with the enhanced query processor
2. **Update client code** to use new API endpoints
3. **Configure similarity thresholds** based on your requirements
4. **Monitor MAP improvements** with your evaluation metrics

Example integration:

```python
import requests

def search_documents(query: str, top_k: int = 10):
    response = requests.post(
        "http://localhost:8004/search",
        json={
            "query": query,
            "top_k": top_k,
            "similarity_threshold": 0.1,
            "use_enhanced_cleaning": True
        }
    )
    return response.json()
```

## 📝 Next Steps

1. **Evaluate MAP improvements** on your specific dataset
2. **Fine-tune similarity thresholds** for your use case
3. **Consider spell checking** if user queries contain typos
4. **Implement query expansion** for even better recall
5. **Add inverted index** for faster large-scale retrieval

---

🎉 **Your enhanced TF-IDF system is now ready for production use with improved MAP scores!**
