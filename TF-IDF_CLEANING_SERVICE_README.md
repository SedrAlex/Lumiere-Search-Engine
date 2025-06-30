# TF-IDF Text Cleaning Service

## Overview

This is a specialized text cleaning service that implements the **EXACT same** text preprocessing pipeline as used in `corrected_tfidf_colab`. It's designed specifically for TF-IDF models and ensures consistent text processing across your search engine.

## Service Architecture

```
🔄 Text Processing Pipeline (matches corrected_tfidf_colab):
1. ⬇️ Lowercase conversion
2. 🗑️ HTML tag removal  
3. 🧹 Special character removal (keep only alphanumeric + spaces)
4. 📏 Whitespace normalization
5. ✂️ Tokenization (using NLTK word_tokenize)
6. 🚫 Token filtering (length >= 2, alphanumeric, non-stopword)
7. 📝 Lemmatization (using WordNetLemmatizer)
8. 🌱 Stemming (using PorterStemmer)
```

## Service Details

- **Port**: 8005 (dedicated for TF-IDF)
- **URL**: `http://localhost:8005`
- **Compatible with**: corrected_tfidf_colab
- **Optimized for**: TF-IDF vectorization

## API Endpoints

### 1. Health Check
```bash
GET /health
```
Returns service health status and component availability.

### 2. Service Information
```bash
GET /info
```
Returns detailed service configuration and processing pipeline info.

### 3. Text Cleaning
```bash
POST /clean
Content-Type: application/json

{
  "text": "Your text to clean",
  "preserve_document_structure": true
}
```

**Response:**
```json
{
  "original_text": "Your text to clean",
  "cleaned_text": "your text clean",
  "tokens": ["your", "text", "clean"],
  "token_count": 3,
  "processing_stats": {
    "original_length": 18,
    "steps_applied": ["lowercase_conversion", "html_tag_removal", ...],
    "final_token_count": 3,
    "compression_ratio": 0.72,
    "nltk_available": true,
    "stemmer_used": true,
    "lemmatizer_used": true
  }
}
```

## Usage

### Starting the Service
```bash
# Start the dedicated TF-IDF cleaning service
python services/shared/tfidf_text_cleaning_service.py
```

### Testing the Service
```bash
# Run the comprehensive test suite
python test_tfidf_cleaning_service.py
```

### Integration with TF-IDF Service
The TF-IDF service (`services/representation/tfidf_service.py`) is already configured to use this cleaning service:

```python
# Configuration in tfidf_service.py
TEXT_CLEANING_SERVICE_URL = "http://localhost:8005"  # Uses dedicated TF-IDF cleaning
```

## Service Comparison

| Service | Port | Purpose | Compatible With |
|---------|------|---------|----------------|
| **TF-IDF Text Cleaning** | 8005 | TF-IDF models | corrected_tfidf_colab |
| Basic Text Cleaning | 8001 | Other models | General purpose |
| Custom Text Cleaning | 8004 | Legacy TF-IDF | Previous implementation |

## Starting All TF-IDF Services

Use the orchestrator to start all TF-IDF related services:

```bash
python start_tfidf_services.py
```

This will start:
1. **TF-IDF Text Cleaning Service** (Port 8005)
2. **TF-IDF Representation Service** (Port 8002)

## Testing Examples

### Basic Text Cleaning
```bash
curl -X POST http://localhost:8005/clean \
  -H "Content-Type: application/json" \
  -d '{"text": "Information Retrieval Systems!"}'
```

### HTML Content Cleaning
```bash
curl -X POST http://localhost:8005/clean \
  -H "Content-Type: application/json" \
  -d '{"text": "<h1>Machine Learning</h1> <p>Natural Language Processing</p>"}'
```

### Check Service Health
```bash
curl http://localhost:8005/health
```

## Key Features

✅ **Exact Compatibility**: Matches corrected_tfidf_colab preprocessing  
✅ **NLTK Integration**: Automatic NLTK data download and initialization  
✅ **Robust Error Handling**: Graceful degradation if NLTK unavailable  
✅ **Detailed Stats**: Comprehensive processing statistics  
✅ **Performance Optimized**: Efficient tokenization and filtering  
✅ **RESTful API**: Easy integration with other services  

## Dependencies

The service automatically handles NLTK data downloads:
- punkt (tokenization)
- wordnet (lemmatization)
- omw-1.4 (WordNet extension)
- stopwords (English stopwords)

## Troubleshooting

### Service Won't Start
1. Check if port 8005 is available
2. Ensure NLTK dependencies are installed: `pip install nltk`
3. Check logs for specific error messages

### Text Cleaning Issues
1. Run the test suite: `python test_tfidf_cleaning_service.py`
2. Check service health: `curl http://localhost:8005/health`
3. Verify NLTK components are loaded properly

### Integration Issues
1. Ensure TF-IDF service is pointing to port 8005
2. Check network connectivity between services
3. Verify request/response formats match API specification

## Performance Notes

- **Tokenization**: Uses NLTK's word_tokenize for accuracy
- **Lemmatization + Stemming**: Applied in the same order as corrected_tfidf_colab
- **Memory Efficient**: Processes text incrementally
- **Caching**: NLTK components are cached after first load

---

**Important**: This service is specifically designed to maintain compatibility with the corrected_tfidf_colab approach. If you need different text preprocessing for other models, use the basic text cleaning service on port 8001.
