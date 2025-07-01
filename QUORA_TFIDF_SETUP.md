# Quora TF-IDF Search Engine Setup

This guide explains how to set up and use the TF-IDF search engine for the Quora dataset, following the same architecture as the Antique implementation.

## Overview

The Quora TF-IDF implementation consists of:

1. **Quora Dataset Loading Service** - Loads and preprocesses Quora data
2. **TF-IDF Quora Representation Service** - Provides TF-IDF-based search functionality
3. **Colab Training Notebook** - Trains TF-IDF models on the Quora dataset
4. **Service Orchestrator** - Manages all services

## Architecture

```
┌─────────────────────┐    ┌──────────────────────────┐
│  Quora Dataset      │    │  TF-IDF Quora           │
│  Loading Service    │───▶│  Representation Service  │
│  (Port: 8004)       │    │  (Port: 8006)            │
└─────────────────────┘    └──────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────┐    ┌──────────────────────────┐
│  Quora Dataset      │    │  TF-IDF Models           │
│  (CSV/JSON/JSONL)   │    │  - vectorizer.joblib     │
└─────────────────────┘    │  - matrix.joblib         │
                           │  - metadata.joblib       │
                           └──────────────────────────┘
```

## Prerequisites

1. Python 3.8+
2. Required packages (installed via requirements.txt):
   - fastapi
   - uvicorn
   - httpx
   - scikit-learn
   - numpy
   - joblib
   - pydantic

## Step 1: Dataset Preparation

Prepare your Quora dataset in one of these formats:

### CSV Format
```csv
id,text
1,"What is machine learning?"
2,"How does neural network work?"
```

### JSON Format
```json
[
  {"id": "1", "text": "What is machine learning?"},
  {"id": "2", "text": "How does neural network work?"}
]
```

### JSONL Format
```jsonl
{"id": "1", "text": "What is machine learning?"}
{"id": "2", "text": "How does neural network work?"}
```

## Step 2: Model Training with Colab

1. **Open the Colab Notebook**:
   ```
   Upload tfidf_quora_colab.ipynb to Google Colab
   ```

2. **Upload Your Dataset**:
   - Run the upload cell in the notebook
   - Select your Quora dataset file

3. **Train the Model**:
   - Follow the notebook steps to:
     - Preprocess the text data
     - Train the TF-IDF vectorizer
     - Evaluate the model
     - Save the trained components

4. **Download Trained Models**:
   - The notebook will generate:
     - `quora_tfidf_vectorizer.joblib`
     - `quora_tfidf_matrix.joblib`
     - `quora_document_metadata.joblib`
     - `quora_tfidf_training_report.json`

## Step 3: Deploy Models

1. **Create Models Directory**:
   ```bash
   mkdir -p models
   ```

2. **Upload Trained Models**:
   ```bash
   # Copy the downloaded models to your server
   cp quora_tfidf_*.joblib models/
   # Or place them in /tmp/ directory
   cp quora_tfidf_*.joblib /tmp/
   ```

## Step 4: Start Services

### Option 1: Use the Orchestrator (Recommended)
```bash
./start_quora_tfidf_services.py
```

### Option 2: Start Services Manually

1. **Start Quora Dataset Loading Service**:
   ```bash
   python services/data/quora_loader_service.py
   ```

2. **Start TF-IDF Quora Representation Service**:
   ```bash
   python services/representation/tfidf_quora_service.py
   ```

## Step 5: Load and Index Data

### Load Quora Dataset
```bash
curl -X POST http://localhost:8004/load \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "/path/to/your/quora_dataset.csv",
    "max_documents": 10000,
    "representation_services": ["tfidf_quora"]
  }'
```

### Response Example
```json
{
  "message": "Loaded and indexed 10000 documents",
  "total_documents": 10000,
  "services_indexed": {
    "tfidf_quora": true
  },
  "processing_time": 45.2
}
```

## Step 6: Search Documents

### Basic Search
```bash
curl -X POST http://localhost:8006/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 5
  }'
```

### Response Example
```json
{
  "query": "machine learning algorithms",
  "results": [
    {
      "document_id": "123",
      "score": 0.8945,
      "text": "What are the best machine learning algorithms for beginners?",
      "metadata": {}
    },
    {
      "document_id": "456",
      "score": 0.7821,
      "text": "How do I choose the right machine learning algorithm?",
      "metadata": {}
    }
  ],
  "total_results": 2,
  "processing_time": 0.045
}
```

## API Endpoints

### Quora Dataset Loading Service (Port 8004)

- `GET /` - Service information
- `GET /health` - Health check
- `POST /load` - Load and index documents
- `POST /load_documents` - Load documents only (no indexing)

### TF-IDF Quora Representation Service (Port 8006)

- `GET /health` - Health check
- `GET /status` - Service status and statistics
- `POST /index` - Index documents
- `POST /search` - Search documents

## Configuration

### Service Ports
- Quora Loading Service: 8004
- TF-IDF Quora Service: 8006

### Model Paths
The service looks for models in these locations:
1. `models/` directory (relative to project root)
2. `/tmp/` directory

### TF-IDF Configuration
```python
tfidf_config = {
    'max_features': 10000,      # Vocabulary size limit
    'ngram_range': (1, 2),      # Unigrams and bigrams
    'min_df': 2,                # Minimum document frequency
    'max_df': 0.8,              # Maximum document frequency
    'sublinear_tf': True,       # Sublinear TF scaling
    'use_idf': True,            # Enable IDF
    'smooth_idf': True,         # Smooth IDF weights
    'norm': 'l2'                # L2 normalization
}
```

## Troubleshooting

### Service Won't Start
1. Check if ports are available:
   ```bash
   lsof -i :8004
   lsof -i :8006
   ```

2. Check dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Models Not Found
1. Verify model files exist:
   ```bash
   ls -la models/quora_tfidf_*.joblib
   ls -la /tmp/quora_tfidf_*.joblib
   ```

2. Re-train models using the Colab notebook

### Poor Search Results
1. Check dataset quality and preprocessing
2. Experiment with TF-IDF parameters in the Colab notebook
3. Increase vocabulary size (`max_features`)
4. Adjust n-gram range

### Memory Issues
1. Reduce `max_documents` when loading
2. Decrease `max_features` in TF-IDF configuration
3. Use document batching for large datasets

## Performance Optimization

### For Large Datasets
1. **Batch Processing**: Process documents in smaller batches
2. **Vocabulary Limitation**: Set appropriate `max_features`
3. **Sparse Matrix Storage**: Models automatically use sparse matrices
4. **Memory Monitoring**: Monitor RAM usage during indexing

### Search Optimization
1. **Pre-filtering**: Filter documents before TF-IDF calculation
2. **Caching**: Cache frequent queries
3. **Index Optimization**: Rebuild index periodically

## Integration with Main Application

To integrate with your main search application:

```python
import httpx

async def search_quora(query: str, top_k: int = 10):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8006/search",
            json={"query": query, "top_k": top_k}
        )
        return response.json()
```

## Monitoring and Logging

### Health Checks
```bash
# Check service health
curl http://localhost:8004/health
curl http://localhost:8006/health

# Get detailed status
curl http://localhost:8006/status
```

### Logs
Services log to stdout. For production, redirect to files:
```bash
python services/data/quora_loader_service.py > quora_loader.log 2>&1 &
python services/representation/tfidf_quora_service.py > tfidf_quora.log 2>&1 &
```

## Scaling Considerations

### Horizontal Scaling
- Run multiple TF-IDF service instances on different ports
- Use a load balancer to distribute requests
- Share model files via network storage

### Vertical Scaling
- Increase RAM for larger vocabularies
- Use SSD storage for faster model loading
- Optimize TF-IDF parameters for your hardware

## Next Steps

1. **Experiment with Parameters**: Try different TF-IDF configurations
2. **Add Text Preprocessing**: Implement custom text cleaning
3. **Evaluation Metrics**: Add relevance scoring and evaluation
4. **Hybrid Search**: Combine with other retrieval methods
5. **Real-time Updates**: Implement incremental indexing

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs
3. Verify dataset format and quality
4. Test with smaller datasets first
