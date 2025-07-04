# TF-IDF Microservices Architecture

A scalable microservices-based implementation of TF-IDF with advanced text processing, inverted index optimization, and MAP evaluation.

## Architecture Overview

The system consists of 4 independent microservices that communicate via HTTP APIs:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Text Cleaning  │    │ TF-IDF Vectorizer│    │Enhanced TF-IDF  │    │ MAP Evaluation  │
│   Service       │    │    Service       │    │   Service       │    │   Service       │
│   Port: 8001    │    │   Port: 8002     │    │   Port: 8003    │    │   Port: 8004    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Details

#### 1. Text Cleaning Service (Port 8001)
- **Purpose**: Advanced text cleaning with spell checking, lemmatization, stemming
- **Features**:
  - Multiple cleaning methods (basic, advanced, tfidf, embedding, query)
  - Conservative spell checking to preserve MAP performance
  - Batch processing support
  - Unicode normalization and contraction expansion

#### 2. TF-IDF Vectorizer Service (Port 8002)
- **Purpose**: Basic TF-IDF vectorization
- **Features**:
  - Sklearn TfidfVectorizer with enhanced tokenizer
  - Training and vectorization endpoints
  - Calls Text Cleaning Service for preprocessing

#### 3. Enhanced TF-IDF Service (Port 8003)
- **Purpose**: Complete TF-IDF system with inverted index and fusion search
- **Features**:
  - Enhanced TF-IDF training with optimized parameters
  - Inverted index construction and management
  - Fusion search (TF-IDF + inverted index scores)
  - Document metadata storage
  - Search caching

#### 4. MAP Evaluation Service (Port 8004)
- **Purpose**: Performance evaluation using MAP and IR metrics
- **Features**:
  - ANTIQUE dataset evaluation
  - MAP, Precision@K, Recall@K calculation
  - Performance recommendations
  - Comprehensive evaluation reports

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn requests scikit-learn nltk numpy ir-datasets textblob
```

### 2. Start All Services

```bash
python start_microservices.py
```

This will:
- Start all 4 services in the correct order
- Wait for each service to be healthy before starting the next
- Test inter-service communication
- Keep all services running

### 3. Run the Demo

```bash
python test_microservices_pipeline.py
```

This demonstrates the complete pipeline with:
- Text cleaning
- TF-IDF training
- Enhanced search
- MAP evaluation

## API Endpoints

### Text Cleaning Service (8001)

```bash
# Basic cleaning
POST /clean
{
  "text": "I'm looking for beautifull antique vases!",
  "method": "tfidf"
}

# TF-IDF optimized cleaning
POST /clean/tfidf
{
  "text": "Your text here"
}

# Query optimized cleaning
POST /clean/query
{
  "query": "Your search query"
}

# Batch cleaning
POST /clean/batch
{
  "texts": ["text1", "text2", "text3"],
  "method": "tfidf"
}
```

### Enhanced TF-IDF Service (8003)

```bash
# Train model
POST /train
{
  "documents": ["doc1 text", "doc2 text"],
  "doc_ids": ["doc1", "doc2"],
  "build_inverted_index": true
}

# Search documents
POST /search
{
  "query": "antique furniture",
  "top_k": 10,
  "method": "enhanced_inverted",
  "fusion_alpha": 0.7
}

# Get training statistics
GET /statistics
```

### MAP Evaluation Service (8004)

```bash
# Evaluate TF-IDF service
POST /evaluate
{
  "dataset_name": "antique",
  "max_queries": 100,
  "k_eval": 10
}

# Calculate MAP from results
POST /calculate_map
{
  "search_results": {
    "query1": ["doc1", "doc2", "doc3"],
    "query2": ["doc4", "doc5", "doc6"]
  },
  "dataset_name": "antique"
}
```

## Advanced Features

### Enhanced Text Processing

The text cleaning includes:

1. **Spell Checking**: Conservative approach using TextBlob
2. **Lemmatization**: WordNet lemmatizer with POS tagging
3. **Stemming**: Porter stemmer for vocabulary reduction
4. **Normalization**: Unicode normalization, contraction expansion
5. **Stopword Removal**: Enhanced stopword lists (technical + domain-specific)

### TF-IDF Optimization

Optimized parameters for high MAP performance:

```python
{
    'max_features': 100000,    # Large vocabulary
    'min_df': 2,               # Remove very rare terms
    'max_df': 0.85,            # Remove very common terms
    'ngram_range': (1, 3),     # Include trigrams
    'sublinear_tf': True,      # Log scaling
    'norm': 'l2',              # L2 normalization
    'smooth_idf': True,        # Smooth IDF weights
}
```

### Inverted Index & Fusion Search

The enhanced service builds an optimized inverted index with:
- Term frequency and document frequency statistics
- Sorted postings by TF-IDF score
- Fusion scoring: `α * tfidf_similarity + (1-α) * index_score`

### MAP Evaluation

Comprehensive evaluation including:
- Mean Average Precision (MAP)
- Precision@K and Recall@K
- Query performance analysis
- Performance recommendations

## Example Workflow

1. **Start Services**:
   ```bash
   python start_microservices.py
   ```

2. **Train Model**:
   ```bash
   curl -X POST http://localhost:8003/train \
     -H "Content-Type: application/json" \
     -d '{
       "documents": ["Beautiful antique furniture", "Vintage collectibles"],
       "doc_ids": ["doc1", "doc2"],
       "build_inverted_index": true
     }'
   ```

3. **Search Documents**:
   ```bash
   curl -X POST http://localhost:8003/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "antique furniture",
       "top_k": 5,
       "method": "enhanced_inverted"
     }'
   ```

4. **Evaluate Performance**:
   ```bash
   curl -X POST http://localhost:8004/evaluate \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_name": "antique",
       "max_queries": 50,
       "k_eval": 10
     }'
   ```

## Benefits of Microservices Architecture

### ✅ **Modularity**
- Each service has a single responsibility
- Independent development and deployment
- Easy to test and maintain

### ✅ **Scalability**
- Scale individual services based on load
- Horizontal scaling support
- Load balancing capabilities

### ✅ **Technology Flexibility**
- Different services can use different technologies
- Easy to upgrade or replace individual components
- Language-agnostic communication via HTTP

### ✅ **Fault Isolation**
- Failure in one service doesn't affect others
- Graceful degradation
- Better error handling and recovery

### ✅ **Development Efficiency**
- Teams can work on different services independently
- Faster development cycles
- Better code organization

## Production Considerations

### Service Discovery
- Implement service registry (e.g., Consul, etcd)
- Use container orchestration (Kubernetes, Docker Swarm)
- Health checks and monitoring

### Load Balancing
- Use reverse proxy (nginx, HAProxy)
- Distribute load across service instances
- Session affinity for stateful operations

### Security
- API authentication and authorization
- Network security (VPN, firewalls)
- Input validation and sanitization

### Monitoring
- Service health monitoring
- Performance metrics collection
- Distributed tracing (Jaeger, Zipkin)
- Centralized logging (ELK stack)

### Data Management
- Shared databases vs. service-specific databases
- Data consistency patterns
- Backup and disaster recovery

## Configuration

### Environment Variables

```bash
# Service URLs (for inter-service communication)
TEXT_CLEANING_URL=http://localhost:8001
TFIDF_VECTORIZER_URL=http://localhost:8002
ENHANCED_TFIDF_URL=http://localhost:8003
MAP_EVALUATION_URL=http://localhost:8004

# Service-specific configurations
ENABLE_SPELL_CHECK=true
ENABLE_LEMMATIZATION=true
ENABLE_STEMMING=true
FUSION_ALPHA=0.7
```

### Docker Deployment

```dockerfile
# Example Dockerfile for text cleaning service
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY services/text_cleaning_service.py .
COPY services/shared/ ./services/shared/

EXPOSE 8001
CMD ["python", "text_cleaning_service.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  text-cleaning:
    build: .
    ports:
      - "8001:8001"
    
  enhanced-tfidf:
    build: .
    ports:
      - "8003:8003"
    depends_on:
      - text-cleaning
    environment:
      - TEXT_CLEANING_URL=http://text-cleaning:8001
```

## Troubleshooting

### Common Issues

1. **Service not starting**: Check port availability and dependencies
2. **Import errors**: Verify Python path and module structure
3. **Service communication failure**: Check network connectivity and URLs
4. **Performance issues**: Monitor resource usage and consider scaling

### Debugging

```bash
# Check service health
curl http://localhost:8001/health
curl http://localhost:8003/health

# Get service information
curl http://localhost:8001/info
curl http://localhost:8003/info

# Test text cleaning
curl -X POST http://localhost:8001/clean/tfidf \
  -H "Content-Type: application/json" \
  -d '{"text": "test text"}'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes to individual services
4. Test with the demo script
5. Submit a pull request

## License

This project is licensed under the MIT License.
