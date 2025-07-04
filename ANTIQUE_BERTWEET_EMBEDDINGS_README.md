# 🦜 Antique Dataset BERTweet Embeddings Pipeline

This comprehensive pipeline provides a complete solution for generating high-quality BERTweet embeddings from the Antique dataset. The pipeline is split into two main components:

1. **Local Text Cleaning Service** - Preprocesses the Antique dataset and prepares it for embedding generation
2. **Google Colab Notebook** - Generates BERTweet embeddings using GPU acceleration

## 📁 File Structure

```
backend/
├── services/shared/
│   ├── enhanced_text_cleaning_service.py      # Base text cleaning service
│   └── antique_text_cleaning_service.py       # Antique-specific cleaning service
├── services/representation/
│   └── bertweet_embedding_service.py           # BERTweet embedding management service
├── run_antique_text_cleaning.py               # Script to run text cleaning
├── antique_bertweet_embeddings_colab.ipynb    # Google Colab notebook
└── ANTIQUE_BERTWEET_EMBEDDINGS_README.md      # This README
```

## 🚀 Quick Start Guide

### Step 1: Clean the Antique Dataset (Local)

First, ensure you have the required dependencies installed:

```bash
pip install aiosqlite asyncio nltk textblob scikit-learn numpy pandas
```

Run the text cleaning process:

```bash
python run_antique_text_cleaning.py
```

This will:
- ✅ Load the antique dataset from your SQLite database (403,666 documents)
- 🧹 Apply advanced text cleaning including:
  - HTML tag removal and decoding
  - Unicode normalization
  - Contraction expansion
  - Spell checking and correction
  - Stemming and lemmatization
  - Stopword removal
  - Special pattern normalization
- 💾 Save cleaned text to database
- 📤 Export cleaned data as JSON for Colab upload

### Step 2: Generate BERTweet Embeddings (Google Colab)

1. **Upload the Colab notebook**: Upload `antique_bertweet_embeddings_colab.ipynb` to Google Colab

2. **Enable GPU**: Go to Runtime → Change runtime type → GPU (T4 or better recommended)

3. **Upload cleaned data**: The notebook will prompt you to upload `antique_cleaned_for_embeddings.json`

4. **Run the notebook**: Execute all cells to generate embeddings

The Colab notebook will:
- 📦 Install required packages (transformers, torch, etc.)
- 🤖 Load the BERTweet model (`vinai/bertweet-base`)
- 🔄 Process cleaned text in batches
- 🚀 Generate high-quality embeddings (768-dimensional)
- 💾 Save embeddings in multiple formats (NumPy, Pickle, CSV)
- 📥 Download all generated files

### Step 3: Load Embeddings Back to Local System

After downloading the embedding files from Colab, use the BERTweet embedding service:

```python
import asyncio
from services.representation.bertweet_embedding_service import BERTweetEmbeddingService

async def load_embeddings():
    # Create service
    service = BERTweetEmbeddingService()
    
    # Load embeddings from Colab files
    success = service.load_embeddings_from_numpy(
        "path/to/antique_bertweet_embeddings.npy",
        "path/to/antique_bertweet_doc_ids.npy", 
        "path/to/antique_bertweet_metadata.json"
    )
    
    if success:
        # Build FAISS index for fast similarity search
        service.build_faiss_index("IVF")
        
        # Save to database for persistence
        await service.save_embeddings_to_database()
        
        # Test similarity search
        results = service.search_similar_documents(query_embedding, k=10)
        
        print(f"✅ Loaded {len(service.doc_ids)} embeddings successfully!")

# Run the loading process
asyncio.run(load_embeddings())
```

## 🔧 Components Deep Dive

### 1. Enhanced Text Cleaning Service

Located in `services/shared/enhanced_text_cleaning_service.py`, this service provides:

- **Multi-level cleaning**: Basic, TF-IDF optimized, and embedding optimized
- **Spell checking**: Using TextBlob with intelligent correction
- **Lemmatization & Stemming**: POS-tagged lemmatization followed by stemming
- **Domain-specific stopwords**: Includes technical and antique-domain specific terms
- **Pattern normalization**: URLs, emails, dates, prices, measurements
- **Unicode handling**: Proper normalization and ASCII conversion

**Key Methods:**
- `preprocess_for_tfidf()`: Aggressive cleaning for TF-IDF
- `preprocess_for_embedding()`: Structure-preserving cleaning for embeddings
- `batch_preprocess()`: Efficient batch processing

### 2. Antique Text Cleaning Service

Located in `services/shared/antique_text_cleaning_service.py`, this service:

- **Database integration**: Direct SQLite database access
- **Batch processing**: Handles large datasets efficiently
- **Progress tracking**: Real-time progress monitoring
- **Export functionality**: Prepares data for Colab upload
- **Statistics**: Detailed cleaning statistics and metrics

**Key Features:**
- Processes documents in configurable batches (default: 1000)
- Creates separate cleaned versions for TF-IDF and embedding use
- Exports up to 50,000 documents for Colab (memory constraints)
- Tracks cleaning progress and provides detailed statistics

### 3. BERTweet Embedding Service

Located in `services/representation/bertweet_embedding_service.py`, this service:

- **Multiple loading formats**: NumPy files or Pickle
- **FAISS integration**: Fast similarity search using FAISS indices
- **Database persistence**: Stores embeddings in SQLite
- **Similarity search**: Efficient document similarity computation
- **Index management**: Save/load FAISS indices to disk

**Supported FAISS Index Types:**
- **Flat**: Exact search (slower, most accurate)
- **IVF**: Inverted file index (balanced speed/accuracy)
- **HNSW**: Hierarchical navigable small world (fastest)

### 4. Google Colab Notebook

The notebook `antique_bertweet_embeddings_colab.ipynb` provides:

- **GPU acceleration**: Automatic GPU detection and usage
- **Batch processing**: Memory-efficient batch processing
- **Progress tracking**: Real-time progress with tqdm
- **Quality validation**: Similarity testing and statistics
- **Multiple output formats**: NumPy, Pickle, CSV for different use cases
- **Memory management**: Automatic cleanup and garbage collection

## 📊 Expected Performance

### Dataset Statistics
- **Total documents**: 403,666 (full Antique dataset)
- **Average processing time**: ~0.001-0.01 seconds per document
- **Text reduction**: ~30-50% character reduction after cleaning
- **Token reduction**: ~40-60% token reduction after cleaning

### Embedding Generation (Colab)
- **Model**: BERTweet (`vinai/bertweet-base`)
- **Embedding dimension**: 768
- **Processing speed**: 
  - GPU (T4): ~100-200 documents/second
  - CPU: ~10-30 documents/second
- **Memory requirements**: 
  - 50K documents: ~1.5GB GPU memory
  - Full dataset: Recommend processing in chunks

### File Sizes (50K documents)
- **Embeddings (.npy)**: ~150MB
- **Complete dataset (.pkl)**: ~200MB
- **Document mapping (.csv)**: ~10MB
- **Metadata (.json)**: <1MB

## 🛠️ Configuration Options

### Text Cleaning Configuration

```python
# Create cleaning service with custom settings
service = AntiqueTextCleaningService(
    db_path="custom/path/to/database.db",
    enable_spell_check=True  # Enable/disable spell checking
)

# Process with custom parameters
await service.process_all_antique_documents(
    batch_size=500,          # Smaller batches for memory-constrained systems
    max_documents=10000      # Limit processing for testing
)
```

### Embedding Service Configuration

```python
# Create embedding service with custom settings
service = BERTweetEmbeddingService(db_path="custom/database.db")

# Build different index types
service.build_faiss_index("Flat")   # Exact search
service.build_faiss_index("IVF")    # Balanced
service.build_faiss_index("HNSW")   # Fast approximate
```

### Colab Notebook Configuration

```python
# Adjust batch size based on available GPU memory
batch_size = 32 if torch.cuda.is_available() else 8

# Modify sequence length for different models
max_length = 256  # Standard for BERTweet

# Control embedding generation
embeddings = generate_embeddings_batch(
    texts, 
    batch_size=batch_size, 
    max_length=max_length
)
```

## 🔍 Usage Examples

### Basic Usage

```python
# 1. Clean the dataset
python run_antique_text_cleaning.py

# 2. Upload to Colab and run notebook
# (Upload antique_cleaned_for_embeddings.json)

# 3. Load embeddings back
service = BERTweetEmbeddingService()
service.load_embeddings_from_pickle("antique_bertweet_complete.pkl")
service.build_faiss_index("IVF")
```

### Advanced Usage

```python
# Custom text cleaning with specific parameters
service = AntiqueTextCleaningService(enable_spell_check=False)
await service.process_all_antique_documents(
    batch_size=2000,
    max_documents=100000
)

# Export specific subset
export_path = await service.export_cleaned_data_for_colab(
    output_file="antique_subset.json",
    max_documents=25000
)

# Load and search embeddings
embedding_service = BERTweetEmbeddingService()
embedding_service.load_embeddings_from_numpy(
    "embeddings.npy", "doc_ids.npy", "metadata.json"
)
embedding_service.build_faiss_index("HNSW")

# Find similar documents
doc_embedding = embedding_service.get_document_embedding("doc_123")
similar_docs = embedding_service.search_similar_documents(
    doc_embedding, k=20, threshold=0.7
)
```

### Query Processing

```python
# Process a search query and find similar documents
from services.shared.enhanced_text_cleaning_service import EnhancedTextCleaningService

# Clean query using the same process
cleaner = EnhancedTextCleaningService()
query = "antique vase from 18th century"
cleaned_query = cleaner.preprocess_for_embedding(query)

# Generate query embedding (would need BERTweet model)
# query_embedding = model.encode(cleaned_query)

# Find similar documents
results = embedding_service.search_similar_documents(
    query_embedding, k=10, threshold=0.5
)

for doc_id, similarity in results:
    print(f"Document {doc_id}: {similarity:.4f}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues in Colab**
   - Reduce batch size: `batch_size = 16` or `batch_size = 8`
   - Process fewer documents: Upload smaller dataset
   - Use CPU if GPU memory insufficient

2. **Database Connection Issues**
   - Check database path in configuration
   - Ensure database file exists and is accessible
   - Verify SQLite database integrity

3. **Text Cleaning Performance**
   - Disable spell checking for faster processing: `enable_spell_check=False`
   - Increase batch size for faster processing: `batch_size=2000`
   - Monitor memory usage during processing

4. **FAISS Index Issues**
   - Ensure embeddings are loaded before building index
   - For large datasets, use IVF or HNSW instead of Flat
   - Check embedding dimension consistency

### Performance Optimization

1. **Text Cleaning**
   ```python
   # Disable spell checking for speed
   service = AntiqueTextCleaningService(enable_spell_check=False)
   
   # Increase batch size
   await service.process_all_antique_documents(batch_size=5000)
   ```

2. **Embedding Generation**
   ```python
   # Use larger batch sizes with sufficient GPU memory
   batch_size = 64  # Adjust based on GPU memory
   
   # Reduce sequence length if needed
   max_length = 128  # Shorter sequences process faster
   ```

3. **FAISS Search**
   ```python
   # Use approximate indices for large datasets
   service.build_faiss_index("IVF")  # or "HNSW"
   
   # Adjust search parameters
   service.faiss_index.nprobe = 20  # Higher for better accuracy
   ```

## 📈 Evaluation and Metrics

### Quality Metrics

The pipeline includes several quality checks:

1. **Text Cleaning Statistics**:
   - Character reduction ratio
   - Token reduction ratio
   - Spell correction count
   - Processing time per document

2. **Embedding Quality**:
   - Embedding norms distribution
   - Pairwise similarity statistics
   - Dimensionality verification
   - Generation time metrics

3. **Search Performance**:
   - Query response time
   - Similarity score distribution
   - Index build time
   - Memory usage

### Example Quality Check

```python
# Get cleaning statistics
stats = await cleaning_service.get_cleaning_statistics()
print(f"Cleaned {stats['cleaned_documents']} documents")
print(f"Average token reduction: {stats['avg_token_reduction_ratio']:.2%}")

# Get embedding statistics
embedding_stats = embedding_service.get_service_statistics()
print(f"Loaded {embedding_stats['total_documents']} embeddings")
print(f"Index type: {embedding_stats['index_type']}")

# Test search performance
import time
start_time = time.time()
results = embedding_service.search_similar_documents(test_embedding, k=100)
search_time = time.time() - start_time
print(f"Search completed in {search_time:.4f} seconds")
```

## 🎯 Next Steps

After completing the embedding generation:

1. **Integration**: Integrate with your search engine's query processing pipeline
2. **Evaluation**: Test retrieval performance using standard IR metrics (MAP, NDCG, etc.)
3. **Optimization**: Fine-tune search parameters based on evaluation results
4. **Scaling**: Consider distributed processing for larger datasets
5. **Monitoring**: Set up monitoring for embedding quality and search performance

## 📝 License and Attribution

This pipeline uses several open-source models and libraries:

- **BERTweet**: `vinai/bertweet-base` - VinAI Research
- **FAISS**: Facebook AI Similarity Search
- **Transformers**: Hugging Face Transformers library
- **NLTK**: Natural Language Toolkit

Please ensure appropriate attribution when using this pipeline in research or production.

---

**Need Help?** Check the troubleshooting section above or review the detailed code documentation in each service file.
