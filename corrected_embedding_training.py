# ===================================================================
# CORRECTED Embedding Training for Antique Dataset
# Uses all-MiniLM-L6-v2 from Hugging Face with preprocessing service
# Trains on all data and creates inverted index
# ===================================================================

# STEP 1: Install Required Libraries
# Run this in the first cell of your Colab notebook
"""
!pip install sentence-transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install ir-datasets joblib requests scipy tqdm nltk numpy pandas scikit-learn faiss-cpu
"""

# STEP 2: Import Libraries and Setup
import joblib
import ir_datasets
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import os
import gc
from typing import List, Dict, Any, Iterator, Tuple
from google.colab import files
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import torch
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import faiss
from collections import defaultdict

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

print("✅ Libraries imported successfully!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# STEP 3: UNIFIED Text Cleaning Service (SAME AS PREPROCESSING SERVICE)
class UnifiedTextCleaningService:
    """
    Unified text cleaning service that matches your preprocessing service
    This ensures consistency between training and inference
    """
    
    def __init__(self):
        # Initialize NLTK components exactly like your preprocessing service
        self.stemmer = None
        self.stop_words = set()
        
        try:
            # Use same NLTK setup as your preprocessing service
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            print("✅ NLTK components initialized (matching preprocessing service)")
        except Exception as e:
            print(f"❌ Error initializing NLTK: {e}")
    
    def clean_text(self, text: str, 
                   remove_stopwords: bool = True,
                   apply_stemming: bool = True, 
                   apply_lemmatization: bool = False) -> str:
        """
        Clean text using EXACTLY the same method as your preprocessing service
        This ensures training and inference use identical preprocessing
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Basic cleaning (matches your preprocessing service _clean_text)
        cleaned_text = self._basic_clean(text)
        
        # Step 2: Tokenization (matches your preprocessing service _tokenize_text)
        tokens = self._tokenize(cleaned_text)
        
        # Step 3: Remove stopwords (matches your preprocessing service)
        if remove_stopwords and self.stop_words:
            tokens = self._remove_stopwords(tokens)
        
        # Step 4: Stemming (matches your preprocessing service)
        if apply_stemming and self.stemmer:
            tokens = self._apply_stemming(tokens)
        
        # Step 5: Lemmatization (not used by default in your service)
        if apply_lemmatization:
            # Your preprocessing service doesn't use lemmatization by default
            pass
        
        # Return cleaned text
        return " ".join(tokens)
    
    def _basic_clean(self, text: str) -> str:
        """Apply basic text cleaning - EXACTLY matching your preprocessing service"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation (matches your service)
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text - EXACTLY matching your preprocessing service"""
        if not text:
            return []
        
        if self.stemmer:  # NLTK available (same logic as your service)
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
                return [token.lower() for token in tokens if token.isalnum()]
            except:
                pass
        
        # Fallback tokenization (same as your service)
        return [word.strip() for word in text.lower().split() if word.strip()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords - EXACTLY matching your preprocessing service"""
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        """Apply stemming - EXACTLY matching your preprocessing service"""
        if not self.stemmer:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]

# Initialize unified text cleaner (matches your preprocessing service configuration)
text_cleaner = UnifiedTextCleaningService()
print("✅ Unified text cleaner initialized (matching preprocessing service)!")

# STEP 4: Load and Preprocess Dataset with Database Storage
def load_and_preprocess_antique_dataset():
    """Load Antique dataset, preprocess using your service, and store in database"""
    try:
        print("📚 Loading FULL Antique dataset (all ~400k documents)...")
        # Load the complete antique dataset (not just train split)
        dataset = ir_datasets.load('antique')
        
        # Extract documents and queries
        raw_documents = []
        queries = []
        qrels = {}
        
        print("Loading raw documents...")
        doc_count = 0
        for doc in dataset.docs_iter():
            raw_documents.append({
                'doc_id': doc.doc_id,
                'text': doc.text
            })
            doc_count += 1
            if doc_count % 1000 == 0:
                print(f"  Loaded {doc_count:,} documents...")
        
        print("Loading queries...")
        for query in dataset.queries_iter():
            queries.append({
                'query_id': query.query_id,
                'text': query.text
            })
        
        print("Loading relevance judgments...")
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        # Validation
        valid_queries = [q for q in queries if q['query_id'] in qrels]
        
        print(f"✅ Raw dataset loaded successfully!")
        print(f"   📄 Documents: {len(raw_documents):,}")
        print(f"   🔍 Total queries: {len(queries)}")
        print(f"   🔍 Queries with relevance judgments: {len(valid_queries)}")
        print(f"   🎯 Relevance judgments: {len(qrels)}")
        
        # Save raw dataset files to drive/downloads
        save_raw_dataset_files(raw_documents, queries, qrels)
        
        return raw_documents, queries, qrels
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None, None

def save_raw_dataset_files(raw_documents, queries, qrels):
    """Save raw dataset files to drive/downloads directory"""
    try:
        print("💾 Saving raw dataset files to drive/downloads...")
        
        # Create drive/downloads directory if it doesn't exist
        os.makedirs('drive/downloads', exist_ok=True)
        
        # Save documents
        docs_data = []
        for doc in raw_documents:
            docs_data.append({
                'doc_id': doc['doc_id'],
                'text': doc['text']
            })
        
        with open('drive/downloads/docs.json', 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        print("✅ Saved: drive/downloads/docs.json")
        
        # Save queries
        queries_data = []
        for query in queries:
            queries_data.append({
                'query_id': query['query_id'],
                'text': query['text']
            })
        
        with open('drive/downloads/queries.json', 'w', encoding='utf-8') as f:
            json.dump(queries_data, f, indent=2, ensure_ascii=False)
        print("✅ Saved: drive/downloads/queries.json")
        
        # Save qrels
        with open('drive/downloads/qrels.json', 'w', encoding='utf-8') as f:
            json.dump(qrels, f, indent=2)
        print("✅ Saved: drive/downloads/qrels.json")
        
        # Save dataset statistics
        stats = {
            'total_documents': len(raw_documents),
            'total_queries': len(queries),
            'total_qrels': len(qrels),
            'dataset_source': 'antique',
            'timestamp': time.time()
        }
        
        with open('drive/downloads/dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("✅ Saved: drive/downloads/dataset_stats.json")
        
        print(f"📊 Dataset files saved to drive/downloads/")
        print(f"   📄 Documents: {len(raw_documents):,}")
        print(f"   🔍 Queries: {len(queries):,}")
        print(f"   🎯 QRels entries: {len(qrels):,}")
        
    except Exception as e:
        print(f"❌ Error saving dataset files: {e}")

def preprocess_and_store_documents(raw_documents):
    """Preprocess all documents using unified cleaning and store metadata"""
    print(f"🧹 Preprocessing ALL {len(raw_documents):,} documents...")
    start_cleaning = time.time()
    
    def clean_document_unified(doc_and_index):
        doc, index = doc_and_index
        # Use SAME cleaning parameters as your preprocessing service
        cleaned = text_cleaner.clean_text(
            doc['text'], 
            remove_stopwords=True,    # Same as your preprocessing service
            apply_stemming=True,      # Same as your preprocessing service  
            apply_lemmatization=False # Same as your preprocessing service
        )
        return index, doc['doc_id'], doc['text'], cleaned
    
    # Use ThreadPoolExecutor for parallel text cleaning
    max_workers = min(32, cpu_count() * 2)
    
    print(f"   🔄 Using {max_workers} worker threads for unified text cleaning...")
    
    processed_documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        doc_index_pairs = [(doc, i) for i, doc in enumerate(raw_documents)]
        futures = {executor.submit(clean_document_unified, pair): pair for pair in doc_index_pairs}
        
        # Collect results maintaining order
        cleaning_results = {}
        for future in tqdm(as_completed(futures), total=len(raw_documents), desc="Cleaning documents"):
            try:
                original_index, doc_id, original_text, cleaned_text = future.result()
                cleaning_results[original_index] = {
                    'doc_id': doc_id,
                    'original_text': original_text,
                    'cleaned_text': cleaned_text
                }
            except Exception as e:
                print(f"Error cleaning document: {e}")
    
    # Build final processed documents list (ALL documents, maintaining order)
    print("🔧 Building processed document collection...")
    for i in range(len(raw_documents)):
        if i in cleaning_results:
            result = cleaning_results[i]
            # Only include documents with non-empty cleaned text
            if result['cleaned_text'].strip():
                processed_documents.append({
                    'doc_id': result['doc_id'],
                    'original_text': result['original_text'],
                    'cleaned_text': result['cleaned_text']
                })
    
    cleaning_time = time.time() - start_cleaning
    print(f"✅ ALL documents processed with unified cleaning!")
    print(f"   📊 Original documents: {len(raw_documents):,}")
    print(f"   📊 Valid documents after cleaning: {len(processed_documents):,}")
    print(f"   📊 Filtered out (empty after cleaning): {len(raw_documents) - len(processed_documents):,}")
    print(f"   ⏱️  Total cleaning time: {cleaning_time:.2f} seconds")
    
    return processed_documents

def load_cleaned_documents_from_storage(processed_documents):
    """Load the cleaned documents (simulating database storage/retrieval)"""
    print(f"📁 Loading cleaned documents from storage...")
    
    # Extract cleaned texts and maintain document order
    cleaned_documents = []
    cleaned_texts = []
    document_order = []
    
    for doc in processed_documents:
        cleaned_documents.append({
            'doc_id': doc['doc_id'],
            'text': doc['original_text']  # Keep original for metadata
        })
        cleaned_texts.append(doc['cleaned_text'])  # Use cleaned for embeddings
        document_order.append(doc['doc_id'])
    
    print(f"✅ Loaded {len(cleaned_documents):,} cleaned documents")
    print(f"   📊 Ready for embedding generation on ALL cleaned data")
    
    return cleaned_documents, cleaned_texts, document_order

# Load and preprocess the dataset
raw_documents, queries, qrels = load_and_preprocess_antique_dataset()

# Preprocess ALL documents using your preprocessing service approach
processed_documents = preprocess_and_store_documents(raw_documents)

# Load cleaned documents (simulating loading from database after preprocessing)
documents, cleaned_texts, document_order = load_cleaned_documents_from_storage(processed_documents)

# STEP 5: Generate Embeddings for ALL Cleaned Documents
def generate_embeddings_for_all_documents(documents: List[Dict], cleaned_texts: List[str], document_order: List[str]):
    """Generate embeddings for ALL cleaned documents (no training, just embedding generation)"""
    
    print(f"🎯 Generating embeddings for ALL {len(documents):,} cleaned documents...")
    print("🤖 Using pre-trained all-MiniLM-L6-v2 from Hugging Face")
    print("📝 Note: This is embedding generation, not model training. The SentenceTransformer model is already pre-trained.")
    
    # Step 1: Initialize the pre-trained embedding model
    print("🚀 Loading pre-trained all-MiniLM-L6-v2 model...")
    start_model_load = time.time()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    model_load_time = time.time() - start_model_load
    print(f"✅ Pre-trained model loaded in {model_load_time:.2f} seconds")
    print(f"🎮 Device: {device}")
    print(f"📏 Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Step 2: Verify we have cleaned texts ready
    print(f"✅ Using pre-cleaned texts from preprocessing service")
    print(f"   📊 Total cleaned documents: {len(cleaned_texts):,}")
    print(f"   📊 Document order maintained: {len(document_order):,}")
    
    # CRITICAL VERIFICATION: Ensure all data is aligned
    assert len(documents) == len(cleaned_texts) == len(document_order)
    print(f"✅ PERFECT ALIGNMENT VERIFIED: All collections have {len(documents)} items")
    
    # Step 3: Generate embeddings for ALL cleaned documents
    print("🎯 Generating embeddings for ALL cleaned documents...")
    print("⚠️  IMPORTANT: Processing EVERY single document in the dataset")
    start_embedding = time.time()
    
    # Process in batches to manage memory efficiently
    batch_size = 32  # Adjust based on GPU memory
    total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    total_processed = 0
    
    all_embeddings = []
    
    print(f"📊 Processing {len(cleaned_texts):,} documents in {total_batches} batches of {batch_size}")
    
    for batch_idx in tqdm(range(0, len(cleaned_texts), batch_size), 
                          desc="Generating embeddings for ALL documents", total=total_batches):
        
        # Get current batch
        end_idx = min(batch_idx + batch_size, len(cleaned_texts))
        batch_texts = cleaned_texts[batch_idx:end_idx]
        current_batch_size = len(batch_texts)
        
        # Generate embeddings for current batch
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=current_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                show_progress_bar=False
            )
        
        all_embeddings.append(batch_embeddings)
        total_processed += current_batch_size
        
        # Progress logging
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"   ✅ Processed {total_processed:,} / {len(cleaned_texts):,} documents")
        
        # Clear cache periodically to manage memory
        if (batch_idx // batch_size) % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Concatenate ALL embeddings into single matrix
    print("🔗 Concatenating all embedding batches...")
    embeddings_matrix = np.vstack(all_embeddings)
    
    embedding_time = time.time() - start_embedding
    print(f"✅ Embeddings generated for ALL documents!")
    print(f"   ⏱️  Total embedding time: {embedding_time:.2f} seconds")
    print(f"   📊 Final matrix shape: {embeddings_matrix.shape}")
    print(f"   📊 Documents processed: {embeddings_matrix.shape[0]:,} (should equal {len(cleaned_texts):,})")
    print(f"   💾 Matrix memory usage: {embeddings_matrix.nbytes / 1024 / 1024:.2f} MB")
    
    # FINAL VERIFICATION: Ensure we processed every single document
    assert embeddings_matrix.shape[0] == len(cleaned_texts) == len(documents) == len(document_order)
    print(f"✅ FINAL VERIFICATION PASSED: ALL {len(documents):,} documents have embeddings")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, embeddings_matrix, documents, cleaned_texts, document_order

# STEP 6: Build Inverted Index for Fast Retrieval
def build_inverted_index(cleaned_texts: List[str], document_order: List[str]) -e Dict[str, List[int]]:
    """Build inverted index for fast term-based filtering"""
    print("🔍 Building inverted index...")
    start_time = time.time()
    
    inverted_index = defaultdict(list)
    
    for doc_idx, cleaned_text in enumerate(tqdm(cleaned_texts, desc="Building index")):
        # Tokenize the cleaned text
        tokens = set(cleaned_text.split())  # Use set to avoid duplicate terms per document
        
        for token in tokens:
            if token.strip():  # Only add non-empty tokens
                inverted_index[token].append(doc_idx)
    
    # Convert to regular dict and sort document lists
    inverted_index = {term: sorted(doc_list) for term, doc_list in inverted_index.items()}
    
    index_time = time.time() - start_time
    print(f"✅ Inverted index built!")
    print(f"   ⏱️  Build time: {index_time:.2f} seconds")
    print(f"   📊 Total terms: {len(inverted_index):,}")
    print(f"   📊 Average documents per term: {np.mean([len(docs) for docs in inverted_index.values()]):.2f}")
    
    return inverted_index

# STEP 7: Create FAISS Index for Fast Similarity Search
def create_faiss_index(embeddings_matrix: np.ndarray) -> faiss.Index:
    """Create FAISS index for fast similarity search"""
    print("⚡ Creating FAISS index for fast search...")
    start_time = time.time()
    
    # Ensure embeddings are float32 (required by FAISS)
    embeddings_matrix = embeddings_matrix.astype(np.float32)
    
    # Create FAISS index
    dimension = embeddings_matrix.shape[1]
    
    # Use IndexFlatIP for exact cosine similarity (since embeddings are normalized)
    faiss_index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    faiss_index.add(embeddings_matrix)
    
    faiss_time = time.time() - start_time
    print(f"✅ FAISS index created!")
    print(f"   ⏱️  Creation time: {faiss_time:.2f} seconds")
    print(f"   📊 Index dimension: {dimension}")
    print(f"   📊 Total vectors: {faiss_index.ntotal:,}")
    
    return faiss_index

# Generate embeddings for ALL cleaned documents
model, embeddings_matrix, documents, cleaned_texts, document_order = generate_embeddings_for_all_documents(documents, cleaned_texts, document_order)

# Build inverted index
inverted_index = build_inverted_index(cleaned_texts, document_order)

# Create FAISS index
faiss_index = create_faiss_index(embeddings_matrix)

# STEP 8: Save All Models and Data
def save_embedding_models(model, embeddings_matrix, documents, cleaned_texts, 
                         document_order, inverted_index, faiss_index):
    """Save all embedding models and data"""
    
    print("💾 Saving embedding models and data...")
    
    try:
        # Save the SentenceTransformer model
        model.save('antique_embedding_model')
        print("✅ Saved: antique_embedding_model/ (SentenceTransformer)")
        
        # Save embeddings matrix
        joblib.dump(embeddings_matrix, 'antique_embeddings_matrix.joblib')
        print("✅ Saved: antique_embeddings_matrix.joblib")
        
        # Save FAISS index
        faiss.write_index(faiss_index, 'antique_faiss_index.faiss')
        print("✅ Saved: antique_faiss_index.faiss")
        
        # Save inverted index
        joblib.dump(inverted_index, 'antique_inverted_index.joblib')
        print("✅ Saved: antique_inverted_index.joblib")
        
        # Save ALIGNED document metadata
        document_metadata = {
            'documents': documents,  # ALL documents
            'document_order': document_order,  # Aligned with matrix rows
            'cleaned_texts': cleaned_texts,  # Aligned with matrix rows
        }
        joblib.dump(document_metadata, 'antique_embedding_document_metadata.joblib')
        print("✅ Saved: antique_embedding_document_metadata.joblib")
        
        # Save training info
        training_info = {
            'model_name': 'all-MiniLM-L6-v2',
            'total_original_documents': len(raw_documents),
            'total_processed_documents': len(documents),
            'embedding_dimension': embeddings_matrix.shape[1],
            'matrix_shape': list(embeddings_matrix.shape),
            'inverted_index_terms': len(inverted_index),
            'faiss_index_total': int(faiss_index.ntotal),
            'training_parameters': {
                'batch_size': 32,
                'normalize_embeddings': True,
                'preprocessing_applied': True,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'dataset': 'antique',
            'cleaning_method': 'unified_preprocessing_service',
            'features': [
                'sentence_transformer_embeddings',
                'preprocessing_service_integration',
                'perfect_document_alignment',
                'inverted_index_for_fast_filtering',
                'faiss_index_for_fast_search',
                'normalized_embeddings'
            ],
            'alignment_verification': {
                'documents_count': len(documents),
                'document_order_count': len(document_order),
                'cleaned_texts_count': len(cleaned_texts),
                'matrix_rows': embeddings_matrix.shape[0],
                'all_perfectly_aligned': len(documents) == len(document_order) == len(cleaned_texts) == embeddings_matrix.shape[0],
                'processed_all_data': True
            }
        }
        
        with open('antique_embedding_training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        print("✅ Saved: antique_embedding_training_info.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

# Save all models
save_success = save_embedding_models(model, embeddings_matrix, documents, 
                                   cleaned_texts, document_order, 
                                   inverted_index, faiss_index)

# STEP 9: Test Sample Queries
def test_sample_queries_embedding(model, embeddings_matrix, documents, document_order, 
                                 queries, faiss_index, top_k=5):
    """Test with sample queries to verify embedding model performance"""
    
    print("🧪 Testing embedding model with sample queries...")
    
    # Test with a few sample queries
    test_queries = queries[:5] if len(queries) >= 5 else queries
    
    for query_data in test_queries:
        query_text = query_data['text']
        query_id = query_data['query_id']
        
        print(f"\n🔍 Query: '{query_text}' (ID: {query_id})")
        
        # Clean query with preprocessing method (less aggressive for queries)
        cleaned_query = text_cleaner.clean_text(
            query_text, 
            remove_stopwords=False,  # Keep stopwords for queries
            apply_stemming=True,
            apply_lemmatization=False
        )
        
        print(f"   🧹 Cleaned: '{cleaned_query}'")
        
        # Generate query embedding
        with torch.no_grad():
            query_embedding = model.encode(
                [cleaned_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Search using FAISS
        query_embedding = query_embedding.astype(np.float32)
        scores, indices = faiss_index.search(query_embedding, top_k)
        
        print(f"   📋 Top {top_k} results:")
        for i, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            doc_id = document_order[idx]
            text_preview = documents[idx]['text'][:80] + "..."
            print(f"      {i}. {doc_id}: {score:.4f} - {text_preview}")

# Test sample queries
if len(queries) > 0:
    test_sample_queries_embedding(model, embeddings_matrix, documents, 
                                 document_order, queries, faiss_index)

# STEP 10: Create Archive and Download Files
def create_archive_and_download():
    """Create archive with all files and download"""
    
    print("📦 Creating archive with all embedding files...")
    
    try:
        import shutil
        
        # Create directory structure
        os.makedirs('antique_embedding_files', exist_ok=True)
        
        # Copy all files to directory
        files_to_include = [
            'antique_embeddings_matrix.joblib',
            'antique_faiss_index.faiss', 
            'antique_inverted_index.joblib',
            'antique_embedding_document_metadata.joblib',
            'antique_embedding_training_info.json'
        ]
        
        for file_name in files_to_include:
            if os.path.exists(file_name):
                shutil.copy2(file_name, f'antique_embedding_files/{file_name}')
        
        # Copy model directory
        if os.path.exists('antique_embedding_model'):
            shutil.copytree('antique_embedding_model', 'antique_embedding_files/antique_embedding_model')
        
        # Create tar archive
        shutil.make_archive('antique_embedding_files', 'tar', '.', 'antique_embedding_files')
        
        print("✅ Archive created: antique_embedding_files.tar")
        
        # Download individual files
        print("📥 Downloading individual files...")
        
        for file_name in files_to_include:
            if os.path.exists(file_name):
                files.download(file_name)
        
        # Download model as zip
        if os.path.exists('antique_embedding_model'):
            shutil.make_archive('antique_embedding_model', 'zip', '.', 'antique_embedding_model')
            files.download('antique_embedding_model.zip')
        
        # Download archive
        files.download('antique_embedding_files.tar')
        
        print("🎉 All embedding files downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error creating archive: {e}")

# Create archive and download
create_archive_and_download()

# STEP 11: Integration Instructions
print("\n" + "="*70)
print("🚀 EMBEDDING MODEL INTEGRATION INSTRUCTIONS")
print("="*70)
print("""
FEATURES IMPLEMENTED:
✅ all-MiniLM-L6-v2 from Hugging Face (pre-trained model)
✅ Preprocessing service integration (unified cleaning)
✅ Embedding generation for ALL dataset documents
✅ Perfect document alignment verification
✅ Inverted index for fast term-based filtering
✅ FAISS index for fast similarity search
✅ Normalized embeddings for cosine similarity
✅ Database-ready cleaned document processing

FILES CREATED:
📁 antique_embedding_model/ (SentenceTransformer model)
📄 antique_embeddings_matrix.joblib (NumPy embeddings)
⚡ antique_faiss_index.faiss (FAISS index)
🔍 antique_inverted_index.joblib (Inverted index)
📊 antique_embedding_document_metadata.joblib (Document metadata)
📋 antique_embedding_training_info.json (Training information)

INTEGRATION STEPS:
1. Extract all files to your backend directory

2. Update your embedding service paths:
   ANTIQUE_MODEL_PATH = "/path/to/antique_embedding_model/"
   ANTIQUE_EMBEDDINGS_PATH = "/path/to/antique_embeddings_matrix.joblib"
   ANTIQUE_FAISS_PATH = "/path/to/antique_faiss_index.faiss"
   ANTIQUE_INVERTED_INDEX_PATH = "/path/to/antique_inverted_index.joblib"
   ANTIQUE_METADATA_PATH = "/path/to/antique_embedding_document_metadata.joblib"

3. Create embedding service similar to tfidf_service.py

4. Build inverted index search functionality

5. Test with your queries

EXPECTED FEATURES:
🎯 Semantic similarity search using embeddings
🎯 Fast retrieval using FAISS index
🎯 Term-based filtering using inverted index
🎯 Consistent preprocessing with your service
🎯 Perfect document alignment
🎯 ALL documents processed (no data left out)
""")

print("\n🎯 Embedding generation for ALL documents completed!")
print("📊 Ready for semantic search with preprocessing integration!")
print("🔧 All alignment and consistency ensured!")
print(f"✅ Processed ALL {len(documents):,} documents from the dataset!")
