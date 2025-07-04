# ===================================================================
# CORRECTED Embedding Training for Antique Dataset
# Uses BERTweet from Hugging Face with cleaned data from SQLite database
# Generates embeddings only - no indexing
# ===================================================================

# STEP 1: Install Required Libraries
# Run this in the first cell of your Colab notebook
"""
!pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install joblib requests scipy tqdm nltk numpy pandas scikit-learn
"""

# STEP 2: Import Libraries and Setup
import joblib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
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
import sqlite3
from datetime import datetime

print("✅ Libraries imported successfully!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# STEP 3: Upload and Load SQLite Database
def upload_and_load_database():
    """
    Upload SQLite database containing cleaned antique dataset
    """
    print("📁 Please upload your SQLite database file:")
    uploaded = files.upload()
    
    # Get the uploaded database filename
    db_file = list(uploaded.keys())[0]
    print(f"✅ Uploaded database: {db_file}")
    
    return db_file

def load_cleaned_antique_from_database(db_file):
    """
    Load cleaned antique documents from SQLite database
    """
    print(f"📀 Loading cleaned antique documents from database: {db_file}")
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Available tables: {[table[0] for table in tables]}")
        
        # Try to load from antique_cleaned_documents table first
        try:
            cursor.execute("""
                SELECT doc_id, cleaned_for_embedding, original_content 
                FROM antique_cleaned_documents 
                ORDER BY doc_id
            """)
            rows = cursor.fetchall()
            
            if rows:
                print(f"✅ Found {len(rows):,} documents in antique_cleaned_documents table")
                documents = []
                for doc_id, cleaned_text, original_text in rows:
                    documents.append({
                        'doc_id': str(doc_id),
                        'text': cleaned_text if cleaned_text else original_text,
                        'original_text': original_text
                    })
                return documents
            
        except sqlite3.Error as e:
            print(f"⚠️ antique_cleaned_documents table not found: {e}")
        
        # Fallback: try documents table
        try:
            cursor.execute("""
                SELECT doc_id, content, title 
                FROM documents 
                WHERE dataset_name = 'antique'
                ORDER BY doc_id
            """)
            rows = cursor.fetchall()
            
            if rows:
                print(f"✅ Found {len(rows):,} documents in documents table")
                documents = []
                for doc_id, content, title in rows:
                    # Combine title and content
                    full_text = ""
                    if title:
                        full_text += title + " "
                    if content:
                        full_text += content
                    
                    documents.append({
                        'doc_id': str(doc_id),
                        'text': full_text.strip(),
                        'original_text': full_text.strip()
                    })
                return documents
                
        except sqlite3.Error as e:
            print(f"❌ Error loading from documents table: {e}")
            
        # If no data found
        print("❌ No suitable data found in database")
        return []
        
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

# STEP 4: Load Dataset from Database
# Upload database and load cleaned documents
db_file = upload_and_load_database()
documents = load_cleaned_antique_from_database(db_file)

if not documents:
    print("❌ No documents loaded. Please check your database.")
    exit()

print(f"✅ Successfully loaded {len(documents):,} documents from database")

# STEP 5: Load BERTweet Model
def load_bertweet_model():
    """
    Load BERTweet model and tokenizer from Hugging Face
    """
    print("🤖 Loading BERTweet model...")
    
    model_name = "vinai/bertweet-base"
    
    try:
        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Load model
        print("   Loading model...")
        model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ BERTweet model loaded successfully!")
        print(f"   Model: {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Embedding dimension: {model.config.hidden_size}")
        print(f"   Max sequence length: {tokenizer.model_max_length}")
        print(f"   Device: {device}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error loading BERTweet model: {e}")
        return None, None, None

# Load BERTweet model
model, tokenizer, device = load_bertweet_model()

if model is None:
    print("❌ Failed to load BERTweet model. Exiting.")
    exit()

# STEP 6: Generate BERTweet Embeddings
def preprocess_text_for_bertweet(text, max_length=256):
    """
    Preprocess text specifically for BERTweet model.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning for BERTweet
    text = text.strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long (leave room for special tokens)
    words = text.split()
    if len(words) > max_length - 10:  # Reserve space for [CLS], [SEP], etc.
        text = ' '.join(words[:max_length - 10])
    
    return text

def generate_bertweet_embeddings(documents, model, tokenizer, device):
    """
    Generate BERTweet embeddings for all documents
    """
    print(f"🚀 Generating BERTweet embeddings for {len(documents):,} documents...")
    
    # Extract texts for embedding
    texts = [doc['text'] for doc in documents]
    doc_ids = [doc['doc_id'] for doc in documents]
    
    # Set batch size based on available memory
    batch_size = 32 if torch.cuda.is_available() else 8
    max_length = 256
    
    print(f"   Batch size: {batch_size}")
    print(f"   Max sequence length: {max_length}")
    print(f"   Device: {device}")
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", total=total_batches):
        batch_texts = texts[i:i + batch_size]
        
        # Preprocess texts
        processed_texts = [preprocess_text_for_bertweet(text, max_length) for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        # Clear GPU memory
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings)
    
    print(f"✅ BERTweet embeddings generated!")
    print(f"   Embeddings shape: {embeddings_matrix.shape}")
    print(f"   Embedding dimension: {embeddings_matrix.shape[1]}")
    print(f"   Memory usage: {embeddings_matrix.nbytes / 1024 / 1024:.2f} MB")
    
    return embeddings_matrix, doc_ids

# Generate embeddings
embeddings_matrix, doc_ids = generate_bertweet_embeddings(documents, model, tokenizer, device)

# STEP 7: Save BERTweet Model and Embeddings
def save_bertweet_embeddings(model, tokenizer, embeddings_matrix, doc_ids, documents):
    """
    Save BERTweet model and embeddings
    """
    print("💾 Saving BERTweet model and embeddings...")
    
    try:
        # Save the BERTweet model and tokenizer
        model.save_pretrained('antique_bertweet_model')
        tokenizer.save_pretrained('antique_bertweet_model')
        print("✅ Saved: antique_bertweet_model/ (BERTweet model and tokenizer)")
        
        # Save embeddings matrix
        np.save('antique_bertweet_embeddings.npy', embeddings_matrix)
        print("✅ Saved: antique_bertweet_embeddings.npy")
        
        # Save document IDs
        np.save('antique_bertweet_doc_ids.npy', np.array(doc_ids))
        print("✅ Saved: antique_bertweet_doc_ids.npy")
        
        # Save document metadata
        document_metadata = {
            'documents': documents,
            'doc_ids': doc_ids,
            'total_documents': len(documents),
            'embedding_dimension': embeddings_matrix.shape[1]
        }
        joblib.dump(document_metadata, 'antique_bertweet_document_metadata.joblib')
        print("✅ Saved: antique_bertweet_document_metadata.joblib")
        
        # Save metadata JSON
        metadata = {
            'model_name': 'vinai/bertweet-base',
            'embedding_dimension': int(embeddings_matrix.shape[1]),
            'total_documents': len(documents),
            'generation_timestamp': datetime.now().isoformat(),
            'device_used': str(device),
            'dataset': 'antique_cleaned_from_database',
            'features': [
                'bertweet_embeddings',
                'database_integration',
                'cleaned_data_processing'
            ]
        }
        
        with open('antique_bertweet_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Saved: antique_bertweet_metadata.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

# Save BERTweet model and embeddings
save_success = save_bertweet_embeddings(model, tokenizer, embeddings_matrix, doc_ids, documents)


# STEP 8: Quality Check
def test_embeddings_quality(embeddings_matrix, doc_ids, documents, sample_size=5):
    """
    Test embedding quality with similarity checks
    """
    print(f"🔍 Testing embedding quality with {sample_size} sample documents...")
    
    # Sample documents for testing
    sample_indices = np.random.choice(len(embeddings_matrix), min(sample_size, len(embeddings_matrix)), replace=False)
    
    for idx in sample_indices:
        doc_id = doc_ids[idx]
        doc_text = documents[idx]['text'][:100] + "..."
        
        # Calculate similarities to all other documents
        embedding = embeddings_matrix[idx:idx+1]
        similarities = np.dot(embeddings_matrix, embedding.T).flatten()
        
        # Find most similar documents (excluding self)
        similarities[idx] = -1  # Exclude self
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\n📄 Document {doc_id}: {doc_text}")
        print("   Most similar documents:")
        for i, similar_idx in enumerate(top_indices, 1):
            similar_doc_id = doc_ids[similar_idx]
            similarity = similarities[similar_idx]
            similar_text = documents[similar_idx]['text'][:80] + "..."
            print(f"      {i}. {similar_doc_id}: {similarity:.4f} - {similar_text}")

# Test embedding quality
test_embeddings_quality(embeddings_matrix, doc_ids, documents)

# STEP 9: Download Files
def download_bertweet_files():
    """
    Download all BERTweet files
    """
    print("📥 Downloading BERTweet files...")
    
    files_to_download = [
        'antique_bertweet_embeddings.npy',
        'antique_bertweet_doc_ids.npy', 
        'antique_bertweet_document_metadata.joblib',
        'antique_bertweet_metadata.json'
    ]
    
    try:
        # Check file sizes
        print("📋 File Information:")
        for filename in files_to_download:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"   {filename}: {size_mb:.2f} MB")
            else:
                print(f"   {filename}: File not found")
        
        # Download individual files
        for filename in files_to_download:
            if os.path.exists(filename):
                print(f"📥 Downloading {filename}...")
                files.download(filename)
        
        # Create and download model archive
        if os.path.exists('antique_bertweet_model'):
            import shutil
            shutil.make_archive('antique_bertweet_model', 'zip', '.', 'antique_bertweet_model')
            print("📥 Downloading antique_bertweet_model.zip...")
            files.download('antique_bertweet_model.zip')
        
        print("✅ All BERTweet files downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading files: {e}")

# Download files
download_bertweet_files()

# STEP 10: Final Summary and Instructions
print("\n" + "="*80)
print("🎉 BERTWEET EMBEDDING GENERATION COMPLETED")
print("="*80)
print(f"""
📊 SUMMARY:
   Model: vinai/bertweet-base
   Documents processed: {len(documents):,}
   Embedding dimension: {embeddings_matrix.shape[1]}
   Device used: {device}
   Data source: SQLite database (cleaned antique dataset)

📁 FILES CREATED:
   ✅ antique_bertweet_model/ (BERTweet model and tokenizer)
   ✅ antique_bertweet_embeddings.npy (NumPy embeddings matrix)
   ✅ antique_bertweet_doc_ids.npy (Document IDs)
   ✅ antique_bertweet_document_metadata.joblib (Document metadata)
   ✅ antique_bertweet_metadata.json (Generation metadata)

🔄 INTEGRATION STEPS:
   1. Download all generated files to your local backend
   2. Load embeddings using the BERTweet embedding service
   3. Use embeddings for semantic similarity search
   4. Integrate with your existing search pipeline

💡 USAGE EXAMPLE:
   # Load embeddings in your backend
   embeddings = np.load('antique_bertweet_embeddings.npy')
   doc_ids = np.load('antique_bertweet_doc_ids.npy')
   
   # Use for similarity search
   similarities = np.dot(embeddings, query_embedding.T)
   
🎯 FEATURES:
   ✅ BERTweet embeddings for high-quality text representation
   ✅ Database integration with cleaned antique dataset
   ✅ Memory-efficient batch processing
   ✅ GPU acceleration support
   ✅ Ready for production use
""")

print(f"✅ Successfully processed {len(documents):,} documents from SQLite database!")
print("🚀 BERTweet embeddings ready for your search engine!")
