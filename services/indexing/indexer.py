"""
Indexing Service
Handles building indices for different document representations:
- TF-IDF Vector Space Model
- Embedding-based (BERT, Word2Vec)
- BM25
- Hybrid representations
"""

import asyncio
import joblib
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import math

# Scikit-learn for TF-IDF and vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Transformers for BERT embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# Gensim for Word2Vec
from gensim.models import Word2Vec

# BM25
from rank_bm25 import BM25Okapi

# Import document class
from services.data_preprocessing.preprocessor import Document

class DocumentIndex:
    """Document index with multiple representations"""
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.documents = []
        self.doc_id_to_idx = {}
        
        # TF-IDF representation
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Embedding representation
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.document_embeddings = None
        
        # Word2Vec representation
        self.word2vec_model = None
        self.word2vec_embeddings = None
        
        # BM25 representation
        self.bm25_model = None
        self.bm25_corpus = None
        
        # Inverted index
        self.inverted_index = defaultdict(list)
        self.term_frequencies = defaultdict(dict)
        self.document_frequencies = defaultdict(int)
        
        # Vocabulary
        self.vocabulary = set()

class IndexingService:
    """Service for building and managing document indices"""
    
    def __init__(self):
        self.indices = {}  # dataset_name -> DocumentIndex
        self.data_dir = "data/indices"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize BERT model for embeddings
        self.bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def build_indices(self, dataset_name: str, documents: List[Document]):
        """Build all types of indices for a dataset"""
        print(f"🔧 Building indices for {dataset_name}...")
        
        # Create document index
        index = DocumentIndex(dataset_name)
        index.documents = documents
        
        # Build document ID mapping
        for i, doc in enumerate(documents):
            index.doc_id_to_idx[doc.doc_id] = i
        
        # Build different representations
        await self._build_tfidf_index(index)
        await self._build_embedding_index(index)
        await self._build_bm25_index(index)
        await self._build_inverted_index(index)
        await self._build_word2vec_index(index)
        
        # Store index
        self.indices[dataset_name] = index
        
        # Save to disk
        await self._save_index(index)
        
        print(f"✅ All indices built for {dataset_name}")
    
    async def _build_tfidf_index(self, index: DocumentIndex):
        """Build TF-IDF vector space model"""
        print("📊 Building TF-IDF index...")
        
        # Prepare document texts
        documents_text = []
        for doc in index.documents:
            # Use lemmatized tokens for better representation
            text = " ".join(doc.lemmatized_tokens) if doc.lemmatized_tokens else doc.processed_text
            documents_text.append(text)
        
        # Create TF-IDF vectorizer
        index.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            stop_words='english'
        )
        
        # Fit and transform documents
        index.tfidf_matrix = index.tfidf_vectorizer.fit_transform(documents_text)
        
        print(f"✅ TF-IDF index built: {index.tfidf_matrix.shape}")
    
    async def _build_embedding_index(self, index: DocumentIndex):
        """Build embedding-based index using sentence transformers"""
        print("🧠 Building embedding index...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load pre-trained sentence transformer model
            model = SentenceTransformer(self.bert_model_name)
            
            # Prepare document texts
            documents_text = []
            for doc in index.documents:
                # Combine title and processed text
                text = f"{doc.title} {doc.processed_text}".strip()
                documents_text.append(text)
            
            # Generate embeddings
            print("🔄 Generating document embeddings...")
            index.document_embeddings = model.encode(
                documents_text,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Store model reference
            index.embedding_model = model
            
            print(f"✅ Embedding index built: {index.document_embeddings.shape}")
            
        except Exception as e:
            print(f"⚠️ Error building embedding index: {e}")
            # Fallback to random embeddings for testing
            index.document_embeddings = np.random.rand(len(index.documents), 384)
    
    async def _build_word2vec_index(self, index: DocumentIndex):
        """Build Word2Vec embeddings and document representations"""
        print("📚 Building Word2Vec index...")
        
        try:
            # Prepare sentences for Word2Vec training
            sentences = []
            for doc in index.documents:
                if doc.lemmatized_tokens:
                    sentences.append(doc.lemmatized_tokens)
            
            if sentences:
                # Train Word2Vec model
                index.word2vec_model = Word2Vec(
                    sentences=sentences,
                    vector_size=100,
                    window=5,
                    min_count=2,
                    workers=4,
                    epochs=10
                )
                
                # Generate document embeddings by averaging word vectors
                index.word2vec_embeddings = []
                for doc in index.documents:
                    doc_vector = self._get_word2vec_document_vector(
                        doc.lemmatized_tokens, 
                        index.word2vec_model
                    )
                    index.word2vec_embeddings.append(doc_vector)
                
                index.word2vec_embeddings = np.array(index.word2vec_embeddings)
                print(f"✅ Word2Vec index built: {index.word2vec_embeddings.shape}")
            else:
                print("⚠️ No valid sentences for Word2Vec training")
                
        except Exception as e:
            print(f"⚠️ Error building Word2Vec index: {e}")
    
    def _get_word2vec_document_vector(self, tokens: List[str], model: Word2Vec) -> np.ndarray:
        """Generate document vector by averaging word vectors"""
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    async def _build_bm25_index(self, index: DocumentIndex):
        """Build BM25 index"""
        print("🎯 Building BM25 index...")
        
        try:
            # Prepare corpus for BM25
            corpus = []
            for doc in index.documents:
                tokens = doc.lemmatized_tokens if doc.lemmatized_tokens else doc.tokens
                corpus.append(tokens)
            
            # Create BM25 model
            index.bm25_model = BM25Okapi(corpus)
            index.bm25_corpus = corpus
            
            print(f"✅ BM25 index built for {len(corpus)} documents")
            
        except Exception as e:
            print(f"⚠️ Error building BM25 index: {e}")
    
    async def _build_inverted_index(self, index: DocumentIndex):
        """Build traditional inverted index"""
        print("📇 Building inverted index...")
        
        # Build inverted index and calculate term frequencies
        for doc_idx, doc in enumerate(index.documents):
            tokens = doc.lemmatized_tokens if doc.lemmatized_tokens else doc.tokens
            
            # Count term frequencies in document
            term_counts = Counter(tokens)
            total_terms = len(tokens)
            
            for term, count in term_counts.items():
                # Add to inverted index
                index.inverted_index[term].append((doc_idx, count))
                
                # Calculate TF
                tf = count / total_terms
                index.term_frequencies[term][doc_idx] = tf
                
                # Add to vocabulary
                index.vocabulary.add(term)
        
        # Calculate document frequencies
        for term in index.vocabulary:
            index.document_frequencies[term] = len(index.inverted_index[term])
        
        print(f"✅ Inverted index built: {len(index.vocabulary)} unique terms")
    
    async def _save_index(self, index: DocumentIndex):
        """Save index to disk"""
        try:
            index_path = os.path.join(self.data_dir, f"{index.dataset_name}_index.pkl")
            
            # Create a serializable version of the index
            index_data = {
                'dataset_name': index.dataset_name,
                'documents': index.documents,
                'doc_id_to_idx': index.doc_id_to_idx,
                'tfidf_matrix': index.tfidf_matrix,
                'document_embeddings': index.document_embeddings,
                'word2vec_embeddings': index.word2vec_embeddings,
                'bm25_corpus': index.bm25_corpus,
                'inverted_index': dict(index.inverted_index),
                'term_frequencies': dict(index.term_frequencies),
                'document_frequencies': dict(index.document_frequencies),
                'vocabulary': index.vocabulary
            }
            
            # Use joblib for better compression and performance
            joblib.dump(index_data, index_path, compress=3)
            
            print(f"💾 Index saved to {index_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving index: {e}")
    
    async def load_index(self, dataset_name: str) -> DocumentIndex:
        """Load index from disk"""
        try:
            index_path = os.path.join(self.data_dir, f"{dataset_name}_index.pkl")
            
            # Load using joblib
            index_data = joblib.load(index_path)
            
            # Reconstruct index
            index = DocumentIndex(dataset_name)
            index.dataset_name = index_data['dataset_name']
            index.documents = index_data['documents']
            index.doc_id_to_idx = index_data['doc_id_to_idx']
            index.tfidf_matrix = index_data['tfidf_matrix']
            index.document_embeddings = index_data['document_embeddings']
            index.word2vec_embeddings = index_data['word2vec_embeddings']
            index.bm25_corpus = index_data['bm25_corpus']
            index.inverted_index = defaultdict(list, index_data['inverted_index'])
            index.term_frequencies = defaultdict(dict, index_data['term_frequencies'])
            index.document_frequencies = defaultdict(int, index_data['document_frequencies'])
            index.vocabulary = index_data['vocabulary']
            
            # Rebuild models that can't be serialized
            if index.bm25_corpus:
                index.bm25_model = BM25Okapi(index.bm25_corpus)
            
            self.indices[dataset_name] = index
            print(f"📂 Index loaded for {dataset_name}")
            
            return index
            
        except Exception as e:
            print(f"⚠️ Error loading index: {e}")
            return None
    
    def get_index(self, dataset_name: str) -> DocumentIndex:
        """Get index for a dataset"""
        return self.indices.get(dataset_name)
    
    async def build_hybrid_representation(self, index: DocumentIndex, query_vector_tfidf: np.ndarray, query_vector_embedding: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Build hybrid representation by combining TF-IDF and embedding similarities"""
        
        # Calculate TF-IDF similarities
        tfidf_similarities = cosine_similarity(query_vector_tfidf, index.tfidf_matrix).flatten()
        
        # Calculate embedding similarities
        if index.document_embeddings is not None:
            embedding_similarities = cosine_similarity(
                query_vector_embedding.reshape(1, -1), 
                index.document_embeddings
            ).flatten()
        else:
            embedding_similarities = np.zeros(len(index.documents))
        
        # Normalize similarities to [0, 1]
        tfidf_similarities = (tfidf_similarities - tfidf_similarities.min()) / (tfidf_similarities.max() - tfidf_similarities.min() + 1e-8)
        embedding_similarities = (embedding_similarities - embedding_similarities.min()) / (embedding_similarities.max() - embedding_similarities.min() + 1e-8)
        
        # Combine using weighted average
        hybrid_similarities = alpha * tfidf_similarities + (1 - alpha) * embedding_similarities
        
        return hybrid_similarities
