"""
Query Processing Service
Handles query preprocessing and representation for different search methods
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np

# Import preprocessing service
from services.data_preprocessing.preprocessor import DataPreprocessingService

class ProcessedQuery:
    """Processed query with multiple representations"""
    def __init__(self, original_query: str):
        self.original = original_query
        self.cleaned = ""
        self.tokens = []
        self.filtered_tokens = []
        self.stemmed_tokens = []
        self.lemmatized_tokens = []
        
        # Representation vectors
        self.tfidf_vector = None
        self.embedding_vector = None
        self.word2vec_vector = None

class QueryProcessingService:
    """Service for processing queries"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessingService()
    
    async def process_query(self, query: str, representation: str) -> ProcessedQuery:
        """Process query for a specific representation type"""
        print(f"🔍 Processing query: '{query}' for {representation} representation")
        
        # Create processed query object
        processed_query = ProcessedQuery(query)
        
        # Basic preprocessing
        query_data = await self.preprocessor.preprocess_query(query)
        
        processed_query.cleaned = query_data['cleaned']
        processed_query.tokens = query_data['tokens']
        processed_query.filtered_tokens = query_data['filtered_tokens']
        processed_query.stemmed_tokens = query_data['stemmed_tokens']
        processed_query.lemmatized_tokens = query_data['lemmatized_tokens']
        
        return processed_query
    
    async def vectorize_query_tfidf(self, processed_query: ProcessedQuery, tfidf_vectorizer) -> np.ndarray:
        """Convert query to TF-IDF vector"""
        try:
            # Use lemmatized tokens for consistency with document indexing
            query_text = " ".join(processed_query.lemmatized_tokens)
            
            # Transform using the fitted vectorizer
            query_vector = tfidf_vectorizer.transform([query_text])
            processed_query.tfidf_vector = query_vector
            
            return query_vector
            
        except Exception as e:
            print(f"⚠️ Error vectorizing query with TF-IDF: {e}")
            return None
    
    async def vectorize_query_embedding(self, processed_query: ProcessedQuery, embedding_model) -> np.ndarray:
        """Convert query to embedding vector"""
        try:
            # Use original query for embeddings (they handle preprocessing internally)
            query_text = processed_query.original
            
            # Generate embedding
            if hasattr(embedding_model, 'encode'):
                # Sentence transformer model
                query_vector = embedding_model.encode([query_text], convert_to_numpy=True)
                processed_query.embedding_vector = query_vector[0]
                return query_vector[0]
            else:
                print("⚠️ Embedding model does not have encode method")
                return None
                
        except Exception as e:
            print(f"⚠️ Error vectorizing query with embeddings: {e}")
            return None
    
    async def vectorize_query_word2vec(self, processed_query: ProcessedQuery, word2vec_model) -> np.ndarray:
        """Convert query to Word2Vec vector by averaging word vectors"""
        try:
            vectors = []
            for token in processed_query.lemmatized_tokens:
                if token in word2vec_model.wv:
                    vectors.append(word2vec_model.wv[token])
            
            if vectors:
                query_vector = np.mean(vectors, axis=0)
                processed_query.word2vec_vector = query_vector
                return query_vector
            else:
                # Return zero vector if no words found
                query_vector = np.zeros(word2vec_model.vector_size)
                processed_query.word2vec_vector = query_vector
                return query_vector
                
        except Exception as e:
            print(f"⚠️ Error vectorizing query with Word2Vec: {e}")
            return np.zeros(100)  # Default size
    
    async def prepare_query_for_bm25(self, processed_query: ProcessedQuery) -> List[str]:
        """Prepare query tokens for BM25 search"""
        # Use lemmatized tokens for consistency
        return processed_query.lemmatized_tokens
    
    def expand_query(self, processed_query: ProcessedQuery, expansion_method: str = "synonyms") -> ProcessedQuery:
        """Expand query with additional terms"""
        # This is a placeholder for query expansion techniques
        # In a real implementation, you might use:
        # - WordNet synonyms
        # - Word2Vec similar words
        # - Relevance feedback
        # - Pseudo-relevance feedback
        
        expanded_query = processed_query
        
        if expansion_method == "synonyms":
            # Add synonym expansion logic here
            pass
        elif expansion_method == "word2vec":
            # Add Word2Vec-based expansion logic here
            pass
        
        return expanded_query
    
    def apply_query_reformulation(self, query: str, reformulation_type: str = "spelling") -> str:
        """Apply query reformulation techniques"""
        reformulated_query = query
        
        if reformulation_type == "spelling":
            # Add spelling correction logic here
            pass
        elif reformulation_type == "stemming":
            # Apply aggressive stemming
            pass
        
        return reformulated_query
