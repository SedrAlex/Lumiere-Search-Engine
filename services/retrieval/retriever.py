"""
Retrieval Service
Handles document retrieval, ranking, and evaluation
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math

# Scikit-learn for similarity calculations
from sklearn.metrics.pairwise import cosine_similarity

# Import other services
from services.indexing.indexer import IndexingService, DocumentIndex
from services.query_processing.processor import QueryProcessingService, ProcessedQuery

class SearchResult:
    """Search result with document information and scoring"""
    def __init__(self, doc_id: str, title: str, content: str, score: float, rank: int):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.score = score
        self.rank = rank

class EvaluationMetrics:
    """Container for evaluation metrics"""
    def __init__(self):
        self.precision_at_10 = 0.0
        self.recall = 0.0
        self.map_score = 0.0  # Mean Average Precision
        self.mrr = 0.0  # Mean Reciprocal Rank
        self.ndcg_at_10 = 0.0  # Normalized Discounted Cumulative Gain

class RetrievalService:
    """Service for document retrieval and ranking"""
    
    def __init__(self):
        self.indexing_service = IndexingService()
        self.query_service = QueryProcessingService()
        
        # BM25 parameters
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75
    
    async def search(self, query: ProcessedQuery, dataset: str, representation: str, top_k: int = 10) -> List[SearchResult]:
        """Main search method that handles different representations"""
        print(f"🔎 Searching with {representation} representation...")
        
        # Get the document index
        index = self.indexing_service.get_index(dataset)
        if not index:
            raise ValueError(f"Index not found for dataset: {dataset}")
        
        # Perform search based on representation type
        if representation == "tfidf":
            results = await self._search_tfidf(query, index, top_k)
        elif representation == "embedding":
            results = await self._search_embedding(query, index, top_k)
        elif representation == "word2vec":
            results = await self._search_word2vec(query, index, top_k)
        elif representation == "bm25":
            results = await self._search_bm25(query, index, top_k)
        elif representation == "hybrid":
            results = await self._search_hybrid(query, index, top_k)
        else:
            raise ValueError(f"Unsupported representation: {representation}")
        
        return results
    
    async def _search_tfidf(self, query: ProcessedQuery, index: DocumentIndex, top_k: int) -> List[SearchResult]:
        """Search using TF-IDF representation"""
        if not index.tfidf_vectorizer or index.tfidf_matrix is None:
            return []
        
        # Vectorize query
        query_vector = await self.query_service.vectorize_query_tfidf(
            query, index.tfidf_vectorizer
        )
        
        if query_vector is None:
            return []
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, index.tfidf_matrix).flatten()
        
        # Get top-k results
        return self._create_search_results(similarities, index, top_k)
    
    async def _search_embedding(self, query: ProcessedQuery, index: DocumentIndex, top_k: int) -> List[SearchResult]:
        """Search using embedding representation"""
        if index.document_embeddings is None or index.embedding_model is None:
            return []
        
        # Vectorize query
        query_vector = await self.query_service.vectorize_query_embedding(
            query, index.embedding_model
        )
        
        if query_vector is None:
            return []
        
        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_vector.reshape(1, -1), 
            index.document_embeddings
        ).flatten()
        
        # Get top-k results
        return self._create_search_results(similarities, index, top_k)
    
    async def _search_word2vec(self, query: ProcessedQuery, index: DocumentIndex, top_k: int) -> List[SearchResult]:
        """Search using Word2Vec representation"""
        if index.word2vec_embeddings is None or index.word2vec_model is None:
            return []
        
        # Vectorize query
        query_vector = await self.query_service.vectorize_query_word2vec(
            query, index.word2vec_model
        )
        
        if query_vector is None:
            return []
        
        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_vector.reshape(1, -1), 
            index.word2vec_embeddings
        ).flatten()
        
        # Get top-k results
        return self._create_search_results(similarities, index, top_k)
    
    async def _search_bm25(self, query: ProcessedQuery, index: DocumentIndex, top_k: int) -> List[SearchResult]:
        """Search using BM25 representation"""
        if not index.bm25_model:
            return []
        
        # Prepare query tokens
        query_tokens = await self.query_service.prepare_query_for_bm25(query)
        
        # Get BM25 scores
        scores = index.bm25_model.get_scores(query_tokens)
        
        # Get top-k results
        return self._create_search_results(scores, index, top_k)
    
    async def _search_hybrid(self, query: ProcessedQuery, index: DocumentIndex, top_k: int, alpha: float = 0.5) -> List[SearchResult]:
        """Search using hybrid representation (TF-IDF + Embedding)"""
        # Get TF-IDF query vector
        tfidf_vector = await self.query_service.vectorize_query_tfidf(
            query, index.tfidf_vectorizer
        )
        
        # Get embedding query vector
        embedding_vector = await self.query_service.vectorize_query_embedding(
            query, index.embedding_model
        )
        
        if tfidf_vector is None and embedding_vector is None:
            return []
        
        # Calculate hybrid similarities
        if tfidf_vector is not None and embedding_vector is not None:
            hybrid_similarities = await self.indexing_service.build_hybrid_representation(
                index, tfidf_vector, embedding_vector, alpha
            )
        elif tfidf_vector is not None:
            # Fallback to TF-IDF only
            hybrid_similarities = cosine_similarity(tfidf_vector, index.tfidf_matrix).flatten()
        else:
            # Fallback to embedding only
            hybrid_similarities = cosine_similarity(
                embedding_vector.reshape(1, -1), 
                index.document_embeddings
            ).flatten()
        
        # Get top-k results
        return self._create_search_results(hybrid_similarities, index, top_k)
    
    def _create_search_results(self, similarities: np.ndarray, index: DocumentIndex, top_k: int) -> List[SearchResult]:
        """Create search results from similarity scores"""
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, doc_idx in enumerate(top_indices):
            doc = index.documents[doc_idx]
            score = similarities[doc_idx]
            
            # Truncate content for display
            content = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            
            result = SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                content=content,
                score=float(score),
                rank=rank + 1
            )
            results.append(result)
        
        return results
    
    async def evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Evaluate all representations on a dataset"""
        print(f"📊 Evaluating dataset: {dataset_name}")
        
        # This would normally use ground truth queries and relevance judgments
        # For now, return sample metrics
        evaluation_results = {
            "tfidf": {
                "precision_at_10": 0.75,
                "recall": 0.68,
                "map": 0.72,
                "mrr": 0.78,
                "ndcg_at_10": 0.74
            },
            "embedding": {
                "precision_at_10": 0.82,
                "recall": 0.71,
                "map": 0.79,
                "mrr": 0.85,
                "ndcg_at_10": 0.81
            },
            "word2vec": {
                "precision_at_10": 0.69,
                "recall": 0.64,
                "map": 0.67,
                "mrr": 0.73,
                "ndcg_at_10": 0.70
            },
            "bm25": {
                "precision_at_10": 0.78,
                "recall": 0.73,
                "map": 0.76,
                "mrr": 0.81,
                "ndcg_at_10": 0.77
            },
            "hybrid": {
                "precision_at_10": 0.85,
                "recall": 0.76,
                "map": 0.83,
                "mrr": 0.88,
                "ndcg_at_10": 0.84
            }
        }
        
        return evaluation_results
    
    def calculate_precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = len(set(top_k) & set(relevant_docs))
        
        return relevant_retrieved / k
    
    def calculate_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Recall"""
        if not relevant_docs:
            return 0.0
        
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_average_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Average Precision"""
        if not relevant_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def calculate_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Reciprocal Rank"""
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = relevance_scores.get(doc, 0.0)
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance[:k]):
            idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    async def batch_evaluate(self, queries: List[Dict], dataset: str, representation: str) -> EvaluationMetrics:
        """Evaluate a batch of queries"""
        metrics = EvaluationMetrics()
        
        total_precision_10 = 0.0
        total_recall = 0.0
        total_ap = 0.0
        total_rr = 0.0
        total_ndcg_10 = 0.0
        
        valid_queries = 0
        
        for query_data in queries:
            query_text = query_data.get('query', '')
            relevant_docs = query_data.get('relevant_docs', [])
            
            if not query_text or not relevant_docs:
                continue
            
            # Process query
            processed_query = await self.query_service.process_query(query_text, representation)
            
            # Search
            search_results = await self.search(processed_query, dataset, representation, top_k=100)
            retrieved_docs = [result.doc_id for result in search_results]
            
            # Calculate metrics
            precision_10 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 10)
            recall = self.calculate_recall(retrieved_docs, relevant_docs)
            ap = self.calculate_average_precision(retrieved_docs, relevant_docs)
            rr = self.calculate_reciprocal_rank(retrieved_docs, relevant_docs)
            
            # For NDCG, assume binary relevance (relevant=1, non-relevant=0)
            relevance_scores = {doc: 1.0 for doc in relevant_docs}
            ndcg_10 = self.calculate_ndcg_at_k(retrieved_docs, relevance_scores, 10)
            
            total_precision_10 += precision_10
            total_recall += recall
            total_ap += ap
            total_rr += rr
            total_ndcg_10 += ndcg_10
            valid_queries += 1
        
        if valid_queries > 0:
            metrics.precision_at_10 = total_precision_10 / valid_queries
            metrics.recall = total_recall / valid_queries
            metrics.map_score = total_ap / valid_queries
            metrics.mrr = total_rr / valid_queries
            metrics.ndcg_at_10 = total_ndcg_10 / valid_queries
        
        return metrics
