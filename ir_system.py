#!/usr/bin/env python3
"""
Information Retrieval System with ANTIQUE and CodeSearchNet datasets
Implements data preprocessing, indexing, and search functionality
"""

import ir_datasets
import nltk
import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import json
from pathlib import Path
import random

# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class DataPreprocessor:
    """Enhanced text preprocessing with normalization and abbreviation expansion"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Country/Location normalization mappings
        self.country_normalizations = {
            'usa': 'united states',
            'us': 'united states', 
            'america': 'united states',
            'uk': 'united kingdom',
            'britain': 'united kingdom',
            'gb': 'united kingdom',
            'eu': 'european union',
            'uae': 'united arab emirates',
            'ussr': 'soviet union',
            'drc': 'democratic republic of congo',
        }
        
        # Technical abbreviation expansions
        self.tech_abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'sql': 'structured query language',
            'html': 'hypertext markup language',
            'css': 'cascading style sheets',
            'js': 'javascript',
            'py': 'python',
            'cpp': 'c plus plus',
            'db': 'database',
            'os': 'operating system',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'url': 'uniform resource locator',
            'uri': 'uniform resource identifier',
            'json': 'javascript object notation',
            'xml': 'extensible markup language',
            'yaml': 'yaml ain\'t markup language',
            'csv': 'comma separated values',
            'pdf': 'portable document format',
            'ide': 'integrated development environment',
            'sdk': 'software development kit',
            'cli': 'command line interface',
            'gui': 'graphical user interface',
            'orm': 'object relational mapping',
            'crud': 'create read update delete',
            'rest': 'representational state transfer',
            'soap': 'simple object access protocol',
            'tcp': 'transmission control protocol',
            'udp': 'user datagram protocol',
            'ip': 'internet protocol',
            'dns': 'domain name system',
            'ssl': 'secure sockets layer',
            'tls': 'transport layer security',
            'vpn': 'virtual private network',
            'cdn': 'content delivery network',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'devops': 'development operations',
            'regex': 'regular expression',
            'xss': 'cross site scripting',
            'csrf': 'cross site request forgery',
            'jwt': 'json web token',
            'oauth': 'open authorization',
        }
    
    def normalize_abbreviations_and_countries(self, text: str) -> str:
        """Normalize abbreviations and countries in text"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        words = text_lower.split()
        
        normalized_words = []
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check for country normalization
            if clean_word in self.country_normalizations:
                normalized_words.append(self.country_normalizations[clean_word])
            # Check for technical abbreviation normalization
            elif clean_word in self.tech_abbreviations:
                normalized_words.append(self.tech_abbreviations[clean_word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_process(self, text: str, use_stemming: bool = True, use_lemmatization: bool = False) -> List[str]:
        """Tokenize text and apply stemming/lemmatization"""
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens

class IRDatasetLoader:
    """Loads and manages ANTIQUE and CodeSearchNet datasets"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.datasets = {}
        self.documents = {}
        self.queries = {}
        self.qrels = {}
        
    def load_antique_dataset(self) -> Dict[str, Any]:
        """Load ANTIQUE dataset"""
        logger.info("Loading ANTIQUE dataset...")
        
        try:
            dataset = ir_datasets.load('antique/train')
            
            # Load documents
            docs = {}
            doc_count = 0
            for doc in dataset.docs_iter():
                docs[doc.doc_id] = {
                    'doc_id': doc.doc_id,
                    'text': doc.text,
                    'processed_text': self.preprocessor.tokenize_and_process(doc.text)
                }
                doc_count += 1
                if doc_count % 10000 == 0:
                    logger.info(f"Processed {doc_count} ANTIQUE documents")
            
            # Load queries
            queries = {}
            for query in dataset.queries_iter():
                queries[query.query_id] = {
                    'query_id': query.query_id,
                    'text': query.text,
                    'processed_text': self.preprocessor.tokenize_and_process(query.text)
                }
            
            # Load qrels (relevance judgments)
            qrels = defaultdict(dict)
            for qrel in dataset.qrels_iter():
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            logger.info(f"ANTIQUE dataset loaded: {len(docs)} documents, {len(queries)} queries, {len(qrels)} query-document pairs")
            
            self.datasets['antique'] = dataset
            self.documents['antique'] = docs
            self.queries['antique'] = queries
            self.qrels['antique'] = dict(qrels)
            
            return {
                'documents': len(docs),
                'queries': len(queries),
                'qrels': len(qrels)
            }
            
        except Exception as e:
            logger.error(f"Error loading ANTIQUE dataset: {e}")
            return {}
    
    def load_codesearchnet_dataset(self) -> Dict[str, Any]:
        """Load CodeSearchNet dataset using ir-datasets with proper qrels"""
        logger.info("Loading CodeSearchNet dataset using ir-datasets...")
        
        try:
            # Load CodeSearchNet dataset using ir-datasets
            dataset = ir_datasets.load('codesearchnet/train')
            
            # Load documents
            docs = {}
            doc_count = 0
            for doc in dataset.docs_iter():
                # The CodeSearchNet documents only have a 'code' field
                # Extract docstring from code if it exists
                code = doc.code
                
                # Try to extract docstring from code if it exists
                docstring = ""
                if code and '"""' in code:
                    # Simple docstring extraction
                    parts = code.split('"""')
                    if len(parts) >= 3:
                        docstring = parts[1].strip()
                
                # Use code as main content, prepend docstring if available
                content = f"{docstring}\n\n{code}" if docstring else code
                
                # Apply enhanced preprocessing with normalization
                normalized_content = self.preprocessor.normalize_abbreviations_and_countries(content)
                processed_text = self.preprocessor.tokenize_and_process(normalized_content)
                
                docs[doc.doc_id] = {
                    'doc_id': doc.doc_id,
                    'text': content,
                    'processed_text': processed_text,
                    'language': getattr(doc, 'language', 'unknown'),
                    'func_name': getattr(doc, 'func_name', '')
                }
                doc_count += 1
                if doc_count % 10000 == 0:
                    logger.info(f"Processed {doc_count} CodeSearchNet documents")
            
            # Load queries
            queries = {}
            for query in dataset.queries_iter():
                # Apply enhanced preprocessing with normalization
                normalized_query = self.preprocessor.normalize_abbreviations_and_countries(query.text)
                processed_text = self.preprocessor.tokenize_and_process(normalized_query)
                
                queries[query.query_id] = {
                    'query_id': query.query_id,
                    'text': query.text,
                    'processed_text': processed_text
                }
            
            # Load qrels (relevance judgments)
            qrels = defaultdict(dict)
            for qrel in dataset.qrels_iter():
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            logger.info(f"CodeSearchNet dataset loaded: {len(docs)} documents, {len(queries)} queries, {len(qrels)} query-document pairs")
            
            self.documents['codesearchnet'] = docs
            self.queries['codesearchnet'] = queries
            self.qrels['codesearchnet'] = dict(qrels)
            
            return {
                'documents': len(docs),
                'queries': len(queries),
                'qrels': len(qrels)
            }
            
        except Exception as e:
            logger.error(f"Error loading CodeSearchNet dataset: {e}")
            return {}
    
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a loaded dataset"""
        if dataset_name not in self.documents:
            return {}
        
        return {
            'name': dataset_name,
            'documents': len(self.documents[dataset_name]),
            'queries': len(self.queries[dataset_name]),
            'qrels': len(self.qrels[dataset_name]),
            'sample_document': list(self.documents[dataset_name].values())[0] if self.documents[dataset_name] else None,
            'sample_query': list(self.queries[dataset_name].values())[0] if self.queries[dataset_name] else None
        }

class SearchEngine:
    """Implementation of search functionality using BM25 and TF-IDF"""
    
    def __init__(self, dataset_loader: IRDatasetLoader):
        self.dataset_loader = dataset_loader
        self.bm25_indexes = {}
        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}
        
    def build_bm25_index(self, dataset_name: str):
        """Build BM25 index for a dataset"""
        if dataset_name not in self.dataset_loader.documents:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Building BM25 index for {dataset_name}...")
        
        documents = self.dataset_loader.documents[dataset_name]
        processed_docs = [doc['processed_text'] for doc in documents.values()]
        
        self.bm25_indexes[dataset_name] = BM25Okapi(processed_docs)
        logger.info(f"BM25 index built for {dataset_name}")
    
    def build_tfidf_index(self, dataset_name: str):
        """Build TF-IDF index for a dataset"""
        if dataset_name not in self.dataset_loader.documents:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Building TF-IDF index for {dataset_name}...")
        
        documents = self.dataset_loader.documents[dataset_name]
        text_docs = [' '.join(doc['processed_text']) for doc in documents.values()]
        
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(text_docs)
        
        self.tfidf_vectorizers[dataset_name] = vectorizer
        self.tfidf_matrices[dataset_name] = tfidf_matrix
        
        logger.info(f"TF-IDF index built for {dataset_name}")
    
    def search_bm25(self, query: str, dataset_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if dataset_name not in self.bm25_indexes:
            raise ValueError(f"BM25 index not built for {dataset_name}")
        
        # Process query
        processed_query = self.dataset_loader.preprocessor.tokenize_and_process(query)
        
        # Get scores
        scores = self.bm25_indexes[dataset_name].get_scores(processed_query)
        
        # Get top documents
        doc_ids = list(self.dataset_loader.documents[dataset_name].keys())
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = doc_ids[idx]
            doc = self.dataset_loader.documents[dataset_name][doc_id]
            results.append({
                'doc_id': doc_id,
                'score': float(scores[idx]),
                'text': doc['text'][:500] + '...' if len(doc['text']) > 500 else doc['text']
            })
        
        return results
    
    def search_tfidf(self, query: str, dataset_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using TF-IDF"""
        if dataset_name not in self.tfidf_vectorizers:
            raise ValueError(f"TF-IDF index not built for {dataset_name}")
        
        # Process query
        processed_query = ' '.join(self.dataset_loader.preprocessor.tokenize_and_process(query))
        
        # Vectorize query
        query_vector = self.tfidf_vectorizers[dataset_name].transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrices[dataset_name]).flatten()
        
        # Get top documents
        doc_ids = list(self.dataset_loader.documents[dataset_name].keys())
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = doc_ids[idx]
            doc = self.dataset_loader.documents[dataset_name][doc_id]
            results.append({
                'doc_id': doc_id,
                'score': float(similarities[idx]),
                'text': doc['text'][:500] + '...' if len(doc['text']) > 500 else doc['text']
            })
        
        return results

class IRSystem:
    """Main Information Retrieval System"""
    
    def __init__(self):
        self.dataset_loader = IRDatasetLoader()
        self.search_engine = SearchEngine(self.dataset_loader)
        self.loaded_datasets = set()
    
    def initialize_datasets(self):
        """Load and initialize both datasets"""
        logger.info("Initializing IR System with ANTIQUE and CodeSearchNet datasets...")
        
        # Load ANTIQUE
        antique_info = self.dataset_loader.load_antique_dataset()
        if antique_info:
            self.loaded_datasets.add('antique')
            self.search_engine.build_bm25_index('antique')
            self.search_engine.build_tfidf_index('antique')
        
        # Load CodeSearchNet
        codesearchnet_info = self.dataset_loader.load_codesearchnet_dataset()
        if codesearchnet_info:
            self.loaded_datasets.add('codesearchnet')
            self.search_engine.build_bm25_index('codesearchnet')
            self.search_engine.build_tfidf_index('codesearchnet')
        
        logger.info(f"IR System initialized with datasets: {self.loaded_datasets}")
        return {
            'antique': antique_info,
            'codesearchnet': codesearchnet_info
        }
    
    def search(self, query: str, dataset_name: str, method: str = 'bm25', top_k: int = 10) -> Dict[str, Any]:
        """Perform search on specified dataset"""
        if dataset_name not in self.loaded_datasets:
            return {'error': f'Dataset {dataset_name} not loaded'}
        
        try:
            if method == 'bm25':
                results = self.search_engine.search_bm25(query, dataset_name, top_k)
            elif method == 'tfidf':
                results = self.search_engine.search_tfidf(query, dataset_name, top_k)
            else:
                return {'error': f'Unknown search method: {method}'}
            
            return {
                'query': query,
                'dataset': dataset_name,
                'method': method,
                'results': results,
                'preprocessing_info': {
                    'cleaned_query': self.dataset_loader.preprocessor.clean_text(query),
                    'processed_tokens': self.dataset_loader.preprocessor.tokenize_and_process(query)
                }
            }
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {'error': str(e)}
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        """Get information about loaded datasets"""
        if dataset_name:
            return self.dataset_loader.get_dataset_info(dataset_name)
        else:
            return {
                name: self.dataset_loader.get_dataset_info(name)
                for name in self.loaded_datasets
            }
    
    def evaluate_query(self, query_id: str, dataset_name: str, method: str = 'bm25', top_k: int = 10) -> Dict[str, Any]:
        """Evaluate search results against ground truth qrels"""
        if dataset_name not in self.loaded_datasets:
            return {'error': f'Dataset {dataset_name} not loaded'}
        
        if query_id not in self.dataset_loader.queries[dataset_name]:
            return {'error': f'Query {query_id} not found in {dataset_name}'}
        
        # Get query text
        query_text = self.dataset_loader.queries[dataset_name][query_id]['text']
        
        # Perform search
        search_results = self.search(query_text, dataset_name, method, top_k)
        
        if 'error' in search_results:
            return search_results
        
        # Get ground truth
        ground_truth = self.dataset_loader.qrels[dataset_name].get(query_id, {})
        
        # Calculate evaluation metrics
        retrieved_docs = [result['doc_id'] for result in search_results['results']]
        relevant_docs = [doc_id for doc_id, relevance in ground_truth.items() if relevance > 0]
        
        # Precision and Recall
        retrieved_relevant = len(set(retrieved_docs) & set(relevant_docs))
        precision = retrieved_relevant / len(retrieved_docs) if retrieved_docs else 0
        recall = retrieved_relevant / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'dataset': dataset_name,
            'method': method,
            'search_results': search_results,
            'evaluation': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'retrieved_docs': len(retrieved_docs),
                'relevant_docs': len(relevant_docs),
                'retrieved_relevant': retrieved_relevant
            },
            'ground_truth': ground_truth
        }

if __name__ == "__main__":
    # Initialize and test the IR system
    ir_system = IRSystem()
    
    # Load datasets
    print("Loading datasets...")
    dataset_info = ir_system.initialize_datasets()
    print("Dataset loading complete!")
    print(json.dumps(dataset_info, indent=2))
    
    # Test search
    test_queries = [
        "How to install Python packages",
        "machine learning algorithms",
        "database connection error"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        
        for dataset in ['antique', 'codesearchnet']:
            if dataset in ir_system.loaded_datasets:
                result = ir_system.search(query, dataset, 'bm25', 5)
                print(f"\n{dataset.upper()} - BM25 Results:")
                if 'results' in result:
                    for i, doc in enumerate(result['results'][:3], 1):
                        print(f"{i}. Score: {doc['score']:.4f}")
                        print(f"   Text: {doc['text'][:100]}...")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
