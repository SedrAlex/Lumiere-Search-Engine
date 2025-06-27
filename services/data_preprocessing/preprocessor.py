"""
Data Preprocessing Service with ir-datasets integration
Supports large-scale IR datasets from ir-datasets.com
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import pandas as pd

# Import ir_datasets if available
try:
    import ir_datasets
    IR_DATASETS_AVAILABLE = True
except ImportError:
    IR_DATASETS_AVAILABLE = False
    print("⚠️ ir_datasets not available. Install with: pip install ir-datasets")

# Import NLTK for text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available. Install with: pip install nltk")

@dataclass
class Document:
    """Document representation"""
    doc_id: str
    title: str = ""
    text: str = ""
    url: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class Query:
    """Query representation"""
    query_id: str
    text: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class QRel:
    """Query relevance judgment"""
    query_id: str
    doc_id: str
    relevance: int
    iteration: str = "Q0"

class DataPreprocessingService:
    """Service for loading and preprocessing IR datasets"""
    
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
        
        # Available datasets from ir-datasets
        self.available_datasets = {
            # Large datasets with >200K documents and test queries
            "msmarco-passage": {
                "name": "MS MARCO Passage Ranking",
                "description": "Microsoft AI & Research dataset for passage ranking",
                "doc_count": 8841823,
                "has_queries": True,
                "has_qrels": True
            },
            "msmarco-document": {
                "name": "MS MARCO Document Ranking", 
                "description": "Microsoft AI & Research dataset for document ranking",
                "doc_count": 3213835,
                "has_queries": True,
                "has_qrels": True
            },
            "trec-covid": {
                "name": "TREC-COVID",
                "description": "COVID-19 research literature dataset",
                "doc_count": 171332,
                "has_queries": True,
                "has_qrels": True
            },
            "beir/nfcorpus": {
                "name": "NFCorpus",
                "description": "Nutrition facts corpus",
                "doc_count": 3633,
                "has_queries": True,
                "has_qrels": True
            },
            "beir/scifact": {
                "name": "SciFact",
                "description": "Scientific fact verification dataset",
                "doc_count": 5183,
                "has_queries": True,
                "has_qrels": True
            }
        }
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available datasets"""
        return self.available_datasets
    
    async def load_dataset(self, dataset_name: str, max_docs: int = 10000) -> Tuple[List[Document], List[Query], List[QRel]]:
        """Load dataset from ir-datasets"""
        if not IR_DATASETS_AVAILABLE:
            raise Exception("ir_datasets not available. Please install: pip install ir-datasets")
        
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {list(self.available_datasets.keys())}")
        
        print(f"📥 Loading dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = ir_datasets.load(dataset_name)
            
            # Load documents
            documents = []
            print(f"📄 Loading documents (max {max_docs})...")
            
            for i, doc in enumerate(dataset.docs_iter()):
                if i >= max_docs:
                    break
                    
                # Extract document fields based on dataset structure
                doc_id = doc.doc_id
                title = getattr(doc, 'title', '')
                text = getattr(doc, 'text', getattr(doc, 'body', ''))
                url = getattr(doc, 'url', '')
                
                documents.append(Document(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    url=url
                ))
                
                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1} documents...")
            
            print(f"✅ Loaded {len(documents)} documents")
            
            # Load queries
            queries = []
            if hasattr(dataset, 'queries_iter'):
                print("📋 Loading queries...")
                for query in dataset.queries_iter():
                    queries.append(Query(
                        query_id=query.query_id,
                        text=query.text
                    ))
                print(f"✅ Loaded {len(queries)} queries")
            
            # Load relevance judgments
            qrels = []
            if hasattr(dataset, 'qrels_iter'):
                print("📊 Loading relevance judgments...")
                for qrel in dataset.qrels_iter():
                    qrels.append(QRel(
                        query_id=qrel.query_id,
                        doc_id=qrel.doc_id,
                        relevance=qrel.relevance,
                        iteration=getattr(qrel, 'iteration', 'Q0')
                    ))
                print(f"✅ Loaded {len(qrels)} relevance judgments")
            
            return documents, queries, qrels
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    async def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess documents (clean text, tokenize, etc.)"""
        print(f"🔧 Preprocessing {len(documents)} documents...")
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                # Clean and preprocess text
                processed_text = self._clean_text(doc.text)
                processed_title = self._clean_text(doc.title)
                
                # Create processed document
                processed_doc = Document(
                    doc_id=doc.doc_id,
                    title=processed_title,
                    text=processed_text,
                    url=doc.url,
                    metadata={
                        **doc.metadata,
                        'original_length': len(doc.text),
                        'processed_length': len(processed_text),
                        'preprocessing_applied': True
                    }
                )
                
                processed_docs.append(processed_doc)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1} documents...")
                    
            except Exception as e:
                print(f"⚠️ Error processing document {doc.doc_id}: {e}")
                # Keep original document if processing fails
                processed_docs.append(doc)
        
        print(f"✅ Preprocessed {len(processed_docs)} documents")
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                return [token.lower() for token in tokens if token.isalnum()]
            except:
                pass
        
        # Fallback tokenization
        return [word.strip() for word in text.lower().split() if word.strip()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        if not self.stemmer:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    async def get_dataset_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics about a dataset"""
        if not IR_DATASETS_AVAILABLE:
            return {"error": "ir_datasets not available"}
        
        try:
            dataset = ir_datasets.load(dataset_name)
            
            stats = {
                "name": dataset_name,
                "has_docs": hasattr(dataset, 'docs_iter'),
                "has_queries": hasattr(dataset, 'queries_iter'),
                "has_qrels": hasattr(dataset, 'qrels_iter'),
            }
            
            # Count documents (sampling approach for large datasets)
            if hasattr(dataset, 'docs_iter'):
                try:
                    doc_count = 0
                    for doc in dataset.docs_iter():
                        doc_count += 1
                        if doc_count >= 1000:  # Sample first 1000 for estimation
                            break
                    stats["sample_doc_count"] = doc_count
                except:
                    stats["sample_doc_count"] = "unknown"
            
            # Count queries
            if hasattr(dataset, 'queries_iter'):
                try:
                    query_count = sum(1 for _ in dataset.queries_iter())
                    stats["query_count"] = query_count
                except:
                    stats["query_count"] = "unknown"
            
            # Count qrels
            if hasattr(dataset, 'qrels_iter'):
                try:
                    qrel_count = sum(1 for _ in dataset.qrels_iter())
                    stats["qrel_count"] = qrel_count
                except:
                    stats["qrel_count"] = "unknown"
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def create_sample_documents(self, count: int = 100) -> List[Document]:
        """Create sample documents for testing when real datasets aren't available"""
        print(f"📝 Creating {count} sample documents for testing...")
        
        sample_docs = []
        
        topics = [
            ("artificial intelligence", "AI machine learning deep learning neural networks computer vision"),
            ("climate change", "global warming environment sustainability carbon emissions renewable energy"),
            ("space exploration", "NASA SpaceX Mars mission astronauts rockets satellites"),
            ("medicine", "healthcare treatment diagnosis disease medical research pharmaceuticals"),
            ("technology", "software development programming computers internet digital innovation")
        ]
        
        for i in range(count):
            topic_idx = i % len(topics)
            topic_name, topic_keywords = topics[topic_idx]
            
            doc_id = f"doc_{i+1:05d}"
            title = f"Research on {topic_name} - Document {i+1}"
            text = f"This document discusses {topic_name}. Key topics include: {topic_keywords}. " * 5
            
            sample_docs.append(Document(
                doc_id=doc_id,
                title=title,
                text=text,
                metadata={"topic": topic_name, "sample": True}
            ))
        
        return sample_docs
