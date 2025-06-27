"""
Data Preprocessing Service with ir-datasets integration
Supports large-scale IR datasets from ir-datasets.com
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass

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
        
        # Available datasets from ir-datasets with >200K documents
        self.available_datasets = {
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
            }
        }
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available datasets"""
        return self.available_datasets
    
    async def load_dataset(self, dataset_name: str, max_docs: int = 10000) -> Tuple[List[Document], List[Query], List[QRel]]:
        """Load dataset from ir-datasets"""
        if not IR_DATASETS_AVAILABLE:
            # Return sample data for testing
            return self._create_sample_data()
        
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
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Tuple[List[Document], List[Query], List[QRel]]:
        """Create sample data for testing"""
        print("📝 Creating sample data for testing...")
        
        # Sample documents
        documents = [
            Document("doc1", "Information Retrieval Systems", 
                    "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources."),
            Document("doc2", "Search Engines and Web Mining", 
                    "Search engines are complex distributed systems that help users find information quickly and efficiently on the web through sophisticated algorithms."),
            Document("doc3", "Natural Language Processing", 
                    "Natural language processing combines computational linguistics with statistical machine learning methods to give computers the ability to understand human language."),
            Document("doc4", "Machine Learning in IR", 
                    "Machine learning techniques have revolutionized information retrieval by enabling systems to learn from data and improve their performance over time."),
            Document("doc5", "Vector Space Models", 
                    "Vector space models represent documents and queries as vectors in a multi-dimensional space where similarity can be computed using cosine similarity.")
        ]
        
        # Sample queries
        queries = [
            Query("q1", "information retrieval systems"),
            Query("q2", "search engines algorithms"),
            Query("q3", "natural language processing machine learning")
        ]
        
        # Sample relevance judgments
        qrels = [
            QRel("q1", "doc1", 2),
            QRel("q1", "doc2", 1),
            QRel("q2", "doc2", 2),
            QRel("q3", "doc3", 2),
            QRel("q3", "doc4", 1)
        ]
        
        return documents, queries, qrels
    
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
