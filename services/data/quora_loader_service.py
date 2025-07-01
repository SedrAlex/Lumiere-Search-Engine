"""
Quora Dataset Loading Service
Loads and streams the Quora dataset for processing by representation services
"""

import asyncio
import logging
import json
import csv
import sqlite3
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPRESENTATION_SERVICES = {
    "tfidf_quora": "http://localhost:8006",
}

# Request/Response Models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class LoadRequest(BaseModel):
    data_path: str
    max_documents: Optional[int] = 10000
    representation_services: Optional[List[str]] = ["tfidf_quora"]

class LoadResponse(BaseModel):
    message: str
    total_documents: int
    services_indexed: Dict[str, bool]
    processing_time: float

class QuoraDatabaseLoader:
    """Database loader for Quora dataset files (docs, queries, qrels)"""
    
    def __init__(self, db_path="data/database/documents.db"):
        self.db_path = db_path
        self.quora_folder = Path.home() / "Downloads" / "quora"
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for Quora dataset"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create quora_docs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_docs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create quora_queries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT UNIQUE NOT NULL,
                        text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create quora_qrels table (query-document relevance)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_qrels (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        relevance INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(query_id, doc_id)
                    )
                """)
                
                conn.commit()
                print("✅ Quora database tables initialized")
                
        except Exception as e:
            print(f"❌ Error initializing Quora database tables: {e}")
    
    def load_docs_to_database(self):
        """Load docs.tsv file to database"""
        docs_file = self.quora_folder / "docs.tsv"
        
        if not docs_file.exists():
            print(f"❌ docs.tsv file not found at {docs_file}")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing data
                cursor.execute("DELETE FROM quora_docs")
                
                with open(docs_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    docs_data = []
                    
                    for row in reader:
                        docs_data.append((
                            row['doc_id'],
                            row['text']
                        ))
                    
                    # Batch insert
                    cursor.executemany("""
                        INSERT OR REPLACE INTO quora_docs (doc_id, text)
                        VALUES (?, ?)
                    """, docs_data)
                
                conn.commit()
                print(f"✅ Loaded {len(docs_data)} documents from docs.tsv")
                return True
                
        except Exception as e:
            print(f"❌ Error loading docs.tsv: {e}")
            return False
    
    def load_queries_to_database(self):
        """Load queries.tsv file to database"""
        queries_file = self.quora_folder / "queries.tsv"
        
        if not queries_file.exists():
            print(f"❌ queries.tsv file not found at {queries_file}")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing data
                cursor.execute("DELETE FROM quora_queries")
                
                with open(queries_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    queries_data = []
                    
                    for row in reader:
                        queries_data.append((
                            row['query_id'],
                            row['text']
                        ))
                    
                    # Batch insert
                    cursor.executemany("""
                        INSERT OR REPLACE INTO quora_queries (query_id, text)
                        VALUES (?, ?)
                    """, queries_data)
                
                conn.commit()
                print(f"✅ Loaded {len(queries_data)} queries from queries.tsv")
                return True
                
        except Exception as e:
            print(f"❌ Error loading queries.tsv: {e}")
            return False
    
    def load_qrels_to_database(self):
        """Load qrels.tsv file to database"""
        qrels_file = self.quora_folder / "qrels.tsv"
        
        if not qrels_file.exists():
            print(f"❌ qrels.tsv file not found at {qrels_file}")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing data
                cursor.execute("DELETE FROM quora_qrels")
                
                with open(qrels_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    qrels_data = []
                    
                    for row in reader:
                        qrels_data.append((
                            row['query_id'],
                            row['doc_id'],
                            int(row['relevance'])
                        ))
                    
                    # Batch insert
                    cursor.executemany("""
                        INSERT OR REPLACE INTO quora_qrels (query_id, doc_id, relevance)
                        VALUES (?, ?, ?)
                    """, qrels_data)
                
                conn.commit()
                print(f"✅ Loaded {len(qrels_data)} relevance judgments from qrels.tsv")
                return True
                
        except Exception as e:
            print(f"❌ Error loading qrels.tsv: {e}")
            return False
    
    def load_all_files(self):
        """Load all Quora dataset files to database"""
        print("🚀 Starting Quora dataset loading to database")
        print("=" * 50)
        
        results = {
            'docs': self.load_docs_to_database(),
            'queries': self.load_queries_to_database(),
            'qrels': self.load_qrels_to_database()
        }
        
        # Show statistics
        self.show_database_stats()
        
        return results
    
    def show_database_stats(self):
        """Show statistics about loaded data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get counts
                cursor.execute("SELECT COUNT(*) FROM quora_docs")
                docs_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM quora_queries")
                queries_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM quora_qrels")
                qrels_count = cursor.fetchone()[0]
                
                print("\n📊 Database Statistics:")
                print("-" * 30)
                print(f"  Documents: {docs_count:,}")
                print(f"  Queries: {queries_count:,}")
                print(f"  Relevance judgments: {qrels_count:,}")
                
                # Show sample data
                cursor.execute("SELECT doc_id, text FROM quora_docs LIMIT 3")
                sample_docs = cursor.fetchall()
                
                cursor.execute("SELECT query_id, text FROM quora_queries LIMIT 3")
                sample_queries = cursor.fetchall()
                
                if sample_docs:
                    print("\n📄 Sample Documents:")
                    for doc_id, text in sample_docs:
                        print(f"  {doc_id}: {text[:100]}...")
                
                if sample_queries:
                    print("\n🔍 Sample Queries:")
                    for query_id, text in sample_queries:
                        print(f"  {query_id}: {text}")
                
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
    
    def get_documents_for_processing(self, limit=None):
        """Get documents from database for processing by other services"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT doc_id, text FROM quora_docs"
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                documents = []
                for doc_id, text in rows:
                    documents.append({
                        'id': doc_id,
                        'text': text,
                        'metadata': {'dataset': 'quora'}
                    })
                
                return documents
                
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return []

class QuoraLoaderService:
    """Service for loading and processing the Quora dataset"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.db_loader = QuoraDatabaseLoader()
    
    async def load_quora_documents(self, data_path: str, max_documents: Optional[int] = None) -> List[Document]:
        """Load documents from the Quora dataset"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        documents = []
        
        # Handle different file formats
        if data_path.suffix.lower() == '.json':
            documents = await self._load_from_json(data_path, max_documents)
        elif data_path.suffix.lower() == '.jsonl':
            documents = await self._load_from_jsonl(data_path, max_documents)
        elif data_path.suffix.lower() in ['.csv', '.tsv']:
            documents = await self._load_from_csv(data_path, max_documents)
        else:
            # Try to auto-detect format
            try:
                documents = await self._load_from_json(data_path, max_documents)
            except:
                try:
                    documents = await self._load_from_jsonl(data_path, max_documents)
                except:
                    documents = await self._load_from_csv(data_path, max_documents)
        
        logger.info(f"Loaded {len(documents)} documents from {data_path}")
        return documents
    
    async def _load_from_json(self, file_path: Path, max_documents: Optional[int]) -> List[Document]:
        """Load documents from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        count = 0
        
        # Handle different JSON structures
        if isinstance(data, list):
            for item in data:
                if max_documents and count >= max_documents:
                    break
                doc = self._parse_document_item(item, count)
                if doc:
                    documents.append(doc)
                    count += 1
        elif isinstance(data, dict):
            for key, value in data.items():
                if max_documents and count >= max_documents:
                    break
                doc = self._parse_document_item(value, key)
                if doc:
                    documents.append(doc)
                    count += 1
        
        return documents
    
    async def _load_from_jsonl(self, file_path: Path, max_documents: Optional[int]) -> List[Document]:
        """Load documents from JSONL file"""
        documents = []
        count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_documents and count >= max_documents:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    doc = self._parse_document_item(item, line_num)
                    if doc:
                        documents.append(doc)
                        count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        return documents
    
    async def _load_from_csv(self, file_path: Path, max_documents: Optional[int]) -> List[Document]:
        """Load documents from CSV/TSV file"""
        documents = []
        count = 0
        
        # Detect delimiter
        delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row_num, row in enumerate(reader):
                if max_documents and count >= max_documents:
                    break
                
                doc = self._parse_document_item(row, row_num)
                if doc:
                    documents.append(doc)
                    count += 1
        
        return documents
    
    def _parse_document_item(self, item: Dict[str, Any], doc_id: Any) -> Optional[Document]:
        """Parse a document item from various formats"""
        # Common field names for document text
        text_fields = ['text', 'content', 'body', 'document', 'passage', 'answer']
        # Common field names for document ID
        id_fields = ['id', 'doc_id', 'document_id', 'qid', 'pid']
        
        # Extract text
        text = None
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field]).strip()
                break
        
        if not text:
            logger.warning(f"No text found in document {doc_id}")
            return None
        
        # Extract ID
        document_id = None
        for field in id_fields:
            if field in item and item[field]:
                document_id = str(item[field])
                break
        
        if not document_id:
            document_id = str(doc_id)
        
        # Extract metadata (everything else)
        metadata = {k: v for k, v in item.items() if k not in text_fields + id_fields}
        
        return Document(
            id=document_id,
            text=text,
            metadata=metadata
        )
    
    async def index_documents_in_services(self, documents: List[Document], services: List[str]) -> Dict[str, bool]:
        """Index documents in specified representation services"""
        results = {}
        
        for service_name in services:
            if service_name not in REPRESENTATION_SERVICES:
                logger.error(f"Unknown service: {service_name}")
                results[service_name] = False
                continue
            
            service_url = REPRESENTATION_SERVICES[service_name]
            
            try:
                # Prepare the request
                request_data = {
                    "documents": [doc.dict() for doc in documents]
                }
                
                # Make the request
                response = await self.http_client.post(
                    f"{service_url}/index",
                    json=request_data
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Successfully indexed {len(documents)} documents in {service_name} service")
                logger.info(f"Service response: {result}")
                results[service_name] = True
                
            except httpx.RequestError as e:
                logger.error(f"Error connecting to {service_name} service: {e}")
                results[service_name] = False
            except Exception as e:
                logger.error(f"Error indexing documents in {service_name}: {e}")
                results[service_name] = False
        
        return results
    
    async def load_and_index(self, data_path: str, max_documents: Optional[int] = 10000, services: List[str] = ["tfidf_quora"]) -> LoadResponse:
        """Load documents and index them in specified services"""
        start_time = asyncio.get_event_loop().time()
        
        # Load documents
        documents = await self.load_quora_documents(data_path, max_documents)
        
        # Index in services
        services_results = await self.index_documents_in_services(documents, services)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return LoadResponse(
            message=f"Loaded and indexed {len(documents)} documents",
            total_documents=len(documents),
            services_indexed=services_results,
            processing_time=processing_time
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# FastAPI app for the data loading service
app = FastAPI(
    title="Quora Dataset Loading Service",
    description="Loads and processes the Quora dataset for representation services",
    version="1.0.0"
)

# Global service instance
loader_service = QuoraLoaderService()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await loader_service.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Quora Dataset Loading Service",
        "version": "1.0.0",
        "description": "Loads and processes the Quora dataset",
        "available_services": list(REPRESENTATION_SERVICES.keys()),
        "endpoints": {
            "POST /load": "Load and index documents",
            "POST /load_documents": "Load documents only",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "quora_loader_service",
        "available_services": list(REPRESENTATION_SERVICES.keys())
    }

@app.post("/load", response_model=LoadResponse)
async def load_and_index_documents(request: LoadRequest):
    """Load documents from Quora dataset and index them in representation services"""
    try:
        result = await loader_service.load_and_index(
            data_path=request.data_path,
            max_documents=request.max_documents,
            services=request.representation_services or ["tfidf_quora"]
        )
        return result
    except Exception as e:
        logger.error(f"Error loading and indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loading error: {str(e)}")

@app.post("/load_documents")
async def load_documents_only(request: LoadRequest):
    """Load documents from Quora dataset without indexing"""
    try:
        documents = await loader_service.load_quora_documents(
            data_path=request.data_path,
            max_documents=request.max_documents
        )
        
        return {
            "message": f"Loaded {len(documents)} documents",
            "total_documents": len(documents),
            "sample_document": documents[0].dict() if documents else None
        }
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loading error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)

