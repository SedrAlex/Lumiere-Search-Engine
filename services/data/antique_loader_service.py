"""
Antique Dataset Loading Service
Loads and streams the Antique dataset for processing by representation services
"""

import asyncio
import logging
import json
import csv
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPRESENTATION_SERVICES = {
    "tfidf": "http://localhost:8002",
    # Add other services as they are created
    # "embeddings": "http://localhost:8003",
    # "bm25": "http://localhost:8004"
}

# Request/Response Models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class LoadRequest(BaseModel):
    data_path: str
    max_documents: Optional[int] = 10000
    representation_services: Optional[List[str]] = ["tfidf"]

class LoadResponse(BaseModel):
    message: str
    total_documents: int
    services_indexed: Dict[str, bool]
    processing_time: float

class StreamRequest(BaseModel):
    data_path: str
    batch_size: int = 100
    max_documents: Optional[int] = None

class AntiqueLoaderService:
    """Service for loading and processing the Antique dataset"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)
    
    async def load_antique_documents(self, data_path: str, max_documents: Optional[int] = None) -> List[Document]:
        """Load documents from the Antique dataset"""
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
    
    async def stream_documents(self, data_path: str, batch_size: int = 100, max_documents: Optional[int] = None) -> AsyncGenerator[List[Document], None]:
        """Stream documents in batches"""
        documents = await self.load_antique_documents(data_path, max_documents)
        
        # Yield documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            yield batch
    
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
    
    async def load_and_index(self, data_path: str, max_documents: Optional[int] = 10000, services: List[str] = ["tfidf"]) -> LoadResponse:
        """Load documents and index them in specified services"""
        start_time = asyncio.get_event_loop().time()
        
        # Load documents
        documents = await self.load_antique_documents(data_path, max_documents)
        
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
    title="Antique Dataset Loading Service",
    description="Loads and processes the Antique dataset for representation services",
    version="1.0.0"
)

# Global service instance
loader_service = AntiqueLoaderService()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await loader_service.close()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Antique Dataset Loading Service",
        "version": "1.0.0",
        "description": "Loads and processes the Antique dataset",
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
        "service": "antique_loader_service",
        "available_services": list(REPRESENTATION_SERVICES.keys())
    }

@app.post("/load", response_model=LoadResponse)
async def load_and_index_documents(request: LoadRequest):
    """Load documents from Antique dataset and index them in representation services"""
    try:
        result = await loader_service.load_and_index(
            data_path=request.data_path,
            max_documents=request.max_documents,
            services=request.representation_services or ["tfidf"]
        )
        return result
    except Exception as e:
        logger.error(f"Error loading and indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loading error: {str(e)}")

@app.post("/load_documents")
async def load_documents_only(request: LoadRequest):
    """Load documents from Antique dataset without indexing"""
    try:
        documents = await loader_service.load_antique_documents(
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
    # This service runs on port 8003
    uvicorn.run(app, host="0.0.0.0", port=8003)
