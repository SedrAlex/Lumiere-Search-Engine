"""
Database Service for Document Storage
Using SQLite for local development and MongoDB option for production
"""

import sqlite3
import json
import asyncio
import aiosqlite
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# MongoDB imports (optional)
try:
    from pymongo import MongoClient
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Document model
from services.data_preprocessing.preprocessor import Document


class DatabaseService:
    """Service for storing and retrieving documents from database"""
    
    def __init__(self, db_type: str = "sqlite", connection_string: str = None):
        self.db_type = db_type
        self.connection_string = connection_string
        self.data_dir = "data/database"
        os.makedirs(self.data_dir, exist_ok=True)
        
        if db_type == "sqlite":
            self.db_path = os.path.join(self.data_dir, "documents.db")
            self._init_sqlite()
        elif db_type == "mongodb" and MONGODB_AVAILABLE:
            self._init_mongodb()
        else:
            print("⚠️ Database type not supported, falling back to SQLite")
            self.db_type = "sqlite"
            self.db_path = os.path.join(self.data_dir, "documents.db")
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        dataset_name TEXT NOT NULL,
                        title TEXT,
                        text TEXT,
                        processed_text TEXT,
                        tokens TEXT,
                        stemmed_tokens TEXT,
                        lemmatized_tokens TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create datasets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        document_count INTEGER DEFAULT 0,
                        indexed BOOLEAN DEFAULT FALSE,
                        representations_available TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indices table for storing vector files
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS indices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_name TEXT NOT NULL,
                        representation_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(dataset_name, representation_type)
                    )
                """)
                
                conn.commit()
                print("✅ SQLite database initialized")
                
        except Exception as e:
            print(f"❌ Error initializing SQLite database: {e}")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            self.mongo_client = AsyncIOMotorClient(
                self.connection_string or "mongodb://localhost:27017/"
            )
            self.db = self.mongo_client.search_engine
            print("✅ MongoDB connection initialized")
        except Exception as e:
            print(f"❌ Error initializing MongoDB: {e}")
    
    async def store_documents(self, documents: List[Document], dataset_name: str) -> bool:
        """Store documents in database"""
        if self.db_type == "sqlite":
            return await self._store_documents_sqlite(documents, dataset_name)
        elif self.db_type == "mongodb":
            return await self._store_documents_mongodb(documents, dataset_name)
        return False
    
    async def _store_documents_sqlite(self, documents: List[Document], dataset_name: str) -> bool:
        """Store documents in SQLite database"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                # Prepare document data
                doc_data = []
                for doc in documents:
                    doc_data.append((
                        doc.doc_id,
                        dataset_name,
                        doc.title,
                        doc.text,
                        doc.processed_text,
                        json.dumps(doc.tokens),
                        json.dumps(doc.stemmed_tokens),
                        json.dumps(doc.lemmatized_tokens),
                        json.dumps(doc.metadata)
                    ))
                
                # Insert documents (with conflict resolution)
                await cursor.executemany("""
                    INSERT OR REPLACE INTO documents (
                        doc_id, dataset_name, title, text, processed_text,
                        tokens, stemmed_tokens, lemmatized_tokens, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, doc_data)
                
                # Update dataset info
                await cursor.execute("""
                    INSERT OR REPLACE INTO datasets (name, description, document_count)
                    VALUES (?, ?, ?)
                """, (dataset_name, f"Dataset: {dataset_name}", len(documents)))
                
                await conn.commit()
                print(f"✅ Stored {len(documents)} documents in SQLite")
                return True
                
        except Exception as e:
            print(f"❌ Error storing documents in SQLite: {e}")
            return False
    
    async def _store_documents_mongodb(self, documents: List[Document], dataset_name: str) -> bool:
        """Store documents in MongoDB"""
        try:
            collection = self.db.documents
            
            # Prepare document data
            doc_data = []
            for doc in documents:
                doc_dict = {
                    "doc_id": doc.doc_id,
                    "dataset_name": dataset_name,
                    "title": doc.title,
                    "text": doc.text,
                    "processed_text": doc.processed_text,
                    "tokens": doc.tokens,
                    "stemmed_tokens": doc.stemmed_tokens,
                    "lemmatized_tokens": doc.lemmatized_tokens,
                    "metadata": doc.metadata
                }
                doc_data.append(doc_dict)
            
            # Insert documents
            await collection.insert_many(doc_data)
            
            # Update dataset collection
            datasets_collection = self.db.datasets
            await datasets_collection.update_one(
                {"name": dataset_name},
                {
                    "$set": {
                        "name": dataset_name,
                        "description": f"Dataset: {dataset_name}",
                        "document_count": len(documents)
                    }
                },
                upsert=True
            )
            
            print(f"✅ Stored {len(documents)} documents in MongoDB")
            return True
            
        except Exception as e:
            print(f"❌ Error storing documents in MongoDB: {e}")
            return False
    
    async def get_documents(self, dataset_name: str, limit: int = None) -> List[Document]:
        """Retrieve documents from database"""
        if self.db_type == "sqlite":
            return await self._get_documents_sqlite(dataset_name, limit)
        elif self.db_type == "mongodb":
            return await self._get_documents_mongodb(dataset_name, limit)
        return []
    
    async def _get_documents_sqlite(self, dataset_name: str, limit: int = None) -> List[Document]:
        """Retrieve documents from SQLite"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                query = "SELECT * FROM documents WHERE dataset_name = ?"
                params = [dataset_name]
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                
                # Get column names
                await cursor.execute("PRAGMA table_info(documents)")
                columns = [col[1] for col in await cursor.fetchall()]
                
                documents = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    doc = Document(
                        doc_id=row_dict['doc_id'],
                        title=row_dict['title'] or "",
                        text=row_dict['text'] or "",
                        metadata=json.loads(row_dict['metadata'] or '{}')
                    )
                    
                    doc.processed_text = row_dict['processed_text'] or ""
                    doc.tokens = json.loads(row_dict['tokens'] or '[]')
                    doc.stemmed_tokens = json.loads(row_dict['stemmed_tokens'] or '[]')
                    doc.lemmatized_tokens = json.loads(row_dict['lemmatized_tokens'] or '[]')
                    
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            print(f"❌ Error retrieving documents from SQLite: {e}")
            return []
    
    async def _get_documents_mongodb(self, dataset_name: str, limit: int = None) -> List[Document]:
        """Retrieve documents from MongoDB"""
        try:
            collection = self.db.documents
            
            query = {"dataset_name": dataset_name}
            cursor = collection.find(query)
            
            if limit:
                cursor = cursor.limit(limit)
            
            documents = []
            async for doc_dict in cursor:
                doc = Document(
                    doc_id=doc_dict['doc_id'],
                    title=doc_dict.get('title', ''),
                    text=doc_dict.get('text', ''),
                    metadata=doc_dict.get('metadata', {})
                )
                
                doc.processed_text = doc_dict.get('processed_text', '')
                doc.tokens = doc_dict.get('tokens', [])
                doc.stemmed_tokens = doc_dict.get('stemmed_tokens', [])
                doc.lemmatized_tokens = doc_dict.get('lemmatized_tokens', [])
                
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Error retrieving documents from MongoDB: {e}")
            return []
    
    async def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset information"""
        if self.db_type == "sqlite":
            return await self._get_dataset_info_sqlite(dataset_name)
        elif self.db_type == "mongodb":
            return await self._get_dataset_info_mongodb(dataset_name)
        return {}
    
    async def _get_dataset_info_sqlite(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset info from SQLite"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                await cursor.execute("""
                    SELECT * FROM datasets WHERE name = ?
                """, (dataset_name,))
                
                row = await cursor.fetchone()
                if row:
                    # Get column names
                    await cursor.execute("PRAGMA table_info(datasets)")
                    columns = [col[1] for col in await cursor.fetchall()]
                    
                    return dict(zip(columns, row))
                
                return {}
                
        except Exception as e:
            print(f"❌ Error getting dataset info from SQLite: {e}")
            return {}
    
    async def _get_dataset_info_mongodb(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset info from MongoDB"""
        try:
            collection = self.db.datasets
            doc = await collection.find_one({"name": dataset_name})
            return doc or {}
        except Exception as e:
            print(f"❌ Error getting dataset info from MongoDB: {e}")
            return {}
    
    async def store_index_metadata(self, dataset_name: str, representation_type: str, file_path: str, file_size: int):
        """Store index file metadata"""
        if self.db_type == "sqlite":
            await self._store_index_metadata_sqlite(dataset_name, representation_type, file_path, file_size)
    
    async def _store_index_metadata_sqlite(self, dataset_name: str, representation_type: str, file_path: str, file_size: int):
        """Store index metadata in SQLite"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                await cursor.execute("""
                    INSERT OR REPLACE INTO indices (dataset_name, representation_type, file_path, file_size)
                    VALUES (?, ?, ?, ?)
                """, (dataset_name, representation_type, file_path, file_size))
                
                await conn.commit()
                
        except Exception as e:
            print(f"❌ Error storing index metadata: {e}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.db_type == "sqlite":
            return await self._get_database_stats_sqlite()
        elif self.db_type == "mongodb":
            return await self._get_database_stats_mongodb()
        return {}
    
    async def _get_database_stats_sqlite(self) -> Dict[str, Any]:
        """Get SQLite database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                # Get total documents
                await cursor.execute("SELECT COUNT(*) FROM documents")
                total_docs = (await cursor.fetchone())[0]
                
                # Get total datasets
                await cursor.execute("SELECT COUNT(*) FROM datasets")
                total_datasets = (await cursor.fetchone())[0]
                
                # Get total indices
                await cursor.execute("SELECT COUNT(*) FROM indices")
                total_indices = (await cursor.fetchone())[0]
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    "database_type": "SQLite",
                    "total_documents": total_docs,
                    "total_datasets": total_datasets,
                    "total_indices": total_indices,
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            print(f"❌ Error getting SQLite stats: {e}")
            return {"error": str(e)}
    
    async def _get_database_stats_mongodb(self) -> Dict[str, Any]:
        """Get MongoDB database statistics"""
        try:
            # Get collection stats
            docs_count = await self.db.documents.count_documents({})
            datasets_count = await self.db.datasets.count_documents({})
            
            return {
                "database_type": "MongoDB",
                "total_documents": docs_count,
                "total_datasets": datasets_count,
                "collections": ["documents", "datasets"]
            }
            
        except Exception as e:
            print(f"❌ Error getting MongoDB stats: {e}")
            return {"error": str(e)}
