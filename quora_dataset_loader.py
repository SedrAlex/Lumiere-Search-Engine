#!/usr/bin/env python3
"""
Quora Dataset Loader
Downloads Quora dataset from multiple sources (Hugging Face, BEIR) and stores in SQLite database
Similar to Antique implementation but for Quora question-answer pairs
"""

import asyncio
import logging
import sqlite3
import aiosqlite
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import ir_datasets
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraDatasetLoader:
    """Loads Quora datasets from various sources and stores them in SQLite database"""
    
    def __init__(self, db_path: str = "data/search_engine.db"):
        self.db_path = db_path
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Available Quora datasets
        self.available_sources = {
            'beir_quora': {
                'name': 'BEIR Quora',
                'description': 'Quora question-answer pairs from BEIR benchmark',
                'ir_datasets_id': 'beir/quora',
                'has_queries': True,
                'has_qrels': True,
                'doc_count_estimate': 522931
            },
            'huggingface_quora': {
                'name': 'Hugging Face Quora',
                'description': 'Original Quora Question Pairs dataset',
                'hf_dataset_id': 'quora',
                'has_queries': False,
                'has_qrels': False,
                'doc_count_estimate': 404290
            }
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with Quora-specific tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create documents table for Quora content
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        dataset_source TEXT NOT NULL,
                        question TEXT,
                        answer TEXT,
                        text TEXT,  -- Combined question + answer
                        processed_text TEXT,
                        question_id TEXT,
                        answer_id TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create queries table for BEIR queries
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT UNIQUE NOT NULL,
                        dataset_source TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create qrels table for relevance judgments
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_qrels (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        relevance INTEGER NOT NULL,
                        dataset_source TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(query_id, doc_id, dataset_source)
                    )
                """)
                
                # Create dataset info table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quora_dataset_info (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_source TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        document_count INTEGER DEFAULT 0,
                        query_count INTEGER DEFAULT 0,
                        qrels_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("✅ Quora database tables initialized")
                
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
            raise
    
    async def load_beir_quora(self, limit_docs: Optional[int] = None, limit_queries: Optional[int] = 100) -> Dict[str, Any]:
        """Load Quora dataset from BEIR via ir_datasets"""
        logger.info("Loading BEIR Quora dataset...")
        
        try:
            # Load the BEIR Quora dataset
            dataset = ir_datasets.load('beir/quora')
            
            # Load documents
            logger.info("Loading documents...")
            documents = []
            
            for i, doc in enumerate(tqdm(dataset.docs_iter(), desc="Loading documents")):
                if limit_docs and i >= limit_docs:
                    break
                
                # For BEIR Quora, documents are usually questions with answers
                documents.append({
                    'doc_id': doc.doc_id,
                    'question': getattr(doc, 'title', ''),
                    'answer': getattr(doc, 'text', ''),
                    'text': f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".strip(),
                    'metadata': {
                        'source': 'beir_quora',
                        'original_doc_id': doc.doc_id
                    }
                })
            
            logger.info(f"✅ Loaded {len(documents)} documents")
            
            # Load queries
            logger.info("Loading queries...")
            queries = []
            
            for i, query in enumerate(tqdm(dataset.queries_iter(), desc="Loading queries")):
                if limit_queries and i >= limit_queries:
                    break
                
                queries.append({
                    'query_id': query.query_id,
                    'query_text': query.text,
                    'metadata': {
                        'source': 'beir_quora',
                        'original_query_id': query.query_id
                    }
                })
            
            logger.info(f"✅ Loaded {len(queries)} queries")
            
            # Load relevance judgments
            logger.info("Loading relevance judgments...")
            qrels = []
            
            for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
                # Only include qrels for loaded queries
                if limit_queries:
                    loaded_query_ids = [q['query_id'] for q in queries]
                    if qrel.query_id not in loaded_query_ids:
                        continue
                
                qrels.append({
                    'query_id': qrel.query_id,
                    'doc_id': qrel.doc_id,
                    'relevance': qrel.relevance
                })
            
            logger.info(f"✅ Loaded {len(qrels)} relevance judgments")
            
            return {
                'dataset_source': 'beir_quora',
                'documents': documents,
                'queries': queries,
                'qrels': qrels,
                'stats': {
                    'total_documents': len(documents),
                    'total_queries': len(queries),
                    'total_qrels': len(qrels)
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading BEIR Quora dataset: {e}")
            raise
    
    async def load_huggingface_quora(self, limit_docs: Optional[int] = None) -> Dict[str, Any]:
        """Load Quora dataset from Hugging Face"""
        logger.info("Loading Hugging Face Quora dataset...")
        
        try:
            # Load the dataset with trust_remote_code=True
            logger.info("Downloading from Hugging Face...")
            dataset = load_dataset('quora', trust_remote_code=True)
            
            # Get the train split (typically the largest)
            train_data = dataset['train']
            
            if limit_docs:
                train_data = train_data.select(range(min(limit_docs, len(train_data))))
            
            documents = []
            
            logger.info(f"Processing {len(train_data)} question pairs...")
            
            for i, item in enumerate(tqdm(train_data, desc="Processing question pairs")):
                # Each item contains question pairs
                questions = item.get('questions', {})
                
                # Extract individual questions as documents
                if 'text' in questions:
                    for j, question_text in enumerate(questions['text']):
                        doc_id = f"hf_quora_{i}_{j}"
                        
                        documents.append({
                            'doc_id': doc_id,
                            'question': question_text,
                            'answer': '',  # HF Quora doesn't have answers
                            'text': question_text,
                            'metadata': {
                                'source': 'huggingface_quora',
                                'original_id': questions.get('id', [None])[j] if j < len(questions.get('id', [])) else None,
                                'is_duplicate': item.get('is_duplicate', False),
                                'pair_id': i
                            }
                        })
            
            logger.info(f"✅ Processed {len(documents)} documents from Hugging Face Quora")
            
            return {
                'dataset_source': 'huggingface_quora',
                'documents': documents,
                'queries': [],  # No queries in HF Quora
                'qrels': [],   # No qrels in HF Quora
                'stats': {
                    'total_documents': len(documents),
                    'total_queries': 0,
                    'total_qrels': 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading Hugging Face Quora dataset: {e}")
            # Fallback: try to load a simpler version or skip
            logger.info("Falling back to basic question pairs format...")
            
            try:
                # Try alternative loading approach
                dataset = load_dataset('glue', 'qqp', trust_remote_code=True)  # Quora Question Pairs from GLUE
                train_data = dataset['train']
                
                if limit_docs:
                    train_data = train_data.select(range(min(limit_docs, len(train_data))))
                
                documents = []
                
                for i, item in enumerate(tqdm(train_data, desc="Processing GLUE QQP")):
                    # Create documents for each question in the pair
                    question1 = item.get('question1', '')
                    question2 = item.get('question2', '')
                    
                    if question1:
                        documents.append({
                            'doc_id': f"glue_qqp_{i}_q1",
                            'question': question1,
                            'answer': '',
                            'text': question1,
                            'metadata': {
                                'source': 'glue_qqp',
                                'pair_id': i,
                                'is_duplicate': item.get('label', 0) == 1,
                                'question_type': 'question1'
                            }
                        })
                    
                    if question2:
                        documents.append({
                            'doc_id': f"glue_qqp_{i}_q2",
                            'question': question2,
                            'answer': '',
                            'text': question2,
                            'metadata': {
                                'source': 'glue_qqp',
                                'pair_id': i,
                                'is_duplicate': item.get('label', 0) == 1,
                                'question_type': 'question2'
                            }
                        })
                
                logger.info(f"✅ Loaded {len(documents)} documents from GLUE QQP as fallback")
                
                return {
                    'dataset_source': 'glue_qqp',
                    'documents': documents,
                    'queries': [],
                    'qrels': [],
                    'stats': {
                        'total_documents': len(documents),
                        'total_queries': 0,
                        'total_qrels': 0
                    }
                }
                
            except Exception as e2:
                logger.error(f"Error loading fallback dataset: {e2}")
                raise e
    
    async def store_dataset_in_db(self, dataset_data: Dict[str, Any]) -> bool:
        """Store dataset in SQLite database"""
        dataset_source = dataset_data['dataset_source']
        documents = dataset_data['documents']
        queries = dataset_data['queries']
        qrels = dataset_data['qrels']
        
        logger.info(f"Storing {dataset_source} dataset in database...")
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                # Store documents
                logger.info(f"Storing {len(documents)} documents...")
                doc_data = []
                for doc in documents:
                    doc_data.append((
                        doc['doc_id'],
                        dataset_source,
                        doc.get('question', ''),
                        doc.get('answer', ''),
                        doc['text'],
                        '',  # processed_text - will be filled later
                        doc.get('metadata', {}).get('original_doc_id', ''),
                        doc.get('metadata', {}).get('original_id', ''),
                        json.dumps(doc.get('metadata', {}))
                    ))
                
                await cursor.executemany("""
                    INSERT OR REPLACE INTO quora_documents (
                        doc_id, dataset_source, question, answer, text, processed_text,
                        question_id, answer_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, doc_data)
                
                # Store queries
                if queries:
                    logger.info(f"Storing {len(queries)} queries...")
                    query_data = []
                    for query in queries:
                        query_data.append((
                            query['query_id'],
                            dataset_source,
                            query['query_text'],
                            json.dumps(query.get('metadata', {}))
                        ))
                    
                    await cursor.executemany("""
                        INSERT OR REPLACE INTO quora_queries (
                            query_id, dataset_source, query_text, metadata
                        ) VALUES (?, ?, ?, ?)
                    """, query_data)
                
                # Store qrels
                if qrels:
                    logger.info(f"Storing {len(qrels)} relevance judgments...")
                    qrel_data = []
                    for qrel in qrels:
                        qrel_data.append((
                            qrel['query_id'],
                            qrel['doc_id'],
                            qrel['relevance'],
                            dataset_source
                        ))
                    
                    await cursor.executemany("""
                        INSERT OR REPLACE INTO quora_qrels (
                            query_id, doc_id, relevance, dataset_source
                        ) VALUES (?, ?, ?, ?)
                    """, qrel_data)
                
                # Store dataset info
                await cursor.execute("""
                    INSERT OR REPLACE INTO quora_dataset_info (
                        dataset_source, name, description, document_count, query_count, qrels_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    dataset_source,
                    self.available_sources.get(dataset_source, {}).get('name', dataset_source),
                    self.available_sources.get(dataset_source, {}).get('description', ''),
                    len(documents),
                    len(queries),
                    len(qrels)
                ))
                
                await conn.commit()
                logger.info(f"✅ Successfully stored {dataset_source} dataset in database")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error storing dataset in database: {e}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.cursor()
                
                # Get document counts by source
                await cursor.execute("""
                    SELECT dataset_source, COUNT(*) as count 
                    FROM quora_documents 
                    GROUP BY dataset_source
                """)
                doc_stats = dict(await cursor.fetchall())
                
                # Get query counts by source
                await cursor.execute("""
                    SELECT dataset_source, COUNT(*) as count 
                    FROM quora_queries 
                    GROUP BY dataset_source
                """)
                query_stats = dict(await cursor.fetchall())
                
                # Get qrels counts by source
                await cursor.execute("""
                    SELECT dataset_source, COUNT(*) as count 
                    FROM quora_qrels 
                    GROUP BY dataset_source
                """)
                qrels_stats = dict(await cursor.fetchall())
                
                # Get total counts
                await cursor.execute("SELECT COUNT(*) FROM quora_documents")
                total_docs = (await cursor.fetchone())[0]
                
                await cursor.execute("SELECT COUNT(*) FROM quora_queries")
                total_queries = (await cursor.fetchone())[0]
                
                await cursor.execute("SELECT COUNT(*) FROM quora_qrels")
                total_qrels = (await cursor.fetchone())[0]
                
                return {
                    'total_documents': total_docs,
                    'total_queries': total_queries,
                    'total_qrels': total_qrels,
                    'documents_by_source': doc_stats,
                    'queries_by_source': query_stats,
                    'qrels_by_source': qrels_stats,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    async def load_all_quora_datasets(self, 
                                     load_beir: bool = True, 
                                     load_hf: bool = True,
                                     limit_docs: Optional[int] = None,
                                     limit_queries: Optional[int] = 100) -> Dict[str, Any]:
        """Load all available Quora datasets"""
        results = {}
        
        if load_beir:
            try:
                logger.info("Loading BEIR Quora dataset...")
                beir_data = await self.load_beir_quora(limit_docs=limit_docs, limit_queries=limit_queries)
                success = await self.store_dataset_in_db(beir_data)
                results['beir_quora'] = {
                    'success': success,
                    'stats': beir_data['stats']
                }
            except Exception as e:
                logger.error(f"Failed to load BEIR Quora: {e}")
                results['beir_quora'] = {'success': False, 'error': str(e)}
        
        if load_hf:
            try:
                logger.info("Loading Hugging Face Quora dataset...")
                hf_data = await self.load_huggingface_quora(limit_docs=limit_docs)
                success = await self.store_dataset_in_db(hf_data)
                results['huggingface_quora'] = {
                    'success': success,
                    'stats': hf_data['stats']
                }
            except Exception as e:
                logger.error(f"Failed to load Hugging Face Quora: {e}")
                results['huggingface_quora'] = {'success': False, 'error': str(e)}
        
        # Get final database stats
        db_stats = await self.get_database_stats()
        results['final_database_stats'] = db_stats
        
        return results

async def main():
    """Main function to load Quora datasets"""
    loader = QuoraDatasetLoader()
    
    print("🎯 Quora Dataset Loader")
    print("=" * 60)
    
    print("Available Quora datasets:")
    for source_id, info in loader.available_sources.items():
        print(f"- {source_id}: {info['name']} (~{info['doc_count_estimate']:,} docs)")
    
    print("\n🚀 Loading Quora datasets...")
    
    # Load datasets with reasonable limits for testing
    results = await loader.load_all_quora_datasets(
        load_beir=True,
        load_hf=True,
        limit_docs=10000,  # Limit for testing
        limit_queries=100
    )
    
    print("\n📊 Loading Results:")
    print("=" * 60)
    
    for dataset_name, result in results.items():
        if dataset_name == 'final_database_stats':
            continue
            
        if result['success']:
            stats = result['stats']
            print(f"✅ {dataset_name}:")
            print(f"   Documents: {stats['total_documents']:,}")
            print(f"   Queries: {stats['total_queries']:,}")
            print(f"   QRels: {stats['total_qrels']:,}")
        else:
            print(f"❌ {dataset_name}: {result.get('error', 'Unknown error')}")
    
    # Print final database stats
    if 'final_database_stats' in results:
        final_stats = results['final_database_stats']
        print(f"\n📈 Final Database Statistics:")
        print(f"Total Documents: {final_stats['total_documents']:,}")
        print(f"Total Queries: {final_stats['total_queries']:,}")
        print(f"Total QRels: {final_stats['total_qrels']:,}")
        print(f"Database: {final_stats['database_path']}")
        
        if 'documents_by_source' in final_stats:
            print("\nDocuments by source:")
            for source, count in final_stats['documents_by_source'].items():
                print(f"  {source}: {count:,}")

if __name__ == "__main__":
    asyncio.run(main())
