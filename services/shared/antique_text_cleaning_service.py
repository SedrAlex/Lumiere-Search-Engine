#!/usr/bin/env python3
"""
Antique Dataset Text Cleaning Service
Dedicated service for cleaning and preprocessing text from the Antique dataset.
This service applies the enhanced text cleaning methods and saves results to database.
"""

import asyncio
import sqlite3
import aiosqlite
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Import the enhanced text cleaning service
from services.shared.enhanced_text_cleaning_service import EnhancedTextCleaningService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiqueTextCleaningService:
    """
    Service for cleaning Antique dataset text and managing cleaned data in database.
    """
    
    def __init__(self, db_path: str = None, enable_spell_check: bool = True):
        """
        Initialize the Antique text cleaning service.
        
        Args:
            db_path: Path to SQLite database
            enable_spell_check: Whether to enable spell checking
        """
        self.db_path = db_path or "data/search_engine.db"
        self.enhanced_cleaner = EnhancedTextCleaningService(
            language='english', 
            enable_spell_check=enable_spell_check
        )
        
        # Ensure database path exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Antique text cleaning service initialized with database: {self.db_path}")
    
    async def create_cleaned_table(self):
        """Create table for storing cleaned antique documents."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS antique_cleaned_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        original_content TEXT,
                        cleaned_content TEXT,
                        cleaned_for_embedding TEXT,
                        cleaning_method TEXT,
                        cleaning_stats TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster lookups
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_antique_cleaned_doc_id 
                    ON antique_cleaned_documents(doc_id)
                """)
                
                await conn.commit()
                logger.info("✅ Created antique_cleaned_documents table")
                
        except Exception as e:
            logger.error(f"❌ Error creating cleaned documents table: {e}")
            raise
    
    async def get_document_count(self) -> int:
        """Get total number of documents in antique dataset."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE dataset_name = 'antique'"
                )
                result = await cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"❌ Error getting document count: {e}")
            return 0
    
    async def get_uncleaned_documents(self, batch_size: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get batch of uncleaned documents from the antique dataset.
        
        Args:
            batch_size: Number of documents to fetch
            offset: Offset for pagination
            
        Returns:
            List of document dictionaries
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Check if cleaned documents table exists
                cursor = await conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='antique_cleaned_documents'
                """)
                table_exists = await cursor.fetchone()
                
                if table_exists:
                    # Use LEFT JOIN to find uncleaned documents
                    cursor = await conn.execute("""
                        SELECT d.doc_id, d.content, d.title, d.processed_content
                        FROM documents d
                        LEFT JOIN antique_cleaned_documents c ON d.doc_id = c.doc_id
                        WHERE d.dataset_name = 'antique' AND c.doc_id IS NULL
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
                else:
                    # Table doesn't exist, get all antique documents
                    cursor = await conn.execute("""
                        SELECT doc_id, content, title, processed_content
                        FROM documents
                        WHERE dataset_name = 'antique'
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
                
                rows = await cursor.fetchall()
                
                documents = []
                for row in rows:
                    doc_id, content, title, processed_content = row
                    
                    # Combine title and content for cleaning
                    full_text = ""
                    if title:
                        full_text += title + " "
                    if content:
                        full_text += content
                    
                    documents.append({
                        'doc_id': doc_id,
                        'original_content': full_text.strip(),
                        'title': title,
                        'content': content,
                        'processed_content': processed_content
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"❌ Error fetching uncleaned documents: {e}")
            return []
    
    async def clean_document_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a batch of documents using enhanced text cleaning.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of cleaned document results
        """
        cleaned_results = []
        
        for doc in documents:
            try:
                original_text = doc['original_content']
                
                if not original_text or not original_text.strip():
                    logger.warning(f"Empty content for document {doc['doc_id']}")
                    continue
                
                # Clean for TF-IDF (aggressive cleaning)
                cleaned_tfidf = self.enhanced_cleaner.preprocess_for_tfidf(original_text)
                
                # Clean for embeddings (preserve more structure)
                cleaned_embedding = self.enhanced_cleaner.preprocess_for_embedding(original_text)
                
                # Get cleaning statistics
                stats = self.enhanced_cleaner.get_preprocessing_statistics(
                    original_text, cleaned_tfidf
                )
                
                cleaned_result = {
                    'doc_id': doc['doc_id'],
                    'original_content': original_text,
                    'cleaned_content': cleaned_tfidf,
                    'cleaned_for_embedding': cleaned_embedding,
                    'cleaning_method': 'enhanced_tfidf_and_embedding',
                    'cleaning_stats': json.dumps(stats)
                }
                
                cleaned_results.append(cleaned_result)
                
            except Exception as e:
                logger.error(f"❌ Error cleaning document {doc.get('doc_id', 'unknown')}: {e}")
                continue
        
        return cleaned_results
    
    async def save_cleaned_documents(self, cleaned_docs: List[Dict[str, Any]]) -> bool:
        """
        Save cleaned documents to database.
        
        Args:
            cleaned_docs: List of cleaned document dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Prepare data for insertion
                doc_data = []
                for doc in cleaned_docs:
                    doc_data.append((
                        doc['doc_id'],
                        doc['original_content'],
                        doc['cleaned_content'],
                        doc['cleaned_for_embedding'],
                        doc['cleaning_method'],
                        doc['cleaning_stats']
                    ))
                
                # Insert cleaned documents
                await conn.executemany("""
                    INSERT OR REPLACE INTO antique_cleaned_documents (
                        doc_id, original_content, cleaned_content, 
                        cleaned_for_embedding, cleaning_method, cleaning_stats
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, doc_data)
                
                await conn.commit()
                logger.info(f"✅ Saved {len(cleaned_docs)} cleaned documents to database")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving cleaned documents: {e}")
            return False
    
    async def process_all_antique_documents(self, batch_size: int = 1000, max_documents: Optional[int] = None):
        """
        Process all antique documents in batches.
        
        Args:
            batch_size: Size of processing batches
            max_documents: Maximum number of documents to process (None for all)
        """
        await self.create_cleaned_table()
        
        total_docs = await self.get_document_count()
        logger.info(f"📊 Total antique documents in database: {total_docs:,}")
        
        if max_documents:
            total_docs = min(total_docs, max_documents)
            logger.info(f"🎯 Processing limited to {max_documents:,} documents")
        
        processed_count = 0
        offset = 0
        
        while processed_count < total_docs:
            current_batch_size = min(batch_size, total_docs - processed_count)
            
            logger.info(f"🔄 Processing batch {offset//batch_size + 1}: documents {offset + 1} to {offset + current_batch_size}")
            
            # Get batch of uncleaned documents
            documents = await self.get_uncleaned_documents(current_batch_size, offset)
            
            if not documents:
                logger.info("✅ No more uncleaned documents found")
                break
            
            # Clean the batch
            cleaned_docs = await self.clean_document_batch(documents)
            
            if cleaned_docs:
                # Save cleaned documents
                success = await self.save_cleaned_documents(cleaned_docs)
                if success:
                    processed_count += len(cleaned_docs)
                    logger.info(f"✅ Processed {processed_count:,}/{total_docs:,} documents ({processed_count/total_docs*100:.1f}%)")
                else:
                    logger.error("❌ Failed to save batch, stopping processing")
                    break
            
            offset += current_batch_size
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        logger.info(f"🎉 Text cleaning completed! Processed {processed_count:,} documents")
        
        # Get final statistics
        await self.get_cleaning_statistics()
    
    async def get_cleaning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the text cleaning process."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Get total counts
                cursor = await conn.execute("SELECT COUNT(*) FROM documents WHERE dataset_name = 'antique'")
                total_docs = (await cursor.fetchone())[0]
                
                # Check if antique_cleaned_documents table exists
                cursor = await conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='antique_cleaned_documents'
                """)
                table_exists = await cursor.fetchone()
                
                if table_exists:
                    cursor = await conn.execute("SELECT COUNT(*) FROM antique_cleaned_documents")
                    cleaned_docs = (await cursor.fetchone())[0]
                else:
                    cleaned_docs = 0
                
                # Get average cleaning statistics only if table exists
                avg_stats = {}
                if table_exists:
                    cursor = await conn.execute("""
                        SELECT cleaning_stats FROM antique_cleaned_documents 
                        WHERE cleaning_stats IS NOT NULL 
                        LIMIT 1000
                    """)
                    
                    stats_rows = await cursor.fetchall()
                    
                    # Parse and aggregate statistics
                    total_stats = {
                        'original_length': 0,
                        'processed_length': 0,
                        'original_tokens': 0,
                        'processed_tokens': 0
                    }
                    
                    valid_stats_count = 0
                    for row in stats_rows:
                        try:
                            stats = json.loads(row[0])
                            for key in total_stats:
                                if key in stats:
                                    total_stats[key] += stats[key]
                            valid_stats_count += 1
                        except:
                            continue
                    
                    # Calculate averages
                    if valid_stats_count > 0:
                        for key, value in total_stats.items():
                            avg_stats[f'avg_{key}'] = value / valid_stats_count
                
                result = {
                    'total_documents': total_docs,
                    'cleaned_documents': cleaned_docs,
                    'completion_percentage': (cleaned_docs / total_docs * 100) if total_docs > 0 else 0,
                    'remaining_documents': total_docs - cleaned_docs,
                    **avg_stats
                }
                
                logger.info("📊 Cleaning Statistics:")
                logger.info(f"   Total documents: {result['total_documents']:,}")
                logger.info(f"   Cleaned documents: {result['cleaned_documents']:,}")
                logger.info(f"   Completion: {result['completion_percentage']:.1f}%")
                logger.info(f"   Remaining: {result['remaining_documents']:,}")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Error getting cleaning statistics: {e}")
            return {}
    
    async def get_cleaned_documents_for_embedding(self, batch_size: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get cleaned documents ready for embedding generation.
        
        Args:
            batch_size: Number of documents to fetch
            offset: Offset for pagination
            
        Returns:
            List of cleaned document dictionaries
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Check if cleaned documents table exists
                cursor = await conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='antique_cleaned_documents'
                """)
                table_exists = await cursor.fetchone()
                
                if not table_exists:
                    logger.warning("antique_cleaned_documents table does not exist. Run text cleaning first.")
                    return []
                
                cursor = await conn.execute("""
                    SELECT doc_id, cleaned_for_embedding, original_content
                    FROM antique_cleaned_documents
                    ORDER BY doc_id
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))
                
                rows = await cursor.fetchall()
                
                documents = []
                for row in rows:
                    doc_id, cleaned_text, original_text = row
                    documents.append({
                        'doc_id': doc_id,
                        'text': cleaned_text,
                        'original_text': original_text
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"❌ Error fetching cleaned documents for embedding: {e}")
            return []
    
    async def export_cleaned_data_for_colab(self, output_file: str = "antique_cleaned_data.json", 
                                          max_documents: Optional[int] = None) -> str:
        """
        Export cleaned data in a format suitable for Colab upload.
        
        Args:
            output_file: Output file name
            max_documents: Maximum number of documents to export
            
        Returns:
            Path to the exported file
        """
        try:
            output_path = Path("data") / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            exported_docs = []
            batch_size = 1000
            offset = 0
            total_exported = 0
            
            logger.info(f"📤 Exporting cleaned data to {output_path}")
            
            while True:
                # Get batch of cleaned documents
                documents = await self.get_cleaned_documents_for_embedding(batch_size, offset)
                
                if not documents:
                    break
                
                for doc in documents:
                    if max_documents and total_exported >= max_documents:
                        break
                    
                    exported_docs.append({
                        'id': doc['doc_id'],
                        'text': doc['text'],
                        'original_text': doc['original_text']
                    })
                    total_exported += 1
                
                if max_documents and total_exported >= max_documents:
                    break
                
                offset += batch_size
                
                if len(documents) < batch_size:
                    break
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset': 'antique',
                    'total_documents': total_exported,
                    'export_timestamp': datetime.now().isoformat(),
                    'cleaning_method': 'enhanced_tfidf_and_embedding',
                    'documents': exported_docs
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Exported {total_exported:,} cleaned documents to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error exporting cleaned data: {e}")
            raise

# Factory function
def create_antique_cleaning_service(db_path: str = None, enable_spell_check: bool = True) -> AntiqueTextCleaningService:
    """
    Factory function to create antique text cleaning service.
    
    Args:
        db_path: Path to SQLite database
        enable_spell_check: Whether to enable spell checking
        
    Returns:
        AntiqueTextCleaningService instance
    """
    return AntiqueTextCleaningService(db_path=db_path, enable_spell_check=enable_spell_check)

# CLI script for running the cleaning process
async def main():
    """Main function for running text cleaning from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Antique dataset text")
    parser.add_argument("--db-path", help="Path to SQLite database", default="data/search_engine.db")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max-docs", type=int, help="Maximum documents to process")
    parser.add_argument("--export-only", action="store_true", help="Only export cleaned data")
    parser.add_argument("--export-file", default="antique_cleaned_data.json", help="Export file name")
    parser.add_argument("--no-spell-check", action="store_true", help="Disable spell checking")
    
    args = parser.parse_args()
    
    # Create service
    service = AntiqueTextCleaningService(
        db_path=args.db_path,
        enable_spell_check=not args.no_spell_check
    )
    
    if args.export_only:
        # Only export cleaned data
        await service.export_cleaned_data_for_colab(args.export_file, args.max_docs)
    else:
        # Process documents and then export
        await service.process_all_antique_documents(args.batch_size, args.max_docs)
        await service.export_cleaned_data_for_colab(args.export_file, args.max_docs)

if __name__ == "__main__":
    asyncio.run(main())
