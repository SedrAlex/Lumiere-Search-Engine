#!/usr/bin/env python3
"""
Script to inspect Quora database contents
"""

import sqlite3
from pathlib import Path

def main():
    """Main function to inspect Quora database"""
    db_path = "data/database/documents.db"
    
    if not Path(db_path).exists():
        print("❌ Database file not found!")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            print("🔍 Quora Database Inspection")
            print("=" * 40)
            
            # Check table structure
            tables = ['quora_docs', 'quora_queries', 'quora_qrels']
            
            for table in tables:
                print(f"\n📋 Table: {table}")
                print("-" * 30)
                
                # Get table info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                print("Columns:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
                
                # Get count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"Total records: {count:,}")
                
                # Show sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                samples = cursor.fetchall()
                if samples:
                    print("Sample data:")
                    for i, sample in enumerate(samples, 1):
                        print(f"  {i}. {sample}")
            
            # Show some interesting queries
            print(f"\n🔗 Query-Document Relationships")
            print("-" * 35)
            
            cursor.execute("""
                SELECT q.query_id, q.text as query_text, d.doc_id, d.text as doc_text, r.relevance
                FROM quora_qrels r
                JOIN quora_queries q ON r.query_id = q.query_id
                JOIN quora_docs d ON r.doc_id = d.doc_id
                LIMIT 5
            """)
            
            relationships = cursor.fetchall()
            for rel in relationships:
                query_id, query_text, doc_id, doc_text, relevance = rel
                print(f"\nQuery {query_id}: {query_text}")
                print(f"Doc {doc_id} (relevance: {relevance}): {doc_text[:100]}...")
            
            # Show relevance distribution
            print(f"\n📊 Relevance Score Distribution")
            print("-" * 35)
            cursor.execute("""
                SELECT relevance, COUNT(*) as count
                FROM quora_qrels
                GROUP BY relevance
                ORDER BY relevance
            """)
            
            for relevance, count in cursor.fetchall():
                print(f"  Relevance {relevance}: {count:,} pairs")
            
    except Exception as e:
        print(f"❌ Error inspecting database: {e}")

if __name__ == "__main__":
    main()
