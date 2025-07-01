#!/usr/bin/env python3
"""
Script to check what tables exist in the database and show sample data
"""

import sqlite3
from pathlib import Path

def main():
    """Check database for Antique and Quora tables"""
    db_path = "data/database/documents.db"
    
    if not Path(db_path).exists():
        print("❌ Database file not found!")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            print("🔍 Database Table Check")
            print("=" * 50)
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print("📋 All Tables in Database:")
            print("-" * 30)
            for table in tables:
                print(f"  - {table[0]}")
            
            print("\n" + "=" * 50)
            
            # Check for Quora tables
            quora_tables = [t[0] for t in tables if 'quora' in t[0].lower()]
            print(f"\n🔍 QUORA Tables Found: {len(quora_tables)}")
            print("-" * 30)
            
            if quora_tables:
                for table in quora_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  ✅ {table}: {count:,} records")
                    
                    # Show sample data
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    samples = cursor.fetchall()
                    for i, sample in enumerate(samples, 1):
                        print(f"    Sample {i}: {sample}")
                    print()
            else:
                print("  ❌ No Quora tables found")
            
            # Check for Antique tables
            antique_tables = [t[0] for t in tables if 'antique' in t[0].lower()]
            print(f"\n🏛️ ANTIQUE Tables Found: {len(antique_tables)}")
            print("-" * 30)
            
            if antique_tables:
                for table in antique_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  ✅ {table}: {count:,} records")
                    
                    # Show sample data
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    samples = cursor.fetchall()
                    for i, sample in enumerate(samples, 1):
                        print(f"    Sample {i}: {sample}")
                    print()
            else:
                print("  ❌ No Antique tables found")
            
            # Check for generic document tables that might contain either dataset
            generic_tables = [t[0] for t in tables if t[0] in ['documents', 'datasets']]
            if generic_tables:
                print(f"\n📄 GENERIC Document Tables Found: {len(generic_tables)}")
                print("-" * 40)
                
                for table in generic_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  ✅ {table}: {count:,} records")
                    
                    # For documents table, check dataset_name if it exists
                    if table == 'documents':
                        try:
                            cursor.execute("SELECT DISTINCT dataset_name, COUNT(*) FROM documents GROUP BY dataset_name")
                            datasets = cursor.fetchall()
                            if datasets:
                                print("    Datasets in documents table:")
                                for dataset_name, doc_count in datasets:
                                    print(f"      - {dataset_name}: {doc_count:,} documents")
                        except:
                            print("    (No dataset_name column)")
                    
                    # Show sample data
                    cursor.execute(f"SELECT * FROM {table} LIMIT 2")
                    samples = cursor.fetchall()
                    for i, sample in enumerate(samples, 1):
                        # Truncate long text fields for readability
                        truncated_sample = []
                        for field in sample:
                            if isinstance(field, str) and len(field) > 100:
                                truncated_sample.append(field[:100] + "...")
                            else:
                                truncated_sample.append(field)
                        print(f"    Sample {i}: {tuple(truncated_sample)}")
                    print()
            
            # Check other tables
            other_tables = [t[0] for t in tables if t[0] not in quora_tables + antique_tables + generic_tables]
            if other_tables:
                print(f"\n📊 OTHER Tables Found: {len(other_tables)}")
                print("-" * 25)
                for table in other_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  - {table}: {count:,} records")
            
            print("\n" + "=" * 50)
            print("✅ Database check completed!")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    main()
