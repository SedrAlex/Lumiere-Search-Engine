"""
RAR File Extractor and Database Storage Service
Extracts files from RAR archives in downloads folder and stores them in database
"""

import os
import rarfile
import sqlite3
import json
from pathlib import Path
import hashlib
from datetime import datetime
import tempfile
import shutil
import patoolib
import subprocess

class RarExtractorService:
    """Service to extract RAR files and store content in database"""
    
    def __init__(self, db_path="data/database/documents.db"):
        self.db_path = db_path
        self.downloads_path = Path.home() / "Downloads"
        self._init_database()
    
    def _init_database(self):
        """Initialize the rar_files table in database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create rar_files table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rar_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rar_filename TEXT NOT NULL,
                        rar_path TEXT NOT NULL,
                        rar_hash TEXT UNIQUE NOT NULL,
                        extracted_file_name TEXT NOT NULL,
                        extracted_file_path TEXT NOT NULL,
                        file_content TEXT,
                        file_size INTEGER,
                        file_type TEXT,
                        extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                conn.commit()
                print("✅ RAR files table initialized")
                
        except Exception as e:
            print(f"❌ Error initializing RAR files table: {e}")
    
    def _get_file_hash(self, file_path):
        """Get SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_type(self, filename):
        """Determine file type from extension"""
        ext = Path(filename).suffix.lower()
        
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.tsv'}
        if ext in text_extensions:
            return 'text'
        elif ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'}:
            return 'image'
        elif ext in {'.pdf'}:
            return 'pdf'
        elif ext in {'.doc', '.docx'}:
            return 'document'
        else:
            return 'binary'
    
    def _read_file_content(self, file_path, file_type):
        """Read file content based on type"""
        try:
            if file_type == 'text':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif file_type == 'binary':
                # For binary files, just store metadata
                return f"Binary file: {os.path.basename(file_path)}"
            else:
                # For other types, try to read as text with error handling
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 100000:  # Limit content size
                        content = content[:100000] + "... [content truncated]"
                    return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def extract_rar_files(self):
        """Extract all RAR files from downloads folder and store in database"""
        rar_files = list(self.downloads_path.glob("*.rar"))
        
        if not rar_files:
            print("No RAR files found in Downloads folder")
            return
        
        print(f"Found {len(rar_files)} RAR file(s) to process")
        
        for rar_path in rar_files:
            print(f"\n📦 Processing: {rar_path.name}")
            self._extract_single_rar(rar_path)
    
    def _extract_single_rar(self, rar_path):
        """Extract a single RAR file and store its contents"""
        try:
            # Get RAR file hash
            rar_hash = self._get_file_hash(rar_path)
            
            # Check if already processed
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM rar_files WHERE rar_hash = ?", (rar_hash,))
                if cursor.fetchone()[0] > 0:
                    print(f"⏭️  RAR file {rar_path.name} already processed")
                    return
            
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Try multiple extraction methods
                extracted = False
                
                # Method 1: Try rarfile library
                try:
                    print("  🔧 Trying rarfile library...")
                    with rarfile.RarFile(str(rar_path)) as rf:
                        rf.extractall(temp_dir)
                    extracted = True
                    print("  ✅ Extracted using rarfile")
                except Exception as e:
                    print(f"  ❌ rarfile failed: {e}")
                
                # Method 2: Try patoolib
                if not extracted:
                    try:
                        print("  🔧 Trying patoolib...")
                        patoolib.extract_archive(str(rar_path), outdir=temp_dir)
                        extracted = True
                        print("  ✅ Extracted using patoolib")
                    except Exception as e:
                        print(f"  ❌ patoolib failed: {e}")
                
                # Method 3: Try system unrar command if available
                if not extracted:
                    try:
                        print("  🔧 Trying system unrar command...")
                        result = subprocess.run([
                            'unrar', 'x', str(rar_path), temp_dir
                        ], capture_output=True, text=True, check=True)
                        extracted = True
                        print("  ✅ Extracted using system unrar")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        print(f"  ❌ system unrar failed: {e}")
                
                if not extracted:
                    print(f"  ❌ All extraction methods failed for {rar_path.name}")
                    return
                
                # Get list of extracted files
                extracted_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        extracted_files.append(Path(root) / file)
                
                print(f"📁 Extracted {len(extracted_files)} files")
                
                # Process each extracted file
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for extracted_file in extracted_files:
                        try:
                            relative_path = extracted_file.relative_to(temp_path)
                            file_size = extracted_file.stat().st_size
                            file_type = self._get_file_type(extracted_file.name)
                            
                            # Read file content
                            file_content = self._read_file_content(extracted_file, file_type)
                            
                            # Prepare metadata
                            metadata = {
                                "original_rar": rar_path.name,
                                "extraction_date": datetime.now().isoformat(),
                                "relative_path_in_rar": str(relative_path),
                                "file_size_bytes": file_size
                            }
                            
                            # Insert into database
                            cursor.execute("""
                                INSERT INTO rar_files (
                                    rar_filename, rar_path, rar_hash, extracted_file_name,
                                    extracted_file_path, file_content, file_size, file_type, metadata
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                rar_path.name,
                                str(rar_path),
                                rar_hash,
                                extracted_file.name,
                                str(relative_path),
                                file_content,
                                file_size,
                                file_type,
                                json.dumps(metadata)
                            ))
                            
                            print(f"  ✅ Stored: {extracted_file.name} ({file_type}, {file_size} bytes)")
                            
                        except Exception as e:
                            print(f"  ❌ Error processing {extracted_file.name}: {e}")
                    
                    conn.commit()
            
            print(f"✅ Successfully processed RAR file: {rar_path.name}")
            
        except Exception as e:
            print(f"❌ Error extracting RAR file {rar_path.name}: {e}")
    
    def get_extracted_files_info(self):
        """Get information about extracted files"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_files,
                        COUNT(DISTINCT rar_filename) as total_rar_files,
                        SUM(file_size) as total_size,
                        file_type,
                        COUNT(*) as count_by_type
                    FROM rar_files 
                    GROUP BY file_type
                """)
                
                results = cursor.fetchall()
                
                print("\n📊 Extracted Files Summary:")
                print("-" * 40)
                
                total_files = 0
                total_size = 0
                
                for row in results:
                    if len(row) == 5:  # Group by file_type
                        file_type, count = row[3], row[4]
                        print(f"  {file_type}: {count} files")
                        total_files += count
                    else:  # Total stats
                        total_files, total_rar_files, total_size = row[0], row[1], row[2] or 0
                
                # Get overall totals
                cursor.execute("SELECT COUNT(*), COUNT(DISTINCT rar_filename), SUM(file_size) FROM rar_files")
                total_files, total_rar_files, total_size = cursor.fetchone()
                total_size = total_size or 0
                
                print(f"\n📈 Overall Statistics:")
                print(f"  Total RAR files processed: {total_rar_files}")
                print(f"  Total extracted files: {total_files}")
                print(f"  Total size: {total_size:,} bytes ({total_size/(1024*1024):.2f} MB)")
                
                # Show recent files
                cursor.execute("""
                    SELECT rar_filename, extracted_file_name, file_type, file_size, extraction_date
                    FROM rar_files 
                    ORDER BY extraction_date DESC 
                    LIMIT 10
                """)
                
                recent_files = cursor.fetchall()
                
                if recent_files:
                    print(f"\n📋 Recent Files:")
                    print("-" * 60)
                    for row in recent_files:
                        rar_name, file_name, file_type, size, date = row
                        print(f"  {file_name} ({file_type}, {size} bytes) from {rar_name}")
                
        except Exception as e:
            print(f"❌ Error getting extracted files info: {e}")

def main():
    """Main function to run the RAR extractor"""
    print("🚀 Starting RAR File Extractor Service")
    print("=" * 50)
    
    extractor = RarExtractorService()
    
    # Extract RAR files
    extractor.extract_rar_files()
    
    # Show summary
    extractor.get_extracted_files_info()
    
    print("\n✅ RAR extraction completed!")

if __name__ == "__main__":
    main()
