#!/usr/bin/env python3
"""
CodeSearchNet Dataset Loader with Advanced Preprocessing
Loads CodeSearchNet from Hugging Face and applies advanced text normalization
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import sqlite3
from pathlib import Path
import json
import re
from datasets import load_dataset
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeDocument:
    """Structured representation of a code document"""
    doc_id: str
    title: str
    content: str
    code: str
    language: str
    url: str
    metadata: Dict[str, Any]

class AdvancedTextProcessor:
    """Advanced text preprocessing with normalization capabilities"""
    
    def __init__(self):
        # Country/Location normalization mappings
        self.country_normalizations = {
            'usa': 'united states',
            'us': 'united states',
            'america': 'united states',
            'uk': 'united kingdom',
            'britain': 'united kingdom',
            'eu': 'european union',
            'uae': 'united arab emirates',
            'ussr': 'soviet union',
            'drc': 'democratic republic of congo',
        }
        
        # Common abbreviation expansions
        self.abbreviation_normalizations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'sql': 'structured query language',
            'html': 'hypertext markup language',
            'css': 'cascading style sheets',
            'js': 'javascript',
            'py': 'python',
            'cpp': 'c plus plus',
            'db': 'database',
            'os': 'operating system',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'url': 'uniform resource locator',
            'uri': 'uniform resource identifier',
            'json': 'javascript object notation',
            'xml': 'extensible markup language',
            'yaml': 'yaml ain\'t markup language',
            'csv': 'comma separated values',
            'pdf': 'portable document format',
            'ide': 'integrated development environment',
            'sdk': 'software development kit',
            'cli': 'command line interface',
            'gui': 'graphical user interface',
            'orm': 'object relational mapping',
            'crud': 'create read update delete',
            'rest': 'representational state transfer',
            'soap': 'simple object access protocol',
            'tcp': 'transmission control protocol',
            'udp': 'user datagram protocol',
            'ip': 'internet protocol',
            'dns': 'domain name system',
            'ssl': 'secure sockets layer',
            'tls': 'transport layer security',
            'vpn': 'virtual private network',
            'cdn': 'content delivery network',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'devops': 'development operations',
            'regex': 'regular expression',
            'xss': 'cross site scripting',
            'csrf': 'cross site request forgery',
            'jwt': 'json web token',
            'oauth': 'open authorization',
            'saml': 'security assertion markup language',
            'ldap': 'lightweight directory access protocol',
            'ssh': 'secure shell',
            'ftp': 'file transfer protocol',
            'smtp': 'simple mail transfer protocol',
            'pop': 'post office protocol',
            'imap': 'internet message access protocol',
        }
        
        # Technical terms that should be preserved as-is
        self.preserve_terms = {
            'github', 'stackoverflow', 'docker', 'kubernetes', 'react', 'angular',
            'vue', 'node', 'npm', 'pip', 'conda', 'git', 'svn', 'maven', 'gradle',
            'jenkins', 'travis', 'circleci', 'terraform', 'ansible', 'chef',
            'puppet', 'vagrant', 'elasticsearch', 'mongodb', 'postgresql', 'mysql',
            'redis', 'memcached', 'rabbitmq', 'kafka', 'spark', 'hadoop', 'hive',
            'airflow', 'luigi', 'celery', 'flask', 'django', 'fastapi', 'spring',
            'hibernate', 'struts', 'junit', 'pytest', 'mocha', 'jest', 'cucumber',
        }
        
        # Compile regex patterns for efficiency
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]*`')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
        self.multiple_spaces_pattern = re.compile(r'\s+')
    
    def normalize_countries_and_abbreviations(self, text: str) -> str:
        """Normalize countries and common abbreviations in text"""
        if not text:
            return ""
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        words = text_lower.split()
        
        normalized_words = []
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a country normalization
            if clean_word in self.country_normalizations:
                normalized_words.append(self.country_normalizations[clean_word])
            # Check if it's an abbreviation normalization
            elif clean_word in self.abbreviation_normalizations:
                normalized_words.append(self.abbreviation_normalizations[clean_word])
            # Preserve technical terms as-is
            elif clean_word in self.preserve_terms:
                normalized_words.append(clean_word)
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def clean_code_content(self, text: str) -> str:
        """Clean code content while preserving important information"""
        if not text:
            return ""
        
        # Remove code blocks but keep the content description
        text = self.code_pattern.sub(' ', text)
        
        # Remove URLs but keep domain info if relevant
        text = self.url_pattern.sub(' web_url ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' email_address ', text)
        
        return text
    
    def advanced_preprocess(self, text: str, is_code: bool = False) -> str:
        """Apply advanced preprocessing with normalization"""
        if not text:
            return ""
        
        # Clean code-specific content if needed
        if is_code:
            text = self.clean_code_content(text)
        
        # Apply country and abbreviation normalizations
        text = self.normalize_countries_and_abbreviations(text)
        
        # Remove extra special characters (but preserve some for technical terms)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Normalize whitespace
        text = self.multiple_spaces_pattern.sub(' ', text).strip()
        
        return text

class CodeSearchNetLoader:
    """Loads CodeSearchNet dataset from Hugging Face with advanced preprocessing"""
    
    def __init__(self, db_path: str = "data/search_engine.db"):
        self.db_path = db_path
        self.processor = AdvancedTextProcessor()
        self.available_languages = [
            'python', 'java', 'javascript', 'php', 'ruby', 'go'
        ]
    
    async def setup_database(self):
        """Setup database tables for CodeSearchNet documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS codesearchnet_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                code TEXT,
                language TEXT,
                url TEXT,
                original_content TEXT,
                processed_content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster searches
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_codesearchnet_language 
            ON codesearchnet_documents(language)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_codesearchnet_doc_id 
            ON codesearchnet_documents(doc_id)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database tables created/verified")
    
    def list_available_languages(self) -> List[str]:
        """List available programming languages in CodeSearchNet"""
        return self.available_languages.copy()
    
    async def load_codesearchnet_dataset(
        self, 
        languages: List[str] = None, 
        limit_per_language: Optional[int] = 10000,
        split: str = 'train'
    ) -> List[CodeDocument]:
        """
        Load CodeSearchNet dataset from Hugging Face
        
        Args:
            languages: List of programming languages to load
            limit_per_language: Maximum documents per language
            split: Dataset split ('train', 'validation', 'test')
        
        Returns:
            List of CodeDocument objects
        """
        if languages is None:
            languages = ['python', 'java', 'javascript']  # Default to popular languages
        
        # Validate languages
        invalid_langs = set(languages) - set(self.available_languages)
        if invalid_langs:
            raise ValueError(f"Invalid languages: {invalid_langs}. Available: {self.available_languages}")
        
        logger.info(f"Loading CodeSearchNet dataset for languages: {languages}")
        
        documents = []
        doc_counter = 0
        
        for language in languages:
            logger.info(f"Loading {language} documents...")
            
            try:
                # Load dataset for specific language
                dataset = load_dataset(
                    'code_search_net', 
                    language,
                    split=split,
                    streaming=True if limit_per_language else False
                )
                
                lang_counter = 0
                for example in dataset:
                    if limit_per_language and lang_counter >= limit_per_language:
                        break
                    
                    # Extract information
                    func_name = example.get('func_name', '')
                    docstring = example.get('docstring', '')
                    code = example.get('code', '')
                    url = example.get('url', '')
                    
                    # Create document ID
                    doc_id = f"csn_{language}_{doc_counter}"
                    
                    # Create title from function name and docstring
                    title = func_name if func_name else f"Code Function {doc_counter}"
                    
                    # Combine docstring and code for content
                    content = f"{docstring}\n\n{code}" if docstring else code
                    
                    # Apply advanced preprocessing
                    processed_content = self.processor.advanced_preprocess(
                        content, is_code=True
                    )
                    
                    # Create document
                    document = CodeDocument(
                        doc_id=doc_id,
                        title=title,
                        content=processed_content,
                        code=code,
                        language=language,
                        url=url,
                        metadata={
                            'dataset': 'codesearchnet',
                            'language': language,
                            'func_name': func_name,
                            'has_docstring': bool(docstring),
                            'code_length': len(code),
                            'split': split
                        }
                    )
                    
                    documents.append(document)
                    doc_counter += 1
                    lang_counter += 1
                    
                    if (lang_counter % 1000) == 0:
                        logger.info(f"Loaded {lang_counter:,} {language} documents...")
                
                logger.info(f"✅ Loaded {lang_counter:,} {language} documents")
                
            except Exception as e:
                logger.error(f"Error loading {language} dataset: {e}")
                continue
        
        logger.info(f"✅ Total loaded: {len(documents):,} CodeSearchNet documents")
        return documents
    
    async def save_documents_to_db(self, documents: List[CodeDocument]):
        """Save CodeSearchNet documents to database"""
        await self.setup_database()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        logger.info(f"Saving {len(documents)} documents to database...")
        
        for doc in documents:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO codesearchnet_documents 
                    (doc_id, title, content, code, language, url, 
                     original_content, processed_content, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc.doc_id,
                    doc.title,
                    doc.content,
                    doc.code,
                    doc.language,
                    doc.url,
                    doc.code,  # Original content is the raw code
                    doc.content,  # Processed content
                    json.dumps(doc.metadata)
                ))
            except Exception as e:
                logger.error(f"Error saving document {doc.doc_id}: {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info("✅ Documents saved to database")
    
    async def load_documents_from_db(
        self, 
        languages: List[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load CodeSearchNet documents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM codesearchnet_documents"
        params = []
        
        if languages:
            placeholders = ','.join('?' * len(languages))
            query += f" WHERE language IN ({placeholders})"
            params.extend(languages)
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        documents = []
        for row in rows:
            doc_dict = dict(zip(columns, row))
            # Parse metadata JSON
            if doc_dict['metadata']:
                doc_dict['metadata'] = json.loads(doc_dict['metadata'])
            documents.append(doc_dict)
        
        conn.close()
        logger.info(f"✅ Loaded {len(documents)} documents from database")
        return documents
    
    async def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored CodeSearchNet dataset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total documents
        cursor.execute("SELECT COUNT(*) FROM codesearchnet_documents")
        total_docs = cursor.fetchone()[0]
        
        # Documents per language
        cursor.execute("""
            SELECT language, COUNT(*) as count 
            FROM codesearchnet_documents 
            GROUP BY language 
            ORDER BY count DESC
        """)
        lang_stats = dict(cursor.fetchall())
        
        # Average code length
        cursor.execute("SELECT AVG(LENGTH(code)) FROM codesearchnet_documents")
        avg_code_length = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_documents': total_docs,
            'languages': lang_stats,
            'average_code_length': round(avg_code_length, 2) if avg_code_length else 0,
            'dataset_name': 'CodeSearchNet'
        }

# Example usage and testing functions
async def load_and_store_codesearchnet():
    """Load CodeSearchNet and store in database"""
    loader = CodeSearchNetLoader()
    
    # Load sample data
    documents = await loader.load_codesearchnet_dataset(
        languages=['python', 'javascript'],
        limit_per_language=5000
    )
    
    # Save to database
    await loader.save_documents_to_db(documents)
    
    # Get statistics
    stats = await loader.get_dataset_stats()
    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    return documents

async def test_text_processing():
    """Test advanced text processing"""
    processor = AdvancedTextProcessor()
    
    test_cases = [
        "I live in USA and work with AI and ML",
        "UK developers use JS and Python APIs",
        "REST API for NLP processing in the US",
        "Machine learning model deployed on AWS using Docker"
    ]
    
    print("Text Processing Examples:")
    for text in test_cases:
        processed = processor.advanced_preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()

async def main():
    """Main function for testing"""
    print("CodeSearchNet Loader Test")
    print("=" * 50)
    
    # Test text processing
    await test_text_processing()
    
    # Load and store dataset
    # documents = await load_and_store_codesearchnet()
    
    # For now, just show what's available
    loader = CodeSearchNetLoader()
    print("Available languages:", loader.list_available_languages())

if __name__ == "__main__":
    asyncio.run(main())
