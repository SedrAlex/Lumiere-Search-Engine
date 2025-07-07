#!/usr/bin/env python3
"""
QUORA TF-IDF Text Processing Service
A standalone microservice for processing text using the exact methods from the QUORA TF-IDF notebook.
This service follows SOA architecture and runs on a separate port.
Built with FastAPI for better performance and automatic API documentation.
"""

import os
import sys
import re
import unicodedata
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
import inflect

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraTFIDFTextProcessor:
    """
    Text processing service using the exact methods from the QUORA TF-IDF notebook.
    This applies the same processing pipeline used in training the TF-IDF model.
    """
    
    def __init__(self):
        """Initialize the text processor with NLTK resources and tools."""
        self.download_nltk_resources()
        self.setup_preprocessing_tools()
        
    def download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            
    def setup_preprocessing_tools(self):
        """Setup preprocessing tools exactly as in the notebook."""
        # Initialize tokenizer
        self.tokenizer = nltk.tokenize
        
        # Initialize stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Setup stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize inflect engine for number conversion
        self.inflect_engine = inflect.engine()
        
    def clean_text(self, text, words_to_remove=None):
        """Remove specific words from text."""
        if words_to_remove is None:
            words_to_remove = []
        words = text.split()
        cleaned_words = [word for word in words if word not in words_to_remove]
        cleaned_text = ' '.join(cleaned_words)
        return cleaned_text

    def number_to_words(self, text):
        """Convert numbers to words."""
        words = self.tokenizer.word_tokenize(text)
        converted_words = []
        for word in words:
            if word.replace('.', '', 1).isdigit():
                converted_words.append(word)
            else:
                if word.isdigit():
                    try:
                        num = int(word)
                        if num <= 999999999999999:
                            converted_word = self.inflect_engine.number_to_words(word)
                            converted_words.append(converted_word)
                        else:
                            converted_words.append('[Number Out of Range]')
                    except:
                        converted_words.append('[Number Out of Range]')
                else:
                    converted_words.append(word)
        return ' '.join(converted_words)

    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        try:
            if '<' in text and '>' in text:
                return BeautifulSoup(text, 'html.parser').get_text()
            else:
                return text
        except:
            logging.warning('MarkupResemblesLocatorWarning: The input looks more like a filename than markup.')
            return text

    def normalize_unicode(self, text):
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKD', text)

    def expand_contractions(self, text):
        """Expand contractions in text."""
        return contractions.fix(text)

    def cleaned_text(self, text):
        """Basic text cleaning - remove special characters and normalize whitespace."""
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalization_example(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def stemming_example(self, text):
        """Apply stemming to text."""
        words = self.tokenizer.word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatization_example(self, text):
        """Apply lemmatization to text."""
        words = self.tokenizer.word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def remove_stopwords(self, text):
        """Remove stopwords from text."""
        words = self.tokenizer.word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls(self, text):
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_special_characters_and_emojis(self, text):
        """Remove special characters and emojis."""
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)

    def replace_synonyms(self, text):
        """Replace words with their synonyms."""
        words = self.tokenizer.word_tokenize(text)
        synonym_words = [self.get_synonym(word) for word in words]
        return ' '.join(synonym_words)

    def get_synonym(self, word):
        """Get synonym for a word using WordNet."""
        synonyms = nltk.corpus.wordnet.synsets(word)
        if synonyms:
            return synonyms[0].lemmas()[0].name()
        return word

    def handle_negations(self, text):
        """Handle negations in text."""
        words = self.tokenizer.word_tokenize(text)
        negated_text = []
        negate = False
        for word in words:
            if word.lower() in ['not', "n't"]:
                negate = True
            elif negate:
                negated_text.append(f'NOT_{word}')
                negate = False
            else:
                negated_text.append(word)
        return ' '.join(negated_text)

    def remove_non_english_words(self, text):
        """Remove non-English words using WordNet."""
        words = self.tokenizer.word_tokenize(text)
        english_words = [word for word in words if wordnet.synsets(word)]
        return ' '.join(english_words)
        
    def processed_text(self, text):
        """
        Apply the complete text processing pipeline exactly as in the notebook.
        This matches the processed_text function used in training.
        """
        if text is None or pd.isna(text):
            return ""
            
        # Apply the exact same processing pipeline as in the notebook
        text = self.cleaned_text(text)
        text = self.normalization_example(text)
        text = self.stemming_example(text)
        text = self.lemmatization_example(text)
        text = self.remove_stopwords(text)
        text = self.number_to_words(text)
        text = self.remove_punctuation(text)
        text = self.expand_contractions(text)
        text = self.normalize_unicode(text)
        text = self.handle_negations(text)
        text = self.remove_urls(text)
        
        return text
        
    def process_batch(self, texts):
        """
        Process a batch of texts.
        
        Args:
            texts (list): List of texts to process
            
        Returns:
            list: List of processed texts
        """
        return [self.processed_text(text) for text in texts]

# Define FastAPI application
app = FastAPI(
    title="QUORA TF-IDF Text Processing Service",
    description="Service using QUORA TF-IDF methods for text processing",
    version="1.0.0"
)

# Initialize the text processor
processor = QuoraTFIDFTextProcessor()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class TextData(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

class QueryData(BaseModel):
    query: str

class DocumentData(BaseModel):
    document: str
    doc_id: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "quora-tfidf-text-processing"}

@app.post("/process")
async def process_text(data: TextData):
    """
    Process text using QUORA TF-IDF text processing methods.
    """
    try:
        if data.text:
            processed = processor.processed_text(data.text)
            return {"processed_text": processed}
        elif data.texts:
            if not isinstance(data.texts, list):
                raise HTTPException(status_code=400, detail="texts must be a list")
            processed = processor.process_batch(data.texts)
            return {"processed_texts": processed}
        raise HTTPException(status_code=400, detail="Either 'text' or 'texts' field required")
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process/query")
async def process_query(data: QueryData):
    """
    Process a search query specifically.
    """
    try:
        processed = processor.processed_text(data.query)
        return {"processed_query": processed}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.post("/process/document")
async def process_document(data: DocumentData):
    """
    Process a document specifically.
    """
    try:
        processed = processor.processed_text(data.document)
        return {"processed_document": processed, "doc_id": data.doc_id}
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "QUORA TF-IDF Text Processing Service",
        "version": "1.0.0",
        "description": "Text preprocessing service using methods from QUORA TF-IDF notebook",
        "processing_pipeline": [
            "cleaned_text",
            "normalization_example", 
            "stemming_example",
            "lemmatization_example",
            "remove_stopwords",
            "number_to_words",
            "remove_punctuation",
            "expand_contractions",
            "normalize_unicode",
            "handle_negations",
            "remove_urls"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "QUORA TF-IDF Text Processing Service",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == '__main__':
    print("🚀 Starting QUORA TF-IDF Text Processing Service with FastAPI...")
    print("📝 This service uses the exact processing pipeline from the QUORA TF-IDF notebook")
    uvicorn.run(app, host="0.0.0.0", port=5003)
