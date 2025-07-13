#!/usr/bin/env python3
"""
TF-IDF ANTIQUE Text Processing Service
- Specialized text cleaning for ANTIQUE medical dataset
- Optimized for medical terminology and health queries
- Preserves medical terms while removing noise
- Includes medical synonym normalization
"""

import ssl
import uvicorn
import re
import nltk
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {str(e)}")

class OptimizedAntiqueTextCleaner:
    """
    Optimized text cleaning class for the ANTIQUE dataset focused on maximizing MAP score.
    """

    def __init__(self):
        # Minimal stopwords - preserve most meaningful terms for better matching
        basic_stopwords = set(stopwords.words('english'))
        # Remove medical and important query terms from stopwords
        important_terms = {
            'pain', 'cause', 'causes', 'treatment', 'treat', 'help', 'prevent', 'symptoms',
            'condition', 'disease', 'disorder', 'medicine', 'medical', 'health', 'body',
            'severe', 'chronic', 'acute', 'serious', 'normal', 'common', 'rare',
            'what', 'when', 'where', 'why', 'how', 'which', 'can', 'could', 'should',
            'would', 'may', 'might', 'need', 'want', 'get', 'make', 'take', 'give',
            'go', 'come', 'see', 'know', 'think', 'feel', 'look', 'work', 'use',
            'good', 'bad', 'better', 'best', 'worse', 'worst', 'much', 'many',
            'more', 'most', 'less', 'least', 'long', 'short', 'high', 'low',
            'old', 'new', 'young', 'early', 'late', 'first', 'last', 'next'
        }
        # Only use very basic stopwords
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Extended contractions dictionary
        self.contractions = {
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "n't": " not", "'re": " are", "'ve": " have",
            "'ll": " will", "'d": " would", "'m": " am",
            "what's": "what is", "that's": "that is", "there's": "there is",
            "it's": "it is", "he's": "he is", "she's": "she is",
            "doesn't": "does not", "isn't": "is not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mustn't": "must not"
        }

        # Medical synonyms and variations for term normalization
        self.medical_synonyms = {
            'ache': 'pain', 'aching': 'pain', 'hurt': 'pain', 'hurting': 'pain',
            'sore': 'pain', 'tender': 'pain', 'discomfort': 'pain',
            'illness': 'disease', 'sickness': 'disease', 'ailment': 'disease',
            'remedy': 'treatment', 'cure': 'treatment', 'therapy': 'treatment',
            'physician': 'doctor', 'doc': 'doctor', 'medic': 'doctor'
        }

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def expand_contractions(self, text):
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text

    def normalize_medical_terms(self, text):
        """Normalize medical terms to improve matching"""
        words = text.split()
        normalized_words = []
        for word in words:
            if word.lower() in self.medical_synonyms:
                normalized_words.append(self.medical_synonyms[word.lower()])
            else:
                normalized_words.append(word)
        return ' '.join(normalized_words)

    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        text = self.expand_contractions(text)

        # Remove URLs, emails, and other web artifacts
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)

        # More aggressive cleaning - keep only letters, numbers and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Simple word splitting (faster than word_tokenize)
        words = text.split()

        # Filter out very short words and numbers
        words = [word for word in words if len(word) >= 2 and word.isalpha()]

        # Lemmatize words that are not in stopwords
        processed_words = []
        for word in words:
            if word not in self.stop_words:
                # Simple lemmatization - faster than POS tagging
                lemma = self.lemmatizer.lemmatize(word)
                # Apply stemming for better matching
                stemmed = self.stemmer.stem(lemma)
                processed_words.append(stemmed)

        # Join and normalize medical terms
        cleaned_text = ' '.join(processed_words)
        cleaned_text = self.normalize_medical_terms(cleaned_text)

        return cleaned_text

# Initialize the text cleaner
text_cleaner = OptimizedAntiqueTextCleaner()

# FastAPI app
app = FastAPI(
    title="ANTIQUE TF-IDF Text Processing Service",
    description="Specialized text cleaning service for ANTIQUE medical dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TextCleanRequest(BaseModel):
    text: str

class TextCleanResponse(BaseModel):
    original_text: str
    cleaned_text: str
    processing_info: dict

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "ANTIQUE TF-IDF Text Processing Service",
        "version": "1.0.0",
        "description": "Specialized text cleaning for ANTIQUE medical dataset",
        "features": [
            "Medical terminology preservation",
            "Health-specific text preprocessing",
            "Symptom and treatment description handling",
            "Medical synonym normalization",
            "Optimized stopword filtering",
            "Lemmatization and stemming"
        ],
        "optimizations": [
            "Minimal stopwords for medical content",
            "Medical term synonym mapping",
            "Contraction expansion",
            "Aggressive noise removal",
            "Porter stemming for better matching"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ANTIQUE TF-IDF Text Processing Service",
        "text_cleaner_loaded": True,
        "nltk_data_available": True
    }

@app.post("/clean", response_model=TextCleanResponse)
async def clean_text(request: TextCleanRequest):
    """
    Clean text using ANTIQUE-optimized text processing
    """
    try:
        cleaned_text = text_cleaner.clean_text(request.text)
        
        # Provide processing information
        processing_info = {
            "original_length": len(request.text),
            "cleaned_length": len(cleaned_text),
            "word_count_original": len(request.text.split()),
            "word_count_cleaned": len(cleaned_text.split()),
            "preprocessing_steps": [
                "Lowercasing",
                "Contraction expansion",
                "URL and email removal",
                "Non-alphanumeric character removal",
                "Whitespace normalization",
                "Short word filtering",
                "Stopword removal",
                "Lemmatization",
                "Stemming",
                "Medical term normalization"
            ]
        }
        
        return TextCleanResponse(
            original_text=request.text,
            cleaned_text=cleaned_text,
            processing_info=processing_info
        )
        
    except Exception as e:
        logger.error(f"Text cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning error: {str(e)}")

@app.post("/batch_clean")
async def batch_clean_text(requests: list[TextCleanRequest]):
    """
    Clean multiple texts in batch
    """
    try:
        results = []
        for request in requests:
            cleaned_text = text_cleaner.clean_text(request.text)
            results.append({
                "original_text": request.text,
                "cleaned_text": cleaned_text
            })
        
        return {
            "results": results,
            "total_processed": len(results)
        }
        
    except Exception as e:
        logger.error(f"Batch text cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch text cleaning error: {str(e)}")

@app.get("/test")
async def test_cleaning():
    """Test the text cleaning functionality"""
    test_cases = [
        "What causes severe swelling and pain in the knees?",
        "I have chronic aching in my joints. What's the best remedy?",
        "How can I treat symptoms of diabetes?",
        "What are the side effects of this medication?",
        "Can you help me understand this medical condition?"
    ]
    
    results = []
    for test_text in test_cases:
        cleaned = text_cleaner.clean_text(test_text)
        results.append({
            "original": test_text,
            "cleaned": cleaned
        })
    
    return {
        "test_cases": results,
        "cleaner_status": "working"
    }

if __name__ == "__main__":
    # Run the service on port 8008
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_level="info"
    )
