"""
TF-IDF Specialized Text Cleaning Service
This service implements the EXACT same text cleaning pipeline as used in corrected_tfidf_colab
Optimized specifically for TF-IDF models and vectorization
"""

import re
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    
    # Download required NLTK data
    nltk_downloads = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
    for download in nltk_downloads:
        try:
            if download == 'punkt':
                nltk.data.find(f'tokenizers/{download}')
            elif download in ['wordnet', 'omw-1.4']:
                nltk.data.find(f'corpora/{download}')
            else:
                nltk.data.find(f'corpora/{download}')
        except (LookupError, FileNotFoundError, Exception) as e:
            logger.info(f"Downloading NLTK data: {download} (Error: {e})")
            try:
                nltk.download(download, quiet=True)
            except Exception as download_error:
                logger.warning(f"Failed to download {download}: {download_error}")
    
    NLTK_AVAILABLE = True
    logger.info("NLTK components initialized successfully for TF-IDF cleaning")
    
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")

# Request/Response Models
class TFIDFCleanRequest(BaseModel):
    text: str
    preserve_document_structure: bool = True

class TFIDFCleanResponse(BaseModel):
    original_text: str
    cleaned_text: str
    tokens: List[str]
    token_count: int
    processing_stats: Dict[str, Any]

class TFIDFTextCleaningService:
    """
    TF-IDF specialized text cleaning service
    Implements the EXACT same cleaning pipeline as corrected_tfidf_colab
    """
    
    def __init__(self):
        self.stemmer = None
        self.lemmatizer = None
        self.stop_words = set()
        
        # Initialize NLTK components (same as corrected_tfidf_colab)
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words = set(stopwords.words('english'))
                logger.info(f"Loaded {len(self.stop_words)} English stopwords for TF-IDF")
            except Exception as e:
                logger.warning(f"Could not load stopwords: {e}")
        else:
            logger.error("NLTK components not available - text cleaning will be limited")
    
    def clean_text_for_tfidf(self, text: str, preserve_document_structure: bool = True) -> TFIDFCleanResponse:
        """
        Clean text specifically for TF-IDF vectorization
        This implements the EXACT same steps as corrected_tfidf_colab EnhancedTokenizer
        
        Args:
            text: Input text to clean
            preserve_document_structure: Whether to preserve some document structure elements
            
        Returns:
            TFIDFCleanResponse with cleaned text optimized for TF-IDF
        """
        if not text or not text.strip():
            return TFIDFCleanResponse(
                original_text=text or "",
                cleaned_text="",
                tokens=[],
                token_count=0,
                processing_stats={"empty_input": True}
            )
        
        original_text = text
        processing_stats = {
            "original_length": len(text),
            "steps_applied": []
        }
        
        # Step 1: Convert to lowercase (same as corrected_tfidf_colab)
        text = text.lower()
        processing_stats["steps_applied"].append("lowercase_conversion")
        
        # Step 2: Remove HTML tags (same as corrected_tfidf_colab)
        text = re.sub(r'<[^>]+>', '', text)
        processing_stats["steps_applied"].append("html_tag_removal")
        
        # Step 3: Clean special characters (same pattern as corrected_tfidf_colab)
        # Keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        processing_stats["steps_applied"].append("special_character_removal")
        
        # Step 4: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        processing_stats["steps_applied"].append("whitespace_normalization")
        
        # Step 5: Tokenization (same as corrected_tfidf_colab)
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        
        processing_stats["tokens_after_tokenization"] = len(tokens)
        processing_stats["steps_applied"].append("tokenization")
        
        # Step 6: Token filtering (same criteria as corrected_tfidf_colab)
        filtered_tokens = []
        for token in tokens:
            # Skip tokens that are too short or not alphanumeric
            if len(token) < 2 or not token.isalnum():
                continue
            
            # Skip stopwords (same as corrected_tfidf_colab)
            if token in self.stop_words:
                continue
                
            filtered_tokens.append(token)
        
        processing_stats["tokens_after_filtering"] = len(filtered_tokens)
        processing_stats["stopwords_removed"] = len(tokens) - len(filtered_tokens) - (len(tokens) - len([t for t in tokens if len(t) >= 2 and t.isalnum()]))
        processing_stats["steps_applied"].append("token_filtering")
        
        # Step 7: Lemmatization THEN Stemming (EXACT same order as corrected_tfidf_colab)
        final_tokens = []
        if NLTK_AVAILABLE and self.lemmatizer and self.stemmer:
            for token in filtered_tokens:
                # First lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                # Then stem the lemmatized form
                stemmed = self.stemmer.stem(lemmatized)
                final_tokens.append(stemmed)
            
            processing_stats["steps_applied"].extend(["lemmatization", "stemming"])
        else:
            final_tokens = filtered_tokens
            processing_stats["steps_applied"].append("no_morphological_processing")
        
        # Step 8: Create final cleaned text
        cleaned_text = " ".join(final_tokens)
        
        # Final processing statistics
        processing_stats.update({
            "final_token_count": len(final_tokens),
            "final_text_length": len(cleaned_text),
            "compression_ratio": len(cleaned_text) / len(original_text) if len(original_text) > 0 else 0,
            "nltk_available": NLTK_AVAILABLE,
            "stemmer_used": self.stemmer is not None,
            "lemmatizer_used": self.lemmatizer is not None,
            "stopwords_count": len(self.stop_words)
        })
        
        logger.debug(f"TF-IDF text cleaning: {len(original_text)} -> {len(cleaned_text)} chars, {len(final_tokens)} tokens")
        
        return TFIDFCleanResponse(
            original_text=original_text,
            cleaned_text=cleaned_text,
            tokens=final_tokens,
            token_count=len(final_tokens),
            processing_stats=processing_stats
        )
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service configuration"""
        return {
            "service_name": "TF-IDF Text Cleaning Service",
            "version": "1.0.0",
            "nltk_available": NLTK_AVAILABLE,
            "stemmer_available": self.stemmer is not None,
            "lemmatizer_available": self.lemmatizer is not None,
            "stopwords_loaded": len(self.stop_words),
            "processing_pipeline": [
                "lowercase_conversion",
                "html_tag_removal", 
                "special_character_removal",
                "whitespace_normalization",
                "tokenization",
                "token_filtering (length >= 2, alphanumeric, non-stopword)",
                "lemmatization",
                "stemming"
            ],
            "compatible_with": "corrected_tfidf_colab",
            "optimized_for": "TF-IDF vectorization"
        }

# FastAPI app
app = FastAPI(
    title="TF-IDF Text Cleaning Service",
    description="Specialized text preprocessing service matching corrected_tfidf_colab approach",
    version="1.0.0"
)

# Global service instance
tfidf_cleaning_service = TFIDFTextCleaningService()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TF-IDF Text Cleaning Service",
        "version": "1.0.0",
        "description": "Specialized text cleaning pipeline optimized for TF-IDF models",
        "compatible_with": "corrected_tfidf_colab",
        "endpoints": {
            "POST /clean": "Clean text for TF-IDF processing",
            "GET /health": "Health check",
            "GET /info": "Service configuration info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tfidf_text_cleaning_service",
        "nltk_available": NLTK_AVAILABLE,
        "components_ready": {
            "stemmer": tfidf_cleaning_service.stemmer is not None,
            "lemmatizer": tfidf_cleaning_service.lemmatizer is not None,
            "stopwords": len(tfidf_cleaning_service.stop_words) > 0
        }
    }

@app.get("/info")
async def get_service_info():
    """Get detailed service information"""
    return tfidf_cleaning_service.get_service_info()

@app.post("/clean", response_model=TFIDFCleanResponse)
async def clean_text_for_tfidf(request: TFIDFCleanRequest):
    """Clean and preprocess text specifically for TF-IDF vectorization"""
    try:
        result = tfidf_cleaning_service.clean_text_for_tfidf(
            text=request.text,
            preserve_document_structure=request.preserve_document_structure
        )
        return result
    except Exception as e:
        logger.error(f"Error cleaning text for TF-IDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TF-IDF text cleaning error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8005 (dedicated for TF-IDF)
    uvicorn.run(app, host="0.0.0.0", port=8005)
