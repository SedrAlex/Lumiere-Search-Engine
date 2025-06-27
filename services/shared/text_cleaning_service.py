"""
Shared Text Cleaning Service
Provides common text preprocessing functionality to all representation services
"""

import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class CleanTextRequest(BaseModel):
    text: str
    remove_stopwords: bool = True
    apply_stemming: bool = True
    apply_lemmatization: bool = False

class CleanTextResponse(BaseModel):
    original_text: str
    cleaned_text: str
    tokens: List[str]
    processing_steps: Dict[str, Any]

class TextCleaningService:
    """Shared text cleaning service for all representation methods"""
    
    def __init__(self):
        self.stemmer = None
        self.lemmatizer = None
        self.stop_words = set()
        
        # Initialize NLTK components if available
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            from nltk.stem import WordNetLemmatizer
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
            
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            logger.info("NLTK components initialized successfully")
            
        except ImportError:
            logger.warning("NLTK not available. Install with: pip install nltk")
        except Exception as e:
            logger.error(f"Error initializing NLTK components: {e}")
    
    def clean_text(self, text: str, 
                   remove_stopwords: bool = True,
                   apply_stemming: bool = True, 
                   apply_lemmatization: bool = False) -> CleanTextResponse:
        """
        Clean and preprocess text with configurable options
        
        Args:
            text: Input text to clean
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            CleanTextResponse with original text, cleaned text, tokens and processing info
        """
        if not text:
            return CleanTextResponse(
                original_text=text,
                cleaned_text="",
                tokens=[],
                processing_steps={"empty_input": True}
            )
        
        original_text = text
        processing_steps = {}
        
        # Step 1: Basic cleaning
        cleaned_text = self._basic_clean(text)
        processing_steps["basic_cleaning"] = {
            "applied": True,
            "steps": ["lowercase", "remove_extra_whitespace", "remove_special_chars"]
        }
        
        # Step 2: Tokenization
        tokens = self._tokenize(cleaned_text)
        processing_steps["tokenization"] = {
            "applied": True,
            "method": "nltk_word_tokenize" if self.stemmer else "simple_split",
            "token_count": len(tokens)
        }
        
        # Step 3: Remove stopwords
        if remove_stopwords and self.stop_words:
            original_count = len(tokens)
            tokens = self._remove_stopwords(tokens)
            processing_steps["stopword_removal"] = {
                "applied": True,
                "removed_count": original_count - len(tokens),
                "remaining_tokens": len(tokens)
            }
        else:
            processing_steps["stopword_removal"] = {"applied": False}
        
        # Step 4: Stemming
        if apply_stemming and self.stemmer:
            tokens = self._apply_stemming(tokens)
            processing_steps["stemming"] = {
                "applied": True,
                "method": "porter_stemmer"
            }
        else:
            processing_steps["stemming"] = {"applied": False}
        
        # Step 5: Lemmatization
        if apply_lemmatization and self.lemmatizer:
            tokens = self._apply_lemmatization(tokens)
            processing_steps["lemmatization"] = {
                "applied": True,
                "method": "wordnet_lemmatizer"
            }
        else:
            processing_steps["lemmatization"] = {"applied": False}
        
        # Rebuild cleaned text from final tokens
        final_cleaned_text = " ".join(tokens)
        
        return CleanTextResponse(
            original_text=original_text,
            cleaned_text=final_cleaned_text,
            tokens=tokens,
            processing_steps=processing_steps
        )
    
    def _basic_clean(self, text: str) -> str:
        """Apply basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        if self.stemmer:  # NLTK available
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
                return [token.lower() for token in tokens if token.isalnum()]
            except:
                pass
        
        # Fallback tokenization
        return [word.strip() for word in text.lower().split() if word.strip()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        if not self.stemmer:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def _apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if not self.lemmatizer:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]

# FastAPI app for the text cleaning service
app = FastAPI(
    title="Text Cleaning Service",
    description="Shared text preprocessing service for all representation methods",
    version="1.0.0"
)

# Global service instance
cleaning_service = TextCleaningService()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Text Cleaning Service",
        "version": "1.0.0",
        "description": "Shared text preprocessing for all representation methods",
        "endpoints": {
            "POST /clean": "Clean and preprocess text",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "text_cleaning_service",
        "nltk_available": cleaning_service.stemmer is not None,
        "stopwords_loaded": len(cleaning_service.stop_words) > 0
    }

@app.post("/clean", response_model=CleanTextResponse)
async def clean_text(request: CleanTextRequest):
    """Clean and preprocess text"""
    try:
        result = cleaning_service.clean_text(
            text=request.text,
            remove_stopwords=request.remove_stopwords,
            apply_stemming=request.apply_stemming,
            apply_lemmatization=request.apply_lemmatization
        )
        return result
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
