"""
Customized Text Cleaning Service for TF-IDF
This service uses the same cleaning steps as the corrected_tfidf_colab
"""

import re
import logging
from typing import List, Dict, Any
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
            nltk.data.find(f'tokenizers/{download}' if download == 'punkt' else f'corpora/{download}')
        except LookupError:
            nltk.download(download, quiet=True)
    
    NLTK_AVAILABLE = True
    logger.info("NLTK components initialized successfully")
    
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")

# Try to initialize spell checker (optional)
try:
    from symspellpy import SymSpell, Verbosity
    SPELLCHECK_AVAILABLE = True
    logger.info("SymSpell spell checker available")
except ImportError:
    SPELLCHECK_AVAILABLE = False
    logger.info("SymSpell not available. Spell checking disabled.")

# Request/Response Models
class CustomCleanRequest(BaseModel):
    text: str

class CustomCleanResponse(BaseModel):
    original_text: str
    cleaned_text: str
    tokens: List[str]

class CustomTextCleaningService:
    """Custom text cleaning service optimized for corrected colab TF-IDF"""
    
    def __init__(self):
        self.stemmer = None
        self.lemmatizer = None
        self.stop_words = set()
        self.spell_checker = None
        
        # Initialize NLTK components
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words = set(stopwords.words('english'))
                logger.info(f"Loaded {len(self.stop_words)} stopwords")
            except:
                logger.warning("Could not load stopwords")
        
        # Initialize spell checker if available
        if SPELLCHECK_AVAILABLE:
            try:
                self.spell_checker = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dict_path = 'frequency_dictionary_en_82_765.txt'
                if self._load_spell_dict(dict_path):
                    logger.info("Spell checker initialized with dictionary")
                else:
                    self.spell_checker = None
                    logger.info("Spell checker disabled - no dictionary found")
            except Exception as e:
                logger.warning(f"Could not initialize spell checker: {e}")
                self.spell_checker = None
    
    def _load_spell_dict(self, dict_path: str) -> bool:
        """Try to load spell checking dictionary"""
        try:
            import os
            if os.path.exists(dict_path):
                self.spell_checker.load_dictionary(dict_path, term_index=0, count_index=1)
                return True
        except:
            pass
        return False
    
    def clean_text(self, text: str) -> CustomCleanResponse:
        """
        Custom text cleaning pipeline
        Implements the EXACT same steps as corrected_tfidf_colab EnhancedTokenizer
        """
        if not text:
            return CustomCleanResponse(
                original_text=text,
                cleaned_text="",
                tokens=[]
            )
        
        original_text = text
        processed_tokens = []
        
        # Basic cleaning (same as corrected_tfidf_colab)
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Replace special chars
        
        # Tokenize
        tokens = word_tokenize(text)
        
        for token in tokens:
            if len(token) < 2 or not token.isalnum():
                continue
            
            # Skip stopwords
            if token in self.stop_words:
                continue
            
            # Spell checking (only if enabled and available)
            if self.spell_checker:
                try:
                    suggestions = self.spell_checker.lookup(token, Verbosity.CLOSEST)
                    if suggestions:
                        token = suggestions[0].term
                except:
                    pass  # Keep original token if spell check fails
            
            # Lemmatization THEN Stemming (same order as corrected_tfidf_colab)
            lemmatized = self.lemmatizer.lemmatize(token)
            stemmed = self.stemmer.stem(lemmatized)
            processed_tokens.append(stemmed)
        
        cleaned_text = " ".join(processed_tokens)
        
        return CustomCleanResponse(
            original_text=original_text,
            cleaned_text=cleaned_text,
            tokens=processed_tokens
        )

# FastAPI app
app = FastAPI(
    title="Customized Text Cleaning Service",
    description="Custom text preprocessing matching the corrected_tfidf_colab",
    version="1.0.0"
)

# Global service instance
cleaning_service = CustomTextCleaningService()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Customized Text Cleaning Service",
        "version": "1.0.0",
        "description": "Custom preprocessing pipeline for TF-IDF",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "custom_text_cleaning_service",
    }

@app.post("/clean", response_model=CustomCleanResponse)
async def clean_text(request: CustomCleanRequest):
    """Clean and preprocess text using custom pipeline"""
    try:
        result = cleaning_service.clean_text(request.text)
        return result
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning error: {str(e)}")

if __name__ == "__main__":
    # This service runs on port 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)
