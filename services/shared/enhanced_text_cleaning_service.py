"""
Enhanced Text Cleaning Service for TF-IDF
Optimized preprocessing pipeline to maximize MAP (Mean Average Precision)
Implements the same cleaning used in the corrected TF-IDF training
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
class EnhancedCleanRequest(BaseModel):
    text: str
    use_lemmatization: bool = True
    use_stemming: bool = True
    use_spellcheck: bool = False
    remove_stopwords: bool = True
    min_token_length: int = 2

class EnhancedCleanResponse(BaseModel):
    original_text: str
    cleaned_text: str
    tokens: List[str]
    processing_stats: Dict[str, Any]

class EnhancedTextCleaningService:
    """Enhanced text cleaning service optimized for TF-IDF performance"""
    
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
                # Try to load dictionary file if it exists
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
    
    def clean_text(self, text: str, **options) -> EnhancedCleanResponse:
        """
        Enhanced text cleaning pipeline optimized for TF-IDF
        Implements: lemmatization → stemming pipeline for better MAP scores
        """
        if not text:
            return EnhancedCleanResponse(
                original_text=text,
                cleaned_text="",
                tokens=[],
                processing_stats={"empty_input": True}
            )
        
        original_text = text
        stats = {
            "original_length": len(text),
            "steps_applied": [],
            "tokens_removed": 0,
            "spell_corrections": 0
        }
        
        # Step 1: Basic cleaning
        cleaned_text = self._basic_clean(text)
        stats["steps_applied"].append("basic_cleaning")
        stats["cleaned_length"] = len(cleaned_text)
        
        # Step 2: Tokenization  
        tokens = self._tokenize(cleaned_text)
        stats["steps_applied"].append("tokenization")
        stats["initial_tokens"] = len(tokens)
        
        # Step 3: Filter short tokens and non-alphanumeric
        min_length = options.get('min_token_length', 2)
        original_count = len(tokens)
        tokens = [token for token in tokens if len(token) >= min_length and token.isalnum()]
        stats["tokens_removed"] += (original_count - len(tokens))
        stats["steps_applied"].append("length_filtering")
        
        # Step 4: Remove stopwords
        if options.get('remove_stopwords', True) and self.stop_words:
            original_count = len(tokens)
            tokens = [token for token in tokens if token not in self.stop_words]
            stats["tokens_removed"] += (original_count - len(tokens))
            stats["steps_applied"].append("stopword_removal")
        
        # Step 5: Spell checking (optional)
        if options.get('use_spellcheck', False) and self.spell_checker:
            corrected_tokens = []
            for token in tokens:
                suggestions = self.spell_checker.lookup(token, Verbosity.CLOSEST)
                if suggestions and len(suggestions) > 0:
                    corrected = suggestions[0].term
                    if corrected != token:
                        stats["spell_corrections"] += 1
                    corrected_tokens.append(corrected)
                else:
                    corrected_tokens.append(token)
            tokens = corrected_tokens
            stats["steps_applied"].append("spell_checking")
        
        # Step 6: Lemmatization (if enabled)
        if options.get('use_lemmatization', True) and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            stats["steps_applied"].append("lemmatization")
        
        # Step 7: Stemming (if enabled) - Applied AFTER lemmatization
        if options.get('use_stemming', True) and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
            stats["steps_applied"].append("stemming")
        
        # Final cleaning
        final_cleaned_text = " ".join(tokens)
        stats["final_tokens"] = len(tokens)
        stats["final_length"] = len(final_cleaned_text)
        
        return EnhancedCleanResponse(
            original_text=original_text,
            cleaned_text=final_cleaned_text,
            tokens=tokens,
            processing_stats=stats
        )
    
    def _basic_clean(self, text: str) -> str:
        """Apply basic text cleaning optimized for information retrieval"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Replace special characters with spaces (preserve word boundaries)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK if available"""
        if not text:
            return []
        
        if NLTK_AVAILABLE and self.stemmer:
            try:
                tokens = word_tokenize(text)
                return [token.lower() for token in tokens if token.isalnum()]
            except:
                pass
        
        # Fallback tokenization
        return [word.strip() for word in text.lower().split() if word.strip().isalnum()]
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "nltk_available": NLTK_AVAILABLE,
            "spellcheck_available": SPELLCHECK_AVAILABLE and self.spell_checker is not None,
            "stopwords_loaded": len(self.stop_words) > 0,
            "components": {
                "stemmer": self.stemmer is not None,
                "lemmatizer": self.lemmatizer is not None,
                "spell_checker": self.spell_checker is not None
            }
        }

# FastAPI app
app = FastAPI(
    title="Enhanced Text Cleaning Service",
    description="Optimized text preprocessing for TF-IDF with improved MAP scores",
    version="2.0.0"
)

# Global service instance
cleaning_service = EnhancedTextCleaningService()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced Text Cleaning Service",
        "version": "2.0.0",
        "description": "Optimized preprocessing pipeline for TF-IDF with lemmatization→stemming",
        "features": [
            "Lemmatization followed by stemming",
            "Optional spell checking with SymSpell",
            "Configurable preprocessing steps",
            "Optimized for information retrieval MAP scores"
        ],
        "endpoints": {
            "POST /clean": "Clean and preprocess text with enhanced pipeline",
            "GET /health": "Health check with component status",
            "GET /stats": "Get cleaning service statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = cleaning_service.get_cleaning_stats()
    return {
        "status": "healthy",
        "service": "enhanced_text_cleaning_service",
        **stats
    }

@app.get("/stats")
async def get_stats():
    """Get detailed service statistics"""
    return cleaning_service.get_cleaning_stats()

@app.post("/clean", response_model=EnhancedCleanResponse)
async def clean_text(request: EnhancedCleanRequest):
    """Clean and preprocess text using enhanced pipeline"""
    try:
        result = cleaning_service.clean_text(
            text=request.text,
            use_lemmatization=request.use_lemmatization,
            use_stemming=request.use_stemming,
            use_spellcheck=request.use_spellcheck,
            remove_stopwords=request.remove_stopwords,
            min_token_length=request.min_token_length
        )
        return result
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text cleaning error: {str(e)}")

@app.post("/clean/batch")
async def clean_batch(texts: List[str]):
    """Clean multiple texts in batch"""
    try:
        results = []
        for text in texts:
            result = cleaning_service.clean_text(text)
            results.append(result)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error in batch cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch cleaning error: {str(e)}")

if __name__ == "__main__":
    # Enhanced text cleaning service runs on port 8003
    uvicorn.run(app, host="0.0.0.0", port=8003)
