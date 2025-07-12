import re
import nltk
import unicodedata
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class EnhancedTextProcessor:
    """
    Enhanced text processor optimized for ANTIQUE dataset to achieve MAP >= 0.2
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.inflect_engine = inflect.engine()
        
        # Custom stopwords - keeping important question words
        base_stopwords = set(stopwords.words('english'))
        important_words = {
            'what', 'when', 'where', 'why', 'who', 'which', 'how',
            'best', 'worst', 'better', 'good', 'bad', 'first', 'last',
            'old', 'new', 'antique', 'vintage', 'ancient', 'modern',
            'most', 'more', 'less', 'least', 'very', 'much',
            'can', 'could', 'should', 'would', 'may', 'might', 'must',
            'cause', 'causes', 'effect', 'effects', 'reason', 'reasons',
            'help', 'helps', 'prevent', 'prevents', 'cure', 'cures',
            'make', 'makes', 'use', 'uses', 'need', 'needs',
            'pain', 'swelling', 'symptom', 'symptoms'
        }
        self.stop_words = base_stopwords - important_words
        
        # Common contractions mapping
        self.contractions_dict = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
    
    def normalize_unicode(self, text):
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKD', text)
    
    def expand_contractions(self, text):
        """Expand contractions for better matching"""
        text = contractions.fix(text)
        for contraction, expansion in self.contractions_dict.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text
    
    def remove_html_tags(self, text):
        """Remove HTML tags if present"""
        if '<' in text and '>' in text:
            try:
                return BeautifulSoup(text, "html.parser").get_text()
            except:
                return text
        return text
    
    def clean_basic_text(self, text):
        """Basic text cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove HTML
        text = self.remove_html_tags(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Expand contractions before other processing
        text = self.expand_contractions(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Keep alphanumeric, spaces, and important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-\'\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def handle_negations(self, tokens):
        """Handle negations by marking negated words"""
        negated_tokens = []
        negate = False
        
        for i, token in enumerate(tokens):
            if token in ['not', 'no', 'never', 'neither', 'nor']:
                negate = True
                negated_tokens.append(token)
            elif negate and i < len(tokens) - 1:
                # Mark the next meaningful word as negated
                negated_tokens.append(f"NOT_{token}")
                negate = False
            else:
                negated_tokens.append(token)
        
        return negated_tokens
    
    def get_wordnet_pos(self, word):
        """Get WordNet POS tag for better lemmatization"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def process_tokens(self, tokens):
        """Process tokens with lemmatization and filtering"""
        processed = []
        
        for token in tokens:
            # Skip very short tokens
            if len(token) <= 1:
                continue
            
            # Skip pure numbers unless they're years or important
            if token.isdigit():
                if len(token) == 4 and 1000 <= int(token) <= 2100:  # Likely a year
                    processed.append(token)
                continue
            
            # Skip if it's in stopwords (unless it's a negated term)
            if token.lower() in self.stop_words and not token.startswith("NOT_"):
                continue
            
            # Lemmatize with POS tagging for better accuracy
            if not token.startswith("NOT_"):
                pos = self.get_wordnet_pos(token)
                lemmatized = self.lemmatizer.lemmatize(token, pos=pos)
                processed.append(lemmatized)
            else:
                # For negated terms, lemmatize the word part only
                word_part = token[4:]  # Remove "NOT_" prefix
                pos = self.get_wordnet_pos(word_part)
                lemmatized = self.lemmatizer.lemmatize(word_part, pos=pos)
                processed.append(f"NOT_{lemmatized}")
        
        return processed
    
    def extract_key_phrases(self, text):
        """Extract important phrases that might be split by tokenization"""
        phrases = []
        
        # Common medical/symptom phrases
        medical_phrases = [
            'chest pain', 'heart attack', 'blood pressure', 'side effect',
            'pain relief', 'home remedy', 'natural cure', 'medical condition'
        ]
        
        # Antique-related phrases
        antique_phrases = [
            'antique furniture', 'vintage item', 'old book', 'ancient artifact',
            'historical piece', 'collector item', 'rare find'
        ]
        
        # Question phrases
        question_phrases = [
            'how to', 'what is', 'why does', 'when should', 'where can',
            'what causes', 'how much', 'how many', 'best way'
        ]
        
        all_phrases = medical_phrases + antique_phrases + question_phrases
        
        text_lower = text.lower()
        for phrase in all_phrases:
            if phrase in text_lower:
                phrases.append(phrase.replace(' ', '_'))
        
        return phrases
    
    def process_text(self, text):
        """
        Main processing function optimized for MAP >= 0.2
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Basic cleaning
        cleaned = self.clean_basic_text(text)
        
        # Step 2: Extract key phrases before tokenization
        key_phrases = self.extract_key_phrases(cleaned)
        
        # Step 3: Tokenize
        tokens = word_tokenize(cleaned)
        
        # Step 4: Handle negations
        tokens = self.handle_negations(tokens)
        
        # Step 5: Process tokens (lemmatization, filtering)
        processed_tokens = self.process_tokens(tokens)
        
        # Step 6: Add key phrases
        processed_tokens.extend(key_phrases)
        
        # Step 7: Remove duplicates while preserving order
        seen = set()
        final_tokens = []
        for token in processed_tokens:
            if token not in seen:
                seen.add(token)
                final_tokens.append(token)
        
        return ' '.join(final_tokens)
    
    def process_query(self, query):
        """
        Special processing for queries to improve matching
        """
        # Process normally first
        processed = self.process_text(query)
        
        # For queries, we might want to add synonyms or related terms
        tokens = processed.split()
        expanded_tokens = tokens.copy()
        
        # Add common query expansions
        expansions = {
            'pain': ['ache', 'hurt', 'discomfort'],
            'swelling': ['inflammation', 'swollen', 'edema'],
            'cause': ['reason', 'why', 'lead'],
            'help': ['aid', 'assist', 'remedy'],
            'best': ['top', 'good', 'recommend'],
            'antique': ['vintage', 'old', 'ancient', 'historical']
        }
        
        for token in tokens:
            if token in expansions:
                expanded_tokens.extend(expansions[token])
        
        # Remove duplicates
        seen = set()
        final_tokens = []
        for token in expanded_tokens:
            if token not in seen:
                seen.add(token)
                final_tokens.append(token)
        
        return ' '.join(final_tokens)


def test_processor():
    """Test the enhanced processor"""
    processor = EnhancedTextProcessor()
    
    test_texts = [
        "What causes severe swelling and pain in the knees?",
        "I can't find any good antique furniture stores.",
        "This is a Victorian-era chair from the 1800s.",
        "Why don't airplanes have parachutes?",
        "The patient is experiencing chest pain and shortness of breath."
    ]
    
    print("Testing Enhanced Text Processor:")
    print("-" * 50)
    
    for text in test_texts:
        processed = processor.process_text(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()
    
    # Test query processing
    print("\nTesting Query Processing:")
    print("-" * 50)
    query = "What causes pain and swelling?"
    processed_query = processor.process_query(query)
    print(f"Query: {query}")
    print(f"Processed Query: {processed_query}")


if __name__ == "__main__":
    test_processor()
