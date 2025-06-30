# ===================================================================
# OPTIMIZED TF-IDF FOR ANTIQUE DATASET
# Features:
# - Enhanced tokenizer with spell check + lemmatization THEN stemming
# - Loads entire dataset at once
# - Cosine similarity evaluation
# - Proper model saving (vectorizer, matrix, metadata)
# ===================================================================

# STEP 1: Install & Import
# STEP 1: Install & Import
!pip install -q textblob symspellpy ir-datasets scikit-learn numpy joblib nltk tqdm

import ir_datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import joblib
from tqdm import tqdm
from typing import List, Dict
import os
import json

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize spell checker with error handling
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    dictionary_path = 'frequency_dictionary_en_82_765.txt'
    if not os.path.exists(dictionary_path):
        !wget https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/english/frequency_dictionary_en_82_765.txt
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    spell_check_available = True
except Exception as e:
    print(f"⚠️ Could not load dictionary: {e}")
    print("⚠️ Continuing without spell checking")
    spell_check_available = False


# STEP 2: Enhanced Tokenizer (Lemmatization THEN Stemming)
class EnhancedTokenizer:
    def __init__(self, use_spellcheck=True):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.use_spellcheck = use_spellcheck
        self.spell_checker = sym_spell if use_spellcheck else None

    def __call__(self, text: str) -> List[str]:
        """Tokenization pipeline: Lemmatization THEN Stemming"""
        if not text:
            return []

        # Basic cleaning
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Replace special chars

        # Tokenize
        tokens = word_tokenize(text)
        processed_tokens = []

        for token in tokens:
            if len(token) < 2 or not token.isalnum():
                continue

            # Skip stopwords
            if token in self.stop_words:
                continue

            # Spell checking (only if enabled)
            if self.use_spellcheck and self.spell_checker:
                suggestions = self.spell_checker.lookup(token, Verbosity.CLOSEST)
                if suggestions:
                    token = suggestions[0].term

            # Lemmatization THEN Stemming
            lemmatized = self.lemmatizer.lemmatize(token)
            stemmed = self.stemmer.stem(lemmatized)
            processed_tokens.append(stemmed)

        return processed_tokens

# STEP 3: Dataset Loading with Metadata Preservation
def load_dataset_with_metadata():
    """Load dataset with complete metadata preservation"""
    print("📚 Loading dataset with metadata...")
    dataset = ir_datasets.load('antique/train')

    documents = []
    doc_metadata = []

    # Load documents with metadata
    for doc in tqdm(dataset.docs_iter(), desc="Loading documents"):
        documents.append(doc.text)
        doc_metadata.append({
            'doc_id': doc.doc_id,
            'raw_text': doc.text,
            'length': len(doc.text)
        })

    # Load queries and qrels
    queries = [{'query_id': q.query_id, 'text': q.text} for q in dataset.queries_iter()]
    qrels = {(qrel.query_id, qrel.doc_id): qrel.relevance for qrel in dataset.qrels_iter()}

    print(f"✅ Loaded {len(documents)} docs, {len(queries)} queries")
    return documents, doc_metadata, queries, qrels

# STEP 4: TF-IDF Training with Cosine Similarity Evaluation
def train_and_evaluate():
    """Complete training and evaluation pipeline"""
    # Load data
    documents, doc_metadata, queries, qrels = load_dataset_with_metadata()

    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        norm='l2',
        tokenizer=EnhancedTokenizer(use_spellcheck=True),
        preprocessor=None,
        lowercase=False
    )
    # Train TF-IDF
    print("🏋️ Training TF-IDF model...")
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"✅ Training complete. Matrix shape: {tfidf_matrix.shape}")

    # Prepare evaluation
    query_tokenizer = EnhancedTokenizer(use_spellcheck=False)
    doc_ids = [meta['doc_id'] for meta in doc_metadata]

    # Evaluate with cosine similarity
    print("📊 Evaluating with cosine similarity...")
    metrics = {
        'map': 0,
        'mrr': 0,
        'precision@10': 0,
        'recall@10': 0,
        'evaluated_queries': 0
    }

    for query in tqdm(queries, desc="Evaluating queries"):
        query_id = query['query_id']

        # Find all relevant docs for this query
        relevant_docs = {doc_id: rel for (q_id, doc_id), rel in qrels.items() if q_id == query_id}
        if not relevant_docs:
            continue

        # Process and vectorize query
        query_tokens = query_tokenizer(query['text'])
        if not query_tokens:
            continue

        query_vec = vectorizer.transform([' '.join(query_tokens)])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_docs = [(doc_ids[i], similarities[i]) for i in ranked_indices if similarities[i] > 0]

        # Calculate metrics
        ap = 0.0
        rr = 0.0
        relevant_count = 0

        for i, (doc_id, score) in enumerate(ranked_docs[:1000], 1):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / i
                ap += precision_at_i

                if rr == 0:  # First relevant document
                    rr = 1 / i

        # Update metrics
        if relevant_docs:
            ap /= len(relevant_docs)
            retrieved_at_10 = ranked_docs[:10]
            relevant_at_10 = sum(1 for doc_id, _ in retrieved_at_10 if doc_id in relevant_docs)

            metrics['map'] += ap
            metrics['mrr'] += rr
            metrics['precision@10'] += relevant_at_10 / 10
            metrics['recall@10'] += relevant_at_10 / len(relevant_docs)
            metrics['evaluated_queries'] += 1

    # Finalize metrics
    if metrics['evaluated_queries'] > 0:
        for key in ['map', 'mrr', 'precision@10', 'recall@10']:
            metrics[key] /= metrics['evaluated_queries']

    print("\n🎯 Evaluation Results:")
    print(f"MAP: {metrics['map']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Precision@10: {metrics['precision@10']:.4f}")
    print(f"Recall@10: {metrics['recall@10']:.4f}")
    print(f"Evaluated on {metrics['evaluated_queries']} queries")

    return vectorizer, tfidf_matrix, doc_metadata, metrics

# STEP 5: Save Complete Model
def save_model(vectorizer, matrix, metadata, metrics, output_dir="model"):
    """Save all model components"""
    os.makedirs(output_dir, exist_ok=True)

    print("💾 Saving model components...")
    # Save vectorizer
    joblib.dump(vectorizer, f"{output_dir}/tfidf_vectorizer.joblib")

    # Save matrix
    joblib.dump(matrix, f"{output_dir}/tfidf_matrix.joblib")

    # Save metadata
    joblib.dump(metadata, f"{output_dir}/document_metadata.joblib")

    # Save metrics
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Model saved to {output_dir}/ directory")

# MAIN EXECUTION
if __name__ == "__main__":
    # Train and evaluate
    vectorizer, matrix, metadata, metrics = train_and_evaluate()

    # Save complete model
    save_model(vectorizer, matrix, metadata, metrics)

    print("\n🔍 Sample Tokenization Test:")
    sample_text = "Running and better organization of words"
    tokenizer = EnhancedTokenizer()
    print(f"Original: {sample_text}")
    print(f"Processed: {tokenizer(sample_text)}")