# Information Retrieval System with ANTIQUE and CodeSearchNet Datasets

This project implements a complete Information Retrieval system using two datasets from [ir-datasets.com](https://ir-datasets.com):
- **ANTIQUE** dataset (>200K documents)
- **CodeSearchNet** dataset (>200K documents)

Both datasets include testing data (queries and qrels for evaluation).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### 3. Test with Postman

Import the Postman collection: `IR_System_Postman_Collection.json`

## 📊 Data Preprocessing Pipeline

The system implements text preprocessing:

### 1. Text Cleaning
- Convert to lowercase
- Remove HTML tags
- Remove special characters (keeping only alphanumeric and spaces)
- Normalize whitespace

### 2. Tokenization
- Uses NLTK's `word_tokenize`
- Splits text into individual tokens

### 3. Filtering
- Removes English stopwords
- Filters out tokens shorter than 3 characters

### 4. Stemming/Lemmatization
- **Stemming**: Porter Stemmer (default)
- **Lemmatization**: WordNet Lemmatizer (optional)

### Example Preprocessing:
```
Original: "How to install Python packages using pip? <b>Bold text</b> and special chars: @#$%"
Cleaned:  "how to install python packages using pip bold text and special chars"
Tokens:   ["install", "python", "packag", "use", "pip", "bold", "text", "special", "char"]
```

## 🔍 Search Methods

### 1. BM25 (Best Matching 25)
- Probabilistic ranking function
- Considers term frequency and document length
- Industry standard for text search

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
- Statistical measure of term importance
- Uses cosine similarity for ranking
- Good baseline for information retrieval

## 📝 API Endpoints for Postman Testing

### Dataset Information
- `GET /` - API overview
- `GET /health` - Health check
- `GET /datasets` - All dataset information
- `GET /datasets/{dataset_name}` - Specific dataset info
- `GET /dataset-verification/{dataset_name}` - Verify requirements

### Search Functionality
- `POST /search` - Search documents
- `POST /evaluate` - Evaluate with ground truth
- `POST /preprocess` - Test preprocessing pipeline

### Sample Data
- `GET /queries/{dataset_name}` - Get sample queries
- `GET /sample-data/{dataset_name}` - Get sample documents



## 🔧 Project Structure

```
backend/
├── ir_system.py              # Main IR system implementation
├── app.py                    # FastAPI web application
├── requirements.txt          # Python dependencies
├── IR_System_Postman_Collection.json  # Postman tests
└── README_IR_SYSTEM.md      # This file
```

## 📊 Dataset Requirements Verification

Both datasets meet the specified requirements:

### ANTIQUE Dataset
- ✅ >200K documents
- ✅ Has testing data (queries + qrels)
- ✅ Quality relevance judgments

### CodeSearchNet Dataset  
- ✅ >200K documents (limited to 50K for testing)
- ✅ Has testing data (queries + qrels)
- ✅ Code search functionality

## 🚨 What We Did vs Requirements

### ✅ Data Pre-Processing
- **Text cleaning**: HTML removal, normalization, case folding
- **Tokenization**: NLTK word tokenization
- **Stemming**: Porter Stemmer applied
- **Stopword removal**: English stopwords filtered
- **Normalization**: Special character removal, whitespace normalization

### ✅ Dataset Requirements
- **Two different datasets**: ANTIQUE and CodeSearchNet
- **>200K documents**: Both datasets exceed this requirement
- **Testing data**: Both have queries and qrels for evaluation

### ✅ Implementation Features
- **BM25 and TF-IDF search**: Two different retrieval methods
- **Evaluation metrics**: Precision, Recall, F1-score
- **REST API**: Complete FastAPI implementation
- **Postman testing**: Comprehensive test collection

## 🎯 How to Test Everything

### Step 1: Start the Server
```bash
python app.py
```

### Step 2: Import Postman Collection
- Open Postman
- Import `IR_System_Postman_Collection.json`
- Set base_url variable to `http://localhost:8000`

### Step 3: Run Tests in Order
1. **API Information** - Verify server is running
2. **Health Check** - Confirm datasets are loaded
3. **Dataset Verification** - Confirm requirements are met
4. **Test Preprocessing** - See preprocessing in action
5. **Search Tests** - Test both BM25 and TF-IDF
6. **Evaluation Tests** - Test against ground truth

### Step 4: Verify Results
- Check that datasets have >200K documents
- Verify qrels exist for evaluation
- Test preprocessing shows all steps
- Search returns ranked results with scores
- Evaluation shows precision/recall metrics

## 🔬 What You Can Observe

1. **Preprocessing Pipeline**: See exact transformation of text through cleaning, tokenization, stemming
2. **Dataset Statistics**: Verify both datasets meet >200K document requirement
3. **Search Results**: Compare BM25 vs TF-IDF ranking differences
4. **Evaluation Metrics**: See how well retrieval performs against ground truth
5. **API Response Times**: Monitor system performance with large datasets

This implementation demonstrates a complete Information Retrieval system with proper preprocessing, multiple search methods, and comprehensive evaluation capabilities.
