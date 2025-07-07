#!/bin/bash

# ANTIQUE Query Processing Service Startup Script
# This script starts the query processing microservice

echo "🚀 Starting ANTIQUE Query Processing Service..."
echo "📍 Service Location: services/processing/embedding_antique_query_processing.py"
echo "🌐 Service Port: 5002"
echo "⏰ Starting at: $(date)"

# Change to the backend directory
cd "$(dirname "$0")/.." || exit 1

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Make sure dependencies are installed."
fi

# Install required packages if needed
echo "📦 Checking dependencies..."
pip install flask flask-cors pandas numpy scikit-learn sentence-transformers faiss-cpu joblib requests > /dev/null 2>&1

# Check if text processing service is running
echo "🔍 Checking if text processing service is available..."
curl -s http://localhost:5001/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Text processing service is running"
else
    echo "⚠️  Text processing service not detected. Starting it first is recommended."
    echo "   Run: ./scripts/start_text_processing_service.sh"
fi

# Start the query processing service
echo "🔥 Starting Query Processing Service on port 5002..."
python services/processing/embedding_antique_query_processing.py
