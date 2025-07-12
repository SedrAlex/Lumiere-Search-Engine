#!/bin/bash

# ANTIQUE Text Processing Service Startup Script
# This script starts the text processing microservice

echo "🚀 Starting ANTIQUE Text Processing Service..."
echo "📍 Service Location: text_preprocessing/embedding_antique_text_processing_service.py"
echo "🌐 Service Port: 5001"
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
pip install flask flask-cors pandas nltk scikit-learn requests numpy > /dev/null 2>&1

# Start the text processing service
echo "🔥 Starting Text Processing Service on port 5001..."
python text_preprocessing/embedding_antique_text_processing_service.py
