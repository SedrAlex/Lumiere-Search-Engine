#!/bin/bash

# Start Quora Query Processing Service
echo "🚀 Starting Quora Query Processing Service..."

cd "$(dirname "$0")/../services/query_processing/quora"

# Activate virtual environment if it exists
if [ -d "../../../venv" ]; then
    source ../../../venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Start the service on port 5004
python embedding_quora_query_processing.py

echo "🔗 Quora Query Processing Service running on http://localhost:5004"
echo "📖 API docs available at: http://localhost:5004/docs"
