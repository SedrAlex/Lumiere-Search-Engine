#!/bin/bash

# Start Quora Text Processing Service
echo "🚀 Starting Quora Text Processing Service..."

cd "$(dirname "$0")/../services/text_preprocessing"

# Activate virtual environment if it exists
if [ -d "../../venv" ]; then
    source ../../venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Start the service on port 5003
python quora_embedding_text_processing_service.py --port 5003 --host 0.0.0.0

echo "🔗 Quora Text Processing Service running on http://localhost:5003"
echo "📖 API docs available at: http://localhost:5003/docs"
