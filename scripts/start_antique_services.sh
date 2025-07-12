#!/bin/bash

# ANTIQUE SOA Services Startup Script
# This script starts both the text processing and query processing services

echo "🎯 Starting ANTIQUE SOA Services..."
echo "🏗️  SOA Architecture: Text Processing + Query Processing"
echo "⏰ Starting at: $(date)"
echo ""

# Change to the backend directory
cd "$(dirname "$0")/.." || exit 1

# Make scripts executable
chmod +x scripts/start_text_processing_service.sh
chmod +x scripts/start_query_processing_service.sh

echo "📋 Service Overview:"
echo "  ├── Text Processing Service (Port 5001)"
echo "  │   └── Handles text cleaning using ANTIQUE notebook methods"
echo "  └── Query Processing Service (Port 5002)"
echo "      └── Performs similarity search with embeddings"
echo ""

# Function to start service in background
start_service() {
    local script_name="$1"
    local service_name="$2"
    local log_file="logs/${service_name}.log"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    echo "🚀 Starting $service_name..."
    nohup ./scripts/$script_name > "$log_file" 2>&1 &
    local pid=$!
    echo "  └── PID: $pid (logs: $log_file)"
    
    # Give service time to start
    sleep 3
    
    return $pid
}

# Start text processing service first
start_service "start_text_processing_service.sh" "text_processing"
TEXT_PROCESSING_PID=$!

# Wait a bit for text processing service to fully start
echo "⏳ Waiting for text processing service to initialize..."
sleep 5

# Start query processing service
start_service "start_query_processing_service.sh" "query_processing"
QUERY_PROCESSING_PID=$!

echo ""
echo "✅ Services started!"
echo ""
echo "📡 Service URLs:"
echo "  ├── Text Processing: http://localhost:5001"
echo "  │   ├── Health: http://localhost:5001/health"
echo "  │   ├── Info: http://localhost:5001/info"
echo "  │   └── Process: http://localhost:5001/process"
echo "  └── Query Processing: http://localhost:5002"
echo "      ├── Health: http://localhost:5002/health"
echo "      ├── Info: http://localhost:5002/info"
echo "      ├── Search: http://localhost:5002/search"
echo "      └── Stats: http://localhost:5002/stats"
echo ""

# Test services
echo "🧪 Testing services..."
sleep 2

# Test text processing service
echo -n "  ├── Text Processing Service: "
if curl -s http://localhost:5001/health > /dev/null; then
    echo "✅ Online"
else
    echo "❌ Failed"
fi

# Test query processing service
echo -n "  └── Query Processing Service: "
if curl -s http://localhost:5002/health > /dev/null; then
    echo "✅ Online"
else
    echo "❌ Failed"
fi

echo ""
echo "📝 Example Usage:"
echo ""
echo "1. Process text:"
echo "   curl -X POST http://localhost:5001/process -H 'Content-Type: application/json' -d '{\"text\": \"Hello world!\"}'"
echo ""
echo "2. Search documents:"
echo "   curl -X POST http://localhost:5002/search -H 'Content-Type: application/json' -d '{\"query\": \"your search query\", \"top_k\": 10}'"
echo ""
echo "3. Stop services:"
echo "   kill $TEXT_PROCESSING_PID $QUERY_PROCESSING_PID"
echo ""

# Save PIDs for easy cleanup
echo "$TEXT_PROCESSING_PID" > .text_processing.pid
echo "$QUERY_PROCESSING_PID" > .query_processing.pid

echo "💡 Service PIDs saved to .text_processing.pid and .query_processing.pid"
echo "🔄 To stop services run: ./scripts/stop_antique_services.sh"
echo ""
echo "🎉 ANTIQUE SOA Services are ready!"

# Keep script running to show logs
echo "📊 Monitoring services... (Press Ctrl+C to stop monitoring)"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $TEXT_PROCESSING_PID $QUERY_PROCESSING_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Monitor the services
while true; do
    # Check if services are still running
    if ! kill -0 $TEXT_PROCESSING_PID 2>/dev/null; then
        echo "❌ Text Processing Service stopped unexpectedly"
        break
    fi
    
    if ! kill -0 $QUERY_PROCESSING_PID 2>/dev/null; then
        echo "❌ Query Processing Service stopped unexpectedly"
        break
    fi
    
    sleep 5
done
