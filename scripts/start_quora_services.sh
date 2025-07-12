#!/bin/bash

# Start Quora Services
echo "🚀 Starting Quora Services..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to start a service in background
start_service() {
    local service_name="$1"
    local script_path="$2"
    local port="$3"
    
    echo "Starting $service_name on port $port..."
    bash "$script_path" &
    
    # Wait a moment for the service to start
    sleep 2
    
    # Check if service is running
    if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "✅ $service_name started successfully on port $port"
    else
        echo "⚠️ $service_name may not have started correctly"
    fi
}

# Start text processing service
start_service "Quora Text Processing Service" "$SCRIPT_DIR/start_quora_text_processing.sh" 5003

# Start query processing service
start_service "Quora Query Processing Service" "$SCRIPT_DIR/start_quora_query_processing.sh" 5004

echo ""
echo "🎉 Quora Services Started!"
echo "📡 Text Processing Service: http://localhost:5003"
echo "🔍 Query Processing Service: http://localhost:5004"
echo ""
echo "📖 API Documentation:"
echo "   Text Processing: http://localhost:5003/docs"
echo "   Query Processing: http://localhost:5004/docs"
echo ""
echo "ℹ️ To stop services, use: pkill -f 'quora.*processing'"
echo "ℹ️ To check status: curl http://localhost:5003/health && curl http://localhost:5004/health"
