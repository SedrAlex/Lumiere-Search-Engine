#!/bin/bash

# ANTIQUE SOA Services Stop Script
# This script stops both services gracefully

echo "🛑 Stopping ANTIQUE SOA Services..."
echo "⏰ Stopping at: $(date)"

# Change to the backend directory
cd "$(dirname "$0")/.." || exit 1

# Function to stop a service
stop_service() {
    local pid_file="$1"
    local service_name="$2"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo "🔍 Checking $service_name (PID: $pid)..."
        
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ├── Sending SIGTERM to $service_name..."
            kill "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "  ├── Force killing $service_name..."
                kill -9 "$pid"
            fi
            
            echo "  └── ✅ $service_name stopped"
        else
            echo "  └── ⚠️  $service_name was not running"
        fi
        
        # Remove PID file
        rm -f "$pid_file"
    else
        echo "⚠️  $service_name PID file not found"
    fi
}

# Stop services
stop_service ".text_processing.pid" "Text Processing Service"
stop_service ".query_processing.pid" "Query Processing Service"

# Also try to kill any remaining processes on the known ports
echo ""
echo "🧹 Cleaning up any remaining processes..."

# Find and kill processes on port 5001 and 5002
for port in 5001 5002; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  ├── Killing process on port $port (PID: $pid)"
        kill -9 "$pid" 2>/dev/null
    fi
done

# Clean up log files older than 7 days
if [ -d "logs" ]; then
    echo "🗂️  Cleaning old log files..."
    find logs -name "*.log" -mtime +7 -delete 2>/dev/null
fi

echo ""
echo "✅ All ANTIQUE SOA Services stopped successfully!"
echo "🎯 You can restart them with: ./scripts/start_antique_services.sh"
