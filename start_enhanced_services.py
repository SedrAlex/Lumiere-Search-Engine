#!/usr/bin/env python3
"""
Start Enhanced TF-IDF Services
Starts the inverted index service and enhanced TF-IDF service for improved MAP scores
"""

import subprocess
import time
import sys
import signal
import os
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

def start_service(service_name, script_path, port):
    """Start a service in a subprocess"""
    print(f"🚀 Starting {service_name} on port {port}...")
    try:
        process = subprocess.Popen([
            sys.executable, script_path
        ], cwd=backend_dir)
        return process
    except Exception as e:
        print(f"❌ Failed to start {service_name}: {e}")
        return None

def main():
    """Start all enhanced services"""
    processes = []
    
    print("🔧 Starting Enhanced TF-IDF Services")
    print("=" * 50)
    
    # Services to start
    services = [
        {
            "name": "TF-IDF Text Cleaning Service",
            "script": "services/shared/tfidf_text_cleaning_service.py",
            "port": 8005
        },
        {
            "name": "Inverted Index Service", 
            "script": "services/indexing/inverted_index_service.py",
            "port": 8006
        },
        {
            "name": "Enhanced TF-IDF Service",
            "script": "services/representation/enhanced_tfidf_service.py", 
            "port": 8007
        }
    ]
    
    # Start each service
    for service in services:
        process = start_service(service["name"], service["script"], service["port"])
        if process:
            processes.append({
                "name": service["name"],
                "process": process,
                "port": service["port"]
            })
            time.sleep(2)  # Give service time to start
    
    print("\n✅ All services started successfully!")
    print("\nService URLs:")
    print(f"🧹 Text Cleaning Service: http://localhost:8005")
    print(f"📚 Inverted Index Service: http://localhost:8006") 
    print(f"🔍 Enhanced TF-IDF Service: http://localhost:8007")
    
    print("\n📝 Key Improvements for Higher MAP Scores:")
    print("  • Increased vocabulary size: 10k → 100k features")
    print("  • N-gram range: (1,2) → (1,3) for better phrase matching")
    print("  • Query expansion using term co-occurrence")
    print("  • Semantic reranking with LSA (300 components)")
    print("  • Hybrid search combining inverted index + TF-IDF")
    print("  • Advanced text cleaning with lemmatization + stemming")
    print("  • BM25 scoring option in inverted index")
    
    print("\n🧪 Quick Test Commands:")
    print("curl -X GET http://localhost:8005/health")
    print("curl -X GET http://localhost:8006/health")
    print("curl -X GET http://localhost:8007/health")
    
    print("\n📊 To evaluate with enhanced parameters:")
    print("python complete_tfidf_evaluation.py --service-url http://localhost:8007")
    
    print("\n⏹️  Press Ctrl+C to stop all services")
    
    def signal_handler(signum, frame):
        print("\n🛑 Stopping all services...")
        for service in processes:
            print(f"  Stopping {service['name']}...")
            service["process"].terminate()
        
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if needed
        for service in processes:
            if service["process"].poll() is None:
                service["process"].kill()
        
        print("✅ All services stopped")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep main process running
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for service in processes:
                if service["process"].poll() is not None:
                    print(f"⚠️  {service['name']} has stopped unexpectedly")
                    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
