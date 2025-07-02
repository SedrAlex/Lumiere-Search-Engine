#!/usr/bin/env python3
"""
Quora TF-IDF Services Orchestrator
Starts all Quora TF-IDF related services in the correct order
"""

import subprocess
import time
import sys
import os
import requests
import signal
from pathlib import Path

# Service configuration
SERVICES = [
    {
        "name": "Quora Dataset Loading Service",
        "script": "services/data/quora_loader_service.py",
        "port": 8004,
        "startup_delay": 2
    },
    {
        "name": "TF-IDF Quora Representation Service", 
        "script": "services/representation/tfidf_quora_service.py",
        "port": 8006,
        "startup_delay": 3
    }
]

class QuoraServiceOrchestrator:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"), 
            ("httpx", "httpx"),
            ("scikit-learn", "sklearn"),
            ("numpy", "numpy"),
            ("joblib", "joblib"),
            ("pydantic", "pydantic")
        ]
        
        missing_packages = []
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install them with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies satisfied")
        return True
    
    def check_models(self):
        """Check if Quora TF-IDF models exist"""
        print("🔍 Checking Quora TF-IDF models...")
        
        model_dir = self.base_dir / "models"
        required_models = [
            "quora_tfidf_vectorizer.joblib",
            "quora_tfidf_matrix.joblib", 
            "quora_document_metadata.joblib"
        ]
        
        missing_models = []
        for model in required_models:
            model_path = model_dir / model
            if not model_path.exists():
                # Also check in /tmp directory
                tmp_path = Path("/tmp") / model
                if not tmp_path.exists():
                    missing_models.append(model)
        
        if missing_models:
            print(f"❌ Missing Quora TF-IDF models: {', '.join(missing_models)}")
            print("Please ensure models are trained and saved in the 'models' directory or /tmp")
            print("You can use the tfidf_quora_colab.ipynb notebook to train models")
            return False
        
        print("✅ All Quora TF-IDF models found")
        return True
    
    def start_service(self, service_config):
        """Start a single service"""
        script_path = self.base_dir / service_config["script"]
        
        if not script_path.exists():
            print(f"❌ Service script not found: {script_path}")
            return None
        
        print(f"🚀 Starting {service_config['name']} on port {service_config['port']}...")
        
        try:
            # Start the service as a subprocess
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_dir)
            )
            
            # Wait for startup
            time.sleep(service_config["startup_delay"])
            
            # Check if service is running
            if self.check_service_health(service_config["port"]):
                print(f"✅ {service_config['name']} started successfully")
                return process
            else:
                print(f"❌ {service_config['name']} failed to start")
                process.terminate()
                return None
                
        except Exception as e:
            print(f"❌ Error starting {service_config['name']}: {e}")
            return None
    
    def check_service_health(self, port, max_retries=5):
        """Check if a service is healthy"""
        for i in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False
    
    def start_all_services(self):
        """Start all services in order"""
        print("🎯 Starting Quora TF-IDF Services...")
        print("=" * 60)
        
        # Check dependencies and models first
        if not self.check_dependencies():
            return False
            
        if not self.check_models():
            print("\n⚠️  WARNING: Models not found, but continuing...")
            print("Services will start but may not function correctly until models are available.")
        
        print("\n🚀 Starting services...")
        
        # Start each service
        for service_config in SERVICES:
            process = self.start_service(service_config)
            if process:
                self.processes.append({
                    "name": service_config["name"],
                    "process": process,
                    "port": service_config["port"]
                })
            else:
                print(f"❌ Failed to start {service_config['name']}")
                self.stop_all_services()
                return False
        
        self.show_status()
        return True
    
    def show_status(self):
        """Show status of all services"""
        print("\n" + "=" * 60)
        print("🎉 ALL QUORA TF-IDF SERVICES STARTED SUCCESSFULLY!")
        print("=" * 60)
        
        for service in self.processes:
            print(f"✅ {service['name']}: http://localhost:{service['port']}")
        
        print(f"""
📋 Quick Test Commands:

1. LOAD QUORA DATASET:
   curl -X POST http://localhost:8004/load \\
     -H "Content-Type: application/json" \\
     -d '{{"data_path": "/path/to/quora/dataset.csv", "max_documents": 1000}}'

2. SEARCH QUORA DOCUMENTS:
   curl -X POST http://localhost:8006/search \\
     -H "Content-Type: application/json" \\
     -d '{{"query": "machine learning", "top_k": 5}}'

3. CHECK SERVICE STATUS:
   curl http://localhost:8004/health
   curl http://localhost:8006/status

🛑 To stop all services: Press Ctrl+C
""")
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        for service in self.processes:
            try:
                service["process"].terminate()
                service["process"].wait(timeout=5)
                print(f"✅ Stopped {service['name']}")
            except subprocess.TimeoutExpired:
                service["process"].kill()
                print(f"🔪 Force killed {service['name']}")
            except Exception as e:
                print(f"⚠️ Error stopping {service['name']}: {e}")
        
        self.processes.clear()
        print("✅ All services stopped")
    
    def wait_for_shutdown(self):
        """Wait for user to stop services"""
        try:
            print("🔄 Services running... Press Ctrl+C to stop")
            while True:
                # Check if any process has died
                for service in self.processes:
                    if service["process"].poll() is not None:
                        print(f"⚠️ {service['name']} has stopped unexpectedly")
                        return
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n📟 Shutdown signal received...")
            self.stop_all_services()

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n📟 Received shutdown signal")
    sys.exit(0)

def main():
    """Main orchestrator function"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    orchestrator = QuoraServiceOrchestrator()
    
    print("🎯 Quora TF-IDF Services Orchestrator")
    print("=" * 60)
    
    if orchestrator.start_all_services():
        orchestrator.wait_for_shutdown()
    else:
        print("❌ Failed to start services")
        sys.exit(1)

if __name__ == "__main__":
    main()
