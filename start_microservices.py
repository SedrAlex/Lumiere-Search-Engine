#!/usr/bin/env python3
"""
Microservices Orchestrator
Starts all TF-IDF microservices in the correct order
"""

import subprocess
import time
import sys
import os
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Service configurations
SERVICES = [
    {
        "name": "Text Cleaning Service",
        "script": "services/text_cleaning_service.py",
        "port": 8001,
        "health_endpoint": "http://localhost:8001/health"
    },
    {
        "name": "TF-IDF Vectorizer Service", 
        "script": "services/tfidf_vectorizer_service.py",
        "port": 8002,
        "health_endpoint": "http://localhost:8002/health"
    },
    {
        "name": "Enhanced TF-IDF Service",
        "script": "services/enhanced_tfidf_service.py", 
        "port": 8003,
        "health_endpoint": "http://localhost:8003/health"
    },
    {
        "name": "MAP Evaluation Service",
        "script": "services/map_evaluation_service.py",
        "port": 8004,
        "health_endpoint": "http://localhost:8004/health"
    }
]

def check_service_health(service):
    """Check if a service is healthy"""
    try:
        response = requests.get(service["health_endpoint"], timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_service(service, max_wait_time=30):
    """Wait for a service to become healthy"""
    logger.info(f"Waiting for {service['name']} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if check_service_health(service):
            logger.info(f"✓ {service['name']} is ready!")
            return True
        time.sleep(2)
    
    logger.error(f"✗ {service['name']} failed to start within {max_wait_time} seconds")
    return False

def start_service(service):
    """Start a service"""
    script_path = Path(service["script"])
    if not script_path.exists():
        logger.error(f"Service script not found: {script_path}")
        return None
    
    logger.info(f"Starting {service['name']} on port {service['port']}...")
    
    # Start the service
    process = subprocess.Popen([
        sys.executable, str(script_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the process to start
    time.sleep(3)
    
    # Check if process started successfully
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"Failed to start {service['name']}")
        logger.error(f"STDOUT: {stdout.decode()}")
        logger.error(f"STDERR: {stderr.decode()}")
        return None
    
    return process

def stop_services(processes):
    """Stop all running services"""
    logger.info("Stopping all services...")
    
    for service_name, process in processes.items():
        if process and process.poll() is None:
            logger.info(f"Stopping {service_name}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {service_name}...")
                process.kill()

def check_dependencies():
    """Check if required dependencies are available"""
    logger.info("Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import requests
        import sklearn
        import nltk
        import numpy
        import ir_datasets
        logger.info("✓ All required dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.error("Please install required dependencies:")
        logger.error("pip install fastapi uvicorn requests scikit-learn nltk numpy ir-datasets textblob")
        return False

def display_service_info():
    """Display information about all services"""
    print("\n" + "="*80)
    print("TF-IDF MICROSERVICES ARCHITECTURE")
    print("="*80)
    print("Port 8001: Text Cleaning Service")
    print("  - Advanced text cleaning with spell checking, lemmatization, stemming")
    print("  - Endpoints: /clean/tfidf, /clean/query, /clean/embedding")
    print()
    print("Port 8002: TF-IDF Vectorizer Service") 
    print("  - Basic TF-IDF vectorization service")
    print("  - Endpoints: /train, /vectorize")
    print()
    print("Port 8003: Enhanced TF-IDF Service")
    print("  - Complete TF-IDF service with inverted index")
    print("  - Endpoints: /train, /search")
    print()
    print("Port 8004: MAP Evaluation Service")
    print("  - MAP and IR metrics evaluation")
    print("  - Endpoints: /evaluate, /calculate_map")
    print()
    print("Service URLs:")
    for service in SERVICES:
        print(f"  {service['name']}: http://localhost:{service['port']}")
    print("="*80)

def test_service_communication():
    """Test communication between services"""
    logger.info("Testing service communication...")
    
    try:
        # Test text cleaning service
        response = requests.post(
            "http://localhost:8001/clean/tfidf",
            json={"text": "This is a test text with beautifull spelling!"}
        )
        if response.status_code == 200:
            logger.info("✓ Text Cleaning Service communication test passed")
        else:
            logger.warning("✗ Text Cleaning Service communication test failed")
        
        # Test Enhanced TF-IDF service health
        response = requests.get("http://localhost:8003/health")
        if response.status_code == 200:
            logger.info("✓ Enhanced TF-IDF Service communication test passed")
        else:
            logger.warning("✗ Enhanced TF-IDF Service communication test failed")
        
        # Test MAP Evaluation service
        response = requests.get("http://localhost:8004/health")
        if response.status_code == 200:
            logger.info("✓ MAP Evaluation Service communication test passed")
        else:
            logger.warning("✗ MAP Evaluation Service communication test failed")
        
        logger.info("✓ Service communication tests completed")
        
    except Exception as e:
        logger.error(f"Service communication test failed: {str(e)}")

def main():
    """Main orchestrator function"""
    print("TF-IDF Microservices Orchestrator")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Display service information
    display_service_info()
    
    # Start services
    processes = {}
    
    try:
        for service in SERVICES:
            # Start the service
            process = start_service(service)
            if process is None:
                logger.error(f"Failed to start {service['name']}")
                stop_services(processes)
                sys.exit(1)
            
            processes[service['name']] = process
            
            # Wait for service to be ready
            if not wait_for_service(service):
                logger.error(f"{service['name']} failed to become ready")
                stop_services(processes)
                sys.exit(1)
        
        logger.info("✓ All services started successfully!")
        
        # Test service communication
        test_service_communication()
        
        print("\n" + "="*80)
        print("ALL SERVICES ARE RUNNING")
        print("="*80)
        print("You can now:")
        print("1. Use the Enhanced TF-IDF Service at http://localhost:8003")
        print("2. Train models by calling POST /train")
        print("3. Search documents by calling POST /search") 
        print("4. Evaluate performance at http://localhost:8004")
        print()
        print("Example workflow:")
        print("1. POST http://localhost:8003/train - Train the model")
        print("2. POST http://localhost:8003/search - Search documents")
        print("3. POST http://localhost:8004/evaluate - Evaluate MAP performance")
        print()
        print("Press Ctrl+C to stop all services")
        print("="*80)
        
        # Keep services running
        try:
            while True:
                time.sleep(10)
                
                # Check if all services are still running
                all_running = True
                for service_name, process in processes.items():
                    if process.poll() is not None:
                        logger.error(f"Service {service_name} has stopped unexpectedly")
                        all_running = False
                        break
                
                if not all_running:
                    logger.error("One or more services have stopped. Shutting down...")
                    break
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
    
    except Exception as e:
        logger.error(f"Error in orchestrator: {str(e)}")
    
    finally:
        # Stop all services
        stop_services(processes)
        logger.info("All services stopped. Goodbye!")

if __name__ == "__main__":
    main()
