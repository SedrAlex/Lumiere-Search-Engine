#!/usr/bin/env python3
"""
Service Starter for Antiqua Processing
=====================================

This script helps start the required services for processing the Antiqua dataset:
1. Enhanced TF-IDF Service (port 8007)
2. Inverted Index Service (port 8006)
3. Text Cleaning Service (port 8003)

Usage:
    python start_services_for_antiqua.py [--check-only]
"""

import subprocess
import time
import sys
import os
import signal
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Service configurations
SERVICES = {
    'text_cleaning': {
        'script': 'services/shared/enhanced_text_cleaning_service.py',
        'port': 8003,
        'url': 'http://localhost:8003'
    },
    'inverted_index': {
        'script': 'services/indexing/inverted_index_service.py',
        'port': 8006,
        'url': 'http://localhost:8006'
    },
    'enhanced_tfidf': {
        'script': 'services/representation/enhanced_tfidf_service.py',
        'port': 8007,
        'url': 'http://localhost:8007'
    }
}

class ServiceManager:
    """Manage multiple microservices"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.base_dir = Path(__file__).parent
    
    def check_service_health(self, service_name: str, url: str) -> bool:
        """Check if a service is healthy"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_all_services(self) -> Dict[str, bool]:
        """Check health of all services"""
        status = {}
        print("🏥 Checking service health...")
        
        for service_name, config in SERVICES.items():
            is_healthy = self.check_service_health(service_name, config['url'])
            status[service_name] = is_healthy
            
            status_icon = "✅" if is_healthy else "❌"
            print(f"  {service_name} (port {config['port']}): {status_icon}")
        
        return status
    
    def start_service(self, service_name: str) -> bool:
        """Start a single service"""
        if service_name in self.processes:
            print(f"⚠️ Service {service_name} is already running")
            return True
        
        config = SERVICES[service_name]
        script_path = self.base_dir / config['script']
        
        if not script_path.exists():
            print(f"❌ Service script not found: {script_path}")
            return False
        
        try:
            print(f"🚀 Starting {service_name} service on port {config['port']}...")
            
            # Start the service
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_dir)
            )
            
            self.processes[service_name] = process
            
            # Wait a bit for startup
            time.sleep(3)
            
            # Check if service is healthy
            max_retries = 10
            for i in range(max_retries):
                if self.check_service_health(service_name, config['url']):
                    print(f"✅ {service_name} service started successfully")
                    return True
                
                if process.poll() is not None:
                    # Process has terminated
                    stdout, stderr = process.communicate()
                    print(f"❌ {service_name} service failed to start")
                    print(f"   stdout: {stdout.decode()[:200]}...")
                    print(f"   stderr: {stderr.decode()[:200]}...")
                    return False
                
                time.sleep(2)
                print(f"   Waiting for {service_name} to become healthy... ({i+1}/{max_retries})")
            
            print(f"❌ {service_name} service failed to become healthy")
            return False
            
        except Exception as e:
            print(f"❌ Error starting {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all required services"""
        print("🚀 Starting all services for Antiqua processing...")
        
        # Start services in order (dependencies first)
        service_order = ['text_cleaning', 'inverted_index', 'enhanced_tfidf']
        
        for service_name in service_order:
            if not self.start_service(service_name):
                print(f"❌ Failed to start {service_name}. Stopping other services...")
                self.stop_all_services()
                return False
        
        print("✅ All services started successfully!")
        return True
    
    def stop_service(self, service_name: str):
        """Stop a single service"""
        if service_name in self.processes:
            process = self.processes[service_name]
            print(f"🛑 Stopping {service_name} service...")
            
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"   Force killing {service_name}...")
                process.kill()
                process.wait()
            
            del self.processes[service_name]
            print(f"✅ {service_name} service stopped")
    
    def stop_all_services(self):
        """Stop all running services"""
        print("🛑 Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        print("✅ All services stopped")
    
    def wait_for_services(self):
        """Wait for services to run and handle cleanup on exit"""
        if not self.processes:
            print("❌ No services are running")
            return
        
        print("🔄 Services are running. Press Ctrl+C to stop all services...")
        
        try:
            # Wait for any process to terminate
            while self.processes:
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        print(f"⚠️ Service {service_name} has terminated unexpectedly")
                        del self.processes[service_name]
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n⏹️ Received interrupt signal...")
        
        finally:
            self.stop_all_services()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Start services for Antiqua processing")
    parser.add_argument('--check-only', action='store_true',
                       help='Only check service health, do not start services')
    parser.add_argument('--service', type=str, choices=list(SERVICES.keys()),
                       help='Start only a specific service')
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n⏹️ Received signal {sig}")
        manager.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.check_only:
            status = manager.check_all_services()
            
            all_healthy = all(status.values())
            if all_healthy:
                print("\n✅ All services are healthy!")
                sys.exit(0)
            else:
                unhealthy = [name for name, healthy in status.items() if not healthy]
                print(f"\n❌ Unhealthy services: {unhealthy}")
                sys.exit(1)
        
        elif args.service:
            # Start specific service
            if manager.start_service(args.service):
                print(f"\n🎉 {args.service} service is ready!")
                manager.wait_for_services()
            else:
                print(f"\n❌ Failed to start {args.service} service")
                sys.exit(1)
        
        else:
            # Start all services
            if manager.start_all_services():
                print("\n🎉 All services are ready for Antiqua processing!")
                print("\nYou can now run:")
                print("  python apply_inverted_index_to_antiqua.py")
                print("\nPress Ctrl+C to stop all services...")
                manager.wait_for_services()
            else:
                print("\n❌ Failed to start all services")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        manager.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()
