#!/usr/bin/env python3
"""
TF-IDF Evaluation Services Startup Script
=========================================

This script starts all the services required for complete TF-IDF evaluation:
1. TF-IDF Text Cleaning Service (port 8005)
2. TF-IDF Query Processing Service (port 8004)

Run this before executing the complete evaluation script.
"""

import asyncio
import subprocess
import time
import sys
import signal
import httpx
from pathlib import Path

# Service configurations
SERVICES = {
    "tfidf_cleaning": {
        "script": "services/shared/tfidf_text_cleaning_service.py",
        "port": 8005,
        "name": "TF-IDF Text Cleaning Service",
        "health_endpoint": "/health"
    },
    "tfidf_query": {
        "script": "services/query_processing/tfidf_query_processor.py", 
        "port": 8004,
        "name": "TF-IDF Query Processing Service",
        "health_endpoint": "/health"
    }
}

class ServiceManager:
    """Manager for TF-IDF evaluation services"""
    
    def __init__(self):
        self.processes = {}
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
        
    def start_service(self, service_name: str, service_config: dict) -> subprocess.Popen:
        """Start a single service"""
        script_path = Path(service_config["script"])
        
        if not script_path.exists():
            raise FileNotFoundError(f"Service script not found: {script_path}")
        
        print(f"🚀 Starting {service_config['name']} on port {service_config['port']}...")
        
        # Start the service
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes[service_name] = process
        return process
    
    async def wait_for_service(self, service_name: str, service_config: dict, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        port = service_config["port"]
        health_url = f"http://localhost:{port}{service_config['health_endpoint']}"
        
        print(f"⏳ Waiting for {service_config['name']} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await self.http_client.get(health_url)
                if response.status_code == 200:
                    print(f"✅ {service_config['name']} is ready!")
                    return True
            except:
                pass
            
            await asyncio.sleep(1)
        
        print(f"❌ {service_config['name']} failed to start within {timeout} seconds")
        return False
    
    async def check_all_services(self) -> dict:
        """Check the status of all services"""
        status = {}
        
        for service_name, service_config in SERVICES.items():
            port = service_config["port"]
            health_url = f"http://localhost:{port}{service_config['health_endpoint']}"
            
            try:
                response = await self.http_client.get(health_url)
                if response.status_code == 200:
                    service_info = response.json()
                    status[service_name] = {
                        "running": True,
                        "port": port,
                        "info": service_info
                    }
                else:
                    status[service_name] = {"running": False, "port": port}
            except:
                status[service_name] = {"running": False, "port": port}
        
        return status
    
    async def start_all_services(self) -> bool:
        """Start all required services"""
        print("🚀 Starting TF-IDF Evaluation Services")
        print("=" * 50)
        
        # Check if services are already running
        print("🔍 Checking existing services...")
        status = await self.check_all_services()
        
        running_services = [name for name, info in status.items() if info["running"]]
        if running_services:
            print(f"✅ Already running: {', '.join(running_services)}")
        
        # Start services that are not running
        for service_name, service_config in SERVICES.items():
            if not status.get(service_name, {}).get("running", False):
                try:
                    self.start_service(service_name, service_config)
                    
                    # Wait for service to be ready
                    if await self.wait_for_service(service_name, service_config):
                        print(f"✅ {service_config['name']} started successfully")
                    else:
                        print(f"❌ Failed to start {service_config['name']}")
                        return False
                        
                except Exception as e:
                    print(f"❌ Error starting {service_config['name']}: {e}")
                    return False
            else:
                print(f"✅ {service_config['name']} already running")
        
        print("\n📊 Final service status:")
        final_status = await self.check_all_services()
        all_running = True
        
        for service_name, service_config in SERVICES.items():
            if final_status[service_name]["running"]:
                print(f"   ✅ {service_config['name']} (port {service_config['port']})")
            else:
                print(f"   ❌ {service_config['name']} (port {service_config['port']})")
                all_running = False
        
        return all_running
    
    def stop_all_services(self):
        """Stop all started services"""
        print("\n🛑 Stopping services...")
        
        for service_name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"   Stopping {SERVICES[service_name]['name']}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        
        self.processes.clear()
        print("✅ All services stopped")

async def main():
    """Main function"""
    manager = ServiceManager()
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        manager.stop_all_services()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        success = await manager.start_all_services()
        
        if success:
            print("\n🎉 All TF-IDF evaluation services are running!")
            print("\n📋 Next steps:")
            print("   1. Run the complete evaluation:")
            print("      python complete_tfidf_evaluation.py")
            print("\n   2. Or run individual tests:")
            print("      python tfidf_evaluation_complete.py")
            print("\n🔧 Services running:")
            for service_name, service_config in SERVICES.items():
                print(f"   • {service_config['name']}: http://localhost:{service_config['port']}")
            
            print(f"\n⏸️  Press Ctrl+C to stop all services")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("\n❌ Failed to start all services")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        manager.stop_all_services()
        await manager.close()
    
    return 0

if __name__ == "__main__":
    print("🎯 TF-IDF Evaluation Services Manager")
    print("📊 Starting services required for complete evaluation")
    print("")
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
