#!/usr/bin/env python3

"""
Simple evaluation runner for Quora hybrid search system.
Make sure the hybrid search service is running on localhost:8005 before running this script.
"""

import sys
import os
import subprocess
import time
import requests

def check_service_running(url="http://localhost:8005"):
    """Check if the hybrid search service is running."""
    try:
        response = requests.get(f"{url}/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_service():
    """Start the hybrid search service."""
    print("Starting hybrid search service...")
    service_script = "hybrid_quora_query_processing.py"
    
    if not os.path.exists(service_script):
        print(f"Error: {service_script} not found in current directory")
        return False
    
    try:
        # Start the service in background
        subprocess.Popen([sys.executable, service_script], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        print("Waiting for service to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_service_running():
                print("Service started successfully!")
                return True
            time.sleep(1)
        
        print("Service failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"Error starting service: {e}")
        return False

def run_evaluation():
    """Run the evaluation."""
    print("Running evaluation...")
    
    try:
        # Import and run the evaluation
        from evaluation_metrics import QuoraEvaluator
        
        evaluator = QuoraEvaluator()
        
        # Run evaluation on first 20 queries for quick testing
        print("Evaluating first 20 queries (for quick testing)...")
        results = evaluator.evaluate_all_queries(
            top_k=50,
            parallel=True,
            max_workers=2
        )
        
        if results:
            evaluator.print_summary(results)
            
            # Save results
            output_file = f"quick_evaluation_results_{int(time.time())}.json"
            evaluator.save_results(results, output_file)
            print(f"\nResults saved to: {output_file}")
            
            return True
        else:
            print("Evaluation failed - no results obtained")
            return False
            
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return False

def main():
    """Main function."""
    print("Quora Hybrid Search Evaluation Runner")
    print("="*50)
    
    # Check if service is already running
    if check_service_running():
        print("Hybrid search service is already running!")
    else:
        print("Hybrid search service is not running.")
        
        # Ask user if they want to start it
        response = input("Do you want to start the service? (y/n): ").strip().lower()
        if response != 'y':
            print("Please start the hybrid search service manually and try again.")
            return
        
        if not start_service():
            print("Failed to start the service. Please start it manually.")
            return
    
    # Run evaluation
    print("\nStarting evaluation process...")
    if run_evaluation():
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed!")

if __name__ == "__main__":
    main()
