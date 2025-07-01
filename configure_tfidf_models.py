#!/usr/bin/env python3
"""
Configure TF-IDF Models for All Services
Ensures all TF-IDF services use the same vectorizer and models
"""

import os
import shutil
import sys
from pathlib import Path

# Configuration
MODELS_DIR = "./models"
TMP_DIR = "/tmp"

# Required model files (using the standard names)
REQUIRED_FILES = [
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.joblib", 
    "document_metadata.joblib"
]

def check_models_exist():
    """Check if all required model files exist in the models directory"""
    missing_files = []
    for filename in REQUIRED_FILES:
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    return missing_files

def copy_models_to_tmp():
    """Copy model files from models/ directory to /tmp/ for tfidf_service.py"""
    print("📁 Copying models to /tmp/ directory...")
    
    for filename in REQUIRED_FILES:
        source = os.path.join(MODELS_DIR, filename)
        dest = os.path.join(TMP_DIR, filename)
        
        try:
            shutil.copy2(source, dest)
            source_size = os.path.getsize(source) / (1024 * 1024)  # MB
            print(f"✅ Copied {filename} ({source_size:.2f} MB)")
        except Exception as e:
            print(f"❌ Failed to copy {filename}: {e}")
            return False
    
    return True

def update_tfidf_service_config():
    """Update TF-IDF service configuration to use proper model paths"""
    service_file = "services/representation/tfidf_service.py"
    
    if not os.path.exists(service_file):
        print(f"⚠️  Service file not found: {service_file}")
        return False
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Update paths to point to /tmp directory
        content = content.replace(
            'ANTIQUE_MODEL_PATH = "/tmp/tfidf_vectorizer.joblib"',
            'ANTIQUE_MODEL_PATH = "/tmp/tfidf_vectorizer.joblib"'
        )
        content = content.replace(
            'ANTIQUE_MATRIX_PATH = "/tmp/tfidf_matrix.joblib"',
            'ANTIQUE_MATRIX_PATH = "/tmp/tfidf_matrix.joblib"'
        )
        content = content.replace(
            'ANTIQUE_METADATA_PATH = "/tmp/document_metadata.joblib"',
            'ANTIQUE_METADATA_PATH = "/tmp/document_metadata.joblib"'
        )
        
        # Ensure USE_PRETRAINED_ANTIQUE is True
        content = content.replace(
            'USE_PRETRAINED_ANTIQUE = False',
            'USE_PRETRAINED_ANTIQUE = True'
        )
        
        with open(service_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {service_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {service_file}: {e}")
        return False

def verify_query_processor_config():
    """Verify TF-IDF query processor configuration"""
    service_file = "services/query_processing/tfidf_query_processor.py"
    
    if not os.path.exists(service_file):
        print(f"⚠️  Query processor file not found: {service_file}")
        return False
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Check if MODEL_BASE_PATH points to the correct location
        expected_path = "/Users/raafatmhanna/Desktop/custom-search-engine/backend/models"
        if expected_path in content:
            print(f"✅ Query processor correctly configured to use {expected_path}")
            return True
        else:
            print(f"⚠️  Query processor may need path updates")
            return False
            
    except Exception as e:
        print(f"❌ Failed to verify {service_file}: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the configured models"""
    print("\n" + "="*60)
    print("🎉 TF-IDF MODELS CONFIGURATION COMPLETE")
    print("="*60)
    print(f"""
📁 Model locations:
   - Source: {MODELS_DIR}/
   - TF-IDF Service: {TMP_DIR}/
   - Query Processor: {MODELS_DIR}/

🚀 To test the configuration:

1. START TF-IDF TEXT CLEANING SERVICE (Port 8005):
   python services/shared/tfidf_text_cleaning_service.py

2. START TF-IDF SERVICE (Port 8002):
   python services/representation/tfidf_service.py

3. START TF-IDF QUERY PROCESSOR (Port 8004):
   python services/query_processing/tfidf_query_processor.py

4. TEST THE SERVICES:
   # Check TF-IDF service status
   curl http://localhost:8002/status
   
   # Check query processor status
   curl http://localhost:8004/status
   
   # Test search
   curl -X POST http://localhost:8004/search \\
     -H "Content-Type: application/json" \\
     -d '{{"query": "information retrieval", "top_k": 5}}'

✅ All services should now use the same vectorizer and models!
""")

def main():
    """Main configuration function"""
    print("🔧 Configuring TF-IDF Models for All Services")
    print("="*60)
    
    # Check if models exist
    missing_files = check_models_exist()
    if missing_files:
        print(f"❌ Missing model files in {MODELS_DIR}/:")
        for filename in missing_files:
            print(f"   - {filename}")
        print("\nPlease ensure all required model files are present in the models directory.")
        return False
    
    print(f"✅ All required model files found in {MODELS_DIR}/")
    
    # Copy models to /tmp for tfidf_service.py
    if not copy_models_to_tmp():
        print("❌ Failed to copy models to /tmp directory")
        return False
    
    # Update TF-IDF service configuration
    if not update_tfidf_service_config():
        print("❌ Failed to update TF-IDF service configuration")
        return False
    
    # Verify query processor configuration
    verify_query_processor_config()
    
    # Show usage instructions
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
