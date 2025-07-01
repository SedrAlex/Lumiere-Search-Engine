#!/usr/bin/env python3
"""
Setup Pre-trained ANTIQUE TF-IDF Models
Downloads and configures pre-trained models for the TF-IDF service
"""

import os
import sys
import requests
import shutil
from pathlib import Path
import joblib
import json

# Configuration
MODEL_DIRECTORY = "/tmp"  # Where to store models
BACKUP_DIRECTORY = "./models"  # Backup location in current directory

def create_directories():
    """Create necessary directories"""
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    os.makedirs(BACKUP_DIRECTORY, exist_ok=True)
    print(f"✅ Created directories: {MODEL_DIRECTORY}, {BACKUP_DIRECTORY}")

def download_from_url(url: str, filename: str, destination: str):
    """Download a file from URL"""
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filepath = os.path.join(destination, filename)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def copy_local_files(source_dir: str, filenames: list):
    """Copy files from local directory to model directory"""
    success_count = 0
    
    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(MODEL_DIRECTORY, filename)
        backup_path = os.path.join(BACKUP_DIRECTORY, filename)
        
        if os.path.exists(source_path):
            try:
                # Copy to /tmp/
                shutil.copy2(source_path, dest_path)
                # Copy to backup location
                shutil.copy2(source_path, backup_path)
                print(f"✅ Copied {filename} to {MODEL_DIRECTORY} and {BACKUP_DIRECTORY}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to copy {filename}: {e}")
        else:
            print(f"⚠️  File not found: {source_path}")
    
    return success_count

def verify_model_files():
    """Verify that all required model files exist and are valid"""
    required_files = [
        "tfidf_vectorizer.joblib",
        "tfidf_matrix.joblib", 
        "document_metadata.joblib"
    ]
    
    print("🔍 Verifying model files...")
    
    all_valid = True
    for filename in required_files:
        filepath = os.path.join(MODEL_DIRECTORY, filename)
        
        if os.path.exists(filepath):
            try:
                # Try to load the file to verify it's valid
                data = joblib.load(filepath)
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"✅ {filename}: {file_size:.2f} MB - Valid")
            except Exception as e:
                print(f"❌ {filename}: Invalid or corrupted - {e}")
                all_valid = False
        else:
            print(f"❌ {filename}: Not found")
            all_valid = False
    
    return all_valid

def update_service_configuration():
    """Update TF-IDF service to use pre-trained models"""
    service_file = "services/representation/tfidf_service.py"
    
    if not os.path.exists(service_file):
        print(f"⚠️  Service file not found: {service_file}")
        return False
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
        
        # Update the configuration
        updated_content = content.replace(
            'USE_PRETRAINED_ANTIQUE = False',
            'USE_PRETRAINED_ANTIQUE = True'
        )
        
        # Ensure paths are correct
        lines = updated_content.split('\n')
        for i, line in enumerate(lines):
            if 'ANTIQUE_MODEL_PATH' in line:
                lines[i] = f'ANTIQUE_MODEL_PATH = "{MODEL_DIRECTORY}/tfidf_vectorizer.joblib"'
            elif 'ANTIQUE_MATRIX_PATH' in line:
                lines[i] = f'ANTIQUE_MATRIX_PATH = "{MODEL_DIRECTORY}/tfidf_matrix.joblib"'
            elif 'ANTIQUE_METADATA_PATH' in line:
                lines[i] = f'ANTIQUE_METADATA_PATH = "{MODEL_DIRECTORY}/document_metadata.joblib"'
        
        updated_content = '\n'.join(lines)
        
        # Write back to file
        with open(service_file, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated service configuration in {service_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update service configuration: {e}")
        return False

def show_integration_info():
    """Show information about how to use the pre-trained models"""
    print("\n" + "="*60)
    print("🎉 PRE-TRAINED MODEL SETUP COMPLETE")
    print("="*60)
    print(f"""
📁 Model files location: {MODEL_DIRECTORY}/
📁 Backup location: {BACKUP_DIRECTORY}/

📋 Files installed:
   - antique_enhanced_tfidf_vectorizer
   - antique_enhanced_tfidf_matrix 
   - antique_enhanced_document_metadata

🚀 Next steps:

1. START THE TF-IDF SERVICE:
   cd /Users/raafatmhanna/Desktop/custom-search-engine/backend
   python services/representation/tfidf_service.py

2. TEST WITH POSTMAN:
   - GET http://localhost:8002/status
   - Should show: "using_pretrained": true

3. SEARCH EXAMPLES:
   POST http://localhost:8002/search
   {{
     "query": "information retrieval systems",
     "top_k": 5
   }}

✅ Benefits:
   - 400K+ ANTIQUE documents ready for search
   - No training time required
   - Production-ready performance
   - Consistent results across deployments

🔧 If you need to retrain or update models:
   - Use the enhanced_colab_tfidf_training.py script in Google Colab
   - Download the new models and run this setup script again
""")

def main():
    """Main setup function"""
    print("🚀 Setting up Pre-trained ANTIQUE TF-IDF Models")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Option 1: Look for models in current directory (downloaded from Colab)
    model_files = [
        "tfidf_vectorizer.joblib",
        "tfidf_matrix.joblib",
        "document_metadata.joblib"
    ]
    
    print("\n📂 Looking for model files in current directory...")
    current_dir_files = copy_local_files(".", model_files)
    
    if current_dir_files == 0:
        print("\n📂 Looking for model files in Downloads directory...")
        downloads_dir = os.path.expanduser("~/Downloads")
        downloads_files = copy_local_files(downloads_dir, model_files)
        
        if downloads_files == 0:
            print("\n❌ No pre-trained model files found!")
            print("\n📋 TO GET PRE-TRAINED MODELS:")
            print("1. Open Google Colab")
            print("2. Upload and run enhanced_colab_tfidf_training.py")
            print("3. Download the generated .joblib files")
            print("4. Place them in this directory and run this script again")
            print("\nAlternatively, you can use the TF-IDF service without pre-trained models")
            print("by training it with your own documents using the /index endpoint.")
            return False
    
    # Verify all files are valid
    if verify_model_files():
        print("\n✅ All model files verified successfully!")
        
        # Update service configuration
        if update_service_configuration():
            show_integration_info()
            return True
        else:
            print("\n⚠️  Models are ready but service configuration update failed.")
            print("You may need to manually update USE_PRETRAINED_ANTIQUE = True")
            return False
    else:
        print("\n❌ Some model files are missing or invalid!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
