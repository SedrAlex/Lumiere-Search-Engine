#!/usr/bin/env python3
"""
Day 1 Setup Script for TF-IDF Implementation
Ensures all dependencies and directories are ready for the 3-day implementation.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    required_packages = [
        "nltk>=3.8",
        "scikit-learn>=1.3.0", 
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
        "ir-datasets>=0.5.0",
        "tqdm>=4.66.0"
    ]
    
    for package in required_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    
    import nltk
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for dataset in datasets:
        try:
            print(f"  Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
        except Exception as e:
            print(f"❌ Failed to download {dataset}: {e}")
            return False
    
    print("✅ NLTK data downloaded successfully!")
    return True

def create_directory_structure():
    """Create required directory structure."""
    print("📁 Creating directory structure...")
    
    directories = [
        "models",
        "services/preprocessing",
        "services/representation", 
        "services/evaluation",
        "evaluation_results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Create __init__.py files for packages
    init_files = [
        "services/__init__.py",
        "services/preprocessing/__init__.py",
        "services/representation/__init__.py",
        "services/evaluation/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"  ✅ Created: {init_file}")
    
    print("✅ Directory structure created!")
    return True

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import nltk
        import sklearn
        import numpy
        import pandas
        import joblib
        import ir_datasets
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "="*60)
    print("🎯 DAY 1 SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. 📚 Open Google Colab")
    print("2. 📤 Upload train_tfidf_antique_colab.ipynb")
    print("3. ▶️  Run the notebook to train TF-IDF models")
    print("4. 📥 Download the generated model files:")
    print("   - tfidf_vectorizer_antique.joblib")
    print("   - tfidf_matrix_antique.joblib")
    print("   - inverted_index_antique.pkl")
    print("   - doc_mappings_antique.json")
    print("   - document_metadata_antique.json (optional)")
    print("   - training_statistics_antique.json (optional)")
    print("5. 📂 Place all files in the 'models/' directory")
    print("6. 🏃 Run evaluation:")
    print("   python run_tfidf_evaluation.py --quick")
    print("7. 🎯 Target: Achieve MAP >= 0.4")
    print("\n💡 Tips:")
    print("- Start with --quick (20 queries) for fast testing")
    print("- Use --test-components to debug issues")
    print("- Check evaluation reports for improvement suggestions")
    print("="*60)

def main():
    """Main setup function."""
    print("🚀 Starting Day 1 TF-IDF Setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        sys.exit(1)
    
    # Create directories
    if not create_directory_structure():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
