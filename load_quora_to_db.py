#!/usr/bin/env python3
"""
Script to load Quora dataset files (docs.tsv, queries.tsv, qrels.tsv) to database
"""

from services.data.quora_loader_service import QuoraDatabaseLoader

def main():
    """Main function to load Quora dataset to database"""
    print("🚀 Loading Quora Dataset to Database")
    print("=" * 50)
    
    # Initialize the database loader
    loader = QuoraDatabaseLoader()
    
    # Load all files
    results = loader.load_all_files()
    
    print("\n📋 Loading Results:")
    print("-" * 20)
    for file_type, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {file_type.upper()}: {status}")
    
    # Check if all files were loaded successfully
    if all(results.values()):
        print("\n🎉 All Quora dataset files loaded successfully!")
    else:
        print("\n⚠️  Some files failed to load. Check the errors above.")
    
    print("\n✅ Process completed!")

if __name__ == "__main__":
    main()
