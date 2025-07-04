#!/usr/bin/env python3
"""
Script to run the Antique dataset text cleaning process.
This script processes the original antique documents and creates cleaned versions
stored in the database, ready for embedding generation.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from services.shared.antique_text_cleaning_service import AntiqueTextCleaningService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the text cleaning process."""
    print("=" * 80)
    print("🧹 ANTIQUE DATASET TEXT CLEANING SERVICE")
    print("=" * 80)
    print()
    
    # Initialize the cleaning service
    logger.info("Initializing Antique text cleaning service...")
    service = AntiqueTextCleaningService(
        db_path="data/search_engine.db",
        enable_spell_check=True  # Enable advanced spell checking
    )
    
    try:
        # Get initial statistics
        print("📊 Checking current dataset status...")
        total_docs = await service.get_document_count()
        print(f"   Total antique documents in database: {total_docs:,}")
        
        # Check if cleaning is already done
        stats = await service.get_cleaning_statistics()
        if stats.get('cleaned_documents', 0) > 0:
            print(f"   Already cleaned documents: {stats['cleaned_documents']:,}")
            print(f"   Completion: {stats['completion_percentage']:.1f}%")
            
            if stats['completion_percentage'] >= 100:
                print("✅ All documents are already cleaned!")
                export_choice = input("\n🤔 Would you like to export the cleaned data for Colab? (y/n): ").lower()
                if export_choice == 'y':
                    print("\n📤 Exporting cleaned data...")
                    export_path = await service.export_cleaned_data_for_colab(
                        output_file="antique_cleaned_for_embeddings.json",
                        max_documents=50000  # Limit for Colab memory constraints
                    )
                    print(f"✅ Exported cleaned data to: {export_path}")
                return
        
        print()
        print("🚀 Starting text cleaning process...")
        print("   This will apply enhanced text cleaning including:")
        print("   - HTML tag removal and decoding")
        print("   - Unicode normalization")
        print("   - Contraction expansion")
        print("   - Spell checking and correction")
        print("   - Stemming and lemmatization")
        print("   - Stopword removal")
        print("   - Special pattern normalization")
        print()
        
        # Ask for confirmation
        confirm = input("Continue with text cleaning? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Text cleaning cancelled.")
            return
        
        # Process documents in batches
        print("\n🔄 Processing documents...")
        
        # For demo purposes, let's process a subset first
        max_docs = None
        batch_size = 1000
        
        # Ask if user wants to limit processing
        limit_choice = input("\n🎯 Process all documents or limit for testing? (all/limit): ").lower()
        if limit_choice == 'limit':
            try:
                max_docs = int(input("   Enter maximum number of documents to process: "))
                print(f"   Limited to {max_docs:,} documents")
            except ValueError:
                print("   Invalid input, processing all documents")
                max_docs = None
        
        # Start processing
        await service.process_all_antique_documents(
            batch_size=batch_size,
            max_documents=max_docs
        )
        
        print("\n🎉 Text cleaning completed successfully!")
        
        # Export cleaned data for Colab
        print("\n📤 Exporting cleaned data for Colab...")
        export_path = await service.export_cleaned_data_for_colab(
            output_file="antique_cleaned_for_embeddings.json",
            max_documents=50000  # Limit for Colab memory constraints
        )
        
        print(f"✅ Cleaned data exported to: {export_path}")
        print()
        print("📋 NEXT STEPS:")
        print("1. Upload the exported JSON file to Google Colab")
        print("2. Use the BERTweet embedding notebook to generate embeddings")
        print("3. The cleaned text is optimized for embedding generation")
        print()
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during text cleaning: {e}")
        print(f"\n❌ Error: {e}")
        return
    
    print("=" * 80)
    print("✅ ANTIQUE TEXT CLEANING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
