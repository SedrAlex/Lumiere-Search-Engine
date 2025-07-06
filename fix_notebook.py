#!/usr/bin/env python3

import json
import re

def clean_notebook():
    # Read the original file with error handling
    with open('Quora_Data_Processing_and_Embeddings_Optimized.ipynb', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Replace all problematic Unicode characters with ASCII equivalents
    replacements = {
        '📦': '[PACKAGE]',
        '⚠️': '[WARNING]',
        '🎉': '[SUCCESS]',
        '❌': '[ERROR]',
        '📁': '[FILES]',
        '🚀': '[READY]',
        '🔥': '[HOT]',
        '✅': '[OK]',
        '📊': '[STATS]',
        '→': '->',
        '&': 'and',
        # Additional problematic characters
        '\u26a0': '[WARNING]',
        '\ufe0f': '',
        '\u1f389': '[SUCCESS]',
        '\u274c': '[ERROR]',
        '\u1f4c1': '[FILES]',
        '\u1f680': '[READY]',
        '\u1f4e6': '[PACKAGE]',
        '\u1f525': '[HOT]',
        '\u2705': '[OK]',
        '\u1f4ca': '[STATS]',
        '\u2192': '->',
    }
    
    # Apply all replacements
    for unicode_char, ascii_equiv in replacements.items():
        content = content.replace(unicode_char, ascii_equiv)
    
    # Remove any remaining non-printable characters (except newlines, tabs, and spaces)
    content = re.sub(r'[^\x20-\x7E\r\n\t]', '', content)
    
    # Parse and reformat the JSON to ensure validity
    try:
        data = json.loads(content)
        # Write the cleaned JSON with proper formatting
        with open('Quora_Data_Processing_and_Embeddings_Optimized_FIXED.ipynb', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
        
        print("✅ Successfully created clean notebook: Quora_Data_Processing_and_Embeddings_Optimized_FIXED.ipynb")
        print(f"Number of cells: {len(data['cells'])}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        # Try to fix common JSON issues
        print("Attempting manual fixes...")
        
        # Fix common issues like trailing commas, unescaped quotes, etc.
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas before }
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas before ]
        
        # Try to parse again
        try:
            data = json.loads(content)
            with open('Quora_Data_Processing_and_Embeddings_Optimized_FIXED.ipynb', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=True)
            
            print("✅ Successfully created clean notebook after manual fixes")
            print(f"Number of cells: {len(data['cells'])}")
            return True
            
        except Exception as e2:
            print(f"❌ Still failed after manual fixes: {e2}")
            return False

if __name__ == "__main__":
    clean_notebook()
