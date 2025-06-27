"""
RAG (Retrieval-Augmented Generation) Service
Enhances search results with generated content using vector stores
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np

# ChromaDB for vector store
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

# Transformers for text generation
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pipeline = None

class RAGService:
    """Service for Retrieval-Augmented Generation"""
    
    def __init__(self):
        self.vector_store = None
        self.text_generator = None
        self.embedding_model = None
        
        # Initialize components
        self._initialize_vector_store()
        self._initialize_text_generator()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            if chromadb:
                # Create in-memory ChromaDB instance
                self.vector_store = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory="data/chroma_db"
                ))
                print("✅ ChromaDB vector store initialized")
            else:
                print("⚠️ ChromaDB not available, using fallback")
        except Exception as e:
            print(f"⚠️ Error initializing vector store: {e}")
    
    def _initialize_text_generator(self):
        """Initialize text generation model"""
        try:
            if pipeline:
                # Use a lightweight model for text generation
                self.text_generator = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    tokenizer="distilgpt2",
                    max_length=150,
                    do_sample=True,
                    temperature=0.7
                )
                print("✅ Text generation model initialized")
            else:
                print("⚠️ Transformers not available, RAG will be limited")
        except Exception as e:
            print(f"⚠️ Error initializing text generator: {e}")
            self.text_generator = None
    
    async def enhance_results(self, query: str, results: List[Any]) -> List[Any]:
        """Enhance search results with RAG"""
        if not self.text_generator:
            return results
        
        print(f"🤖 Enhancing {len(results)} results with RAG...")
        
        enhanced_results = []
        
        for result in results:
            try:
                # Generate enhanced summary
                enhanced_content = await self._generate_enhanced_content(
                    query, result.title, result.content
                )
                
                # Update result with enhanced content
                if enhanced_content:
                    result.content = enhanced_content
                
                enhanced_results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error enhancing result {result.doc_id}: {e}")
                enhanced_results.append(result)  # Keep original if enhancement fails
        
        return enhanced_results
    
    async def _generate_enhanced_content(self, query: str, title: str, content: str) -> str:
        """Generate enhanced content using the text generation model"""
        try:
            # Create a prompt that combines query context with document content
            prompt = f"Query: {query}\nDocument: {title}\nContent: {content[:100]}...\nSummary:"
            
            # Generate enhanced content
            if self.text_generator:
                generated = self.text_generator(
                    prompt,
                    max_length=len(prompt.split()) + 50,
                    num_return_sequences=1,
                    pad_token_id=50256  # GPT-2 pad token
                )
                
                enhanced_text = generated[0]['generated_text']
                
                # Extract only the generated part (after "Summary:")
                if "Summary:" in enhanced_text:
                    summary = enhanced_text.split("Summary:")[-1].strip()
                    if summary:
                        return f"{content[:200]}... [RAG Summary: {summary}]"
            
            return content
            
        except Exception as e:
            print(f"⚠️ Error generating enhanced content: {e}")
            return content
    
    async def create_vector_store_from_documents(self, documents: List[Any], dataset_name: str):
        """Create vector store from documents for RAG"""
        if not self.vector_store:
            return
        
        try:
            # Create or get collection
            collection_name = f"{dataset_name}_rag"
            
            try:
                collection = self.vector_store.get_collection(collection_name)
                print(f"📚 Using existing collection: {collection_name}")
            except:
                collection = self.vector_store.create_collection(collection_name)
                print(f"📚 Created new collection: {collection_name}")
            
            # Prepare documents for vector store
            doc_texts = []
            doc_ids = []
            doc_metadata = []
            
            for doc in documents[:1000]:  # Limit for demo purposes
                doc_texts.append(f"{doc.title} {doc.text}")
                doc_ids.append(doc.doc_id)
                doc_metadata.append({
                    "title": doc.title,
                    "dataset": dataset_name
                })
            
            # Add documents to collection
            collection.add(
                documents=doc_texts,
                ids=doc_ids,
                metadatas=doc_metadata
            )
            
            print(f"✅ Added {len(doc_texts)} documents to vector store")
            
        except Exception as e:
            print(f"⚠️ Error creating vector store: {e}")
    
    async def retrieve_similar_documents(self, query: str, dataset_name: str, n_results: int = 5) -> List[Dict]:
        """Retrieve similar documents from vector store"""
        if not self.vector_store:
            return []
        
        try:
            collection_name = f"{dataset_name}_rag"
            collection = self.vector_store.get_collection(collection_name)
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            similar_docs = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    similar_docs.append({
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return similar_docs
            
        except Exception as e:
            print(f"⚠️ Error retrieving similar documents: {e}")
            return []
    
    async def generate_query_answer(self, query: str, context_documents: List[str]) -> str:
        """Generate an answer to a query based on context documents"""
        if not self.text_generator:
            return "RAG text generation not available."
        
        try:
            # Combine context documents
            context = " ".join(context_documents[:3])  # Use top 3 documents
            context = context[:500]  # Limit context length
            
            # Create prompt for answer generation
            prompt = f"Question: {query}\nContext: {context}\nAnswer:"
            
            # Generate answer
            generated = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + 30,
                num_return_sequences=1,
                pad_token_id=50256
            )
            
            generated_text = generated[0]['generated_text']
            
            # Extract answer
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
                return answer if answer else "No answer generated."
            
            return "Could not generate answer."
            
        except Exception as e:
            print(f"⚠️ Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def enhance_with_context(self, query: str, search_results: List[Any], dataset_name: str) -> Dict[str, Any]:
        """Enhance search results with contextual information"""
        enhanced_response = {
            "original_results": search_results,
            "context_summary": "",
            "generated_answer": "",
            "similar_documents": []
        }
        
        try:
            # Get similar documents from vector store
            similar_docs = await self.retrieve_similar_documents(query, dataset_name)
            enhanced_response["similar_documents"] = similar_docs
            
            # Extract content from search results
            result_contents = [result.content for result in search_results[:3]]
            
            # Generate contextual answer
            if result_contents:
                answer = await self.generate_query_answer(query, result_contents)
                enhanced_response["generated_answer"] = answer
            
            # Create context summary
            if similar_docs:
                context_summary = f"Found {len(similar_docs)} similar documents. "
                context_summary += f"Generated answer based on top search results."
                enhanced_response["context_summary"] = context_summary
            
        except Exception as e:
            print(f"⚠️ Error enhancing with context: {e}")
            enhanced_response["error"] = str(e)
        
        return enhanced_response
    
    def get_rag_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get RAG statistics for a dataset"""
        stats = {
            "vector_store_available": self.vector_store is not None,
            "text_generator_available": self.text_generator is not None,
            "collection_exists": False,
            "document_count": 0
        }
        
        if self.vector_store:
            try:
                collection_name = f"{dataset_name}_rag"
                collection = self.vector_store.get_collection(collection_name)
                stats["collection_exists"] = True
                stats["document_count"] = collection.count()
            except:
                pass
        
        return stats
