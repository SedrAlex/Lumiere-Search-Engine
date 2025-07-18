#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Service for Quora Dataset
Combines document retrieval using embeddings with text generation for conversational responses.
"""

import os
import sys
import logging
import json
import time
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Add the parent directory to the path to import the embedding service
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Try to import transformers for generation
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

# Import the existing embedding service
from services.query_processing.quora.embedding_quora_query_processing import QuoraQueryProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Context for maintaining conversation state"""
    conversation_id: str
    messages: List[Dict[str, str]]
    last_retrieval_results: List[Dict[str, Any]]
    timestamp: datetime

class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_context_docs: int = 5
    temperature: float = 0.7
    max_length: int = 512

class RAGResponse(BaseModel):
    response: str
    conversation_id: str
    retrieved_documents: List[Dict[str, Any]]
    generation_time: float
    retrieval_time: float
    total_time: float

class QuoraRAGService:
    """
    RAG service that combines document retrieval with text generation
    for conversational search over Quora dataset.
    """
    
    def __init__(self, generation_model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the RAG service.
        
        Args:
            generation_model_name: Hugging Face model name for text generation
        """
        self.generation_model_name = generation_model_name
        self.retrieval_processor = None
        self.generation_pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_conversations = 100
        self.conversation_ttl = 3600  # 1 hour in seconds
        
        # Initialize components
        self._initialize_retrieval()
        if TRANSFORMERS_AVAILABLE:
            self._initialize_generation()
        else:
            logger.warning("Transformers not available. Generation will use template-based responses.")
    
    def _initialize_retrieval(self):
        """Initialize the retrieval component using existing embedding service."""
        try:
            logger.info("Initializing retrieval component...")
            # Initialize with database-based approach instead of joblib files
            self.retrieval_processor = QuoraQueryProcessor(
                text_processing_service_url="http://localhost:5003",
                use_faiss=True  # Use FAISS for faster retrieval
            )
            
            # Override the joblib loading to use database instead
            self._load_embeddings_from_database()
            
            logger.info("Retrieval component initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval component: {e}")
            raise
    
    def _initialize_generation(self):
        """Initialize the text generation component."""
        try:
            logger.info(f"Initializing generation model: {self.generation_model_name}")
            
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'CUDA' if device == 0 else 'CPU'}")
            
            # Try to load a smaller, faster model first
            small_models = [
                "distilgpt2",  # Much smaller and faster
                "gpt2",       # Still smaller than DialoGPT
                self.generation_model_name  # Original model as fallback
            ]
            
            model_loaded = False
            for model_name in small_models:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    
                    # Set timeout and other parameters to prevent hanging
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        local_files_only=False
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        local_files_only=False,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    
                    # Add padding token if not present
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Create generation pipeline
                    self.generation_pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    logger.info(f"Successfully loaded model: {model_name}")
                    self.generation_model_name = model_name  # Update to reflect actual loaded model
                    model_loaded = True
                    break
                    
                except Exception as model_error:
                    logger.warning(f"Failed to load model {model_name}: {model_error}")
                    continue
            
            if not model_loaded:
                raise Exception("Failed to load any generation model")
                
            logger.info("Generation component initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize generation component: {e}")
            # Fallback to template-based generation
            self.generation_pipeline = None
            self.tokenizer = None
            self.model = None
            logger.warning("Falling back to template-based generation")
    
    def _load_embeddings_from_database(self):
        """Load embeddings from database instead of joblib files."""
        try:
            logger.info("Loading embeddings from Quora database...")
            
            # Load documents from SQLite database
            sqlite_db_path = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_documents.db'
            
            if not os.path.exists(sqlite_db_path):
                raise FileNotFoundError(f"Quora database not found at {sqlite_db_path}")
            
            conn = sqlite3.connect(sqlite_db_path)
            cursor = conn.cursor()
            
            # Get all documents with processed text
            cursor.execute('SELECT doc_id, processed_text FROM documents WHERE processed_text IS NOT NULL')
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                raise ValueError("No documents found in Quora database")
            
            # Extract document IDs and texts
            doc_ids = [row[0] for row in rows]
            doc_texts = [row[1] for row in rows]
            
            logger.info(f"Loaded {len(doc_ids)} documents from Quora database")
            
            # Load the SentenceTransformer model for encoding
            model_path = '/Users/raafatmhanna/Downloads/quora_Embeddings/sentence-transformers_all-MiniLM-L6-v2'
            if os.path.exists(model_path):
                model = SentenceTransformer(model_path)
                logger.info("SentenceTransformer model loaded successfully from local path.")
            else:
                logger.warning("Local model not found, loading from HuggingFace...")
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded from HuggingFace.")
            
            # Check if we need to compute embeddings
            embeddings_cache_path = '/Users/raafatmhanna/Desktop/custom-search-engine/backend/services/query_processing/quora/quora_embeddings_cache.pkl'
            
            if os.path.exists(embeddings_cache_path):
                logger.info("Loading cached embeddings...")
                import pickle
                with open(embeddings_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Verify cache is valid
                if (len(cache_data.get('doc_ids', [])) == len(doc_ids) and
                    cache_data.get('doc_ids') == doc_ids):
                    doc_embeddings = cache_data['doc_embeddings']
                    logger.info(f"Using cached embeddings for {len(doc_ids)} documents")
                else:
                    logger.info("Cache invalid, computing new embeddings...")
                    doc_embeddings = self._compute_embeddings(model, doc_texts)
                    self._save_embeddings_cache(doc_ids, doc_texts, doc_embeddings, embeddings_cache_path)
            else:
                logger.info("No cache found, computing embeddings...")
                doc_embeddings = self._compute_embeddings(model, doc_texts)
                self._save_embeddings_cache(doc_ids, doc_texts, doc_embeddings, embeddings_cache_path)
            
            # Set the data in the processor
            self.retrieval_processor.doc_ids_cache = doc_ids
            self.retrieval_processor.doc_texts_cache = doc_texts
            self.retrieval_processor.doc_embeddings = doc_embeddings
            self.retrieval_processor.model = model
            
            logger.info(f"Successfully loaded {len(doc_ids)} documents from Quora database")
            
        except Exception as e:
            logger.error(f"Error loading embeddings from database: {e}")
            raise
    
    def _compute_embeddings(self, model, doc_texts):
        """Compute embeddings for document texts."""
        logger.info(f"Computing embeddings for {len(doc_texts)} documents...")
        
        # Compute embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i+batch_size]
            batch_embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=True)
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(doc_texts) + batch_size - 1)//batch_size}")
        
        return np.array(all_embeddings)
    
    def _save_embeddings_cache(self, doc_ids, doc_texts, doc_embeddings, cache_path):
        """Save embeddings to cache file."""
        try:
            import pickle
            cache_data = {
                'doc_ids': doc_ids,
                'doc_texts': doc_texts,
                'doc_embeddings': doc_embeddings,
                'timestamp': time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Embeddings cache saved to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {e}")
    
    def _clean_conversations(self):
        """Remove expired conversations."""
        current_time = datetime.now()
        expired_ids = []
        
        for conv_id, context in self.conversations.items():
            if (current_time - context.timestamp).total_seconds() > self.conversation_ttl:
                expired_ids.append(conv_id)
        
        for conv_id in expired_ids:
            del self.conversations[conv_id]
        
        logger.info(f"Cleaned {len(expired_ids)} expired conversations")
    
    def _get_or_create_conversation(self, conversation_id: Optional[str]) -> ConversationContext:
        """Get existing conversation or create new one."""
        if conversation_id and conversation_id in self.conversations:
            # Update timestamp
            self.conversations[conversation_id].timestamp = datetime.now()
            return self.conversations[conversation_id]
        
        # Create new conversation
        new_id = conversation_id or f"conv_{int(time.time() * 1000)}"
        context = ConversationContext(
            conversation_id=new_id,
            messages=[],
            last_retrieval_results=[],
            timestamp=datetime.now()
        )
        
        # Clean old conversations if needed
        if len(self.conversations) >= self.max_conversations:
            self._clean_conversations()
        
        self.conversations[new_id] = context
        return context
    
    def retrieve_context(self, query: str, max_docs: int = 5) -> tuple[List[Dict[str, Any]], float]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: User query
            max_docs: Maximum number of documents to retrieve
            
        Returns:
            Tuple of (retrieved documents, retrieval time)
        """
        start_time = time.time()
        
        try:
            # Use existing embedding service for retrieval
            results = self.retrieval_processor.search_similar_documents(
                query=query,
                top_k=max_docs,
                use_faiss=True
            )
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
            
            return results, retrieval_time
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return [], time.time() - start_time
    
    def _build_context_prompt(self, query: str, documents: List[Dict[str, Any]], 
                            conversation_history: List[Dict[str, str]]) -> str:
        """Build context prompt for generation."""
        
        # Add conversation history
        context_parts = []
        
        if conversation_history:
            context_parts.append("Previous conversation:")
            for msg in conversation_history[-6:]:  # Last 6 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                context_parts.append(f"{role.capitalize()}: {content}")
            context_parts.append("")
        
        # Add retrieved documents
        context_parts.append("Relevant information from Quora:")
        for i, doc in enumerate(documents[:3], 1):  # Top 3 documents
            doc_text = doc.get('document', '')[:300]  # Truncate long documents
            context_parts.append(f"{i}. {doc_text}")
        
        context_parts.append("")
        context_parts.append(f"User question: {query}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, documents: List[Dict[str, Any]], 
                         conversation_history: List[Dict[str, str]],
                         temperature: float = 0.7, max_length: int = 512) -> tuple[str, float]:
        """
        Generate response using retrieved documents and conversation history.
        
        Args:
            query: User query
            documents: Retrieved documents
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_length: Maximum response length
            
        Returns:
            Tuple of (generated response, generation time)
        """
        start_time = time.time()
        
        if not documents:
            response = "I couldn't find relevant information to answer your question. Could you please rephrase or ask something else?"
            return response, time.time() - start_time
        
        try:
            if self.generation_pipeline is None:
                # Template-based fallback
                response = self._generate_template_response(query, documents)
            else:
                # Neural generation
                prompt = self._build_context_prompt(query, documents, conversation_history)
                
                # Generate response
                outputs = self.generation_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                response = outputs[0]['generated_text'].strip()
                
                # Clean up response
                response = self._clean_generated_response(response)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.3f}s")
            
            return response, generation_time
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            # Fallback to template-based response
            response = self._generate_template_response(query, documents)
            return response, time.time() - start_time
    
    def _generate_template_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate template-based response when neural generation is not available."""
        
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Find the most relevant document
        best_doc = documents[0]
        best_text = best_doc.get('document', '')
        similarity_score = best_doc.get('similarity_score', 0)
        
        # Create a template response
        if similarity_score > 0.8:
            response = f"Based on similar questions from Quora, here's what I found:\n\n{best_text}"
        elif similarity_score > 0.6:
            response = f"I found some related information that might help:\n\n{best_text}"
        else:
            response = f"Here's some potentially relevant information:\n\n{best_text}"
        
        # Add additional context if available
        if len(documents) > 1:
            response += f"\n\nI also found {len(documents)-1} other related questions that might be helpful."
        
        return response
    
    def _clean_generated_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove potential prompt artifacts
        response = response.replace("Assistant:", "").strip()
        response = response.replace("User:", "").strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def process_message(self, message: str, conversation_id: Optional[str] = None,
                       max_context_docs: int = 5, temperature: float = 0.7,
                       max_length: int = 512) -> RAGResponse:
        """
        Process a user message and generate a response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            max_context_docs: Maximum documents to retrieve
            temperature: Generation temperature
            max_length: Maximum response length
            
        Returns:
            RAGResponse with generated response and metadata
        """
        total_start_time = time.time()
        
        # Get or create conversation context
        conversation = self._get_or_create_conversation(conversation_id)
        
        # Retrieve relevant documents
        documents, retrieval_time = self.retrieve_context(message, max_context_docs)
        
        # Generate response
        response, generation_time = self.generate_response(
            message, documents, conversation.messages, temperature, max_length
        )
        
        # Update conversation history
        conversation.messages.append({"role": "user", "content": message})
        conversation.messages.append({"role": "assistant", "content": response})
        conversation.last_retrieval_results = documents
        
        # Keep conversation history manageable
        if len(conversation.messages) > 20:
            conversation.messages = conversation.messages[-20:]
        
        total_time = time.time() - total_start_time
        
        return RAGResponse(
            response=response,
            conversation_id=conversation.conversation_id,
            retrieved_documents=documents,
            generation_time=generation_time,
            retrieval_time=retrieval_time,
            total_time=total_time
        )

# FastAPI app setup
app = FastAPI(
    title="Quora RAG Service",
    description="Retrieval-Augmented Generation service for Quora dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup."""
    global rag_service
    try:
        rag_service = QuoraRAGService()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # Create a basic service that can at least handle retrieval
        try:
            rag_service = QuoraRAGService()
            logger.info("RAG service initialized with limited functionality")
        except Exception as fallback_error:
            logger.error(f"Complete failure to initialize RAG service: {fallback_error}")
            rag_service = None

@app.post("/chat", response_model=RAGResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Chat endpoint for RAG conversations.
    
    Args:
        chat_message: ChatMessage with user query and parameters
        
    Returns:
        RAGResponse with generated response and metadata
    """
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        response = rag_service.process_message(
            message=chat_message.message,
            conversation_id=chat_message.conversation_id,
            max_context_docs=chat_message.max_context_docs,
            temperature=chat_message.temperature,
            max_length=chat_message.max_length
        )
        return response
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Quora RAG Service"}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    if conversation_id not in rag_service.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = rag_service.conversations[conversation_id]
    return {
        "conversation_id": conversation_id,
        "messages": conversation.messages,
        "last_retrieval_results": conversation.last_retrieval_results
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    if conversation_id in rag_service.conversations:
        del rag_service.conversations[conversation_id]
        return {"message": "Conversation deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    return {
        "active_conversations": len(rag_service.conversations),
        "generation_model": rag_service.generation_model_name,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "retrieval_stats": rag_service.retrieval_processor.get_service_stats() if rag_service.retrieval_processor else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
