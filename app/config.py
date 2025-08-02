from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os
from pathlib import Path
from typing import Optional
from secrets import token_urlsafe
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths using pathlib
VECTORSTORE_DIR = BASE_DIR / "data" / "vector_store"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
DATASETS_DIR = BASE_DIR / "data" / "datasets"

# Ensure directories exist using pathlib
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Convert back to string for printing if needed, but keep as Path objects for usage
print(f"DOCUMENTS_DIR: {str(DOCUMENTS_DIR)}")
print(f"VECTORSTORE_DIR: {str(VECTORSTORE_DIR)}")
print(f"Both directories exist: {DOCUMENTS_DIR.exists() and VECTORSTORE_DIR.exists()}")

# Embedding Models
EMBEDDING_MODEL: Optional = None
CROSS_ENCODER_MODEL: Optional = None

try:
    # Try to load BGE large model for better embeddings
    try:
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        print("BGE large embedding model loaded successfully")
    except Exception as e:
        print(f"Error loading BGE large model: {str(e)}")
        print("Falling back to all-MiniLM-L6-v2")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Fallback embedding model loaded successfully")
    
    # Try to load cross-encoder model
    try:
        from sentence_transformers import CrossEncoder
        
        CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Cross-encoder model loaded successfully")
    except Exception as e:
        print(f"Error loading cross-encoder model: {str(e)}")
        CROSS_ENCODER_MODEL = None
        
except Exception as e:
    print(f"Error loading embedding models: {str(e)}")
    # Fallback to HuggingFace embeddings if all else fails
    try:
        EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Fallback to HuggingFace embedding model successful")
    except Exception as e:
        print(f"Error loading fallback embedding model: {str(e)}")
        EMBEDDING_MODEL = None

# LLM setup - will be initialized on demand with fallback handling
LLM_MODEL_NAME = "llama3:8b"  # Using llama3:8b model

# Ollama API URL
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval settings - Updated for sliding window chunking
CHUNK_SIZE = 800  # Increased from 500 for better context preservation
CHUNK_OVERLAP = 300  # Increased from 50 for sliding window effect
RETRIEVAL_K = 8  # Final number of documents to retrieve after reranking (increased for better coverage)
RETRIEVAL_CANDIDATES = 30  # Number of initial candidates to retrieve before reranking (increased for better coverage)

# Security Configuration
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", token_urlsafe(32))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # Default for demo, should be changed

# File Upload Security
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
ALLOWED_EXTENSIONS = {'.pdf'}
QUARANTINE_DIR = BASE_DIR / "quarantine"

# Ensure quarantine directory exists
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

# Sentry Configuration - Security: Use environment variables
SENTRY_DSN = os.getenv("SENTRY_DSN", "")  # Security: Moved to environment variable
if not SENTRY_DSN:
    print("Warning: SENTRY_DSN not set in environment variables. Sentry monitoring disabled.")
else:
    print("Sentry monitoring enabled") 