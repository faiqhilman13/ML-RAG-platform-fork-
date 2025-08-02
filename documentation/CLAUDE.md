# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Retrieval-Augmented Generation (RAG) chatbot built with FastAPI and React that provides intelligent document-based question answering with advanced source filtering capabilities. The system uses BAAI/bge-large-en-v1.5 embeddings with cross-encoder reranking and llama3:8b LLM via Ollama.

## Development Commands

### Backend Development
```bash
# Start the FastAPI server
python -m uvicorn app.main:app --reload --port 8080

# Alternative: Run from main.py (uses port 8001)
cd chatbot-RAG
python app/main.py

# Install Python dependencies
pip install -r requirements.txt

# Run tests
pytest
pytest tests/test_auth.py  # Run specific test file
```

### Frontend Development
```bash
# Navigate to frontend directory
cd frontend-react

# Install dependencies
npm install

# Start development server (runs on port 5170)
npm start

# Build for production
npm run build

# Run tests
npm test
```

### Ollama Setup (Required)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3:8b
ollama serve  # Start Ollama service
```

## Architecture Overview

### Core System Components

**Backend (FastAPI):**
- `app/main.py` - Main FastAPI application with routing and middleware
- `app/config.py` - Configuration including model settings and security
- `app/retrievers/rag.py` - Core RAG retriever with hybrid retrieval
- `app/routers/` - API endpoints (ask, auth, monitoring, eval, feedback)
- `app/utils/` - Utilities (evaluation, feedback system, performance monitoring)

**Frontend (React):**
- `frontend-react/src/App.js` - Main React application
- `frontend-react/src/components/` - Reusable UI components
- `frontend-react/src/pages/` - Page components (Chat, Upload, Documents, etc.)
- `frontend-react/src/context/` - React contexts for state management

**Data Layer:**
- `data/documents/` - Uploaded PDF documents with UUID prefixes
- `data/vector_store/` - FAISS vector store (index.faiss, index.pkl)
- `data/document_index.json` - Document metadata and mapping
- `data/` - Various JSON files for metrics, feedback, and system health

### Key Architectural Patterns

**Two-Stage Retrieval:**
1. Initial retrieval of 30 candidates using vector similarity (configurable via `RETRIEVAL_CANDIDATES`)
2. Cross-encoder reranking to select top 8 results (configurable via `RETRIEVAL_K`)

**Intelligent Source Filtering:**
- Automatic intent detection for personal vs. financial queries
- Dynamic document filtering based on query analysis
- Fallback expansion when filtering returns insufficient results

**Authentication System:**
- Session-based authentication with middleware
- Admin password protection (configurable via `ADMIN_PASSWORD` env var)
- CORS configuration for frontend-backend communication

**Feedback Loop System:**
- Real-time user feedback collection (thumbs up/down)
- Automatic parameter optimization based on feedback patterns
- Comprehensive analytics dashboard for monitoring system performance

## Configuration Settings

Key settings in `app/config.py`:
- `LLM_MODEL_NAME = "llama3:8b"` - Ollama model
- `EMBEDDING_MODEL` - BAAI/bge-large-en-v1.5 with fallback
- `CROSS_ENCODER_MODEL` - cross-encoder/ms-marco-MiniLM-L-6-v2
- `CHUNK_SIZE = 800` - Document chunk size with 300 token overlap
- `RETRIEVAL_K = 8` - Final results after reranking
- `RETRIEVAL_CANDIDATES = 30` - Initial candidates before reranking

## Development Guidelines

### Working with the RAG System

**Document Processing:**
- Documents are processed through `app/utils/file_loader.py`
- Each document gets a UUID prefix for safe storage
- Vector store is incrementally updated, not rebuilt from scratch
- Document index tracks metadata in JSON format

**Query Processing Flow:**
1. Query analysis and intent detection (`app/utils/query_analyzer.py`)
2. Source filtering based on detected intent
3. Hybrid retrieval combining vector similarity and keyword matching
4. Cross-encoder reranking for relevance
5. LLM response generation with source attribution

**Testing the System:**
- Upload test documents (CV/resume and financial reports work well)
- Test automatic filtering: "Tell me about Faiq's experience" vs "What was the revenue?"
- Verify source attribution and filtering behavior
- Check feedback system integration

### Security Considerations

**File Upload Security:**
- File type validation (PDF only)
- File size limits (10MB max)
- Filename sanitization to prevent path traversal
- Quarantine directory for suspicious files

**Authentication:**
- Session-based auth with secure headers
- CORS properly configured for specific origins
- Environment variables for sensitive config

### Performance Monitoring

The system includes comprehensive monitoring:
- Real-time performance metrics via `app/utils/performance_monitor.py`
- System health monitoring with alerts
- User feedback analytics and parameter optimization
- Query metrics and response time tracking

### Common Issues

**Ollama Connection Issues:**
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Pull required model if missing: `ollama pull llama3:8b`

**Vector Store Issues:**
- Vector store is auto-created on first document upload
- Deletion rebuilds entire vector store from remaining documents
- FAISS files are stored in `data/vector_store/`

**Frontend Proxy Issues:**
- Frontend runs on port 5170, backend on 8080/8001
- Proxy configuration in `package.json` handles API routing
- CORS middleware allows credentials for authentication