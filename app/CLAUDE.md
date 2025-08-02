# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Retrieval-Augmented Generation (RAG) chatbot built with FastAPI and React that provides intelligent document-based question answering with advanced ML features, source filtering capabilities, and comprehensive monitoring systems.

## Development Commands

### Backend Development
```bash
# Start the FastAPI server (recommended approach)
python -m uvicorn app.main:app --reload --port 8000

# Alternative: Direct execution (uses port 8001)
python app/main.py

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_auth.py

# Run tests by category
pytest -m "ml_api"         # ML API tests
pytest -m "ml_models"      # ML model tests  
pytest -m "integration"    # Integration tests
pytest -m "fast"           # Fast running tests
```

### Frontend Development
```bash
cd frontend-react
npm install
npm start                  # Runs on port 5170
npm run build
npm test
```

### Required External Services
```bash
# Ollama LLM service (required)
ollama serve
ollama pull llama3:8b

# Optional: Test with other models
ollama pull mistral
```

## Core Architecture

### Application Structure
- **main.py** - FastAPI app initialization, middleware, and core endpoints
- **config.py** - Central configuration with environment variable support
- **auth.py** - Session-based authentication middleware
- **database.py** - Database abstraction layer
- **routers/** - API endpoint modules (ask, auth, eval, feedback, ml, monitoring)
- **services/** - Business logic layer (ml_pipeline_service)
- **utils/** - Shared utilities and helper functions
- **models/** - Data models and ML model definitions
- **retrievers/** - RAG retrieval engine and query processing
- **llm/** - Language model integration (Ollama)

### Key Architectural Patterns

**Two-Stage RAG Retrieval:**
1. Initial semantic search retrieves `RETRIEVAL_CANDIDATES` (30) documents using BAAI/bge-large-en-v1.5 embeddings
2. Cross-encoder reranking selects top `RETRIEVAL_K` (8) most relevant results
3. Intelligent source filtering based on query intent detection

**ML Pipeline Integration:**
- Seamless integration with existing authentication and routing patterns
- Support for multiple problem types (classification, regression, clustering)
- Modular algorithm registry with configurable hyperparameters
- Training pipeline with evaluation metrics and model persistence

**Session-Based Authentication:**
- Middleware-based auth with session management
- Admin password protection via environment variables
- CORS configuration for React frontend integration

**Document Processing:**
- UUID-prefixed file storage for security
- Incremental vector store updates (no full rebuilds)
- JSON-based document index for metadata tracking
- Automatic chunk processing with sliding window (800 chars, 300 overlap)

**Real-Time Monitoring:**
- Performance metrics tracking via `performance_monitor.py`
- User feedback loop with automatic parameter optimization
- System health monitoring with configurable alerts
- Query analytics and response time tracking

## Configuration Settings

### Key Environment Variables
```bash
SESSION_SECRET_KEY=<your-secret-key>
ADMIN_PASSWORD=<admin-password>
SENTRY_DSN=<sentry-dsn>              # Optional monitoring
```

### Model Configuration (config.py)
- **LLM**: `llama3:8b` via Ollama (localhost:11434)
- **Embeddings**: `BAAI/bge-large-en-v1.5` with fallback to `all-MiniLM-L6-v2`
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Chunking**: 800 chars with 300 char overlap
- **Retrieval**: 30 candidates → 8 final results

## Development Guidelines

### Working with the RAG System

**Document Upload Flow:**
1. Security validation (file type, size, filename sanitization)
2. UUID generation and safe filename creation
3. Document processing via `prepare_documents()`
4. Incremental vector store update using FAISS `add_documents()`
5. JSON index update for metadata tracking

**Query Processing:**
1. Query analysis and intent detection (`query_analyzer.py`)
2. Dynamic source filtering based on detected intent
3. Hybrid retrieval combining semantic and keyword matching
4. Cross-encoder reranking for relevance optimization
5. LLM response generation with source attribution

**Testing RAG Features:**
- Upload diverse document types (CVs, financial reports, technical docs)
- Test automatic filtering: personal queries vs. financial queries
- Verify source attribution accuracy
- Check fallback behavior when filtering returns insufficient results

### Working with ML Features

**ML Pipeline Usage:**
- Dataset upload via `/api/ml/upload-dataset` 
- Training configuration via algorithm registry
- Pipeline execution with progress tracking
- Model evaluation and persistence
- Results visualization and analysis

**Adding New Algorithms:**
1. Register in `workflows/ml/algorithm_registry.py`
2. Implement in appropriate module (preprocessing, evaluation)
3. Add tests in `tests/test_ml_*.py`
4. Update API documentation

### Security Considerations

**File Upload Security:**
- Strict file type validation (PDF only for documents)
- 10MB size limit enforcement
- Filename sanitization to prevent path traversal
- Quarantine directory for suspicious files

**Authentication Security:**
- Session-based auth with secure headers
- CORS restricted to specific origins
- Environment variable protection for secrets
- Security headers middleware (`X-Content-Type-Options`, `X-Frame-Options`)

### Performance Monitoring

**Built-in Monitoring Systems:**
- Real-time performance metrics collection
- System health dashboard with alerts
- User feedback analytics and optimization
- Query performance tracking and analysis

**Key Metrics Tracked:**
- Response times and error rates
- Vector store performance
- User satisfaction scores
- Resource utilization patterns

## Common Development Tasks

### Adding New API Endpoints
1. Create router module in `routers/`
2. Add authentication dependency: `Depends(require_auth)`
3. Include router in `main.py` with appropriate prefix
4. Add corresponding tests in `tests/`

### Extending RAG Capabilities
1. Modify retrieval logic in `retrievers/rag.py`
2. Update query analysis in `utils/query_analyzer.py`
3. Add evaluation metrics in `utils/evaluation.py`
4. Test with diverse document types

### Troubleshooting Common Issues

**Ollama Connection Issues:**
- Verify service: `ollama serve`
- Check model availability: `ollama list`
- Test API: `curl http://localhost:11434/api/generate`

**Vector Store Corruption:**
- Delete `data/vector_store/` directory
- Re-upload documents to rebuild FAISS index
- Check document index JSON integrity

**Authentication Problems:**
- Verify `SESSION_SECRET_KEY` is set
- Check CORS configuration for frontend origin
- Ensure session middleware is properly configured

**ML Pipeline Issues:**
- Verify dataset format and encoding
- Check algorithm configuration parameters
- Review training logs for error details
- Ensure sufficient disk space for model storage