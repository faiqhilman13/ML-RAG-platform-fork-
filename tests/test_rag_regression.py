"""
Test RAG System Regression

Tests to ensure that the new ML features do not break existing RAG functionality.
This includes testing existing API endpoints, authentication, document management,
and chat functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, mock_open
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth_session():
    """Mock authenticated session."""
    def _mock_require_auth(request):
        return "test_user"
    
    with patch('app.auth.require_auth', side_effect=_mock_require_auth):
        yield


class TestExistingAPIEndpoints:
    """Test that existing API endpoints still work after ML integration."""
    
    def test_health_endpoint_still_works(self, client):
        """Test that health endpoint is not affected by ML changes."""
        with patch('app.main.VECTORSTORE_DIR') as mock_vectorstore_dir:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.iterdir.return_value = ['some_file']
            mock_vectorstore_dir.exists.return_value = True
            mock_vectorstore_dir.iterdir.return_value = ['some_file']
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "ok"
            assert "vector_store_initialized" in data
    
    def test_auth_endpoints_still_work(self, client):
        """Test that authentication endpoints are not affected."""
        # Test login endpoint
        login_data = {"username": "admin", "password": "admin123"}
        
        with patch('app.routers.auth.authenticate_user') as mock_auth, \
             patch('app.routers.auth.create_session') as mock_session:
            
            mock_auth.return_value = {"username": "admin", "is_active": True}
            
            response = client.post("/login", data=login_data)
            
            # Should redirect or return success
            assert response.status_code in [200, 307, 302]
    
    def test_document_upload_still_works(self, client, mock_auth_session):
        """Test that document upload functionality is not affected."""
        with patch('app.main.prepare_documents') as mock_prepare, \
             patch('app.main.rag_retriever') as mock_retriever, \
             patch('app.main.load_document_index') as mock_load_index, \
             patch('app.main.save_document_index') as mock_save_index, \
             patch('builtins.open', mock_open()) as mock_file:
            
            # Mock document processing
            mock_prepare.return_value = [Mock()]  # Mock chunks
            mock_retriever.load_vectorstore.return_value = True
            mock_retriever.save_vectorstore.return_value = True
            mock_load_index.return_value = {}
            
            # Test file upload
            test_file_content = b"Test document content"
            
            response = client.post(
                "/api/upload",
                files={"file": ("test.txt", test_file_content, "text/plain")},
                data={"title": "Test Document"}
            )
            
            # Should work regardless of ML features
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_document_list_still_works(self, client, mock_auth_session):
        """Test that document listing is not affected."""
        with patch('app.main.load_document_index') as mock_load_index:
            mock_load_index.return_value = {
                "doc1": {"title": "Document 1", "filename": "doc1.txt", "id": "doc1"},
                "doc2": {"title": "Document 2", "filename": "doc2.txt", "id": "doc2"}
            }
            
            response = client.get("/api/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
            assert len(data["documents"]) == 2
    
    def test_document_delete_still_works(self, client, mock_auth_session):
        """Test that document deletion is not affected."""
        with patch('app.main.load_document_index') as mock_load_index, \
             patch('app.main.save_document_index') as mock_save_index, \
             patch('app.main.rag_retriever') as mock_retriever, \
             patch('app.main.prepare_documents') as mock_prepare, \
             patch('os.remove') as mock_remove, \
             patch('os.path.exists') as mock_exists:
            
            # Mock document index
            mock_load_index.return_value = {
                "test-doc-id": {"title": "Test Doc", "filename": "test.txt", "id": "test-doc-id"}
            }
            mock_exists.return_value = True
            mock_prepare.return_value = []
            mock_retriever.build_vectorstore.return_value = True
            
            response = client.delete("/api/documents/test-doc-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_ask_endpoint_still_works(self, client, mock_auth_session):
        """Test that the chat/ask functionality is not affected."""
        with patch('app.routers.ask.rag_retriever') as mock_retriever, \
             patch('app.routers.ask.save_chat_context') as mock_save_context:
            
            # Mock RAG retriever response
            mock_retriever.retrieve_and_generate.return_value = {
                "answer": "This is a test answer",
                "sources": ["document1.txt", "document2.txt"],
                "context_used": ["relevant context chunk"],
                "confidence_score": 0.85
            }
            
            response = client.post(
                "/api/ask",
                json={"question": "What is the test about?"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert data["answer"] == "This is a test answer"
    
    def test_monitoring_endpoints_still_work(self, client, mock_auth_session):
        """Test that monitoring endpoints are not affected."""
        with patch('app.routers.monitoring.get_system_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "memory_usage": 50.0,
                "cpu_usage": 25.0
            }
            
            response = client.get("/api/monitoring/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
    
    def test_evaluation_endpoints_still_work(self, client, mock_auth_session):
        """Test that evaluation endpoints are not affected."""
        with patch('app.routers.eval.answer_evaluator') as mock_evaluator:
            mock_evaluator.evaluate_answer.return_value = {"score": 0.85}
            
            response = client.post(
                "/api/eval/evaluate",
                json={
                    "question": "Test question",
                    "answer": "Test answer",
                    "context": "Test context"
                }
            )
            
            assert response.status_code == 200
    
    def test_feedback_endpoints_still_work(self, client, mock_auth_session):
        """Test that feedback endpoints are not affected."""
        response = client.post(
            "/api/feedback/submit",
            json={
                "question": "Test question",
                "answer": "Test answer", 
                "rating": 5,
                "feedback": "Great answer"
            }
        )
        
        # Should work regardless of ML features
        assert response.status_code == 200


class TestApplicationStartup:
    """Test that application starts up correctly with ML features."""
    
    def test_app_initialization_includes_ml_router(self):
        """Test that ML router is properly included in app."""
        # Check that ML router is included
        routes = [route.path for route in app.routes]
        
        # ML endpoints should be available
        ml_routes = [route for route in routes if route.startswith("/api/ml")]
        assert len(ml_routes) > 0 or any("/api/ml" in str(route) for route in app.routes)
    
    def test_middleware_still_works_with_ml(self, client):
        """Test that middleware (CORS, sessions, security) still works."""
        # Test CORS headers
        response = client.options("/api/documents", headers={"Origin": "http://localhost:5170"})
        
        # Should not fail due to CORS issues
        assert response.status_code in [200, 405]
        
        # Test security headers on any endpoint
        response = client.get("/health")
        
        # Security headers should still be present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
    
    def test_static_file_serving_still_works(self, client):
        """Test that static file serving is not affected."""
        with patch('pathlib.Path.is_dir') as mock_is_dir, \
             patch('pathlib.Path.is_file') as mock_is_file:
            
            mock_is_dir.return_value = True
            mock_is_file.return_value = True
            
            # Try to access root endpoint (should serve frontend)
            response = client.get("/")
            
            # Should attempt to serve frontend (may fail due to missing file, but routing should work)
            assert response.status_code in [200, 404]


class TestDatabaseIntegration:
    """Test that database integration doesn't break existing functionality."""
    
    def test_existing_database_operations_still_work(self):
        """Test that existing database operations are not affected."""
        # Test that we can import and use existing database functionality
        from app.config import BASE_DIR
        
        # Should be able to access existing configuration
        assert BASE_DIR is not None
        
        # Test document index operations
        from app.main import load_document_index, save_document_index
        
        with patch('builtins.open', mock_open(read_data='{}')) as mock_file:
            index = load_document_index()
            assert isinstance(index, dict)
            
            # Should be able to save index
            save_document_index({"test": "data"})
            mock_file.assert_called()
    
    def test_ml_database_doesnt_interfere_with_existing_data(self):
        """Test that ML database setup doesn't interfere with existing data."""
        # Test that we can import ML database components without errors
        from app.database import db_manager
        from app.models.ml_models import MLPipelineRun, MLModel
        
        # Should be able to import without affecting existing functionality
        assert MLPipelineRun is not None
        assert MLModel is not None


class TestErrorHandling:
    """Test that error handling is not affected by ML integration."""
    
    def test_existing_error_handling_still_works(self, client):
        """Test that existing error handling is not broken."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # Test auth required error
        response = client.get("/api/documents")
        assert response.status_code == 401
        
        # Test invalid request format
        response = client.post("/api/ask", json={"invalid": "request"})
        assert response.status_code in [400, 422]
    
    def test_ml_errors_dont_affect_existing_endpoints(self, client, mock_auth_session):
        """Test that ML service errors don't break existing endpoints."""
        # Even if ML service is broken, existing endpoints should work
        with patch('app.services.ml_pipeline_service.get_ml_pipeline_service') as mock_ml_service:
            mock_ml_service.side_effect = Exception("ML service is down")
            
            # Existing endpoints should still work
            with patch('app.main.load_document_index') as mock_load_index:
                mock_load_index.return_value = {}
                
                response = client.get("/api/documents")
                assert response.status_code == 200


class TestPerformanceRegression:
    """Test that ML integration doesn't cause performance regression."""
    
    def test_response_times_not_significantly_affected(self, client, mock_auth_session):
        """Test that response times are not significantly affected."""
        import time
        
        with patch('app.main.load_document_index') as mock_load_index:
            mock_load_index.return_value = {}
            
            # Measure response time for existing endpoint
            start_time = time.time()
            response = client.get("/api/documents")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Response should be reasonably fast (less than 1 second for mocked operation)
            assert response_time < 1.0
            assert response.status_code == 200
    
    def test_memory_usage_not_significantly_increased(self):
        """Test that importing ML modules doesn't significantly increase memory usage."""
        import psutil
        import os
        
        # Get memory usage before importing ML modules
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Import ML modules
        from app.services.ml_pipeline_service import MLPipelineService
        from workflows.ml.algorithm_registry import get_algorithm_registry
        
        # Get memory usage after
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for imports)
        assert memory_increase < 100 * 1024 * 1024  # 100MB


class TestConfigurationCompatibility:
    """Test that configuration and environment setup is compatible."""
    
    def test_existing_config_still_accessible(self):
        """Test that existing configuration is still accessible."""
        from app.config import (
            DOCUMENTS_DIR, VECTORSTORE_DIR, BASE_DIR, 
            SESSION_SECRET_KEY, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
        )
        
        # All existing config should still be accessible
        assert DOCUMENTS_DIR is not None
        assert VECTORSTORE_DIR is not None
        assert BASE_DIR is not None
        assert SESSION_SECRET_KEY is not None
        assert MAX_FILE_SIZE is not None
        assert ALLOWED_EXTENSIONS is not None
    
    def test_new_ml_dependencies_dont_break_existing_imports(self):
        """Test that new ML dependencies don't break existing imports."""
        # Test that we can still import all existing modules
        from app.auth import require_auth, authenticate_user
        from app.retrievers.rag import rag_retriever
        from app.utils.file_loader import prepare_documents
        from app.routers import ask, auth, monitoring, eval, feedback
        
        # All imports should work without errors
        assert require_auth is not None
        assert authenticate_user is not None
        assert rag_retriever is not None
        assert prepare_documents is not None
    
    def test_requirements_compatibility(self):
        """Test that new requirements don't conflict with existing ones."""
        # This test ensures that we can import both existing and new dependencies
        try:
            # Existing dependencies
            import fastapi
            import uvicorn
            import langchain
            import sentence_transformers
            import faiss
            
            # New ML dependencies
            import pandas
            import sklearn
            import matplotlib
            import seaborn
            import joblib
            
            # All imports should succeed
            assert True
            
        except ImportError as e:
            pytest.fail(f"Import conflict detected: {e}")


class TestBackwardCompatibility:
    """Test backward compatibility of API responses."""
    
    def test_existing_api_response_formats_unchanged(self, client, mock_auth_session):
        """Test that existing API response formats are unchanged."""
        with patch('app.main.load_document_index') as mock_load_index:
            mock_load_index.return_value = {
                "doc1": {"title": "Test Doc", "filename": "test.txt", "id": "doc1"}
            }
            
            response = client.get("/api/documents")
            data = response.json()
            
            # Response format should be unchanged
            assert "documents" in data
            assert isinstance(data["documents"], list)
            
            if len(data["documents"]) > 0:
                doc = data["documents"][0]
                assert "title" in doc
                assert "filename" in doc
                assert "id" in doc
    
    def test_health_check_response_format_unchanged(self, client):
        """Test that health check response format is unchanged."""
        with patch('app.main.VECTORSTORE_DIR') as mock_vectorstore_dir:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.iterdir.return_value = ['some_file']
            mock_vectorstore_dir.exists.return_value = True
            mock_vectorstore_dir.iterdir.return_value = ['some_file']
            
            response = client.get("/health")
            data = response.json()
            
            # Original fields should still be present
            assert "status" in data
            assert "vector_store_initialized" in data
            
            # Values should have expected types
            assert isinstance(data["status"], str)
            assert isinstance(data["vector_store_initialized"], bool)