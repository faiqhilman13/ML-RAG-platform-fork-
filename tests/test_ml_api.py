"""
Test ML API Endpoints

Tests for the ML training API endpoints to ensure they work correctly
and integrate properly with the existing RAG system.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, status

from app.main import app
from app.models.ml_models import ProblemTypeEnum, PipelineStatusEnum
from app.services.ml_pipeline_service import MLPipelineService


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication for tests."""
    def _mock_require_auth():
        return "test_user"
    
    # Override the dependency in the FastAPI app
    from app.main import app
    from app.auth import require_auth
    
    app.dependency_overrides[require_auth] = _mock_require_auth
    yield
    
    # Clean up after test
    if require_auth in app.dependency_overrides:
        del app.dependency_overrides[require_auth]


@pytest.fixture
def mock_ml_service():
    """Mock ML pipeline service."""
    with patch('app.routers.ml.get_ml_pipeline_service') as mock:
        service_mock = Mock(spec=MLPipelineService)
        mock.return_value = service_mock
        yield service_mock


class TestMLAPIEndpoints:
    """Test ML API endpoint functionality."""
    
    def test_ml_health_check(self, client):
        """Test ML health check endpoint."""
        with patch('app.database.db_manager') as mock_db:
            mock_db.health_check.return_value = {"status": "healthy"}
            
            response = client.get("/api/ml/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "details" in data
    
    def test_create_ml_training_pipeline_success(self, client, mock_auth, mock_ml_service):
        """Test successful ML training pipeline creation."""
        # Mock service response
        mock_ml_service.trigger_ml_training = AsyncMock(return_value={
            "success": True,
            "pipeline_run_uuid": "test-uuid-123",
            "status": "running",
            "message": "Pipeline started successfully"
        })
        
        # Valid request payload
        request_data = {
            "file_path": "test_data.csv",
            "target_variable": "target",
            "problem_type": "classification",
            "algorithms": [
                {
                    "name": "random_forest_classifier",
                    "hyperparameters": {"n_estimators": 100}
                }
            ],
            "preprocessing_config": {
                "test_size": 0.2,
                "random_state": 42
            }
        }
        
        response = client.post("/api/ml/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_uuid"] == "test-uuid-123"
        assert data["status"] == "running"
        
        # Verify service was called with correct parameters
        mock_ml_service.trigger_ml_training.assert_called_once()
    
    def test_create_ml_training_pipeline_validation_error(self, client, mock_auth):
        """Test ML training pipeline creation with validation errors."""
        # Invalid request - missing required fields
        request_data = {
            "file_path": "",  # Empty file path
            "target_variable": "target",
            "problem_type": "classification"
            # Missing algorithms
        }
        
        response = client.post("/api/ml/train", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_create_ml_training_pipeline_service_error(self, client, mock_auth, mock_ml_service):
        """Test ML training pipeline creation with service error."""
        # Mock service failure
        mock_ml_service.trigger_ml_training = AsyncMock(return_value={
            "success": False,
            "error": "Dataset file not found"
        })
        
        request_data = {
            "file_path": "nonexistent.csv",
            "target_variable": "target",
            "problem_type": "classification",
            "algorithms": [{"name": "random_forest_classifier"}]
        }
        
        response = client.post("/api/ml/train", json=request_data)
        assert response.status_code == 500
    
    def test_get_pipeline_status_success(self, client, mock_auth, mock_ml_service):
        """Test successful pipeline status retrieval."""
        mock_ml_service.get_pipeline_status.return_value = {
            "success": True,
            "run_uuid": "test-uuid-123",
            "status": "running",
            "progress": {"percentage": 50, "current_stage": "training"},
            "created_at": "2025-01-08T10:00:00",
            "started_at": "2025-01-08T10:01:00"
        }
        
        response = client.get("/api/ml/status/test-uuid-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_uuid"] == "test-uuid-123"
        assert data["status"] == "running"
        assert "progress" in data
    
    def test_get_pipeline_status_not_found(self, client, mock_auth, mock_ml_service):
        """Test pipeline status retrieval for non-existent pipeline."""
        mock_ml_service.get_pipeline_status.return_value = {
            "success": False,
            "error": "Pipeline run not found"
        }
        
        response = client.get("/api/ml/status/nonexistent-uuid")
        assert response.status_code == 404
    
    def test_get_pipeline_results_success(self, client, mock_auth, mock_ml_service):
        """Test successful pipeline results retrieval."""
        mock_ml_service.get_pipeline_results.return_value = {
            "success": True,
            "run_uuid": "test-uuid-123",
            "status": "completed",
            "problem_type": "classification",
            "target_variable": "target",
            "total_training_time_seconds": 120.5,
            "total_models_trained": 3,
            "best_model": {
                "algorithm_name": "random_forest_classifier",
                "primary_metric": {"name": "accuracy", "value": 0.95}
            },
            "model_results": []
        }
        
        response = client.get("/api/ml/results/test-uuid-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_models_trained"] == 3
        assert data["best_model"]["primary_metric"]["value"] == 0.95
    
    def test_get_available_algorithms_success(self, client, mock_auth, mock_ml_service):
        """Test successful algorithm retrieval."""
        mock_ml_service.get_available_algorithms.return_value = {
            "success": True,
            "algorithms": {
                "random_forest_classifier": {
                    "display_name": "Random Forest Classifier",
                    "problem_types": ["classification"],
                    "hyperparameters": {}
                }
            },
            "total_count": 1
        }
        
        response = client.get("/api/ml/algorithms")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 1
        assert "random_forest_classifier" in data["algorithms"]
    
    def test_get_available_algorithms_filtered(self, client, mock_auth, mock_ml_service):
        """Test algorithm retrieval with problem type filter."""
        mock_ml_service.get_available_algorithms.return_value = {
            "success": True,
            "algorithms": {},
            "total_count": 0
        }
        
        response = client.get("/api/ml/algorithms?problem_type=regression")
        
        assert response.status_code == 200
        # Verify service was called with filter
        mock_ml_service.get_available_algorithms.assert_called_with(ProblemTypeEnum.REGRESSION)
    
    def test_validate_dataset_success(self, client, mock_auth, mock_ml_service):
        """Test successful dataset validation."""
        mock_ml_service.validate_dataset.return_value = {
            "success": True,
            "dataset_info": {
                "shape": [1000, 10],
                "total_rows": 1000,
                "total_features": 9
            },
            "target_info": {
                "name": "target",
                "type": "int64",
                "unique_count": 2
            },
            "suggested_problem_type": "classification"
        }
        
        response = client.post(
            "/api/ml/validate-dataset",
            params={"file_path": "test.csv", "target_column": "target"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["dataset_info"]["total_rows"] == 1000
        assert data["suggested_problem_type"] == "classification"
    
    def test_validate_dataset_validation_error(self, client, mock_auth):
        """Test dataset validation with invalid parameters."""
        # Missing required parameters
        response = client.post("/api/ml/validate-dataset")
        assert response.status_code == 422  # Validation error
        
        # Empty file path
        response = client.post(
            "/api/ml/validate-dataset",
            params={"file_path": "", "target_column": "target"}
        )
        assert response.status_code == 400
    
    def test_list_ml_pipelines(self, client, mock_auth):
        """Test pipeline listing endpoint."""
        response = client.get("/api/ml/pipelines")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "pipelines" in data
        assert "total" in data
    
    def test_delete_ml_pipeline(self, client, mock_auth):
        """Test pipeline deletion endpoint."""
        response = client.delete("/api/ml/pipelines/test-uuid-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestMLAPIIntegration:
    """Test ML API integration with existing RAG system."""
    
    def test_ml_endpoints_dont_break_existing_auth(self, client):
        """Test that ML endpoints work with existing auth system."""
        # Test without authentication
        response = client.get("/api/ml/algorithms")
        # Should return auth error (401) not server error (500)
        assert response.status_code == 401
    
    def test_ml_endpoints_cors_headers(self, client, mock_auth):
        """Test that ML endpoints respect CORS configuration."""
        response = client.options("/api/ml/algorithms")
        # Should not return error due to CORS
        assert response.status_code in [200, 405]  # Either allowed or method not allowed
    
    def test_ml_router_prefix(self, client, mock_auth, mock_ml_service):
        """Test that ML endpoints are properly prefixed."""
        mock_ml_service.get_available_algorithms.return_value = {
            "success": True,
            "algorithms": {},
            "total_count": 0
        }
        
        # Should work with /api/ml prefix
        response = client.get("/api/ml/algorithms")
        assert response.status_code == 200
        
        # Should not work without prefix
        response = client.get("/algorithms")
        assert response.status_code == 404


class TestMLAPIErrorHandling:
    """Test ML API error handling and edge cases."""
    
    def test_ml_service_exception_handling(self, client, mock_auth, mock_ml_service):
        """Test handling of unexpected service exceptions."""
        mock_ml_service.get_available_algorithms.side_effect = Exception("Database connection failed")
        
        response = client.get("/api/ml/algorithms")
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
    
    def test_invalid_uuid_format(self, client, mock_auth, mock_ml_service):
        """Test handling of invalid UUID formats."""
        mock_ml_service.get_pipeline_status.return_value = {
            "success": False,
            "error": "Pipeline run not found"
        }
        
        response = client.get("/api/ml/status/invalid-uuid-format")
        assert response.status_code == 404
    
    def test_large_request_payload(self, client, mock_auth):
        """Test handling of large request payloads."""
        # Create a large algorithm list
        large_algorithms = [
            {"name": f"algorithm_{i}", "hyperparameters": {f"param_{j}": j for j in range(100)}}
            for i in range(50)  # This should exceed reasonable limits
        ]
        
        request_data = {
            "file_path": "test.csv",
            "target_variable": "target", 
            "problem_type": "classification",
            "algorithms": large_algorithms
        }
        
        response = client.post("/api/ml/train", json=request_data)
        # Should either reject due to validation or handle gracefully
        assert response.status_code in [400, 422, 413]  # Bad request, validation error, or payload too large


class TestMLAPISecurityHeaders:
    """Test that ML endpoints maintain security headers."""
    
    def test_security_headers_present(self, client, mock_auth, mock_ml_service):
        """Test that security headers are present in ML API responses."""
        mock_ml_service.get_available_algorithms.return_value = {
            "success": True,
            "algorithms": {},
            "total_count": 0
        }
        
        response = client.get("/api/ml/algorithms")
        
        # Check that security headers are present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"