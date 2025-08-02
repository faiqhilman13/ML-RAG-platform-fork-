"""
Test ML Pipeline Service

Tests for the ML pipeline service layer functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock, mock_open
from pathlib import Path
from datetime import datetime

from app.services.ml_pipeline_service import MLPipelineService, get_ml_pipeline_service
from app.models.ml_models import ProblemTypeEnum, PipelineStatusEnum, AlgorithmNameEnum


@pytest.fixture
def ml_service():
    """Create ML service instance for testing."""
    return MLPipelineService()


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'feature3': [10.1, 20.2, 30.3, 40.4, 50.5],
        'target': [0, 1, 0, 1, 0]
    })


class TestMLPipelineServiceInit:
    """Test ML pipeline service initialization."""
    
    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = MLPipelineService()
        
        assert service.executor is not None
        assert service.active_pipelines == {}
        assert service.executor._max_workers == 2
    
    def test_singleton_service_instance(self):
        """Test that get_ml_pipeline_service returns singleton."""
        service1 = get_ml_pipeline_service()
        service2 = get_ml_pipeline_service()
        
        assert service1 is service2


class TestTriggerMLTraining:
    """Test ML training pipeline triggering."""
    
    @pytest.mark.asyncio
    async def test_trigger_ml_training_success(self, ml_service):
        """Test successful ML training trigger."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db, \
             patch('app.services.ml_pipeline_service.create_pipeline_run') as mock_create, \
             patch.object(ml_service.executor, 'run_in_executor') as mock_executor:
            
            # Mock database operations
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_pipeline_run = Mock()
            mock_pipeline_run.id = 1
            mock_create.return_value = mock_pipeline_run
            
            mock_executor.return_value = AsyncMock()
            
            # Test parameters
            result = await ml_service.trigger_ml_training(
                file_path="test.csv",
                target_variable="target",
                problem_type=ProblemTypeEnum.CLASSIFICATION,
                algorithms=[{"name": "random_forest_classifier"}],
                preprocessing_config={"test_size": 0.2},
                experiment_id="exp-1",
                user_id="test_user"
            )
            
            assert result["success"] is True
            assert "pipeline_run_uuid" in result
            assert result["status"] == PipelineStatusEnum.RUNNING.value
            
            # Verify database operations
            mock_create.assert_called_once()
            assert mock_session.add.called
            assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_trigger_ml_training_database_error(self, ml_service):
        """Test ML training trigger with database error."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            result = await ml_service.trigger_ml_training(
                file_path="test.csv",
                target_variable="target",
                problem_type=ProblemTypeEnum.CLASSIFICATION,
                algorithms=[{"name": "random_forest_classifier"}]
            )
            
            assert result["success"] is False
            assert "Database connection failed" in result["error"]


class TestPipelineStatus:
    """Test pipeline status functionality."""
    
    def test_get_pipeline_status_success(self, ml_service):
        """Test successful pipeline status retrieval."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock pipeline run
            mock_pipeline_run = Mock()
            mock_pipeline_run.run_uuid = "test-uuid-123"
            mock_pipeline_run.status = PipelineStatusEnum.RUNNING.value
            mock_pipeline_run.created_at = datetime(2025, 1, 8, 10, 0, 0)
            mock_pipeline_run.started_at = datetime(2025, 1, 8, 10, 1, 0)
            mock_pipeline_run.completed_at = None
            mock_pipeline_run.error_message = None
            mock_pipeline_run.total_models_trained = None
            mock_pipeline_run.best_model_score = None
            
            mock_session.exec.return_value.first.return_value = mock_pipeline_run
            
            result = ml_service.get_pipeline_status("test-uuid-123")
            
            assert result["success"] is True
            assert result["run_uuid"] == "test-uuid-123"
            assert result["status"] == PipelineStatusEnum.RUNNING.value
            assert "progress" in result
            assert "created_at" in result
    
    def test_get_pipeline_status_not_found(self, ml_service):
        """Test pipeline status retrieval for non-existent pipeline."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.exec.return_value.first.return_value = None
            
            result = ml_service.get_pipeline_status("nonexistent-uuid")
            
            assert result["success"] is False
            assert "Pipeline run not found" in result["error"]
    
    def test_get_pipeline_status_progress_calculation(self, ml_service):
        """Test progress calculation in pipeline status."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock completed pipeline
            mock_pipeline_run = Mock()
            mock_pipeline_run.run_uuid = "test-uuid-123"
            mock_pipeline_run.status = PipelineStatusEnum.COMPLETED.value
            mock_pipeline_run.created_at = datetime(2025, 1, 8, 10, 0, 0)
            mock_pipeline_run.started_at = datetime(2025, 1, 8, 10, 1, 0)
            mock_pipeline_run.completed_at = datetime(2025, 1, 8, 10, 5, 0)
            mock_pipeline_run.error_message = None
            mock_pipeline_run.total_models_trained = 3
            mock_pipeline_run.best_model_score = 0.95
            
            mock_session.exec.return_value.first.return_value = mock_pipeline_run
            
            result = ml_service.get_pipeline_status("test-uuid-123")
            
            assert result["progress"]["percentage"] == 100
            assert result["progress"]["current_stage"] == "completed"


class TestPipelineResults:
    """Test pipeline results functionality."""
    
    def test_get_pipeline_results_success(self, ml_service):
        """Test successful pipeline results retrieval."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock pipeline run
            mock_pipeline_run = Mock()
            mock_pipeline_run.run_uuid = "test-uuid-123"
            mock_pipeline_run.status = PipelineStatusEnum.COMPLETED.value
            mock_pipeline_run.problem_type = ProblemTypeEnum.CLASSIFICATION.value
            mock_pipeline_run.target_variable = "target"
            mock_pipeline_run.created_at = datetime(2025, 1, 8, 10, 0, 0)
            mock_pipeline_run.completed_at = datetime(2025, 1, 8, 10, 5, 0)
            mock_pipeline_run.total_training_time_seconds = 300.0
            mock_pipeline_run.ml_config = {"test": "config"}
            mock_pipeline_run.id = 1
            
            # Mock models
            mock_model = Mock()
            mock_model.model_id = "model-123"
            mock_model.algorithm_name = "random_forest_classifier"
            mock_model.algorithm_display_name = "Random Forest"
            mock_model.primary_metric_name = "accuracy"
            mock_model.primary_metric_value = 0.95
            mock_model.performance_metrics = {"accuracy": 0.95, "f1_score": 0.93}
            mock_model.training_time_seconds = 150.0
            mock_model.hyperparameters = {"n_estimators": 100}
            mock_model.feature_importance = {}
            mock_model.n_features = 10
            mock_model.is_best_model = True
            mock_model.rank_in_pipeline = 1
            
            # Mock preprocessing log
            mock_preprocessing_log = Mock()
            mock_preprocessing_log.original_shape = [1000, 10]
            mock_preprocessing_log.final_shape = [800, 10]
            mock_preprocessing_log.steps_applied = ["remove_missing", "scale"]
            mock_preprocessing_log.warnings = []
            
            # Setup mock returns
            mock_session.exec.side_effect = [
                Mock(first=Mock(return_value=mock_pipeline_run)),  # Pipeline run query
                Mock(all=Mock(return_value=[mock_model])),          # Models query
                Mock(first=Mock(return_value=mock_preprocessing_log)) # Preprocessing log query
            ]
            
            result = ml_service.get_pipeline_results("test-uuid-123")
            
            assert result["success"] is True
            assert result["run_uuid"] == "test-uuid-123"
            assert result["total_models_trained"] == 1
            assert result["best_model"]["primary_metric"]["value"] == 0.95
            assert result["preprocessing_info"]["original_shape"] == [1000, 10]
    
    def test_get_pipeline_results_not_found(self, ml_service):
        """Test pipeline results retrieval for non-existent pipeline."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.exec.return_value.first.return_value = None
            
            result = ml_service.get_pipeline_results("nonexistent-uuid")
            
            assert result["success"] is False
            assert "Pipeline run not found" in result["error"]


class TestAvailableAlgorithms:
    """Test available algorithms functionality."""
    
    def test_get_available_algorithms_all(self, ml_service):
        """Test getting all available algorithms."""
        with patch('app.services.ml_pipeline_service.get_available_algorithms') as mock_get_algos:
            mock_get_algos.return_value = {
                "random_forest_classifier": {
                    "display_name": "Random Forest Classifier",
                    "problem_types": ["classification"]
                },
                "linear_regression": {
                    "display_name": "Linear Regression", 
                    "problem_types": ["regression"]
                }
            }
            
            result = ml_service.get_available_algorithms()
            
            assert result["success"] is True
            assert result["total_count"] == 2
            assert "random_forest_classifier" in result["algorithms"]
            assert "linear_regression" in result["algorithms"]
    
    def test_get_available_algorithms_filtered(self, ml_service):
        """Test getting algorithms filtered by problem type."""
        with patch('app.services.ml_pipeline_service.get_available_algorithms') as mock_get_algos:
            mock_get_algos.return_value = {
                "random_forest_classifier": {
                    "display_name": "Random Forest Classifier",
                    "problem_types": ["classification"]
                }
            }
            
            result = ml_service.get_available_algorithms(ProblemTypeEnum.CLASSIFICATION)
            
            assert result["success"] is True
            assert result["total_count"] == 1
            mock_get_algos.assert_called_with(ProblemTypeEnum.CLASSIFICATION)
    
    def test_get_available_algorithms_error(self, ml_service):
        """Test error handling in get_available_algorithms."""
        with patch('app.services.ml_pipeline_service.get_available_algorithms') as mock_get_algos:
            mock_get_algos.side_effect = Exception("Registry error")
            
            result = ml_service.get_available_algorithms()
            
            assert result["success"] is False
            assert "Registry error" in result["error"]


class TestDatasetValidation:
    """Test dataset validation functionality."""
    
    def test_validate_dataset_success(self, ml_service, sample_dataset):
        """Test successful dataset validation."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pandas.read_csv') as mock_read_csv:
            
            mock_exists.return_value = True
            mock_read_csv.return_value = sample_dataset
            
            result = ml_service.validate_dataset("test.csv", "target")
            
            assert result["success"] is True
            assert result["dataset_info"]["total_rows"] == 5
            assert result["dataset_info"]["total_features"] == 3
            assert result["target_info"]["name"] == "target"
            assert result["suggested_problem_type"] == "classification"
            assert len(result["feature_info"]) == 3
    
    def test_validate_dataset_file_not_found(self, ml_service):
        """Test dataset validation with missing file."""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = ml_service.validate_dataset("nonexistent.csv", "target")
            
            assert result["success"] is False
            assert "Dataset file not found" in result["error"]
    
    def test_validate_dataset_invalid_csv(self, ml_service):
        """Test dataset validation with invalid CSV."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pandas.read_csv') as mock_read_csv:
            
            mock_exists.return_value = True
            mock_read_csv.side_effect = Exception("Invalid CSV format")
            
            result = ml_service.validate_dataset("invalid.csv", "target")
            
            assert result["success"] is False
            assert "Could not load dataset" in result["error"]
    
    def test_validate_dataset_target_not_found(self, ml_service, sample_dataset):
        """Test dataset validation with missing target column."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pandas.read_csv') as mock_read_csv:
            
            mock_exists.return_value = True
            mock_read_csv.return_value = sample_dataset
            
            result = ml_service.validate_dataset("test.csv", "nonexistent_target")
            
            assert result["success"] is False
            assert "Target column 'nonexistent_target' not found" in result["error"]
            assert "available_columns" in result
    
    def test_validate_dataset_regression_detection(self, ml_service):
        """Test dataset validation with regression problem detection."""
        regression_dataset = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'price': [100.5, 200.7, 300.2, 400.8, 500.1]  # Continuous target
        })
        
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pandas.read_csv') as mock_read_csv:
            
            mock_exists.return_value = True
            mock_read_csv.return_value = regression_dataset
            
            result = ml_service.validate_dataset("regression_data.csv", "price")
            
            assert result["success"] is True
            assert result["suggested_problem_type"] == "regression"
    
    def test_validation_warnings(self, ml_service):
        """Test dataset validation warnings generation."""
        # Create dataset with various issues
        problematic_dataset = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5] * 10,  # Missing values
            'high_card_feature': [f"cat_{i}" for i in range(50)],  # High cardinality
            'mostly_missing': [1, None, None, None, None] * 10,  # >50% missing
            'target': [0, 1, 0, 1, 0] * 10
        })
        
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pandas.read_csv') as mock_read_csv:
            
            mock_exists.return_value = True
            mock_read_csv.return_value = problematic_dataset
            
            result = ml_service.validate_dataset("problematic.csv", "target")
            
            assert result["success"] is True
            assert len(result["validation_warnings"]) > 0
            assert any("High cardinality" in warning for warning in result["validation_warnings"])
            assert any(">50% missing values" in warning for warning in result["validation_warnings"])


class TestServiceErrorHandling:
    """Test service error handling."""
    
    def test_service_database_connection_error(self, ml_service):
        """Test handling of database connection errors."""
        with patch('app.services.ml_pipeline_service.get_db_session') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            result = ml_service.get_pipeline_status("test-uuid")
            
            assert result["success"] is False
            assert "Database connection failed" in result["error"]
    
    def test_service_unexpected_error(self, ml_service):
        """Test handling of unexpected errors."""
        with patch('app.services.ml_pipeline_service.get_available_algorithms') as mock_get_algos:
            mock_get_algos.side_effect = RuntimeError("Unexpected error")
            
            result = ml_service.get_available_algorithms()
            
            assert result["success"] is False
            assert "Unexpected error" in result["error"]


class TestPipelineExecution:
    """Test pipeline execution functionality."""
    
    def test_run_training_pipeline_thread(self, ml_service):
        """Test training pipeline execution in thread."""
        with patch('app.services.ml_pipeline_service.run_ml_training_sync') as mock_training, \
             patch('app.services.ml_pipeline_service.get_db_session') as mock_db, \
             patch.object(ml_service, '_save_pipeline_results') as mock_save:
            
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_pipeline_run = Mock()
            mock_session.get.return_value = mock_pipeline_run
            
            mock_training.return_value = {
                "success": True,
                "total_time_seconds": 120.0,
                "best_model_summary": {
                    "algorithm": "random_forest_classifier",
                    "primary_metric": {"value": 0.95}
                },
                "training_results": [{"algorithm_name": "random_forest_classifier"}]
            }
            
            training_config = {
                "file_path": "test.csv",
                "target_column": "target",
                "problem_type": "classification",
                "algorithms": [],
                "preprocessing_config": {},
                "pipeline_run_id": "test-uuid"
            }
            
            # Call the method
            ml_service._run_training_pipeline(training_config, 1)
            
            # Verify training was called
            mock_training.assert_called_once_with(training_config)
            mock_save.assert_called_once()
            
            # Verify database updates
            assert mock_pipeline_run.status == PipelineStatusEnum.COMPLETED.value
            assert mock_session.commit.called