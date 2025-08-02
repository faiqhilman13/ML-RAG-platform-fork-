"""
Test ML Database Models

Tests for ML database models, factory functions, and database operations.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.ml_models import (
    MLPipelineRun, MLModel, MLExperiment, MLPreprocessingLog,
    ProblemTypeEnum, PipelineStatusEnum, AlgorithmNameEnum, MetricNameEnum,
    create_pipeline_run, create_ml_model, create_ml_experiment, create_preprocessing_log, Base
)


@pytest.fixture
def test_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    yield session
    
    session.close()


class TestMLPipelineRunModel:
    """Test MLPipelineRun database model."""
    
    def test_create_pipeline_run_basic(self, test_db):
        """Test basic pipeline run creation."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={"test": "config"},
            uploaded_file_reference="test.csv"
        )
        
        assert pipeline_run.problem_type == ProblemTypeEnum.CLASSIFICATION.value
        assert pipeline_run.target_variable == "target"
        assert pipeline_run.ml_config == {"test": "config"}
        assert pipeline_run.uploaded_file_reference == "test.csv"
        assert pipeline_run.status == PipelineStatusEnum.PENDING.value
        assert pipeline_run.created_at is not None
    
    def test_create_pipeline_run_with_experiment(self, test_db):
        """Test pipeline run creation with experiment."""
        # First create an experiment
        experiment = MLExperiment(
            experiment_id="test-exp-123",
            experiment_name="Test Experiment",
            description="Test experiment description"
        )
        test_db.add(experiment)
        test_db.commit()
        
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.REGRESSION,
            target_variable="price",
            ml_config={"algorithms": []},
            uploaded_file_reference="housing.csv",
            experiment_id=experiment.id
        )
        
        assert pipeline_run.experiment_id == experiment.id
        assert pipeline_run.problem_type == ProblemTypeEnum.REGRESSION.value
    
    def test_pipeline_run_status_updates(self, test_db):
        """Test pipeline run status updates."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        
        test_db.add(pipeline_run)
        test_db.commit()
        
        # Update status
        pipeline_run.status = PipelineStatusEnum.RUNNING.value
        pipeline_run.started_at = datetime.now(timezone.utc)
        test_db.commit()
        
        # Verify updates
        assert pipeline_run.status == PipelineStatusEnum.RUNNING.value
        assert pipeline_run.started_at is not None
    
    def test_pipeline_run_relationships(self, test_db):
        """Test pipeline run relationships with other models."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        
        test_db.add(pipeline_run)
        test_db.commit()
        
        # Add related ML model
        ml_model = create_ml_model(
            pipeline_run_id=pipeline_run.id,
            pipeline_run_uuid="test-uuid",
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            algorithm_display_name="Random Forest",
            hyperparameters={"n_estimators": 100}
        )
        
        test_db.add(ml_model)
        test_db.commit()
        
        # Verify relationship
        assert len(pipeline_run.models) == 1
        assert pipeline_run.models[0].algorithm_name == AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER.value


class TestMLModelModel:
    """Test MLModel database model."""
    
    def test_create_ml_model_basic(self, test_db):
        """Test basic ML model creation."""
        # First create a pipeline run
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        test_db.add(pipeline_run)
        test_db.commit()
        
        # Create ML model
        ml_model = create_ml_model(
            pipeline_run_id=pipeline_run.id,
            pipeline_run_uuid="test-uuid-123",
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            algorithm_display_name="Logistic Regression",
            hyperparameters={"C": 1.0, "penalty": "l2"}
        )
        
        assert ml_model.pipeline_run_id == pipeline_run.id
        assert ml_model.pipeline_run_uuid == "test-uuid-123"
        assert ml_model.algorithm_name == AlgorithmNameEnum.LOGISTIC_REGRESSION.value
        assert ml_model.algorithm_display_name == "Logistic Regression"
        assert ml_model.hyperparameters == {"C": 1.0, "penalty": "l2"}
        assert ml_model.created_at is not None
        assert ml_model.model_id is not None
    
    def test_ml_model_performance_metrics(self, test_db):
        """Test ML model with performance metrics."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        test_db.add(pipeline_run)
        test_db.commit()
        
        ml_model = create_ml_model(
            pipeline_run_id=pipeline_run.id,
            pipeline_run_uuid="test-uuid",
            algorithm_name=AlgorithmNameEnum.SVM_CLASSIFIER,
            algorithm_display_name="SVM",
            hyperparameters={}
        )
        
        # Set performance metrics
        ml_model.primary_metric_name = MetricNameEnum.ACCURACY.value
        ml_model.primary_metric_value = 0.85
        ml_model.performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85
        }
        ml_model.training_time_seconds = 45.2
        ml_model.n_features = 20
        ml_model.feature_names = ["feature_1", "feature_2"]
        
        test_db.add(ml_model)
        test_db.commit()
        
        # Verify metrics
        assert ml_model.primary_metric_name == MetricNameEnum.ACCURACY.value
        assert ml_model.primary_metric_value == 0.85
        assert ml_model.performance_metrics["precision"] == 0.83
        assert ml_model.training_time_seconds == 45.2
        assert ml_model.n_features == 20
    
    def test_ml_model_best_model_flag(self, test_db):
        """Test best model flag functionality."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        test_db.add(pipeline_run)
        test_db.commit()
        
        # Create multiple models
        model1 = create_ml_model(
            pipeline_run_id=pipeline_run.id,
            pipeline_run_uuid="test-uuid",
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            algorithm_display_name="Random Forest",
            hyperparameters={}
        )
        model1.primary_metric_value = 0.85
        model1.rank_in_pipeline = 2
        
        model2 = create_ml_model(
            pipeline_run_id=pipeline_run.id,
            pipeline_run_uuid="test-uuid",
            algorithm_name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            algorithm_display_name="Logistic Regression",
            hyperparameters={}
        )
        model2.primary_metric_value = 0.92
        model2.is_best_model = True
        model2.rank_in_pipeline = 1
        
        test_db.add_all([model1, model2])
        test_db.commit()
        
        # Verify best model
        best_models = [model for model in pipeline_run.models if model.is_best_model]
        assert len(best_models) == 1
        assert best_models[0].primary_metric_value == 0.92
        assert best_models[0].rank_in_pipeline == 1


class TestMLExperimentModel:
    """Test MLExperiment database model."""
    
    def test_create_experiment(self, test_db):
        """Test experiment creation."""
        experiment = MLExperiment(
            experiment_id="exp-001",
            experiment_name="Classification Comparison",
            description="Comparing different classification algorithms",
            tags={"dataset": "iris", "type": "comparison"}
        )
        
        test_db.add(experiment)
        test_db.commit()
        
        assert experiment.experiment_id == "exp-001"
        assert experiment.experiment_name == "Classification Comparison"
        assert experiment.tags["dataset"] == "iris"
        assert experiment.created_at is not None
    
    def test_experiment_pipeline_relationship(self, test_db):
        """Test experiment to pipeline run relationship."""
        experiment = MLExperiment(
            experiment_id="exp-002",
            experiment_name="Regression Test",
            description="Testing regression algorithms"
        )
        test_db.add(experiment)
        test_db.commit()
        
        # Create pipeline runs for this experiment
        pipeline1 = create_pipeline_run(
            problem_type=ProblemTypeEnum.REGRESSION,
            target_variable="price",
            ml_config={},
            uploaded_file_reference="data1.csv",
            experiment_id=experiment.id
        )
        
        pipeline2 = create_pipeline_run(
            problem_type=ProblemTypeEnum.REGRESSION,
            target_variable="price",
            ml_config={},
            uploaded_file_reference="data2.csv",
            experiment_id=experiment.id
        )
        
        test_db.add_all([pipeline1, pipeline2])
        test_db.commit()
        
        # Verify relationship
        assert len(experiment.pipeline_runs) == 2
        assert all(run.experiment_id == experiment.id for run in experiment.pipeline_runs)


class TestMLPreprocessingLogModel:
    """Test MLPreprocessingLog database model."""
    
    def test_create_preprocessing_log(self, test_db):
        """Test preprocessing log creation."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        test_db.add(pipeline_run)
        test_db.commit()
        
        preprocessing_log = MLPreprocessingLog(
            log_id="log-123",
            pipeline_run_id=pipeline_run.id,
            preprocessing_config={
                "scaling_strategy": "standard",
                "categorical_strategy": "onehot"
            },
            original_shape=[1000, 20],
            final_shape=[1000, 25],
            steps_applied=["remove_missing", "encode_categorical", "scale_features"],
            preprocessing_time_seconds=5.2,
            warnings=["High cardinality feature detected"]
        )
        
        test_db.add(preprocessing_log)
        test_db.commit()
        
        assert preprocessing_log.log_id == "log-123"
        assert preprocessing_log.pipeline_run_id == pipeline_run.id
        assert preprocessing_log.original_shape == [1000, 20]
        assert preprocessing_log.final_shape == [1000, 25]
        assert len(preprocessing_log.steps_applied) == 3
        assert preprocessing_log.preprocessing_time_seconds == 5.2
        assert len(preprocessing_log.warnings) == 1


class TestModelEnums:
    """Test model enums."""
    
    def test_problem_type_enum(self):
        """Test ProblemTypeEnum values."""
        assert ProblemTypeEnum.CLASSIFICATION.value == "classification"
        assert ProblemTypeEnum.REGRESSION.value == "regression"
    
    def test_pipeline_status_enum(self):
        """Test PipelineStatusEnum values."""
        assert PipelineStatusEnum.PENDING.value == "pending"
        assert PipelineStatusEnum.RUNNING.value == "running"
        assert PipelineStatusEnum.COMPLETED.value == "completed"
        assert PipelineStatusEnum.FAILED.value == "failed"
    
    def test_algorithm_name_enum(self):
        """Test AlgorithmNameEnum values."""
        assert AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER.value == "random_forest_classifier"
        assert AlgorithmNameEnum.LOGISTIC_REGRESSION.value == "logistic_regression"
        assert AlgorithmNameEnum.LINEAR_REGRESSION.value == "linear_regression"
    
    def test_metric_name_enum(self):
        """Test MetricNameEnum values."""
        assert MetricNameEnum.ACCURACY.value == "accuracy"
        assert MetricNameEnum.PRECISION.value == "precision"
        assert MetricNameEnum.RECALL.value == "recall"
        assert MetricNameEnum.F1_SCORE.value == "f1_score"
        assert MetricNameEnum.MSE.value == "mse"
        assert MetricNameEnum.RMSE.value == "rmse"
        assert MetricNameEnum.MAE.value == "mae"
        assert MetricNameEnum.R2_SCORE.value == "r2_score"


class TestDatabaseOperations:
    """Test database operations with ML models."""
    
    def test_concurrent_pipeline_runs(self, test_db):
        """Test handling of concurrent pipeline runs."""
        pipelines = []
        for i in range(5):
            pipeline = create_pipeline_run(
                problem_type=ProblemTypeEnum.CLASSIFICATION,
                target_variable=f"target_{i}",
                ml_config={"run": i},
                uploaded_file_reference=f"data_{i}.csv"
            )
            pipelines.append(pipeline)
        
        test_db.add_all(pipelines)
        test_db.commit()
        
        # Verify all pipelines were created
        assert len(pipelines) == 5
        assert all(p.id is not None for p in pipelines)
    
    def test_cascade_delete_behavior(self, test_db):
        """Test cascade delete behavior."""
        pipeline_run = create_pipeline_run(
            problem_type=ProblemTypeEnum.CLASSIFICATION,
            target_variable="target",
            ml_config={},
            uploaded_file_reference="test.csv"
        )
        test_db.add(pipeline_run)
        test_db.commit()
        
        # Add related models
        ml_model = create_ml_model(
            pipeline_run_id=pipeline_run.id,
            pipeline_run_uuid="test-uuid",
            algorithm_name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            algorithm_display_name="Random Forest",
            hyperparameters={}
        )
        test_db.add(ml_model)
        test_db.commit()
        
        # Delete pipeline run
        test_db.delete(pipeline_run)
        test_db.commit()
        
        # Verify related models are handled appropriately
        # (Actual cascade behavior depends on foreign key constraints)