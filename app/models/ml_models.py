"""
ML Models for the RAG System

This module defines the database models for ML training functionality,
seamlessly integrated with the existing RAG system architecture.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

class Base(DeclarativeBase):
    pass

class ProblemTypeEnum(str, Enum):
    """Supported ML problem types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class AlgorithmNameEnum(str, Enum):
    """Supported ML algorithms"""
    # Classification algorithms
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    SVM_CLASSIFIER = "svm_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    NAIVE_BAYES = "naive_bayes"
    
    # Regression algorithms
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    SVM_REGRESSOR = "svm_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"

class MetricNameEnum(str, Enum):
    """Supported evaluation metrics"""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    
    # Regression metrics
    MAE = "mae"  # Mean Absolute Error
    MSE = "mse"  # Mean Squared Error
    RMSE = "rmse"  # Root Mean Squared Error
    R2_SCORE = "r2_score"  # R-squared

class PipelineStatusEnum(str, Enum):
    """ML pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MLPipelineRun(Base):
    """
    Tracks individual ML training pipeline executions
    
    Links to existing document management system while providing
    comprehensive ML-specific tracking and configuration storage.
    """
    __tablename__ = "ml_pipeline_run"
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    run_uuid = Column(String, unique=True, index=True, nullable=False)
    
    # Link to existing RAG system (when documents are used as datasets)
    # Note: This would link to UploadedFileLog if it existed as a proper model
    # For now, we'll store the document reference as a string
    uploaded_file_reference = Column(String, nullable=True)  # Can be file path or document ID
    
    # ML-specific configuration
    problem_type = Column(String, nullable=False)  # ProblemTypeEnum
    target_variable = Column(String, nullable=False)
    selected_features = Column(JSON, nullable=True)  # List of feature names
    ml_config = Column(JSON, nullable=True)  # Complete ML configuration
    
    # Execution tracking
    status = Column(String, default=PipelineStatusEnum.PENDING.value, nullable=False)
    progress_percentage = Column(Float, default=0.0)
    current_stage = Column(String, nullable=True)  # Current execution stage
    
    # Results summary
    best_model_id = Column(String, nullable=True)
    best_model_algorithm = Column(String, nullable=True)
    best_model_score = Column(Float, nullable=True)
    total_models_trained = Column(Integer, default=0)
    
    # Performance metrics
    total_training_time_seconds = Column(Float, nullable=True)
    preprocessing_time_seconds = Column(Float, nullable=True)
    evaluation_time_seconds = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Foreign key relationships
    experiment_id = Column(Integer, ForeignKey("ml_experiment.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    models = relationship("MLModel", back_populates="pipeline_run", cascade="all, delete-orphan")
    preprocessing_logs = relationship("MLPreprocessingLog", back_populates="pipeline_run", cascade="all, delete-orphan")
    experiment = relationship("MLExperiment", back_populates="pipeline_runs")
    
    def __repr__(self):
        return f"<MLPipelineRun(id={self.id}, uuid={self.run_uuid}, status={self.status})>"

class MLModel(Base):
    """
    Stores individual trained model metadata and performance metrics
    
    Each model represents one algorithm trained within a pipeline run,
    with comprehensive performance tracking and hyperparameter storage.
    """
    __tablename__ = "ml_model"
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True, nullable=False)
    
    # Relationship to pipeline run
    pipeline_run_id = Column(Integer, ForeignKey("ml_pipeline_run.id"), nullable=False)
    pipeline_run_uuid = Column(String, nullable=False)  # For easy lookup
    
    # Model configuration
    algorithm_name = Column(String, nullable=False)  # AlgorithmNameEnum
    algorithm_display_name = Column(String, nullable=False)
    hyperparameters = Column(JSON, nullable=False)  # Algorithm hyperparameters
    random_state = Column(Integer, nullable=True)  # For reproducibility
    
    # Performance metrics
    performance_metrics = Column(JSON, nullable=False)  # All calculated metrics
    primary_metric_name = Column(String, nullable=False)  # Primary metric for comparison
    primary_metric_value = Column(Float, nullable=False)  # Primary metric value
    cross_validation_scores = Column(JSON, nullable=True)  # CV scores if available
    
    # Feature information
    feature_names = Column(JSON, nullable=True)  # List of feature names used
    feature_importance = Column(JSON, nullable=True)  # Feature importance scores
    n_features = Column(Integer, nullable=True)  # Number of features used
    
    # Training information
    training_time_seconds = Column(Float, nullable=False)
    training_samples = Column(Integer, nullable=True)  # Number of training samples
    test_samples = Column(Integer, nullable=True)  # Number of test samples
    
    # Model persistence
    model_file_path = Column(String, nullable=True)  # Path to saved model file
    model_size_bytes = Column(Integer, nullable=True)  # Model file size
    
    # Status and ranking
    is_best_model = Column(Boolean, default=False, nullable=False)
    rank_in_pipeline = Column(Integer, nullable=True)  # Rank among models in pipeline
    training_status = Column(String, default="COMPLETED", nullable=False)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    pipeline_run = relationship("MLPipelineRun", back_populates="models")
    
    def __repr__(self):
        return f"<MLModel(id={self.id}, algorithm={self.algorithm_name}, score={self.primary_metric_value})>"

class MLExperiment(Base):
    """
    Groups related ML pipeline runs for comparison and experiment tracking
    
    Enables users to organize and compare multiple training runs
    across different configurations, datasets, or time periods.
    """
    __tablename__ = "ml_experiment"
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, unique=True, index=True, nullable=False)
    
    # Experiment metadata
    name = Column(String, nullable=False)
    experiment_name = Column(String, nullable=True)  # For backwards compatibility
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags for organization
    
    # Dataset information
    dataset_reference = Column(String, nullable=True)  # Reference to dataset used
    dataset_name = Column(String, nullable=True)
    problem_type = Column(String, nullable=False)  # ProblemTypeEnum
    target_variable = Column(String, nullable=False)
    
    # Experiment summary
    total_runs = Column(Integer, default=0, nullable=False)
    successful_runs = Column(Integer, default=0, nullable=False)
    failed_runs = Column(Integer, default=0, nullable=False)
    
    # Best results across all runs
    best_run_id = Column(String, nullable=True)
    best_model_id = Column(String, nullable=True)
    best_score = Column(Float, nullable=True)
    best_algorithm = Column(String, nullable=True)
    
    # Experiment status
    status = Column(String, default="ACTIVE", nullable=False)  # ACTIVE, ARCHIVED, DELETED
    is_favorite = Column(Boolean, default=False, nullable=False)
    
    # User and collaboration
    created_by = Column(String, nullable=True)  # User identifier if available
    shared_with = Column(JSON, nullable=True)  # List of users with access
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    last_run_at = Column(DateTime, nullable=True)
    
    # Relationships
    pipeline_runs = relationship("MLPipelineRun", back_populates="experiment")
    
    def __init__(self, **kwargs):
        # Handle backwards compatibility - if experiment_name is provided but name is not
        if 'experiment_name' in kwargs and 'name' not in kwargs:
            kwargs['name'] = kwargs['experiment_name']
        elif 'experiment_name' in kwargs and kwargs.get('name') is None:
            kwargs['name'] = kwargs['experiment_name']
        
        # Set defaults for required fields if not provided
        if 'problem_type' not in kwargs:
            kwargs['problem_type'] = ProblemTypeEnum.CLASSIFICATION.value
        if 'target_variable' not in kwargs:
            kwargs['target_variable'] = "target"
            
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<MLExperiment(id={self.id}, name={self.name}, runs={self.total_runs})>"

class MLPreprocessingLog(Base):
    """
    Tracks preprocessing steps and transformations applied to datasets
    
    Provides detailed logging of all preprocessing operations for
    reproducibility and debugging purposes.
    """
    __tablename__ = "ml_preprocessing_log"
    
    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String, unique=True, index=True, nullable=False)
    
    # Relationship to pipeline run
    pipeline_run_id = Column(Integer, ForeignKey("ml_pipeline_run.id"), nullable=False)
    
    # Preprocessing configuration
    preprocessing_config = Column(JSON, nullable=False)  # Complete preprocessing config
    selected_features = Column(JSON, nullable=True)  # Features selected by user
    
    # Data information
    original_features = Column(JSON, nullable=False, default=lambda: [])  # List of original feature names
    final_features = Column(JSON, nullable=False, default=lambda: [])  # List of final feature names after preprocessing
    feature_types = Column(JSON, nullable=True)  # Types of each feature
    
    # Data statistics
    original_shape = Column(JSON, nullable=True)  # [rows, columns] before preprocessing
    final_shape = Column(JSON, nullable=True)  # [rows, columns] after preprocessing
    missing_values_summary = Column(JSON, nullable=True)  # Missing values per feature
    categorical_encodings = Column(JSON, nullable=True)  # Categorical encoding mappings
    
    # Preprocessing steps applied
    steps_applied = Column(JSON, nullable=False)  # List of preprocessing steps
    transformations = Column(JSON, nullable=True)  # Details of transformations applied
    high_cardinality_features = Column(JSON, nullable=True)  # Features with high cardinality
    
    # Performance metrics
    preprocessing_time_seconds = Column(Float, nullable=False)
    memory_usage_mb = Column(Float, nullable=True)
    
    # Quality checks
    warnings = Column(JSON, nullable=True)  # List of warnings generated
    errors = Column(JSON, nullable=True)  # List of errors encountered
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    pipeline_run = relationship("MLPipelineRun", back_populates="preprocessing_logs")
    
    def __repr__(self):
        return f"<MLPreprocessingLog(id={self.id}, pipeline_run_id={self.pipeline_run_id})>"

# Utility functions for model creation and management

def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique ID for ML entities"""
    return f"{prefix}{uuid.uuid4()}" if prefix else str(uuid.uuid4())

def create_pipeline_run(
    problem_type: ProblemTypeEnum,
    target_variable: str,
    ml_config: Dict[str, Any],
    uploaded_file_reference: Optional[str] = None,
    experiment_id: Optional[int] = None
) -> MLPipelineRun:
    """Factory function to create a new ML pipeline run"""
    now = datetime.now(timezone.utc)
    return MLPipelineRun(
        run_uuid=generate_unique_id("run_"),
        uploaded_file_reference=uploaded_file_reference,
        problem_type=problem_type.value,
        target_variable=target_variable,
        ml_config=ml_config,
        status=PipelineStatusEnum.PENDING.value,
        experiment_id=experiment_id,
        created_at=now,
        updated_at=now
    )

def create_ml_model(
    pipeline_run_id: int,
    pipeline_run_uuid: str,
    algorithm_name: AlgorithmNameEnum,
    algorithm_display_name: str,
    hyperparameters: Dict[str, Any],
) -> MLModel:
    """Factory function to create a new ML model record"""
    now = datetime.now(timezone.utc)
    return MLModel(
        model_id=generate_unique_id("model_"),
        pipeline_run_id=pipeline_run_id,
        pipeline_run_uuid=pipeline_run_uuid,
        algorithm_name=algorithm_name.value,
        algorithm_display_name=algorithm_display_name,
        hyperparameters=hyperparameters,
        performance_metrics={},
        primary_metric_name="",
        primary_metric_value=0.0,
        training_time_seconds=0.0,
        created_at=now
    )

def create_ml_experiment(
    name: str,
    problem_type: ProblemTypeEnum = None,
    target_variable: str = None,
    description: Optional[str] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> MLExperiment:
    """Factory function to create a new ML experiment"""
    # Handle backwards compatibility with experiment_name
    if experiment_name and not name:
        name = experiment_name
    
    # Set defaults for optional parameters if not provided
    if problem_type is None:
        problem_type = ProblemTypeEnum.CLASSIFICATION
    if target_variable is None:
        target_variable = "target"
        
    now = datetime.now(timezone.utc)
    experiment = MLExperiment(
        experiment_id=generate_unique_id("exp_"),
        name=name,
        description=description,
        problem_type=problem_type.value if hasattr(problem_type, 'value') else problem_type,
        target_variable=target_variable,
        created_at=now,
        updated_at=now,
        **kwargs
    )
    
    # Set experiment_name for backwards compatibility
    if experiment_name:
        experiment.experiment_name = experiment_name
        
    return experiment

def create_preprocessing_log(
    pipeline_run_id: int,
    preprocessing_config: Dict[str, Any],
    steps_applied: List[str],
    preprocessing_time_seconds: float,
    original_features: Optional[List[str]] = None,
    final_features: Optional[List[str]] = None,
    **kwargs
) -> MLPreprocessingLog:
    """Factory function to create a new preprocessing log"""
    now = datetime.now(timezone.utc)
    return MLPreprocessingLog(
        log_id=generate_unique_id("log_"),
        pipeline_run_id=pipeline_run_id,
        preprocessing_config=preprocessing_config,
        steps_applied=steps_applied,
        preprocessing_time_seconds=preprocessing_time_seconds,
        original_features=original_features or [],
        final_features=final_features or [],
        created_at=now,
        **kwargs
    )