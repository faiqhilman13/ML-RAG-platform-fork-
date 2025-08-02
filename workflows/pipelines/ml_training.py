"""
ML Training Pipeline with Prefect Orchestration

This module provides Prefect-orchestrated ML training workflows with
comprehensive error handling, deterministic results, and progress tracking.
"""

import logging
import pandas as pd
import numpy as np
import time
import traceback
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import pickle
import uuid

# Prefect imports with fallback for environments without Prefect
# Temporarily disable Prefect due to Pydantic v2 compatibility issues
PREFECT_AVAILABLE = False

if PREFECT_AVAILABLE:
    try:
        from prefect import flow, task, get_run_logger
        from prefect.task_runners import SequentialTaskRunner
    except ImportError:
        PREFECT_AVAILABLE = False

if not PREFECT_AVAILABLE:
    # Create mock decorators for environments without Prefect
    def flow(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def task(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def get_run_logger():
        return logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from sklearn.base import clone

from app.models.ml_models import ProblemTypeEnum, AlgorithmNameEnum, PipelineStatusEnum
from workflows.ml.algorithm_registry import get_algorithm_registry, generate_unique_random_state
from workflows.ml.preprocessing import DataPreprocessor, PreprocessingConfig, create_preprocessing_config, preprocess_ml_data
from workflows.ml.evaluation import get_model_evaluator, get_model_comparator, ModelAnalysis
from app.config import BASE_DIR

logger = logging.getLogger(__name__)

# Directory for storing trained models with organized structure
MODELS_DIR = BASE_DIR / "data" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def create_organized_training_directory(pipeline_run_id: str) -> Dict[str, Path]:
    """
    Create organized directory structure for training results
    
    Creates a structured directory layout:
    data/trained_models/
    ├── pipeline_{run_id}/
    │   ├── models/           # .pkl model files
    │   ├── metadata/         # .json metadata files
    │   ├── logs/            # training logs
    │   └── results/         # evaluation results and visualizations
    └── pipeline_{run_id}.json  # Main pipeline metadata
    
    Returns:
        Dictionary with paths to each subdirectory
    """
    try:
        # Create main pipeline directory
        pipeline_dir = MODELS_DIR / f"pipeline_{pipeline_run_id}"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = {
            "pipeline_dir": pipeline_dir,
            "models_dir": pipeline_dir / "models",
            "metadata_dir": pipeline_dir / "metadata", 
            "logs_dir": pipeline_dir / "logs",
            "results_dir": pipeline_dir / "results"
        }
        
        for subdir_path in subdirs.values():
            if subdir_path != pipeline_dir:  # Skip the main pipeline dir as it's already created
                subdir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Created organized training directory structure at: {pipeline_dir}")
        
        return subdirs
        
    except Exception as e:
        logger.error(f"❌ Failed to create organized directory structure: {str(e)}")
        # Fallback to flat structure
        return {
            "pipeline_dir": MODELS_DIR,
            "models_dir": MODELS_DIR,
            "metadata_dir": MODELS_DIR,
            "logs_dir": MODELS_DIR,
            "results_dir": MODELS_DIR
        }

def save_model_metadata(
    metadata_path: Path,
    model_id: str,
    algorithm_name: str,
    algorithm_display_name: str,
    hyperparameters: Dict[str, Any],
    training_time_seconds: float,
    model_file_path: str,
    random_state: int,
    created_at: str,
    **kwargs
) -> None:
    """
    Save model training metadata as JSON file
    
    Creates a comprehensive metadata file alongside the .pkl model file
    containing all training configuration, hyperparameters, and basic info.
    """
    try:
        metadata = {
            "model_info": {
                "model_id": model_id,
                "algorithm_name": algorithm_name,
                "algorithm_display_name": algorithm_display_name,
                "model_file_path": model_file_path,
                "created_at": created_at
            },
            "training_config": {
                "hyperparameters": hyperparameters,
                "random_state": random_state,
                "training_time_seconds": training_time_seconds
            },
            "file_info": {
                "metadata_version": "1.0",
                "model_filename": os.path.basename(model_file_path),
                "metadata_filename": os.path.basename(str(metadata_path))
            }
        }
        
        # Add any additional metadata passed via kwargs
        if kwargs:
            metadata["additional_info"] = kwargs
            
        # Write metadata to JSON file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📄 Saved model metadata: {metadata_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model metadata {metadata_path}: {str(e)}")
        # Don't raise - metadata saving failure shouldn't break training

def save_pipeline_metadata(
    pipeline_run_id: str,
    config: Dict[str, Any],
    pipeline_result: 'PipelineResult',
    models_dir: Path = MODELS_DIR
) -> None:
    """
    Save comprehensive pipeline metadata as JSON file
    
    Creates a pipeline-level metadata file containing training configuration,
    results summary, and references to all trained models.
    """
    try:
        pipeline_filename = f"pipeline_{pipeline_run_id}.json"
        pipeline_metadata_path = models_dir / pipeline_filename
        
        # Extract key information from pipeline result
        training_results = pipeline_result.training_results or []
        evaluation_results = pipeline_result.evaluation_results or []
        
        # Prepare model summaries
        model_summaries = []
        for i, training_result in enumerate(training_results):
            evaluation_result = evaluation_results[i] if i < len(evaluation_results) else None
            
            model_summary = {
                "model_id": training_result.model_id,
                "algorithm_name": training_result.algorithm_name,
                "algorithm_display_name": training_result.algorithm_display_name,
                "hyperparameters": training_result.hyperparameters,
                "training_time_seconds": training_result.training_time_seconds,
                "model_file_path": training_result.model_file_path,
                "random_state": training_result.random_state,
                "error": training_result.error,
                "performance_metrics": evaluation_result.to_dict() if evaluation_result else None
            }
            model_summaries.append(model_summary)
        
        # Prepare pipeline metadata
        pipeline_metadata = {
            "pipeline_info": {
                "pipeline_run_id": pipeline_run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "success": pipeline_result.success,
                "total_training_time_seconds": pipeline_result.total_time_seconds
            },
            "training_config": {
                "file_path": config.get("file_path"),
                "target_column": config.get("target_column"),
                "problem_type": config.get("problem_type"),
                "algorithms_config": config.get("algorithms", []),
                "preprocessing_config": config.get("preprocessing_config", {})
            },
            "results_summary": {
                "total_models_trained": len(model_summaries),
                "successful_models": len([m for m in model_summaries if not m.get("error")]),
                "failed_models": len([m for m in model_summaries if m.get("error")]),
                "best_model": (
                    pipeline_result.best_model_analysis.to_dict() 
                    if pipeline_result.best_model_analysis 
                    else None
                )
            },
            "preprocessing_summary": (
                pipeline_result.preprocessing_result.to_dict() 
                if pipeline_result.preprocessing_result 
                else None
            ),
            "models": model_summaries,
            "file_info": {
                "metadata_version": "1.0",
                "pipeline_metadata_filename": pipeline_filename
            }
        }
        
        # Write pipeline metadata to JSON file
        with open(pipeline_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📋 Saved pipeline metadata: {pipeline_metadata_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save pipeline metadata: {str(e)}")
        # Don't raise - metadata saving failure shouldn't break training

@dataclass
class TrainingResult:
    """Result of training a single algorithm"""
    model_id: str
    algorithm_name: str
    algorithm_display_name: str
    model: Any = None
    model_analysis: Optional[ModelAnalysis] = None
    training_time_seconds: float = 0.0
    model_file_path: Optional[str] = None
    hyperparameters: Dict[str, Any] = None
    random_state: int = 42
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_id": self.model_id,
            "algorithm_name": self.algorithm_name,
            "algorithm_display_name": self.algorithm_display_name,
            "training_time_seconds": self.training_time_seconds,
            "model_file_path": self.model_file_path,
            "hyperparameters": self.hyperparameters,
            "random_state": self.random_state,
            "error": self.error,
            "has_model": self.model is not None,
            "model_analysis_summary": self.model_analysis.get_summary() if self.model_analysis else None
        }

@dataclass
class PipelineResult:
    """Result of complete ML training pipeline"""
    success: bool
    pipeline_run_id: str
    preprocessing_result: Any = None
    training_results: List[TrainingResult] = None
    evaluation_results: List[ModelAnalysis] = None
    best_model_analysis: Optional[ModelAnalysis] = None
    comparison_results: Optional[Dict[str, Any]] = None
    total_time_seconds: float = 0.0
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.training_results is None:
            self.training_results = []
        if self.evaluation_results is None:
            self.evaluation_results = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "pipeline_run_id": self.pipeline_run_id,
            "total_time_seconds": self.total_time_seconds,
            "error": self.error,
            "preprocessing_summary": self.preprocessing_result.get_summary() if self.preprocessing_result else None,
            "training_results": [result.to_dict() for result in self.training_results],
            "evaluation_results": [analysis.get_summary() for analysis in self.evaluation_results],
            "best_model_summary": self.best_model_analysis.get_summary() if self.best_model_analysis else None,
            "comparison_results": self.comparison_results
        }

@task(name="load_and_validate_data")
def load_and_validate_data_task(
    file_path: str,
    target_column: str,
    problem_type: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate dataset for ML training
    
    Args:
        file_path: Path to the dataset file
        target_column: Name of the target column
        problem_type: Type of ML problem ('classification' or 'regression')
    
    Returns:
        Tuple of (dataframe, validation_info)
    """
    logger = get_run_logger()
    logger.info(f"Loading data from: {file_path}")
    
    try:
        # Convert to absolute path if needed
        from pathlib import Path
        file_path_obj = Path(file_path)
        
        # If path is not absolute, resolve it correctly
        if not file_path_obj.is_absolute():
            from app.config import BASE_DIR, DATASETS_DIR
            
            # DEFINITIVE FIX: Always check DATASETS_DIR first since that's where files actually are
            filename = file_path_obj.name  # Get just the filename
            datasets_path = DATASETS_DIR / filename
            
            if datasets_path.exists():
                # File exists in the correct datasets directory
                file_path_obj = datasets_path
                logger.info(f"Found file in datasets directory: {file_path_obj}")
            else:
                # Fallback: try the provided path relative to BASE_DIR
                file_path_obj = BASE_DIR / file_path
                logger.info(f"Trying relative to BASE_DIR: {file_path_obj}")
        
        # Convert back to string for pandas
        absolute_file_path = str(file_path_obj)
        logger.info(f"Resolved absolute path: {absolute_file_path}")
        
        # Verify file exists
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {absolute_file_path}")
        
        # Load data (support CSV for now, can extend to other formats)
        if absolute_file_path.endswith('.csv'):
            df = pd.read_csv(absolute_file_path)
        else:
            raise ValueError(f"Unsupported file format: {absolute_file_path}")
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Basic data validation
        validation_info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "target_column": target_column,
            "target_type": str(df[target_column].dtype),
            "missing_values": df.isnull().sum().sum(),
            "target_unique_values": df[target_column].nunique(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Problem type validation
        if problem_type == ProblemTypeEnum.CLASSIFICATION.value:
            if df[target_column].nunique() > len(df) * 0.5:
                logger.warning("High number of unique target values for classification problem")
        
        logger.info(f"Data validation completed: {validation_info}")
        return df, validation_info
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@task(name="preprocess_data")
def preprocess_data_task(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    preprocessing_config: Dict[str, Any],
    pipeline_run_id: str
) -> Any:
    """
    Preprocess data for ML training
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        problem_type: Type of ML problem
        preprocessing_config: Preprocessing configuration
        pipeline_run_id: Pipeline run identifier
    
    Returns:
        PreprocessingResult object
    """
    logger = get_run_logger()
    logger.info("Starting data preprocessing")
    
    try:
        # Create preprocessing configuration
        config = create_preprocessing_config(
            problem_type=ProblemTypeEnum(problem_type),
            **preprocessing_config
        )
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Perform preprocessing
        result = preprocessor.preprocess_data(
            df=df,
            target_col=target_column,
            problem_type=ProblemTypeEnum(problem_type),
            pipeline_run_id=pipeline_run_id
        )
        
        logger.info(f"Preprocessing completed: {result.get_summary()}")
        return result
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

@task(name="train_single_algorithm")
def train_single_algorithm_task(
    preprocessing_result: Any,
    algorithm_config: Dict[str, Any],
    pipeline_run_id: str,
    problem_type: str,
    **kwargs
) -> TrainingResult:
    """
    Train a single ML algorithm
    
    Args:
        preprocessing_result: Result from preprocessing step
        algorithm_config: Configuration for the algorithm
        pipeline_run_id: Pipeline run identifier
        problem_type: Type of ML problem
    
    Returns:
        TrainingResult object
    """
    logger = get_run_logger()
    
    algorithm_name = algorithm_config.get("name")
    hyperparameters = algorithm_config.get("hyperparameters", {})
    
    logger.info(f"Training algorithm: {algorithm_name}")
    
    # Generate unique model ID
    model_id = f"{algorithm_name}_{pipeline_run_id}_{int(time.time())}"
    
    try:
        # Get algorithm registry
        registry = get_algorithm_registry()
        algorithm_enum = AlgorithmNameEnum(algorithm_name)
        algorithm_def = registry.get_algorithm(algorithm_enum)
        
        if not algorithm_def:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in registry")
        
        # Generate unique random state for deterministic results
        random_state = generate_unique_random_state(pipeline_run_id, algorithm_name)
        
        # Validate and prepare hyperparameters
        validated_params = algorithm_def.get_default_hyperparameters()
        validated_params.update(hyperparameters)
        validated_params['random_state'] = random_state
        
        # Validate hyperparameters
        is_valid, errors = registry.validate_hyperparameters(algorithm_enum, validated_params)
        if not is_valid:
            raise ValueError(f"Invalid hyperparameters: {errors}")
        
        # Import and instantiate the model
        module_path, class_name = algorithm_def.sklearn_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Filter parameters that the model actually accepts
        import inspect
        model_signature = inspect.signature(model_class.__init__)
        model_params = {
            k: v for k, v in validated_params.items()
            if k in model_signature.parameters
        }
        
        model = model_class(**model_params)
        
        # Train the model
        start_time = time.time()
        logger.info(f"Starting training with parameters: {model_params}")
        
        model.fit(preprocessing_result.X_train, preprocessing_result.y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model to organized structure
        model_filename = f"{model_id}.pkl"
        models_dir = kwargs.get('models_dir', MODELS_DIR)
        model_path = models_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model metadata as JSON to organized structure
        metadata_filename = f"{model_id}.json"
        metadata_dir = kwargs.get('metadata_dir', MODELS_DIR)
        metadata_path = metadata_dir / metadata_filename
        save_model_metadata(
            metadata_path=metadata_path,
            model_id=model_id,
            algorithm_name=algorithm_name,
            algorithm_display_name=algorithm_def.display_name,
            hyperparameters=validated_params,
            training_time_seconds=training_time,
            model_file_path=str(model_path),
            random_state=random_state,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        return TrainingResult(
            model_id=model_id,
            algorithm_name=algorithm_name,
            algorithm_display_name=algorithm_def.display_name,
            model=model,
            training_time_seconds=training_time,
            model_file_path=str(model_path),
            hyperparameters=validated_params,
            random_state=random_state
        )
        
    except Exception as e:
        logger.error(f"Error training {algorithm_name}: {str(e)}")
        return TrainingResult(
            model_id=model_id,
            algorithm_name=algorithm_name,
            algorithm_display_name=algorithm_config.get("display_name", algorithm_name),
            error=str(e),
            error_traceback=traceback.format_exc(),
            hyperparameters=hyperparameters
        )

@task(name="evaluate_model")
def evaluate_model_task(
    training_result: TrainingResult,
    preprocessing_result: Any,
    problem_type: str
) -> Optional[ModelAnalysis]:
    """
    Evaluate a trained model
    
    Args:
        training_result: Result from training step
        preprocessing_result: Result from preprocessing step
        problem_type: Type of ML problem
    
    Returns:
        ModelAnalysis object or None if evaluation failed
    """
    logger = get_run_logger()
    
    if training_result.error:
        logger.warning(f"Skipping evaluation for failed model: {training_result.algorithm_name}")
        return None
    
    logger.info(f"Evaluating model: {training_result.algorithm_display_name}")
    
    try:
        evaluator = get_model_evaluator()
        
        analysis = evaluator.evaluate_model(
            model=training_result.model,
            X_train=preprocessing_result.X_train,
            X_test=preprocessing_result.X_test,
            y_train=preprocessing_result.y_train,
            y_test=preprocessing_result.y_test,
            algorithm_name=training_result.algorithm_name,
            algorithm_display_name=training_result.algorithm_display_name,
            problem_type=ProblemTypeEnum(problem_type),
            hyperparameters=training_result.hyperparameters,
            training_time=training_result.training_time_seconds,
            model_id=training_result.model_id
        )
        
        # Update training result with analysis
        training_result.model_analysis = analysis
        
        logger.info(f"Evaluation completed for {training_result.algorithm_display_name}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error evaluating model {training_result.algorithm_display_name}: {str(e)}")
        return None

@task(name="select_best_model")
def select_best_model_task(
    evaluation_results: List[ModelAnalysis],
    problem_type: str
) -> Optional[ModelAnalysis]:
    """
    Select the best model from evaluation results
    
    Args:
        evaluation_results: List of ModelAnalysis objects
        problem_type: Type of ML problem
    
    Returns:
        Best ModelAnalysis or None
    """
    logger = get_run_logger()
    
    valid_results = [r for r in evaluation_results if r is not None and r.error is None]
    
    if not valid_results:
        logger.warning("No valid evaluation results to compare")
        return None
    
    logger.info(f"Selecting best model from {len(valid_results)} candidates")
    
    try:
        comparator = get_model_comparator()
        comparison = comparator.compare_models(valid_results)
        
        if comparison.get("best_model"):
            best_model_id = comparison["best_model"]["model_id"]
            best_analysis = next(
                (r for r in valid_results if r.model_id == best_model_id),
                None
            )
            
            if best_analysis:
                logger.info(f"Best model selected: {best_analysis.algorithm_display_name} "
                           f"(Score: {best_analysis.primary_metric.value:.4f})")
                return best_analysis
        
        # Fallback: select first valid result
        logger.warning("Could not determine best model from comparison, using first valid result")
        return valid_results[0]
        
    except Exception as e:
        logger.error(f"Error selecting best model: {str(e)}")
        return valid_results[0] if valid_results else None

@flow(name="ml_training_flow")
def ml_training_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete ML training pipeline flow
    
    Args:
        config: Pipeline configuration containing:
            - file_path: Path to dataset
            - target_column: Name of target column
            - problem_type: 'classification' or 'regression'
            - algorithms: List of algorithm configurations
            - preprocessing_config: Preprocessing configuration
            - pipeline_run_id: Unique pipeline identifier
    
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    pipeline_start_time = time.time()
    
    pipeline_run_id = config.get("pipeline_run_id", str(uuid.uuid4()))
    
    # Create organized directory structure for this training session
    directory_structure = create_organized_training_directory(pipeline_run_id)
    
    logger.info(f"🚀 Starting ML training pipeline: {pipeline_run_id}")
    
    try:
        # Step 1: Load and validate data
        logger.info(f"📂 Step 1: Loading and validating data from {config['file_path']}")
        df, validation_info = load_and_validate_data_task(
            config["file_path"],
            config["target_column"],
            config["problem_type"]
        )
        logger.info(f"✅ Step 1 completed: Data loaded with shape {validation_info.get('shape', 'unknown')}")
        
        # Step 2: Preprocess data
        logger.info(f"🔧 Step 2: Preprocessing data...")
        preprocessing_result = preprocess_data_task(
            df,
            config["target_column"],
            config["problem_type"],
            config.get("preprocessing_config", {}),
            pipeline_run_id
        )
        logger.info(f"✅ Step 2 completed: Data preprocessed")
        
        # Step 3: Train multiple algorithms
        training_results = []
        for algo_config in config["algorithms"]:
            result = train_single_algorithm_task(
                preprocessing_result,
                algo_config,
                pipeline_run_id,
                config["problem_type"],
                models_dir=directory_structure["models_dir"],
                metadata_dir=directory_structure["metadata_dir"]
            )
            training_results.append(result)
        
        # Step 4: Evaluate all models
        evaluation_results = []
        for training_result in training_results:
            eval_result = evaluate_model_task(
                training_result,
                preprocessing_result,
                config["problem_type"]
            )
            if eval_result:
                evaluation_results.append(eval_result)
        
        # Step 5: Select best model and compare
        best_model_analysis = select_best_model_task(
            evaluation_results,
            config["problem_type"]
        )
        
        # Step 6: Generate comparison results
        comparison_results = None
        if len(evaluation_results) > 1:
            try:
                comparator = get_model_comparator()
                comparison_results = comparator.compare_models(evaluation_results)
            except Exception as e:
                logger.warning(f"Could not generate comparison results: {e}")
        
        # Calculate total time
        total_time = time.time() - pipeline_start_time
        
        # Create pipeline result
        pipeline_result = PipelineResult(
            success=True,
            pipeline_run_id=pipeline_run_id,
            preprocessing_result=preprocessing_result,
            training_results=training_results,
            evaluation_results=evaluation_results,
            best_model_analysis=best_model_analysis,
            comparison_results=comparison_results,
            total_time_seconds=total_time
        )
        
        # Save comprehensive pipeline metadata
        save_pipeline_metadata(
            pipeline_run_id=pipeline_run_id,
            config=config,
            pipeline_result=pipeline_result,
            models_dir=directory_structure["pipeline_dir"].parent  # Save main pipeline file at trained_models level
        )
        
        logger.info(f"ML training pipeline completed successfully in {total_time:.2f}s")
        logger.info(f"Trained {len(training_results)} models, {len(evaluation_results)} successful")
        
        return pipeline_result.to_dict()
        
    except Exception as e:
        total_time = time.time() - pipeline_start_time
        logger.error(f"ML training pipeline failed after {total_time:.2f}s: {str(e)}")
        
        return {
            "success": False,
            "pipeline_run_id": pipeline_run_id,
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "total_time_seconds": total_time
        }

# Convenience functions for running without Prefect

def run_ml_training_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run ML training pipeline synchronously (without Prefect)
    
    This function provides the same functionality as the Prefect flow
    but can be used in environments where Prefect is not available.
    """
    try:
        logger.info(f"🎯 Starting run_ml_training_sync with config: {config}")
        result = ml_training_flow(config)
        logger.info(f"🎯 run_ml_training_sync completed with success: {result.get('success', False)}")
        return result
    except Exception as e:
        logger.error(f"🎯 run_ml_training_sync failed: {str(e)}")
        import traceback
        logger.error(f"🎯 Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "error_traceback": traceback.format_exc()
        }

def create_algorithm_config(
    algorithm_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create algorithm configuration for training pipeline
    
    Args:
        algorithm_name: Name of the algorithm (from AlgorithmNameEnum)
        hyperparameters: Optional custom hyperparameters
    
    Returns:
        Algorithm configuration dictionary
    """
    registry = get_algorithm_registry()
    
    try:
        algorithm_enum = AlgorithmNameEnum(algorithm_name)
        algorithm_def = registry.get_algorithm(algorithm_enum)
        
        if not algorithm_def:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        config = {
            "name": algorithm_name,
            "display_name": algorithm_def.display_name,
            "hyperparameters": hyperparameters or {}
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Error creating algorithm config for {algorithm_name}: {e}")
        raise

def get_default_algorithm_configs(problem_type: ProblemTypeEnum) -> List[Dict[str, Any]]:
    """
    Get default algorithm configurations for a problem type
    
    Args:
        problem_type: Type of ML problem
    
    Returns:
        List of default algorithm configurations
    """
    registry = get_algorithm_registry()
    recommended_algorithms = registry.recommend_algorithms(problem_type, "medium")
    
    configs = []
    for algo_name in recommended_algorithms[:5]:  # Limit to top 5
        try:
            config = create_algorithm_config(algo_name.value)
            configs.append(config)
        except Exception as e:
            logger.warning(f"Could not create config for {algo_name}: {e}")
    
    return configs

# Example usage and testing functions

def create_example_pipeline_config(
    file_path: str = "example.csv",
    target_column: str = "target",
    problem_type: str = "classification",
    selected_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create an example pipeline configuration
    
    Args:
        file_path: Path to dataset file
        target_column: Name of target column
        problem_type: 'classification' or 'regression'
        selected_features: Optional list of features to use
    
    Returns:
        Complete pipeline configuration
    """
    problem_type_enum = ProblemTypeEnum(problem_type)
    
    return {
        "file_path": file_path,
        "target_column": target_column,
        "problem_type": problem_type,
        "algorithms": get_default_algorithm_configs(problem_type_enum),
        "preprocessing_config": {
            "selected_features": selected_features,
            "respect_user_selection": True,
            "categorical_strategy": "onehot",
            "scaling_strategy": "standard",
            "missing_strategy": "mean",
            "test_size": 0.2,
            "random_state": 42
        },
        "pipeline_run_id": f"example_{int(time.time())}"
    }

# CLI functions for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument("file_path", help="Path to dataset CSV file")
    parser.add_argument("target_column", help="Name of target column")
    parser.add_argument("--problem-type", choices=["classification", "regression"], 
                       default="classification", help="Type of ML problem")
    parser.add_argument("--features", nargs="*", help="List of features to use")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_example_pipeline_config(
        args.file_path,
        args.target_column,
        args.problem_type,
        args.features
    )
    
    # Run pipeline
    print("Starting ML training pipeline...")
    result = run_ml_training_sync(config)
    
    # Print results
    if result["success"]:
        print(f"✅ Pipeline completed successfully!")
        print(f"⏱️ Total time: {result['total_time_seconds']:.2f}s")
        print(f"🤖 Models trained: {len(result['training_results'])}")
        
        if result.get("best_model_summary"):
            best = result["best_model_summary"]
            print(f"🏆 Best model: {best['algorithm']} (Score: {best['primary_metric']['value']:.4f})")
    else:
        print(f"❌ Pipeline failed: {result['error']}")
        print(f"⏱️ Failed after: {result['total_time_seconds']:.2f}s")