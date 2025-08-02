"""
ML Pipeline Service Layer

This module provides the service layer for ML operations, handling business logic
and coordinating between the API layer and the ML workflows.
"""

import logging
import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database import get_db_session
from app.models.ml_models import (
    MLPipelineRun, MLModel, MLExperiment, MLPreprocessingLog,
    ProblemTypeEnum, PipelineStatusEnum, AlgorithmNameEnum,
    create_pipeline_run, create_ml_model
)
from workflows.ml.algorithm_registry import get_algorithm_registry, get_available_algorithms
from workflows.pipelines.ml_training import run_ml_training_sync, create_example_pipeline_config
from app.config import BASE_DIR

logger = logging.getLogger(__name__)

class MLPipelineService:
    """
    Service layer for ML pipeline operations
    
    Handles all business logic for ML training, model management,
    and pipeline orchestration.
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent training
        self.active_pipelines = {}  # Track running pipelines
    
    async def trigger_ml_training(
        self,
        file_path: str,
        target_variable: str,
        problem_type: ProblemTypeEnum,
        algorithms: List[Dict[str, Any]],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger ML training pipeline asynchronously
        
        Args:
            file_path: Path to dataset file
            target_variable: Name of target column
            problem_type: Type of ML problem
            algorithms: List of algorithm configurations
            preprocessing_config: Preprocessing configuration
            experiment_id: Optional experiment ID for grouping
            user_id: Optional user identifier
        
        Returns:
            Dictionary with pipeline run information
        """
        logger.info(f"Triggering ML training for {file_path}")
        
        try:
            # Generate unique pipeline run ID
            pipeline_run_uuid = str(uuid.uuid4())
            
            # Create pipeline run record in database
            with get_db_session() as db:
                pipeline_run = create_pipeline_run(
                    problem_type=problem_type,
                    target_variable=target_variable,
                    ml_config={
                        "algorithms": algorithms,
                        "preprocessing_config": preprocessing_config or {},
                        "experiment_id": experiment_id,
                        "user_id": user_id
                    },
                    uploaded_file_reference=file_path
                )
                pipeline_run.run_uuid = pipeline_run_uuid
                pipeline_run.status = PipelineStatusEnum.PENDING.value
                
                db.add(pipeline_run)
                db.commit()
                
                pipeline_run_id = pipeline_run.id
            
            # Create training configuration
            training_config = {
                "file_path": file_path,
                "target_column": target_variable,
                "problem_type": problem_type.value,
                "algorithms": algorithms,
                "preprocessing_config": preprocessing_config or {},
                "pipeline_run_id": pipeline_run_uuid
            }
            
            # Start training asynchronously
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.executor,
                self._run_training_pipeline,
                training_config,
                pipeline_run_id
            )
            
            # Track the running pipeline
            self.active_pipelines[pipeline_run_uuid] = {
                "future": future,
                "started_at": datetime.now(timezone.utc),
                "status": PipelineStatusEnum.RUNNING.value
            }
            
            # Update status to running
            with get_db_session() as db:
                pipeline_run = db.get(MLPipelineRun, pipeline_run_id)
                if pipeline_run:
                    pipeline_run.status = PipelineStatusEnum.RUNNING.value
                    pipeline_run.started_at = datetime.now(timezone.utc)
                    db.commit()
            
            logger.info(f"ML training started for pipeline {pipeline_run_uuid}")
            
            return {
                "success": True,
                "pipeline_run_uuid": pipeline_run_uuid,
                "status": PipelineStatusEnum.RUNNING.value,
                "message": "ML training pipeline started successfully"
            }
            
        except Exception as e:
            logger.error(f"Error triggering ML training: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start ML training pipeline"
            }
    
    def _run_training_pipeline(
        self,
        training_config: Dict[str, Any],
        pipeline_run_id: int
    ) -> None:
        """
        Run the training pipeline in a separate thread
        
        Args:
            training_config: Training configuration
            pipeline_run_id: Database ID of pipeline run
        """
        pipeline_run_uuid = training_config["pipeline_run_id"]
        
        try:
            logger.info(f"Starting training execution for {pipeline_run_uuid}")
            
            # Run the ML training pipeline
            result = run_ml_training_sync(training_config)
            
            # Process and save results
            self._save_pipeline_results(result, pipeline_run_id)
            
            # Update pipeline status
            with get_db_session() as db:
                pipeline_run = db.get(MLPipelineRun, pipeline_run_id)
                if pipeline_run:
                    if result.get("success"):
                        pipeline_run.status = PipelineStatusEnum.COMPLETED.value
                        pipeline_run.total_training_time_seconds = result.get("total_time_seconds", 0)
                        
                        # Set best model information
                        best_model = result.get("best_model_summary")
                        if best_model:
                            pipeline_run.best_model_algorithm = best_model.get("algorithm")
                            pipeline_run.best_model_score = best_model.get("primary_metric", {}).get("value")
                        
                        pipeline_run.total_models_trained = len(result.get("training_results", []))
                    else:
                        pipeline_run.status = PipelineStatusEnum.FAILED.value
                        pipeline_run.error_message = result.get("error", "Unknown error")
                    
                    pipeline_run.completed_at = datetime.now(timezone.utc)
                    db.commit()
            
            logger.info(f"Training pipeline {pipeline_run_uuid} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in training pipeline {pipeline_run_uuid}: {str(e)}")
            
            # Update pipeline status to failed
            try:
                with get_db_session() as db:
                    pipeline_run = db.get(MLPipelineRun, pipeline_run_id)
                    if pipeline_run:
                        pipeline_run.status = PipelineStatusEnum.FAILED.value
                        pipeline_run.error_message = str(e)
                        pipeline_run.completed_at = datetime.now(timezone.utc)
                        db.commit()
            except Exception as db_error:
                logger.error(f"Error updating failed pipeline status: {db_error}")
        
        finally:
            # Remove from active pipelines
            if pipeline_run_uuid in self.active_pipelines:
                del self.active_pipelines[pipeline_run_uuid]
    
    def _save_pipeline_results(
        self,
        result: Dict[str, Any],
        pipeline_run_id: int
    ) -> None:
        """
        Save pipeline results to database
        
        Args:
            result: Pipeline execution result
            pipeline_run_id: Database ID of pipeline run
        """
        try:
            with get_db_session() as db:
                # Save individual model results
                training_results = result.get("training_results", [])
                
                for training_result in training_results:
                    if training_result.get("error"):
                        # Skip failed models
                        continue
                    
                    model_analysis = training_result.get("model_analysis_summary")
                    if not model_analysis:
                        continue
                    
                    # Create ML model record
                    ml_model = create_ml_model(
                        pipeline_run_id=pipeline_run_id,
                        pipeline_run_uuid=result["pipeline_run_id"],
                        algorithm_name=AlgorithmNameEnum(training_result["algorithm_name"]),
                        algorithm_display_name=training_result["algorithm_display_name"],
                        hyperparameters=training_result.get("hyperparameters", {})
                    )
                    
                    # Set performance metrics
                    if model_analysis.get("primary_metric"):
                        ml_model.primary_metric_name = model_analysis["primary_metric"]["name"]
                        ml_model.primary_metric_value = model_analysis["primary_metric"]["value"]
                    
                    ml_model.performance_metrics = model_analysis.get("metrics", {})
                    ml_model.training_time_seconds = training_result.get("training_time_seconds", 0)
                    ml_model.model_file_path = training_result.get("model_file_path")
                    ml_model.random_state = training_result.get("random_state")
                    ml_model.n_features = model_analysis.get("n_features", 0)
                    ml_model.feature_names = model_analysis.get("feature_names", [])
                    
                    # Set feature importance if available
                    if model_analysis.get("has_feature_importance"):
                        # Feature importance would be saved here if available in the result
                        pass
                    
                    # Mark as best model if it matches
                    best_model = result.get("best_model_summary")
                    if best_model and best_model.get("model_id") == training_result.get("model_id"):
                        ml_model.is_best_model = True
                        ml_model.rank_in_pipeline = 1
                    
                    db.add(ml_model)
                
                # Save preprocessing log
                preprocessing_summary = result.get("preprocessing_summary")
                if preprocessing_summary:
                    preprocessing_log = MLPreprocessingLog(
                        log_id=str(uuid.uuid4()),
                        pipeline_run_id=pipeline_run_id,
                        preprocessing_config=result.get("preprocessing_config", {}),
                        original_shape=preprocessing_summary.get("original_shape"),
                        final_shape=preprocessing_summary.get("final_shape"),
                        steps_applied=preprocessing_summary.get("preprocessing_steps", []),
                        preprocessing_time_seconds=preprocessing_summary.get("processing_time", 0),
                        warnings=preprocessing_summary.get("warnings", [])
                    )
                    db.add(preprocessing_log)
                
                db.commit()
                logger.info(f"Saved results for pipeline run {pipeline_run_id}")
                
        except Exception as e:
            logger.error(f"Error saving pipeline results: {str(e)}")
            raise
    
    def get_pipeline_status(self, run_uuid: str) -> Dict[str, Any]:
        """
        Get the status of a pipeline run
        
        Args:
            run_uuid: Pipeline run UUID
        
        Returns:
            Pipeline status information
        """
        try:
            with get_db_session() as db:
                pipeline_run = db.exec(
                    select(MLPipelineRun).where(MLPipelineRun.run_uuid == run_uuid)
                ).first()
                
                if not pipeline_run:
                    return {
                        "success": False,
                        "error": "Pipeline run not found"
                    }
                
                # Calculate progress
                progress_percentage = 0
                current_stage = "initializing"
                
                if pipeline_run.status == PipelineStatusEnum.COMPLETED.value:
                    progress_percentage = 100
                    current_stage = "completed"
                elif pipeline_run.status == PipelineStatusEnum.RUNNING.value:
                    # Estimate progress based on elapsed time and typical training time
                    if pipeline_run.started_at:
                        elapsed_seconds = (datetime.now(timezone.utc) - pipeline_run.started_at).total_seconds()
                        # Rough estimate: assume 2-5 minutes for typical training
                        estimated_total = 300  # 5 minutes
                        progress_percentage = min(90, (elapsed_seconds / estimated_total) * 100)
                        current_stage = "training"
                elif pipeline_run.status == PipelineStatusEnum.FAILED.value:
                    current_stage = "failed"
                
                # Estimate completion time
                estimated_completion_time = None
                if pipeline_run.status == PipelineStatusEnum.RUNNING.value and pipeline_run.started_at:
                    elapsed_seconds = (datetime.now(timezone.utc) - pipeline_run.started_at).total_seconds()
                    if progress_percentage > 0:
                        estimated_total_seconds = (elapsed_seconds / progress_percentage) * 100
                        remaining_seconds = estimated_total_seconds - elapsed_seconds
                        estimated_completion_time = (
                            datetime.now(timezone.utc).timestamp() + remaining_seconds
                        )
                
                return {
                    "success": True,
                    "run_uuid": run_uuid,
                    "status": pipeline_run.status,
                    "progress": {
                        "percentage": progress_percentage,
                        "current_stage": current_stage
                    },
                    "created_at": pipeline_run.created_at.isoformat() if pipeline_run.created_at else None,
                    "started_at": pipeline_run.started_at.isoformat() if pipeline_run.started_at else None,
                    "completed_at": pipeline_run.completed_at.isoformat() if pipeline_run.completed_at else None,
                    "estimated_completion_time": estimated_completion_time,
                    "error_message": pipeline_run.error_message,
                    "total_models_trained": pipeline_run.total_models_trained,
                    "best_model_score": pipeline_run.best_model_score
                }
                
        except Exception as e:
            logger.error(f"Error getting pipeline status: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_pipeline_results(self, run_uuid: str) -> Dict[str, Any]:
        """
        Get complete results for a pipeline run
        
        Args:
            run_uuid: Pipeline run UUID
        
        Returns:
            Complete pipeline results
        """
        try:
            with get_db_session() as db:
                # Get pipeline run
                pipeline_run = db.exec(
                    select(MLPipelineRun).where(MLPipelineRun.run_uuid == run_uuid)
                ).first()
                
                if not pipeline_run:
                    return {
                        "success": False,
                        "error": "Pipeline run not found"
                    }
                
                # Get all models for this pipeline run
                models = db.exec(
                    select(MLModel).where(MLModel.pipeline_run_id == pipeline_run.id)
                ).all()
                
                # Get preprocessing log
                preprocessing_log = db.exec(
                    select(MLPreprocessingLog).where(MLPreprocessingLog.pipeline_run_id == pipeline_run.id)
                ).first()
                
                # Format model results
                model_results = []
                best_model = None
                
                for model in models:
                    model_data = {
                        "model_id": model.model_id,
                        "algorithm_name": model.algorithm_name,
                        "algorithm_display_name": model.algorithm_display_name,
                        "primary_metric": {
                            "name": model.primary_metric_name,
                            "value": model.primary_metric_value
                        },
                        "performance_metrics": model.performance_metrics,
                        "training_time_seconds": model.training_time_seconds,
                        "hyperparameters": model.hyperparameters,
                        "feature_importance": model.feature_importance,
                        "n_features": model.n_features,
                        "is_best_model": model.is_best_model,
                        "rank": model.rank_in_pipeline
                    }
                    
                    model_results.append(model_data)
                    
                    if model.is_best_model:
                        best_model = model_data
                
                # Sort models by rank
                model_results.sort(key=lambda x: x.get("rank", 999))
                
                return {
                    "success": True,
                    "run_uuid": run_uuid,
                    "status": pipeline_run.status,
                    "problem_type": pipeline_run.problem_type,
                    "target_variable": pipeline_run.target_variable,
                    "created_at": pipeline_run.created_at.isoformat() if pipeline_run.created_at else None,
                    "completed_at": pipeline_run.completed_at.isoformat() if pipeline_run.completed_at else None,
                    "total_training_time_seconds": pipeline_run.total_training_time_seconds,
                    "total_models_trained": len(model_results),
                    "best_model": best_model,
                    "model_results": model_results,
                    "preprocessing_info": {
                        "original_shape": preprocessing_log.original_shape if preprocessing_log else None,
                        "final_shape": preprocessing_log.final_shape if preprocessing_log else None,
                        "steps_applied": preprocessing_log.steps_applied if preprocessing_log else [],
                        "warnings": preprocessing_log.warnings if preprocessing_log else []
                    } if preprocessing_log else None,
                    "ml_config": pipeline_run.ml_config
                }
                
        except Exception as e:
            logger.error(f"Error getting pipeline results: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_algorithms(self, problem_type: Optional[ProblemTypeEnum] = None) -> Dict[str, Any]:
        """
        Get available algorithms for ML training
        
        Args:
            problem_type: Optional filter by problem type
        
        Returns:
            Available algorithms information
        """
        try:
            algorithms = get_available_algorithms(problem_type)
            
            return {
                "success": True,
                "algorithms": algorithms,
                "total_count": len(algorithms)
            }
            
        except Exception as e:
            logger.error(f"Error getting available algorithms: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_dataset(self, file_path: str, target_column: str) -> Dict[str, Any]:
        """
        Validate a dataset for ML training
        
        Args:
            file_path: Path to dataset file
            target_column: Name of target column
        
        Returns:
            Validation results
        """
        try:
            import pandas as pd
            
            # Load and validate data
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "error": "Dataset file not found"
                }
            
            # Try to load the dataset
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not load dataset: {str(e)}"
                }
            
            # Validate target column
            if target_column not in df.columns:
                return {
                    "success": False,
                    "error": f"Target column '{target_column}' not found in dataset",
                    "available_columns": df.columns.tolist()
                }
            
            # Analyze dataset
            features = [col for col in df.columns if col != target_column]
            
            feature_info = []
            for col in features:
                feature_data = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_percentage": float(df[col].isnull().sum() / len(df) * 100),
                    "unique_count": int(df[col].nunique())
                }
                
                # Add category information for categorical features
                if df[col].dtype in ['object', 'category']:
                    feature_data["is_categorical"] = True
                    feature_data["high_cardinality"] = feature_data["unique_count"] > 20
                else:
                    feature_data["is_categorical"] = False
                    feature_data["high_cardinality"] = False
                
                feature_info.append(feature_data)
            
            # Analyze target variable
            target_info = {
                "name": target_column,
                "type": str(df[target_column].dtype),
                "unique_count": int(df[target_column].nunique()),
                "missing_count": int(df[target_column].isnull().sum()),
                "missing_percentage": float(df[target_column].isnull().sum() / len(df) * 100)
            }
            
            # Suggest problem type
            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < len(df) * 0.05:
                suggested_problem_type = "classification"
            else:
                suggested_problem_type = "regression"
            
            return {
                "success": True,
                "dataset_info": {
                    "shape": df.shape,
                    "total_rows": len(df),
                    "total_features": len(features),
                    "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
                },
                "target_info": target_info,
                "feature_info": feature_info,
                "suggested_problem_type": suggested_problem_type,
                "high_cardinality_features": [f for f in feature_info if f["high_cardinality"]],
                "features_with_missing": [f for f in feature_info if f["missing_count"] > 0],
                "validation_warnings": self._generate_validation_warnings(df, target_column, feature_info)
            }
            
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_validation_warnings(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_info: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate validation warnings for dataset"""
        warnings = []
        
        # Check dataset size
        if len(df) < 100:
            warnings.append("Dataset is very small (< 100 rows). Results may not be reliable.")
        elif len(df) < 1000:
            warnings.append("Dataset is small (< 1000 rows). Consider gathering more data for better results.")
        
        # Check missing values in target
        target_missing = df[target_column].isnull().sum()
        if target_missing > 0:
            warnings.append(f"Target column has {target_missing} missing values. These rows will be dropped.")
        
        # Check high cardinality features
        high_card_features = [f["name"] for f in feature_info if f["high_cardinality"]]
        if high_card_features:
            warnings.append(f"High cardinality features detected: {high_card_features}. Consider feature engineering.")
        
        # Check features with many missing values
        high_missing_features = [f["name"] for f in feature_info if f["missing_percentage"] > 50]
        if high_missing_features:
            warnings.append(f"Features with >50% missing values: {high_missing_features}. Consider dropping these.")
        
        return warnings

# Global service instance
_ml_pipeline_service = None

def get_ml_pipeline_service() -> MLPipelineService:
    """Get the global ML pipeline service instance"""
    global _ml_pipeline_service
    if _ml_pipeline_service is None:
        _ml_pipeline_service = MLPipelineService()
    return _ml_pipeline_service