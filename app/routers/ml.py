"""
ML Router - API Endpoints for Machine Learning Features

This module provides REST API endpoints for ML training pipeline operations,
seamlessly integrated with the existing RAG system authentication and patterns.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Query, File, UploadFile
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import os
import shutil
import uuid
from pathlib import Path

from app.auth import require_auth
from app.services.ml_pipeline_service import get_ml_pipeline_service
from app.models.ml_models import ProblemTypeEnum
from app.config import DATASETS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Pipeline"])

# Pydantic Models for API Requests and Responses

class AlgorithmConfig(BaseModel):
    """Configuration for a single algorithm"""
    name: str = Field(..., description="Algorithm name (e.g., 'random_forest_classifier')")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters")

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing"""
    selected_features: Optional[List[str]] = Field(None, description="List of features to use")
    respect_user_selection: bool = Field(True, description="Respect user feature selection")
    categorical_strategy: str = Field("onehot", description="Strategy for categorical encoding")
    scaling_strategy: str = Field("standard", description="Strategy for feature scaling")
    missing_strategy: str = Field("mean", description="Strategy for handling missing values")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Proportion of data for testing")
    random_state: int = Field(42, description="Random state for reproducibility")

class MLPipelineCreateRequest(BaseModel):
    """Request model for creating ML training pipeline"""
    file_path: str = Field(..., description="Path to dataset file")
    target_variable: str = Field(..., description="Name of target column")
    problem_type: ProblemTypeEnum = Field(..., description="Type of ML problem")
    algorithms: List[AlgorithmConfig] = Field(..., min_length=1, max_length=10, description="Algorithms to train")
    preprocessing_config: Optional[PreprocessingConfig] = Field(None, description="Preprocessing configuration")
    experiment_id: Optional[str] = Field(None, description="Optional experiment ID for grouping")
    
    @field_validator('target_variable')
    @classmethod
    def validate_target_variable(cls, v):
        if not v or not v.strip():
            raise ValueError('Target variable cannot be empty')
        # Basic validation for column name safety
        if not v.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise ValueError('Target variable contains invalid characters')
        return v.strip()
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        if not v or not v.strip():
            raise ValueError('File path cannot be empty')
        # Basic path validation
        if '..' in v or v.startswith('/') or ':' in v:
            raise ValueError('Invalid file path')
        return v.strip()
    
    @field_validator('algorithms')
    @classmethod
    def validate_algorithms(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 algorithms allowed per pipeline')
        return v

class MLPipelineCreateResponse(BaseModel):
    """Response model for ML pipeline creation"""
    success: bool
    message: str
    run_uuid: Optional[str] = None
    status: Optional[str] = None

class MLPipelineStatusResponse(BaseModel):
    """Response model for ML pipeline status"""
    success: bool
    message: str = ""
    run_uuid: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_completion_time: Optional[float] = None
    error_message: Optional[str] = None
    total_models_trained: Optional[int] = None
    best_model_score: Optional[float] = None

class MLPipelineResultsResponse(BaseModel):
    """Response model for ML pipeline results"""
    success: bool
    message: str = ""
    run_uuid: Optional[str] = None
    status: Optional[str] = None
    problem_type: Optional[str] = None
    target_variable: Optional[str] = None
    total_training_time_seconds: Optional[float] = None
    total_models_trained: Optional[int] = None
    best_model: Optional[Dict[str, Any]] = None
    model_results: Optional[List[Dict[str, Any]]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None

class DatasetValidationResponse(BaseModel):
    """Response model for dataset validation"""
    success: bool
    message: str = ""
    dataset_info: Optional[Dict[str, Any]] = None
    target_info: Optional[Dict[str, Any]] = None
    feature_info: Optional[List[Dict[str, Any]]] = None
    suggested_problem_type: Optional[str] = None
    validation_warnings: Optional[List[str]] = None

class AlgorithmsResponse(BaseModel):
    """Response model for available algorithms"""
    success: bool
    message: str = ""
    algorithms: Optional[Dict[str, Any]] = None
    total_count: Optional[int] = None

# API Endpoints

@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: str = Depends(require_auth)
):
    """
    Upload a CSV dataset for ML training
    
    This endpoint accepts CSV files and saves them to the datasets directory
    for use in ML training pipelines.
    """
    try:
        logger.info(f"Uploading dataset file for user {current_user}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension (only CSV for ML datasets)
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.csv']:
            raise HTTPException(status_code=400, detail="Only CSV files are allowed for ML datasets")
        
        # Validate filename characters (prevent path traversal)
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.() ")
        if not all(c in safe_chars for c in file.filename):
            raise HTTPException(status_code=400, detail="Invalid characters in filename")
        
        # Generate unique filename to avoid collisions
        dataset_id = str(uuid.uuid4())
        safe_filename = f"{dataset_id}_{file.filename}"
        save_path = DATASETS_DIR / safe_filename
        
        # Save the uploaded file
        logger.info(f"Saving dataset to: {save_path}")
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Dataset saved successfully: {save_path}")
        
        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset_id,
            "filename": file.filename,
            "file_path": str(save_path.relative_to(DATASETS_DIR.parent.parent)),  # Return path relative to project root (BASE_DIR)
            "file_size": save_path.stat().st_size if save_path.exists() else 0
        }
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while uploading dataset"
        )

@router.post("/train", response_model=MLPipelineCreateResponse)
async def create_ml_training_pipeline(
    request: MLPipelineCreateRequest,
    current_user: str = Depends(require_auth)
):
    """
    Create and start ML training pipeline
    
    This endpoint creates a new ML training pipeline with the specified configuration
    and starts the training process asynchronously.
    """
    try:
        logger.info(f"Creating ML pipeline for user {current_user}")
        
        ml_service = get_ml_pipeline_service()
        
        # Convert preprocessing config to dict
        preprocessing_dict = None
        if request.preprocessing_config:
            preprocessing_dict = request.preprocessing_config.model_dump()
        
        # Convert algorithm configs to list of dicts
        algorithms_list = [algo.model_dump() for algo in request.algorithms]
        
        # Trigger ML training
        result = await ml_service.trigger_ml_training(
            file_path=request.file_path,
            target_variable=request.target_variable,
            problem_type=request.problem_type,
            algorithms=algorithms_list,
            preprocessing_config=preprocessing_dict,
            experiment_id=request.experiment_id,
            user_id=current_user
        )
        
        if result["success"]:
            return MLPipelineCreateResponse(
                success=True,
                message=result["message"],
                run_uuid=result["pipeline_run_uuid"],
                status=result["status"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to create ML pipeline")
            )
            
    except ValueError as e:
        logger.warning(f"Validation error in ML pipeline creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating ML pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating ML training pipeline"
        )

@router.get("/status/{run_uuid}", response_model=MLPipelineStatusResponse)
async def get_ml_pipeline_status(
    run_uuid: str,
    current_user: str = Depends(require_auth)
):
    """
    Get ML training pipeline status and progress
    
    Returns the current status and progress information for a running
    or completed ML training pipeline.
    """
    try:
        logger.info(f"Getting pipeline status for {run_uuid}")
        
        ml_service = get_ml_pipeline_service()
        result = ml_service.get_pipeline_status(run_uuid)
        
        if result["success"]:
            return MLPipelineStatusResponse(
                success=True,
                message="Pipeline status retrieved successfully",
                **{k: v for k, v in result.items() if k != "success"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Pipeline run not found")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving pipeline status"
        )

@router.get("/results/{run_uuid}", response_model=MLPipelineResultsResponse)
async def get_ml_pipeline_results(
    run_uuid: str,
    current_user: str = Depends(require_auth)
):
    """
    Get complete ML training pipeline results
    
    Returns comprehensive results including model performance, comparisons,
    and training details for a completed ML training pipeline.
    """
    try:
        logger.info(f"Getting pipeline results for {run_uuid}")
        
        ml_service = get_ml_pipeline_service()
        result = ml_service.get_pipeline_results(run_uuid)
        
        if result["success"]:
            return MLPipelineResultsResponse(
                success=True,
                message="Pipeline results retrieved successfully",
                **{k: v for k, v in result.items() if k not in ["success", "message"]}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Pipeline run not found")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving pipeline results"
        )

@router.get("/algorithms", response_model=AlgorithmsResponse)
async def get_available_algorithms(
    problem_type: Optional[ProblemTypeEnum] = None,
    current_user: str = Depends(require_auth)
):
    """
    Get available ML algorithms and their configurations
    
    Returns information about all available algorithms, optionally filtered
    by problem type, including their hyperparameters and descriptions.
    """
    try:
        logger.info(f"Getting available algorithms for problem type: {problem_type}")
        
        ml_service = get_ml_pipeline_service()
        result = ml_service.get_available_algorithms(problem_type)
        
        if result["success"]:
            return AlgorithmsResponse(
                success=True,
                message="Available algorithms retrieved successfully",
                algorithms=result["algorithms"],
                total_count=result["total_count"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to retrieve algorithms")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available algorithms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving available algorithms"
        )

@router.post("/validate-dataset", response_model=DatasetValidationResponse)
async def validate_dataset(
    file_path: str = Query(..., description="Path to dataset file"),
    target_column: str = Query(..., description="Name of target column"),
    current_user: str = Depends(require_auth)
):
    """
    Validate dataset for ML training
    
    Analyzes a dataset and target column to provide validation information,
    feature analysis, and recommendations for ML training configuration.
    """
    try:
        logger.info(f"Validating dataset: {file_path}, target: {target_column}")
        
        # Basic input validation
        if not file_path or not file_path.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File path cannot be empty"
            )
        
        if not target_column or not target_column.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Target column cannot be empty"
            )
        
        ml_service = get_ml_pipeline_service()
        result = ml_service.validate_dataset(file_path.strip(), target_column.strip())
        
        if result["success"]:
            return DatasetValidationResponse(
                success=True,
                message="Dataset validation completed successfully",
                **{k: v for k, v in result.items() if k not in ["success", "message"]}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Dataset validation failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while validating dataset"
        )

@router.get("/datasets", response_model=Dict[str, Any])
async def list_available_datasets(
    current_user: str = Depends(require_auth)
):
    """
    List available datasets for ML training
    
    Returns a list of uploaded CSV datasets that can be used for ML training.
    """
    try:
        logger.info(f"Listing available datasets for user {current_user}")
        
        # List all CSV files in the datasets directory
        datasets = []
        if DATASETS_DIR.exists():
            for file_path in DATASETS_DIR.glob("*.csv"):
                try:
                    # Parse UUID and original filename from the stored filename
                    # Format: {uuid}_{original_filename}
                    filename = file_path.name
                    if '_' in filename:
                        uuid_part, original_filename = filename.split('_', 1)
                        
                        # Get file info
                        file_stat = file_path.stat()
                        
                        datasets.append({
                            "id": uuid_part,
                            "original_filename": original_filename,
                            "stored_filename": filename,
                            "file_path": str(file_path.relative_to(DATASETS_DIR.parent)),
                            "size_bytes": file_stat.st_size,
                            "uploaded_at": file_stat.st_mtime,
                            "display_name": original_filename.replace('.csv', '').replace('_', ' ').title()
                        })
                except Exception as file_error:
                    logger.warning(f"Error processing dataset file {file_path}: {file_error}")
                    continue
        
        # Sort by upload time (newest first)
        datasets.sort(key=lambda x: x["uploaded_at"], reverse=True)
        
        return {
            "success": True,
            "message": f"Found {len(datasets)} available datasets",
            "datasets": datasets,
            "total": len(datasets)
        }
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing datasets"
        )

@router.get("/pipelines", response_model=Dict[str, Any])
async def list_ml_pipelines(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of pipelines to return"),
    offset: int = Query(0, ge=0, description="Number of pipelines to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by pipeline status"),
    current_user: str = Depends(require_auth)
):
    """
    List ML training pipelines for the current user
    
    Returns a paginated list of ML training pipelines with basic information
    and status for the authenticated user.
    """
    try:
        logger.info(f"Listing ML pipelines for user {current_user}")
        
        # This is a placeholder for pipeline listing functionality
        # In a full implementation, you would query the database for user's pipelines
        
        return {
            "success": True,
            "message": "Pipeline listing not yet implemented",
            "total": 0,
            "pipelines": [],
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing ML pipelines: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing pipelines"
        )

@router.delete("/pipelines/{run_uuid}")
async def delete_ml_pipeline(
    run_uuid: str,
    current_user: str = Depends(require_auth)
):
    """
    Delete an ML training pipeline and its results
    
    Removes a pipeline run and all associated models and results from the database.
    This action cannot be undone.
    """
    try:
        logger.info(f"Deleting ML pipeline {run_uuid} for user {current_user}")
        
        # This is a placeholder for pipeline deletion functionality
        # In a full implementation, you would:
        # 1. Verify the pipeline belongs to the user
        # 2. Remove associated files and models
        # 3. Delete database records
        
        return {
            "success": True,
            "message": "Pipeline deletion not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error deleting ML pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while deleting pipeline"
        )

# Health check endpoint for ML services
@router.get("/health")
async def ml_health_check():
    """
    Health check for ML services
    
    Returns the health status of ML-related services and dependencies.
    """
    try:
        from app.database import db_manager
        
        # Check database health
        db_health = db_manager.health_check()
        
        # Check if required directories exist
        from app.config import BASE_DIR
        models_dir = BASE_DIR / "data" / "trained_models"
        models_dir_exists = models_dir.exists()
        
        health_status = {
            "ml_services": "healthy",
            "database": db_health.get("status", "unknown"),
            "models_directory": "exists" if models_dir_exists else "missing",
            "algorithm_registry": "loaded",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Determine overall health
        all_healthy = (
            db_health.get("status") == "healthy" and
            models_dir_exists
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Error in ML health check: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Import datetime for health check
from datetime import datetime, timezone