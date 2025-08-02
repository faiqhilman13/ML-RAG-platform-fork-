"""
Database Configuration and Management

This module provides database connectivity and session management
for ML features while maintaining compatibility with existing RAG system.
"""

import os
import logging
from pathlib import Path
from typing import Generator, Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager

from app.config import BASE_DIR
from app.models.ml_models import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_DIR = BASE_DIR / "data" / "ml_database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite for simplicity and to avoid external dependencies
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/ml_features.db"

# Create engine with appropriate settings for SQLite
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    connect_args={
        "check_same_thread": False,  # Allow multiple threads for SQLite
        "timeout": 30,  # 30 second timeout for database operations
    },
    echo=False,  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """
    Create all ML tables in the database.
    
    This function is safe to call multiple times - it will only create
    tables that don't already exist.
    """
    try:
        logger.info("Creating ML database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("ML database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        return False

def get_database_info():
    """Get information about the database and tables."""
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        return {
            "database_url": DATABASE_URL,
            "database_path": str(DATABASE_DIR / "ml_features.db"),
            "tables": list(metadata.tables.keys()),
            "database_exists": (DATABASE_DIR / "ml_features.db").exists()
        }
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e)}

def drop_all_tables():
    """
    Drop all ML tables. USE WITH CAUTION!
    
    This is primarily for development and testing purposes.
    """
    try:
        logger.warning("!!! DROPPING ALL ML TABLES !!!")
        Base.metadata.drop_all(bind=engine)
        logger.info("All ML tables dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        return False

# Database session dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session for FastAPI endpoints.
    
    Usage in FastAPI routes:
        @app.get("/endpoint")
        def my_endpoint(db: Session = Depends(get_db)):
            # Use db session here
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_session() as db:
            # Use db session here
            db.add(my_object)
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

class DatabaseManager:
    """
    Database manager class for ML operations.
    
    Provides high-level database operations and management functions.
    """
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def initialize(self) -> bool:
        """Initialize the database (create tables if needed)."""
        return create_tables()
    
    def health_check(self) -> dict:
        """Perform a health check on the database."""
        try:
            with get_db_session() as db:
                # Simple query to test connectivity
                db.execute("SELECT 1").fetchone()
            
            info = get_database_info()
            return {
                "status": "healthy",
                "database_info": info,
                "connection": "successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection": "failed"
            }
    
    def get_table_counts(self) -> dict:
        """Get row counts for all ML tables."""
        try:
            with get_db_session() as db:
                from app.models.ml_models import MLPipelineRun, MLModel, MLExperiment, MLPreprocessingLog
                
                return {
                    "ml_pipeline_run": db.query(MLPipelineRun).count(),
                    "ml_model": db.query(MLModel).count(),
                    "ml_experiment": db.query(MLExperiment).count(),
                    "ml_preprocessing_log": db.query(MLPreprocessingLog).count()
                }
        except Exception as e:
            logger.error(f"Error getting table counts: {e}")
            return {"error": str(e)}
    
    def cleanup_old_runs(self, days_old: int = 30) -> dict:
        """
        Clean up old pipeline runs and associated data.
        
        Args:
            days_old: Remove runs older than this many days
        """
        try:
            from datetime import datetime, timedelta, timezone
            from app.models.ml_models import MLPipelineRun, MLModel, MLPreprocessingLog
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            with get_db_session() as db:
                # Find old pipeline runs
                old_runs = db.query(MLPipelineRun).filter(
                    MLPipelineRun.created_at < cutoff_date
                ).all()
                
                run_ids = [run.id for run in old_runs]
                
                if not run_ids:
                    return {"message": "No old runs to clean up", "removed": 0}
                
                # Remove associated models
                model_count = db.query(MLModel).filter(
                    MLModel.pipeline_run_id.in_(run_ids)
                ).count()
                
                db.query(MLModel).filter(
                    MLModel.pipeline_run_id.in_(run_ids)
                ).delete(synchronize_session=False)
                
                # Remove preprocessing logs
                log_count = db.query(MLPreprocessingLog).filter(
                    MLPreprocessingLog.pipeline_run_id.in_(run_ids)
                ).count()
                
                db.query(MLPreprocessingLog).filter(
                    MLPreprocessingLog.pipeline_run_id.in_(run_ids)
                ).delete(synchronize_session=False)
                
                # Remove pipeline runs
                run_count = len(run_ids)
                db.query(MLPipelineRun).filter(
                    MLPipelineRun.id.in_(run_ids)
                ).delete(synchronize_session=False)
                
                db.commit()
                
                return {
                    "message": f"Cleaned up data older than {days_old} days",
                    "removed": {
                        "pipeline_runs": run_count,
                        "models": model_count,
                        "preprocessing_logs": log_count
                    }
                }
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}

# Global database manager instance
db_manager = DatabaseManager()

# Initialize database on import
def init_database():
    """Initialize the database when the module is imported."""
    try:
        success = db_manager.initialize()
        if success:
            logger.info("ML database initialized successfully")
        else:
            logger.error("Failed to initialize ML database")
        return success
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

# Auto-initialize when module is imported
if __name__ != "__main__":
    init_database()

# CLI functions for database management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Database Management")
    parser.add_argument("command", choices=["init", "info", "health", "counts", "cleanup", "drop"])
    parser.add_argument("--days", type=int, default=30, help="Days for cleanup command")
    
    args = parser.parse_args()
    
    if args.command == "init":
        success = create_tables()
        print(f"Database initialization: {'SUCCESS' if success else 'FAILED'}")
    
    elif args.command == "info":
        info = get_database_info()
        print("Database Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    elif args.command == "health":
        health = db_manager.health_check()
        print("Database Health Check:")
        for key, value in health.items():
            print(f"  {key}: {value}")
    
    elif args.command == "counts":
        counts = db_manager.get_table_counts()
        print("Table Row Counts:")
        for table, count in counts.items():
            print(f"  {table}: {count}")
    
    elif args.command == "cleanup":
        result = db_manager.cleanup_old_runs(args.days)
        print("Cleanup Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    elif args.command == "drop":
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == "yes":
            success = drop_all_tables()
            print(f"Drop tables: {'SUCCESS' if success else 'FAILED'}")
        else:
            print("Operation cancelled")