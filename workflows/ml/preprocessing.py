"""
Data Preprocessing Pipeline for ML Training

This module provides comprehensive data preprocessing capabilities with
manual feature selection support and intelligent handling of various data types.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

from app.models.ml_models import ProblemTypeEnum

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline"""
    
    # Missing value handling
    missing_strategy: str = "mean"  # mean, median, mode, drop, constant
    missing_threshold: float = 0.8  # Drop columns with >80% missing values
    fill_value: Optional[Union[str, int, float]] = None  # For constant strategy
    
    # Categorical encoding
    categorical_strategy: str = "onehot"  # onehot, label, target
    max_categories: int = 20  # Max categories for one-hot encoding
    handle_unknown: str = "ignore"  # ignore, error
    
    # Feature selection (CRITICAL ENHANCEMENT)
    selected_features: Optional[List[str]] = None
    respect_user_selection: bool = True
    max_categories_override: Optional[int] = None
    
    # Feature scaling
    scaling_strategy: str = "standard"  # standard, minmax, none
    
    # Data splitting
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True  # For classification problems
    
    # Advanced options
    remove_duplicates: bool = True
    remove_constant_features: bool = True
    correlation_threshold: float = 0.95  # Remove highly correlated features

@dataclass
class PreprocessingResult:
    """Result of data preprocessing pipeline"""
    
    # Processed data
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    
    # Preprocessing information
    preprocessing_steps: List[str] = field(default_factory=list)
    transformations: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    # Data quality information
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    dropped_features: List[str] = field(default_factory=list)
    encoded_features: Dict[str, List[str]] = field(default_factory=dict)
    
    # Warnings and issues
    preprocessing_warnings: List[str] = field(default_factory=list)
    high_cardinality_features: List[Dict[str, Any]] = field(default_factory=list)
    missing_value_summary: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    preprocessing_time_seconds: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of preprocessing results"""
        return {
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "n_features_original": self.original_shape[1] if len(self.original_shape) > 1 else 0,
            "n_features_final": len(self.feature_names),
            "n_samples_train": len(self.X_train),
            "n_samples_test": len(self.X_test),
            "preprocessing_steps": self.preprocessing_steps,
            "dropped_features": self.dropped_features,
            "warnings_count": len(self.preprocessing_warnings),
            "high_cardinality_count": len(self.high_cardinality_features),
            "processing_time": self.preprocessing_time_seconds
        }

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    
    Handles missing values, categorical encoding, feature scaling,
    and manual feature selection with intelligent warnings.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessing_steps = []
        self.transformations = {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names_ = None
        self.preprocessing_steps_ = []
        
    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: ProblemTypeEnum,
        pipeline_run_id: Optional[str] = None
    ) -> PreprocessingResult:
        """
        Main preprocessing pipeline with enhanced user feedback
        
        Args:
            df: Input dataframe
            target_col: Name of the target column
            problem_type: Type of ML problem
            pipeline_run_id: Optional pipeline run ID for logging
            
        Returns:
            PreprocessingResult with processed data and metadata
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting preprocessing for pipeline {pipeline_run_id}")
        logger.info(f"Original data shape: {df.shape}")
        
        original_shape = df.shape
        preprocessing_warnings = []
        high_cardinality_features = []
        
        try:
            # Step 0: Early data validation
            self._validate_data_for_ml(df, target_col, problem_type, preprocessing_warnings)
            
            # Step 1: Apply manual feature selection FIRST
            df_selected = self._apply_manual_feature_selection(
                df, target_col, preprocessing_warnings, high_cardinality_features
            )
            
            # Step 2: Basic data cleaning
            df_cleaned = self._basic_data_cleaning(df_selected, target_col, preprocessing_warnings)
            
            # Step 3: Handle missing values
            df_imputed = self._handle_missing_values(df_cleaned, target_col, preprocessing_warnings)
            
            # Step 4: Encode categorical variables
            df_encoded = self._encode_categorical_variables(
                df_imputed, target_col, problem_type, preprocessing_warnings
            )
            
            # Step 5: Split data
            X_train, X_test, y_train, y_test = self._split_data(
                df_encoded, target_col, problem_type
            )
            
            # Step 6: Scale features
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            
            # Step 7: Final feature engineering
            X_train_final, X_test_final = self._final_feature_engineering(
                X_train_scaled, X_test_scaled, preprocessing_warnings
            )
            
            # Create result
            preprocessing_time = time.time() - start_time
            
            result = PreprocessingResult(
                X_train=X_train_final,
                X_test=X_test_final,
                y_train=y_train,
                y_test=y_test,
                preprocessing_steps=self.preprocessing_steps.copy(),
                transformations=self.transformations.copy(),
                feature_names=X_train_final.columns.tolist(),
                original_shape=original_shape,
                final_shape=X_train_final.shape,
                preprocessing_warnings=preprocessing_warnings,
                high_cardinality_features=high_cardinality_features,
                missing_value_summary=self._calculate_missing_summary(df),
                preprocessing_time_seconds=preprocessing_time
            )
            
            logger.info(f"Preprocessing completed in {preprocessing_time:.2f}s")
            logger.info(f"Final data shape: {X_train_final.shape[1]} features, {len(X_train_final)} train samples")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
    
    def _validate_data_for_ml(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: ProblemTypeEnum,
        warnings: List[str]
    ) -> None:
        """
        Early validation to catch common ML data issues before processing
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            problem_type: Type of ML problem
            warnings: List to append warnings to
        """
        logger.info("🔍 Performing early data validation...")
        
        # Check if target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Analyze target variable
        target_series = df[target_col]
        
        # Check for identifier columns that should be excluded
        potential_id_columns = []
        feature_columns = [col for col in df.columns if col != target_col]
        
        for col in feature_columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio >= 0.95:  # 95% or more unique values
                potential_id_columns.append({
                    'column': col, 
                    'unique_count': df[col].nunique(),
                    'unique_ratio': unique_ratio
                })
        
        if potential_id_columns:
            logger.warning("🆔 Potential identifier columns detected:")
            for col_info in potential_id_columns:
                logger.warning(f"   - {col_info['column']}: {col_info['unique_count']} unique values ({col_info['unique_ratio']:.1%})")
            
            id_column_names = [col['column'] for col in potential_id_columns]
            warning_msg = f"Potential ID columns found: {id_column_names}. Consider excluding these from features."
            warnings.append(warning_msg)
            logger.warning("💡 These columns may cause overfitting and should typically be excluded from ML models")
        
        # Classification-specific validation
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            class_counts = target_series.value_counts()
            min_class_size = class_counts.min()
            total_classes = len(class_counts)
            
            logger.info(f"📊 Classification target analysis:")
            logger.info(f"   - Total classes: {total_classes}")
            logger.info(f"   - Smallest class: {min_class_size} samples")
            logger.info(f"   - Class distribution:")
            for class_val, count in class_counts.head(10).items():
                percentage = count / len(target_series) * 100
                logger.info(f"     {class_val}: {count} samples ({percentage:.1f}%)")
            
            if total_classes > 100:
                warning_msg = f"Very high number of classes ({total_classes}). Consider if this should be a regression problem or if classes should be grouped."
                warnings.append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
            
            # Critical class imbalance check
            if min_class_size == 1:
                error_msg = f"CRITICAL: Target column '{target_col}' has classes with only 1 sample. This will cause train_test_split to fail with stratification."
                logger.error(f"❌ {error_msg}")
                
                # Check if target column is actually an ID column
                if target_series.nunique() == len(target_series):
                    raise ValueError(f"Target column '{target_col}' appears to be an identifier (all unique values). Please choose a different target column for prediction.")
                
                # Provide suggestions
                logger.error("💡 Possible solutions:")
                logger.error("   1. Choose a different target column")
                logger.error("   2. Remove or combine rare classes")
                logger.error("   3. Use regression instead of classification")
                logger.error("   4. Collect more data for rare classes")
                
                raise ValueError(error_msg)
            
            elif min_class_size < 5:
                warning_msg = f"Small class sizes detected (minimum: {min_class_size}). This may cause issues with cross-validation and model evaluation."
                warnings.append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
        
        # Regression-specific validation
        elif problem_type == ProblemTypeEnum.REGRESSION:
            if not pd.api.types.is_numeric_dtype(target_series):
                if target_series.nunique() < 20:
                    warning_msg = f"Target column appears categorical with {target_series.nunique()} unique values. Consider using classification instead."
                    warnings.append(warning_msg)
                    logger.warning(f"⚠️ {warning_msg}")
                else:
                    # Try to convert to numeric
                    try:
                        target_numeric = pd.to_numeric(target_series, errors='coerce')
                        if target_numeric.isnull().sum() > len(target_series) * 0.1:
                            raise ValueError(f"Target column '{target_col}' cannot be converted to numeric for regression. Consider classification or data cleaning.")
                    except:
                        raise ValueError(f"Target column '{target_col}' is not numeric and cannot be used for regression.")
        
        # Check for missing target values
        missing_target = target_series.isnull().sum()
        if missing_target > 0:
            warning_msg = f"Target column has {missing_target} missing values. These rows will be dropped."
            warnings.append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
            
            if missing_target > len(df) * 0.1:
                logger.warning(f"⚠️ High proportion of missing target values ({missing_target/len(df):.1%}). Consider data quality review.")
        
        # Dataset size validation
        if len(df) < 50:
            warning_msg = "Very small dataset (< 50 rows). Results may not be reliable."
            warnings.append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
        elif len(df) < 200:
            warning_msg = "Small dataset (< 200 rows). Consider gathering more data for better model performance."
            warnings.append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
        
        logger.info("✅ Early validation completed")
    
    def _apply_manual_feature_selection(
        self,
        df: pd.DataFrame,
        target_col: str,
        warnings: List[str],
        high_cardinality_features: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply manual feature selection with comprehensive validation"""
        
        if not self.config.selected_features:
            logger.info("No manual feature selection specified, using all features")
            return df
        
        logger.info(f"Applying manual feature selection: {len(self.config.selected_features)} features selected")
        self.preprocessing_steps.append("manual_feature_selection")
        
        # Comprehensive feature validation
        available_features = [col for col in df.columns if col != target_col]
        valid_features = [col for col in self.config.selected_features if col in available_features]
        invalid_features = [col for col in self.config.selected_features if col not in available_features]
        
        if invalid_features:
            warning_msg = f"Invalid features ignored: {invalid_features}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        if not valid_features:
            error_msg = "No valid features found in selection"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # High cardinality analysis with user choice respect
        for col in valid_features:
            if df[col].dtype in ['object', 'category']:
                unique_count = df[col].nunique()
                max_cats = self.config.max_categories_override or self.config.max_categories
                
                if unique_count > max_cats:
                    high_cardinality_info = {
                        "feature": col,
                        "unique_count": unique_count,
                        "threshold": max_cats,
                        "action": "label_encoding" if self.config.respect_user_selection else "dropped",
                        "user_selected": True
                    }
                    high_cardinality_features.append(high_cardinality_info)
                    
                    if self.config.respect_user_selection:
                        warning_msg = f"⚠️ HIGH CARDINALITY: '{col}' has {unique_count} unique values (using label encoding)"
                        warnings.append(warning_msg)
                        logger.warning(warning_msg)
        
        # Return selected features plus target
        selected_df = df[valid_features + [target_col]].copy()
        
        dropped_count = len(available_features) - len(valid_features)
        if dropped_count > 0:
            info_msg = f"Manual selection: kept {len(valid_features)} features, dropped {dropped_count} features"
            logger.info(info_msg)
        
        return selected_df
    
    def _basic_data_cleaning(
        self,
        df: pd.DataFrame,
        target_col: str,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Basic data cleaning operations"""
        
        logger.info("Performing basic data cleaning")
        df_clean = df.copy()
        
        # Remove duplicates
        if self.config.remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = initial_rows - len(df_clean)
            
            if duplicates_removed > 0:
                self.preprocessing_steps.append("remove_duplicates")
                warning_msg = f"Removed {duplicates_removed} duplicate rows"
                warnings.append(warning_msg)
                logger.info(warning_msg)
        
        # Check for and handle infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        for col in numeric_cols:
            inf_count = np.isinf(df_clean[col]).sum()
            if inf_count > 0:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                warning_msg = f"Replaced {inf_count} infinite values with NaN in '{col}'"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        return df_clean
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        target_col: str,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Handle missing values with various strategies"""
        
        logger.info(f"Handling missing values with strategy: {self.config.missing_strategy}")
        df_imputed = df.copy()
        
        # Check missing values
        missing_info = df_imputed.isnull().sum()
        missing_features = missing_info[missing_info > 0]
        
        if len(missing_features) == 0:
            logger.info("No missing values found")
            return df_imputed
        
        self.preprocessing_steps.append("handle_missing_values")
        
        # Drop columns with too many missing values
        features_to_drop = []
        for col, missing_count in missing_features.items():
            if col == target_col:
                continue
                
            missing_ratio = missing_count / len(df_imputed)
            if missing_ratio > self.config.missing_threshold:
                features_to_drop.append(col)
                warning_msg = f"Dropped '{col}': {missing_ratio:.1%} missing values"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        if features_to_drop:
            df_imputed = df_imputed.drop(columns=features_to_drop)
        
        # Handle remaining missing values
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        
        # Remove target column from imputation
        numeric_cols = [col for col in numeric_cols if col != target_col]
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            if self.config.missing_strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=self.config.missing_strategy)
                df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                self.imputers['numeric'] = imputer
            elif self.config.missing_strategy == 'constant':
                fill_val = self.config.fill_value if self.config.fill_value is not None else 0
                df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(fill_val)
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            if self.config.missing_strategy == 'mode':
                for col in categorical_cols:
                    mode_val = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else 'Unknown'
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
            elif self.config.missing_strategy == 'constant':
                fill_val = self.config.fill_value if self.config.fill_value is not None else 'Unknown'
                df_imputed[categorical_cols] = df_imputed[categorical_cols].fillna(fill_val)
        
        # Store transformation info
        self.transformations['missing_handling'] = {
            'strategy': self.config.missing_strategy,
            'dropped_columns': features_to_drop,
            'imputed_columns': list(numeric_cols) + list(categorical_cols)
        }
        
        logger.info(f"Missing value handling completed. Dropped {len(features_to_drop)} columns")
        return df_imputed
    
    def _encode_categorical_variables(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: ProblemTypeEnum,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Encode categorical variables with intelligent strategy selection"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        if len(categorical_cols) == 0:
            logger.info("No categorical features to encode")
            return df
        
        logger.info(f"Encoding {len(categorical_cols)} categorical features")
        self.preprocessing_steps.append("encode_categorical")
        
        df_encoded = df.copy()
        encoded_features = {}
        
        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()
            max_cats = self.config.max_categories_override or self.config.max_categories
            
            # Strategy selection based on cardinality and user preferences
            if unique_count <= max_cats and self.config.categorical_strategy == "onehot":
                # One-hot encoding for low cardinality
                encoder = OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown=self.config.handle_unknown
                )
                
                encoded_values = encoder.fit_transform(df_encoded[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                
                # Create encoded dataframe
                encoded_df = pd.DataFrame(
                    encoded_values,
                    columns=feature_names,
                    index=df_encoded.index
                )
                
                # Replace original column
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                
                self.encoders[col] = encoder
                encoded_features[col] = feature_names
                
                logger.info(f"One-hot encoded '{col}': {unique_count} categories → {len(feature_names)} features")
                
            else:
                # Label encoding for high cardinality or user preference
                encoder = LabelEncoder()
                df_encoded[f"{col}_encoded"] = encoder.fit_transform(df_encoded[col])
                df_encoded = df_encoded.drop(columns=[col])
                
                self.encoders[col] = encoder
                encoded_features[col] = [f"{col}_encoded"]
                
                logger.info(f"Label encoded '{col}': {unique_count} categories → 1 feature")
                
                if unique_count > max_cats:
                    warning_msg = f"High cardinality feature '{col}' ({unique_count} categories) encoded with label encoding"
                    warnings.append(warning_msg)
        
        # Handle target encoding for classification problems if needed
        if problem_type == ProblemTypeEnum.CLASSIFICATION and target_col in df_encoded.columns:
            if df_encoded[target_col].dtype in ['object', 'category']:
                target_encoder = LabelEncoder()
                df_encoded[target_col] = target_encoder.fit_transform(df_encoded[target_col])
                self.encoders[target_col] = target_encoder
                logger.info(f"Label encoded target column '{target_col}'")
        
        # Store transformation info
        self.transformations['categorical_encoding'] = {
            'strategy': self.config.categorical_strategy,
            'encoded_features': encoded_features,
            'max_categories_threshold': max_cats
        }
        
        return df_encoded
    
    def _split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: ProblemTypeEnum
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets with enhanced class balance validation"""
        
        logger.info(f"Splitting data: {self.config.test_size:.1%} test size")
        self.preprocessing_steps.append("split_data")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Enhanced class balance validation for classification
        stratify_param = None
        if (self.config.stratify and 
            problem_type == ProblemTypeEnum.CLASSIFICATION and 
            y.nunique() > 1):
            
            # Check class distribution
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            total_classes = len(class_counts)
            
            # Calculate minimum samples needed for stratified split
            min_samples_needed = max(2, int(1 / self.config.test_size))
            
            logger.info(f"Class distribution analysis:")
            logger.info(f"  - Total classes: {total_classes}")
            logger.info(f"  - Smallest class size: {min_class_size}")
            logger.info(f"  - Minimum samples needed for stratification: {min_samples_needed}")
            
            # Improved stratification decision logic
            if min_class_size >= min_samples_needed and y.nunique() <= len(y) * 0.5:
                stratify_param = y
                logger.info("✅ Using stratified sampling")
            else:
                if min_class_size < min_samples_needed:
                    logger.warning(f"❌ Cannot use stratified sampling: smallest class has only {min_class_size} samples, need at least {min_samples_needed}")
                    logger.warning("🔄 Falling back to random sampling")
                else:
                    logger.warning(f"❌ Cannot use stratified sampling: too many classes ({total_classes})")
                    logger.warning("🔄 Falling back to random sampling")
                
                # Additional warning if this might be a problem
                if min_class_size == 1:
                    logger.error(f"⚠️ CRITICAL: Target column '{target_col}' has classes with only 1 sample!")
                    logger.error("💡 Suggestions:")
                    logger.error("   1. Check if this is the correct target column")
                    logger.error("   2. Consider binning continuous variables into categories")
                    logger.error("   3. Remove identifier columns (like student_id, customer_id, etc.)")
                    
                    # Check for potential ID columns in features
                    potential_ids = []
                    for col in X.columns:
                        if X[col].nunique() == len(X):
                            potential_ids.append(col)
                    
                    if potential_ids:
                        logger.error(f"   4. Found potential ID columns to exclude: {potential_ids}")
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_param
            )
            
        except ValueError as e:
            if "least populated class" in str(e):
                logger.error(f"❌ Stratified sampling failed: {str(e)}")
                logger.warning("🔄 Retrying with random sampling...")
                
                # Retry without stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=None
                )
                stratify_param = None
                logger.info("✅ Random sampling successful")
            else:
                # Re-raise other ValueError types
                raise
        
        # Store transformation info
        self.transformations['data_split'] = {
            'test_size': self.config.test_size,
            'stratified': stratify_param is not None,
            'random_state': self.config.random_state,
            'class_balance_info': {
                'total_classes': y.nunique(),
                'min_class_size': y.value_counts().min() if problem_type == ProblemTypeEnum.CLASSIFICATION else None,
                'stratification_attempted': self.config.stratify and problem_type == ProblemTypeEnum.CLASSIFICATION
            }
        }
        
        logger.info(f"Data split completed: {len(X_train)} train, {len(X_test)} test samples")
        if stratify_param is not None:
            logger.info("📊 Stratified sampling preserved class distribution")
        else:
            logger.info("🎲 Random sampling used")
            
        return X_train, X_test, y_train, y_test
    
    def _scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features"""
        
        if self.config.scaling_strategy == "none":
            logger.info("No feature scaling applied")
            return X_train, X_test
        
        logger.info(f"Scaling features with strategy: {self.config.scaling_strategy}")
        self.preprocessing_steps.append("scale_features")
        
        # Select only numeric columns for scaling
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.info("No numeric features to scale")
            return X_train, X_test
        
        # Choose scaler
        if self.config.scaling_strategy == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_strategy == "minmax":
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling strategy: {self.config.scaling_strategy}")
            return X_train, X_test
        
        # Fit scaler on training data and transform both sets
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        self.scalers['features'] = scaler
        
        # Store transformation info
        self.transformations['feature_scaling'] = {
            'strategy': self.config.scaling_strategy,
            'scaled_columns': numeric_cols
        }
        
        logger.info(f"Scaled {len(numeric_cols)} numeric features")
        return X_train_scaled, X_test_scaled
    
    def _final_feature_engineering(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        warnings: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final feature engineering steps"""
        
        logger.info("Applying final feature engineering")
        
        # Remove constant features
        if self.config.remove_constant_features:
            constant_features = []
            for col in X_train.columns:
                if X_train[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                X_train = X_train.drop(columns=constant_features)
                X_test = X_test.drop(columns=constant_features)
                
                warning_msg = f"Removed {len(constant_features)} constant features: {constant_features}"
                warnings.append(warning_msg)
                logger.info(warning_msg)
                self.preprocessing_steps.append("remove_constant_features")
        
        # Remove highly correlated features
        if self.config.correlation_threshold < 1.0:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = X_train[numeric_cols].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                highly_correlated = [
                    column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > self.config.correlation_threshold)
                ]
                
                if highly_correlated:
                    X_train = X_train.drop(columns=highly_correlated)
                    X_test = X_test.drop(columns=highly_correlated)
                    
                    warning_msg = f"Removed {len(highly_correlated)} highly correlated features"
                    warnings.append(warning_msg)
                    logger.info(warning_msg)
                    self.preprocessing_steps.append("remove_correlated_features")
        
        return X_train, X_test
    
    def _calculate_missing_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing value summary"""
        missing_summary = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_summary[col] = missing_count / len(df)
        return missing_summary
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed dataframe
        """
        logger.info("Transforming new data using fitted preprocessors")
        
        df_transformed = df.copy()
        
        # Apply same transformations as training data
        # Note: This is a simplified version - in production, you'd want
        # to store and apply the exact same pipeline
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in df_transformed.columns:
                if isinstance(encoder, LabelEncoder):
                    # Handle unseen categories
                    unique_values = df_transformed[col].unique()
                    known_values = encoder.classes_
                    unknown_mask = ~np.isin(unique_values, known_values)
                    
                    if unknown_mask.any():
                        logger.warning(f"Unknown categories found in '{col}': {unique_values[unknown_mask]}")
                        # Replace unknown categories with the most frequent known category
                        most_frequent = encoder.classes_[0]  # Assuming first is most frequent
                        df_transformed[col] = df_transformed[col].replace(
                            unique_values[unknown_mask], most_frequent
                        )
                    
                    df_transformed[f"{col}_encoded"] = encoder.transform(df_transformed[col])
                    df_transformed = df_transformed.drop(columns=[col])
        
        # Apply scalers
        for scaler_name, scaler in self.scalers.items():
            if scaler_name == 'features':
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                df_transformed[numeric_cols] = scaler.transform(df_transformed[numeric_cols])
        
        return df_transformed

def create_preprocessing_config(
    selected_features: Optional[List[str]] = None,
    problem_type: ProblemTypeEnum = ProblemTypeEnum.CLASSIFICATION,
    **kwargs
) -> PreprocessingConfig:
    """
    Factory function to create preprocessing configuration
    
    Args:
        selected_features: List of features selected by user
        problem_type: Type of ML problem
        **kwargs: Additional configuration parameters
        
    Returns:
        PreprocessingConfig instance
    """
    config = PreprocessingConfig(
        selected_features=selected_features,
        **kwargs
    )
    
    # Adjust defaults based on problem type
    if problem_type == ProblemTypeEnum.REGRESSION:
        config.stratify = False  # No stratification for regression
    
    return config

# Create aliases for the test imports
MLDataPreprocessor = DataPreprocessor

def preprocess_ml_data(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main preprocessing function for ML data
    
    Args:
        X: Feature dataframe
        y: Target series
        config: Preprocessing configuration
        
    Returns:
        Dictionary with preprocessed data and metadata
    """
    # Combine X and y back into a single dataframe for processing
    df = X.copy()
    target_col = 'target'
    df[target_col] = y
    
    # Determine problem type based on target
    problem_type = ProblemTypeEnum.CLASSIFICATION
    if y.dtype in ['float64', 'float32'] and y.nunique() > 10:
        problem_type = ProblemTypeEnum.REGRESSION
    
    # Create preprocessing config
    preprocessing_config = create_preprocessing_config(
        problem_type=problem_type,
        **config
    )
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(preprocessing_config)
    
    # Perform preprocessing
    result = preprocessor.preprocess_data(
        df=df,
        target_col=target_col,
        problem_type=problem_type
    )
    
    # Return in expected format
    return {
        'X_train': result.X_train,
        'X_test': result.X_test,
        'y_train': result.y_train,
        'y_test': result.y_test,
        'preprocessor': preprocessor,
        'feature_names': result.feature_names,
        'warnings': result.preprocessing_warnings
    }

def validate_data_quality(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate data quality for ML training
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'issues': []
    }
    
    # Check if target column exists
    if target_column not in df.columns:
        validation_result['is_valid'] = False
        validation_result['issues'].append(f"Target column not found: '{target_column}' not in dataset")
        return validation_result
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        validation_result['warnings'].append(f"Dataset contains {missing_count} missing values")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_result['warnings'].append(f"Dataset contains {duplicate_count} duplicate rows")
    
    # Check target distribution
    target_unique = df[target_column].nunique()
    if target_unique == 1:
        validation_result['is_valid'] = False
        validation_result['issues'].append("Target column has only one unique value")
    elif target_unique == len(df):
        validation_result['warnings'].append("Target has unique value for each row (possible regression problem)")
    
    return validation_result

def generate_preprocessing_summary(
    original_data: pd.DataFrame,
    preprocessing_result: Dict[str, Any],
    config: Dict[str, Any],
    processing_time: float
) -> Dict[str, Any]:
    """
    Generate preprocessing summary
    
    Args:
        original_data: Original dataframe
        preprocessing_result: Result from preprocessing
        config: Preprocessing configuration
        processing_time: Time taken for preprocessing
        
    Returns:
        Summary dictionary
    """
    X_train = preprocessing_result['X_train']
    X_test = preprocessing_result['X_test']
    
    return {
        'original_shape': original_data.shape,
        'final_shape': (len(X_train) + len(X_test), len(X_train.columns)),
        'preprocessing_steps': [
            'Data loading',
            'Missing value handling',
            'Categorical encoding',
            'Feature scaling',
            'Train/test split'
        ],
        'processing_time': processing_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(X_train.columns),
        'warnings': preprocessing_result.get('warnings', [])
    }