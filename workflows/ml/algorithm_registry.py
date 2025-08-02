"""
Algorithm Registry for ML Pipeline

This module provides a centralized registry for all supported ML algorithms
with their configurations, hyperparameters, and validation logic.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from enum import Enum

from app.models.ml_models import AlgorithmNameEnum, ProblemTypeEnum, MetricNameEnum

logger = logging.getLogger(__name__)

class PreprocessingStepEnum(str, Enum):
    """Preprocessing steps that can be recommended for algorithms"""
    SCALE_FEATURES = "scale_features"
    NORMALIZE_FEATURES = "normalize_features"
    HANDLE_MISSING = "handle_missing"
    ENCODE_CATEGORICAL = "encode_categorical"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

@dataclass
class HyperparameterSpec:
    """
    Specification for a single hyperparameter
    
    Defines the type, constraints, and validation rules for hyperparameters
    used by ML algorithms.
    """
    name: str
    type: Type
    default: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    is_required: bool = False
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a hyperparameter value
        
        Args:
            value: The value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None and self.is_required:
            return False, f"Parameter '{self.name}' is required"
        
        if value is None and not self.is_required:
            return True, None
        
        # Type validation
        if not isinstance(value, self.type):
            try:
                value = self.type(value)
            except (ValueError, TypeError):
                return False, f"Parameter '{self.name}' must be of type {self.type.__name__}"
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False, f"Parameter '{self.name}' must be >= {self.min_value}"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Parameter '{self.name}' must be <= {self.max_value}"
        
        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Parameter '{self.name}' must be one of {self.allowed_values}"
        
        return True, None

@dataclass
class AlgorithmDefinition:
    """
    Complete definition of an ML algorithm
    
    Contains all information needed to configure, train, and evaluate
    a specific ML algorithm.
    """
    name: AlgorithmNameEnum
    display_name: str
    description: str
    problem_types: List[ProblemTypeEnum] = field(default_factory=list)
    hyperparameters: List[HyperparameterSpec] = field(default_factory=list)
    default_metrics: List[MetricNameEnum] = field(default_factory=list)
    recommended_preprocessing: List[PreprocessingStepEnum] = field(default_factory=list)
    sklearn_class: str = ""
    supports_feature_importance: bool = True
    supports_probability: bool = True
    is_ensemble: bool = False
    training_complexity: str = "medium"  # low, medium, high
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameter values"""
        return {param.name: param.default for param in self.hyperparameters}
    
    def validate_hyperparameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a set of hyperparameters
        
        Args:
            params: Dictionary of parameter values to validate
            
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        
        # Only validate provided parameters
        for param_name, param_value in params.items():
            # Find the corresponding parameter spec
            param_spec = next((p for p in self.hyperparameters if p.name == param_name), None)
            if param_spec is None:
                errors.append(f"Unknown parameter: '{param_name}'")
                continue
                
            is_valid, error = param_spec.validate(param_value)
            if not is_valid:
                errors.append(error)
                
        # Check for any required parameters that are missing
        for param_spec in self.hyperparameters:
            if param_spec.is_required and param_spec.name not in params:
                errors.append(f"Required parameter '{param_spec.name}' is missing")
        
        return len(errors) == 0, errors

class AlgorithmRegistry:
    """
    Centralized registry for all ML algorithms
    
    Provides access to algorithm definitions, validation, and configuration
    management for the ML training pipeline.
    """
    
    def __init__(self):
        self._algorithms: Dict[AlgorithmNameEnum, AlgorithmDefinition] = {}
        self._initialize_algorithms()
        
    @property
    def algorithms(self) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get all algorithms - property for test compatibility"""
        return self._algorithms
    
    def _initialize_algorithms(self):
        """Initialize all supported algorithms"""
        try:
            # Classification algorithms
            self._register_logistic_regression()
            self._register_random_forest_classifier()
            self._register_svm_classifier()
            self._register_gradient_boosting_classifier()
            self._register_naive_bayes()
            
            # Regression algorithms
            self._register_linear_regression()
            self._register_random_forest_regressor()
            self._register_svm_regressor()
            self._register_gradient_boosting_regressor()
            
            logger.info(f"Initialized {len(self._algorithms)} algorithms in registry")
            
        except Exception as e:
            logger.error(f"Error initializing algorithm registry: {e}")
            raise
    
    def _register_logistic_regression(self):
        """Register Logistic Regression algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="C", type=float, default=1.0,
                min_value=0.001, max_value=1000.0,
                description="Regularization strength (inverse)"
            ),
            HyperparameterSpec(
                name="max_iter", type=int, default=1000,
                min_value=100, max_value=10000,
                description="Maximum number of iterations"
            ),
            HyperparameterSpec(
                name="solver", type=str, default="lbfgs",
                allowed_values=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                description="Optimization algorithm"
            ),
            HyperparameterSpec(
                name="penalty", type=str, default="l2",
                allowed_values=["l1", "l2", "elasticnet", "none"],
                description="Regularization penalty"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.LOGISTIC_REGRESSION] = AlgorithmDefinition(
            name=AlgorithmNameEnum.LOGISTIC_REGRESSION,
            display_name="Logistic Regression",
            description="Linear classifier using logistic function for binary and multiclass classification",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.ACCURACY, MetricNameEnum.F1_SCORE, MetricNameEnum.ROC_AUC],
            recommended_preprocessing=[PreprocessingStepEnum.SCALE_FEATURES, PreprocessingStepEnum.HANDLE_MISSING],
            sklearn_class="sklearn.linear_model.LogisticRegression",
            supports_feature_importance=False,
            supports_probability=True,
            training_complexity="low"
        )
    
    def _register_random_forest_classifier(self):
        """Register Random Forest Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_estimators", type=int, default=100,
                min_value=10, max_value=1000,
                description="Number of trees in the forest"
            ),
            HyperparameterSpec(
                name="max_depth", type=int, default=None,
                min_value=1, max_value=50,
                description="Maximum depth of trees", is_required=False
            ),
            HyperparameterSpec(
                name="min_samples_split", type=int, default=2,
                min_value=2, max_value=20,
                description="Minimum samples required to split a node",
                is_required=False
            ),
            HyperparameterSpec(
                name="min_samples_leaf", type=int, default=1,
                min_value=1, max_value=20,
                description="Minimum samples required at each leaf node",
                is_required=False
            ),
            HyperparameterSpec(
                name="max_features", type=str, default="sqrt",
                allowed_values=["auto", "sqrt", "log2"],
                description="Number of features to consider for best split",
                is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
            display_name="Random Forest Classifier",
            description="Ensemble method using multiple decision trees for robust classification",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.ACCURACY, MetricNameEnum.F1_SCORE, MetricNameEnum.PRECISION, MetricNameEnum.RECALL],
            recommended_preprocessing=[PreprocessingStepEnum.HANDLE_MISSING, PreprocessingStepEnum.ENCODE_CATEGORICAL],
            sklearn_class="sklearn.ensemble.RandomForestClassifier",
            supports_feature_importance=True,
            supports_probability=True,
            is_ensemble=True,
            training_complexity="medium"
        )
    
    def _register_svm_classifier(self):
        """Register Support Vector Machine Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="C", type=float, default=1.0,
                min_value=0.001, max_value=1000.0,
                description="Regularization parameter"
            ),
            HyperparameterSpec(
                name="kernel", type=str, default="rbf",
                allowed_values=["linear", "poly", "rbf", "sigmoid"],
                description="Kernel type for the algorithm"
            ),
            HyperparameterSpec(
                name="gamma", type=str, default="scale",
                allowed_values=["scale", "auto"],
                description="Kernel coefficient"
            ),
            HyperparameterSpec(
                name="degree", type=int, default=3,
                min_value=1, max_value=10,
                description="Degree for polynomial kernel"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.SVM_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.SVM_CLASSIFIER,
            display_name="Support Vector Machine",
            description="Support Vector Machine for classification with various kernel functions",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.ACCURACY, MetricNameEnum.F1_SCORE, MetricNameEnum.PRECISION, MetricNameEnum.RECALL],
            recommended_preprocessing=[PreprocessingStepEnum.SCALE_FEATURES, PreprocessingStepEnum.HANDLE_MISSING],
            sklearn_class="sklearn.svm.SVC",
            supports_feature_importance=False,
            supports_probability=True,
            training_complexity="high"
        )
    
    def _register_gradient_boosting_classifier(self):
        """Register Gradient Boosting Classifier algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_estimators", type=int, default=100,
                min_value=10, max_value=1000,
                description="Number of boosting stages"
            ),
            HyperparameterSpec(
                name="learning_rate", type=float, default=0.1,
                min_value=0.01, max_value=1.0,
                description="Learning rate for boosting"
            ),
            HyperparameterSpec(
                name="max_depth", type=int, default=3,
                min_value=1, max_value=10,
                description="Maximum depth of individual estimators"
            ),
            HyperparameterSpec(
                name="subsample", type=float, default=1.0,
                min_value=0.1, max_value=1.0,
                description="Fraction of samples used for fitting individual base learners"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.GRADIENT_BOOSTING_CLASSIFIER] = AlgorithmDefinition(
            name=AlgorithmNameEnum.GRADIENT_BOOSTING_CLASSIFIER,
            display_name="Gradient Boosting Classifier",
            description="Gradient boosting ensemble method for classification tasks",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.ACCURACY, MetricNameEnum.F1_SCORE, MetricNameEnum.ROC_AUC],
            recommended_preprocessing=[PreprocessingStepEnum.HANDLE_MISSING, PreprocessingStepEnum.SCALE_FEATURES],
            sklearn_class="sklearn.ensemble.GradientBoostingClassifier",
            supports_feature_importance=True,
            supports_probability=True,
            is_ensemble=True,
            training_complexity="high"
        )
    
    def _register_naive_bayes(self):
        """Register Naive Bayes algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="alpha", type=float, default=1.0,
                min_value=0.01, max_value=10.0,
                description="Additive (Laplace/Lidstone) smoothing parameter"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.NAIVE_BAYES] = AlgorithmDefinition(
            name=AlgorithmNameEnum.NAIVE_BAYES,
            display_name="Naive Bayes",
            description="Probabilistic classifier based on Bayes' theorem with naive independence assumption",
            problem_types=[ProblemTypeEnum.CLASSIFICATION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.ACCURACY, MetricNameEnum.F1_SCORE, MetricNameEnum.PRECISION, MetricNameEnum.RECALL],
            recommended_preprocessing=[PreprocessingStepEnum.HANDLE_MISSING, PreprocessingStepEnum.ENCODE_CATEGORICAL],
            sklearn_class="sklearn.naive_bayes.MultinomialNB",
            supports_feature_importance=False,
            supports_probability=True,
            training_complexity="low"
        )
    
    def _register_linear_regression(self):
        """Register Linear Regression algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="fit_intercept", type=bool, default=True,
                description="Whether to calculate the intercept"
            ),
            HyperparameterSpec(
                name="normalize", type=bool, default=False,
                description="Whether to normalize features before regression", is_required=False
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.LINEAR_REGRESSION] = AlgorithmDefinition(
            name=AlgorithmNameEnum.LINEAR_REGRESSION,
            display_name="Linear Regression",
            description="Ordinary least squares linear regression for continuous target prediction",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.MAE, MetricNameEnum.MSE, MetricNameEnum.R2_SCORE],
            recommended_preprocessing=[PreprocessingStepEnum.SCALE_FEATURES, PreprocessingStepEnum.HANDLE_MISSING],
            sklearn_class="sklearn.linear_model.LinearRegression",
            supports_feature_importance=False,
            supports_probability=False,
            training_complexity="low"
        )
    
    def _register_random_forest_regressor(self):
        """Register Random Forest Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_estimators", type=int, default=100,
                min_value=10, max_value=1000,
                description="Number of trees in the forest"
            ),
            HyperparameterSpec(
                name="max_depth", type=int, default=None,
                min_value=1, max_value=50,
                description="Maximum depth of trees", is_required=False
            ),
            HyperparameterSpec(
                name="min_samples_split", type=int, default=2,
                min_value=2, max_value=20,
                description="Minimum samples required to split a node"
            ),
            HyperparameterSpec(
                name="min_samples_leaf", type=int, default=1,
                min_value=1, max_value=20,
                description="Minimum samples required at each leaf node"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
            display_name="Random Forest Regressor",
            description="Ensemble method using multiple decision trees for robust regression",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.MAE, MetricNameEnum.MSE, MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE],
            recommended_preprocessing=[PreprocessingStepEnum.HANDLE_MISSING, PreprocessingStepEnum.ENCODE_CATEGORICAL],
            sklearn_class="sklearn.ensemble.RandomForestRegressor",
            supports_feature_importance=True,
            supports_probability=False,
            is_ensemble=True,
            training_complexity="medium"
        )
    
    def _register_svm_regressor(self):
        """Register Support Vector Machine Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="C", type=float, default=1.0,
                min_value=0.001, max_value=1000.0,
                description="Regularization parameter"
            ),
            HyperparameterSpec(
                name="kernel", type=str, default="rbf",
                allowed_values=["linear", "poly", "rbf", "sigmoid"],
                description="Kernel type for the algorithm"
            ),
            HyperparameterSpec(
                name="gamma", type=str, default="scale",
                allowed_values=["scale", "auto"],
                description="Kernel coefficient"
            ),
            HyperparameterSpec(
                name="epsilon", type=float, default=0.1,
                min_value=0.001, max_value=1.0,
                description="Epsilon parameter in the epsilon-SVR model"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.SVM_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.SVM_REGRESSOR,
            display_name="Support Vector Regressor",
            description="Support Vector Machine for regression with various kernel functions",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.MAE, MetricNameEnum.MSE, MetricNameEnum.R2_SCORE],
            recommended_preprocessing=[PreprocessingStepEnum.SCALE_FEATURES, PreprocessingStepEnum.HANDLE_MISSING],
            sklearn_class="sklearn.svm.SVR",
            supports_feature_importance=False,
            supports_probability=False,
            training_complexity="high"
        )
    
    def _register_gradient_boosting_regressor(self):
        """Register Gradient Boosting Regressor algorithm"""
        hyperparams = [
            HyperparameterSpec(
                name="n_estimators", type=int, default=100,
                min_value=10, max_value=1000,
                description="Number of boosting stages"
            ),
            HyperparameterSpec(
                name="learning_rate", type=float, default=0.1,
                min_value=0.01, max_value=1.0,
                description="Learning rate for boosting"
            ),
            HyperparameterSpec(
                name="max_depth", type=int, default=3,
                min_value=1, max_value=10,
                description="Maximum depth of individual estimators"
            ),
            HyperparameterSpec(
                name="subsample", type=float, default=1.0,
                min_value=0.1, max_value=1.0,
                description="Fraction of samples used for fitting individual base learners"
            )
        ]
        
        self._algorithms[AlgorithmNameEnum.GRADIENT_BOOSTING_REGRESSOR] = AlgorithmDefinition(
            name=AlgorithmNameEnum.GRADIENT_BOOSTING_REGRESSOR,
            display_name="Gradient Boosting Regressor",
            description="Gradient boosting ensemble method for regression tasks",
            problem_types=[ProblemTypeEnum.REGRESSION],
            hyperparameters=hyperparams,
            default_metrics=[MetricNameEnum.MAE, MetricNameEnum.MSE, MetricNameEnum.RMSE, MetricNameEnum.R2_SCORE],
            recommended_preprocessing=[PreprocessingStepEnum.HANDLE_MISSING, PreprocessingStepEnum.SCALE_FEATURES],
            sklearn_class="sklearn.ensemble.GradientBoostingRegressor",
            supports_feature_importance=True,
            supports_probability=False,
            is_ensemble=True,
            training_complexity="high"
        )
    
    # Public API methods
    
    
    def get_all_algorithms(self) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get all registered algorithms"""
        return self._algorithms.copy()
    
    def get_algorithms_for_problem_type(self, problem_type: ProblemTypeEnum) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get algorithms suitable for a specific problem type"""
        return {
            name: algo for name, algo in self._algorithms.items()
            if problem_type in algo.problem_types
        }
    
    def validate_hyperparameters(self, algorithm_name: Union[str, AlgorithmNameEnum], params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters for a specific algorithm"""
        # Convert string to enum if needed
        if isinstance(algorithm_name, str):
            try:
                algorithm_enum = AlgorithmNameEnum(algorithm_name)
            except ValueError:
                return {'valid': False, 'errors': [f"Unknown algorithm: {algorithm_name}"]}
        else:
            algorithm_enum = algorithm_name
            
        algo = self._algorithms.get(algorithm_enum)
        if not algo:
            return {'valid': False, 'errors': [f"Algorithm '{algorithm_name}' not found"]}
        
        is_valid, errors = algo.validate_hyperparameters(params)
        return {'valid': is_valid, 'errors': errors}
    
    def get_default_hyperparameters(self, algorithm_name: AlgorithmNameEnum) -> Optional[Dict[str, Any]]:
        """Get default hyperparameters for an algorithm"""
        algo = self.get_algorithm(algorithm_name)
        if not algo:
            return None
        
        return algo.get_default_hyperparameters()
    
    def get_algorithms_by_complexity(self, complexity: str) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get algorithms filtered by training complexity"""
        return {
            name: algo for name, algo in self._algorithms.items()
            if algo.training_complexity == complexity
        }
    
    def get_ensemble_algorithms(self) -> Dict[AlgorithmNameEnum, AlgorithmDefinition]:
        """Get all ensemble algorithms"""
        return {
            name: algo for name, algo in self._algorithms.items()
            if algo.is_ensemble
        }
    
    def recommend_algorithms(self, problem_type: ProblemTypeEnum, complexity_preference: str = "medium") -> List[AlgorithmNameEnum]:
        """
        Recommend algorithms based on problem type and complexity preference
        
        Args:
            problem_type: The type of ML problem
            complexity_preference: Preferred complexity level (low, medium, high)
            
        Returns:
            List of recommended algorithm names in order of preference
        """
        suitable_algos = self.get_algorithms_for_problem_type(problem_type)
        
        # Priority order based on general effectiveness and ease of use
        if problem_type == ProblemTypeEnum.CLASSIFICATION:
            priority_order = [
                AlgorithmNameEnum.RANDOM_FOREST_CLASSIFIER,
                AlgorithmNameEnum.GRADIENT_BOOSTING_CLASSIFIER,
                AlgorithmNameEnum.LOGISTIC_REGRESSION,
                AlgorithmNameEnum.SVM_CLASSIFIER,
                AlgorithmNameEnum.NAIVE_BAYES
            ]
        else:  # REGRESSION
            priority_order = [
                AlgorithmNameEnum.RANDOM_FOREST_REGRESSOR,
                AlgorithmNameEnum.GRADIENT_BOOSTING_REGRESSOR,
                AlgorithmNameEnum.LINEAR_REGRESSION,
                AlgorithmNameEnum.SVM_REGRESSOR
            ]
        
        # Filter by complexity preference
        recommended = []
        for algo_name in priority_order:
            if algo_name in suitable_algos:
                algo = suitable_algos[algo_name]
                if complexity_preference == "any" or algo.training_complexity == complexity_preference:
                    recommended.append(algo_name)
        
        # Add remaining algorithms if not enough recommendations
        remaining = [name for name in suitable_algos.keys() if name not in recommended]
        recommended.extend(remaining)
        
        return recommended
    
    def generate_deterministic_random_state(self, pipeline_run_id: str, algorithm_name: str) -> int:
        """
        Generate unique but deterministic random state for reproducible results
        
        Args:
            pipeline_run_id: Unique identifier for the pipeline run
            algorithm_name: Name of the algorithm
            
        Returns:
            Unique integer random state
        """
        return generate_unique_random_state(pipeline_run_id, algorithm_name)
    
    def get_algorithm_instance(self, name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a configured algorithm instance
        
        Args:
            name: Algorithm name (string or enum)
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Configured scikit-learn model instance
        """
        import importlib
        import inspect
        
        # Convert string to enum if needed
        if isinstance(name, str):
            try:
                algorithm_enum = AlgorithmNameEnum(name)
            except ValueError:
                raise ValueError(f"Unknown algorithm: {name}")
        else:
            algorithm_enum = name
        
        # Get algorithm definition
        algo_def = self._algorithms.get(algorithm_enum)
        if not algo_def:
            raise ValueError(f"Algorithm '{algorithm_enum}' not found in registry")
        
        # Prepare hyperparameters
        params = algo_def.get_default_hyperparameters()
        if hyperparameters:
            params.update(hyperparameters)
        
        # Import and instantiate the model
        module_path, class_name = algo_def.sklearn_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Filter parameters that the model actually accepts
        model_signature = inspect.signature(model_class.__init__)
        model_params = {
            k: v for k, v in params.items()
            if k in model_signature.parameters
        }
        
        return model_class(**model_params)
    
    def get_algorithm(self, name: Union[str, AlgorithmNameEnum], hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a configured sklearn algorithm instance (overloaded method for backward compatibility)
        
        Args:
            name: Algorithm name (string or enum)
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Configured scikit-learn model instance or AlgorithmDefinition
        """
        if hyperparameters is not None:
            # If hyperparameters provided, return sklearn instance
            return self.get_algorithm_instance(name, hyperparameters)
        else:
            # If no hyperparameters, return algorithm definition (old behavior)
            return self.get_algorithm_definition(name)
    
    def get_algorithm_definition(self, name: Union[str, AlgorithmNameEnum]) -> Optional[AlgorithmDefinition]:
        """Get algorithm definition by name"""
        # Convert string to enum if needed
        if isinstance(name, str):
            try:
                algorithm_enum = AlgorithmNameEnum(name)
            except ValueError:
                return None
        else:
            algorithm_enum = name
            
        return self._algorithms.get(algorithm_enum)

def generate_unique_random_state(pipeline_run_id: str, algorithm_name: str) -> int:
    """
    Generate unique but deterministic random state for reproducible results
    
    Args:
        pipeline_run_id: Unique identifier for the pipeline run
        algorithm_name: Name of the algorithm
        
    Returns:
        Unique integer random state
    """
    hash_input = f"{pipeline_run_id}_{algorithm_name}"
    hash_object = hashlib.md5(hash_input.encode())
    unique_seed = int(hash_object.hexdigest()[:8], 16) % (2**31 - 1)
    return unique_seed

# Global registry instance
_algorithm_registry = None

def get_algorithm_registry() -> AlgorithmRegistry:
    """Get the global algorithm registry instance"""
    global _algorithm_registry
    if _algorithm_registry is None:
        _algorithm_registry = AlgorithmRegistry()
    return _algorithm_registry

# Convenience functions
def get_available_algorithms(problem_type: Optional[ProblemTypeEnum] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get available algorithms in a format suitable for frontend consumption
    
    Args:
        problem_type: Optional filter by problem type
        
    Returns:
        Dictionary with algorithm information
    """
    registry = get_algorithm_registry()
    
    if problem_type:
        algorithms = registry.get_algorithms_for_problem_type(problem_type)
    else:
        algorithms = registry.get_all_algorithms()
    
    result = {}
    for name, algo in algorithms.items():
        result[name.value] = {
            "name": name.value,
            "display_name": algo.display_name,
            "description": algo.description,
            "problem_types": [pt.value for pt in algo.problem_types],
            "hyperparameters": [
                {
                    "name": param.name,
                    "type": param.type.__name__,
                    "default": param.default,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "allowed_values": param.allowed_values,
                    "description": param.description,
                    "required": param.is_required
                }
                for param in algo.hyperparameters
            ],
            "default_metrics": [metric.value for metric in algo.default_metrics],
            "supports_feature_importance": algo.supports_feature_importance,
            "supports_probability": algo.supports_probability,
            "is_ensemble": algo.is_ensemble,
            "training_complexity": algo.training_complexity
        }
    
    return result

def get_algorithm_recommendation(
    problem_type: str,
    dataset_size: int,
    n_features: int,
    complexity_preference: str = "medium"
) -> Dict[str, Any]:
    """
    Get algorithm recommendations based on dataset characteristics
    
    Args:
        problem_type: Type of ML problem ('classification' or 'regression')
        dataset_size: Number of samples in the dataset
        n_features: Number of features in the dataset
        complexity_preference: Preferred complexity level ('low', 'medium', 'high')
        
    Returns:
        Dictionary with recommended algorithms and reasoning
    """
    registry = get_algorithm_registry()
    
    # Convert string to enum
    problem_type_enum = ProblemTypeEnum(problem_type)
    
    # Get base recommendations
    recommended_algos = registry.recommend_algorithms(problem_type_enum, complexity_preference)
    
    # Add dataset-specific filtering and ranking
    recommendations = []
    
    for algo_name in recommended_algos:
        algo_def = registry.get_algorithm(algo_name)
        if algo_def:
            score = 0.8  # Base score
            reasoning = []
            
            # Adjust score based on dataset characteristics
            if dataset_size < 1000:
                if algo_def.training_complexity == "low":
                    score += 0.1
                    reasoning.append("Good for small datasets")
                elif algo_def.training_complexity == "high":
                    score -= 0.2
                    reasoning.append("May overfit on small datasets")
            elif dataset_size > 10000:
                if algo_def.is_ensemble:
                    score += 0.1
                    reasoning.append("Ensemble methods work well with large datasets")
            
            if n_features > 100:
                if algo_def.supports_feature_importance:
                    score += 0.1
                    reasoning.append("Provides feature importance for high-dimensional data")
            
            recommendations.append({
                "algorithm": algo_name.value,
                "display_name": algo_def.display_name,
                "score": score,
                "reasoning": reasoning,
                "complexity": algo_def.training_complexity,
                "is_ensemble": algo_def.is_ensemble
            })
    
    # Sort by score
    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)
    
    return {
        "recommended_algorithms": [r["algorithm"] for r in recommendations],
        "detailed_recommendations": recommendations,
        "dataset_characteristics": {
            "problem_type": problem_type,
            "dataset_size": dataset_size,
            "n_features": n_features
        },
        "reasoning": f"Based on {problem_type} problem with {dataset_size} samples and {n_features} features"
    }