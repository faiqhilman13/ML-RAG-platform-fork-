"""
Test ML Workflows and Algorithms

Tests for ML workflow components including algorithm registry,
preprocessing, evaluation, and training pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from workflows.ml.algorithm_registry import (
    get_algorithm_registry, get_available_algorithms, 
    AlgorithmRegistry, get_algorithm_recommendation
)
from workflows.ml.preprocessing import (
    MLDataPreprocessor, preprocess_ml_data,
    validate_data_quality, generate_preprocessing_summary
)
from workflows.ml.evaluation import (
    evaluate_model_performance, get_classification_metrics,
    get_regression_metrics, compare_models, extract_feature_importance
)
from workflows.pipelines.ml_training import (
    run_ml_training_sync, create_example_pipeline_config
)


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'numeric_feature1': np.random.normal(0, 1, n_samples),
        'numeric_feature2': np.random.normal(5, 2, n_samples),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
        'missing_feature': np.where(np.random.random(n_samples) < 0.3, np.nan, np.random.random(n_samples)),
        'target': np.random.choice([0, 1], n_samples)
    })


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 100
    
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(5, 2, n_samples)
    
    return pd.DataFrame({
        'numeric_feature1': X1,
        'numeric_feature2': X2,
        'categorical_feature': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'target': 2 * X1 + 3 * X2 + np.random.normal(0, 0.5, n_samples)
    })


class TestAlgorithmRegistry:
    """Test algorithm registry functionality."""
    
    def test_get_algorithm_registry(self):
        """Test algorithm registry retrieval."""
        registry = get_algorithm_registry()
        
        assert isinstance(registry, AlgorithmRegistry)
        assert hasattr(registry, 'algorithms')
        assert len(registry.algorithms) > 0
    
    def test_classification_algorithms_available(self):
        """Test that classification algorithms are available."""
        algorithms = get_available_algorithms()
        
        classification_algos = [
            name for name, info in algorithms.items() 
            if 'classification' in info.get('problem_types', [])
        ]
        
        assert len(classification_algos) >= 5
        assert 'random_forest_classifier' in classification_algos
        assert 'logistic_regression' in classification_algos
        assert 'svm_classifier' in classification_algos
    
    def test_regression_algorithms_available(self):
        """Test that regression algorithms are available."""
        algorithms = get_available_algorithms()
        
        regression_algos = [
            name for name, info in algorithms.items()
            if 'regression' in info.get('problem_types', [])
        ]
        
        assert len(regression_algos) >= 4
        assert 'linear_regression' in regression_algos
        assert 'random_forest_regressor' in regression_algos
        assert 'gradient_boosting_regressor' in regression_algos
    
    def test_algorithm_filtering_by_problem_type(self):
        """Test algorithm filtering by problem type."""
        # Test classification filter
        classification_algos = get_available_algorithms(problem_type="classification")
        for algo_info in classification_algos.values():
            assert 'classification' in algo_info['problem_types']
        
        # Test regression filter
        regression_algos = get_available_algorithms(problem_type="regression")
        for algo_info in regression_algos.values():
            assert 'regression' in algo_info['problem_types']
    
    def test_algorithm_hyperparameters_validation(self):
        """Test algorithm hyperparameter validation."""
        registry = get_algorithm_registry()
        
        # Test valid hyperparameters
        valid_params = {'n_estimators': 100, 'max_depth': 5}
        result = registry.validate_hyperparameters('random_forest_classifier', valid_params)
        assert result['valid'] is True
        
        # Test invalid hyperparameters
        invalid_params = {'invalid_param': 'invalid_value'}
        result = registry.validate_hyperparameters('random_forest_classifier', invalid_params)
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_algorithm_recommendation(self):
        """Test algorithm recommendation system."""
        # Test classification recommendation
        recommendation = get_algorithm_recommendation(
            problem_type="classification",
            dataset_size=1000,
            n_features=10
        )
        
        assert 'recommended_algorithms' in recommendation
        assert len(recommendation['recommended_algorithms']) > 0
        
        # Test regression recommendation
        recommendation = get_algorithm_recommendation(
            problem_type="regression",
            dataset_size=500,
            n_features=5
        )
        
        assert 'recommended_algorithms' in recommendation
        assert len(recommendation['recommended_algorithms']) > 0
    
    def test_deterministic_random_state_generation(self):
        """Test deterministic random state generation."""
        registry = get_algorithm_registry()
        
        # Same inputs should produce same random state
        state1 = registry.generate_deterministic_random_state("test_pipeline", "random_forest_classifier")
        state2 = registry.generate_deterministic_random_state("test_pipeline", "random_forest_classifier")
        
        assert state1 == state2
        
        # Different inputs should produce different states
        state3 = registry.generate_deterministic_random_state("different_pipeline", "random_forest_classifier")
        assert state1 != state3


class TestMLDataPreprocessor:
    """Test ML data preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        config = {
            'categorical_strategy': 'onehot',
            'scaling_strategy': 'standard',
            'missing_strategy': 'mean'
        }
        
        preprocessor = MLDataPreprocessor(config)
        
        assert preprocessor.config == config
        assert preprocessor.feature_names_ is None
        assert preprocessor.preprocessing_steps_ == []
    
    def test_data_quality_validation(self, sample_classification_data):
        """Test data quality validation."""
        validation_result = validate_data_quality(sample_classification_data, 'target')
        
        assert 'is_valid' in validation_result
        assert 'warnings' in validation_result
        assert 'issues' in validation_result
        assert validation_result['is_valid'] is True
    
    def test_data_quality_validation_missing_target(self, sample_classification_data):
        """Test data quality validation with missing target."""
        validation_result = validate_data_quality(sample_classification_data, 'nonexistent_target')
        
        assert validation_result['is_valid'] is False
        assert any('Target column not found' in issue for issue in validation_result['issues'])
    
    def test_preprocessing_pipeline_classification(self, sample_classification_data):
        """Test preprocessing pipeline for classification data."""
        X = sample_classification_data.drop('target', axis=1)
        y = sample_classification_data['target']
        
        config = {
            'categorical_strategy': 'onehot',
            'scaling_strategy': 'standard',
            'missing_strategy': 'mean',
            'test_size': 0.2,
            'random_state': 42
        }
        
        result = preprocess_ml_data(X, y, config)
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'preprocessor' in result
        assert 'feature_names' in result
        
        # Check shapes
        assert len(result['X_train']) + len(result['X_test']) == len(X)
        assert len(result['y_train']) + len(result['y_test']) == len(y)
        
        # Check that categorical encoding worked
        assert result['X_train'].shape[1] >= X.shape[1]  # Should have more columns due to one-hot encoding
    
    def test_preprocessing_with_feature_selection(self, sample_classification_data):
        """Test preprocessing with manual feature selection."""
        X = sample_classification_data.drop('target', axis=1)
        y = sample_classification_data['target']
        
        selected_features = ['numeric_feature1', 'categorical_feature']
        config = {
            'selected_features': selected_features,
            'respect_user_selection': True,
            'categorical_strategy': 'onehot',
            'test_size': 0.2
        }
        
        result = preprocess_ml_data(X, y, config)
        
        # Should only use selected features
        original_selected_columns = len(selected_features)
        # One-hot encoding may increase the number of columns
        assert result['X_train'].shape[1] >= original_selected_columns
    
    def test_preprocessing_high_cardinality_warning(self):
        """Test preprocessing with high cardinality categorical features."""
        # Create data with high cardinality categorical feature
        high_card_data = pd.DataFrame({
            'numeric_feature': np.random.normal(0, 1, 100),
            'high_cardinality_cat': [f'cat_{i}' for i in range(100)],  # 100 unique categories
            'target': np.random.choice([0, 1], 100)
        })
        
        X = high_card_data.drop('target', axis=1)
        y = high_card_data['target']
        
        config = {
            'categorical_strategy': 'onehot',
            'test_size': 0.2
        }
        
        result = preprocess_ml_data(X, y, config)
        
        # Should have warnings about high cardinality
        assert 'warnings' in result
        assert any('high cardinality' in warning.lower() for warning in result['warnings'])
    
    def test_generate_preprocessing_summary(self, sample_classification_data):
        """Test preprocessing summary generation."""
        X = sample_classification_data.drop('target', axis=1)
        y = sample_classification_data['target']
        
        config = {'test_size': 0.2}
        preprocessing_result = preprocess_ml_data(X, y, config)
        
        summary = generate_preprocessing_summary(
            original_data=sample_classification_data,
            preprocessing_result=preprocessing_result,
            config=config,
            processing_time=1.5
        )
        
        assert 'original_shape' in summary
        assert 'final_shape' in summary
        assert 'preprocessing_steps' in summary
        assert 'processing_time' in summary
        assert summary['processing_time'] == 1.5


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        # Create sample predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1],
                           [0.2, 0.8], [0.4, 0.6], [0.1, 0.9], [0.3, 0.7]])
        
        metrics = get_classification_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check that all metrics are between 0 and 1
        for metric_name, metric_value in metrics.items():
            if metric_name != 'confusion_matrix':
                assert 0 <= metric_value <= 1
    
    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.2, 4.8])
        
        metrics = get_regression_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        
        # Check that MSE and RMSE are positive
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        
        # Check that R² is reasonable (can be negative for very bad models)
        assert -10 <= metrics['r2_score'] <= 1
    
    def test_model_performance_evaluation(self):
        """Test comprehensive model performance evaluation."""
        # Mock a trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 1, 1, 1])
        
        evaluation = evaluate_model_performance(
            model=mock_model,
            X_test=X_test,
            y_test=y_test,
            problem_type='classification'
        )
        
        assert 'metrics' in evaluation
        assert 'predictions' in evaluation
        assert 'probabilities' in evaluation
        assert 'primary_metric' in evaluation
        
        # Verify model was called correctly
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction."""
        # Test with mock model that has feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.7])
        feature_names = ['feature1', 'feature2']
        
        importance = extract_feature_importance(mock_model, feature_names)
        
        assert importance is not None
        assert len(importance) == 2
        assert importance[0]['feature'] == 'feature2'  # Should be sorted by importance
        assert importance[0]['importance'] == 0.7
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        model_results = [
            {
                'model_name': 'Model A',
                'metrics': {'accuracy': 0.85, 'f1_score': 0.83},
                'primary_metric': {'name': 'accuracy', 'value': 0.85}
            },
            {
                'model_name': 'Model B', 
                'metrics': {'accuracy': 0.90, 'f1_score': 0.88},
                'primary_metric': {'name': 'accuracy', 'value': 0.90}
            },
            {
                'model_name': 'Model C',
                'metrics': {'accuracy': 0.82, 'f1_score': 0.80},
                'primary_metric': {'name': 'accuracy', 'value': 0.82}
            }
        ]
        
        comparison = compare_models(model_results, 'accuracy')
        
        assert 'best_model' in comparison
        assert 'model_ranking' in comparison
        assert 'performance_summary' in comparison
        
        # Best model should be Model B
        assert comparison['best_model']['model_name'] == 'Model B'
        
        # Models should be ranked by accuracy
        rankings = comparison['model_ranking']
        assert rankings[0]['model_name'] == 'Model B'
        assert rankings[1]['model_name'] == 'Model A'
        assert rankings[2]['model_name'] == 'Model C'


class TestMLTrainingPipeline:
    """Test ML training pipeline functionality."""
    
    def test_create_example_pipeline_config(self):
        """Test example pipeline configuration creation."""
        config = create_example_pipeline_config()
        
        assert 'file_path' in config
        assert 'target_column' in config
        assert 'problem_type' in config
        assert 'algorithms' in config
        assert 'preprocessing_config' in config
        
        # Check that algorithms list is not empty
        assert len(config['algorithms']) > 0
        
        # Check that each algorithm has required fields
        for algo in config['algorithms']:
            assert 'name' in algo
            assert 'hyperparameters' in algo
    
    @patch('workflows.pipelines.ml_training.pd.read_csv')
    @patch('workflows.pipelines.ml_training.preprocess_ml_data')
    @patch('workflows.pipelines.ml_training.get_algorithm_registry')
    def test_ml_training_sync_success(self, mock_registry, mock_preprocess, mock_read_csv, sample_classification_data):
        """Test successful ML training pipeline execution."""
        # Mock data loading
        mock_read_csv.return_value = sample_classification_data
        
        # Mock preprocessing
        X_train = np.random.random((80, 5))
        X_test = np.random.random((20, 5)) 
        y_train = np.random.choice([0, 1], 80)
        y_test = np.random.choice([0, 1], 20)
        
        mock_preprocess.return_value = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': Mock(),
            'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5'],
            'warnings': []
        }
        
        # Mock algorithm registry
        mock_algo_registry = Mock()
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = y_test
        mock_model.predict_proba.return_value = np.random.random((20, 2))
        
        mock_algo_registry.get_algorithm.return_value = mock_model
        mock_algo_registry.validate_hyperparameters.return_value = {'valid': True, 'errors': []}
        mock_algo_registry.generate_deterministic_random_state.return_value = 42
        
        mock_registry.return_value = mock_algo_registry
        
        # Training configuration
        config = {
            'file_path': 'test.csv',
            'target_column': 'target',
            'problem_type': 'classification',
            'algorithms': [
                {'name': 'random_forest_classifier', 'hyperparameters': {'n_estimators': 10}}
            ],
            'preprocessing_config': {'test_size': 0.2},
            'pipeline_run_id': 'test-uuid'
        }
        
        result = run_ml_training_sync(config)
        
        assert result['success'] is True
        assert 'training_results' in result
        assert 'best_model_summary' in result
        assert 'preprocessing_summary' in result
        assert len(result['training_results']) == 1
    
    @patch('workflows.pipelines.ml_training.pd.read_csv')
    def test_ml_training_sync_file_not_found(self, mock_read_csv):
        """Test ML training pipeline with file not found."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        config = {
            'file_path': 'nonexistent.csv',
            'target_column': 'target',
            'problem_type': 'classification',
            'algorithms': [{'name': 'random_forest_classifier'}],
            'pipeline_run_id': 'test-uuid'
        }
        
        result = run_ml_training_sync(config)
        
        assert result['success'] is False
        assert 'File not found' in result['error']
    
    @patch('workflows.pipelines.ml_training.pd.read_csv')
    def test_ml_training_sync_missing_target_column(self, mock_read_csv, sample_classification_data):
        """Test ML training pipeline with missing target column."""
        mock_read_csv.return_value = sample_classification_data
        
        config = {
            'file_path': 'test.csv',
            'target_column': 'nonexistent_target',
            'problem_type': 'classification',
            'algorithms': [{'name': 'random_forest_classifier'}],
            'pipeline_run_id': 'test-uuid'
        }
        
        result = run_ml_training_sync(config)
        
        assert result['success'] is False
        assert 'Target column' in result['error']


class TestWorkflowIntegration:
    """Test integration between workflow components."""
    
    def test_algorithm_registry_preprocessing_integration(self, sample_classification_data):
        """Test integration between algorithm registry and preprocessing."""
        registry = get_algorithm_registry()
        
        # Get an algorithm
        model = registry.get_algorithm('random_forest_classifier', {'n_estimators': 10})
        
        # Preprocess data
        X = sample_classification_data.drop('target', axis=1)
        y = sample_classification_data['target']
        
        config = {'test_size': 0.2, 'random_state': 42}
        preprocessing_result = preprocess_ml_data(X, y, config)
        
        # Train model
        model.fit(preprocessing_result['X_train'], preprocessing_result['y_train'])
        
        # Evaluate
        evaluation = evaluate_model_performance(
            model=model,
            X_test=preprocessing_result['X_test'],
            y_test=preprocessing_result['y_test'],
            problem_type='classification'
        )
        
        assert evaluation['metrics']['accuracy'] is not None
        assert 0 <= evaluation['metrics']['accuracy'] <= 1
    
    def test_end_to_end_workflow_simulation(self, sample_regression_data):
        """Test end-to-end workflow simulation."""
        # This simulates the complete workflow without actual file I/O
        
        # 1. Data validation
        validation = validate_data_quality(sample_regression_data, 'target')
        assert validation['is_valid'] is True
        
        # 2. Algorithm recommendation
        recommendation = get_algorithm_recommendation(
            problem_type='regression',
            dataset_size=len(sample_regression_data),
            n_features=sample_regression_data.shape[1] - 1
        )
        assert len(recommendation['recommended_algorithms']) > 0
        
        # 3. Preprocessing
        X = sample_regression_data.drop('target', axis=1)
        y = sample_regression_data['target']
        
        preprocessing_result = preprocess_ml_data(X, y, {'test_size': 0.2})
        
        # 4. Model training and evaluation
        registry = get_algorithm_registry()
        recommended_algo = recommendation['recommended_algorithms'][0]
        
        model = registry.get_algorithm(recommended_algo, {})
        model.fit(preprocessing_result['X_train'], preprocessing_result['y_train'])
        
        evaluation = evaluate_model_performance(
            model=model,
            X_test=preprocessing_result['X_test'],
            y_test=preprocessing_result['y_test'],
            problem_type='regression'
        )
        
        # 5. Verify results
        assert evaluation['metrics']['r2_score'] is not None
        assert evaluation['primary_metric']['value'] is not None