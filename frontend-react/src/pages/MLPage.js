import React, { useState, useEffect, useCallback } from 'react';
import Logo from '../components/Logo';
import MLTrainingForm from '../components/MLTrainingForm';
import MLPipelineCard from '../components/MLPipelineCard';
import MLResultsDisplay from '../components/MLResultsDisplay';
import { usePage, PAGES } from '../context/PageContext';
import * as mlApi from '../services/mlApi';
import './MLPage.css';

// Chart.js imports
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const MLPage = () => {
  const { setActivePage } = usePage();
  // Dataset Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState({ message: '', type: '' });
  const [isUploading, setIsUploading] = useState(false);

  // Dataset Management State
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);

  // Training Configuration State
  const [trainingConfig, setTrainingConfig] = useState({
    problemType: 'classification',
    targetVariable: '',
    selectedFeatures: [], // Array of selected feature column names
    algorithms: [],
    preprocessing: {
      scaleFeatures: true,
      handleMissing: 'mean',
      encodeCategories: 'auto'
    },
    testSize: 0.2,
    randomState: 42
  });

  // Dataset Column Information State
  const [datasetColumns, setDatasetColumns] = useState([]);
  const [isLoadingColumns, setIsLoadingColumns] = useState(false);
  const [columnLoadError, setColumnLoadError] = useState(null);

  // Pipeline State
  const [activePipelines, setActivePipelines] = useState([]);
  const [pipelineHistory, setPipelineHistory] = useState([]);
  const [isLoadingPipelines, setIsLoadingPipelines] = useState(false);

  // Results State
  const [selectedResult, setSelectedResult] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [isLoadingResults, setIsLoadingResults] = useState(false);

  // Available algorithms
  const availableAlgorithms = {
    classification: [
      { id: 'random_forest', name: 'Random Forest', description: 'Ensemble method with good accuracy' },
      { id: 'logistic_regression', name: 'Logistic Regression', description: 'Linear model for binary/multiclass' },
      { id: 'svm', name: 'Support Vector Machine', description: 'Effective for high-dimensional data' },
      { id: 'xgboost', name: 'XGBoost', description: 'Gradient boosting with high performance' },
      { id: 'neural_network', name: 'Neural Network', description: 'Deep learning for complex patterns' }
    ],
    regression: [
      { id: 'random_forest_reg', name: 'Random Forest Regressor', description: 'Ensemble method for regression' },
      { id: 'linear_regression', name: 'Linear Regression', description: 'Simple linear relationship modeling' },
      { id: 'svr', name: 'Support Vector Regression', description: 'SVM for continuous targets' },
      { id: 'xgboost_reg', name: 'XGBoost Regressor', description: 'Gradient boosting for regression' },
      { id: 'neural_network_reg', name: 'Neural Network Regressor', description: 'Deep learning for regression' }
    ]
  };


  // Fetch dataset columns
  const fetchDatasetColumns = useCallback(async (dataset) => {
    if (!dataset || !dataset.file_path) return;
    
    try {
      setIsLoadingColumns(true);
      setColumnLoadError(null);
      
      const response = await mlApi.getDatasetColumns(dataset.file_path);
      if (response.success && response.data) {
        setDatasetColumns(response.data.columns || []);
        
        // Reset configuration when columns change
        setTrainingConfig(prev => ({
          ...prev,
          targetVariable: '',
          selectedFeatures: []
        }));
      } else {
        throw new Error(response.error || 'Failed to fetch dataset columns');
      }
    } catch (error) {
      console.error('Error fetching dataset columns:', error);
      setColumnLoadError(error.message);
      setDatasetColumns([]);
    } finally {
      setIsLoadingColumns(false);
    }
  }, []);

  // Fetch available datasets
  const fetchDatasets = useCallback(async () => {
    try {
      setIsLoadingDatasets(true);
      const response = await mlApi.listAvailableDatasets();
      if (response.success) {
        setAvailableDatasets(response.data.datasets || []);
        // Auto-select first dataset if none selected
        if (!selectedDataset && response.data.datasets && response.data.datasets.length > 0) {
          const firstDataset = response.data.datasets[0];
          setSelectedDataset(firstDataset);
          // Also fetch columns for the first dataset
          fetchDatasetColumns(firstDataset);
        }
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error('Error fetching datasets:', error);
      setUploadStatus({ message: `Error fetching datasets: ${error.message}`, type: 'error' });
    } finally {
      setIsLoadingDatasets(false);
    }
  }, [selectedDataset, fetchDatasetColumns]);

  // Fetch pipelines
  const fetchPipelines = useCallback(async () => {
    try {
      setIsLoadingPipelines(true);
      const response = await mlApi.listUserPipelines();
      if (response.success) {
        const allPipelines = response.data.pipelines || [];
        setActivePipelines(allPipelines.filter(p => mlApi.isPipelineRunning(p.status)));
        setPipelineHistory(allPipelines.filter(p => mlApi.isPipelineComplete(p.status)).slice(0, 10));
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error('Error fetching pipelines:', error);
      setUploadStatus({ message: `Error fetching pipelines: ${error.message}`, type: 'error' });
    } finally {
      setIsLoadingPipelines(false);
    }
  }, []);

  // Fetch results
  const fetchResults = useCallback(async (pipelineId) => {
    try {
      setIsLoadingResults(true);
      const response = await mlApi.getPipelineResults(pipelineId);
      if (response.success) {
        setModelMetrics(response.data);
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error('Error fetching results:', error);
      setUploadStatus({ message: `Error fetching results: ${error.message}`, type: 'error' });
    } finally {
      setIsLoadingResults(false);
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchDatasets();
    fetchPipelines();
  }, [fetchDatasets, fetchPipelines]);

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setUploadStatus({ message: '', type: '' });
      } else {
        setUploadStatus({ message: 'Please select a CSV file.', type: 'error' });
        setSelectedFile(null);
      }
    }
  };

  // Handle dataset upload
  const handleDatasetUpload = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
      setUploadStatus({ message: 'Please select a CSV file.', type: 'error' });
      return;
    }

    setIsUploading(true);
    setUploadStatus({ message: 'Uploading dataset...', type: 'loading' });

    try {
      const response = await mlApi.uploadDataset(selectedFile);
      
      if (response.success) {
        setUploadStatus({
          message: `Dataset uploaded successfully!`,
          type: 'success'
        });
        
        // Clear file selection
        setSelectedFile(null);
        document.getElementById('datasetUpload').value = '';
        
        // Refresh datasets list
        fetchDatasets();
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus({ message: `Error: ${error.message}`, type: 'error' });
    } finally {
      setIsUploading(false);
    }
  };

  // Handle dataset selection
  const handleDatasetSelect = (dataset) => {
    setSelectedDataset(dataset);
    fetchDatasetColumns(dataset);
  };

  // Handle configuration changes
  const handleConfigChange = (field, value) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      setTrainingConfig(prev => ({
        ...prev,
        [parent]: {
          ...prev[parent],
          [child]: value
        }
      }));
    } else {
      setTrainingConfig(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  // Handle feature selection (multi-select)
  const handleFeatureToggle = (featureName) => {
    setTrainingConfig(prev => ({
      ...prev,
      selectedFeatures: prev.selectedFeatures.includes(featureName)
        ? prev.selectedFeatures.filter(f => f !== featureName)
        : [...prev.selectedFeatures, featureName]
    }));
  };

  // Get available features (exclude target variable)
  const getAvailableFeatures = () => {
    if (!datasetColumns || datasetColumns.length === 0) return [];
    return datasetColumns.filter(col => col.name !== trainingConfig.targetVariable);
  };

  // Get potential target variables (exclude ID columns)
  const getPotentialTargets = () => {
    if (!datasetColumns || datasetColumns.length === 0) return [];
    
    // Get total rows from first available column or estimate
    const totalRows = datasetColumns.length > 0 ? 
      (datasetColumns.find(col => col.total_rows) || {}).total_rows || 1000 : 1000;
    
    // Filter out likely ID columns (high uniqueness ratio)
    return datasetColumns.filter(col => {
      const uniquenessRatio = col.unique_count / totalRows;
      return uniquenessRatio < 0.95; // Exclude columns where >95% of values are unique
    });
  };

  // Get default hyperparameters for algorithms
  const getDefaultHyperparameters = (algorithmId) => {
    const defaultParams = {
      'random_forest': { n_estimators: 100, max_depth: null, random_state: 42 },
      'random_forest_reg': { n_estimators: 100, max_depth: null, random_state: 42 },
      'logistic_regression': { C: 1.0, max_iter: 1000, random_state: 42 },
      'linear_regression': { fit_intercept: true },
      'svm': { C: 1.0, kernel: 'rbf', random_state: 42 },
      'svr': { C: 1.0, kernel: 'rbf' },
      'xgboost': { n_estimators: 100, learning_rate: 0.1, max_depth: 6, random_state: 42 },
      'xgboost_reg': { n_estimators: 100, learning_rate: 0.1, max_depth: 6, random_state: 42 },
      'neural_network': { hidden_layer_sizes: [100], learning_rate: 'adaptive', random_state: 42 },
      'neural_network_reg': { hidden_layer_sizes: [100], learning_rate: 'adaptive', random_state: 42 }
    };
    return defaultParams[algorithmId] || {};
  };

  // Handle algorithm selection
  const handleAlgorithmToggle = (algorithmId) => {
    setTrainingConfig(prev => ({
      ...prev,
      algorithms: prev.algorithms.includes(algorithmId)
        ? prev.algorithms.filter(id => id !== algorithmId)
        : [...prev.algorithms, algorithmId]
    }));
  };

  // Start training
  const handleStartTraining = async () => {
    if (!selectedDataset) {
      setUploadStatus({ message: 'Please select a dataset for training.', type: 'error' });
      return;
    }

    if (!trainingConfig.targetVariable) {
      setUploadStatus({ message: 'Please specify the target variable.', type: 'error' });
      return;
    }

    if (trainingConfig.algorithms.length === 0) {
      setUploadStatus({ message: 'Please select at least one algorithm.', type: 'error' });
      return;
    }

    if (trainingConfig.selectedFeatures.length === 0) {
      setUploadStatus({ message: 'Please select at least one feature variable.', type: 'error' });
      return;
    }

    try {
      setUploadStatus({ message: 'Starting training pipeline...', type: 'loading' });

      // Convert selected algorithms to backend schema format
      const algorithms = trainingConfig.algorithms.map(algorithmId => {
        // Map frontend algorithm IDs to backend algorithm names
        const algorithmNameMap = {
          'random_forest': 'random_forest_classifier',
          'random_forest_reg': 'random_forest_regressor',
          'logistic_regression': 'logistic_regression',
          'linear_regression': 'linear_regression',
          'svm': 'support_vector_classifier',
          'svr': 'support_vector_regressor',
          'xgboost': 'xgboost_classifier',
          'xgboost_reg': 'xgboost_regressor',
          'neural_network': 'neural_network_classifier',
          'neural_network_reg': 'neural_network_regressor'
        };

        return {
          name: algorithmNameMap[algorithmId] || algorithmId,
          hyperparameters: getDefaultHyperparameters(algorithmId)
        };
      });

      // Map preprocessing config to backend schema
      const preprocessingConfig = {
        selected_features: trainingConfig.selectedFeatures.length > 0 ? trainingConfig.selectedFeatures : null,
        respect_user_selection: trainingConfig.selectedFeatures.length > 0,
        scaling_strategy: trainingConfig.preprocessing.scaleFeatures ? 'standard' : 'none',
        missing_strategy: trainingConfig.preprocessing.handleMissing,
        test_size: trainingConfig.testSize,
        random_state: trainingConfig.randomState
      };

      // Build request matching MLPipelineCreateRequest schema
      const trainingRequest = {
        file_path: selectedDataset.file_path,
        target_variable: trainingConfig.targetVariable,
        problem_type: trainingConfig.problemType,
        algorithms: algorithms,
        preprocessing_config: preprocessingConfig
      };

      const response = await mlApi.startTraining(trainingRequest);
      
      if (response.success) {
        setUploadStatus({
          message: `Training started successfully! Pipeline ID: ${response.data.uuid || response.data.pipeline_id}`,
          type: 'success'
        });
        
        // Refresh pipelines
        fetchPipelines();
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error('Training start error:', error);
      setUploadStatus({ message: `Error starting training: ${error.message}`, type: 'error' });
    }
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return '#00ff88';
      case 'running': return '#fbbf24';
      case 'failed': return '#ef4444';
      case 'queued': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return '✅';
      case 'running': return '🔄';
      case 'failed': return '❌';
      case 'queued': return '⏳';
      default: return '❓';
    }
  };

  // Render metrics chart
  const renderMetricsChart = () => {
    if (!modelMetrics?.results) return null;

    const algorithms = Object.keys(modelMetrics.results);
    const accuracies = algorithms.map(alg => modelMetrics.results[alg].accuracy || 0);
    const precisions = algorithms.map(alg => modelMetrics.results[alg].precision || 0);

    const chartData = {
      labels: algorithms.map(alg => alg.replace('_', ' ').toUpperCase()),
      datasets: [
        {
          label: 'Accuracy',
          data: accuracies,
          backgroundColor: 'rgba(0, 255, 136, 0.6)',
          borderColor: '#00ff88',
          borderWidth: 2,
        },
        {
          label: 'Precision',
          data: precisions,
          backgroundColor: 'rgba(194, 124, 185, 0.6)',
          borderColor: '#c27cb9',
          borderWidth: 2,
        }
      ]
    };

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#e5e7eb' }
        }
      },
      scales: {
        x: {
          ticks: { color: '#9ca3af' },
          grid: { color: '#374151' }
        },
        y: {
          ticks: { color: '#9ca3af' },
          grid: { color: '#374151' },
          min: 0,
          max: 1
        }
      }
    };

    return <Bar data={chartData} options={chartOptions} />;
  };

  const handleViewDocuments = () => {
    setActivePage(PAGES.DOCUMENTS);
  };

  return (
    <div className="ml-page">
      <Logo />
      <h1>Machine Learning Training</h1>
      <p>
        Upload datasets, configure training parameters, and monitor ML pipelines or{' '}
        <button className="link-button" onClick={handleViewDocuments}>
          view your documents
        </button>
      </p>

      {/* Status Message */}
      {uploadStatus.message && (
        <div className={`status-message ${uploadStatus.type}`}>
          {uploadStatus.message}
        </div>
      )}

      {/* Dataset Upload Section */}
      <div className="card">
        <h2>📊 Dataset Upload</h2>
        <p>Upload CSV files for machine learning training</p>
        <form onSubmit={handleDatasetUpload} className="upload-form">
          <div className="file-input-container">
            <input
              type="file"
              id="datasetUpload"
              accept=".csv"
              onChange={handleFileChange}
              disabled={isUploading}
            />
            <label htmlFor="datasetUpload" className="file-label">
              {selectedFile ? selectedFile.name : 'Choose CSV file...'}
            </label>
          </div>
          <button
            type="submit"
            disabled={!selectedFile || isUploading}
            className="upload-btn"
          >
            {isUploading ? 'Uploading...' : 'Upload Dataset'}
          </button>
        </form>
      </div>

      {/* Dataset Selection Section */}
      <div className="card">
        <h2>📂 Dataset Selection</h2>
        <p>Choose a dataset for ML training</p>
        
        {isLoadingDatasets ? (
          <div className="loading">Loading datasets...</div>
        ) : availableDatasets.length > 0 ? (
          <div className="dataset-selection">
            <div className="dataset-grid">
              {availableDatasets.map(dataset => (
                <div 
                  key={dataset.id} 
                  className={`dataset-card ${selectedDataset?.id === dataset.id ? 'selected' : ''}`}
                  onClick={() => handleDatasetSelect(dataset)}
                >
                  <div className="dataset-header">
                    <h4>{dataset.display_name}</h4>
                    <span className="dataset-size">
                      {(dataset.size_bytes / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <div className="dataset-info">
                    <p><strong>File:</strong> {dataset.original_filename}</p>
                    <p><strong>Uploaded:</strong> {new Date(dataset.uploaded_at * 1000).toLocaleString()}</p>
                  </div>
                  {selectedDataset?.id === dataset.id && (
                    <div className="dataset-selected-indicator">✓ Selected</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="no-datasets">
            <p>No datasets available. Please upload a CSV file above to get started.</p>
          </div>
        )}
      </div>

      {/* Training Configuration Section */}
      <div className="card">
        <h2>⚙️ Training Configuration</h2>
        
        {!selectedDataset && (
          <div className="warning-message">
            ⚠️ Please select a dataset above before configuring training parameters.
          </div>
        )}
        
        <div className="config-grid">
          <div className="config-group">
            <label>Problem Type</label>
            <select
              value={trainingConfig.problemType}
              onChange={(e) => handleConfigChange('problemType', e.target.value)}
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>

          <div className="config-group">
            <label>Target Variable</label>
            {isLoadingColumns ? (
              <div className="loading-columns">Loading columns...</div>
            ) : columnLoadError ? (
              <div className="error-message">Error: {columnLoadError}</div>
            ) : (
              <select
                value={trainingConfig.targetVariable}
                onChange={(e) => handleConfigChange('targetVariable', e.target.value)}
                disabled={!selectedDataset || datasetColumns.length === 0}
              >
                <option value="">Select target column...</option>
                {getPotentialTargets().map(column => (
                  <option key={column.name} value={column.name}>
                    {column.name} ({column.type}, {column.unique_count} unique)
                    {column.is_categorical ? ' [Categorical]' : ' [Numerical]'}
                  </option>
                ))}
              </select>
            )}
            {datasetColumns.length > 0 && getPotentialTargets().length < datasetColumns.length && (
              <div className="helper-text">
                💡 ID columns with unique values are automatically excluded
              </div>
            )}
          </div>

          <div className="config-group">
            <label>Feature Variables</label>
            {isLoadingColumns ? (
              <div className="loading-columns">Loading columns...</div>
            ) : columnLoadError ? (
              <div className="error-message">Error: {columnLoadError}</div>
            ) : (
              <div className="feature-dropdown-container">
                <select
                  multiple
                  value={trainingConfig.selectedFeatures}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions, option => option.value);
                    setTrainingConfig(prev => ({
                      ...prev,
                      selectedFeatures: values
                    }));
                  }}
                  disabled={!selectedDataset || getAvailableFeatures().length === 0}
                  className="feature-multiselect"
                  size={Math.min(8, Math.max(4, getAvailableFeatures().length))}
                >
                  {getAvailableFeatures().map(column => (
                    <option key={column.name} value={column.name}>
                      {column.name} ({column.type}, {column.unique_count} unique)
                      {column.is_categorical ? ' [Categorical]' : ' [Numerical]'}
                    </option>
                  ))}
                </select>
                {getAvailableFeatures().length > 0 && (
                  <div className="feature-controls-inline">
                    <button
                      type="button"
                      className="select-all-btn"
                      onClick={() => setTrainingConfig(prev => ({
                        ...prev,
                        selectedFeatures: getAvailableFeatures().map(f => f.name)
                      }))}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="clear-all-btn"
                      onClick={() => setTrainingConfig(prev => ({
                        ...prev,
                        selectedFeatures: []
                      }))}
                    >
                      Clear All
                    </button>
                  </div>
                )}
                <div className="helper-text">
                  {getAvailableFeatures().length > 0 ? (
                    <>
                      <div>Tip: Hold Ctrl/Cmd to select multiple features. Selected: {trainingConfig.selectedFeatures.length} of {getAvailableFeatures().length} features</div>
                      {datasetColumns.length > 0 && getPotentialTargets().length < datasetColumns.length && (
                        <div>Note: ID columns with unique values are automatically excluded</div>
                      )}
                    </>
                  ) : (
                    trainingConfig.targetVariable 
                      ? "No features available (all columns are target or ID columns)"
                      : "Please select a target variable first"
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="config-group">
            <label>Test Split</label>
            <input
              type="number"
              min="0.1"
              max="0.5"
              step="0.05"
              value={trainingConfig.testSize}
              onChange={(e) => handleConfigChange('testSize', parseFloat(e.target.value))}
            />
          </div>

          <div className="config-group">
            <label>Random State</label>
            <input
              type="number"
              value={trainingConfig.randomState}
              onChange={(e) => handleConfigChange('randomState', parseInt(e.target.value))}
            />
          </div>
        </div>

        {/* Algorithm Selection */}
        <div className="algorithm-selection">
          <h3>Select Algorithms</h3>
          <div className="algorithm-grid">
            {availableAlgorithms[trainingConfig.problemType].map(algorithm => (
              <div key={algorithm.id} className="algorithm-card">
                <label className="algorithm-label">
                  <input
                    type="checkbox"
                    checked={trainingConfig.algorithms.includes(algorithm.id)}
                    onChange={() => handleAlgorithmToggle(algorithm.id)}
                  />
                  <div className="algorithm-info">
                    <h4>{algorithm.name}</h4>
                    <p>{algorithm.description}</p>
                  </div>
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Preprocessing Options */}
        <div className="preprocessing-section">
          <h3>Preprocessing Options</h3>
          <div className="preprocessing-grid">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={trainingConfig.preprocessing.scaleFeatures}
                onChange={(e) => handleConfigChange('preprocessing.scaleFeatures', e.target.checked)}
              />
              Scale Features
            </label>

            <div className="config-group">
              <label>Handle Missing Values</label>
              <select
                value={trainingConfig.preprocessing.handleMissing}
                onChange={(e) => handleConfigChange('preprocessing.handleMissing', e.target.value)}
              >
                <option value="mean">Mean Imputation</option>
                <option value="median">Median Imputation</option>
                <option value="mode">Mode Imputation</option>
                <option value="drop">Drop Rows</option>
              </select>
            </div>

            <div className="config-group">
              <label>Encode Categories</label>
              <select
                value={trainingConfig.preprocessing.encodeCategories}
                onChange={(e) => handleConfigChange('preprocessing.encodeCategories', e.target.value)}
              >
                <option value="auto">Auto Detection</option>
                <option value="onehot">One-Hot Encoding</option>
                <option value="label">Label Encoding</option>
                <option value="none">No Encoding</option>
              </select>
            </div>
          </div>
        </div>

        <button onClick={handleStartTraining} className="start-training-btn">
          🚀 Start Training
        </button>
      </div>

      {/* Active Pipelines Section */}
      <div className="card">
        <h2>🔄 Active Pipelines</h2>
        {isLoadingPipelines ? (
          <div className="loading">Loading pipelines...</div>
        ) : activePipelines.length > 0 ? (
          <div className="pipeline-grid">
            {activePipelines.map(pipeline => (
              <div key={pipeline.id} className="pipeline-card">
                <div className="pipeline-header">
                  <span className="pipeline-status">
                    {getStatusIcon(pipeline.status)}
                    <span style={{ color: getStatusColor(pipeline.status) }}>
                      {pipeline.status.toUpperCase()}
                    </span>
                  </span>
                  <span className="pipeline-id">ID: {pipeline.id}</span>
                </div>
                <div className="pipeline-info">
                  <p><strong>Type:</strong> {pipeline.problem_type}</p>
                  <p><strong>Target:</strong> {pipeline.target_variable}</p>
                  <p><strong>Algorithms:</strong> {pipeline.algorithms.join(', ')}</p>
                  <p><strong>Started:</strong> {new Date(pipeline.started_at).toLocaleString()}</p>
                  {pipeline.progress && (
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${pipeline.progress}%` }}
                      ></div>
                      <span className="progress-text">{pipeline.progress}%</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p>No active pipelines</p>
        )}
      </div>

      {/* Recent Results Section */}
      <div className="card">
        <h2>Recent Results</h2>
        <div className="pipeline-history">
          {pipelineHistory.length > 0 ? (
            <div className="history-table">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Status</th>
                    <th>Type</th>
                    <th>Best Score</th>
                    <th>Duration</th>
                    <th>Completed</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {pipelineHistory.map(pipeline => (
                    <tr key={pipeline.id}>
                      <td>{pipeline.id}</td>
                      <td>
                        <span style={{ color: getStatusColor(pipeline.status) }}>
                          {getStatusIcon(pipeline.status)} {pipeline.status}
                        </span>
                      </td>
                      <td>{pipeline.problem_type}</td>
                      <td>{pipeline.best_score?.toFixed(4) || 'N/A'}</td>
                      <td>{pipeline.duration ? `${Math.round(pipeline.duration)}s` : 'N/A'}</td>
                      <td>{pipeline.completed_at ? new Date(pipeline.completed_at).toLocaleString() : 'N/A'}</td>
                      <td>
                        {pipeline.status === 'completed' && (
                          <button
                            onClick={() => {
                              setSelectedResult(pipeline.id);
                              fetchResults(pipeline.id);
                            }}
                            className="view-results-btn"
                          >
                            View Results
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p>No pipeline history available</p>
          )}
        </div>
      </div>

      {/* Results Visualization Section */}
      {selectedResult && (
        <div className="card">
          <h2>📊 Training Results</h2>
          {isLoadingResults ? (
            <div className="loading">Loading results...</div>
          ) : modelMetrics ? (
            <div className="results-container">
              <div className="results-header">
                <h3>Pipeline ID: {selectedResult}</h3>
                <p>Best Algorithm: {modelMetrics.best_algorithm}</p>
              </div>

              <div className="metrics-chart">
                <h4>Model Comparison</h4>
                <div className="chart-wrapper">
                  {renderMetricsChart()}
                </div>
              </div>

              <div className="detailed-metrics">
                <h4>Detailed Metrics</h4>
                <div className="metrics-grid">
                  {Object.entries(modelMetrics.results).map(([algorithm, metrics]) => (
                    <div key={algorithm} className="metric-card">
                      <h5>{algorithm.replace('_', ' ').toUpperCase()}</h5>
                      <div className="metric-values">
                        <span>Accuracy: {metrics.accuracy?.toFixed(4) || 'N/A'}</span>
                        <span>Precision: {metrics.precision?.toFixed(4) || 'N/A'}</span>
                        <span>Recall: {metrics.recall?.toFixed(4) || 'N/A'}</span>
                        <span>F1 Score: {metrics.f1_score?.toFixed(4) || 'N/A'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p>No results available</p>
          )}
        </div>
      )}
    </div>
  );
};

export default MLPage;