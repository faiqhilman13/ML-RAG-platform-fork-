/**
 * ML API Service Layer
 * Provides standardized API functions for machine learning endpoints
 * Follows existing app patterns for error handling and authentication
 */

// Base API configuration
const BASE_URL = 'http://localhost:8000';

/**
 * Generic API request handler with standardized error handling
 * @param {string} endpoint - API endpoint path
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Standardized response object
 */
const apiRequest = async (endpoint, options = {}) => {
  try {
    const url = endpoint.startsWith('http') ? endpoint : `${BASE_URL}${endpoint}`;
    
    const defaultOptions = {
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    // Remove Content-Type for FormData
    if (options.body instanceof FormData) {
      delete defaultOptions.headers['Content-Type'];
    }

    const response = await fetch(url, {
      ...defaultOptions,
      ...options,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || data.error || `Request failed with status ${response.status}`);
    }

    return {
      success: true,
      data: data,
      status: response.status,
    };
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    return {
      success: false,
      error: error.message || 'Network error',
      status: error.status || null,
    };
  }
};

/**
 * Check ML service health status
 * @returns {Promise<Object>} Health status response
 */
export const getMLHealth = async () => {
  return await apiRequest('/api/ml/health');
};

/**
 * Get list of available ML algorithms
 * @returns {Promise<Object>} Available algorithms response
 */
export const getAvailableAlgorithms = async () => {
  return await apiRequest('/api/ml/algorithms');
};

/**
 * Validate dataset for ML training
 * @param {string} filePath - Path to the dataset file
 * @param {string} targetVariable - Target variable for prediction
 * @returns {Promise<Object>} Validation response
 */
export const validateDataset = async (filePath, targetVariable) => {
  if (!filePath || !targetVariable) {
    return {
      success: false,
      error: 'File path and target variable are required',
    };
  }

  return await apiRequest('/api/ml/validate-dataset', {
    method: 'POST',
    body: JSON.stringify({
      file_path: filePath,
      target_variable: targetVariable,
    }),
  });
};

/**
 * Start ML model training
 * @param {Object} config - Training configuration matching MLPipelineCreateRequest schema
 * @param {string} config.file_path - Path to dataset file
 * @param {string} config.target_variable - Target variable column name
 * @param {string} config.problem_type - 'classification' or 'regression'
 * @param {Array} config.algorithms - Array of algorithm objects with name and hyperparameters
 * @param {Object} config.preprocessing_config - Preprocessing configuration
 * @returns {Promise<Object>} Training start response with UUID
 */
export const startTraining = async (config) => {
  if (!config || !config.file_path || !config.target_variable || !config.problem_type) {
    return {
      success: false,
      error: 'file_path, target_variable, and problem_type are required',
    };
  }

  if (!config.algorithms || !Array.isArray(config.algorithms) || config.algorithms.length === 0) {
    return {
      success: false,
      error: 'algorithms array with at least one algorithm is required',
    };
  }

  return await apiRequest('/api/ml/train', {
    method: 'POST',
    body: JSON.stringify(config),
  });
};

/**
 * Get training/pipeline status by UUID
 * @param {string} uuid - Pipeline UUID
 * @returns {Promise<Object>} Pipeline status response
 */
export const getPipelineStatus = async (uuid) => {
  if (!uuid) {
    return {
      success: false,
      error: 'Pipeline UUID is required',
    };
  }

  return await apiRequest(`/api/ml/status/${uuid}`);
};

/**
 * Get training results by UUID
 * @param {string} uuid - Pipeline UUID
 * @returns {Promise<Object>} Pipeline results response
 */
export const getPipelineResults = async (uuid) => {
  if (!uuid) {
    return {
      success: false,
      error: 'Pipeline UUID is required',
    };
  }

  return await apiRequest(`/api/ml/results/${uuid}`);
};

/**
 * Get list of user's ML pipelines
 * @returns {Promise<Object>} User pipelines response
 */
export const listUserPipelines = async () => {
  return await apiRequest('/api/ml/pipelines');
};

/**
 * Delete a ML pipeline by UUID
 * @param {string} uuid - Pipeline UUID to delete
 * @returns {Promise<Object>} Deletion response
 */
export const deletePipeline = async (uuid) => {
  if (!uuid) {
    return {
      success: false,
      error: 'Pipeline UUID is required',
    };
  }

  return await apiRequest(`/api/ml/pipelines/${uuid}`, {
    method: 'DELETE',
  });
};

/**
 * Upload dataset file for ML training
 * @param {File} file - Dataset file to upload
 * @param {string} name - Optional dataset name
 * @returns {Promise<Object>} Upload response
 */
export const uploadDataset = async (file, name = '') => {
  if (!file) {
    return {
      success: false,
      error: 'File is required',
    };
  }

  const formData = new FormData();
  formData.append('file', file);
  if (name.trim()) {
    formData.append('name', name.trim());
  }

  return await apiRequest('/api/ml/upload-dataset', {
    method: 'POST',
    body: formData,
  });
};

/**
 * Get model predictions
 * @param {string} uuid - Trained model UUID
 * @param {Object} input_data - Data for prediction
 * @returns {Promise<Object>} Prediction response
 */
export const getPrediction = async (uuid, input_data) => {
  if (!uuid || !input_data) {
    return {
      success: false,
      error: 'Model UUID and input data are required',
    };
  }

  return await apiRequest(`/api/ml/predict/${uuid}`, {
    method: 'POST',
    body: JSON.stringify({ input_data }),
  });
};

/**
 * Utility function to poll pipeline status until completion
 * @param {string} uuid - Pipeline UUID
 * @param {number} interval - Polling interval in milliseconds (default: 2000)
 * @param {number} maxAttempts - Maximum polling attempts (default: 150)
 * @returns {Promise<Object>} Final status response
 */
export const pollPipelineStatus = async (uuid, interval = 2000, maxAttempts = 150) => {
  if (!uuid) {
    return {
      success: false,
      error: 'Pipeline UUID is required',
    };
  }

  let attempts = 0;
  
  while (attempts < maxAttempts) {
    const statusResponse = await getPipelineStatus(uuid);
    
    if (!statusResponse.success) {
      return statusResponse;
    }

    const status = statusResponse.data.status;
    
    // Terminal states
    if (status === 'completed' || status === 'failed' || status === 'error') {
      return statusResponse;
    }

    // Wait before next poll
    await new Promise(resolve => setTimeout(resolve, interval));
    attempts++;
  }

  return {
    success: false,
    error: 'Polling timeout: Pipeline status check exceeded maximum attempts',
  };
};

/**
 * Utility function to format error messages for UI display
 * @param {Object} response - API response object
 * @returns {string} Formatted error message
 */
export const formatErrorMessage = (response) => {
  if (response.success) {
    return '';
  }
  
  return response.error || 'An unexpected error occurred';
};

/**
 * Utility function to check if a pipeline is in a terminal state
 * @param {string} status - Pipeline status
 * @returns {boolean} True if pipeline is in terminal state
 */
export const isPipelineComplete = (status) => {
  return ['completed', 'failed', 'error'].includes(status);
};

/**
 * Utility function to check if a pipeline is currently running
 * @param {string} status - Pipeline status
 * @returns {boolean} True if pipeline is running
 */
export const isPipelineRunning = (status) => {
  return ['running', 'training', 'processing', 'queued'].includes(status);
};

// Export all functions as default object for easier importing
export default {
  getMLHealth,
  getAvailableAlgorithms,
  validateDataset,
  startTraining,
  getPipelineStatus,
  getPipelineResults,
  listUserPipelines,
  deletePipeline,
  uploadDataset,
  getPrediction,
  pollPipelineStatus,
  formatErrorMessage,
  isPipelineComplete,
  isPipelineRunning,
};