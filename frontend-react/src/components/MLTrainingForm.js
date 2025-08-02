import React, { useState } from 'react';
import './MLTrainingForm.css';

const MLTrainingForm = ({ onTrainingSubmit }) => {
  const [formData, setFormData] = useState({
    datasetFile: null,
    problemType: '',
    targetVariable: '',
    algorithms: []
  });
  const [status, setStatus] = useState({ type: '', message: '' });

  const problemTypes = [
    { value: 'classification', label: 'Classification' },
    { value: 'regression', label: 'Regression' },
    { value: 'clustering', label: 'Clustering' }
  ];

  const algorithms = [
    { id: 'random_forest', name: 'Random Forest' },
    { id: 'svm', name: 'Support Vector Machine' },
    { id: 'logistic_regression', name: 'Logistic Regression' },
    { id: 'linear_regression', name: 'Linear Regression' },
    { id: 'kmeans', name: 'K-Means' },
    { id: 'gradient_boosting', name: 'Gradient Boosting' }
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFormData(prev => ({ ...prev, datasetFile: file }));
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleAlgorithmChange = (algorithmId) => {
    setFormData(prev => ({
      ...prev,
      algorithms: prev.algorithms.includes(algorithmId)
        ? prev.algorithms.filter(id => id !== algorithmId)
        : [...prev.algorithms, algorithmId]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.datasetFile) {
      setStatus({ type: 'error', message: 'Please select a dataset file' });
      return;
    }

    if (!formData.problemType) {
      setStatus({ type: 'error', message: 'Please select a problem type' });
      return;
    }

    if (!formData.targetVariable) {
      setStatus({ type: 'error', message: 'Please specify the target variable' });
      return;
    }

    if (formData.algorithms.length === 0) {
      setStatus({ type: 'error', message: 'Please select at least one algorithm' });
      return;
    }

    setStatus({ type: 'loading', message: 'Starting ML training...' });

    try {
      await onTrainingSubmit(formData);
      setStatus({ type: 'success', message: 'Training started successfully!' });
    } catch (error) {
      setStatus({ type: 'error', message: error.message || 'Failed to start training' });
    }
  };

  return (
    <div className="ml-training-form card">
      <h2>ML Training Configuration</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="dataset-file">Dataset File</label>
          <input
            type="file"
            id="dataset-file"
            accept=".csv,.xlsx,.json"
            onChange={handleFileChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="problem-type">Problem Type</label>
          <select
            id="problem-type"
            name="problemType"
            value={formData.problemType}
            onChange={handleInputChange}
            required
          >
            <option value="">Select problem type...</option>
            {problemTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="target-variable">Target Variable</label>
          <input
            type="text"
            id="target-variable"
            name="targetVariable"
            value={formData.targetVariable}
            onChange={handleInputChange}
            placeholder="Column name for prediction target"
            required
          />
        </div>

        <div className="form-group">
          <label>Algorithms to Train</label>
          <div className="algorithm-checkboxes">
            {algorithms.map(algorithm => (
              <label key={algorithm.id} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={formData.algorithms.includes(algorithm.id)}
                  onChange={() => handleAlgorithmChange(algorithm.id)}
                />
                <span className="checkbox-custom"></span>
                {algorithm.name}
              </label>
            ))}
          </div>
        </div>

        <button type="submit">
          Start Training
        </button>

        {status.message && (
          <div className={`status ${status.type}`}>
            {status.message}
          </div>
        )}
      </form>
    </div>
  );
};

export default MLTrainingForm;