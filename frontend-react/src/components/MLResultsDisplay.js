import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import './MLResultsDisplay.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const MLResultsDisplay = ({ pipelineId, initialResults, onClose, className }) => {
  const [results, setResults] = useState(initialResults || null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedView, setSelectedView] = useState('overview');

  // Fetch results if not provided initially
  useEffect(() => {
    if (!results && pipelineId) {
      fetchResults();
    }
  }, [pipelineId]);

  const fetchResults = async () => {
    if (!pipelineId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/ml/results/${pipelineId}`, {
        credentials: 'include'
      });

      const result = await response.json();

      if (response.ok && result.success) {
        setResults(result);
      } else {
        setError(result.detail || 'Failed to fetch results');
      }
    } catch (err) {
      console.error('Error fetching results:', err);
      setError('Error fetching results');
    } finally {
      setIsLoading(false);
    }
  };

  // Format metric values
  const formatMetricValue = (value, metricName) => {
    if (value === null || value === undefined) return 'N/A';
    
    // Percentage metrics
    if (metricName.toLowerCase().includes('accuracy') || 
        metricName.toLowerCase().includes('precision') || 
        metricName.toLowerCase().includes('recall') ||
        metricName.toLowerCase().includes('f1')) {
      return `${(value * 100).toFixed(2)}%`;
    }
    
    // Score metrics (usually 0-1)
    if (metricName.toLowerCase().includes('score')) {
      return value.toFixed(4);
    }
    
    // Default formatting
    return typeof value === 'number' ? value.toFixed(4) : value;
  };

  // Create chart data for model comparison
  const createModelComparisonChart = () => {
    if (!results?.model_results || results.model_results.length === 0) return null;

    const modelNames = results.model_results.map(model => 
      model.algorithm_name || model.model_name || 'Unknown'
    );
    
    // Get the primary metric (first metric in best model)
    const primaryMetricName = results.best_model?.metrics ? 
      Object.keys(results.best_model.metrics)[0] : 'score';
    
    const scores = results.model_results.map(model => {
      const metrics = model.metrics || {};
      return metrics[primaryMetricName] || metrics.score || 0;
    });

    const chartData = {
      labels: modelNames,
      datasets: [
        {
          label: primaryMetricName || 'Score',
          data: scores,
          backgroundColor: 'rgba(0, 255, 255, 0.6)',
          borderColor: 'rgba(0, 255, 255, 1)',
          borderWidth: 1,
        }
      ]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#c9d1d9'
          }
        },
        title: {
          display: true,
          text: 'Model Performance Comparison',
          color: '#c9d1d9'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: '#30363d'
          },
          ticks: {
            color: '#c9d1d9'
          }
        },
        x: {
          grid: {
            color: '#30363d'
          },
          ticks: {
            color: '#c9d1d9'
          }
        }
      }
    };

    return <Bar data={chartData} options={options} />;
  };

  // Create algorithm performance breakdown chart
  const createAlgorithmBreakdownChart = () => {
    if (!results?.model_results || results.model_results.length === 0) return null;

    // Group by algorithm family
    const algorithmGroups = {};
    results.model_results.forEach(model => {
      const algName = model.algorithm_name || 'Unknown';
      const family = algName.split('_')[0]; // e.g., 'random' from 'random_forest'
      
      if (!algorithmGroups[family]) {
        algorithmGroups[family] = [];
      }
      algorithmGroups[family].push(model);
    });

    const labels = Object.keys(algorithmGroups);
    const counts = Object.values(algorithmGroups).map(group => group.length);
    const avgScores = Object.values(algorithmGroups).map(group => {
      const scores = group.map(model => {
        const metrics = model.metrics || {};
        return metrics.score || metrics.accuracy || 0;
      });
      return scores.reduce((sum, score) => sum + score, 0) / scores.length;
    });

    const chartData = {
      labels: labels,
      datasets: [
        {
          label: 'Count',
          data: counts,
          backgroundColor: 'rgba(194, 124, 185, 0.6)',
          borderColor: 'rgba(194, 124, 185, 1)',
          borderWidth: 1,
        }
      ]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#c9d1d9'
          }
        },
        title: {
          display: true,
          text: 'Algorithm Family Distribution',
          color: '#c9d1d9'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: '#30363d'
          },
          ticks: {
            color: '#c9d1d9'
          }
        },
        x: {
          grid: {
            color: '#30363d'
          },
          ticks: {
            color: '#c9d1d9'
          }
        }
      }
    };

    return <Doughnut data={chartData} options={options} />;
  };

  // Render loading state
  if (isLoading) {
    return (
      <div className={`ml-results-display loading ${className || ''}`}>
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <p>Loading training results...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className={`ml-results-display error ${className || ''}`}>
        <div className="error-content">
          <h3>Error Loading Results</h3>
          <p>{error}</p>
          <div className="error-actions">
            <button onClick={fetchResults} className="retry-btn">
              Retry
            </button>
            {onClose && (
              <button onClick={onClose} className="close-btn">
                Close
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Render no results state
  if (!results) {
    return (
      <div className={`ml-results-display no-results ${className || ''}`}>
        <div className="no-results-content">
          <h3>No Results Available</h3>
          <p>No training results found for this pipeline.</p>
          {onClose && (
            <button onClick={onClose} className="close-btn">
              Close
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`ml-results-display ${className || ''}`}>
      {/* Header */}
      <div className="results-header">
        <div className="header-info">
          <h2>Training Results</h2>
          <div className="pipeline-meta">
            <span className="meta-item">Pipeline: {pipelineId}</span>
            <span className="meta-item">
              Problem: {results.problem_type || 'Unknown'}
            </span>
            <span className="meta-item">
              Target: {results.target_variable || 'Unknown'}
            </span>
          </div>
        </div>
        
        {onClose && (
          <button onClick={onClose} className="close-button" title="Close results">
            ✕
          </button>
        )}
      </div>

      {/* Navigation Tabs */}
      <div className="results-nav">
        <button
          className={`nav-tab ${selectedView === 'overview' ? 'active' : ''}`}
          onClick={() => setSelectedView('overview')}
        >
          Overview
        </button>
        <button
          className={`nav-tab ${selectedView === 'models' ? 'active' : ''}`}
          onClick={() => setSelectedView('models')}
        >
          All Models
        </button>
        <button
          className={`nav-tab ${selectedView === 'charts' ? 'active' : ''}`}
          onClick={() => setSelectedView('charts')}
        >
          Visualizations
        </button>
        <button
          className={`nav-tab ${selectedView === 'details' ? 'active' : ''}`}
          onClick={() => setSelectedView('details')}
        >
          Details
        </button>
      </div>

      {/* Content based on selected view */}
      <div className="results-content">
        {selectedView === 'overview' && (
          <div className="overview-content">
            {/* Best Model Card */}
            {results.best_model && (
              <div className="best-model-card">
                <h3>🏆 Best Model</h3>
                <div className="model-info">
                  <div className="model-name">
                    {results.best_model.algorithm_name || 'Unknown Algorithm'}
                  </div>
                  <div className="model-metrics">
                    {results.best_model.metrics && Object.entries(results.best_model.metrics).map(([metric, value]) => (
                      <div key={metric} className="metric-item">
                        <span className="metric-name">{metric}:</span>
                        <span className="metric-value">{formatMetricValue(value, metric)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Summary Stats */}
            <div className="summary-stats">
              <div className="stat-card">
                <div className="stat-value">{results.total_models_trained || 0}</div>
                <div className="stat-label">Models Trained</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">
                  {results.total_training_time_seconds ? 
                    `${Math.round(results.total_training_time_seconds)}s` : 'N/A'}
                </div>
                <div className="stat-label">Training Time</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{results.problem_type || 'Unknown'}</div>
                <div className="stat-label">Problem Type</div>
              </div>
            </div>

            {/* Top 3 Models */}
            {results.model_results && results.model_results.length > 0 && (
              <div className="top-models">
                <h3>Top Performing Models</h3>
                <div className="models-list">
                  {results.model_results.slice(0, 3).map((model, index) => (
                    <div key={index} className="model-summary">
                      <div className="model-rank">#{index + 1}</div>
                      <div className="model-details">
                        <div className="model-name">
                          {model.algorithm_name || `Model ${index + 1}`}
                        </div>
                        <div className="model-score">
                          {model.metrics && Object.keys(model.metrics).length > 0 ? 
                            formatMetricValue(Object.values(model.metrics)[0], Object.keys(model.metrics)[0]) : 
                            'No metrics'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {selectedView === 'models' && (
          <div className="models-content">
            {results.model_results && results.model_results.length > 0 ? (
              <div className="models-table-container">
                <table className="models-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Algorithm</th>
                      <th>Metrics</th>
                      <th>Training Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.model_results.map((model, index) => (
                      <tr key={index} className={index === 0 ? 'best-model' : ''}>
                        <td className="rank-cell">
                          {index === 0 ? '🏆' : `#${index + 1}`}
                        </td>
                        <td className="algorithm-cell">
                          {model.algorithm_name || `Model ${index + 1}`}
                        </td>
                        <td className="metrics-cell">
                          {model.metrics ? (
                            <div className="metrics-list">
                              {Object.entries(model.metrics).map(([metric, value]) => (
                                <div key={metric} className="metric-row">
                                  <span className="metric-name">{metric}:</span>
                                  <span className="metric-value">
                                    {formatMetricValue(value, metric)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <span className="no-metrics">No metrics available</span>
                          )}
                        </td>
                        <td className="time-cell">
                          {model.training_time ? 
                            `${model.training_time.toFixed(2)}s` : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="no-models">
                <p>No model results available</p>
              </div>
            )}
          </div>
        )}

        {selectedView === 'charts' && (
          <div className="charts-content">
            <div className="chart-container">
              <div className="chart-item">
                {createModelComparisonChart()}
              </div>
              <div className="chart-item">
                {createAlgorithmBreakdownChart()}
              </div>
            </div>
          </div>
        )}

        {selectedView === 'details' && (
          <div className="details-content">
            {/* Preprocessing Info */}
            {results.preprocessing_info && (
              <div className="detail-section">
                <h3>Preprocessing Configuration</h3>
                <div className="detail-grid">
                  {Object.entries(results.preprocessing_info).map(([key, value]) => (
                    <div key={key} className="detail-item">
                      <span className="detail-label">{key}:</span>
                      <span className="detail-value">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pipeline Metadata */}
            <div className="detail-section">
              <h3>Pipeline Information</h3>
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Pipeline ID:</span>
                  <span className="detail-value monospace">{pipelineId}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Problem Type:</span>
                  <span className="detail-value">{results.problem_type || 'Unknown'}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Target Variable:</span>
                  <span className="detail-value">{results.target_variable || 'Unknown'}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Total Training Time:</span>
                  <span className="detail-value">
                    {results.total_training_time_seconds ? 
                      `${results.total_training_time_seconds.toFixed(2)} seconds` : 'N/A'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Models Trained:</span>
                  <span className="detail-value">{results.total_models_trained || 0}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

MLResultsDisplay.propTypes = {
  pipelineId: PropTypes.string.isRequired,
  initialResults: PropTypes.object,
  onClose: PropTypes.func,
  className: PropTypes.string
};

export default MLResultsDisplay;