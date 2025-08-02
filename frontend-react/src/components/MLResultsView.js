import React from 'react';
import './MLResultsView.css';

const MLResultsView = ({ results }) => {
  const formatMetricValue = (value) => {
    if (typeof value === 'number') {
      return value < 1 ? (value * 100).toFixed(2) + '%' : value.toFixed(4);
    }
    return value || 'N/A';
  };

  const getBestAlgorithm = () => {
    if (!results?.algorithms || results.algorithms.length === 0) return null;
    
    // Find algorithm with highest accuracy/score
    return results.algorithms.reduce((best, current) => {
      const bestScore = best.accuracy || best.score || 0;
      const currentScore = current.accuracy || current.score || 0;
      return currentScore > bestScore ? current : best;
    });
  };

  const getMetricTableData = () => {
    if (!results?.algorithms) return [];
    
    return results.algorithms.map(algorithm => ({
      name: algorithm.name || 'Unknown',
      accuracy: algorithm.accuracy,
      precision: algorithm.precision,
      recall: algorithm.recall,
      f1_score: algorithm.f1_score,
      training_time: algorithm.training_time,
      ...algorithm.metrics
    }));
  };

  const bestAlgorithm = getBestAlgorithm();

  if (!results) {
    return (
      <div className="ml-results-view card">
        <h2>ML Training Results</h2>
        <div className="no-results">
          No results available. Start a training pipeline to see results here.
        </div>
      </div>
    );
  }

  return (
    <div className="ml-results-view card">
      <h2>ML Training Results</h2>
      
      {/* Best Algorithm Summary */}
      {bestAlgorithm && (
        <div className="best-algorithm-section">
          <h3>Best Performing Algorithm</h3>
          <div className="best-algorithm-card">
            <div className="algorithm-name">{bestAlgorithm.name}</div>
            <div className="algorithm-score">
              Accuracy: {formatMetricValue(bestAlgorithm.accuracy || bestAlgorithm.score)}
            </div>
          </div>
        </div>
      )}

      {/* Metrics Comparison Table */}
      <div className="metrics-table-section">
        <h3>Algorithm Comparison</h3>
        <div className="metrics-table-container">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Training Time</th>
              </tr>
            </thead>
            <tbody>
              {getMetricTableData().map((algorithm, index) => (
                <tr 
                  key={index}
                  className={algorithm.name === bestAlgorithm?.name ? 'best-row' : ''}
                >
                  <td className="algorithm-name-cell">{algorithm.name}</td>
                  <td>{formatMetricValue(algorithm.accuracy)}</td>
                  <td>{formatMetricValue(algorithm.precision)}</td>
                  <td>{formatMetricValue(algorithm.recall)}</td>
                  <td>{formatMetricValue(algorithm.f1_score)}</td>
                  <td>{algorithm.training_time ? `${algorithm.training_time.toFixed(2)}s` : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Training Summary */}
      <div className="training-summary-section">
        <h3>Training Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="summary-label">Problem Type:</span>
            <span className="summary-value">{results.problemType || 'N/A'}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Target Variable:</span>
            <span className="summary-value">{results.targetVariable || 'N/A'}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Dataset Size:</span>
            <span className="summary-value">{results.datasetSize || 'N/A'}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Training Duration:</span>
            <span className="summary-value">
              {results.totalDuration ? `${results.totalDuration.toFixed(2)}s` : 'N/A'}
            </span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Algorithms Trained:</span>
            <span className="summary-value">{results.algorithms?.length || 0}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Date Completed:</span>
            <span className="summary-value">
              {results.completedAt 
                ? new Date(results.completedAt).toLocaleString() 
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Feature Importance (if available) */}
      {results.featureImportance && (
        <div className="feature-importance-section">
          <h3>Feature Importance</h3>
          <div className="feature-list">
            {results.featureImportance.map((feature, index) => (
              <div key={index} className="feature-item">
                <span className="feature-name">{feature.name}</span>
                <div className="importance-bar-container">
                  <div 
                    className="importance-bar"
                    style={{ width: `${(feature.importance * 100)}%` }}
                  ></div>
                  <span className="importance-value">
                    {(feature.importance * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chart Container (placeholder for future chart integration) */}
      <div className="chart-container">
        <h3>Performance Visualization</h3>
        <div className="chart-placeholder">
          <div className="chart-placeholder-content">
            <span>📊</span>
            <p>Chart visualization will be available in future updates</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLResultsView;