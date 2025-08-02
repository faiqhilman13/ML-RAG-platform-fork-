import React from 'react';
import './MLPipelineCard.css';

const MLPipelineCard = ({ pipeline }) => {
  const getStatusBadgeClass = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
      case 'success':
        return 'status-badge success';
      case 'running':
      case 'training':
      case 'in_progress':
        return 'status-badge running';
      case 'failed':
      case 'error':
        return 'status-badge error';
      case 'pending':
      case 'queued':
        return 'status-badge pending';
      default:
        return 'status-badge';
    }
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const getProgressPercentage = () => {
    if (pipeline.status === 'completed') return 100;
    if (pipeline.status === 'failed') return 0;
    return pipeline.progress || 0;
  };

  return (
    <div className="ml-pipeline-card card">
      <div className="pipeline-header">
        <h3 className="pipeline-title">{pipeline.name || 'ML Pipeline'}</h3>
        <span className={getStatusBadgeClass(pipeline.status)}>
          {pipeline.status || 'unknown'}
        </span>
      </div>

      <div className="pipeline-details">
        <div className="detail-row">
          <span className="detail-label">Problem Type:</span>
          <span className="detail-value">{pipeline.problemType || 'N/A'}</span>
        </div>
        
        <div className="detail-row">
          <span className="detail-label">Target Variable:</span>
          <span className="detail-value">{pipeline.targetVariable || 'N/A'}</span>
        </div>
        
        <div className="detail-row">
          <span className="detail-label">Algorithms:</span>
          <span className="detail-value">
            {pipeline.algorithms?.length 
              ? pipeline.algorithms.join(', ') 
              : 'N/A'}
          </span>
        </div>
        
        <div className="detail-row">
          <span className="detail-label">Duration:</span>
          <span className="detail-value">{formatDuration(pipeline.duration)}</span>
        </div>
        
        {pipeline.accuracy && (
          <div className="detail-row">
            <span className="detail-label">Best Accuracy:</span>
            <span className="detail-value accuracy">
              {(pipeline.accuracy * 100).toFixed(2)}%
            </span>
          </div>
        )}
      </div>

      {(pipeline.status === 'running' || pipeline.status === 'training') && (
        <div className="progress-section">
          <div className="progress-header">
            <span className="progress-label">Training Progress</span>
            <span className="progress-percentage">{getProgressPercentage()}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${getProgressPercentage()}%` }}
            ></div>
          </div>
          {pipeline.currentAlgorithm && (
            <div className="current-algorithm">
              Currently training: {pipeline.currentAlgorithm}
            </div>
          )}
        </div>
      )}

      <div className="pipeline-metadata">
        <div className="metadata-item">
          <span className="metadata-label">Created:</span>
          <span className="metadata-value">
            {pipeline.createdAt 
              ? new Date(pipeline.createdAt).toLocaleString() 
              : 'N/A'}
          </span>
        </div>
        
        {pipeline.completedAt && (
          <div className="metadata-item">
            <span className="metadata-label">Completed:</span>
            <span className="metadata-value">
              {new Date(pipeline.completedAt).toLocaleString()}
            </span>
          </div>
        )}
      </div>

      {pipeline.error && (
        <div className="error-section">
          <span className="error-label">Error:</span>
          <span className="error-message">{pipeline.error}</span>
        </div>
      )}
    </div>
  );
};

export default MLPipelineCard;