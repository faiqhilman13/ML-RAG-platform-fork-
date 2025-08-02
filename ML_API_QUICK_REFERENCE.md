# ML Pipeline API - Quick Reference Guide

## 🚀 Getting Started

### 1. Check System Health
```bash
curl -X GET "http://localhost:8001/api/ml/health"
```

### 2. Login (required for all ML endpoints)
```bash
curl -X POST "http://localhost:8001/auth/login" \
  -H "Content-Type: application/json" \
  -c cookies.txt \
  -d '{"username": "admin", "password": "admin123"}'
```

### 3. Get Available Algorithms
```bash
curl -X GET "http://localhost:8001/api/ml/algorithms" \
  -b cookies.txt
```

## 🎯 Core ML Workflow

### Step 1: Validate Your Dataset
```bash
curl -X POST "http://localhost:8001/api/ml/validate-dataset?file_path=data/your_data.csv&target_column=target" \
  -b cookies.txt
```

### Step 2: Start Training Pipeline
```bash
curl -X POST "http://localhost:8001/api/ml/train" \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{
    "file_path": "data/your_data.csv",
    "target_variable": "target",
    "problem_type": "classification",
    "algorithms": [
      {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 100}},
      {"name": "logistic_regression", "hyperparameters": {}}
    ]
  }'
```

### Step 3: Monitor Progress
```bash
curl -X GET "http://localhost:8001/api/ml/status/{run_uuid}" \
  -b cookies.txt
```

### Step 4: Get Results (when completed)
```bash
curl -X GET "http://localhost:8001/api/ml/results/{run_uuid}" \
  -b cookies.txt
```

## 📋 Available Endpoints

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/api/ml/health` | GET | System health check | ❌ |
| `/api/ml/algorithms` | GET | Available algorithms | ✅ |
| `/api/ml/validate-dataset` | POST | Dataset validation | ✅ |
| `/api/ml/train` | POST | Start training pipeline | ✅ |
| `/api/ml/status/{uuid}` | GET | Pipeline status | ✅ |
| `/api/ml/results/{uuid}` | GET | Pipeline results | ✅ |
| `/api/ml/pipelines` | GET | List user pipelines | ✅ |
| `/api/ml/pipelines/{uuid}` | DELETE | Delete pipeline | ✅ |

## 🤖 Available Algorithms

### Classification (9 algorithms)
- `random_forest_classifier` - Ensemble method with decision trees
- `logistic_regression` - Linear classification method
- `svm_classifier` - Support Vector Machine
- `gradient_boosting_classifier` - Gradient boosting ensemble
- `naive_bayes` - Probabilistic classifier

### Regression (4 algorithms)  
- `linear_regression` - Simple linear regression
- `random_forest_regressor` - Ensemble regression method
- `svm_regressor` - Support Vector Regression
- `gradient_boosting_regressor` - Gradient boosting regression

## ⚙️ Configuration Options

### Problem Types
- `"classification"` - Predicting categories/classes
- `"regression"` - Predicting continuous values

### Preprocessing Options
```json
{
  "selected_features": ["feature1", "feature2"],
  "categorical_strategy": "onehot",
  "scaling_strategy": "standard", 
  "missing_strategy": "mean",
  "test_size": 0.2,
  "random_state": 42,
  "respect_user_selection": true
}
```

### Pipeline Status Values
- `"pending"` - Waiting to start
- `"running"` - Currently training
- `"completed"` - Successfully finished
- `"failed"` - Training failed
- `"cancelled"` - Manually stopped

## 🎯 Example Training Requests

### Classification Example
```json
{
  "file_path": "data/iris.csv",
  "target_variable": "species",
  "problem_type": "classification",
  "algorithms": [
    {
      "name": "random_forest_classifier",
      "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
      }
    },
    {
      "name": "logistic_regression",
      "hyperparameters": {
        "max_iter": 1000
      }
    }
  ],
  "preprocessing_config": {
    "categorical_strategy": "onehot",
    "scaling_strategy": "standard",
    "test_size": 0.2
  }
}
```

### Regression Example
```json
{
  "file_path": "data/housing.csv", 
  "target_variable": "price",
  "problem_type": "regression",
  "algorithms": [
    {
      "name": "random_forest_regressor",
      "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 15
      }
    },
    {
      "name": "linear_regression",
      "hyperparameters": {}
    }
  ],
  "preprocessing_config": {
    "scaling_strategy": "standard",
    "missing_strategy": "mean",
    "test_size": 0.25
  }
}
```

## 📊 Response Examples

### Training Started Response
```json
{
  "success": true,
  "message": "ML training pipeline started successfully",
  "run_uuid": "ml-run-12345-abcde",
  "status": "running"
}
```

### Status Response
```json
{
  "success": true,
  "run_uuid": "ml-run-12345-abcde",
  "status": "running",
  "progress": {
    "current_step": "training",
    "completed_algorithms": 1,
    "total_algorithms": 2,
    "progress_percentage": 50.0
  },
  "created_at": "2024-01-15T10:00:00Z",
  "total_models_trained": 1,
  "best_model_score": 0.85
}
```

### Final Results Response
```json
{
  "success": true,
  "run_uuid": "ml-run-12345-abcde",
  "status": "completed",
  "problem_type": "classification",
  "target_variable": "species",
  "total_training_time_seconds": 127.3,
  "total_models_trained": 2,
  "best_model": {
    "algorithm": "Random Forest Classifier",
    "score": 0.97,
    "metrics": {
      "accuracy": 0.97,
      "precision": 0.96,
      "recall": 0.95,
      "f1_score": 0.96
    }
  },
  "model_results": [
    {
      "algorithm": "Random Forest Classifier",
      "score": 0.97,
      "training_time": 89.2,
      "status": "completed"
    },
    {
      "algorithm": "Logistic Regression", 
      "score": 0.91,
      "training_time": 12.1,
      "status": "completed"
    }
  ]
}
```

## 🔒 Security & Limits

- **Authentication**: All endpoints except `/health` require valid session
- **File Validation**: Paths checked for security (no `..`, absolute paths blocked)
- **Algorithm Limits**: Maximum 10 algorithms per pipeline
- **Resource Monitoring**: Memory and CPU usage tracked
- **User Isolation**: Each user's pipelines and results are isolated

## 🚨 Common Error Responses

### Authentication Error (401)
```json
{
  "detail": "Authentication required"
}
```

### Validation Error (400)
```json
{
  "detail": "Target variable cannot be empty"
}
```

### Not Found Error (404)
```json
{
  "detail": "Pipeline run not found"
}
```

### Server Error (500) 
```json
{
  "detail": "Internal server error while creating ML training pipeline"
}
```

## 🎉 Success! Your RAG System Now Has ML Capabilities

The ML Pipeline API seamlessly integrates with your existing RAG chatbot system, providing powerful machine learning capabilities while maintaining the same authentication and security standards.

**Ready to start training? Use the examples above or check the full API documentation in `API_REFERENCE.md`**