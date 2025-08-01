# DrugBAN MLOps Pipeline - Complete Project Documentation

A comprehensive MLOps pipeline for drug-target interaction prediction using the DrugBAN model, featuring data validation, model training, deployment infrastructure, production monitoring, and automated model lifecycle management.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-green.svg)](https://kubernetes.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-latest-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/mlflow-tracking-orange.svg)](https://mlflow.org/)

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Model Architecture & Management](#model-architecture--management)
5. [Data Management with DVC](#data-management-with-dvc)
6. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
7. [API Usage](#api-usage)
8. [Deployment Guide](#deployment-guide)
9. [Production Monitoring & Lifecycle Management](#production-monitoring--lifecycle-management)
10. [Environment-Specific Setup](#environment-specific-setup)
11. [Development Workflow](#development-workflow)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)

## Overview

This project implements a complete MLOps pipeline for drug repurposing using the DrugBAN (Drug-Target Interaction Prediction using Bilinear Attention Network) model. The pipeline includes:

### Core Components
- **Data Management**: DVC for data versioning, Great Expectations for validation
- **Model Training**: PyTorch Lightning with MLflow tracking and Optuna optimization
- **Deployment**: Docker containers with Kubernetes orchestration
- **API Service**: FastAPI with authentication, rate limiting, and batch processing
- **Monitoring**: Prometheus metrics, Grafana dashboards, and model drift detection
- **Infrastructure**: Terraform for cloud provisioning across multiple environments

### Advanced MLOps Features (Phase 7)
- **Automated Retraining**: Intelligent scheduling with drift detection and performance monitoring
- **A/B Testing Framework**: Statistical testing for safe model rollouts with traffic allocation
- **Model Lifecycle Management**: Comprehensive staging with approval workflows and lineage tracking
- **Performance Evaluation**: Automated evaluation with threshold-based alerts and reporting
- **Infrastructure Optimization**: Cost monitoring and automated scaling adjustments

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git
- Make (optional, for convenience commands)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd MLOpsProject

# Install dependencies
pip install -r requirements.txt

# Install additional development dependencies (optional)
pip install -r requirements-dev.txt
```

### 2. Quick Local Development

```bash
# Start the complete local development stack
make local-dev

# Or manually with docker-compose
docker-compose up -d
```

This will start:
- **API Service**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Redis**: localhost:6379

### 3. Run Your First Prediction

```bash
# Get authentication token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json"

# Make a prediction (replace TOKEN with the token from above)
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "drug": {
      "drug_id": "CHEMBL85",
      "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "drug_name": "Caffeine"
    },
    "target": {
      "target_id": "ENSP003",
      "target_name": "Adenosine receptor A2A",
      "uniprot_id": "P29274",
      "organism": "Homo sapiens",
      "target_class": "GPCR"
    }
  }'
```

## Project Structure

```
MLOpsProject/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── params.yaml              # Pipeline parameters
├── Dockerfile               # Container build
├── docker-compose.yml       # Local development stack
├── Makefile                 # Build and deployment automation
├── deploy.sh                # Deployment script
├── changelog.txt            # Complete implementation history
│
├── api/                     # FastAPI application
│   ├── __init__.py
│   └── main.py              # API server with auth and rate limiting
│
├── scripts/                 # Data processing and training scripts
│   ├── validate_data.py     # Data validation with Great Expectations
│   ├── import_labeled_datasets.py    # Data import and splitting
│   ├── generate_molecular_fingerprints.py  # Feature extraction
│   ├── apply_data_augmentation.py    # Data augmentation
│   ├── train_model.py       # PyTorch Lightning training
│   ├── hyperparameter_tuning.py     # Optuna optimization
│   ├── cross_validation.py  # K-fold cross-validation
│   ├── error_analysis.py    # Model analysis and interpretation
│   ├── statistical_testing.py       # Statistical significance tests
│   ├── model_comparison.py  # Multi-model performance comparison
│   │
│   ├── retraining/          # Automated retraining system
│   │   └── scheduled_retraining.py  # Intelligent retraining pipeline
│   ├── ab_testing/          # A/B testing framework
│   │   ├── ab_test_framework.py     # Statistical A/B testing
│   │   └── prepare_ab_test.py       # Test preparation and configuration
│   ├── evaluation/          # Automated performance evaluation
│   │   └── automated_performance_evaluation.py
│   └── lifecycle/           # Model lifecycle management
│       ├── model_staging.py        # Stage transitions and management
│       └── approval_workflows.py   # Enterprise approval workflows
│
├── configs/                 # Configuration files
│   ├── retraining_config.yaml      # Retraining pipeline configuration
│   ├── ab_test_config.yaml         # A/B testing configuration
│   ├── evaluation_config.yaml      # Performance evaluation settings
│   └── model_staging_config.yaml   # Model staging configuration
│
├── k8s/                     # Kubernetes deployment manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── redis.yaml
│   ├── api-deployment.yaml
│   ├── hpa.yaml             # Horizontal Pod Autoscaler
│   └── ingress.yaml         # NGINX Ingress Controller
│
├── terraform/               # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── environments/        # Environment-specific configs
│
├── .github/workflows/       # CI/CD workflows
│   └── scheduled-retraining.yml     # Automated retraining workflow
│
├── data/                    # Data directory
│   ├── raw/                 # Original datasets
│   ├── processed/           # Processed features and splits
│   └── labeled/             # Labeled datasets for training
│
├── models/                  # Model storage
│   ├── pretrained/          # Pre-trained model weights
│   ├── trained/             # Newly trained models
│   └── checkpoints/         # Training checkpoints
│
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Unit and integration tests
└── docs/                    # Additional documentation
```

## Model Architecture & Management

### DrugBAN (Bilinear Attention Network)

**Purpose**: Drug-target interaction prediction for drug repurposing applications

**Architecture Components**:
- **Drug Encoder**: Processes molecular fingerprints (Morgan fingerprints, 2048-bit)
- **Target Encoder**: Processes protein sequence embeddings (1024-dimensional)
- **Bilinear Attention**: Multi-head attention mechanism for interaction modeling
- **Classifier**: Final prediction layer with sigmoid activation

**Model Configuration**:
```python
{
    "drug_dim": 2048,        # Morgan fingerprint size
    "target_dim": 1024,      # Target embedding dimension
    "hidden_dim": 512,       # Hidden layer dimension
    "num_heads": 8,          # Attention heads
    "dropout": 0.1          # Dropout rate
}
```

### MLflow Integration

**Experiments Structure**:
1. **drug_repurposing_baseline**: Baseline model evaluations
2. **drug_repurposing_training**: Model training experiments
3. **drug_repurposing_evaluation**: Model comparison and validation
4. **drug_repurposing_production**: Production model deployments

**Model Registry**:
Models are registered in MLflow Model Registry with the following naming convention:
- `DrugBAN_baseline`: Pre-trained baseline model
- `DrugBAN_v{version}`: Trained model versions
- `DrugBAN_production`: Currently deployed production model

**Environment-Specific Tracking**:
```bash
# Development
export MLFLOW_TRACKING_URI="file:///path/to/mlruns_dev"

# Staging
export MLFLOW_TRACKING_URI="file:///path/to/mlruns_staging"

# Production
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
```

### Model Lifecycle Management

**Development → Staging → Production Pipeline**:
1. **Development**: Model trained and validated locally
2. **Staging**: Model deployed to staging environment for testing
3. **Review**: Performance review and approval by team
4. **Production**: Model promoted to production deployment

**Automated Model Staging** (Phase 7 Implementation):
- Intelligent stage transitions based on performance metrics
- Digital signature-based approval workflows
- Emergency deployment procedures
- Comprehensive audit trails and lineage tracking

### Feature Engineering

**Drug Features - Morgan Fingerprints**:
```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
```

**Target Features - Protein Sequence Embeddings**:
- ESM (Evolutionary Scale Modeling) embeddings
- ProtBERT embeddings
- Custom sequence-based features

### Model Performance Monitoring

**Training Metrics**:
- Loss curves (training/validation)
- Learning rate schedules
- Gradient norms
- Model convergence indicators

**Performance Metrics**:
- Classification metrics (accuracy, precision, recall, F1)
- Ranking metrics (ROC-AUC, AP)
- Calibration metrics (Brier score, reliability diagrams)

## Data Management with DVC

### Overview

DVC (Data Version Control) enables version control for datasets, machine learning models, and ensures reproducible data pipelines in the MLOps Drug Repurposing project.

### Configuration

**Automated Setup**:
```bash
# Configure for development environment
python dvc_config.py dev

# Configure for staging environment  
python dvc_config.py staging

# Configure for production environment
python dvc_config.py prod
```

**Manual Configuration**:
```bash
# Add S3 remote for data storage
dvc remote add -d data s3://mlops-drug-repurposing-dev-data-bucket

# Add S3 remote for models
dvc remote add models s3://mlops-drug-repurposing-dev-models-bucket

# Configure AWS credentials
dvc remote modify data credentialpath ~/.aws/credentials
```

### Data Structure

```
data/
├── raw/                    # Original datasets (DVC tracked)
│   ├── biosnap/           # BIOSNAP drug-target data
│   ├── bindingdb/         # BindingDB binding data  
│   └── sample/            # Sample datasets for testing
├── processed/             # Processed datasets (pipeline outputs)
└── external/              # External reference data
```

### Pipeline Management

**DVC Pipeline Configuration** (`dvc.yaml`):
```yaml
stages:
  data_download:
    cmd: python data/download_datasets.py
    deps:
      - data/download_datasets.py
    outs:
      - data/raw/sample/
      - data/raw/biosnap/
      - data/raw/bindingdb/

  data_validation:
    cmd: python scripts/validate_data.py
    deps:
      - scripts/validate_data.py
      - data/raw/sample/
    outs:
      - data/validation_report.json
    metrics:
      - data/validation_metrics.json
```

**Running Pipelines**:
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro data_validation

# Show pipeline status
dvc status

# Visualize pipeline
dvc dag
```

**Best Practices**:
- Always add large files (>10MB) to DVC tracking
- Use descriptive commit messages for data changes
- Tag important data versions: `git tag data-v1.0`
- Always run `dvc pull` before starting work
- Push data changes with `dvc push` before git push

## Phase-by-Phase Implementation

### Phase 1: Infrastructure Setup & Data Management ✅

```bash
# Set up DVC for data versioning
python dvc_config.py

# Validate data quality
python scripts/validate_data.py

# Set up MLflow tracking
python mlflow_setup.py
```

### Phase 2: Data Labelling & Organization ✅

```bash
# Set up Label Studio
python scripts/setup_label_studio.py

# Import and organize datasets
python scripts/import_labeled_datasets.py

# Generate molecular fingerprints
python scripts/generate_molecular_fingerprints.py

# Apply data augmentation
python scripts/apply_data_augmentation.py

# Run complete Phase 2 pipeline
python run_pipeline.py
```

### Phase 3: Model Training & Error Analysis ✅

```bash
# Train model with PyTorch Lightning
python scripts/train_model.py

# Hyperparameter optimization
python scripts/hyperparameter_tuning.py 20  # 20 trials

# Cross-validation
python scripts/cross_validation.py

# Comprehensive error analysis
python scripts/error_analysis.py

# Statistical testing
python scripts/statistical_testing.py

# Model comparison
python scripts/model_comparison.py
```

### Phase 4: Deployment Infrastructure ✅

```bash
# Local development
make local-dev

# Build Docker image
make docker-build

# Deploy to Kubernetes
make k8s-deploy

# Or use deployment script
./deploy.sh -e dev deploy
```

### Phase 5: Production Monitoring & Observability ✅

```bash
# Set up Prometheus monitoring
python scripts/setup_monitoring.py

# Configure Grafana dashboards
python scripts/setup_grafana.py

# Enable model drift detection
python scripts/setup_drift_detection.py
```

### Phase 6: Advanced MLOps Features ✅

```bash
# Set up automated testing
python scripts/setup_automated_testing.py

# Configure security scanning
python scripts/setup_security_scanning.py

# Enable advanced logging
python scripts/setup_advanced_logging.py
```

### Phase 7: Model Maintenance & Lifecycle Management 🔄 (In Progress)

#### Step 1: Schedule Periodic Model Retraining ✅
```bash
# Configure automated retraining
python scripts/retraining/scheduled_retraining.py

# Set up GitHub Actions workflow
# .github/workflows/scheduled-retraining.yml
```

**Key Features Implemented**:
- Intelligent triggering based on schedule, performance degradation, and data/concept drift
- Multi-condition retraining system with comprehensive drift detection
- Integration with MLflow Model Registry for seamless model transitions
- Automated validation and staging deployment
- Comprehensive logging and monitoring throughout the retraining process

#### Step 2: A/B Testing Framework ✅
```bash
# Set up A/B testing framework
python scripts/ab_testing/ab_test_framework.py

# Prepare A/B tests
python scripts/ab_testing/prepare_ab_test.py
```

**Key Features Implemented**:
- Statistical A/B testing with power analysis and sample size calculations
- Traffic allocation strategies (hash-based routing, gradual ramp-up)
- Comprehensive metrics collection and analysis
- Integration with Kubernetes for traffic management
- Statistical significance testing with multiple comparison corrections

#### Step 3: Automated Model Performance Evaluation ✅
```bash
# Set up automated evaluation
python scripts/evaluation/automated_performance_evaluation.py
```

**Key Features Implemented**:
- Comprehensive performance metrics calculation (accuracy, AUC, precision, recall, F1, MCC)
- Threshold-based alerting and automated reporting
- Integration with monitoring systems and notification channels
- Historical performance tracking and trend analysis

#### Step 4: Model Staging and Lifecycle Management ✅
```bash
# Configure model staging
python scripts/lifecycle/model_staging.py
```

**Key Features Implemented**:
- Complete model staging system with lifecycle management (None → Staging → Production → Archived)
- Stage transition automation based on performance criteria
- Integration with MLflow Model Registry for centralized model management
- Automated deployment pipeline with rollback capabilities

#### Step 5: Model Approval Workflows ✅
```bash
# Set up approval workflows
python scripts/lifecycle/approval_workflows.py
```

**Key Features Implemented**:
- Enterprise-grade approval workflow system with multi-tier approval
- Digital signature-based approvals with audit trails
- Emergency deployment procedures for critical updates
- Integration with notification systems and documentation requirements

#### Remaining Steps (Pending):
- **Step 6**: Create model lineage tracking
- **Step 7**: Automated dependency updates
- **Step 8**: Infrastructure cost optimization
- **Step 9**: Performance tuning and scaling adjustments

## API Usage

### Authentication

First, get an authentication token:

```bash
curl -X POST "http://localhost:8000/auth/token"
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "drug": {
      "drug_id": "CHEMBL85",
      "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "drug_name": "Caffeine"
    },
    "target": {
      "target_id": "ENSP003",
      "target_name": "Adenosine receptor A2A",
      "uniprot_id": "P29274",
      "organism": "Homo sapiens",
      "target_class": "GPCR"
    },
    "return_confidence": true
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "drug": {"drug_id": "CHEMBL85", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        "target": {"target_id": "ENSP003", "target_name": "Adenosine receptor A2A"}
      }
    ]
  }'
```

### Interactive API Documentation

Visit http://localhost:8000/docs for interactive Swagger documentation.

## Deployment Guide

### Local Development Setup

#### Option 1: Docker Compose (Recommended)

This is the easiest way to get started with the complete stack:

```bash
# Start all services
make local-dev

# View logs
make local-dev-logs

# Stop services
make local-dev-stop
```

**Services included:**
- DrugBAN API server
- Redis for caching and rate limiting
- MLflow tracking server
- NGINX reverse proxy

#### Option 2: Manual Setup

For development with more control:

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Start Redis**
```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install Redis locally and start it
redis-server
```

3. **Set Environment Variables**
```bash
export REDIS_URL="redis://localhost:6379"
export SECRET_KEY="your-secret-key-for-development"
export MLFLOW_TRACKING_URI="file:///path/to/project/mlruns"
```

4. **Start API Server**
```bash
# Start FastAPI server
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Kubernetes

```bash
# Deploy to Kubernetes cluster
make k8s-deploy

# Check deployment status
make k8s-status

# View logs
make k8s-logs

# Clean up
make k8s-delete
```

### Multi-Environment Deployment

```bash
# Development environment
./deploy.sh -e dev deploy

# Staging environment
./deploy.sh -e staging deploy

# Production environment
./deploy.sh -e prod deploy
```

### Docker Only

```bash
# Build and run container
make docker-build
make docker-run

# Stop container
make docker-stop
```

## Production Monitoring & Lifecycle Management

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Kubernetes health
make health-check-k8s
```

### Metrics and Monitoring

- **API Metrics**: http://localhost:8000/metrics
- **MLflow UI**: http://localhost:5000
- **Application Logs**: `make local-dev-logs`

### Performance Monitoring

The API includes built-in monitoring:
- Request/response times
- Prediction accuracy tracking
- Error rates and patterns
- Resource utilization

### Automated Retraining System

**Configuration** (`configs/retraining_config.yaml`):
```yaml
schedule:
  weekly: true
  monthly: true
  cron_expression: "0 2 * * 0"  # Weekly on Sunday at 2 AM

triggers:
  performance_degradation:
    enabled: true
    threshold: 0.05  # 5% drop in performance
  data_drift:
    enabled: true
    threshold: 0.1
  concept_drift:
    enabled: true
    threshold: 0.15
```

**Monitoring and Alerts**:
- Real-time performance tracking
- Automated drift detection
- Slack/email notifications for retraining events
- Comprehensive logging and audit trails

## Environment-Specific Setup

### Development Environment

```bash
# Use development configuration
python mlflow_setup.py dev
python dvc_config.py dev

# Enable debug logging
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export DVC_LOG_LEVEL=DEBUG
```

### Staging Environment

```bash
# Use staging configuration
python mlflow_setup.py staging
python dvc_config.py staging

# Deploy to staging infrastructure
terraform workspace select staging
terraform apply -var-file="environments/staging.tfvars"
```

### Production Environment

```bash
# Use production configuration
python mlflow_setup.py prod
python dvc_config.py prod

# Deploy to production infrastructure
terraform workspace select prod
terraform apply -var-file="environments/prod.tfvars"
```

## Development Workflow

### 1. Code Changes

```bash
# Make your changes
git checkout -b feature/your-feature

# Test locally
make test

# Lint code
make lint

# Format code
make format
```

### 2. Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-api
python -m pytest tests/test_training.py -v
```

### 3. Building and Deployment

```bash
# Full build pipeline
make build

# Deploy to development
make deploy

# Or specific steps
make docker-build
make docker-push
make k8s-deploy
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .
pylint scripts/

# Type checking
mypy scripts/
```

### Pre-commit Hooks

```bash
# Install development requirements
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### 1. Redis Connection Failed

```bash
# Check if Redis is running
docker ps | grep redis

# Restart Redis
docker restart redis

# Or start new instance
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### 2. Model Loading Failed

```bash
# Check MLflow runs
mlflow ui

# Verify model files
ls -la models/

# Run training to create model
python scripts/train_model.py
```

#### 3. Docker Build Issues

```bash
# Clean Docker cache
docker system prune -af

# Rebuild without cache
docker-compose build --no-cache
```

#### 4. DVC Remote Access Issues

```bash
# Check AWS credentials
aws configure list

# Test S3 access
aws s3 ls s3://your-bucket-name

# Update DVC remote
dvc remote modify data url s3://correct-bucket-name
```

#### 5. MLflow Server Not Starting

```bash
# Check if port is available
lsof -i :5000

# Kill existing process
pkill -f "mlflow server"

# Start server manually
mlflow server --host 127.0.0.1 --port 5000
```

#### 6. Python Package Conflicts

```bash
# Create fresh environment
conda create -n mlops-fresh python=3.10
conda activate mlops-fresh

# Install minimal requirements first
pip install -r requirements-minimal.txt

# Then install additional packages
pip install -r requirements.txt
```

#### 7. OpenMP Errors on macOS

```bash
# Set environment variable
export KMP_DUPLICATE_LIB_OK=TRUE

# Add to your shell profile
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bashrc
source ~/.bashrc
```

### Performance Optimization

#### 1. Local Development

- Use docker-compose for consistent environment
- Enable Redis for caching
- Allocate sufficient memory to Docker

#### 2. Production Deployment

- Configure resource limits appropriately
- Enable horizontal pod autoscaling
- Use persistent volumes for model storage
- Configure proper health checks

#### 3. Memory Usage

```bash
# Monitor memory usage
python -m memory_profiler scripts/evaluate_baseline.py

# Reduce batch size for large datasets
# Edit params.yaml and set smaller batch_size
```

#### 4. GPU Support

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-specific packages
pip install torch[cuda]
```

### Security Best Practices

1. **Access Control**:
   - Use IAM roles for S3 access
   - Environment-specific permissions
   - Audit trail through CloudTrail

2. **Data Privacy**:
   - No personal data in tracked datasets
   - Encryption at rest and in transit
   - Regular security audits

3. **API Security**:
   - JWT-based authentication
   - Rate limiting per client
   - Input validation and sanitization

### Getting Help

1. Check the [changelog.txt](changelog.txt) for recent changes
2. Review logs: `make local-dev-logs`
3. Check health endpoints: `curl http://localhost:8000/health`
4. Verify configuration in `params.yaml`
5. Review existing [GitHub issues](link-to-issues)
6. Create a new issue with detailed error information

## Development Tools

### Useful Make Targets

```bash
make help                 # Show all available targets
make setup               # Install dependencies
make quickstart          # Complete setup and start
make test                # Run tests
make lint                # Code linting
make format              # Code formatting
make docker-build        # Build Docker image
make k8s-deploy          # Deploy to Kubernetes
make clean               # Clean up resources
```

### Environment Variables

Key environment variables for configuration:

- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key
- `MLFLOW_TRACKING_URI`: MLflow tracking server
- `MODEL_PATH`: Path to model files
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Monitoring and Logging

#### MLflow UI
Access the MLflow UI at http://localhost:5000 to:
- View experiment runs and metrics
- Compare model performance
- Manage model registry
- Track artifacts and parameters

#### DVC Pipeline Visualization
```bash
# Show pipeline DAG
dvc dag

# Show pipeline status
dvc status

# Show metrics comparison
dvc metrics show
```

#### Logs and Debugging
```bash
# View application logs
tail -f logs/application.log

# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/evaluate_baseline.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use Black for code formatting
- Add type hints where possible
- Include docstrings for functions and classes
- Write unit tests for new features
- Maintain test coverage above 80%
- Update documentation for new features

### Model Governance

**Model Card Template**:
```yaml
model_name: DrugBAN_v1.0
model_type: drug_target_interaction
training_data:
  dataset: BIOSNAP + BindingDB
  size: 100000 interactions
  date_range: 2020-2023
performance:
  accuracy: 0.85
  roc_auc: 0.92
  precision: 0.83
limitations:
  - Limited to known drug-target pairs
  - Bias toward well-studied targets
ethical_considerations:
  - Responsible drug repurposing applications
  - Privacy-preserving inference
```

**Model Approval Process**:
1. **Development**: Model trained and validated locally
2. **Staging**: Model deployed to staging environment for testing
3. **Review**: Performance review and approval by team
4. **Production**: Model promoted to production deployment

### Integration with Other Tools

#### Git Integration
```bash
# DVC files should be committed to git
git add *.dvc .gitignore
git commit -m "Update dataset version"

# Data files are automatically ignored
```

#### MLflow Integration
```bash
# Log DVC data versions in MLflow experiments
mlflow log_param "data_version" $(git rev-parse HEAD)
mlflow log_artifact "data/validation_report.json"
```

#### CI/CD Integration
```bash
# In CI pipeline
dvc pull  # Get latest data
dvc repro # Run pipeline
dvc push  # Push any new outputs
```

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DrugBAN model from the original research paper
- MLflow and PyTorch Lightning communities
- FastAPI and Kubernetes ecosystems
- Open source drug discovery initiatives

---

This unified documentation consolidates all project information from multiple documentation files and includes comprehensive details about the completed Phase 7 implementation with advanced MLOps features for model maintenance and lifecycle management.