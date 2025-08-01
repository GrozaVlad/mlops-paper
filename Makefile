# DrugBAN MLOps Project Makefile
# Comprehensive build and deployment automation

.PHONY: help build test deploy clean docker-build docker-push k8s-deploy k8s-delete local-dev

# Variables
DOCKER_REGISTRY ?= drugban
IMAGE_NAME ?= api
IMAGE_TAG ?= latest
NAMESPACE ?= drugban
KUBECONFIG ?= ~/.kube/config

# Help target
help:
	@echo "DrugBAN MLOps Project - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  local-dev        - Start local development environment with docker-compose"
	@echo "  test            - Run all tests"
	@echo "  lint            - Run code linting and formatting"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-push     - Push Docker image to registry"
	@echo "  docker-run      - Run Docker container locally"
	@echo ""
	@echo "Kubernetes:"
	@echo "  k8s-deploy      - Deploy to Kubernetes cluster"
	@echo "  k8s-delete      - Delete Kubernetes deployment"
	@echo "  k8s-status      - Check deployment status"
	@echo "  k8s-logs        - View application logs"
	@echo ""
	@echo "CI/CD:"
	@echo "  build           - Full build pipeline"
	@echo "  deploy          - Full deployment pipeline"
	@echo "  clean           - Clean up build artifacts"

# Development targets
local-dev:
	@echo "🚀 Starting local development environment..."
	docker-compose up -d
	@echo "✅ Environment started!"
	@echo "📊 API available at: http://localhost:8000"
	@echo "📊 API docs at: http://localhost:8000/docs"
	@echo "📊 MLflow UI at: http://localhost:5000"
	@echo "📊 Redis at: localhost:6379"

local-dev-logs:
	docker-compose logs -f

local-dev-stop:
	@echo "🛑 Stopping local development environment..."
	docker-compose down
	@echo "✅ Environment stopped!"

test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --cov=api --cov-report=html
	@echo "✅ Tests completed!"

test-api:
	@echo "🧪 Testing API endpoints..."
	python -m pytest tests/test_api.py -v
	@echo "✅ API tests completed!"

lint:
	@echo "🔍 Running code linting..."
	python -m flake8 api/ scripts/ --max-line-length=120
	python -m black api/ scripts/ --check
	python -m isort api/ scripts/ --check-only
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code..."
	python -m black api/ scripts/
	python -m isort api/ scripts/
	@echo "✅ Code formatted!"

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG) .
	docker build -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):latest .
	@echo "✅ Docker image built: $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)"

docker-push:
	@echo "📤 Pushing Docker image to registry..."
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):latest
	@echo "✅ Docker image pushed!"

docker-run:
	@echo "🏃 Running Docker container..."
	docker run -d \
		--name drugban-api-test \
		-p 8000:8000 \
		-e REDIS_URL=redis://host.docker.internal:6379 \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "✅ Container started at http://localhost:8000"

docker-stop:
	@echo "🛑 Stopping Docker container..."
	docker stop drugban-api-test || true
	docker rm drugban-api-test || true
	@echo "✅ Container stopped!"

# Kubernetes targets
k8s-deploy:
	@echo "☸️  Deploying to Kubernetes..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/redis.yaml
	kubectl apply -f k8s/api-deployment.yaml
	kubectl apply -f k8s/hpa.yaml
	kubectl apply -f k8s/ingress.yaml
	@echo "✅ Kubernetes deployment completed!"
	@echo "⏳ Waiting for pods to be ready..."
	kubectl wait --namespace=$(NAMESPACE) --for=condition=ready pod --selector=app=drugban-api --timeout=300s

k8s-delete:
	@echo "🗑️  Deleting Kubernetes deployment..."
	kubectl delete -f k8s/ingress.yaml --ignore-not-found=true
	kubectl delete -f k8s/hpa.yaml --ignore-not-found=true
	kubectl delete -f k8s/api-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/redis.yaml --ignore-not-found=true
	kubectl delete -f k8s/configmap.yaml --ignore-not-found=true
	kubectl delete -f k8s/namespace.yaml --ignore-not-found=true
	@echo "✅ Kubernetes deployment deleted!"

k8s-status:
	@echo "📊 Kubernetes deployment status:"
	@echo ""
	@echo "Namespace:"
	kubectl get namespace $(NAMESPACE) || echo "Namespace not found"
	@echo ""
	@echo "Pods:"
	kubectl get pods -n $(NAMESPACE) -o wide || echo "No pods found"
	@echo ""
	@echo "Services:"
	kubectl get services -n $(NAMESPACE) || echo "No services found"
	@echo ""
	@echo "Ingress:"
	kubectl get ingress -n $(NAMESPACE) || echo "No ingress found"
	@echo ""
	@echo "HPA:"
	kubectl get hpa -n $(NAMESPACE) || echo "No HPA found"

k8s-logs:
	@echo "📋 API logs:"
	kubectl logs -n $(NAMESPACE) -l app=drugban-api --tail=100 -f

k8s-describe:
	@echo "📝 Describing API deployment:"
	kubectl describe deployment drugban-api -n $(NAMESPACE)

k8s-shell:
	@echo "🐚 Opening shell in API pod..."
	kubectl exec -it -n $(NAMESPACE) $(shell kubectl get pods -n $(NAMESPACE) -l app=drugban-api -o jsonpath='{.items[0].metadata.name}') -- /bin/bash

# CI/CD Pipeline targets
build: lint test docker-build
	@echo "✅ Build pipeline completed!"

deploy: build docker-push k8s-deploy
	@echo "✅ Deployment pipeline completed!"

# Infrastructure targets
infra-plan:
	@echo "📋 Planning Terraform infrastructure..."
	cd terraform && terraform plan -var-file=environments/dev.tfvars

infra-apply:
	@echo "🏗️  Applying Terraform infrastructure..."
	cd terraform && terraform apply -var-file=environments/dev.tfvars -auto-approve

infra-destroy:
	@echo "💥 Destroying Terraform infrastructure..."
	cd terraform && terraform destroy -var-file=environments/dev.tfvars -auto-approve

# Monitoring targets
monitor-setup:
	@echo "📊 Setting up monitoring stack..."
	kubectl apply -f monitoring/prometheus.yaml
	kubectl apply -f monitoring/grafana.yaml
	@echo "✅ Monitoring stack deployed!"

# Security targets
security-scan:
	@echo "🔒 Running security scans..."
	docker run --rm -v $(PWD):/app clair/clair:latest clairctl analyze $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "✅ Security scan completed!"

# Data and Model targets
data-validate:
	@echo "✅ Running data validation..."
	python scripts/validate_data.py
	@echo "✅ Data validation completed!"

model-train:
	@echo "🎯 Training model..."
	python scripts/train_model.py
	@echo "✅ Model training completed!"

model-evaluate:
	@echo "📊 Evaluating model..."
	python scripts/error_analysis.py
	@echo "✅ Model evaluation completed!"

# Cleanup targets
clean:
	@echo "🧹 Cleaning up..."
	docker image prune -f
	docker container prune -f
	docker volume prune -f
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cleanup completed!"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	docker system prune -af
	@echo "✅ Deep cleanup completed!"

# Quick start targets
setup:
	@echo "⚡ Quick setup..."
	pip install -r requirements.txt
	@echo "✅ Setup completed!"

quickstart: setup local-dev
	@echo "🚀 Quick start completed!"
	@echo "📊 Access the API at: http://localhost:8000/docs"

# Health check targets
health-check:
	@echo "🏥 Checking service health..."
	curl -f http://localhost:8000/health || echo "Service not healthy"

health-check-k8s:
	@echo "🏥 Checking Kubernetes service health..."
	kubectl get pods -n $(NAMESPACE) -l app=drugban-api
	kubectl exec -n $(NAMESPACE) $(shell kubectl get pods -n $(NAMESPACE) -l app=drugban-api -o jsonpath='{.items[0].metadata.name}') -- curl -f http://localhost:8000/health