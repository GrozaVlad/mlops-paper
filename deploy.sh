#!/bin/bash
# DrugBAN Deployment Script
# Automated deployment pipeline for different environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="drugban"
DEFAULT_REGISTRY="drugban"
DEFAULT_ENVIRONMENT="dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
DrugBAN Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build Docker image
    push            Push Docker image to registry
    deploy          Deploy to Kubernetes
    local           Start local development environment
    test            Run tests
    clean           Clean up resources

Options:
    -e, --environment   Target environment (dev, staging, prod) [default: dev]
    -r, --registry      Docker registry [default: drugban]
    -t, --tag          Image tag [default: latest]
    -n, --namespace    Kubernetes namespace [default: drugban]
    -h, --help         Show this help message

Environment Variables:
    DOCKER_REGISTRY    Docker registry URL
    KUBECONFIG        Kubernetes config file path
    SECRET_KEY        API secret key for JWT tokens

Examples:
    $0 build                        # Build Docker image
    $0 -e prod deploy              # Deploy to production
    $0 -r myregistry.com push      # Push to custom registry
    $0 local                       # Start local development

EOF
}

# Parse command line arguments
ENVIRONMENT="$DEFAULT_ENVIRONMENT"
REGISTRY="$DEFAULT_REGISTRY"
TAG="latest"
NAMESPACE="drugban"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|push|deploy|local|test|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "${COMMAND:-}" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Set environment-specific variables
case "$ENVIRONMENT" in
    dev)
        REPLICAS=1
        RESOURCES_REQUESTS_CPU="100m"
        RESOURCES_REQUESTS_MEMORY="256Mi"
        RESOURCES_LIMITS_CPU="500m"
        RESOURCES_LIMITS_MEMORY="1Gi"
        ;;
    staging)
        REPLICAS=2
        RESOURCES_REQUESTS_CPU="200m"
        RESOURCES_REQUESTS_MEMORY="512Mi"
        RESOURCES_LIMITS_CPU="1000m"
        RESOURCES_LIMITS_MEMORY="2Gi"
        ;;
    prod)
        REPLICAS=3
        RESOURCES_REQUESTS_CPU="500m"
        RESOURCES_REQUESTS_MEMORY="1Gi"
        RESOURCES_LIMITS_CPU="2000m"
        RESOURCES_LIMITS_MEMORY="4Gi"
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Image name
IMAGE_NAME="$REGISTRY/drugban-api:$TAG"

# Functions
check_dependencies() {
    local deps=("docker" "kubectl")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is not installed or not in PATH"
            exit 1
        fi
    done
}

check_docker_running() {
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
}

build_image() {
    log_info "Building Docker image: $IMAGE_NAME"
    
    # Build multi-stage Docker image
    docker build \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --tag "$IMAGE_NAME" \
        --tag "$REGISTRY/drugban-api:latest" \
        "$SCRIPT_DIR"
    
    log_success "Docker image built successfully"
}

push_image() {
    log_info "Pushing Docker image: $IMAGE_NAME"
    
    # Login to registry if credentials are available
    if [[ -n "${DOCKER_REGISTRY_USERNAME:-}" && -n "${DOCKER_REGISTRY_PASSWORD:-}" ]]; then
        echo "$DOCKER_REGISTRY_PASSWORD" | docker login "$REGISTRY" --username "$DOCKER_REGISTRY_USERNAME" --password-stdin
    fi
    
    docker push "$IMAGE_NAME"
    docker push "$REGISTRY/drugban-api:latest"
    
    log_success "Docker image pushed successfully"
}

deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes environment: $ENVIRONMENT"
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations with environment-specific values
    envsubst < "$SCRIPT_DIR/k8s/configmap.yaml" | kubectl apply -f -
    
    # Deploy Redis
    kubectl apply -f "$SCRIPT_DIR/k8s/redis.yaml"
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=drugban-redis --timeout=120s
    
    # Deploy API with environment-specific settings
    sed "s|replicas: 3|replicas: $REPLICAS|g; \
         s|drugban/api:latest|$IMAGE_NAME|g; \
         s|memory: \"512Mi\"|memory: \"$RESOURCES_REQUESTS_MEMORY\"|g; \
         s|cpu: \"200m\"|cpu: \"$RESOURCES_REQUESTS_CPU\"|g; \
         s|memory: \"2Gi\"|memory: \"$RESOURCES_LIMITS_MEMORY\"|g; \
         s|cpu: \"1000m\"|cpu: \"$RESOURCES_LIMITS_CPU\"|g" \
         "$SCRIPT_DIR/k8s/api-deployment.yaml" | kubectl apply -f -
    
    # Deploy HPA
    kubectl apply -f "$SCRIPT_DIR/k8s/hpa.yaml"
    
    # Deploy Ingress
    kubectl apply -f "$SCRIPT_DIR/k8s/ingress.yaml"
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=drugban-api --timeout=300s
    
    # Get service information
    log_success "Deployment completed successfully!"
    
    # Show deployment status
    echo ""
    log_info "Deployment Status:"
    kubectl get pods -n "$NAMESPACE" -l app=drugban-api
    echo ""
    kubectl get services -n "$NAMESPACE"
    echo ""
    
    # Get external IP if available
    EXTERNAL_IP=$(kubectl get service drugban-api-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$EXTERNAL_IP" ]]; then
        log_success "API available at: http://$EXTERNAL_IP"
        log_info "API documentation: http://$EXTERNAL_IP/docs"
    else
        log_info "Use 'kubectl port-forward' to access the service locally"
        log_info "kubectl port-forward -n $NAMESPACE service/drugban-api-service 8000:80"
    fi
}

start_local_environment() {
    log_info "Starting local development environment"
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be ready..."
    
    # Wait for API to be healthy
    timeout=60
    while [[ $timeout -gt 0 ]]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            break
        fi
        sleep 2
        ((timeout -= 2))
    done
    
    if [[ $timeout -le 0 ]]; then
        log_warning "API health check timed out"
    else
        log_success "Local environment started successfully!"
        echo ""
        log_info "Services available:"
        log_info "  API: http://localhost:8000"
        log_info "  API Docs: http://localhost:8000/docs"
        log_info "  MLflow: http://localhost:5000"
        log_info "  Redis: localhost:6379"
        echo ""
        log_info "View logs: docker-compose logs -f"
        log_info "Stop services: docker-compose down"
    fi
}

run_tests() {
    log_info "Running tests"
    
    # Install test dependencies if needed
    pip install -r requirements-dev.txt 2>/dev/null || true
    
    # Run different types of tests
    log_info "Running unit tests..."
    python -m pytest tests/ -v --cov=api --cov-report=term-missing
    
    log_info "Running API integration tests..."
    if curl -f http://localhost:8000/health &> /dev/null; then
        python -m pytest tests/integration/ -v
    else
        log_warning "API not running, skipping integration tests"
    fi
    
    log_success "Tests completed"
}

cleanup_resources() {
    log_info "Cleaning up resources"
    
    case "$1" in
        local)
            docker-compose down --volumes --remove-orphans
            docker image prune -f
            ;;
        kubernetes)
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
        all)
            cleanup_resources local
            cleanup_resources kubernetes
            docker system prune -af
            ;;
        *)
            log_error "Invalid cleanup target: $1"
            exit 1
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "DrugBAN Deployment Script"
    log_info "Environment: $ENVIRONMENT"
    log_info "Registry: $REGISTRY"
    log_info "Image: $IMAGE_NAME"
    log_info "Namespace: $NAMESPACE"
    echo ""
    
    check_dependencies
    
    case "$COMMAND" in
        build)
            check_docker_running
            build_image
            ;;
        push)
            check_docker_running
            push_image
            ;;
        deploy)
            deploy_to_kubernetes
            ;;
        local)
            check_docker_running
            start_local_environment
            ;;
        test)
            run_tests
            ;;
        clean)
            cleanup_resources "${2:-all}"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Trap errors
trap 'log_error "Script failed on line $LINENO"' ERR

# Run main function
main "$@"