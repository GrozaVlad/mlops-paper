#!/bin/bash
# Deploy Complete Monitoring Stack for DrugBAN MLOps Pipeline
# Deploys Prometheus, Grafana, AlertManager, ELK Stack, and Jaeger

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="monitoring"
DRUGBAN_NAMESPACE="drugban"

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
DrugBAN Monitoring Stack Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy          Deploy complete monitoring stack
    prometheus      Deploy only Prometheus
    grafana         Deploy only Grafana
    alertmanager    Deploy only AlertManager
    elk             Deploy only ELK stack
    jaeger          Deploy only Jaeger
    cleanup         Remove all monitoring components
    status          Check deployment status

Options:
    -n, --namespace     Monitoring namespace [default: monitoring]
    -h, --help          Show this help message

Examples:
    $0 deploy                    # Deploy complete stack
    $0 prometheus               # Deploy only Prometheus
    $0 cleanup                  # Remove all components

EOF
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy|prometheus|grafana|alertmanager|elk|jaeger|cleanup|status)
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
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Check dependencies
check_dependencies() {
    local deps=("kubectl")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    log_success "Namespace created/updated"
}

# Deploy Prometheus
deploy_prometheus() {
    log_info "Deploying Prometheus..."
    
    kubectl apply -f "$SCRIPT_DIR/prometheus-deployment.yaml"
    
    # Wait for deployment
    log_info "Waiting for Prometheus to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=prometheus --timeout=300s
    
    log_success "Prometheus deployed successfully"
}

# Deploy Grafana
deploy_grafana() {
    log_info "Deploying Grafana..."
    
    # Create dashboard ConfigMap
    kubectl create configmap grafana-dashboards \
        --from-file="$SCRIPT_DIR/dashboards/" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl apply -f "$SCRIPT_DIR/grafana-deployment.yaml"
    
    # Wait for deployment
    log_info "Waiting for Grafana to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=grafana --timeout=300s
    
    log_success "Grafana deployed successfully"
    log_info "Grafana admin password: drugban123"
}

# Deploy AlertManager
deploy_alertmanager() {
    log_info "Deploying AlertManager..."
    
    kubectl apply -f "$SCRIPT_DIR/alertmanager-deployment.yaml"
    
    # Wait for deployment
    log_info "Waiting for AlertManager to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=alertmanager --timeout=300s
    
    log_success "AlertManager deployed successfully"
}

# Deploy ELK Stack
deploy_elk() {
    log_info "Deploying ELK Stack..."
    
    # Deploy Elasticsearch
    log_info "Deploying Elasticsearch..."
    kubectl apply -f "$SCRIPT_DIR/elasticsearch-deployment.yaml"
    
    # Wait for Elasticsearch
    log_info "Waiting for Elasticsearch to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=elasticsearch --timeout=600s
    
    # Deploy Logstash
    log_info "Deploying Logstash..."
    kubectl apply -f "$SCRIPT_DIR/logstash-deployment.yaml"
    
    # Wait for Logstash
    log_info "Waiting for Logstash to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=logstash --timeout=300s
    
    # Deploy Kibana
    log_info "Deploying Kibana..."
    kubectl apply -f "$SCRIPT_DIR/kibana-deployment.yaml"
    
    # Wait for Kibana
    log_info "Waiting for Kibana to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=kibana --timeout=300s
    
    log_success "ELK Stack deployed successfully"
}

# Deploy Jaeger
deploy_jaeger() {
    log_info "Deploying Jaeger..."
    
    kubectl apply -f "$SCRIPT_DIR/jaeger-deployment.yaml"
    
    # Wait for deployment
    log_info "Waiting for Jaeger to be ready..."
    kubectl wait --namespace="$NAMESPACE" --for=condition=ready pod --selector=app=jaeger --timeout=300s
    
    log_success "Jaeger deployed successfully"
}

# Deploy complete monitoring stack
deploy_complete_stack() {
    log_info "Deploying complete monitoring stack..."
    
    create_namespace
    
    # Deploy in order
    deploy_prometheus
    deploy_grafana
    deploy_alertmanager
    deploy_elk
    deploy_jaeger
    
    # Update DrugBAN API to include monitoring
    update_drugban_monitoring
    
    log_success "Complete monitoring stack deployed successfully!"
    
    # Show access information
    show_access_info
}

# Update DrugBAN API for monitoring
update_drugban_monitoring() {
    log_info "Updating DrugBAN API with monitoring annotations..."
    
    # Add Prometheus scraping annotations to DrugBAN pods
    kubectl patch deployment drugban-api -n "$DRUGBAN_NAMESPACE" -p '{
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": "8001",
                        "prometheus.io/path": "/metrics"
                    }
                }
            }
        }
    }' 2>/dev/null || log_warning "Could not update DrugBAN API deployment (may not exist yet)"
    
    log_success "DrugBAN monitoring annotations updated"
}

# Show access information
show_access_info() {
    log_info "Monitoring Stack Access Information:"
    echo ""
    
    # Get external IPs or provide port-forward commands
    echo "📊 Access URLs (use kubectl port-forward for local access):"
    echo ""
    echo "Prometheus:     kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090"
    echo "                http://localhost:9090"
    echo ""
    echo "Grafana:        kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"
    echo "                http://localhost:3000 (admin/drugban123)"
    echo ""
    echo "AlertManager:   kubectl port-forward -n $NAMESPACE svc/alertmanager 9093:9093"
    echo "                http://localhost:9093"
    echo ""
    echo "Kibana:         kubectl port-forward -n $NAMESPACE svc/kibana 5601:5601"
    echo "                http://localhost:5601"
    echo ""
    echo "Jaeger:         kubectl port-forward -n $NAMESPACE svc/jaeger 16686:16686"
    echo "                http://localhost:16686"
    echo ""
    
    # Check for LoadBalancer IPs
    log_info "Checking for external LoadBalancer IPs..."
    
    for service in prometheus grafana alertmanager kibana jaeger; do
        external_ip=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [[ -n "$external_ip" ]]; then
            echo "$service: http://$external_ip"
        fi
    done
}

# Check deployment status
check_status() {
    log_info "Checking monitoring stack status..."
    echo ""
    
    # Check namespace
    echo "📁 Namespace:"
    kubectl get namespace "$NAMESPACE" 2>/dev/null || echo "Namespace $NAMESPACE not found"
    echo ""
    
    # Check deployments
    echo "🚀 Deployments:"
    kubectl get deployments -n "$NAMESPACE" -o wide 2>/dev/null || echo "No deployments found"
    echo ""
    
    # Check pods
    echo "🎭 Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || echo "No pods found"
    echo ""
    
    # Check services
    echo "🌐 Services:"
    kubectl get services -n "$NAMESPACE" -o wide 2>/dev/null || echo "No services found"
    echo ""
    
    # Check ingresses
    echo "🌍 Ingresses:"
    kubectl get ingresses -n "$NAMESPACE" -o wide 2>/dev/null || echo "No ingresses found"
    echo ""
    
    # Check PVCs
    echo "💾 Persistent Volume Claims:"
    kubectl get pvc -n "$NAMESPACE" -o wide 2>/dev/null || echo "No PVCs found"
}

# Cleanup monitoring stack
cleanup_monitoring() {
    log_warning "This will remove the entire monitoring stack. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleaning up monitoring stack..."
        
        # Delete deployments
        kubectl delete -f "$SCRIPT_DIR/jaeger-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$SCRIPT_DIR/kibana-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$SCRIPT_DIR/logstash-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$SCRIPT_DIR/elasticsearch-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$SCRIPT_DIR/alertmanager-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$SCRIPT_DIR/grafana-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$SCRIPT_DIR/prometheus-deployment.yaml" --ignore-not-found=true
        
        # Delete ConfigMaps
        kubectl delete configmap grafana-dashboards -n "$NAMESPACE" --ignore-not-found=true
        
        # Delete namespace (optional)
        log_warning "Delete monitoring namespace? (y/N)"
        read -r ns_response
        if [[ "$ns_response" =~ ^[Yy]$ ]]; then
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        fi
        
        log_success "Monitoring stack cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Main execution
main() {
    log_info "DrugBAN Monitoring Stack Deployment"
    log_info "Namespace: $NAMESPACE"
    log_info "Command: $COMMAND"
    echo ""
    
    check_dependencies
    
    case "$COMMAND" in
        deploy)
            deploy_complete_stack
            ;;
        prometheus)
            create_namespace
            deploy_prometheus
            ;;
        grafana)
            create_namespace
            deploy_grafana
            ;;
        alertmanager)
            create_namespace
            deploy_alertmanager
            ;;
        elk)
            create_namespace
            deploy_elk
            ;;
        jaeger)
            create_namespace
            deploy_jaeger
            ;;
        status)
            check_status
            ;;
        cleanup)
            cleanup_monitoring
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